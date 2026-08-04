"""
Per-row uplift scoring for machine callers — built for Clay HTTP API columns.

POST /api/score           — score one contact (one Clay table row per call)
POST /api/score/batch     — score up to 1000 contacts in one call
POST /api/score/train     — fit + persist the tenant's scoring model
GET  /api/score/status    — model availability + metadata
GET  /api/score/srm       — audit: assignment split + holdout violations

Auth: X-API-Key header (minted at /api/alpha/auth/api-key), Bearer JWT, or
unauthenticated (demo tenant, synthetic data).

Design rules:
    - No Claude call on the scoring path — Clay fires one request per row
      with retries; responses must be fast, cheap, and idempotent.
    - Holdout/assignment is deterministic (hash of segment + email) and uses
      the same segment_id convention as core/outreach.py, so a contact keeps
      one bucket across scoring, sending, and lift measurement.
    - Cold start: a tenant with no trained model gets randomized
      treat/control assignment — the first campaign wave IS the experiment
      that produces the training data for wave two.
    - Every scored row is logged (scored_rows table) so the SRM check can
      compare the intended split against what was actually sent.
"""

import hashlib
import logging
import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.db import get_tenant_conn, tenant_has_data
from api.deps import OptionalMachineUser, tenant_id_from
from api.rate_limit import train_rate_limit
from core.causal import detect_srm
from core.model_store import (
    ModelStoreError,
    fit_and_save,
    load_latest,
    score_rows,
)
from core.outreach import _is_holdout, _sqlite_path, segment_message_angle
from ingestion.clay_normalizer import (
    normalize_channel,
    normalize_company_size,
    normalize_industry,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/score", tags=["score"])


def _cold_start_control_fraction() -> float:
    return float(os.getenv("COLD_START_CONTROL_FRACTION", "0.50"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ScoreContact(BaseModel):
    email: str = Field(..., min_length=3, description="Contact email — the deterministic assignment key")
    company_size: str | int | None = Field(None, description="Label ('Enterprise'), range ('51-200'), or employee count")
    employee_count: int | None = Field(None, ge=0, description="Explicit employee count — wins over company_size")
    industry: str | None = Field(None, description="Industry vertical (free text)")
    channel: str | None = Field(None, description="Acquisition channel or source label ('Google Ads', 'LinkedIn', …)")
    source: str | None = Field(None, description="Alias for channel — used when channel is absent (Clay often names it 'source')")


class ScoreResult(BaseModel):
    email: str
    segment_id: str
    company_size: str
    channel: str
    mode: str = Field(..., description="'scored' when a model served the row, 'cold_start' for randomized assignment")
    assignment: str = Field(..., description="'treat' → eligible to contact, 'control' → suppress (holdout)")
    holdout_flag: bool = Field(..., description="True means DO NOT send — this contact measures lift")
    cate_estimate: float | None = Field(None, description="Predicted activation uplift (null in cold_start mode)")
    uplift_tier: str | None = Field(None, description="high | mid | deprioritize (null in cold_start mode)")
    message_angle: str = Field(..., description="One-line causal angle for the AI-writing step")
    model_version: str | None = None


class ScoreBatchRequest(BaseModel):
    contacts: list[ScoreContact] = Field(..., min_length=1, max_length=1000)


class ScoreBatchResponse(BaseModel):
    results: list[ScoreResult]
    total: int
    mode: str


class TrainResponse(BaseModel):
    model_version: str
    trained_at: str
    n_train: int
    n_treatment: int
    n_control: int
    high_cutoff: float
    deprioritize_cutoff: float


class ScoreStatusResponse(BaseModel):
    has_model: bool
    mode: str
    model_version: str | None = None
    trained_at: str | None = None
    n_train: int | None = None


class SrmModeAudit(BaseModel):
    mode: str
    n_treat: int
    n_control: int
    expected_treat_split: float
    srm_detected: bool
    p_value: float
    recommendation: str


class SrmAuditResponse(BaseModel):
    splits: list[SrmModeAudit]
    holdout_violations: int = Field(
        ..., description="Contacts assigned to control at scoring time that were sent anyway — a misconfigured downstream filter (e.g. Clay)"
    )
    total_sends: int
    n_scored: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=ScoreResult, summary="Score one contact (Clay HTTP API column)")
def score_one(contact: ScoreContact, user: OptionalMachineUser) -> dict:
    """
    Score a single contact for uplift and return targeting instructions.

    With a trained model: CATE estimate, uplift tier, holdout flag.
    Without one (cold start): randomized treat/control assignment so the
    first wave doubles as the training experiment.
    """
    tenant_id = tenant_id_from(user)
    return _score_and_log([contact], tenant_id)[0]


@router.post("/batch", response_model=ScoreBatchResponse, summary="Score up to 1000 contacts")
def score_batch(body: ScoreBatchRequest, user: OptionalMachineUser) -> dict:
    """Score a batch of contacts in one call — same semantics as /score per row."""
    tenant_id = tenant_id_from(user)
    results = _score_and_log(body.contacts, tenant_id)
    return {
        "results": results,
        "total": len(results),
        "mode": results[0]["mode"] if results else "cold_start",
    }


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(train_rate_limit)],
    summary="Fit and persist the tenant's uplift scoring model",
)
def train(user: OptionalMachineUser) -> dict:
    """
    Train the scoring model from the tenant's funnel data.

    Run after importing campaign outcomes (wave 0), and re-run after each
    subsequent wave — every wave's outcomes are the next wave's training data.
    """
    tenant_id = tenant_id_from(user)
    if not tenant_has_data(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "No funnel data",
                "detail": "Upload a funnel CSV (or import outcomes) before training a scoring model.",
            },
        )

    with get_tenant_conn(tenant_id) as conn:
        df = conn.execute(
            """
            SELECT activated, treatment, company_size, channel, industry
            FROM users
            WHERE signed_up = 1
            """
        ).df()

    try:
        meta = fit_and_save(df, tenant_id)
    except ModelStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Training failed", "detail": str(exc)},
        )

    logger.info("score/train | tenant=%s version=%s n=%d", tenant_id, meta["model_version"], meta["n_train"])
    return meta


@router.get("/status", response_model=ScoreStatusResponse, summary="Scoring model availability")
def score_status(user: OptionalMachineUser) -> dict:
    """Report whether the tenant has a trained model and which version serves /score."""
    tenant_id = tenant_id_from(user)
    payload = load_latest(tenant_id)
    if payload is None:
        return {"has_model": False, "mode": "cold_start"}
    return {
        "has_model": True,
        "mode": "scored",
        "model_version": payload["model_version"],
        "trained_at": payload["trained_at"],
        "n_train": payload["n_train"],
    }


@router.get("/srm", response_model=SrmAuditResponse, summary="Audit assignment split and holdout enforcement")
def srm_audit(user: OptionalMachineUser) -> dict:
    """
    Cross-check the scoring log against actual sends.

    Two failure modes are surfaced:
    1. SRM on the assignment split — the observed treat/control ratio among
       scored contacts deviates from the intended split (chi-square, α=0.01,
       per CLAUDE.md run BEFORE interpreting any lift result).
    2. Holdout violations — contacts assigned to control at scoring time that
       appear in contact_sends anyway. Any nonzero count means the downstream
       send filter (typically a Clay table filter) is misconfigured and the
       lift measurement is contaminated.
    """
    tenant_id = tenant_id_from(user)

    conn = sqlite3.connect(_sqlite_path())
    try:
        split_rows = conn.execute(
            """
            SELECT mode, assignment, COUNT(*) FROM scored_rows
            WHERE tenant_id = ?
            GROUP BY mode, assignment
            """,
            (tenant_id,),
        ).fetchall()

        # SQLite has no SHA-256 SQL function — hash sent emails in Python and
        # intersect with the control-assigned hashes from the scoring log.
        control_hashes = {
            row[0] for row in conn.execute(
                "SELECT email_hash FROM scored_rows WHERE tenant_id = ? AND assignment = 'control'",
                (tenant_id,),
            )
        }
        sent_emails = [
            row[0] for row in conn.execute(
                "SELECT email FROM contact_sends WHERE tenant_id = ? AND status = 'sent'",
                (tenant_id,),
            )
        ]
    finally:
        conn.close()

    total_sends = len(sent_emails)
    violations = sum(
        1 for email in sent_emails
        if hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16] in control_hashes
    )

    counts: dict[str, dict[str, int]] = {}
    for mode, assignment, n in split_rows:
        counts.setdefault(mode, {"treat": 0, "control": 0})[assignment] = n

    splits = []
    for mode, c in counts.items():
        control_frac = (
            _cold_start_control_fraction() if mode == "cold_start"
            else float(os.getenv("HOLDOUT_FRACTION", "0.20"))
        )
        expected_treat = 1.0 - control_frac
        if c["treat"] + c["control"] == 0:
            continue
        srm = detect_srm(c["treat"], c["control"], expected_split=expected_treat)
        splits.append({
            "mode": mode,
            "n_treat": c["treat"],
            "n_control": c["control"],
            "expected_treat_split": expected_treat,
            "srm_detected": srm["srm_detected"],
            "p_value": srm["p_value"],
            "recommendation": srm["recommendation"],
        })

    n_scored = sum(c["treat"] + c["control"] for c in counts.values())
    return {
        "splits": splits,
        "holdout_violations": int(violations),
        "total_sends": int(total_sends),
        "n_scored": n_scored,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _score_and_log(contacts: list[ScoreContact], tenant_id: str) -> list[dict]:
    """Normalize, score (or cold-start assign), and log a list of contacts."""
    normalized = [
        {
            "email": c.email.strip().lower(),
            "company_size": normalize_company_size(c.company_size, c.employee_count),
            "channel": normalize_channel(c.channel or c.source),
            "industry": normalize_industry(c.industry),
        }
        for c in contacts
    ]

    try:
        payload = load_latest(tenant_id)
    except ModelStoreError as exc:
        logger.error("score: model load failed for tenant=%s: %s", tenant_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Model load failed", "detail": str(exc)},
        )

    if payload is None:
        results = [_cold_start_result(row) for row in normalized]
    else:
        scored = score_rows(payload, normalized)
        results = [_scored_result(row, s) for row, s in zip(normalized, scored)]

    _log_scored_rows(results, tenant_id)
    return results


def _cold_start_result(row: dict) -> dict:
    """Randomized treat/control assignment for tenants with no model yet."""
    segment_id = f"{row['company_size']}_{row['channel']}"
    holdout = _is_holdout(segment_id, row["email"], _cold_start_control_fraction())
    return {
        "email":         row["email"],
        "segment_id":    segment_id,
        "company_size":  row["company_size"],
        "channel":       row["channel"],
        "mode":          "cold_start",
        "assignment":    "control" if holdout else "treat",
        "holdout_flag":  holdout,
        "cate_estimate": None,
        "uplift_tier":   None,
        "message_angle": segment_message_angle(row["company_size"], row["channel"], "cold_start"),
        "model_version": None,
    }


def _scored_result(row: dict, scored: dict) -> dict:
    """Combine a normalized row with its model score into the response shape."""
    segment_id = f"{row['company_size']}_{row['channel']}"
    holdout = _is_holdout(segment_id, row["email"])
    return {
        "email":         row["email"],
        "segment_id":    segment_id,
        "company_size":  row["company_size"],
        "channel":       row["channel"],
        "mode":          "scored",
        "assignment":    "control" if holdout else "treat",
        "holdout_flag":  holdout,
        "cate_estimate": round(scored["cate_estimate"], 4),
        "uplift_tier":   scored["uplift_tier"],
        "message_angle": segment_message_angle(row["company_size"], row["channel"], scored["uplift_tier"]),
        "model_version": scored["model_version"],
    }


def _log_scored_rows(results: list[dict], tenant_id: str) -> None:
    """
    Persist scored rows for SRM auditing.

    UNIQUE(tenant_id, email_hash) ON CONFLICT REPLACE in the schema makes
    re-scores idempotent — Clay column re-runs do not double-count.
    Logging failure is non-fatal: scoring responses still return.
    """
    try:
        conn = sqlite3.connect(_sqlite_path())
        try:
            conn.executemany(
                """
                INSERT INTO scored_rows
                    (tenant_id, email_hash, segment_id, company_size, channel,
                     assignment, holdout_flag, uplift_tier, cate_estimate,
                     model_version, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        tenant_id,
                        hashlib.sha256(r["email"].encode()).hexdigest()[:16],
                        r["segment_id"],
                        r["company_size"],
                        r["channel"],
                        r["assignment"],
                        int(r["holdout_flag"]),
                        r["uplift_tier"],
                        r["cate_estimate"],
                        r["model_version"],
                        r["mode"],
                    )
                    for r in results
                ],
            )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.error("score: failed to log %d scored rows: %s", len(results), exc)
