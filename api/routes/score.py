"""
Per-row uplift scoring for machine callers — built for Clay HTTP API columns.

POST /api/score           — score one contact (one Clay table row per call)
POST /api/score/batch     — score up to 1000 contacts in one call
POST /api/score/outcomes  — import activations for scored contacts (closes the flywheel)
POST /api/score/train     — fit + persist the tenant's scoring model (funnel or campaign source)
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

import pandas as pd
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


class TrainRequest(BaseModel):
    source: str = Field(
        "auto",
        description="'funnel' = DuckDB funnel upload; 'campaign' = scored contacts with "
                    "imported outcomes (the Clay flywheel); 'auto' = campaign when it has "
                    "enough outcome data, else funnel",
    )
    include_scored_waves: bool = Field(
        False,
        description="Campaign source only. False (default) trains exclusively on "
                    "cold_start-mode rows, whose treat/control assignment was randomized. "
                    "Rows scored by a model are excluded because their sends were targeted "
                    "by that model's tiers — training on them lets the model confound its "
                    "own future training data. Enable only if you know what you're doing.",
    )
    mde: float = Field(
        0.05, gt=0, lt=1,
        description="Minimum detectable effect (absolute pp) used for the power check",
    )


class PowerCheck(BaseModel):
    baseline_rate: float
    mde: float
    required_n_per_arm: int
    n_treatment: int
    n_control: int
    adequately_powered: bool
    warning: str | None = None


class TrainResponse(BaseModel):
    model_version: str
    trained_at: str
    n_train: int
    n_treatment: int
    n_control: int
    high_cutoff: float
    deprioritize_cutoff: float
    source_used: str
    power_check: PowerCheck


class OutcomesRequest(BaseModel):
    activated_emails: list[str] = Field(..., min_length=1, max_length=10000)
    activated_at: str | None = Field(None, description="ISO datetime; defaults to now")


class OutcomesResponse(BaseModel):
    updated: int
    not_found: int
    n_with_outcomes: int = Field(..., description="Total scored contacts with an outcome recorded")


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
def train(user: OptionalMachineUser, req: TrainRequest | None = None) -> dict:
    """
    Train the scoring model from funnel data or scored-campaign outcomes.

    The campaign source closes the Clay flywheel: contacts scored by
    /api/score whose outcomes were imported via /api/score/outcomes become
    training rows (treatment = randomized assignment, outcome = activated).
    Only randomized cold-start cohorts are used by default — waves targeted
    by a model's own tiers would confound its retraining.

    Every response includes a power check: whether the training arms are
    large enough to detect the requested MDE at 80% power. An underpowered
    model still trains, but its tiers should be treated as provisional.
    """
    req = req or TrainRequest()
    if req.source not in {"auto", "funnel", "campaign"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Invalid source", "detail": "source must be auto, funnel, or campaign"},
        )
    tenant_id = tenant_id_from(user)

    campaign_df = _campaign_training_frame(tenant_id, req.include_scored_waves)
    use_campaign = req.source == "campaign" or (
        req.source == "auto"
        and len(campaign_df) >= 50
        and campaign_df["activated"].sum() > 0
    )

    if use_campaign:
        if len(campaign_df) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "No campaign data",
                    "detail": "Score contacts via /api/score and import outcomes via "
                              "/api/score/outcomes before campaign training.",
                },
            )
        df, source_used = campaign_df, "campaign"
    else:
        if not tenant_has_data(tenant_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "No funnel data",
                    "detail": "Upload a funnel CSV (or import campaign outcomes) before training.",
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
        source_used = "funnel"

    try:
        meta = fit_and_save(df, tenant_id)
    except ModelStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Training failed", "detail": str(exc)},
        )

    power = _power_check(df, req.mde)
    logger.info(
        "score/train | tenant=%s version=%s n=%d source=%s powered=%s",
        tenant_id, meta["model_version"], meta["n_train"], source_used,
        power["adequately_powered"],
    )
    return {**meta, "source_used": source_used, "power_check": power}


@router.post("/outcomes", response_model=OutcomesResponse, summary="Import campaign outcomes for scored contacts")
def import_outcomes(body: OutcomesRequest, user: OptionalMachineUser) -> dict:
    """
    Record activations (replies, meetings, signups) for scored contacts.

    Matches on the same email hash used at scoring time — send the same
    email addresses Clay scored. Outcomes are recorded for BOTH arms:
    holdout/control outcomes are the control side of every lift and
    training computation. Contacts never scored by this tenant count as
    not_found. This is the bridge that closes the flywheel: after importing,
    run POST /api/score/train (source=campaign or auto).
    """
    from datetime import datetime, timezone

    tenant_id = tenant_id_from(user)
    activated_at = body.activated_at or datetime.now(tz=timezone.utc).isoformat()

    updated = 0
    conn = sqlite3.connect(_sqlite_path())
    try:
        for email in body.activated_emails:
            cursor = conn.execute(
                "UPDATE scored_rows SET activated_at = ? WHERE tenant_id = ? AND email_hash = ?",
                (activated_at, tenant_id, _email_hash(email)),
            )
            updated += cursor.rowcount
        conn.commit()
        n_with = conn.execute(
            "SELECT COUNT(*) FROM scored_rows WHERE tenant_id = ? AND activated_at IS NOT NULL",
            (tenant_id,),
        ).fetchone()[0]
    finally:
        conn.close()

    not_found = len(body.activated_emails) - updated
    logger.info(
        "score/outcomes | tenant=%s | updated=%d | not_found=%d", tenant_id, updated, not_found,
    )
    return {"updated": updated, "not_found": not_found, "n_with_outcomes": int(n_with)}


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
    violations = sum(1 for email in sent_emails if _email_hash(email) in control_hashes)

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
        "industry":      row["industry"],
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
        "industry":      row["industry"],
        "mode":          "scored",
        "assignment":    "control" if holdout else "treat",
        "holdout_flag":  holdout,
        "cate_estimate": round(scored["cate_estimate"], 4),
        "uplift_tier":   scored["uplift_tier"],
        "message_angle": segment_message_angle(row["company_size"], row["channel"], scored["uplift_tier"]),
        "model_version": scored["model_version"],
    }


def _email_hash(email: str) -> str:
    """Privacy-preserving contact key used throughout scored_rows."""
    return hashlib.sha256(email.strip().lower().encode()).hexdigest()[:16]


def _campaign_training_frame(tenant_id: str, include_scored_waves: bool) -> pd.DataFrame:
    """
    Build a training frame from scored contacts and their imported outcomes.

    treatment = the randomized assignment made at scoring time (treat=1).
    activated = an outcome was imported for the contact (no outcome = 0 —
    the standard convention for conversion imports).

    Cohort rule: only cold_start-mode rows are included unless the caller
    opts in — waves scored by a model were *sent* according to that model's
    tiers, so their assignment-vs-send relationship is no longer independent
    of the model's predictions.
    """
    mode_clause = "" if include_scored_waves else "AND mode = 'cold_start'"
    conn = sqlite3.connect(_sqlite_path())
    try:
        rows = conn.execute(
            f"""
            SELECT company_size, channel, industry, assignment, activated_at
            FROM scored_rows
            WHERE tenant_id = ? {mode_clause}
            """,
            (tenant_id,),
        ).fetchall()
    finally:
        conn.close()

    return pd.DataFrame(
        [
            {
                "company_size": r[0],
                "channel":      r[1],
                "industry":     r[2] or "unknown",
                "treatment":    1 if r[3] == "treat" else 0,
                "activated":    1 if r[4] is not None else 0,
            }
            for r in rows
        ],
        columns=["company_size", "channel", "industry", "treatment", "activated"],
    )


def _power_check(df: pd.DataFrame, mde: float) -> dict:
    """
    Check whether the training arms can detect `mde` at alpha=0.05, power=0.80.

    Uses the experiment designer's sample-size formula on the control arm's
    realized activation rate. An underpowered model is still returned to the
    caller — but flagged, so tiers fit on noise are not mistaken for signal.
    """
    from core.experiment import _required_n

    control = df[df["treatment"] == 0]["activated"]
    baseline = float(control.mean()) if len(control) else 0.0
    # _required_n needs rates strictly inside (0, 1); clamp degenerate baselines
    clamped = min(max(baseline, 0.01), 1.0 - mde - 0.01)
    required = _required_n(clamped, clamped + mde, alpha=0.05, power=0.80)

    n_t = int((df["treatment"] == 1).sum())
    n_c = int((df["treatment"] == 0).sum())
    powered = min(n_t, n_c) >= required

    warning = None
    if not powered:
        warning = (
            f"Underpowered: {required:,} users per arm needed to detect {mde:.0%} "
            f"absolute lift at 80% power (have treatment={n_t:,}, control={n_c:,}). "
            f"Treat tiers as provisional and keep collecting outcomes."
        )
    return {
        "baseline_rate":      round(baseline, 4),
        "mde":                mde,
        "required_n_per_arm": required,
        "n_treatment":        n_t,
        "n_control":          n_c,
        "adequately_powered": powered,
        "warning":            warning,
    }


def _log_scored_rows(results: list[dict], tenant_id: str) -> None:
    """
    Persist scored rows for SRM auditing and campaign training.

    Explicit UPSERT (not the table's ON CONFLICT REPLACE): re-scores from
    Clay retries/column re-runs refresh the scoring fields but PRESERVE
    activated_at — a REPLACE would silently wipe imported outcomes.
    Logging failure is non-fatal: scoring responses still return.
    """
    try:
        conn = sqlite3.connect(_sqlite_path())
        try:
            conn.executemany(
                """
                INSERT INTO scored_rows
                    (tenant_id, email_hash, segment_id, company_size, channel,
                     industry, assignment, holdout_flag, uplift_tier,
                     cate_estimate, model_version, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, email_hash) DO UPDATE SET
                    scored_at     = datetime('now'),
                    segment_id    = excluded.segment_id,
                    company_size  = excluded.company_size,
                    channel       = excluded.channel,
                    industry      = excluded.industry,
                    assignment    = excluded.assignment,
                    holdout_flag  = excluded.holdout_flag,
                    uplift_tier   = excluded.uplift_tier,
                    cate_estimate = excluded.cate_estimate,
                    model_version = excluded.model_version,
                    mode          = excluded.mode
                """,
                [
                    (
                        tenant_id,
                        _email_hash(r["email"]),
                        r["segment_id"],
                        r["company_size"],
                        r["channel"],
                        r["industry"],
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
