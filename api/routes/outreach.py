"""
POST /api/outreach/generate — personalised outreach message for a high-uplift segment.
GET  /api/outreach/results  — recent outreach log from SQLite.

Rules enforced here (mirroring CLAUDE.md / core/outreach.py):
    - Only segments above CATE threshold receive messages (OutreachError → 422)
    - 20% holdout auto-assigned per segment; holdout_flag=True means DO NOT send
    - Claude API errors return graceful fallback; route never returns 500 for API errors
    - Raw API errors never exposed — UI receives sanitised fallback shape
    - Rate limited to 20 requests / minute / IP (protects Anthropic API key budget)
"""

import logging
import sqlite3

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from api.db import get_tenant_conn
from api.deps import CurrentUser, OptionalUser, get_optional_user, tenant_id_from
from api.rate_limit import claude_rate_limit
from core.email_sender import EmailDeliveryError, send_email
from core.outreach import OutreachError, _sqlite_path, generate_outreach

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/outreach", tags=["outreach"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class OutreachSegment(BaseModel):
    cate_estimate: float = Field(..., description="Predicted activation lift (CATE) for this segment")
    company_size: str | None = Field(None, description="SMB | mid_market | enterprise")
    industry: str | None = Field(None, description="Industry vertical")
    channel: str | None = Field(None, description="organic | paid_search | social | referral | email")
    funnel_stage: str | None = Field(None, description="Stage where segment dropped off")
    segment_id: str | None = Field(None, description="Override auto-generated segment ID")


class OutreachGenerateRequest(BaseModel):
    segment: OutreachSegment
    product_context: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Short description of what the product does",
    )
    tone: str = Field("direct", description="warm | direct | technical")
    cate_threshold: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Minimum CATE to qualify. Reads CATE_UPLIFT_THRESHOLD env if omitted.",
    )
    holdout_fraction: float | None = Field(
        None, ge=0.0, lt=1.0,
        description="Fraction to withhold. Reads HOLDOUT_FRACTION env if omitted.",
    )
    user_id: str = Field("", description="User ID for deterministic holdout assignment")


class OutreachGenerateResponse(BaseModel):
    # is_fallback is populated from the "_error" key in the core module's
    # fallback dict via the alias — Pydantic v2 maps "_error" → is_fallback.
    model_config = ConfigDict(populate_by_name=True)

    subject: str
    body: str
    cta: str
    predicted_uplift_group: str
    holdout_flag: bool
    segment_id: str
    is_fallback: bool = Field(False, alias="_error", description="True when Claude API unavailable")


class OutreachLogEntry(BaseModel):
    sent_at: str
    segment_id: str
    company_size: str
    industry: str
    channel: str
    cate_estimate: float
    tone: str
    subject_hash: str
    body_hash: str
    is_holdout: bool


class OutreachResultsResponse(BaseModel):
    results: list[OutreachLogEntry]
    total: int


class LiftSegment(BaseModel):
    segment_id: str
    company_size: str
    channel: str
    predicted_cate: float   # mean CATE from T-Learner at send time
    observed_lift: float    # treatment_rate - control_rate in DuckDB users
    treatment_rate: float
    control_rate: float
    n_sent: int             # non-holdout messages logged
    n_holdout: int          # holdout messages logged
    last_sent_at: str


class LiftSummary(BaseModel):
    total_sent: int
    total_holdout: int
    avg_predicted_cate: float
    avg_observed_lift: float
    n_segments: int


class LiftResponse(BaseModel):
    segments: list[LiftSegment]
    summary: LiftSummary
    data_source: str   # "campaign" (real activation data) | "baseline" (historical DuckDB)


class SendSegmentRequest(BaseModel):
    segment_id: str = Field(..., description="e.g. enterprise_paid_search")
    company_size: str = Field(..., description="SMB | mid_market | enterprise")
    channel: str = Field(..., description="organic | paid_search | social | referral | email")
    cate_estimate: float = Field(..., ge=0.0, le=1.0)
    product_context: str = Field(..., min_length=10, max_length=500)
    tone: str = Field("direct", description="warm | direct | technical")


class SendSegmentResult(BaseModel):
    sent: int
    held_out: int
    failed: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=OutreachGenerateResponse,
    dependencies=[Depends(claude_rate_limit)],
)
def outreach_generate(req: OutreachGenerateRequest, user: OptionalUser) -> dict:
    """
    Generate a personalised outreach email for a B2B SaaS segment.

    Requires CATE estimate above threshold — low-uplift segments are rejected
    with a 422 to prevent budget waste and lift dilution.

    Holdout flag: if True, the message was generated but MUST NOT be sent.
    The holdout user is the control group for measuring outreach lift.

    Rate limited: 20 requests per minute per IP (Claude API cost protection).
    """
    tone = req.tone
    if tone not in {"warm", "direct", "technical"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Invalid tone", "detail": "tone must be warm, direct, or technical"},
        )

    segment_dict = {k: v for k, v in req.segment.model_dump().items() if v is not None}
    tenant_id = tenant_id_from(user)

    try:
        result = generate_outreach(
            segment=segment_dict,
            product_context=req.product_context,
            tone=tone,  # type: ignore[arg-type]
            cate_threshold=req.cate_threshold,
            holdout_fraction=req.holdout_fraction,
            user_id=req.user_id,
            log_to_db=True,
            tenant_id=tenant_id,
        )
    except OutreachError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Segment below CATE threshold", "detail": str(exc)},
        )
    except Exception as exc:
        logger.error("outreach generate error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Outreach generation failed", "detail": str(exc)},
        )

    return result


@router.get("/results", response_model=OutreachResultsResponse)
def outreach_results(limit: int = 50, user: dict | None = Depends(get_optional_user)) -> dict:
    """
    Return the most recent outreach log entries from SQLite.

    Useful for reviewing which segments received messages and which were
    held out, along with CATE estimates and tone used.
    """
    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "limit must be between 1 and 500"},
        )
    tenant_id = tenant_id_from(user)
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT sent_at, segment_id, company_size, industry, channel,
                   cate_estimate, tone, subject_hash, body_hash, is_holdout
            FROM outreach_log
            WHERE tenant_id = ?
            ORDER BY sent_at DESC
            LIMIT ?
            """,
            (tenant_id, limit),
        ).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM outreach_log WHERE tenant_id = ?", (tenant_id,)
        ).fetchone()[0]
    except Exception as exc:
        logger.error("outreach results query error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to fetch outreach results", "detail": str(exc)},
        )
    finally:
        conn.close()

    entries = [
        {
            "sent_at":       row["sent_at"],
            "segment_id":    row["segment_id"],
            "company_size":  row["company_size"],
            "industry":      row["industry"],
            "channel":       row["channel"],
            "cate_estimate": row["cate_estimate"],
            "tone":          row["tone"],
            "subject_hash":  row["subject_hash"],
            "body_hash":     row["body_hash"],
            "is_holdout":    bool(row["is_holdout"]),
        }
        for row in rows
    ]
    return {"results": entries, "total": total}


@router.get("/lift", response_model=LiftResponse)
def outreach_lift(user: dict | None = Depends(get_optional_user)) -> dict:
    """
    Compare predicted CATE (T-Learner estimate at send time) against observed lift
    (treatment minus control activation rate in DuckDB) for each targeted segment.

    This closes the causal loop: did the segments we predicted would respond well
    actually respond well? Each row is one (company_size × channel) segment that
    received at least one outreach message.

    Observed lift is computed from the full signed-up user population in that
    segment — not just the outreach recipients — giving a stable denominator.
    """
    tenant_id = tenant_id_from(user)

    # 1. Pull segment aggregates from outreach_log (SQLite)
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        log_rows = conn.execute(
            """
            SELECT
                company_size,
                channel,
                AVG(cate_estimate)                               AS predicted_cate,
                SUM(CASE WHEN is_holdout = 0 THEN 1 ELSE 0 END) AS n_sent,
                SUM(CASE WHEN is_holdout = 1 THEN 1 ELSE 0 END) AS n_holdout,
                MAX(sent_at)                                     AS last_sent_at
            FROM outreach_log
            WHERE tenant_id = ?
              AND company_size IS NOT NULL AND company_size != ''
              AND channel      IS NOT NULL AND channel      != ''
            GROUP BY company_size, channel
            ORDER BY predicted_cate DESC
            """,
            (tenant_id,),
        ).fetchall()

        # Check for real activation data in contact_sends
        has_activations = conn.execute(
            """
            SELECT COUNT(*) FROM contact_sends
            WHERE tenant_id = ? AND activated_at IS NOT NULL
            """,
            (tenant_id,),
        ).fetchone()[0] > 0

        if has_activations:
            campaign_rows = conn.execute(
                """
                SELECT
                    segment_id,
                    company_size,
                    channel,
                    AVG(cate_estimate) AS predicted_cate,
                    SUM(CASE WHEN is_holdout = 0 THEN 1 ELSE 0 END)                         AS n_sent,
                    SUM(CASE WHEN is_holdout = 1 THEN 1 ELSE 0 END)                         AS n_holdout,
                    AVG(CASE WHEN is_holdout = 0 AND activated_at IS NOT NULL THEN 1.0 ELSE 0.0 END) AS sent_activation_rate,
                    AVG(CASE WHEN is_holdout = 1 AND activated_at IS NOT NULL THEN 1.0 ELSE 0.0 END) AS holdout_activation_rate,
                    MAX(sent_at) AS last_sent_at
                FROM contact_sends
                WHERE tenant_id = ?
                GROUP BY segment_id, company_size, channel
                HAVING n_sent > 0 OR n_holdout > 0
                ORDER BY predicted_cate DESC
                """,
                (tenant_id,),
            ).fetchall()
        else:
            campaign_rows = []
    except Exception as exc:
        logger.error("lift: sqlite query error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to read outreach log", "detail": str(exc)},
        )
    finally:
        conn.close()

    empty_response = {
        "segments": [],
        "summary": {
            "total_sent": 0, "total_holdout": 0,
            "avg_predicted_cate": 0.0, "avg_observed_lift": 0.0,
            "n_segments": 0,
        },
        "data_source": "campaign",
    }

    if not log_rows:
        return empty_response

    # 2a. Real campaign lift — use contact_sends activation data when available
    if campaign_rows:
        segments_out: list[dict] = []
        total_sent = total_holdout = 0
        for row in campaign_rows:
            n_sent    = int(row["n_sent"])
            n_holdout = int(row["n_holdout"])
            total_sent    += n_sent
            total_holdout += n_holdout
            t_rate  = round(float(row["sent_activation_rate"]),    4)
            c_rate  = round(float(row["holdout_activation_rate"]), 4)
            predicted = round(float(row["predicted_cate"]), 4)
            segments_out.append({
                "segment_id":     row["segment_id"],
                "company_size":   row["company_size"],
                "channel":        row["channel"],
                "predicted_cate": predicted,
                "observed_lift":  round(t_rate - c_rate, 4),
                "treatment_rate": t_rate,
                "control_rate":   c_rate,
                "n_sent":         n_sent,
                "n_holdout":      n_holdout,
                "last_sent_at":   row["last_sent_at"],
            })
        avg_pred = sum(s["predicted_cate"] for s in segments_out) / len(segments_out)
        avg_obs  = sum(s["observed_lift"]  for s in segments_out) / len(segments_out)
        return {
            "segments": segments_out,
            "summary": {
                "total_sent":         total_sent,
                "total_holdout":      total_holdout,
                "avg_predicted_cate": round(avg_pred, 4),
                "avg_observed_lift":  round(avg_obs, 4),
                "n_segments":         len(segments_out),
            },
            "data_source": "campaign",
        }

    # 2b. Baseline lift — compare treatment vs. control in tenant DuckDB
    segments_out = []
    total_sent = total_holdout = 0

    try:
        with get_tenant_conn(tenant_id) as dconn:
            users_df: pd.DataFrame = dconn.execute(
                """
                SELECT company_size, channel, treatment,
                       CAST(activated AS DOUBLE) AS activated
                FROM users
                WHERE signed_up = 1
                """
            ).df()
    except Exception as exc:
        logger.error("lift: duckdb query error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to read user data", "detail": str(exc)},
        )

    for row in log_rows:
        cs = row["company_size"]
        ch = row["channel"]
        n_sent    = int(row["n_sent"])
        n_holdout = int(row["n_holdout"])
        total_sent    += n_sent
        total_holdout += n_holdout

        seg_df = users_df[(users_df["company_size"] == cs) & (users_df["channel"] == ch)]
        t_df = seg_df[seg_df["treatment"] == 1]
        c_df = seg_df[seg_df["treatment"] == 0]
        if len(t_df) < 5 or len(c_df) < 5:
            continue

        t_rate = float(t_df["activated"].mean())
        c_rate = float(c_df["activated"].mean())
        segments_out.append({
            "segment_id":     f"{cs}_{ch}",
            "company_size":   cs,
            "channel":        ch,
            "predicted_cate": round(float(row["predicted_cate"]), 4),
            "observed_lift":  round(t_rate - c_rate, 4),
            "treatment_rate": round(t_rate, 4),
            "control_rate":   round(c_rate, 4),
            "n_sent":         n_sent,
            "n_holdout":      n_holdout,
            "last_sent_at":   row["last_sent_at"],
        })

    avg_pred = sum(s["predicted_cate"] for s in segments_out) / len(segments_out) if segments_out else 0.0
    avg_obs  = sum(s["observed_lift"]  for s in segments_out) / len(segments_out) if segments_out else 0.0

    return {
        "segments": segments_out,
        "summary": {
            "total_sent":         total_sent,
            "total_holdout":      total_holdout,
            "avg_predicted_cate": round(avg_pred, 4),
            "avg_observed_lift":  round(avg_obs, 4),
            "n_segments":         len(segments_out),
        },
        "data_source": "baseline",
    }


@router.post(
    "/send-segment",
    response_model=SendSegmentResult,
    dependencies=[Depends(claude_rate_limit)],
)
def send_segment(req: SendSegmentRequest, user: CurrentUser) -> dict:
    """
    Generate personalised outreach and send to all matching contacts in the segment.

    For each contact matching (company_size × channel):
        - Generates a message via Claude (or uses fallback if API unavailable)
        - Assigns holdout deterministically via hash(segment_id + email)
        - Sends via Resend only if holdout_flag is False
        - Logs every contact (sent and held out) to contact_sends

    Returns counts of sent, held out, and failed deliveries.
    Requires RESEND_API_KEY and RESEND_FROM_EMAIL to be configured.
    Individual delivery failures do not abort the batch — errors are
    collected and returned so the caller can inspect them.
    """
    if req.tone not in {"warm", "direct", "technical"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "tone must be warm, direct, or technical"},
        )

    tenant_id = user["email"]

    # 1. Fetch matching contacts from SQLite (tenant-scoped)
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        contacts = conn.execute(
            """
            SELECT id, email, first_name, company
            FROM contacts
            WHERE tenant_id = ? AND company_size = ? AND channel = ?
            ORDER BY id
            """,
            (tenant_id, req.company_size, req.channel),
        ).fetchall()
    except Exception as exc:
        conn.close()
        logger.error("send-segment: contact fetch error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to fetch contacts", "detail": str(exc)},
        )

    if not contacts:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "No contacts found for this segment",
                "detail": (
                    f"Upload a CSV with company_size='{req.company_size}' "
                    f"and channel='{req.channel}' before sending."
                ),
            },
        )

    # 2. Generate one shared message for the segment (not per-contact —
    #    CATE is segment-level, not individual-level, so one message per segment
    #    is statistically coherent). Personalisation of greeting is done below.
    segment_dict = {
        "cate_estimate": req.cate_estimate,
        "company_size":  req.company_size,
        "channel":       req.channel,
        "segment_id":    req.segment_id,
    }

    try:
        template = generate_outreach(
            segment=segment_dict,
            product_context=req.product_context,
            tone=req.tone,
            log_to_db=False,   # we log per-contact below instead
            tenant_id=tenant_id,
        )
    except OutreachError as exc:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Segment below CATE threshold", "detail": str(exc)},
        )

    # 3. Send to each contact, log every row (sent + holdout)
    sent = held_out = failed = 0
    errors: list[str] = []

    try:
        for contact in contacts:
            email      = contact["email"]
            first_name = contact["first_name"] or ""
            company    = contact["company"] or ""

            # Personalise greeting if we have a name
            body = template["body"]
            if first_name:
                body = f"Hi {first_name},\n\n{body}"

            # Deterministic holdout assignment keyed to segment + email
            from core.outreach import _is_holdout
            holdout = _is_holdout(req.segment_id, email)

            provider_id: str | None = None
            row_status = "holdout" if holdout else "sent"

            if not holdout:
                try:
                    provider_id = send_email(
                        to=email,
                        subject=template["subject"],
                        body=body,
                    )
                    sent += 1
                except EmailDeliveryError as exc:
                    errors.append(f"{email}: {exc}")
                    failed += 1
                    row_status = "failed"
            else:
                held_out += 1

            # Log every contact regardless of holdout/failure
            try:
                conn.execute(
                    """
                    INSERT INTO contact_sends
                        (tenant_id, email, contact_id, segment_id, company_size, channel,
                         cate_estimate, tone, subject, body, cta,
                         is_holdout, provider_message_id, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        email, contact["id"], req.segment_id,
                        req.company_size, req.channel,
                        req.cate_estimate, req.tone,
                        template["subject"], body, template["cta"],
                        int(holdout), provider_id, row_status,
                    ),
                )
            except Exception as log_exc:
                logger.warning("Failed to log contact send for %s: %s", email, log_exc)

        conn.commit()
    finally:
        conn.close()

    logger.info(
        "send-segment | segment=%s | sent=%d | held_out=%d | failed=%d",
        req.segment_id, sent, held_out, failed,
    )
    return {"sent": sent, "held_out": held_out, "failed": failed, "errors": errors[:20]}
