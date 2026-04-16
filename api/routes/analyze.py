"""
POST /api/analyze — funnel summary + CUPED ATE + SRM check.

Queries the seeded DuckDB funnel data, optionally filtered by segment,
and returns:
  - Funnel conversion rates at each stage
  - CUPED-adjusted ATE (if enough signed-up users)
  - SRM result (always run before reporting ATE)
  - Daily trend data for the UI chart

Statistical correctness notes:
  - CUPED outcome is the OBSERVED binary activation flag (activated=0/1),
    NOT the latent activation_prob. Binary outcomes are NOT winsorized.
  - Only the continuous pre_activation_rate covariate is winsorized.
  - SRM uses the caller-supplied expected_split (default 0.5); do not
    assume 50/50 if the experiment design used a different split.
  - The daily_summary filter is rebuilt separately — it only supports
    date-based conditions because that table has no segment columns.
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.db import get_tenant_conn, tenant_has_data
from api.deps import OptionalUser, tenant_id_from
from core.causal import CausalEstimationError, cuped_adjustment, detect_srm
from core.preprocess import winsorize

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["analyze"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    company_size: str | None = Field(None, description="Filter: SMB | mid_market | enterprise")
    channel: str | None = Field(None, description="Filter: organic | paid_search | social | referral | email")
    date_from: str | None = Field(None, description="ISO date filter start (YYYY-MM-DD)")
    date_to: str | None = Field(None, description="ISO date filter end (YYYY-MM-DD)")
    expected_split: float = Field(
        0.5,
        ge=0.01,
        le=0.99,
        description=(
            "Intended treatment fraction used during randomisation. "
            "Must match the experiment design — wrong value will cause "
            "false SRM positives or negatives."
        ),
    )


class FunnelStage(BaseModel):
    stage: str
    n: int
    rate_from_prev: float | None
    rate_from_impression: float


class SrmResult(BaseModel):
    srm_detected: bool
    p_value: float
    observed_split: float
    expected_split: float
    recommendation: str


class CupedResult(BaseModel):
    ate: float
    ate_se: float
    p_value: float
    ci_lower: float
    ci_upper: float
    variance_reduction_pct: float
    n_treatment: int
    n_control: int


class AnalyzeResponse(BaseModel):
    total_users: int
    funnel: list[FunnelStage]
    srm: SrmResult
    cuped: CupedResult | None
    daily_trend: list[dict]
    filters_applied: dict


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest, user: OptionalUser) -> dict:
    """
    Return funnel summary, SRM check, and CUPED-adjusted ATE.

    Authenticated requests analyse the caller's uploaded data.
    Unauthenticated requests analyse the shared synthetic demo dataset.
    """
    tenant_id = tenant_id_from(user)
    if not tenant_has_data(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "No data uploaded yet",
                "detail": "Upload a funnel CSV on the Data tab to analyse your own data.",
            },
        )
    try:
        return _run_analysis(req, tenant_id)
    except Exception as exc:
        logger.error("analyze error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Analysis failed", "detail": str(exc)},
        )


def _build_where(req: AnalyzeRequest) -> tuple[str, list]:
    """Build parameterised WHERE clause for the users table."""
    clauses, params = [], []
    if req.company_size:
        clauses.append("company_size = ?")
        params.append(req.company_size)
    if req.channel:
        clauses.append("channel = ?")
        params.append(req.channel)
    if req.date_from:
        clauses.append("impression_date >= ?")
        params.append(req.date_from)
    if req.date_to:
        clauses.append("impression_date <= ?")
        params.append(req.date_to)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _build_daily_where(req: AnalyzeRequest) -> tuple[str, list]:
    """
    Build WHERE clause for the daily_summary table.

    daily_summary has no company_size or channel columns — only date.
    Attempting to filter by segment on this table causes a column-not-found
    error in DuckDB.  This function only passes date-based conditions.
    """
    clauses, params = [], []
    if req.date_from:
        clauses.append("date >= ?")
        params.append(req.date_from)
    if req.date_to:
        clauses.append("date <= ?")
        params.append(req.date_to)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def _run_analysis(req: AnalyzeRequest, tenant_id: str = "demo") -> dict:
    where, params = _build_where(req)
    daily_where, daily_params = _build_daily_where(req)

    with get_tenant_conn(tenant_id) as conn:
        users_df = conn.execute(
            f"""
            SELECT user_id, treatment, clicked, signed_up, activated, converted,
                   pre_activation_rate,
                   CAST(activated AS DOUBLE) AS post_activation_rate
            FROM users
            {where}
            """,
            params,
        ).df()

        daily_df = conn.execute(
            f"""
            SELECT date, impressions, clicks, signups, activations, conversions,
                   treatment_activation_rate, control_activation_rate
            FROM daily_summary
            {daily_where}
            ORDER BY date
            """,
            daily_params,
        ).df()

    if users_df.empty:
        raise ValueError("No users match the applied filters.")

    n_total = len(users_df)

    # Funnel counts
    n_click    = int(users_df["clicked"].sum())
    n_signup   = int(users_df["signed_up"].sum())
    n_activate = int(users_df["activated"].sum())
    n_convert  = int(users_df["converted"].sum())

    funnel = [
        FunnelStage(stage="impression",  n=n_total,    rate_from_prev=None,                                                   rate_from_impression=1.0),
        FunnelStage(stage="click",       n=n_click,    rate_from_prev=round(n_click/n_total, 4) if n_total else 0,            rate_from_impression=round(n_click/n_total, 4) if n_total else 0),
        FunnelStage(stage="signup",      n=n_signup,   rate_from_prev=round(n_signup/n_click, 4) if n_click else 0,           rate_from_impression=round(n_signup/n_total, 4) if n_total else 0),
        FunnelStage(stage="activation",  n=n_activate, rate_from_prev=round(n_activate/n_signup, 4) if n_signup else 0,       rate_from_impression=round(n_activate/n_total, 4) if n_total else 0),
        FunnelStage(stage="conversion",  n=n_convert,  rate_from_prev=round(n_convert/n_activate, 4) if n_activate else 0,    rate_from_impression=round(n_convert/n_total, 4) if n_total else 0),
    ]

    # SRM — always run first, using the experiment's actual intended split
    n_t = int(users_df["treatment"].sum())
    n_c = n_total - n_t
    srm_result = detect_srm(n_t, n_c, expected_split=req.expected_split)
    srm = SrmResult(
        srm_detected=srm_result["srm_detected"],
        p_value=round(srm_result["p_value"], 6),
        observed_split=round(srm_result["observed_split"], 4),
        expected_split=srm_result["expected_split"],
        recommendation=srm_result["recommendation"],
    )

    # CUPED — only on signed-up users (activation is the outcome)
    # Outcome: binary activated flag (0/1) — do NOT winsorize binary outcomes.
    # Covariate: pre_activation_rate (continuous) — winsorize to reduce outlier influence.
    cuped: CupedResult | None = None
    signed_up_df = users_df[users_df["signed_up"] == 1].copy()
    if len(signed_up_df) >= 200:
        try:
            # Only winsorize the continuous covariate, not the binary outcome
            signed_up_df["pre_w"] = winsorize(signed_up_df["pre_activation_rate"])
            res = cuped_adjustment(signed_up_df, "post_activation_rate", "pre_w", "treatment")
            cuped = CupedResult(
                ate=round(res["ate"], 5),
                ate_se=round(res["ate_se"], 5),
                p_value=round(res["p_value"], 6),
                ci_lower=round(res["ci_lower"], 5),
                ci_upper=round(res["ci_upper"], 5),
                variance_reduction_pct=round(res["variance_reduction_pct"], 2),
                n_treatment=res["n_treatment"],
                n_control=res["n_control"],
            )
        except CausalEstimationError as exc:
            logger.warning("CUPED skipped: %s", exc)

    daily_trend = daily_df.to_dict(orient="records") if not daily_df.empty else []

    return {
        "total_users":     n_total,
        "funnel":          funnel,
        "srm":             srm,
        "cuped":           cuped,
        "daily_trend":     daily_trend,
        "filters_applied": {k: v for k, v in req.model_dump().items() if v is not None and k != "expected_split"},
    }
