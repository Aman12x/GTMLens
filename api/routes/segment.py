"""
POST /api/segment/cate — CATE estimation by segment with BH correction.

Fits a T-Learner on all signed-up users, then aggregates per-user CATE
estimates to the (company_size × channel) segment level. Segment-level
significance is tested via Welch's t-test on the binary activation outcome
within each segment (treated vs. control), with Benjamini-Hochberg
correction applied across all segments simultaneously.

This is the core causal capability of GTMLens:
    - T-Learner gives a per-user estimate of the treatment effect
    - Segment aggregation identifies which buckets have the highest lift
    - BH correction controls false discovery rate — avoids cherry-picking
    - recommended_for_outreach flags segments that clear both the
      significance bar AND the CATE_UPLIFT_THRESHOLD

CLAUDE.md correctness rules applied here:
    - T-Learner is the default method (defensible; separate response surfaces)
    - CausalForest gated at N > 5000 per arm per segment (not global N)
    - BH correction used — never Bonferroni
    - Log-transform applied to continuous features before T-Learner
    - Outcome is binary activated flag (not the latent activation_prob)
"""

import logging
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from scipy import stats

from api.db import get_tenant_conn, tenant_has_data
from api.deps import OptionalUser, tenant_id_from
from core.causal import CausalEstimationError, bh_correction, estimate_cate
from core.preprocess import log_transform

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/segment", tags=["segment"])

_CATE_THRESHOLD = float(os.getenv("CATE_UPLIFT_THRESHOLD", "0.40"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CateRequest(BaseModel):
    method: str = Field("t_learner", description="t_learner | s_learner | causal_forest")
    min_segment_n: int = Field(30, ge=10, description="Minimum users per segment to include in output")
    apply_bh: bool = Field(True, description="Apply Benjamini-Hochberg FDR correction")
    bh_alpha: float = Field(0.05, gt=0, lt=1, description="FDR level for BH correction")
    date_from: str | None = Field(None, description="ISO date filter start (YYYY-MM-DD)")
    date_to: str | None = Field(None, description="ISO date filter end (YYYY-MM-DD)")


class SegmentCateResult(BaseModel):
    company_size: str
    channel: str
    mean_cate: float
    n_treatment: int
    n_control: int
    segment_ate: float
    p_value_raw: float
    significant_bh: bool
    recommended_for_outreach: bool


class CateResponse(BaseModel):
    segments: list[SegmentCateResult]
    method: str
    n_users: int
    bh_applied: bool
    bh_alpha: float
    cate_threshold: float
    n_significant: int


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/cate", response_model=CateResponse)
def segment_cate(req: CateRequest, user: OptionalUser) -> dict:
    """
    Estimate CATE for every (company_size × channel) segment.

    Authenticated requests run on the caller's uploaded data.
    Unauthenticated requests run on the synthetic demo dataset.
    """
    if req.method not in {"t_learner", "s_learner", "causal_forest"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Invalid method", "detail": "method must be t_learner, s_learner, or causal_forest"},
        )
    tenant_id = tenant_id_from(user)
    if not tenant_has_data(tenant_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "No data uploaded yet",
                "detail": "Upload a funnel CSV on the Data tab before running CATE estimation.",
            },
        )
    try:
        return _run_cate(req, tenant_id)
    except CausalEstimationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "CATE estimation failed", "detail": str(exc)},
        )
    except Exception as exc:
        logger.error("segment/cate error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Segment CATE failed", "detail": str(exc)},
        )


def _run_cate(req: CateRequest, tenant_id: str = "demo") -> dict:
    # 1. Load signed-up users (activation is the outcome)
    clauses, params = [], []
    if req.date_from:
        clauses.append("impression_date >= ?")
        params.append(req.date_from)
    if req.date_to:
        clauses.append("impression_date <= ?")
        params.append(req.date_to)

    with get_tenant_conn(tenant_id) as conn:
        df = conn.execute(
            f"""
            SELECT user_id, treatment, activated, company_size, channel, industry,
                   pre_activation_rate
            FROM users
            WHERE signed_up = 1
            {("AND " + " AND ".join(clauses)) if clauses else ""}
            ORDER BY user_id
            """,
            params,
        ).df()

    if len(df) < 50:
        raise CausalEstimationError(
            f"Too few signed-up users ({len(df)}) for CATE estimation. "
            "Run demo reset to regenerate the dataset."
        )

    # 2. Feature engineering
    # One-hot encode categoricals; drop_first avoids perfect multicollinearity
    df_enc = pd.get_dummies(df, columns=["company_size", "channel", "industry"], drop_first=True)

    # Log-transform the continuous pre_activation_rate (CATE model feature)
    df_enc["pre_activation_rate_log"] = log_transform(df["pre_activation_rate"])

    feature_cols = [
        c for c in df_enc.columns
        if c.startswith(("company_size_", "channel_", "industry_"))
        or c == "pre_activation_rate_log"
    ]

    # 3. Fit T-Learner and get per-user CATE estimates
    df_with_cate = estimate_cate(
        df_enc,
        outcome_col="activated",
        treatment_col="treatment",
        feature_cols=feature_cols,
        method=req.method,  # type: ignore[arg-type]
    )

    # Re-attach original segment labels (dropped during encoding)
    df_with_cate["company_size"] = df["company_size"].values
    df_with_cate["channel"] = df["channel"].values

    # 4. Segment-level aggregation
    segments: list[dict] = []
    p_values_raw: list[float] = []

    for (company_size, channel), grp in df_with_cate.groupby(["company_size", "channel"]):
        t_grp = grp[grp["treatment"] == 1]
        c_grp = grp[grp["treatment"] == 0]

        n_t = len(t_grp)
        n_c = len(c_grp)

        if n_t < 5 or n_c < 5 or (n_t + n_c) < req.min_segment_n:
            continue

        mean_cate = float(grp["cate_estimate"].mean())

        # Within-segment t-test on the binary outcome (activated)
        t_activated = t_grp["activated"].to_numpy(dtype=float)
        c_activated = c_grp["activated"].to_numpy(dtype=float)
        segment_ate = float(t_activated.mean() - c_activated.mean())

        if np.var(t_activated, ddof=1) == 0 and np.var(c_activated, ddof=1) == 0:
            p_raw = 1.0  # both arms are constant — no evidence of effect
        else:
            _, p_raw = stats.ttest_ind(t_activated, c_activated, equal_var=False)

        p_values_raw.append(float(p_raw))
        segments.append({
            "company_size":  str(company_size),
            "channel":       str(channel),
            "mean_cate":     round(mean_cate, 4),
            "n_treatment":   n_t,
            "n_control":     n_c,
            "segment_ate":   round(segment_ate, 4),
            "p_value_raw":   round(float(p_raw), 6),
        })

    if not segments:
        raise CausalEstimationError(
            f"No segments met the minimum N={req.min_segment_n} threshold."
        )

    # 5. BH correction across all segments simultaneously
    if req.apply_bh and len(p_values_raw) > 1:
        significant = bh_correction(p_values_raw, alpha=req.bh_alpha)
    else:
        # Single segment or BH disabled: use raw p-value threshold
        significant = [p < req.bh_alpha for p in p_values_raw]

    # CATE_UPLIFT_THRESHOLD is a *fraction* (0.40 = top 40% of segments by CATE).
    # Convert to an absolute cutoff using the empirical percentile so that the
    # flag is meaningful regardless of the effect-size scale.
    top_fraction = float(os.getenv("CATE_UPLIFT_THRESHOLD", "0.40"))
    cate_values = [s["mean_cate"] for s in segments]
    if cate_values:
        cutoff_percentile = (1.0 - top_fraction) * 100
        cate_threshold = float(np.percentile(cate_values, cutoff_percentile))
    else:
        cate_threshold = 0.0

    for seg, sig in zip(segments, significant):
        seg["significant_bh"] = bool(sig)
        seg["recommended_for_outreach"] = bool(sig and seg["mean_cate"] >= cate_threshold)

    # Sort by mean_cate descending
    segments.sort(key=lambda s: s["mean_cate"], reverse=True)

    n_significant = sum(s["significant_bh"] for s in segments)

    logger.info(
        "segment/cate | method=%s | n_users=%d | n_segments=%d | n_significant=%d",
        req.method, len(df), len(segments), n_significant,
    )

    return {
        "segments":       segments,
        "method":         req.method,
        "n_users":        len(df),
        "bh_applied":     req.apply_bh and len(p_values_raw) > 1,
        "bh_alpha":       req.bh_alpha,
        "cate_threshold": cate_threshold,
        "n_significant":  n_significant,
    }
