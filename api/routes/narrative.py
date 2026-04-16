"""
POST /api/narrative — PM-ready experiment result narrative via Claude.

Wraps core/narrative.generate_result_narrative(). The route never returns 500
for Claude API failures — the core module handles those with a graceful fallback
that is clearly marked (is_fallback: True) so the UI can adapt.

Expected call flow:
    1. Client runs POST /api/analyze to get cuped results
    2. Client sends the cuped block + metric hierarchy to POST /api/narrative
    3. UI renders the structured narrative (outcome, driver, guardrails, recommendation)

Rate limited to 20 requests / minute / IP (protects Anthropic API key budget).
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from api.rate_limit import claude_rate_limit
from core.narrative import generate_result_narrative

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/narrative", tags=["narrative"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ExperimentResultInput(BaseModel):
    """Subset of cuped_adjustment() output required by the narrative generator."""
    ate: float = Field(..., description="Average treatment effect (absolute rate change)")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Two-sided p-value")
    ci_lower: float | None = Field(None, description="95% CI lower bound")
    ci_upper: float | None = Field(None, description="95% CI upper bound")
    ate_se: float | None = Field(None, description="Standard error of the ATE")
    variance_reduction_pct: float | None = Field(None, description="CUPED variance reduction %")
    n_treatment: int | None = Field(None, description="Users in treatment arm")
    n_control: int | None = Field(None, description="Users in control arm")
    guardrail_results: dict | None = Field(
        None, description="Metric name → held/breached description"
    )
    segment_breakdown: list[dict] | None = Field(
        None, description="Per-segment ATE results [{segment, ate, p_value}]"
    )


class MetricHierarchyInput(BaseModel):
    nsm: str = Field("weekly_active_accounts", description="North Star Metric name")
    primary_metric: str = Field("activation_rate", description="What the experiment measures")
    secondary_metrics: list[str] = Field(
        default_factory=list, description="Secondary metric names"
    )
    guardrail_metrics: list[str] = Field(
        default_factory=list, description="Guardrail metric names to assess"
    )


class NarrativeRequest(BaseModel):
    experiment_result: ExperimentResultInput
    metric_hierarchy: MetricHierarchyInput = Field(default_factory=MetricHierarchyInput)
    recommendation: str = Field(
        "iterate",
        description="Caller's suggested recommendation (ship | iterate | abort). "
                    "Claude may override based on evidence.",
    )


class NarrativeResponse(BaseModel):
    # is_fallback is populated from the "_error" key in the core module's
    # fallback dict via the alias — Pydantic v2 maps "_error" → is_fallback.
    model_config = ConfigDict(populate_by_name=True)

    outcome: str
    driver: str
    guardrails: dict
    recommendation: str
    rationale: str
    is_fallback: bool = Field(False, alias="_error", description="True when Claude API unavailable")


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=NarrativeResponse,
    dependencies=[Depends(claude_rate_limit)],
)
def narrative(req: NarrativeRequest) -> dict:
    """
    Generate a structured PM narrative for an A/B experiment result.

    Input: CUPED result block + metric hierarchy (from /api/analyze).
    Output: {outcome, driver, guardrails, recommendation, rationale}.

    Claude may override the caller's recommendation hint based on the
    evidence (p-value, CI, guardrail state). The recommendation will always
    be exactly one of: SHIP, ITERATE, ABORT.

    If the Claude API is unavailable, returns a fallback dict with
    is_fallback=True. This route never returns 5xx for Claude API failures.

    Rate limited: 20 requests per minute per IP (Claude API cost protection).
    """
    rec = req.recommendation.lower()
    if rec not in {"ship", "iterate", "abort"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Invalid recommendation",
                "detail": "recommendation must be ship, iterate, or abort",
            },
        )

    exp_result = req.experiment_result.model_dump(exclude_none=True)
    hierarchy = req.metric_hierarchy.model_dump()

    try:
        result = generate_result_narrative(
            experiment_result=exp_result,
            metric_hierarchy=hierarchy,
            recommendation=rec,  # type: ignore[arg-type]
            log_to_db=True,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Invalid experiment result", "detail": str(exc)},
        )
    except Exception as exc:
        logger.error("narrative generation error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Narrative generation failed", "detail": str(exc)},
        )

    return result
