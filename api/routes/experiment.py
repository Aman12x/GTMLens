"""
POST /api/experiment/design — statistical power calculation with optional CUPED.

Delegates to core/experiment.design_experiment() and returns the full spec:
  - Required N per arm (with and without CUPED)
  - Experiment duration at current traffic
  - Guardrail metrics to monitor
  - Human-readable design notes
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from core.experiment import ExperimentDesignError, design_experiment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiment", tags=["experiment"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ExperimentDesignRequest(BaseModel):
    baseline_rate: float = Field(..., gt=0, lt=1, description="Current activation rate (e.g. 0.42)")
    mde: float = Field(..., gt=0, description="Minimum detectable effect as absolute rate change (e.g. 0.05)")
    alpha: float = Field(0.05, gt=0, lt=0.5, description="Type I error rate (default 0.05)")
    power: float = Field(0.80, ge=0.5, lt=1.0, description="Statistical power (default 0.80)")
    daily_traffic: int | None = Field(None, gt=0, description="Average daily users entering funnel")
    use_cuped: bool = Field(True, description="Apply CUPED variance reduction to required N")
    variance_reduction: float | None = Field(
        None, ge=0.0, lt=1.0,
        description="Expected CUPED variance reduction (0–1). Reads env if omitted.",
    )
    treatment_split: float = Field(0.50, ge=0.1, le=0.9, description="Fraction assigned to treatment")
    guardrail_metrics: list[str] | None = Field(
        None, description="Metrics to monitor for regressions"
    )

    @field_validator("mde")
    @classmethod
    def mde_not_too_large(cls, v: float) -> float:
        if v >= 1.0:
            raise ValueError("mde must be less than 1.0 (absolute rate change, not percentage)")
        return v


class ExperimentDesignResponse(BaseModel):
    required_n_per_arm: int
    required_n_total: int
    naive_n_per_arm: int
    duration_days: int | None
    treatment_split: float
    alpha: float
    power: float
    baseline_rate: float
    mde: float
    treatment_rate: float
    cuped_applied: bool
    variance_reduction_pct: float
    primary_metric: str
    guardrail_metrics: list[str]
    notes: str


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/design", response_model=ExperimentDesignResponse)
def experiment_design(req: ExperimentDesignRequest) -> dict:
    """
    Compute required sample size for a two-proportion experiment.

    With use_cuped=True, N is reduced by (1 - variance_reduction) to account
    for the variance reduction CUPED provides. This is the pre-experiment
    planning step — run CUPED-adjusted analysis via POST /api/analyze.
    """
    try:
        return design_experiment(
            baseline_rate=req.baseline_rate,
            mde=req.mde,
            alpha=req.alpha,
            power=req.power,
            daily_traffic=req.daily_traffic,
            use_cuped=req.use_cuped,
            variance_reduction=req.variance_reduction,
            treatment_split=req.treatment_split,
            guardrail_metrics=req.guardrail_metrics,
        )
    except ExperimentDesignError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "Invalid experiment parameters", "detail": str(exc)},
        )
    except Exception as exc:
        logger.error("experiment design error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Experiment design failed", "detail": str(exc)},
        )
