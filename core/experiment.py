"""
Experiment designer: statistical power calculation with optional CUPED adjustment.

Produces an experiment spec that answers:
    - How many users do we need per arm?
    - How many days does that take at current traffic?
    - What is the recommended treatment split?
    - What guardrail metrics should we monitor?

CUPED toggle: when use_cuped=True, the required N is reduced by
    (1 - variance_reduction) because CUPED pre-reduces metric variance.

All thresholds and parameters are read from environment variables where
applicable, with arguments taking precedence.
"""

import logging
import os

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ExperimentDesignError(ValueError):
    """Raised when experiment parameters are invalid or under-powered."""


def design_experiment(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int | None = None,
    use_cuped: bool = True,
    variance_reduction: float | None = None,
    treatment_split: float = 0.50,
    guardrail_metrics: list[str] | None = None,
) -> dict:
    """
    Design a two-sample proportion experiment with optional CUPED adjustment.

    Uses the standard two-proportion z-test sample size formula:

        n = (z_α/2 + z_β)² × [p1(1-p1) + p2(1-p2)] / (p1 - p2)²

    where p2 = baseline_rate + mde (the expected rate under treatment).

    With CUPED (use_cuped=True):
        n_cuped = ceil(n × (1 - variance_reduction))

    Args:
        baseline_rate:      Current activation/conversion rate (e.g. 0.42 = 42%).
        mde:                Minimum detectable effect as an absolute rate change
                            (e.g. 0.05 = 5pp lift). Must be positive.
        alpha:              Type I error rate (significance level). Default 0.05.
        power:              Statistical power (1 - β). Default 0.80.
        daily_traffic:      Average users entering the funnel per day. Used to
                            compute experiment duration. None means duration is
                            not included in the output.
        use_cuped:          Apply CUPED variance reduction to required N.
        variance_reduction: Expected fractional variance reduction from CUPED
                            (e.g. 0.30 = 30%). Reads CUPED_VARIANCE_REDUCTION_TARGET
                            env var if None. Default 0.30.
        treatment_split:    Fraction of users assigned to treatment (default 0.50).
        guardrail_metrics:  List of metric names to monitor for regressions.
                            Defaults to ["unsubscribe_rate", "spam_complaint_rate"].

    Returns:
        Dict with keys:
            required_n_per_arm      — users needed per arm (ceil)
            required_n_total        — total users needed across both arms
            duration_days           — experiment duration at daily_traffic (or None)
            treatment_split         — recommended split (echoed back)
            alpha                   — significance level used
            power                   — power level used
            baseline_rate           — baseline rate used
            mde                     — MDE used
            treatment_rate          — expected treatment rate (baseline + mde)
            cuped_applied           — whether CUPED reduction was applied
            variance_reduction_pct  — CUPED variance reduction % (or 0)
            naive_n_per_arm         — N without CUPED (for comparison)
            primary_metric          — "activation_rate" (default)
            guardrail_metrics       — list of guardrail metric names
            notes                   — human-readable design summary

    Raises:
        ExperimentDesignError: If parameters are out of range or the
                               design is infeasible (e.g. treatment_rate > 1).
    """
    _validate_inputs(baseline_rate, mde, alpha, power, treatment_split)

    vr = variance_reduction
    if vr is None:
        vr = float(os.getenv("CUPED_VARIANCE_REDUCTION_TARGET", "0.30"))
    if not (0.0 <= vr < 1.0):
        raise ExperimentDesignError(
            f"variance_reduction must be in [0, 1), got {vr}."
        )

    treatment_rate = baseline_rate + mde
    if treatment_rate > 1.0:
        raise ExperimentDesignError(
            f"baseline_rate ({baseline_rate}) + mde ({mde}) = {treatment_rate} > 1.0. "
            f"Reduce the MDE."
        )

    naive_n = _required_n(baseline_rate, treatment_rate, alpha, power)

    if use_cuped and vr > 0:
        adjusted_n = int(np.ceil(naive_n * (1 - vr)))
        cuped_applied = True
    else:
        adjusted_n = naive_n
        cuped_applied = False
        vr = 0.0

    n_total = int(np.ceil(adjusted_n / treatment_split))
    duration = None
    if daily_traffic is not None and daily_traffic > 0:
        duration = int(np.ceil(n_total / daily_traffic))

    guardrails = guardrail_metrics or ["unsubscribe_rate", "spam_complaint_rate"]

    cuped_reduction_pct = vr * 100 if cuped_applied else 0.0
    notes = _build_notes(naive_n, adjusted_n, cuped_applied, vr, duration)

    logger.info(
        "design_experiment | baseline=%.3f mde=%.3f alpha=%.3f power=%.3f "
        "| naive_n=%d adjusted_n=%d duration=%s",
        baseline_rate, mde, alpha, power, naive_n, adjusted_n, duration,
    )

    return {
        "required_n_per_arm":     adjusted_n,
        "required_n_total":       n_total,
        "duration_days":          duration,
        "treatment_split":        treatment_split,
        "alpha":                  alpha,
        "power":                  power,
        "baseline_rate":          baseline_rate,
        "mde":                    mde,
        "treatment_rate":         treatment_rate,
        "cuped_applied":          cuped_applied,
        "variance_reduction_pct": cuped_reduction_pct,
        "naive_n_per_arm":        naive_n,
        "primary_metric":         "activation_rate",
        "guardrail_metrics":      guardrails,
        "notes":                  notes,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _required_n(
    p1: float,
    p2: float,
    alpha: float,
    power: float,
) -> int:
    """
    Compute per-arm sample size for a two-proportion z-test.

    Formula:
        n = (z_α/2 + z_β)² × [p1(1-p1) + p2(1-p2)] / (p1 - p2)²

    Args:
        p1:    Baseline (control) proportion.
        p2:    Expected treatment proportion.
        alpha: Significance level (two-sided).
        power: Desired power (1 - β).

    Returns:
        Required sample size per arm (ceiling integer).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)

    numerator   = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    denominator = (p2 - p1) ** 2

    return int(np.ceil(numerator / denominator))


def _validate_inputs(
    baseline_rate: float,
    mde: float,
    alpha: float,
    power: float,
    treatment_split: float,
) -> None:
    """Validate experiment design parameters, raising ExperimentDesignError if invalid."""
    if not (0.0 < baseline_rate < 1.0):
        raise ExperimentDesignError(
            f"baseline_rate must be in (0, 1), got {baseline_rate}."
        )
    if mde <= 0:
        raise ExperimentDesignError(f"mde must be positive, got {mde}.")
    if not (0.0 < alpha < 0.5):
        raise ExperimentDesignError(
            f"alpha must be in (0, 0.5), got {alpha}. "
            f"Common values: 0.05 (standard), 0.01 (conservative)."
        )
    if not (0.5 <= power < 1.0):
        raise ExperimentDesignError(
            f"power must be in [0.5, 1.0), got {power}. "
            f"Common values: 0.80 (standard), 0.90 (high confidence)."
        )
    if not (0.1 <= treatment_split <= 0.9):
        raise ExperimentDesignError(
            f"treatment_split must be in [0.1, 0.9], got {treatment_split}."
        )


def _build_notes(
    naive_n: int,
    adjusted_n: int,
    cuped_applied: bool,
    variance_reduction: float,
    duration_days: int | None,
) -> str:
    """Build a human-readable design summary note."""
    lines = []
    if cuped_applied:
        savings_pct = (1 - adjusted_n / naive_n) * 100
        lines.append(
            f"CUPED reduces required N by {savings_pct:.0f}% "
            f"({naive_n:,} → {adjusted_n:,} per arm) assuming "
            f"{variance_reduction * 100:.0f}% variance reduction."
        )
    else:
        lines.append(f"Required N per arm: {adjusted_n:,} (no CUPED adjustment).")

    if duration_days is not None:
        lines.append(f"Estimated duration: {duration_days} days at given daily traffic.")

    return " ".join(lines)
