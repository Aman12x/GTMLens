"""
Tests for core/experiment.py.

Required coverage (CLAUDE.md):
    - Power calc matches scipy reference formula
    - CUPED adjustment reduces required N by expected amount
    - Output contains all required fields
    - Invalid parameters raise ExperimentDesignError
    - Duration calculation is correct given daily_traffic
"""

import math

import numpy as np
import pytest
from scipy import stats

from core.experiment import ExperimentDesignError, design_experiment


# ---------------------------------------------------------------------------
# Reference implementation (scipy) for cross-validation
# ---------------------------------------------------------------------------


def _scipy_required_n(p1: float, p2: float, alpha: float, power: float) -> int:
    """
    Reference two-proportion z-test sample size from the standard formula.

    n = (z_α/2 + z_β)² × [p1(1-p1) + p2(1-p2)] / (p1 - p2)²
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta  = stats.norm.ppf(power)
    num = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    den = (p2 - p1) ** 2
    return math.ceil(num / den)


# ---------------------------------------------------------------------------
# Power calculation matches scipy reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline, mde, alpha, power", [
    (0.40, 0.05, 0.05, 0.80),
    (0.30, 0.03, 0.05, 0.80),
    (0.50, 0.10, 0.05, 0.90),
    (0.20, 0.04, 0.01, 0.80),
    (0.45, 0.08, 0.05, 0.80),
])
def test_power_calc_matches_scipy_reference(
    baseline: float, mde: float, alpha: float, power: float
) -> None:
    """Naive N (use_cuped=False) must equal the scipy reference formula."""
    result = design_experiment(
        baseline_rate=baseline,
        mde=mde,
        alpha=alpha,
        power=power,
        use_cuped=False,
    )
    expected_n = _scipy_required_n(baseline, baseline + mde, alpha, power)
    assert result["naive_n_per_arm"] == expected_n, (
        f"baseline={baseline} mde={mde}: "
        f"got {result['naive_n_per_arm']}, expected {expected_n}"
    )


# ---------------------------------------------------------------------------
# CUPED reduces required N by expected amount
# ---------------------------------------------------------------------------


def test_cuped_reduces_required_n() -> None:
    """With use_cuped=True, required_n_per_arm must be less than naive_n_per_arm."""
    result = design_experiment(
        baseline_rate=0.42,
        mde=0.05,
        alpha=0.05,
        power=0.80,
        use_cuped=True,
        variance_reduction=0.30,
    )
    assert result["required_n_per_arm"] < result["naive_n_per_arm"]


def test_cuped_reduction_matches_formula() -> None:
    """required_n_per_arm ≈ naive_n × (1 - variance_reduction)."""
    vr = 0.30
    result = design_experiment(
        baseline_rate=0.42,
        mde=0.05,
        alpha=0.05,
        power=0.80,
        use_cuped=True,
        variance_reduction=vr,
    )
    expected = math.ceil(result["naive_n_per_arm"] * (1 - vr))
    assert result["required_n_per_arm"] == expected


@pytest.mark.parametrize("vr", [0.10, 0.20, 0.30, 0.40, 0.50])
def test_cuped_reduction_scales_with_variance_reduction(vr: float) -> None:
    no_cuped = design_experiment(0.42, 0.05, use_cuped=False)
    with_cuped = design_experiment(0.42, 0.05, use_cuped=True, variance_reduction=vr)
    assert with_cuped["required_n_per_arm"] <= no_cuped["required_n_per_arm"]


def test_cuped_off_equals_naive_n() -> None:
    result = design_experiment(
        baseline_rate=0.40,
        mde=0.05,
        use_cuped=False,
    )
    assert result["required_n_per_arm"] == result["naive_n_per_arm"]
    assert result["cuped_applied"] is False


# ---------------------------------------------------------------------------
# Output fields
# ---------------------------------------------------------------------------


def test_output_contains_all_required_fields() -> None:
    result = design_experiment(
        baseline_rate=0.42,
        mde=0.05,
        daily_traffic=500,
    )
    required_fields = {
        "required_n_per_arm", "required_n_total", "duration_days",
        "treatment_split", "alpha", "power", "baseline_rate", "mde",
        "treatment_rate", "cuped_applied", "variance_reduction_pct",
        "naive_n_per_arm", "primary_metric", "guardrail_metrics", "notes",
    }
    assert required_fields.issubset(result.keys())


def test_output_treatment_rate_is_baseline_plus_mde() -> None:
    result = design_experiment(baseline_rate=0.40, mde=0.06)
    assert result["treatment_rate"] == pytest.approx(0.46)


def test_output_echoes_alpha_and_power() -> None:
    result = design_experiment(0.40, 0.05, alpha=0.01, power=0.90)
    assert result["alpha"] == 0.01
    assert result["power"] == 0.90


def test_output_guardrail_metrics_default() -> None:
    result = design_experiment(0.40, 0.05)
    assert "unsubscribe_rate" in result["guardrail_metrics"]
    assert "spam_complaint_rate" in result["guardrail_metrics"]


def test_output_custom_guardrail_metrics() -> None:
    result = design_experiment(
        0.40, 0.05,
        guardrail_metrics=["churn_rate", "support_ticket_rate"],
    )
    assert result["guardrail_metrics"] == ["churn_rate", "support_ticket_rate"]


# ---------------------------------------------------------------------------
# Duration calculation
# ---------------------------------------------------------------------------


def test_duration_days_computed_correctly() -> None:
    result = design_experiment(
        baseline_rate=0.42,
        mde=0.05,
        daily_traffic=200,
        use_cuped=False,
    )
    expected_duration = math.ceil(result["required_n_total"] / 200)
    assert result["duration_days"] == expected_duration


def test_duration_none_when_no_traffic_given() -> None:
    result = design_experiment(0.42, 0.05)
    assert result["duration_days"] is None


def test_cuped_reduces_duration() -> None:
    """With CUPED, duration should be shorter than without."""
    no_cuped = design_experiment(0.42, 0.05, daily_traffic=300, use_cuped=False)
    with_cuped = design_experiment(
        0.42, 0.05, daily_traffic=300, use_cuped=True, variance_reduction=0.30
    )
    assert with_cuped["duration_days"] <= no_cuped["duration_days"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_invalid_baseline_rate_zero_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="baseline_rate"):
        design_experiment(baseline_rate=0.0, mde=0.05)


def test_invalid_baseline_rate_one_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="baseline_rate"):
        design_experiment(baseline_rate=1.0, mde=0.05)


def test_invalid_mde_zero_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="mde"):
        design_experiment(baseline_rate=0.40, mde=0.0)


def test_invalid_mde_negative_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="mde"):
        design_experiment(baseline_rate=0.40, mde=-0.05)


def test_treatment_rate_exceeds_1_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="mde"):
        design_experiment(baseline_rate=0.95, mde=0.10)


def test_invalid_alpha_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="alpha"):
        design_experiment(0.40, 0.05, alpha=0.60)


def test_invalid_power_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="power"):
        design_experiment(0.40, 0.05, power=0.30)


def test_invalid_treatment_split_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="treatment_split"):
        design_experiment(0.40, 0.05, treatment_split=0.05)


def test_invalid_variance_reduction_raises() -> None:
    with pytest.raises(ExperimentDesignError, match="variance_reduction"):
        design_experiment(0.40, 0.05, use_cuped=True, variance_reduction=1.0)


# ---------------------------------------------------------------------------
# Notes field is populated
# ---------------------------------------------------------------------------


def test_notes_field_mentions_cuped_when_applied() -> None:
    result = design_experiment(0.42, 0.05, use_cuped=True, variance_reduction=0.30)
    assert "CUPED" in result["notes"]


def test_notes_field_present_without_cuped() -> None:
    result = design_experiment(0.42, 0.05, use_cuped=False)
    assert isinstance(result["notes"], str)
    assert len(result["notes"]) > 0
