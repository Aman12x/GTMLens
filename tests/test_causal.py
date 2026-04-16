"""
Tests for core/causal.py.

Required coverage (CLAUDE.md):
    - CUPED recovers known ATE on synthetic data
    - SRM detects 60/40 split at p < 0.01
    - BH rejects fewer hypotheses than Bonferroni on the same p-values
    - BH controls FDR (all true nulls accepted under global null)
    - detect_srm: balanced split passes, extreme imbalance fails
    - cuped_adjustment: raises on missing cols, zero-variance covariate
    - diff_in_diff: recovers known treatment effect on panel data
"""

import numpy as np
import pandas as pd
import pytest

from core.causal import (
    CausalEstimationError,
    bh_correction,
    cuped_adjustment,
    detect_srm,
    diff_in_diff,
    estimate_cate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cuped_df(
    n: int = 2000,
    true_ate: float = 0.10,
    treatment_split: float = 0.50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic DataFrame for CUPED tests with known ATE.

    Pre-experiment covariate X_pre is correlated with the outcome Y_post
    so CUPED provides meaningful variance reduction.
    """
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < treatment_split).astype(int)
    x_pre = rng.normal(0.45, 0.10, n).clip(0.01, 0.99)
    noise = rng.normal(0, 0.05, n)
    y_post = x_pre + true_ate * treatment + noise
    y_post = y_post.clip(0.01, 0.99)

    return pd.DataFrame({
        "user_id":    [f"u{i}" for i in range(n)],
        "treatment":  treatment,
        "pre_rate":   x_pre,
        "post_rate":  y_post,
    })


# ---------------------------------------------------------------------------
# CUPED — recovers known ATE
# ---------------------------------------------------------------------------


def test_cuped_recovers_known_ate() -> None:
    """CUPED-adjusted ATE must be within ±2pp of the true ATE on large synthetic data."""
    true_ate = 0.10
    df = _make_cuped_df(n=5000, true_ate=true_ate, seed=0)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    assert abs(result["ate"] - true_ate) < 0.02, (
        f"CUPED ATE {result['ate']:.4f} is more than 2pp from true ATE {true_ate}"
    )


def test_cuped_p_value_significant_for_real_effect() -> None:
    """When a real effect is present at N=5000, CUPED should detect it (p < 0.05)."""
    df = _make_cuped_df(n=5000, true_ate=0.08, seed=1)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    assert result["p_value"] < 0.05


def test_cuped_variance_reduction_positive() -> None:
    """Variance reduction must be positive when the covariate is correlated."""
    df = _make_cuped_df(n=2000, true_ate=0.05, seed=2)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    assert result["variance_reduction_pct"] > 0.0


def test_cuped_returns_all_expected_keys() -> None:
    df = _make_cuped_df(n=1000, true_ate=0.05)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    expected_keys = {
        "ate", "ate_se", "p_value", "ci_lower", "ci_upper",
        "variance_reduction_pct", "theta", "covariate_pearson_r",
        "n_treatment", "n_control",
    }
    assert expected_keys.issubset(result.keys())


def test_cuped_ci_contains_true_ate() -> None:
    """95% CI should contain the true ATE at large N (non-deterministic, but very likely)."""
    true_ate = 0.10
    df = _make_cuped_df(n=10000, true_ate=true_ate, seed=99)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    assert result["ci_lower"] <= true_ate <= result["ci_upper"]


def test_cuped_null_effect_not_significant() -> None:
    """When true ATE=0, CUPED should not reject at alpha=0.05 most of the time."""
    df = _make_cuped_df(n=2000, true_ate=0.0, seed=7)
    result = cuped_adjustment(df, "post_rate", "pre_rate", "treatment")
    # With ATE=0 and reasonable N, p should not be extremely small
    assert result["p_value"] > 0.001


# ---------------------------------------------------------------------------
# CUPED — error cases
# ---------------------------------------------------------------------------


def test_cuped_missing_metric_col_raises() -> None:
    df = _make_cuped_df()
    with pytest.raises(CausalEstimationError, match="not found"):
        cuped_adjustment(df, "nonexistent", "pre_rate", "treatment")


def test_cuped_missing_covariate_raises() -> None:
    df = _make_cuped_df()
    with pytest.raises(CausalEstimationError, match="not found"):
        cuped_adjustment(df, "post_rate", "nonexistent", "treatment")


def test_cuped_zero_variance_covariate_raises() -> None:
    df = _make_cuped_df()
    df["constant_pre"] = 0.5  # zero variance
    with pytest.raises(CausalEstimationError, match="zero variance"):
        cuped_adjustment(df, "post_rate", "constant_pre", "treatment")


def test_cuped_too_few_rows_raises() -> None:
    df = _make_cuped_df(n=2)
    with pytest.raises(CausalEstimationError, match="Too few"):
        cuped_adjustment(df, "post_rate", "pre_rate", "treatment")


# ---------------------------------------------------------------------------
# SRM Detection
# ---------------------------------------------------------------------------


def test_srm_detects_60_40_split() -> None:
    """A 60/40 split intended to be 50/50 should be detected at p < 0.01."""
    result = detect_srm(n_treatment=6000, n_control=4000, expected_split=0.50, alpha=0.01)
    assert result["srm_detected"] is True
    assert result["p_value"] < 0.01


def test_srm_passes_balanced_split() -> None:
    """Near-50/50 split should not trigger SRM."""
    result = detect_srm(n_treatment=5010, n_control=4990, expected_split=0.50, alpha=0.01)
    assert result["srm_detected"] is False
    assert result["p_value"] > 0.01


def test_srm_extreme_imbalance_detected() -> None:
    result = detect_srm(n_treatment=9000, n_control=1000, expected_split=0.50, alpha=0.01)
    assert result["srm_detected"] is True


def test_srm_custom_split_respected() -> None:
    """70/30 intended split with near-exact counts should not trigger SRM."""
    result = detect_srm(n_treatment=700, n_control=300, expected_split=0.70, alpha=0.01)
    assert result["srm_detected"] is False


def test_srm_returns_all_expected_keys() -> None:
    result = detect_srm(1000, 1000)
    expected = {
        "srm_detected", "p_value", "observed_split",
        "expected_split", "chi2_stat", "recommendation",
    }
    assert expected.issubset(result.keys())


def test_srm_invalid_n_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        detect_srm(-100, 1000)


def test_srm_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="expected_split"):
        detect_srm(500, 500, expected_split=0.0)


def test_srm_zero_total_raises() -> None:
    with pytest.raises(ValueError, match="0"):
        detect_srm(0, 0)


# ---------------------------------------------------------------------------
# BH correction — rejects fewer than Bonferroni
# ---------------------------------------------------------------------------


def test_bh_rejects_fewer_than_bonferroni() -> None:
    """
    CLAUDE.md: BH rejects fewer null hypotheses than Bonferroni at the same alpha.
    On a realistic mix of null and non-null p-values, BH should reject more than
    Bonferroni (not fewer) — but the key CLAUDE.md rule is to never use Bonferroni.
    Here we verify BH rejects AT LEAST as many as Bonferroni (Bonferroni is stricter).
    """
    rng = np.random.default_rng(0)
    # 5 small p-values (real effects) + 10 large ones (null)
    real_effects = [0.001, 0.003, 0.008, 0.012, 0.020]
    nulls = rng.uniform(0.3, 0.9, 10).tolist()
    p_values = real_effects + nulls
    alpha = 0.05

    bh_rejects = sum(bh_correction(p_values, alpha=alpha))

    # Bonferroni threshold
    bonferroni_threshold = alpha / len(p_values)
    bonferroni_rejects = sum(1 for p in p_values if p <= bonferroni_threshold)

    assert bh_rejects >= bonferroni_rejects, (
        f"BH rejected {bh_rejects} but Bonferroni rejected {bonferroni_rejects}. "
        f"BH should be at least as liberal."
    )


def test_bh_all_null_rejects_none() -> None:
    """Under the global null (all p-values uniform), BH should reject close to 0."""
    rng = np.random.default_rng(42)
    p_values = rng.uniform(0.1, 1.0, 20).tolist()
    rejects = bh_correction(p_values, alpha=0.05)
    # With all-large p-values there should be no rejections
    assert sum(rejects) == 0


def test_bh_all_significant_rejects_all() -> None:
    """When all p-values are well below alpha/m, all should be rejected."""
    p_values = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
    rejects = bh_correction(p_values, alpha=0.05)
    assert all(rejects)


def test_bh_preserves_original_order() -> None:
    """Output list must be indexed to match input order, not sorted order."""
    p_values = [0.90, 0.001, 0.80, 0.002]
    rejects = bh_correction(p_values, alpha=0.05)
    assert rejects[0] is False   # 0.90 — not rejected
    assert rejects[1] is True    # 0.001 — rejected
    assert rejects[2] is False   # 0.80 — not rejected
    assert rejects[3] is True    # 0.002 — rejected


def test_bh_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        bh_correction([])


def test_bh_invalid_alpha_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        bh_correction([0.01, 0.05], alpha=1.5)


def test_bh_returns_list_of_bool() -> None:
    result = bh_correction([0.01, 0.5, 0.9])
    assert isinstance(result, list)
    assert all(isinstance(v, bool) for v in result)


# ---------------------------------------------------------------------------
# DiD — recovers known treatment effect
# ---------------------------------------------------------------------------


def _make_did_df(
    n_users: int = 500,
    true_effect: float = 0.08,
    seed: int = 42,
) -> pd.DataFrame:
    """Panel dataset with known DiD treatment effect for unit testing."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i in range(n_users):
        is_treated = int(i < n_users // 2)
        # Pre-period: both groups at same baseline
        for d in range(10):
            dt = base_date + pd.Timedelta(days=d)
            outcome = rng.binomial(1, 0.40 + is_treated * 0.0)  # same baseline
            rows.append({"user_id": f"u{i}", "event_date": dt,
                         "treated": is_treated, "outcome": outcome})
        # Post-period: treated group gets lift
        for d in range(10, 20):
            dt = base_date + pd.Timedelta(days=d)
            p = 0.40 + is_treated * true_effect
            outcome = rng.binomial(1, p)
            rows.append({"user_id": f"u{i}", "event_date": dt,
                         "treated": is_treated, "outcome": outcome})
    return pd.DataFrame(rows)


def test_did_recovers_known_effect() -> None:
    """DiD β3 should be within ±3pp of true_effect at N=500."""
    true_effect = 0.08
    df = _make_did_df(n_users=1000, true_effect=true_effect, seed=0)
    result = diff_in_diff(
        df,
        pre_window=("2024-01-01", "2024-01-09"),
        post_window=("2024-01-10", "2024-01-19"),
        treatment_group_col="treated",
        outcome_col="outcome",
        date_col="event_date",
    )
    assert abs(result["did_estimate"] - true_effect) < 0.03, (
        f"DiD estimate {result['did_estimate']:.4f} far from true effect {true_effect}"
    )


def test_did_returns_all_expected_keys() -> None:
    df = _make_did_df()
    result = diff_in_diff(
        df,
        pre_window=("2024-01-01", "2024-01-09"),
        post_window=("2024-01-10", "2024-01-19"),
        treatment_group_col="treated",
        outcome_col="outcome",
    )
    expected = {"did_estimate", "std_err", "p_value", "ci_lower", "ci_upper", "n_pre", "n_post"}
    assert expected.issubset(result.keys())


def test_did_missing_col_raises() -> None:
    df = _make_did_df()
    with pytest.raises(CausalEstimationError, match="not found"):
        diff_in_diff(
            df,
            pre_window=("2024-01-01", "2024-01-09"),
            post_window=("2024-01-10", "2024-01-19"),
            treatment_group_col="nonexistent",
            outcome_col="outcome",
        )


def test_did_empty_pre_window_raises() -> None:
    df = _make_did_df()
    with pytest.raises(CausalEstimationError, match="No rows in pre-period"):
        diff_in_diff(
            df,
            pre_window=("2023-01-01", "2023-01-09"),   # before data starts
            post_window=("2024-01-10", "2024-01-19"),
            treatment_group_col="treated",
            outcome_col="outcome",
        )


# ---------------------------------------------------------------------------
# estimate_cate — basic smoke tests
# ---------------------------------------------------------------------------


def _make_cate_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    treatment = (rng.random(n) < 0.5).astype(int)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # HTE: effect is larger for high x1
    p = 0.40 + 0.05 * treatment + 0.03 * treatment * (x1 > 0)
    outcome = rng.binomial(1, p.clip(0.01, 0.99))
    return pd.DataFrame({
        "treatment": treatment,
        "outcome":   outcome,
        "x1":        x1,
        "x2":        x2,
    })


def test_estimate_cate_returns_cate_column() -> None:
    df = _make_cate_df(n=200)
    result = estimate_cate(df, "outcome", "treatment", ["x1", "x2"], method="t_learner")
    assert "cate_estimate" in result.columns


def test_estimate_cate_output_length_matches_input() -> None:
    df = _make_cate_df(n=200)
    result = estimate_cate(df, "outcome", "treatment", ["x1", "x2"])
    assert len(result) == len(df)


def test_estimate_cate_causal_forest_small_n_raises() -> None:
    df = _make_cate_df(n=200)
    with pytest.raises(CausalEstimationError, match="5000"):
        estimate_cate(df, "outcome", "treatment", ["x1", "x2"], method="causal_forest")


def test_estimate_cate_missing_col_raises() -> None:
    df = _make_cate_df(n=200)
    with pytest.raises(CausalEstimationError, match="not found"):
        estimate_cate(df, "outcome", "treatment", ["x1", "nonexistent"])


def test_estimate_cate_s_learner_runs() -> None:
    df = _make_cate_df(n=200)
    result = estimate_cate(df, "outcome", "treatment", ["x1", "x2"], method="s_learner")
    assert "cate_estimate" in result.columns
    assert result["cate_estimate"].notna().sum() > 0
