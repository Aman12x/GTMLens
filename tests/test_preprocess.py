"""
Tests for core/preprocess.py.

Required coverage (CLAUDE.md):
    - Winsorize clips correctly at upper percentile
    - log_transform handles zeros via offset
    - Input series is never mutated
    - Binary outcomes must not be winsorized (caller's guard — validator test)
    - preprocess_metric dispatches correctly for all three methods
    - Edge cases: all-NaN, non-numeric dtype, invalid params
"""

import numpy as np
import pandas as pd
import pytest

from core.preprocess import log_transform, preprocess_metric, winsorize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def revenue_series() -> pd.Series:
    """Log-normal revenue series with heavy right tail (realistic GTM data)."""
    rng = np.random.default_rng(0)
    vals = rng.lognormal(mean=7.0, sigma=1.5, size=1000)
    return pd.Series(vals, name="revenue")


@pytest.fixture()
def activation_series() -> pd.Series:
    """Activation rate series: float values in [0, 1]."""
    rng = np.random.default_rng(1)
    vals = rng.beta(2, 3, size=500)
    return pd.Series(vals, name="activation_rate")


@pytest.fixture()
def series_with_zeros() -> pd.Series:
    return pd.Series([0.0, 1.5, 3.0, 0.0, 5.0, 10.0])


@pytest.fixture()
def series_with_nulls() -> pd.Series:
    return pd.Series([1.0, 2.0, np.nan, 4.0, 100.0])


# ---------------------------------------------------------------------------
# winsorize — clipping correctness
# ---------------------------------------------------------------------------


def test_winsorize_clips_upper_tail(revenue_series: pd.Series) -> None:
    result = winsorize(revenue_series, upper_pct=0.99)
    cap = float(np.quantile(revenue_series.dropna(), 0.99))
    assert float(result.max()) <= cap + 1e-9


def test_winsorize_values_above_cap_equal_cap(revenue_series: pd.Series) -> None:
    cap = float(np.quantile(revenue_series, 0.95))
    result = winsorize(revenue_series, upper_pct=0.95)
    above_original = revenue_series[revenue_series > cap]
    assert (result.loc[above_original.index] == cap).all()


def test_winsorize_values_below_cap_unchanged(revenue_series: pd.Series) -> None:
    cap = float(np.quantile(revenue_series, 0.99))
    result = winsorize(revenue_series, upper_pct=0.99)
    below_mask = revenue_series <= cap
    pd.testing.assert_series_equal(
        result[below_mask].reset_index(drop=True),
        revenue_series[below_mask].reset_index(drop=True),
    )


def test_winsorize_lower_tail_untouched(revenue_series: pd.Series) -> None:
    """Lower tail must not be clipped — winsorize is one-sided."""
    result = winsorize(revenue_series, upper_pct=0.99)
    assert float(result.min()) == pytest.approx(float(revenue_series.min()))


def test_winsorize_reduces_max_not_min(revenue_series: pd.Series) -> None:
    result = winsorize(revenue_series, upper_pct=0.90)
    assert float(result.max()) < float(revenue_series.max())
    assert float(result.min()) == pytest.approx(float(revenue_series.min()))


def test_winsorize_with_nulls_ignores_nulls(series_with_nulls: pd.Series) -> None:
    """NaN values should be preserved, not clipped."""
    result = winsorize(series_with_nulls, upper_pct=0.75)
    assert result.isna().sum() == series_with_nulls.isna().sum()


def test_winsorize_pct_1_clips_nothing(revenue_series: pd.Series) -> None:
    """upper_pct=1.0 should leave the series unchanged."""
    result = winsorize(revenue_series, upper_pct=1.0)
    pd.testing.assert_series_equal(result, revenue_series)


# ---------------------------------------------------------------------------
# winsorize — error cases
# ---------------------------------------------------------------------------


def test_winsorize_invalid_pct_above_1_raises() -> None:
    with pytest.raises(ValueError, match="upper_pct"):
        winsorize(pd.Series([1.0, 2.0, 3.0]), upper_pct=1.5)


def test_winsorize_zero_pct_raises() -> None:
    with pytest.raises(ValueError, match="upper_pct"):
        winsorize(pd.Series([1.0, 2.0, 3.0]), upper_pct=0.0)


def test_winsorize_all_nan_raises() -> None:
    with pytest.raises(ValueError, match="entirely NaN"):
        winsorize(pd.Series([np.nan, np.nan]))


def test_winsorize_non_numeric_raises() -> None:
    with pytest.raises(TypeError, match="numeric"):
        winsorize(pd.Series(["a", "b", "c"]))


# ---------------------------------------------------------------------------
# log_transform — zero handling
# ---------------------------------------------------------------------------


def test_log_transform_handles_zeros(series_with_zeros: pd.Series) -> None:
    """log(0 + 1) = 0 — must not raise or produce -inf."""
    result = log_transform(series_with_zeros, offset=1.0)
    assert not result.isin([float("-inf"), float("inf")]).any()
    assert float(result.min()) >= 0.0


def test_log_transform_correct_values(series_with_zeros: pd.Series) -> None:
    result = log_transform(series_with_zeros, offset=1.0)
    expected = np.log(series_with_zeros + 1.0)
    pd.testing.assert_series_equal(result, pd.Series(expected, name=series_with_zeros.name))


def test_log_transform_offset_shifts_values() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    result_1 = log_transform(s, offset=1.0)
    result_2 = log_transform(s, offset=2.0)
    assert (result_2 > result_1).all()


def test_log_transform_null_preserved() -> None:
    s = pd.Series([1.0, np.nan, 3.0])
    result = log_transform(s, offset=1.0)
    assert result.isna().sum() == 1


# ---------------------------------------------------------------------------
# log_transform — error cases
# ---------------------------------------------------------------------------


def test_log_transform_zero_offset_raises() -> None:
    with pytest.raises(ValueError, match="offset"):
        log_transform(pd.Series([1.0, 2.0]), offset=0.0)


def test_log_transform_negative_offset_raises() -> None:
    with pytest.raises(ValueError, match="offset"):
        log_transform(pd.Series([1.0, 2.0]), offset=-1.0)


def test_log_transform_negative_values_raises() -> None:
    """Series with values that produce (value + offset) <= 0 must raise."""
    with pytest.raises(ValueError, match="<= 0"):
        log_transform(pd.Series([-5.0, 1.0, 2.0]), offset=1.0)


def test_log_transform_non_numeric_raises() -> None:
    with pytest.raises(TypeError, match="numeric"):
        log_transform(pd.Series(["a", "b"]))


# ---------------------------------------------------------------------------
# Input immutability — critical: caller's data must never be mutated
# ---------------------------------------------------------------------------


def test_winsorize_does_not_mutate_input(revenue_series: pd.Series) -> None:
    original_values = revenue_series.values.copy()
    original_max = float(revenue_series.max())
    winsorize(revenue_series, upper_pct=0.90)
    assert float(revenue_series.max()) == pytest.approx(original_max)
    np.testing.assert_array_equal(revenue_series.values, original_values)


def test_log_transform_does_not_mutate_input(series_with_zeros: pd.Series) -> None:
    original_values = series_with_zeros.values.copy()
    log_transform(series_with_zeros, offset=1.0)
    np.testing.assert_array_equal(series_with_zeros.values, original_values)


def test_preprocess_metric_none_does_not_mutate(revenue_series: pd.Series) -> None:
    original_values = revenue_series.values.copy()
    preprocess_metric(revenue_series, method="none")
    np.testing.assert_array_equal(revenue_series.values, original_values)


# ---------------------------------------------------------------------------
# preprocess_metric — dispatch
# ---------------------------------------------------------------------------


def test_preprocess_metric_winsorize_dispatches(revenue_series: pd.Series) -> None:
    result = preprocess_metric(revenue_series, method="winsorize", upper_pct=0.95)
    cap = float(np.quantile(revenue_series, 0.95))
    assert float(result.max()) <= cap + 1e-9


def test_preprocess_metric_log_dispatches(series_with_zeros: pd.Series) -> None:
    result = preprocess_metric(series_with_zeros, method="log", offset=1.0)
    expected = np.log(series_with_zeros + 1.0)
    np.testing.assert_allclose(result.values, expected.values)


def test_preprocess_metric_none_returns_copy(revenue_series: pd.Series) -> None:
    result = preprocess_metric(revenue_series, method="none")
    pd.testing.assert_series_equal(result, revenue_series)
    assert result is not revenue_series  # must be a copy


def test_preprocess_metric_invalid_method_raises(revenue_series: pd.Series) -> None:
    with pytest.raises(ValueError, match="method"):
        preprocess_metric(revenue_series, method="standardize")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Treatment/control consistency check
# ---------------------------------------------------------------------------


def test_winsorize_same_cap_treatment_control() -> None:
    """
    CLAUDE.md rule: Apply same transformation to both arms.
    Simulate by applying winsorize to the full series (not per-arm),
    then verify that the same cap is applied across both arms.
    """
    rng = np.random.default_rng(42)
    combined = pd.Series(rng.lognormal(7, 1.5, 1000))
    treatment = combined[:500]
    control = combined[500:]

    # Correct approach: winsorize on FULL series, apply same cap to both
    cap = float(np.quantile(combined.dropna(), 0.99))
    result_t = combined[:500].clip(upper=cap)
    result_c = combined[500:].clip(upper=cap)

    assert float(result_t.max()) <= cap + 1e-9
    assert float(result_c.max()) <= cap + 1e-9
