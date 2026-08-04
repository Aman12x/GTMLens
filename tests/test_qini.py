"""
Tests for Qini curve evaluation (core/causal.py) and POST /api/segment/qini.

Coverage:
    - qini_curve: perfect-knowledge scores beat random scores; random scores
      hover near zero; the curve's endpoint equals the overall incremental;
      capture shares are monotone in the targeted fraction
    - qini_curve input validation: length mismatch, single-arm data,
      all-NaN scores
    - cross_fitted_cate: returns aligned out-of-fold scores; unsupported
      method and bad fold counts raise
    - API: /api/segment/qini on the demo tenant returns a positive Qini
      coefficient (the demo has designed heterogeneity, so the ranking must
      beat random) with consistent curve arrays
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.causal import (
    CausalEstimationError,
    cross_fitted_cate,
    qini_curve,
)


def _uplift_population(n: int = 6000, seed: int = 11):
    """
    Population with known heterogeneous uplift:
    half the users have +25pp treatment effect, half have zero.
    Returns (y, t, true_effect_flag).
    """
    rng = np.random.default_rng(seed)
    responsive = rng.random(n) < 0.5
    t = (rng.random(n) < 0.5).astype(float)
    p = 0.30 + 0.25 * responsive * t
    y = (rng.random(n) < p).astype(float)
    return y, t, responsive.astype(float)


# ---------------------------------------------------------------------------
# qini_curve
# ---------------------------------------------------------------------------


class TestQiniCurve:
    def test_perfect_scores_beat_random_scores(self):
        y, t, responsive = _uplift_population()
        rng = np.random.default_rng(0)

        perfect = qini_curve(y, t, responsive + rng.normal(0, 1e-6, len(y)))
        random_ = qini_curve(y, t, rng.random(len(y)))

        assert perfect["qini_coefficient"] > random_["qini_coefficient"]
        assert perfect["qini_coefficient"] > 0

    def test_random_scores_near_zero_coefficient(self):
        y, t, _ = _uplift_population()
        rng = np.random.default_rng(1)
        result = qini_curve(y, t, rng.random(len(y)))
        # Coefficient is in "incremental conversions × fraction" units; for a
        # random ranking it should be a small share of the total lift
        assert abs(result["qini_coefficient"]) < 0.15 * result["total_incremental"]

    def test_curve_endpoint_equals_overall_incremental(self):
        y, t, responsive = _uplift_population(n=2000)
        result = qini_curve(y, t, responsive)
        n_t, n_c = t.sum(), (1 - t).sum()
        overall = float((y * t).sum() - (y * (1 - t)).sum() * n_t / n_c)
        assert result["incremental"][-1] == pytest.approx(overall, abs=0.01)
        assert result["fractions"][-1] == 1.0

    def test_capture_shares_monotone(self):
        y, t, responsive = _uplift_population()
        result = qini_curve(y, t, responsive)
        shares = [result["capture_shares"][k] for k in ("0.10", "0.20", "0.30", "0.50")]
        assert shares == sorted(shares)
        # With half the population responsive, targeting the top 50% by a
        # perfect score should capture essentially all incremental conversions
        assert result["capture_shares"]["0.50"] > 0.85

    def test_length_mismatch_raises(self):
        with pytest.raises(CausalEstimationError, match="same length"):
            qini_curve(np.ones(10), np.ones(9), np.ones(10))

    def test_single_arm_raises(self):
        with pytest.raises(CausalEstimationError, match="both treatment and control"):
            qini_curve(np.ones(20), np.ones(20), np.random.default_rng(0).random(20))

    def test_all_nan_scores_raise(self):
        with pytest.raises(CausalEstimationError, match="non-NaN"):
            qini_curve(np.ones(10), np.array([0, 1] * 5), np.full(10, np.nan))


# ---------------------------------------------------------------------------
# cross_fitted_cate
# ---------------------------------------------------------------------------


class TestCrossFittedCate:
    def _frame(self, n: int = 1500, seed: int = 5) -> pd.DataFrame:
        y, t, responsive = _uplift_population(n=n, seed=seed)
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "outcome":   y,
            "treatment": t,
            "resp_flag": responsive,
            "noise":     rng.normal(0, 1, n),
        })

    def test_returns_aligned_finite_scores(self):
        df = self._frame()
        scores = cross_fitted_cate(df, "outcome", "treatment", ["resp_flag", "noise"])
        assert len(scores) == len(df)
        assert np.isfinite(scores).all()

    def test_cross_fitted_ranking_recovers_heterogeneity(self):
        df = self._frame()
        scores = cross_fitted_cate(df, "outcome", "treatment", ["resp_flag", "noise"])
        result = qini_curve(
            df["outcome"].to_numpy(), df["treatment"].to_numpy(), scores
        )
        assert result["qini_coefficient"] > 0
        assert result["capture_shares"]["0.50"] > 0.7

    def test_unsupported_method_raises(self):
        df = self._frame(n=300)
        with pytest.raises(CausalEstimationError, match="t_learner and s_learner"):
            cross_fitted_cate(df, "outcome", "treatment", ["noise"], method="causal_forest")

    def test_bad_fold_count_raises(self):
        df = self._frame(n=300)
        with pytest.raises(CausalEstimationError, match="n_folds"):
            cross_fitted_cate(df, "outcome", "treatment", ["noise"], n_folds=1)


# ---------------------------------------------------------------------------
# POST /api/segment/qini (demo tenant — designed heterogeneity)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestQiniEndpoint:
    def test_demo_qini_beats_random(self, client):
        resp = client.post("/api/segment/qini", json={"n_folds": 3})
        assert resp.status_code == 200
        d = resp.json()
        assert d["cross_fitted"] is True
        assert d["qini_coefficient"] > 0
        assert d["total_incremental"] > 0
        assert len(d["fractions"]) == len(d["incremental"]) == len(d["random_baseline"])
        shares = [d["capture_shares"][k] for k in ("0.10", "0.20", "0.30", "0.50")]
        assert shares == sorted(shares)

    def test_invalid_method_rejected(self, client):
        resp = client.post("/api/segment/qini", json={"method": "causal_forest"})
        assert resp.status_code == 422
