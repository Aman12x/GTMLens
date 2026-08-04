"""
Tests for the Clay integration surface:
    ingestion/clay_normalizer.py — raw Clay values → internal vocabulary
    core/model_store.py          — fit/persist/load/score round-trip
    core/outreach.segment_message_angle — angle composition
    api/routes/score.py          — cold-start RCT, scored mode, holdout
                                   determinism, SRM audit, API-key auth

Coverage:
    - Employee counts, range strings, and labels bucket correctly
    - Channel labels (Google Ads, LinkedIn, SEO…) map to internal channels
    - fit_and_save rejects under-sized arms; round-trips through load_latest
    - Scoring recovers designed heterogeneity (enterprise > SMB CATE)
    - Unseen category levels fall back to the reference level, still score
    - Cold-start assignment is deterministic and splits ~50/50
    - /score/train on the demo tenant flips /score from cold_start to scored
    - X-API-Key resolves to the minting user's tenant; revoked keys degrade
    - /score/srm reports splits and zero violations when nothing was sent
"""

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ingestion.clay_normalizer import (
    normalize_channel,
    normalize_company_size,
    normalize_industry,
)

# ---------------------------------------------------------------------------
# Fixtures — isolated temp stores so tests don't touch real data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def isolated_env(tmp_path_factory):
    """Point auth DB, SQLite log, and model store at temp paths for the module."""
    base = tmp_path_factory.mktemp("score")
    os.environ["AUTH_DB_PATH"] = str(base / "auth.db")
    os.environ["SQLITE_PATH"] = str(base / "logs.db")
    os.environ["MODELS_DIR"] = str(base / "models")
    os.environ["JWT_SECRET_KEY"] = "test-secret-do-not-use-in-production"

    from core.auth import init_auth_db
    init_auth_db()

    yield

    for key in ("AUTH_DB_PATH", "SQLITE_PATH", "MODELS_DIR", "JWT_SECRET_KEY"):
        os.environ.pop(key, None)


@pytest.fixture(scope="module")
def client(isolated_env):
    """TestClient with lifespan fired (seeds SQLite tables + demo DuckDB)."""
    from api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    """Register a fresh user and return Bearer headers (isolated tenant)."""
    client.post(
        "/api/alpha/auth/register",
        json={"email": "clay-test@example.com", "password": "testpass123"},
    )
    resp = client.post(
        "/api/alpha/auth/login",
        data={"username": "clay-test@example.com", "password": "testpass123"},
    )
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _training_frame(n: int = 2000, seed: int = 7) -> pd.DataFrame:
    """
    Synthetic funnel frame with designed heterogeneity:
    enterprise responds strongly to treatment (+30pp), SMB barely (+2pp).
    """
    rng = np.random.default_rng(seed)
    company_size = rng.choice(["SMB", "enterprise"], size=n)
    channel = rng.choice(["organic", "paid_search"], size=n)
    industry = rng.choice(["SaaS", "FinTech"], size=n)
    treatment = rng.integers(0, 2, size=n)
    base = np.where(company_size == "enterprise", 0.40, 0.30)
    effect = np.where(company_size == "enterprise", 0.30, 0.02)
    prob = base + effect * treatment
    activated = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "activated": activated,
        "treatment": treatment,
        "company_size": company_size,
        "channel": channel,
        "industry": industry,
    })


# ---------------------------------------------------------------------------
# clay_normalizer
# ---------------------------------------------------------------------------


class TestNormalizeCompanySize:
    def test_employee_counts_bucket_at_boundaries(self):
        assert normalize_company_size(50) == "SMB"
        assert normalize_company_size(51) == "mid_market"
        assert normalize_company_size(500) == "mid_market"
        assert normalize_company_size(501) == "enterprise"

    def test_range_string_buckets_by_lower_bound(self):
        assert normalize_company_size("51-200") == "mid_market"
        assert normalize_company_size("1,001-5,000") == "enterprise"

    def test_labels_map_case_insensitively(self):
        assert normalize_company_size("Enterprise") == "enterprise"
        assert normalize_company_size("small business") == "SMB"
        assert normalize_company_size("Mid Market") == "mid_market"

    def test_explicit_employee_count_wins_over_label(self):
        assert normalize_company_size("SMB", employee_count=10_000) == "enterprise"

    def test_missing_values_return_unknown(self):
        assert normalize_company_size(None) == "unknown"
        assert normalize_company_size("") == "unknown"
        assert normalize_company_size(0) == "unknown"


class TestNormalizeChannel:
    def test_paid_search_labels(self):
        assert normalize_channel("Google Ads") == "paid_search"
        assert normalize_channel("PPC") == "paid_search"

    def test_social_labels(self):
        assert normalize_channel("LinkedIn") == "social"
        assert normalize_channel("paid social") == "social"

    def test_organic_and_default(self):
        assert normalize_channel("SEO") == "organic"
        assert normalize_channel(None) == "organic"
        assert normalize_channel("  ") == "organic"

    def test_compound_label_substring_match(self):
        assert normalize_channel("google_ads_brand_campaign") == "paid_search"

    def test_unrecognised_passes_through_as_slug(self):
        assert normalize_channel("Carrier Pigeon") == "carrier_pigeon"


def test_normalize_industry_defaults_to_unknown():
    assert normalize_industry(None) == "unknown"
    assert normalize_industry("  ") == "unknown"
    assert normalize_industry(" FinTech ") == "FinTech"


# ---------------------------------------------------------------------------
# model_store
# ---------------------------------------------------------------------------


class TestModelStore:
    def test_rejects_undersized_arms(self):
        from core.model_store import ModelStoreError, fit_and_save
        tiny = _training_frame(n=30)
        with pytest.raises(ModelStoreError, match="at least 25 users per arm"):
            fit_and_save(tiny, "tiny-tenant@example.com")

    def test_fit_load_score_round_trip(self):
        from core.model_store import fit_and_save, has_model, load_latest, score_rows

        tenant = "roundtrip@example.com"
        meta = fit_and_save(_training_frame(), tenant)

        assert meta["model_version"].startswith("v")
        assert meta["n_train"] == 2000
        assert meta["high_cutoff"] >= meta["deprioritize_cutoff"]
        assert has_model(tenant)

        payload = load_latest(tenant)
        assert payload["model_version"] == meta["model_version"]

        rows = [
            {"email": "a@x.com", "company_size": "enterprise", "channel": "paid_search", "industry": "SaaS"},
            {"email": "b@x.com", "company_size": "SMB", "channel": "organic", "industry": "SaaS"},
        ]
        results = score_rows(payload, rows)
        assert len(results) == 2
        # Designed heterogeneity: enterprise CATE must exceed SMB CATE
        assert results[0]["cate_estimate"] > results[1]["cate_estimate"]
        assert all(r["uplift_tier"] in {"high", "mid", "deprioritize"} for r in results)

    def test_unseen_level_falls_back_to_reference(self):
        from core.model_store import fit_and_save, load_latest, score_rows

        tenant = "unseen@example.com"
        fit_and_save(_training_frame(), tenant)
        payload = load_latest(tenant)

        unseen = score_rows(payload, [{"email": "c@x.com", "company_size": "mid_market", "channel": "fax", "industry": "Zeppelins"}])
        assert len(unseen) == 1
        assert isinstance(unseen[0]["cate_estimate"], float)

    def test_load_latest_returns_none_for_fresh_tenant(self):
        from core.model_store import has_model, load_latest
        assert load_latest("never-trained@example.com") is None
        assert not has_model("never-trained@example.com")


# ---------------------------------------------------------------------------
# segment_message_angle
# ---------------------------------------------------------------------------


def test_message_angle_composition():
    from core.outreach import segment_message_angle

    high = segment_message_angle("enterprise", "paid_search", "high")
    assert high.startswith("High predicted uplift")
    assert "ROI at scale" in high

    cold = segment_message_angle("SMB", "email", "cold_start")
    assert "Randomised pilot wave" in cold

    fallback = segment_message_angle("unheard_of", "smoke_signal", "mid")
    assert fallback.startswith("Moderate predicted uplift")


# ---------------------------------------------------------------------------
# /api/score routes
# ---------------------------------------------------------------------------


class TestScoreColdStart:
    def test_cold_start_shape_and_determinism(self, client, auth_headers):
        body = {"email": "prospect@corp.com", "employee_count": 800, "channel": "Google Ads"}
        first = client.post("/api/score", json=body, headers=auth_headers)
        assert first.status_code == 200
        r = first.json()
        assert r["mode"] == "cold_start"
        assert r["company_size"] == "enterprise"
        assert r["channel"] == "paid_search"
        assert r["segment_id"] == "enterprise_paid_search"
        assert r["cate_estimate"] is None
        assert r["assignment"] in {"treat", "control"}
        assert r["holdout_flag"] == (r["assignment"] == "control")
        assert "Randomised pilot wave" in r["message_angle"]

        second = client.post("/api/score", json=body, headers=auth_headers).json()
        assert second["assignment"] == r["assignment"]

    def test_cold_start_split_is_roughly_half(self, client, auth_headers):
        contacts = [
            {"email": f"user{i}@corp.com", "employee_count": 40, "channel": "SEO"}
            for i in range(400)
        ]
        resp = client.post("/api/score/batch", json={"contacts": contacts}, headers=auth_headers)
        assert resp.status_code == 200
        results = resp.json()["results"]
        treat_frac = sum(r["assignment"] == "treat" for r in results) / len(results)
        assert 0.40 < treat_frac < 0.60

    def test_status_reports_cold_start(self, client, auth_headers):
        resp = client.get("/api/score/status", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json() == {
            "has_model": False, "mode": "cold_start",
            "model_version": None, "trained_at": None, "n_train": None,
        }


class TestScoreTrainedFlow:
    def test_train_then_score_demo_tenant(self, client):
        train = client.post("/api/score/train")
        assert train.status_code == 200
        meta = train.json()
        assert meta["n_train"] > 1000
        assert meta["n_treatment"] > 0 and meta["n_control"] > 0

        resp = client.post(
            "/api/score",
            json={"email": "vp@bigco.com", "company_size": "Enterprise", "channel": "Google Ads", "industry": "SaaS"},
        )
        assert resp.status_code == 200
        r = resp.json()
        assert r["mode"] == "scored"
        assert r["model_version"] == meta["model_version"]
        assert r["uplift_tier"] in {"high", "mid", "deprioritize"}
        assert isinstance(r["cate_estimate"], float)

        status = client.get("/api/score/status").json()
        assert status["has_model"] is True
        assert status["model_version"] == meta["model_version"]

    def test_srm_audit_reports_splits_and_no_violations(self, client):
        resp = client.get("/api/score/srm")
        assert resp.status_code == 200
        audit = resp.json()
        assert audit["holdout_violations"] == 0
        assert audit["n_scored"] > 0
        modes = {s["mode"] for s in audit["splits"]}
        assert modes <= {"scored", "cold_start"}
        for s in audit["splits"]:
            assert 0.0 <= s["p_value"] <= 1.0


class TestApiKeyAuth:
    def test_key_scopes_to_minting_tenant_and_revocation_degrades(self, client, auth_headers):
        mint = client.post("/api/alpha/auth/api-key", json={"label": "clay-test"}, headers=auth_headers)
        assert mint.status_code == 201
        key = mint.json()["api_key"]
        prefix = mint.json()["key_prefix"]
        assert key.startswith("gtml_")

        # Key resolves to the fresh tenant (no model) even though demo has one
        with_key = client.get("/api/score/status", headers={"X-API-Key": key}).json()
        assert with_key["has_model"] is False

        listed = client.get("/api/alpha/auth/api-keys", headers=auth_headers).json()["keys"]
        assert any(k["key_prefix"] == prefix and k["is_active"] for k in listed)

        revoke = client.delete(f"/api/alpha/auth/api-key/{prefix}", headers=auth_headers)
        assert revoke.status_code == 204

        # Revoked key degrades to the demo tenant (which was trained above)
        after = client.get("/api/score/status", headers={"X-API-Key": key}).json()
        assert after == client.get("/api/score/status").json()

    def test_bogus_key_serves_demo_tenant(self, client):
        resp = client.get("/api/score/status", headers={"X-API-Key": "gtml_not_a_real_key"})
        assert resp.status_code == 200
