"""
Per-tenant persistence for the T-Learner scoring model.

Why this exists: the dashboard (api/routes/segment.py) refits the T-Learner
on every request over the tenant's full dataset — acceptable for an
interactive tab, wrong for per-row scoring where Clay's HTTP API column
fires one request per table row and expects a fast, stable answer.

This module fits the same T-Learner response surfaces (via
core.causal.fit_t_learner_models) once, persists them with joblib, and
serves single-row predictions from the saved artefact.

Feature policy — send-time features only:
    The scoring model is trained on company_size, channel, and industry.
    It deliberately EXCLUDES pre_activation_rate (used by the dashboard's
    CATE tab): a cold prospect arriving from Clay has no product-usage
    history, and imputing it would score on a feature unavailable at send
    time (leakage). Segment-level CATE is driven by the categoricals anyway.

Artefact layout:
    {MODELS_DIR}/{sha256(tenant)[:24]}/model_v{UTC timestamp}.joblib
    The newest file by name is the active model (timestamps sort lexically).

joblib is not a new dependency — it ships with scikit-learn, which is
already required transitively by econml.
"""

import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.causal import CausalEstimationError, fit_t_learner_models

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

_CATEGORICAL_FEATURES = ["company_size", "channel", "industry"]
_OUTCOME_COL = "activated"
_TREATMENT_COL = "treatment"


class ModelStoreError(ValueError):
    """Raised when a scoring model cannot be trained, saved, or loaded."""


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------


def _models_dir() -> Path:
    return Path(os.getenv("MODELS_DIR", str(_PROJECT_ROOT / "data" / "models")))


def _tenant_dir(tenant_id: str) -> Path:
    """Per-tenant model directory keyed by email hash (matches api/db.py)."""
    safe = hashlib.sha256(tenant_id.encode()).hexdigest()[:24]
    return _models_dir() / safe


def _top_fraction() -> float:
    return float(os.getenv("CATE_UPLIFT_THRESHOLD", "0.40"))


def _deprioritize_fraction() -> float:
    return float(os.getenv("SCORE_DEPRIORITIZE_FRACTION", "0.30"))


# ---------------------------------------------------------------------------
# Training + persistence
# ---------------------------------------------------------------------------


def fit_and_save(df: pd.DataFrame, tenant_id: str) -> dict:
    """
    Fit a T-Learner scoring model from funnel data and persist it.

    Args:
        df:        One row per signed-up user with columns: activated,
                   treatment, company_size, channel, industry.
        tenant_id: Tenant identifier ("demo" or user email).

    Returns:
        Model metadata dict: model_version, trained_at, n_train, n_treatment,
        n_control, feature_columns, high_cutoff, deprioritize_cutoff.

    Raises:
        ModelStoreError: If required columns are missing or either arm is
                         too small to fit response surfaces.
    """
    required = [_OUTCOME_COL, _TREATMENT_COL] + _CATEGORICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ModelStoreError(f"Training data is missing columns: {missing}.")

    sub = df[required].dropna().copy()
    n_t = int((sub[_TREATMENT_COL] == 1).sum())
    n_c = int((sub[_TREATMENT_COL] == 0).sum())
    if n_t < 25 or n_c < 25:
        raise ModelStoreError(
            f"Need at least 25 users per arm to train a scoring model. "
            f"Got treatment={n_t}, control={n_c}."
        )

    # Fixed category levels are stored with the model so scoring-time encoding
    # is identical regardless of which levels appear in a given Clay batch.
    categorical_levels = {
        col: sorted(sub[col].astype(str).unique()) for col in _CATEGORICAL_FEATURES
    }
    X, feature_columns = _encode_frame(sub, categorical_levels)
    Y = sub[_OUTCOME_COL].to_numpy(dtype=float)
    T = sub[_TREATMENT_COL].to_numpy(dtype=float)

    try:
        model_t, model_c = fit_t_learner_models(Y, T, X)
    except (ValueError, CausalEstimationError) as exc:
        raise ModelStoreError(f"T-Learner fit failed: {exc}") from exc

    # Tier cutoffs come from the training CATE distribution so that
    # "high uplift" means the same thing at scoring time as it does on the
    # dashboard: top CATE_UPLIFT_THRESHOLD fraction of users.
    train_cate = model_t.predict(X) - model_c.predict(X)
    top_frac = _top_fraction()
    dep_frac = _deprioritize_fraction()
    high_cutoff = float(np.percentile(train_cate, (1.0 - top_frac) * 100))
    deprioritize_cutoff = float(np.percentile(train_cate, dep_frac * 100))

    trained_at = datetime.now(tz=timezone.utc)
    model_version = trained_at.strftime("v%Y%m%dT%H%M%SZ")

    payload = {
        "model_t":              model_t,
        "model_c":              model_c,
        "feature_columns":      feature_columns,
        "categorical_levels":   categorical_levels,
        "model_version":        model_version,
        "trained_at":           trained_at.isoformat(),
        "n_train":              len(sub),
        "n_treatment":          n_t,
        "n_control":            n_c,
        "high_cutoff":          high_cutoff,
        "deprioritize_cutoff":  deprioritize_cutoff,
        "top_fraction":         top_frac,
        "deprioritize_fraction": dep_frac,
    }

    tenant_dir = _tenant_dir(tenant_id)
    tenant_dir.mkdir(parents=True, exist_ok=True)
    path = tenant_dir / f"model_{model_version}.joblib"
    joblib.dump(payload, path)

    logger.info(
        "model_store | trained %s for tenant_hash=%s | n=%d (t=%d c=%d) | "
        "high_cutoff=%.4f deprioritize_cutoff=%.4f",
        model_version, tenant_dir.name, len(sub), n_t, n_c,
        high_cutoff, deprioritize_cutoff,
    )

    return {k: v for k, v in payload.items() if k not in ("model_t", "model_c")}


def load_latest(tenant_id: str) -> dict | None:
    """
    Load the most recently trained scoring model for a tenant.

    Args:
        tenant_id: Tenant identifier ("demo" or user email).

    Returns:
        Full model payload dict (including fitted models), or None when the
        tenant has never trained a model (callers use this for cold-start).
    """
    tenant_dir = _tenant_dir(tenant_id)
    if not tenant_dir.exists():
        return None
    candidates = sorted(tenant_dir.glob("model_v*.joblib"))
    if not candidates:
        return None
    path = candidates[-1]  # timestamps in filenames sort lexically
    try:
        return joblib.load(path)
    except (OSError, EOFError, KeyError) as exc:
        raise ModelStoreError(f"Failed to load model artefact {path.name}: {exc}") from exc


def has_model(tenant_id: str) -> bool:
    """Return True if the tenant has at least one trained scoring model."""
    tenant_dir = _tenant_dir(tenant_id)
    return tenant_dir.exists() and any(tenant_dir.glob("model_v*.joblib"))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_rows(payload: dict, rows: list[dict]) -> list[dict]:
    """
    Score normalized contact rows with a loaded model payload.

    Args:
        payload: Model payload from load_latest().
        rows:    List of dicts each containing company_size, channel, industry
                 (already normalized to the internal vocabulary).

    Returns:
        One dict per input row: cate_estimate (float), uplift_tier
        ("high" | "mid" | "deprioritize"), model_version.
    """
    if not rows:
        return []

    frame = pd.DataFrame(rows)
    for col in _CATEGORICAL_FEATURES:
        if col not in frame.columns:
            frame[col] = "unknown"
    _warn_unseen_levels(frame, payload["categorical_levels"])

    X, _ = _encode_frame(frame, payload["categorical_levels"])
    cate = payload["model_t"].predict(X) - payload["model_c"].predict(X)

    high = payload["high_cutoff"]
    low = payload["deprioritize_cutoff"]
    results = []
    for estimate in cate:
        if estimate >= high:
            tier = "high"
        elif estimate < low:
            tier = "deprioritize"
        else:
            tier = "mid"
        results.append({
            "cate_estimate": float(estimate),
            "uplift_tier":   tier,
            "model_version": payload["model_version"],
        })
    return results


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_frame(
    frame: pd.DataFrame,
    categorical_levels: dict[str, list[str]],
) -> tuple[np.ndarray, list[str]]:
    """
    One-hot encode categoricals against a fixed level vocabulary.

    Matches the dashboard's pd.get_dummies(..., drop_first=True) convention:
    the first sorted level of each column is the reference (all-zero) level.
    Values outside the stored vocabulary — including "unknown" — also encode
    as all zeros, i.e. they fall back to the reference level.

    Args:
        frame:              DataFrame containing the categorical columns.
        categorical_levels: Column → sorted list of levels seen at training.

    Returns:
        (feature_matrix, feature_column_names)
    """
    columns: list[str] = []
    vectors: list[np.ndarray] = []
    for col, levels in categorical_levels.items():
        values = frame[col].astype(str)
        for level in levels[1:]:  # drop_first: skip the reference level
            columns.append(f"{col}_{level}")
            vectors.append((values == level).to_numpy(dtype=float))
    X = np.column_stack(vectors) if vectors else np.zeros((len(frame), 0))
    return X, columns


def _warn_unseen_levels(frame: pd.DataFrame, categorical_levels: dict[str, list[str]]) -> None:
    """Log a warning for category values the model never saw during training."""
    for col, levels in categorical_levels.items():
        unseen = set(frame[col].astype(str).unique()) - set(levels)
        if unseen:
            logger.warning(
                "score: unseen %s values %s — encoded as reference level '%s'.",
                col, sorted(unseen), levels[0],
            )
