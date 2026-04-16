"""
Preprocessing utilities for GTMLens experiment and causal pipelines.

Decision rule (from CLAUDE.md):
    experiment / CUPED pipeline  → winsorize   (preserves units, reduces variance)
    ML model inputs (CATE)       → log_transform (stabilises regression residuals)
    raw reporting                → none

Critical correctness rules:
    - Always winsorize BEFORE computing CUPED theta
    - WINSORIZE_UPPER_PCT read from env (default 0.99)
    - Never winsorize binary outcomes (0/1 flags)
    - Apply the SAME transformation to both treatment and control
    - Input series is NEVER mutated
"""

import logging
import os
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _winsorize_upper_pct() -> float:
    """Read WINSORIZE_UPPER_PCT from env, default 0.99."""
    return float(os.getenv("WINSORIZE_UPPER_PCT", "0.99"))


def winsorize(series: pd.Series, upper_pct: float | None = None) -> pd.Series:
    """
    Cap values at the given upper percentile (one-sided winsorization).

    Only the upper tail is clipped. The lower tail is left untouched because
    GTM metrics (revenue, session count) have a meaningful zero floor.
    Winsorizing the upper tail reduces variance driven by extreme outliers
    without shifting the estimand.

    Args:
        series:    Numeric Series to winsorize. Must not contain all-NaN values.
        upper_pct: Upper percentile cap in (0, 1]. Reads WINSORIZE_UPPER_PCT
                   from env if None. Default env value is 0.99.

    Returns:
        New Series with values above the upper_pct quantile clipped.
        The input series is never mutated.

    Raises:
        ValueError: If upper_pct is not in (0, 1].
        ValueError: If series is entirely NaN.
        TypeError:  If series is not a numeric dtype.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"winsorize requires a numeric Series, got dtype '{series.dtype}'."
        )

    pct = upper_pct if upper_pct is not None else _winsorize_upper_pct()
    if not (0.0 < pct <= 1.0):
        raise ValueError(f"upper_pct must be in (0, 1], got {pct}.")

    clean = series.dropna()
    if clean.empty:
        raise ValueError("Series is entirely NaN — cannot winsorize.")

    cap = float(np.quantile(clean.values, pct))
    result = series.clip(upper=cap)

    n_clipped = int((series > cap).sum())
    if n_clipped > 0:
        logger.debug(
            "winsorize: clipped %d values (%.2f%%) above %.4f",
            n_clipped, n_clipped / len(series) * 100, cap,
        )

    return result


def log_transform(series: pd.Series, offset: float = 1.0) -> pd.Series:
    """
    Apply log(x + offset) transformation.

    Used for CATE model inputs (EconML feature matrix) to stabilise
    regression residuals. Do NOT use on the experiment estimator or CUPED
    — log-transform changes the estimand (ratio rather than difference).

    Args:
        series: Numeric Series. Zero values are handled via the offset.
        offset: Constant added before taking the log. Must be > 0.
                Default 1.0 (reads LOG_OFFSET env var if not supplied).

    Returns:
        New Series with log(series + offset) values.
        The input series is never mutated.

    Raises:
        ValueError: If offset <= 0.
        ValueError: If any (value + offset) <= 0 after applying offset.
        TypeError:  If series is not a numeric dtype.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"log_transform requires a numeric Series, got dtype '{series.dtype}'."
        )

    eff_offset = float(os.getenv("LOG_OFFSET", str(offset)))
    if eff_offset <= 0:
        raise ValueError(f"offset must be > 0, got {eff_offset}.")

    shifted = series + eff_offset
    if (shifted.dropna() <= 0).any():
        raise ValueError(
            f"log_transform: series + offset ({eff_offset}) contains values <= 0. "
            f"Increase offset so all values are positive."
        )

    return np.log(shifted)


def preprocess_metric(
    series: pd.Series,
    method: Literal["winsorize", "log", "none"] = "winsorize",
    upper_pct: float | None = None,
    offset: float = 1.0,
) -> pd.Series:
    """
    Apply the appropriate preprocessing transformation for the given pipeline context.

    Decision guide:
        "winsorize" — experiment ATE estimator, CUPED covariate, revenue metrics
        "log"       — EconML CATE model feature inputs
        "none"      — binary outcomes (reply rate, conversion flag), raw reporting

    Args:
        series:    Numeric Series to transform.
        method:    Transformation to apply. One of "winsorize", "log", "none".
        upper_pct: Passed to winsorize(). Uses WINSORIZE_UPPER_PCT env if None.
        offset:    Passed to log_transform(). Uses LOG_OFFSET env if not overridden.

    Returns:
        Transformed Series. Input is never mutated.

    Raises:
        ValueError: If method is not one of the allowed values.
        TypeError:  If series dtype is not numeric.
    """
    if method == "winsorize":
        return winsorize(series, upper_pct=upper_pct)
    if method == "log":
        return log_transform(series, offset=offset)
    if method == "none":
        return series.copy()
    raise ValueError(
        f"method must be one of 'winsorize', 'log', 'none'. Got '{method}'."
    )
