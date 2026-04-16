"""
Causal inference core: CUPED, CATE/HTE, DiD, SRM detection, BH correction.

All functions follow the statistical correctness rules in CLAUDE.md:
    - Winsorize BEFORE calling cuped_adjustment (caller's responsibility)
    - CUPED covariate must be pre-experiment period only
    - T-Learner is the default CATE method
    - SRM check runs at alpha=0.01 (more conservative than experiment alpha)
    - BH correction for multi-segment results; Bonferroni is never used

Raises domain-specific CausalEstimationError for estimation failures.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CausalEstimationError(ValueError):
    """Raised when a causal estimation step cannot be completed."""


# ---------------------------------------------------------------------------
# CUPED
# ---------------------------------------------------------------------------


def cuped_adjustment(
    df: pd.DataFrame,
    metric_col: str,
    covariate_col: str,
    treatment_col: str,
) -> dict:
    """
    Compute CUPED-adjusted ATE between treatment and control groups.

    CUPED (Controlled-experiment Using Pre-Experiment Data) reduces variance
    by removing the component of the outcome that is predictable from a
    pre-experiment covariate, without shifting the estimand.

    Formula:
        theta    = Cov(Y, X_pre) / Var(X_pre)
        Y_cuped  = Y - theta * (X_pre - E[X_pre])
        ATE      = E[Y_cuped | T=1] - E[Y_cuped | T=0]

    The covariate MUST be from the pre-experiment period only. Using post-
    experiment data as a covariate introduces bias (leakage).

    Caller must winsorize the covariate_col before calling. Do NOT winsorize
    binary outcomes (0/1 flags) — the metric_col decision is the caller's.

    Args:
        df:             DataFrame with one row per user.
        metric_col:     Post-experiment outcome column.
        covariate_col:  Pre-experiment covariate column (winsorized).
        treatment_col:  Binary treatment indicator column (0/1).

    Returns:
        Dict with keys:
            ate               — adjusted average treatment effect
            ate_se            — standard error of ATE
            p_value           — two-sided p-value (Welch's t-test on adjusted outcomes)
            ci_lower          — 95% CI lower bound (Welch-Satterthwaite t critical value)
            ci_upper          — 95% CI upper bound
            variance_reduction_pct — % variance reduction vs. unadjusted outcome
            theta             — CUPED coefficient (for diagnostics)
            covariate_pearson_r — Pearson r between outcome and covariate (warn if < 0.1)
            n_treatment       — sample size in treatment arm
            n_control         — sample size in control arm

    Raises:
        CausalEstimationError: If required columns are missing, covariate has
                               zero variance, or either arm has fewer than 2 rows.
    """
    for col in (metric_col, covariate_col, treatment_col):
        if col not in df.columns:
            raise CausalEstimationError(f"Column '{col}' not found in DataFrame.")

    sub = df[[metric_col, covariate_col, treatment_col]].dropna()
    if len(sub) < 4:
        raise CausalEstimationError(
            f"Too few non-null rows ({len(sub)}) for CUPED adjustment."
        )

    Y = sub[metric_col].to_numpy(dtype=float)
    X = sub[covariate_col].to_numpy(dtype=float)
    T = sub[treatment_col].to_numpy(dtype=int)

    var_x = float(np.var(X, ddof=1))
    if var_x == 0.0:
        raise CausalEstimationError(
            f"Covariate '{covariate_col}' has zero variance — cannot compute theta."
        )

    theta = float(np.cov(Y, X, ddof=1)[0, 1] / var_x)
    Y_cuped = Y - theta * (X - X.mean())

    # Pearson r diagnostic — warn if covariate is weakly correlated
    r, _ = stats.pearsonr(Y, X)
    if abs(r) < 0.1:
        logger.warning(
            "CUPED: covariate '%s' has low Pearson r=%.3f with outcome '%s'. "
            "Variance reduction will be minimal.",
            covariate_col, r, metric_col,
        )

    treatment_mask = T == 1
    control_mask = T == 0
    y_t = Y_cuped[treatment_mask]
    y_c = Y_cuped[control_mask]

    if len(y_t) < 2 or len(y_c) < 2:
        raise CausalEstimationError(
            f"Each arm needs at least 2 observations. "
            f"Got treatment={len(y_t)}, control={len(y_c)}."
        )

    ate = float(y_t.mean() - y_c.mean())
    _, p_value = stats.ttest_ind(y_t, y_c, equal_var=False)

    # Welch-Satterthwaite SE and degrees of freedom
    var_t = float(np.var(y_t, ddof=1)) / len(y_t)
    var_c = float(np.var(y_c, ddof=1)) / len(y_c)
    se = float(np.sqrt(var_t + var_c))

    # Use t critical value with Welch-Satterthwaite df — more accurate than z=1.96
    # at small N (e.g. df≈100 gives t=1.984 vs z=1.960 — CIs would be too narrow
    # at the minimum 200-user threshold if z were used).
    welch_df = (var_t + var_c) ** 2 / (
        var_t ** 2 / (len(y_t) - 1) + var_c ** 2 / (len(y_c) - 1)
    )
    t_crit = float(stats.t.ppf(0.975, df=welch_df))
    ci_lower = ate - t_crit * se
    ci_upper = ate + t_crit * se

    var_raw = float(np.var(Y, ddof=1))
    var_adj = float(np.var(Y_cuped, ddof=1))
    variance_reduction_pct = float((1 - var_adj / var_raw) * 100) if var_raw > 0 else 0.0

    if variance_reduction_pct < 0:
        logger.warning(
            "CUPED: adjustment INCREASED variance by %.1f%% (theta=%.4f, r=%.3f). "
            "The covariate may be negatively correlated or near-zero correlated with the outcome. "
            "Reporting variance_reduction_pct=%.1f%% (not clamped — negative value is informative).",
            abs(variance_reduction_pct), theta, r, variance_reduction_pct,
        )

    logger.info(
        "CUPED | ATE=%.4f SE=%.4f p=%.4f | var_reduction=%.1f%% | r=%.3f | theta=%.4f | df=%.1f",
        ate, se, p_value, variance_reduction_pct, r, theta, welch_df,
    )

    return {
        "ate":                    ate,
        "ate_se":                 se,
        "p_value":                float(p_value),
        "ci_lower":               ci_lower,
        "ci_upper":               ci_upper,
        "variance_reduction_pct": variance_reduction_pct,
        "theta":                  theta,
        "covariate_pearson_r":    float(r),
        "n_treatment":            int(treatment_mask.sum()),
        "n_control":              int(control_mask.sum()),
    }


# ---------------------------------------------------------------------------
# CATE / HTE via EconML
# ---------------------------------------------------------------------------


def estimate_cate(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: list[str],
    method: Literal["s_learner", "t_learner", "causal_forest"] = "t_learner",
) -> pd.DataFrame:
    """
    Estimate Conditional Average Treatment Effects (CATE) per user.

    Default method: T-Learner (most defensible in interviews — separate models
    for treatment and control, no shared parametric assumptions).

    CausalForest is gated: only runs when N > 5000 per arm. At smaller N it
    over-fits on the treatment effect heterogeneity — use T-Learner instead.

    Caller should log-transform continuous features before calling:
        df[feat] = log_transform(df[feat])

    Args:
        df:           DataFrame with one row per user.
        outcome_col:  Binary or continuous outcome (e.g. 'activated').
        treatment_col: Binary treatment indicator (0/1).
        feature_cols: List of pre-treatment covariate columns for the CATE model.
        method:       Estimation method — "t_learner", "s_learner", or
                      "causal_forest" (requires N > 5000 per arm).

    Returns:
        Original DataFrame with three new columns appended:
            cate_estimate  — individual-level CATE point estimate
            cate_lower     — 95% CI lower (NaN for learner methods)
            cate_upper     — 95% CI upper (NaN for learner methods)

    Raises:
        CausalEstimationError: If columns are missing, method is unsupported,
                               or CausalForest is requested at N < 5000.
    """
    for col in [outcome_col, treatment_col] + feature_cols:
        if col not in df.columns:
            raise CausalEstimationError(f"Column '{col}' not found in DataFrame.")

    sub = df[[outcome_col, treatment_col] + feature_cols].dropna().copy()
    if len(sub) < 10:
        raise CausalEstimationError(
            f"Too few rows ({len(sub)}) for CATE estimation. Need at least 10."
        )

    Y = sub[outcome_col].to_numpy(dtype=float)
    T = sub[treatment_col].to_numpy(dtype=float)
    X = sub[feature_cols].to_numpy(dtype=float)

    n_t = int(T.sum())
    n_c = int((1 - T).sum())

    # CausalForest gate: N must exceed 5000 per arm (not ≤ 5000)
    if method == "causal_forest" and min(n_t, n_c) < 5000:
        raise CausalEstimationError(
            f"CausalForest requires N > 5000 per arm. "
            f"Got treatment={n_t}, control={n_c}. Use t_learner instead."
        )

    logger.info(
        "estimate_cate | method=%s | n=%d | n_treatment=%d | n_control=%d | features=%s",
        method, len(sub), n_t, n_c, feature_cols,
    )

    if method == "t_learner":
        cate_estimates = _t_learner(Y, T, X)
        ci_lower = ci_upper = None
    elif method == "s_learner":
        cate_estimates = _s_learner(Y, T, X)
        ci_lower = ci_upper = None
    elif method == "causal_forest":
        cate_estimates, ci_lower, ci_upper = _causal_forest(Y, T, X)
    else:
        raise CausalEstimationError(
            f"Unknown method '{method}'. Must be t_learner, s_learner, or causal_forest."
        )

    result = df.copy()
    cate_series = pd.Series(np.nan, index=df.index)
    cate_series.loc[sub.index] = cate_estimates
    result["cate_estimate"] = cate_series

    if ci_lower is not None:
        lo = pd.Series(np.nan, index=df.index)
        hi = pd.Series(np.nan, index=df.index)
        lo.loc[sub.index] = ci_lower
        hi.loc[sub.index] = ci_upper
        result["cate_lower"] = lo
        result["cate_upper"] = hi
    else:
        result["cate_lower"] = np.nan
        result["cate_upper"] = np.nan

    logger.info(
        "CATE estimates | mean=%.4f | std=%.4f | min=%.4f | max=%.4f",
        float(np.nanmean(cate_estimates)),
        float(np.nanstd(cate_estimates)),
        float(np.nanmin(cate_estimates)),
        float(np.nanmax(cate_estimates)),
    )

    return result


def _t_learner(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    T-Learner: fit separate response surfaces for treatment and control.

    mu1(x) = E[Y | T=1, X=x]
    mu0(x) = E[Y | T=0, X=x]
    CATE(x) = mu1(x) - mu0(x)
    """
    from sklearn.ensemble import GradientBoostingRegressor

    idx_t = T == 1
    idx_c = T == 0

    model_t = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model_c = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

    model_t.fit(X[idx_t], Y[idx_t])
    model_c.fit(X[idx_c], Y[idx_c])

    return model_t.predict(X) - model_c.predict(X)


def _s_learner(Y: np.ndarray, T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    S-Learner: single model with treatment as a feature.

    mu(x, t) = E[Y | X=x, T=t]
    CATE(x)  = mu(x, 1) - mu(x, 0)
    """
    from sklearn.ensemble import GradientBoostingRegressor

    XT = np.column_stack([X, T])
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(XT, Y)

    X1 = np.column_stack([X, np.ones(len(X))])
    X0 = np.column_stack([X, np.zeros(len(X))])
    return model.predict(X1) - model.predict(X0)


def _causal_forest(
    Y: np.ndarray,
    T: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CausalForest via EconML. Only called when N > 5000 per arm.

    Returns (point_estimates, ci_lower, ci_upper).
    """
    from econml.grf import CausalForest

    cf = CausalForest(n_estimators=200, min_samples_leaf=5, random_state=42)
    cf.fit(X, T, Y)
    point, lb, ub = cf.predict(X, interval=True, alpha=0.05)
    return point.flatten(), lb.flatten(), ub.flatten()


# ---------------------------------------------------------------------------
# DiD
# ---------------------------------------------------------------------------


def diff_in_diff(
    df: pd.DataFrame,
    pre_window: tuple[str, str],
    post_window: tuple[str, str],
    treatment_group_col: str,
    outcome_col: str,
    date_col: str = "event_date",
) -> dict:
    """
    Estimate causal lift of a campaign/policy change via Difference-in-Differences.

    Fits the regression:
        Y = β0 + β1·Post + β2·Treated + β3·(Post × Treated) + ε

    β3 is the DiD estimator — the causal lift attributable to treatment.

    Visual parallel-trends check: the caller should plot pre-period trends
    for treatment and control groups before interpreting results.

    Args:
        df:                 Long-format event DataFrame (one row per user-period).
        pre_window:         (start_date, end_date) ISO strings for pre-period.
        post_window:        (start_date, end_date) ISO strings for post-period.
        treatment_group_col: Binary column: 1 if user is in treatment group, 0 if control.
        outcome_col:        Outcome metric column (binary or continuous).
        date_col:           Date/datetime column to filter periods.

    Returns:
        Dict with keys:
            did_estimate  — β3 (interaction coefficient)
            std_err       — standard error of β3
            p_value       — two-sided p-value for β3
            ci_lower      — 95% CI lower
            ci_upper      — 95% CI upper
            n_pre         — rows in pre-period
            n_post        — rows in post-period

    Raises:
        CausalEstimationError: If required columns are missing or a window is empty.
    """
    import statsmodels.formula.api as smf

    for col in (treatment_group_col, outcome_col, date_col):
        if col not in df.columns:
            raise CausalEstimationError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    pre_start, pre_end = pd.Timestamp(pre_window[0], tz="UTC"), pd.Timestamp(pre_window[1], tz="UTC")
    post_start, post_end = pd.Timestamp(post_window[0], tz="UTC"), pd.Timestamp(post_window[1], tz="UTC")

    pre_mask  = (df[date_col] >= pre_start)  & (df[date_col] <= pre_end)
    post_mask = (df[date_col] >= post_start) & (df[date_col] <= post_end)

    pre_df  = df[pre_mask].copy()
    post_df = df[post_mask].copy()

    if pre_df.empty:
        raise CausalEstimationError(f"No rows in pre-period {pre_window}.")
    if post_df.empty:
        raise CausalEstimationError(f"No rows in post-period {post_window}.")

    pre_df["_post"]    = 0
    post_df["_post"]   = 1
    panel = pd.concat([pre_df, post_df], ignore_index=True)
    panel = panel.rename(columns={treatment_group_col: "_treated", outcome_col: "_outcome"})

    formula = "_outcome ~ _post + _treated + _post:_treated"
    model = smf.ols(formula, data=panel).fit()

    coef = model.params["_post:_treated"]
    se   = model.bse["_post:_treated"]
    pval = model.pvalues["_post:_treated"]
    ci   = model.conf_int().loc["_post:_treated"]

    logger.info(
        "DiD | β3=%.4f SE=%.4f p=%.4f | n_pre=%d n_post=%d",
        coef, se, pval, len(pre_df), len(post_df),
    )

    return {
        "did_estimate": float(coef),
        "std_err":      float(se),
        "p_value":      float(pval),
        "ci_lower":     float(ci.iloc[0]),
        "ci_upper":     float(ci.iloc[1]),
        "n_pre":        len(pre_df),
        "n_post":       len(post_df),
    }


# ---------------------------------------------------------------------------
# SRM Detection
# ---------------------------------------------------------------------------


def detect_srm(
    n_treatment: int,
    n_control: int,
    expected_split: float = 0.5,
    alpha: float = 0.01,
) -> dict:
    """
    Detect Sample Ratio Mismatch (SRM) using a chi-square test.

    SRM occurs when the observed treatment/control split differs significantly
    from the intended split. It is a data-quality issue (often caused by logging
    bugs or selective dropout) that invalidates causal interpretation.

    Always run BEFORE reporting any experiment results.
    Uses alpha=0.01 (more conservative than experiment alpha) per CLAUDE.md.

    Args:
        n_treatment:    Observed count in treatment arm.
        n_control:      Observed count in control arm.
        expected_split: Intended fraction assigned to treatment (default 0.5).
                        Must match the split used during randomisation — do NOT
                        default to 0.5 if the experiment used a different split.
        alpha:          Significance level for SRM test (default 0.01).

    Returns:
        Dict with keys:
            srm_detected    — True if SRM is present at the given alpha
            p_value         — chi-square p-value
            observed_split  — actual treatment fraction
            expected_split  — intended treatment fraction
            chi2_stat       — test statistic
            recommendation  — human-readable action string

    Raises:
        ValueError: If n_treatment or n_control is negative, or expected_split
                    is outside (0, 1).
    """
    if n_treatment < 0 or n_control < 0:
        raise ValueError("n_treatment and n_control must be non-negative.")
    if not (0.0 < expected_split < 1.0):
        raise ValueError(f"expected_split must be in (0, 1), got {expected_split}.")

    n_total = n_treatment + n_control
    if n_total == 0:
        raise ValueError("Total sample size is 0 — no data to test.")

    expected_t = n_total * expected_split
    expected_c = n_total * (1 - expected_split)

    chi2, p_value = stats.chisquare(
        f_obs=[n_treatment, n_control],
        f_exp=[expected_t, expected_c],
    )

    srm_detected = bool(p_value < alpha)
    observed_split = n_treatment / n_total

    if srm_detected:
        recommendation = (
            f"SRM DETECTED (p={p_value:.4f} < {alpha}). "
            f"Observed split {observed_split:.3f} vs. expected {expected_split:.3f}. "
            f"Do NOT interpret experiment results — investigate logging and assignment pipeline."
        )
        logger.warning(recommendation)
    else:
        recommendation = (
            f"No SRM detected (p={p_value:.4f} >= {alpha}). "
            f"Observed split {observed_split:.3f} is consistent with expected {expected_split:.3f}."
        )

    return {
        "srm_detected":    srm_detected,
        "p_value":         float(p_value),
        "observed_split":  observed_split,
        "expected_split":  expected_split,
        "chi2_stat":       float(chi2),
        "recommendation":  recommendation,
    }


# ---------------------------------------------------------------------------
# Multiple Testing Correction
# ---------------------------------------------------------------------------


def bh_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Apply Benjamini-Hochberg (BH) False Discovery Rate correction.

    BH controls the expected proportion of false discoveries among all
    rejected hypotheses. It is less conservative than Bonferroni
    (which controls FWER) — more rejections at the same alpha, while still
    providing a meaningful guarantee.

    Use whenever reporting results for more than one segment simultaneously.
    Never use Bonferroni for this use case (too conservative for FDR context).

    Algorithm:
        1. Sort p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
        2. Find the largest k such that p_(k) ≤ (k/m) × alpha
        3. Reject all null hypotheses p_(1) through p_(k)

    Args:
        p_values: List of raw p-values from m simultaneous tests.
        alpha:    FDR level (default 0.05).

    Returns:
        Boolean list of length m — True means the null hypothesis is rejected
        at the BH-corrected threshold.

    Raises:
        ValueError: If p_values is empty or alpha is outside (0, 1).
    """
    if not p_values:
        raise ValueError("p_values list is empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * m

    # Find the largest k where p_(k) <= k/m * alpha
    last_reject = -1
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            last_reject = rank

    # Reject all hypotheses up through last_reject
    for rank, (orig_idx, _) in enumerate(indexed, start=1):
        if rank <= last_reject:
            reject[orig_idx] = True

    n_rejected = sum(reject)
    logger.info(
        "BH correction | m=%d | alpha=%.3f | rejected=%d (%.1f%%)",
        m, alpha, n_rejected, n_rejected / m * 100,
    )

    return reject
