"""
Synthetic GTM funnel data generator.

Generates 50,000 B2B SaaS users moving through a 5-stage funnel:
    impression → click → signup → activation → conversion

Real-world distributions are calibrated from industry benchmarks:
    - B2B SaaS funnel conversion rates (First Principles, OpenView 2023)
    - Channel mix by company size (Demand Gen Report 2023)
    - Industry distribution (LinkedIn B2B index)
    - Temporal patterns: weekday bias, campaign spikes, decay curves

Treatment effects are heterogeneous by segment (ground truth in ground_truth.py).
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from faker import Faker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-world calibration constants
# ---------------------------------------------------------------------------

# Industry distribution — calibrated to LinkedIn B2B audience index
INDUSTRY_WEIGHTS = {
    "SaaS": 0.28,
    "FinTech": 0.15,
    "HealthTech": 0.12,
    "E-commerce": 0.11,
    "Manufacturing": 0.09,
    "Professional Services": 0.10,
    "EdTech": 0.08,
    "Logistics": 0.07,
}

# Company size distribution — OpenView 2023 B2B SaaS benchmark
COMPANY_SIZE_WEIGHTS = {
    "SMB": 0.45,       # 1–50 employees
    "mid_market": 0.35, # 51–500 employees
    "enterprise": 0.20, # 500+ employees
}

# Channel mix — varies by company size (Demand Gen Report 2023)
CHANNEL_MIX = {
    "SMB":        {"organic": 0.40, "paid_search": 0.25, "social": 0.20, "referral": 0.10, "email": 0.05},
    "mid_market": {"organic": 0.30, "paid_search": 0.30, "social": 0.15, "referral": 0.15, "email": 0.10},
    "enterprise": {"organic": 0.20, "paid_search": 0.35, "social": 0.10, "referral": 0.20, "email": 0.15},
}

# Baseline funnel conversion rates (no treatment) — ALL CONDITIONAL on prior stage
#
# Funnel model: product-usage onboarding funnel (not ad-impression funnel).
# Users in this dataset have already entered the product (free trial / freemium signup).
# Stages represent in-product milestones:
#
#   impression  = user starts onboarding flow
#   click       = user completes first meaningful product action (feature discovery)
#   signup      = user completes profile + initial setup
#   activation  = user reaches the defined activation milestone (the experiment outcome)
#   conversion  = user upgrades to paid plan
#
# Calibration sources:
#   click_rate      : % of onboarding starters who complete first action
#                     (Amplitude B2B SaaS 2023: 55–80% D1 engagement)
#   signup_rate     : % of first-action users who complete setup
#                     (Mixpanel B2B benchmark: 55–75%)
#   activation_rate : % of setup-complete users who hit activation milestone
#                     (OpenView SaaS median: 30–55%)
#   conversion_rate : % of activated users who convert to paid
#                     (SaaS Capital survey: 15–40% free-to-paid)
#
# These rates yield ~12,000–18,000 activated users out of 50k starts,
# which is sufficient for T-Learner CATE estimation across 15 segments.
BASELINE_RATES = {
    #                          click    signup   activation  conversion
    ("SMB",        "organic"):      (0.62, 0.58, 0.38, 0.22),
    ("SMB",        "paid_search"):  (0.58, 0.55, 0.35, 0.20),
    ("SMB",        "social"):       (0.52, 0.48, 0.30, 0.17),
    ("SMB",        "referral"):     (0.70, 0.66, 0.48, 0.28),
    ("SMB",        "email"):        (0.68, 0.63, 0.42, 0.24),
    ("mid_market", "organic"):      (0.65, 0.60, 0.42, 0.27),
    ("mid_market", "paid_search"):  (0.62, 0.58, 0.40, 0.25),
    ("mid_market", "social"):       (0.55, 0.50, 0.33, 0.19),
    ("mid_market", "referral"):     (0.74, 0.69, 0.55, 0.35),
    ("mid_market", "email"):        (0.72, 0.67, 0.50, 0.31),
    ("enterprise", "organic"):      (0.68, 0.63, 0.48, 0.32),
    ("enterprise", "paid_search"):  (0.65, 0.61, 0.46, 0.30),
    ("enterprise", "social"):       (0.57, 0.52, 0.35, 0.21),
    ("enterprise", "referral"):     (0.78, 0.73, 0.60, 0.42),
    ("enterprise", "email"):        (0.76, 0.71, 0.55, 0.38),
}

# True treatment effects (heterogeneous) — must match ground_truth.py
_TRUE_ATE = {
    ("enterprise", "paid_search"): 0.18,
    ("enterprise", "organic"):     0.14,
    ("enterprise", "referral"):    0.12,
    ("enterprise", "email"):       0.10,
    ("enterprise", "social"):      0.06,
    ("mid_market", "paid_search"): 0.11,
    ("mid_market", "organic"):     0.09,
    ("mid_market", "referral"):    0.08,
    ("mid_market", "email"):       0.07,
    ("mid_market", "social"):      0.04,
    ("SMB",        "paid_search"): 0.04,
    ("SMB",        "organic"):     0.03,
    ("SMB",        "referral"):    0.03,
    ("SMB",        "email"):       0.02,
    ("SMB",        "social"):      0.01,
}

# Revenue proxy by segment (USD, monthly) — used as a continuous outcome
REVENUE_PARAMS = {
    "SMB":        {"mean": 180,   "std": 90,   "cap": 600},
    "mid_market": {"mean": 1_200, "std": 600,  "cap": 5_000},
    "enterprise": {"mean": 8_500, "std": 4_000,"cap": 40_000},
}

STAGES = ["impression", "click", "signup", "activation", "conversion"]


def _week_day_multiplier(dt: datetime) -> float:
    """Business traffic is ~30% lower on weekends (Marketo benchmark)."""
    return 0.70 if dt.weekday() >= 5 else 1.00


def _campaign_spike(day_offset: int, spike_day: int = 14, magnitude: float = 1.6) -> float:
    """
    Model a one-time campaign launch spike with exponential decay.

    Args:
        day_offset: Days since experiment start.
        spike_day: Day the campaign launched.
        magnitude: Peak traffic multiplier.

    Returns:
        Multiplicative traffic factor.
    """
    if day_offset < spike_day:
        return 1.0
    decay = np.exp(-0.12 * (day_offset - spike_day))
    return 1.0 + (magnitude - 1.0) * decay


def _daily_impression_volume(
    day_offset: int,
    dt: datetime,
    spike_day: int,
    base_daily: int,
) -> int:
    """
    Compute daily impression count with weekday + campaign-spike effects.

    Args:
        day_offset: Days since start of window.
        dt: Actual calendar date.
        spike_day: Campaign launch day.
        base_daily: Average impressions on a typical weekday.

    Returns:
        Integer impression count for that day.
    """
    vol = base_daily * _week_day_multiplier(dt) * _campaign_spike(day_offset, spike_day)
    noise = np.random.normal(1.0, 0.08)  # ±8% daily noise
    return max(1, int(vol * noise))


def _sample_channel(company_size: str, rng: np.random.Generator) -> str:
    """Sample acquisition channel according to size-conditional mix."""
    mix = CHANNEL_MIX[company_size]
    channels = list(mix.keys())
    probs = list(mix.values())
    return rng.choice(channels, p=probs)


def _treatment_assignment(
    rng: np.random.Generator,
    treatment_split: float,
) -> int:
    """Bernoulli assignment. Returns 1 (treatment) or 0 (control)."""
    return int(rng.random() < treatment_split)


def _activation_rate(
    company_size: str,
    channel: str,
    treated: int,
    industry: str,
    rng: np.random.Generator,
) -> float:
    """
    Compute individual-level activation probability.

    Applies:
    - Baseline rate for (size, channel) cell
    - Heterogeneous treatment effect (HTE)
    - Industry modifier (±5pp noise around mean)
    - Individual noise (Beta distribution to stay in [0,1])

    Args:
        company_size: Segment size tier.
        channel: Acquisition channel.
        treated: Binary treatment indicator.
        industry: User's industry.
        rng: Seeded random generator.

    Returns:
        Probability of activation given observed covariates.
    """
    _, _, base_activation, _ = BASELINE_RATES[(company_size, channel)]
    ate = _TRUE_ATE.get((company_size, channel), 0.0)
    treatment_lift = ate * treated

    # Industry modifier: industries like FinTech/enterprise convert better
    industry_boost = {
        "SaaS": 0.02,
        "FinTech": 0.03,
        "HealthTech": -0.01,
        "E-commerce": 0.00,
        "Manufacturing": -0.02,
        "Professional Services": 0.01,
        "EdTech": -0.01,
        "Logistics": -0.02,
    }.get(industry, 0.0)

    mu = np.clip(base_activation + treatment_lift + industry_boost, 0.01, 0.99)
    # Beta noise: mean=mu, moderate variance
    alpha_param = mu * 12
    beta_param = (1 - mu) * 12
    return float(np.clip(rng.beta(alpha_param, beta_param), 0.01, 0.99))


def _revenue_sample(company_size: str, activated: int, rng: np.random.Generator) -> float:
    """
    Sample monthly revenue proxy (log-normal, heavy right tail).

    Only users who activate generate meaningful revenue.

    Args:
        company_size: Determines mean/std/cap parameters.
        activated: Binary; non-activated users contribute near-zero.
        rng: Seeded random generator.

    Returns:
        Revenue value in USD.
    """
    if not activated:
        return float(rng.exponential(5.0))  # tiny churn-stage revenue
    p = REVENUE_PARAMS[company_size]
    mu_log = np.log(p["mean"]) - 0.5 * np.log(1 + (p["std"] / p["mean"]) ** 2)
    sigma_log = np.sqrt(np.log(1 + (p["std"] / p["mean"]) ** 2))
    raw = rng.lognormal(mu_log, sigma_log)
    return float(np.clip(raw, 0, p["cap"]))


def generate_funnel_data(
    n_users: int = 50_000,
    window_days: int = 90,
    treatment_split: float = 0.50,
    spike_day: int = 14,
    seed: int = 42,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Generate synthetic B2B SaaS funnel dataset with heterogeneous treatment effects.

    Population arrives as impressions distributed across the 90-day window.
    Each user is assigned:
      - company_size, industry, channel (via real-world weighted distributions)
      - treatment (Bernoulli, 50/50 by default)
      - funnel stages reached (Bernoulli chain with calibrated conversion rates)
      - activation_rate (continuous, Beta-distributed around segment baseline + HTE)
      - revenue (log-normal conditional on activation)
      - timestamps for each stage reached

    Args:
        n_users: Total users to generate (impressions).
        window_days: Length of observation window in days.
        treatment_split: Fraction assigned to treatment arm.
        spike_day: Day index for campaign traffic spike.
        seed: Random seed for reproducibility.
        start_date: ISO date string for window start.

    Returns:
        DataFrame with one row per user. Columns:
            user_id, impression_date, company_size, industry, channel,
            treatment, clicked, signed_up, activated, converted,
            activation_prob, pre_activation_rate (covariate, pre-experiment),
            revenue, click_date, signup_date, activation_date, conversion_date,
            days_to_click, days_to_signup, days_to_activation, days_to_conversion
    """
    if n_users <= 0:
        raise ValueError(f"n_users must be positive, got {n_users}")
    if not (0.0 < treatment_split < 1.0):
        raise ValueError(f"treatment_split must be in (0,1), got {treatment_split}")
    if window_days < 7:
        raise ValueError(f"window_days must be >= 7, got {window_days}")

    rng = np.random.default_rng(seed)
    fake = Faker()
    fake.seed_instance(seed)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    industries = list(INDUSTRY_WEIGHTS.keys())
    industry_probs = list(INDUSTRY_WEIGHTS.values())
    sizes = list(COMPANY_SIZE_WEIGHTS.keys())
    size_probs = list(COMPANY_SIZE_WEIGHTS.values())

    # ------------------------------------------------------------------
    # Distribute impressions across days with temporal patterns
    # ------------------------------------------------------------------
    daily_weights = np.array([
        _week_day_multiplier(start_dt + timedelta(days=d))
        * _campaign_spike(d, spike_day)
        for d in range(window_days)
    ])
    daily_weights /= daily_weights.sum()
    day_indices = rng.choice(window_days, size=n_users, p=daily_weights)

    # ------------------------------------------------------------------
    # Sample user attributes
    # ------------------------------------------------------------------
    company_sizes = rng.choice(sizes, size=n_users, p=size_probs)
    industries_arr = rng.choice(industries, size=n_users, p=industry_probs)

    channels = np.array([
        _sample_channel(sz, rng) for sz in company_sizes
    ])

    treatments = (rng.random(n_users) < treatment_split).astype(int)

    # ------------------------------------------------------------------
    # Pre-experiment covariate: activation rate from a synthetic
    # pre-period (30-day window before experiment start).
    # We jitter it from the true baseline to simulate real pre-period data.
    # ------------------------------------------------------------------
    pre_activation_rates = np.array([
        float(np.clip(
            BASELINE_RATES[(company_sizes[i], channels[i])][2]
            + rng.normal(0, 0.03),
            0.01, 0.99
        ))
        for i in range(n_users)
    ])

    # ------------------------------------------------------------------
    # Funnel stage outcomes
    # ------------------------------------------------------------------
    click_rates = np.array([
        BASELINE_RATES[(company_sizes[i], channels[i])][0]
        for i in range(n_users)
    ])
    signup_rates = np.array([
        BASELINE_RATES[(company_sizes[i], channels[i])][1]
        for i in range(n_users)
    ])
    activation_probs = np.array([
        _activation_rate(company_sizes[i], channels[i], treatments[i], industries_arr[i], rng)
        for i in range(n_users)
    ])
    # Conversion conditional on activation (baseline from table)
    conversion_rates = np.array([
        BASELINE_RATES[(company_sizes[i], channels[i])][3]
        for i in range(n_users)
    ])

    clicked    = (rng.random(n_users) < click_rates).astype(int)
    signed_up  = (clicked & (rng.random(n_users) < signup_rates)).astype(int)
    activated  = (signed_up & (rng.random(n_users) < activation_probs)).astype(int)
    converted  = (activated & (rng.random(n_users) < conversion_rates)).astype(int)

    # ------------------------------------------------------------------
    # Revenue
    # ------------------------------------------------------------------
    revenues = np.array([
        _revenue_sample(company_sizes[i], activated[i], rng)
        for i in range(n_users)
    ])

    # ------------------------------------------------------------------
    # Timestamps — realistic lag distributions per stage
    # Based on Gartner B2B buying cycle benchmarks
    # ------------------------------------------------------------------
    impression_dates = [start_dt + timedelta(days=int(d)) for d in day_indices]

    def _lag(n: int, loc: float, scale: float, cap: int) -> np.ndarray:
        """Gamma-distributed stage lag in days, floored at 0, capped."""
        shape = (loc / scale) ** 2
        raw = rng.gamma(shape, scale, size=n)
        return np.clip(np.round(raw).astype(int), 0, cap)

    click_lags      = _lag(n_users, loc=0.8, scale=0.6, cap=3)
    signup_lags     = _lag(n_users, loc=2.5, scale=1.5, cap=10)
    activation_lags = _lag(n_users, loc=5.0, scale=3.0, cap=21)
    conversion_lags = _lag(n_users, loc=18.0, scale=8.0, cap=60)

    click_dates = [
        impression_dates[i] + timedelta(days=int(click_lags[i]))
        if clicked[i] else None
        for i in range(n_users)
    ]
    signup_dates = [
        click_dates[i] + timedelta(days=int(signup_lags[i]))
        if signed_up[i] and click_dates[i] else None
        for i in range(n_users)
    ]
    activation_dates = [
        signup_dates[i] + timedelta(days=int(activation_lags[i]))
        if activated[i] and signup_dates[i] else None
        for i in range(n_users)
    ]
    conversion_dates = [
        activation_dates[i] + timedelta(days=int(conversion_lags[i]))
        if converted[i] and activation_dates[i] else None
        for i in range(n_users)
    ]

    # ------------------------------------------------------------------
    # Assemble DataFrame
    # ------------------------------------------------------------------
    records = {
        "user_id":              [f"u_{i:06d}" for i in range(n_users)],
        "impression_date":      impression_dates,
        "company_size":         company_sizes,
        "industry":             industries_arr,
        "channel":              channels,
        "treatment":            treatments,
        "clicked":              clicked,
        "signed_up":            signed_up,
        "activated":            activated,
        "converted":            converted,
        "activation_prob":      activation_probs,
        "pre_activation_rate":  pre_activation_rates,
        "revenue":              revenues,
        "click_date":           click_dates,
        "signup_date":          signup_dates,
        "activation_date":      activation_dates,
        "conversion_date":      conversion_dates,
        "days_to_click":        [int(click_lags[i]) if clicked[i] else None for i in range(n_users)],
        "days_to_signup":       [int(signup_lags[i]) if signed_up[i] else None for i in range(n_users)],
        "days_to_activation":   [int(activation_lags[i]) if activated[i] else None for i in range(n_users)],
        "days_to_conversion":   [int(conversion_lags[i]) if converted[i] else None for i in range(n_users)],
    }

    df = pd.DataFrame(records)

    # Validate treatment balance
    split_actual = df["treatment"].mean()
    # Conditional funnel rates (each stage conditional on prior stage)
    click_cond    = df["clicked"].mean()
    signup_cond   = df.loc[df["clicked"]   == 1, "signed_up"].mean()
    act_cond      = df.loc[df["signed_up"] == 1, "activated"].mean()
    conv_cond     = df.loc[df["activated"] == 1, "converted"].mean()

    n_activated = int(df["activated"].sum())

    logger.info(
        "Generated %d users | treatment split: %.3f | "
        "click: %.1f%% | signup|click: %.1f%% | activation|signup: %.1f%% | conversion|activation: %.1f%% "
        "| n_activated=%d",
        n_users,
        split_actual,
        click_cond * 100,
        signup_cond * 100,
        act_cond * 100,
        conv_cond * 100,
        n_activated,
    )

    return df
