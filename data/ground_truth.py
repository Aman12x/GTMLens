"""
Ground truth effect sizes for GTMLens evaluation.

These values are the single source of truth for the experiment.
They mirror _TRUE_ATE in synthetic.py exactly — any change here
MUST be mirrored there.

Used exclusively for:
  - Evaluating how close CATE estimates are to truth (README accuracy table)
  - Test assertions in tests/test_causal.py

Do NOT import this file from production code paths.
"""

from typing import TypedDict


class SegmentEffect(TypedDict):
    company_size: str
    channel: str
    true_ate: float          # absolute lift on activation rate
    significance: str        # "significant" | "not_significant" (at alpha=0.05, N=50k)
    sample_share: float      # approx fraction of total population in this cell
    interview_note: str      # talking point for product DS interviews


# ---------------------------------------------------------------------------
# Ground truth table
# ---------------------------------------------------------------------------
# Activation rate lifts are on the treated group in steady-state.
# At N=50k (50/50 split), MDE at 80% power ≈ ±2pp for activation — so
# effects below ~2pp are "not significant" at the full-population level,
# though they may survive segment-level analysis with BH correction.

GROUND_TRUTH: list[SegmentEffect] = [
    # --- Enterprise ---
    {
        "company_size": "enterprise",
        "channel": "paid_search",
        "true_ate": 0.18,
        "significance": "significant",
        "sample_share": 0.070,  # 20% enterprise × 35% paid_search
        "interview_note": (
            "Largest effect. Enterprise buyers on paid search are high-intent — "
            "new onboarding flow removes friction at the key decision point. "
            "Classic 'right message, right moment' dynamic."
        ),
    },
    {
        "company_size": "enterprise",
        "channel": "organic",
        "true_ate": 0.14,
        "significance": "significant",
        "sample_share": 0.040,
        "interview_note": (
            "Strong effect for organic enterprise — these users self-discovered "
            "the product, so a smoother onboarding converts latent intent to activation."
        ),
    },
    {
        "company_size": "enterprise",
        "channel": "referral",
        "true_ate": 0.12,
        "significance": "significant",
        "sample_share": 0.040,
        "interview_note": (
            "Referral enterprise has high baseline; treatment still adds 12pp. "
            "Social proof + improved onboarding is a compounding advantage."
        ),
    },
    {
        "company_size": "enterprise",
        "channel": "email",
        "true_ate": 0.10,
        "significance": "significant",
        "sample_share": 0.030,
        "interview_note": (
            "Email-acquired enterprise users respond well — email sets expectations "
            "that the new onboarding flow meets more precisely."
        ),
    },
    {
        "company_size": "enterprise",
        "channel": "social",
        "true_ate": 0.06,
        "significance": "significant",
        "sample_share": 0.020,
        "interview_note": (
            "Smaller effect on social enterprise — social-acquired users are "
            "more exploratory; activation depends more on product depth than UX."
        ),
    },
    # --- Mid-market ---
    {
        "company_size": "mid_market",
        "channel": "paid_search",
        "true_ate": 0.11,
        "significance": "significant",
        "sample_share": 0.105,  # 35% mid_market × 30% paid_search
        "interview_note": (
            "Second-largest segment by volume. 11pp lift is commercially significant — "
            "drives most of the aggregate ATE you'd see in a naive ITT analysis."
        ),
    },
    {
        "company_size": "mid_market",
        "channel": "organic",
        "true_ate": 0.09,
        "significance": "significant",
        "sample_share": 0.105,
        "interview_note": (
            "Mid-market organic is the modal GTM segment. 9pp lift is the 'headline' "
            "number you'd cite when presenting to a PM — material, defensible."
        ),
    },
    {
        "company_size": "mid_market",
        "channel": "referral",
        "true_ate": 0.08,
        "significance": "significant",
        "sample_share": 0.053,
        "interview_note": "Referral mid-market has high baseline; effect is real but smaller in absolute terms.",
    },
    {
        "company_size": "mid_market",
        "channel": "email",
        "true_ate": 0.07,
        "significance": "significant",
        "sample_share": 0.035,
        "interview_note": "Consistent with email-nurture patterns — high intent, moderate treatment sensitivity.",
    },
    {
        "company_size": "mid_market",
        "channel": "social",
        "true_ate": 0.04,
        "significance": "significant",
        "sample_share": 0.053,
        "interview_note": (
            "4pp is above MDE but borderline. At segment level after BH correction "
            "this may or may not survive — good teaching example for multiple testing."
        ),
    },
    # --- SMB ---
    {
        "company_size": "SMB",
        "channel": "paid_search",
        "true_ate": 0.04,
        "significance": "not_significant",
        "sample_share": 0.113,  # 45% SMB × 25% paid_search
        "interview_note": (
            "Near-zero effect. SMB users churn or activate based on product "
            "value, not onboarding polish. Targeting them costs budget with minimal lift."
        ),
    },
    {
        "company_size": "SMB",
        "channel": "organic",
        "true_ate": 0.03,
        "significance": "not_significant",
        "sample_share": 0.180,  # largest single cell
        "interview_note": (
            "Largest population cell; near-zero effect. This is why naive ATE "
            "dilutes the signal — SMB organic drowns out the enterprise effect."
        ),
    },
    {
        "company_size": "SMB",
        "channel": "referral",
        "true_ate": 0.03,
        "significance": "not_significant",
        "sample_share": 0.045,
        "interview_note": "Marginal. Referral SMB already has high baseline; ceiling effect limits treatment lift.",
    },
    {
        "company_size": "SMB",
        "channel": "email",
        "true_ate": 0.02,
        "significance": "not_significant",
        "sample_share": 0.023,
        "interview_note": "Negligible. Email SMB users are often low-qualification leads.",
    },
    {
        "company_size": "SMB",
        "channel": "social",
        "true_ate": 0.01,
        "significance": "not_significant",
        "sample_share": 0.090,
        "interview_note": (
            "Effectively zero. Social SMB is the highest-volume, lowest-quality segment. "
            "Do not generate outreach — threshold filter should eliminate this cell."
        ),
    },
]


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_true_ate(company_size: str, channel: str) -> float:
    """
    Look up ground-truth ATE for a (company_size, channel) cell.

    Args:
        company_size: One of 'SMB', 'mid_market', 'enterprise'.
        channel: One of 'organic', 'paid_search', 'social', 'referral', 'email'.

    Returns:
        True ATE as a float (activation rate lift).

    Raises:
        KeyError: If the (company_size, channel) combination is not in ground truth.
    """
    for entry in GROUND_TRUTH:
        if entry["company_size"] == company_size and entry["channel"] == channel:
            return entry["true_ate"]
    raise KeyError(f"No ground truth for ({company_size}, {channel})")


def significant_segments() -> list[SegmentEffect]:
    """Return only the segments with statistically significant treatment effects."""
    return [e for e in GROUND_TRUTH if e["significance"] == "significant"]


def segment_summary() -> dict[str, dict]:
    """
    Return a nested dict keyed by (company_size, channel) for fast lookup.

    Returns:
        Dict mapping (company_size, channel) → SegmentEffect dict.
    """
    return {(e["company_size"], e["channel"]): e for e in GROUND_TRUTH}


# ---------------------------------------------------------------------------
# High-level aggregate ATEs (for README / interview)
# ---------------------------------------------------------------------------

# Weighted by sample_share — matches what a naive ITT would return
AGGREGATE_ATE_WEIGHTED = sum(
    e["true_ate"] * e["sample_share"] for e in GROUND_TRUTH
)

# Enterprise aggregate (all channels, equal-weighted)
ENTERPRISE_ATE = sum(
    e["true_ate"] for e in GROUND_TRUTH if e["company_size"] == "enterprise"
) / len([e for e in GROUND_TRUTH if e["company_size"] == "enterprise"])

# Mid-market aggregate
MID_MARKET_ATE = sum(
    e["true_ate"] for e in GROUND_TRUTH if e["company_size"] == "mid_market"
) / len([e for e in GROUND_TRUTH if e["company_size"] == "mid_market"])

# SMB aggregate (should be near zero)
SMB_ATE = sum(
    e["true_ate"] for e in GROUND_TRUTH if e["company_size"] == "SMB"
) / len([e for e in GROUND_TRUTH if e["company_size"] == "SMB"])
