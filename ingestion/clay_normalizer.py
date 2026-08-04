"""
Value normalization for Clay-shaped scoring payloads.

Clay enrichment columns deliver raw vendor values — an employee count as an
integer, a source label like "Google Ads" or "LinkedIn" — while the scoring
model was trained on the GTMLens internal vocabulary:

    company_size — SMB | mid_market | enterprise
    channel      — organic | paid_search | social | referral | email

These helpers map raw values onto that vocabulary. Unrecognised strings are
passed through in normalized form (lowercase, underscores) rather than
silently coerced: core/model_store.py encodes unseen levels as the reference
level and logs a warning, which is more honest than guessing.

Bucket boundaries match data/synthetic.py:
    SMB 1–50 employees · mid_market 51–500 · enterprise 500+
"""

import logging
import re

logger = logging.getLogger(__name__)

_SMB_MAX = 50
_MID_MARKET_MAX = 500

_SIZE_LABELS: dict[str, str] = {
    "smb": "SMB",
    "small": "SMB",
    "small_business": "SMB",
    "startup": "SMB",
    "mid_market": "mid_market",
    "midmarket": "mid_market",
    "mid": "mid_market",
    "medium": "mid_market",
    "enterprise": "enterprise",
    "large": "enterprise",
}

# Common Clay / UTM / CRM source labels → internal channel vocabulary
_CHANNEL_LABELS: dict[str, str] = {
    # paid search
    "paid_search": "paid_search", "google_ads": "paid_search", "adwords": "paid_search",
    "ppc": "paid_search", "sem": "paid_search", "paid": "paid_search",
    "bing_ads": "paid_search", "cpc": "paid_search",
    # social
    "social": "social", "linkedin": "social", "facebook": "social",
    "twitter": "social", "x": "social", "instagram": "social",
    "paid_social": "social", "organic_social": "social",
    # referral
    "referral": "referral", "partner": "referral", "affiliate": "referral",
    "word_of_mouth": "referral", "community": "referral",
    # email
    "email": "email", "newsletter": "email", "outbound": "email",
    "email_marketing": "email", "cold_email": "email",
    # organic
    "organic": "organic", "organic_search": "organic", "seo": "organic",
    "direct": "organic", "google": "organic", "website": "organic",
}


def _slug(value: str) -> str:
    """Lowercase and collapse separators to underscores: "Google Ads" → "google_ads"."""
    return re.sub(r"[\s/-]+", "_", value.strip().lower())


def normalize_company_size(value: str | int | float | None, employee_count: int | None = None) -> str:
    """
    Map a raw company-size value onto SMB | mid_market | enterprise.

    Accepts, in order of precedence:
        1. employee_count int (typical Clay enrichment field)
        2. numeric value ("250", 250) — treated as an employee count
        3. range strings ("51-200", "1,001-5,000") — bucketed by lower bound
        4. label strings ("Enterprise", "small business")

    Args:
        value:          Raw company-size field from the caller.
        employee_count: Explicit employee count, wins over value if provided.

    Returns:
        Internal bucket name, or "unknown" when nothing is interpretable.
    """
    if employee_count is not None:
        return _bucket_from_count(employee_count)

    if value is None:
        return "unknown"
    if isinstance(value, (int, float)):
        return _bucket_from_count(int(value))

    raw = value.strip()
    if not raw:
        return "unknown"

    digits = re.findall(r"\d[\d,]*", raw)
    if digits:
        # Range or bare number: bucket by the lower bound ("51-200" → 51)
        return _bucket_from_count(int(digits[0].replace(",", "")))

    label = _SIZE_LABELS.get(_slug(raw))
    if label:
        return label

    logger.info("normalize_company_size: unrecognised value '%s'", raw)
    return _slug(raw)


def _bucket_from_count(count: int) -> str:
    """Bucket an employee count using the synthetic.py boundaries."""
    if count <= 0:
        return "unknown"
    if count <= _SMB_MAX:
        return "SMB"
    if count <= _MID_MARKET_MAX:
        return "mid_market"
    return "enterprise"


def normalize_channel(value: str | None) -> str:
    """
    Map a raw source/channel label onto the internal channel vocabulary.

    Args:
        value: Raw channel string ("Google Ads", "LinkedIn", "utm: cpc", …).

    Returns:
        organic | paid_search | social | referral | email when recognised;
        the slugified raw value otherwise (encoder treats it as unseen);
        "organic" for empty/None (the modal acquisition channel).
    """
    if value is None or not value.strip():
        return "organic"

    slug = _slug(value)
    mapped = _CHANNEL_LABELS.get(slug)
    if mapped:
        return mapped

    # Substring pass catches compound labels like "google_ads_brand"
    for needle, channel in _CHANNEL_LABELS.items():
        if needle in slug:
            return channel

    logger.info("normalize_channel: unrecognised value '%s'", value)
    return slug


def normalize_industry(value: str | None) -> str:
    """
    Normalize an industry label (light touch — pass through trimmed value).

    Industry vocabularies vary per tenant; the encoder handles unseen levels,
    so no fixed mapping is imposed here.

    Args:
        value: Raw industry string.

    Returns:
        Trimmed industry string, or "unknown" for empty/None.
    """
    if value is None or not value.strip():
        return "unknown"
    return value.strip()
