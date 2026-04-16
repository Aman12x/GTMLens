"""
Outreach generation: segment-conditional message creation via Claude API.

Outreach messages are treated as treatment variants — each generated message
is an experiment arm whose causal lift is measured against a holdout group.

Key rules (CLAUDE.md):
    - Only generate outreach for segments where cate_estimate > threshold
      (default: CATE_UPLIFT_THRESHOLD env var, 0.40 = top 40% by uplift)
    - 20% holdout auto-assigned per segment (HOLDOUT_FRACTION env var)
    - All sends logged to SQLite (outreach_log table)
    - Claude API errors return a graceful fallback — never crash
    - Do not mock the Claude API in this module — mocking is for tests only

Holdout assignment:
    Deterministic — derived from hash(segment_id + user_id) so the same
    user always lands in the same bucket across experiment runs. This
    prevents contamination from re-randomisation.
"""

import hashlib
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Literal

import anthropic

# Pricing table keyed by model ID (input_$/1M, output_$/1M).
# Update this when switching models — a stale price silently mis-tracks cost.
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-5":   (15.0,  75.0),
    "claude-opus-4-6":   (15.0,  75.0),
    "claude-sonnet-4-6": (3.0,   15.0),
    "claude-haiku-4-5":  (0.25,  1.25),
}

_PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MODEL = "claude-opus-4-5"
_MAX_TOKENS = 1024

_SYSTEM_PROMPT = """\
You are a GTM strategist. Generate outreach for the following B2B SaaS segment.

Structure your message as:
1. [Persona insight] — one sentence showing you understand this buyer
2. [Specific pain point] — grounded in the segment's behavioral data provided
3. [Value proposition] — concrete, tied to the pain point
4. [Single CTA] — one clear next step, no alternatives

Hard constraints:
- Max 120 words in the body
- No generic openers ("I hope this finds you well", "Hope you're doing well", etc.)
- Do not fabricate metrics or percentages not provided in the segment data
- Do not use the word "leverage"
- No em-dashes (— or –) anywhere in the output

Output ONLY valid JSON with this exact schema:
{"subject": "<email subject line>", "body": "<email body>", "cta": "<call to action text>"}

No markdown, no explanation, no preamble — only the JSON object."""

# Fallback returned when Claude API call fails — safe to show in UI
_FALLBACK_RESPONSE = {
    "subject": "[Outreach unavailable — please retry]",
    "body": "Message generation is temporarily unavailable. Please try again.",
    "cta": "Contact us",
    "predicted_uplift_group": "unknown",
    "holdout_flag": False,
    "_error": True,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutreachError(ValueError):
    """
    Raised when outreach cannot be generated for a segment.

    This is a correctness error (segment below threshold, missing data)
    not a transient API error. Callers should surface this to the UI.
    """


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _cate_threshold() -> float:
    return float(os.getenv("CATE_UPLIFT_THRESHOLD", "0.40"))


def _holdout_fraction() -> float:
    return float(os.getenv("HOLDOUT_FRACTION", "0.20"))


def _sqlite_path() -> str:
    return os.getenv(
        "SQLITE_PATH",
        str(_PROJECT_ROOT / "data" / "logs.db"),
    )


# ---------------------------------------------------------------------------
# Holdout assignment (deterministic)
# ---------------------------------------------------------------------------


def _is_holdout(segment_id: str, user_id: str = "", holdout_fraction: float | None = None) -> bool:
    """
    Deterministically assign a user to holdout using a hash of segment + user.

    The hash ensures the same (segment_id, user_id) pair always gets the same
    assignment, preventing contamination from re-randomisation across runs.

    Args:
        segment_id:       Identifies the outreach segment.
        user_id:          Identifies the individual user. Empty string for
                          segment-level (non-user) calls — assigns ~holdout_fraction
                          of segment-level calls to holdout.
        holdout_fraction: Fraction to hold out. Reads HOLDOUT_FRACTION env if None.

    Returns:
        True if this (segment, user) combination is in the holdout group.
    """
    frac = holdout_fraction if holdout_fraction is not None else _holdout_fraction()
    key = f"{segment_id}:{user_id}"
    digest = int(hashlib.md5(key.encode(), usedforsecurity=False).hexdigest(), 16)
    return (digest % 100) < int(frac * 100)


# ---------------------------------------------------------------------------
# SQLite logging
# ---------------------------------------------------------------------------


@contextmanager
def _db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(_sqlite_path())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _log_send(
    segment_id: str,
    company_size: str,
    industry: str,
    channel: str,
    cate_estimate: float,
    tone: str,
    subject: str,
    body: str,
    is_holdout: bool,
    tenant_id: str = "demo",
) -> None:
    """
    Record an outreach send (or holdout) to the outreach_log SQLite table.

    Args are the same fields as the outreach_log schema in seed_db.py.
    Never raises — log failures are warnings, not errors.
    """
    subject_hash = hashlib.sha256(subject.encode()).hexdigest()[:16]
    body_hash    = hashlib.sha256(body.encode()).hexdigest()[:16]
    try:
        with _db() as conn:
            conn.execute("""
                INSERT INTO outreach_log
                    (tenant_id, sent_at, segment_id, company_size, industry, channel,
                     cate_estimate, tone, subject_hash, body_hash, is_holdout)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tenant_id,
                datetime.now(tz=timezone.utc).isoformat(),
                segment_id, company_size, industry, channel,
                cate_estimate, tone, subject_hash, body_hash,
                int(is_holdout),
            ))
    except Exception as exc:
        logger.warning("Failed to log outreach send: %s", exc)


def _log_api_usage(
    endpoint: str,
    model: str,
    usage: anthropic.types.Usage,
) -> None:
    """Record Claude API token usage to api_usage SQLite table for cost tracking."""
    input_rate, output_rate = _MODEL_PRICING.get(model, (15.0, 75.0))
    cost_usd = (usage.input_tokens * input_rate + usage.output_tokens * output_rate) / 1_000_000
    try:
        with _db() as conn:
            conn.execute("""
                INSERT INTO api_usage
                    (called_at, endpoint, model, input_tokens, output_tokens, cost_usd)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(tz=timezone.utc).isoformat(),
                endpoint, model,
                usage.input_tokens, usage.output_tokens, cost_usd,
            ))
    except Exception as exc:
        logger.warning("Failed to log API usage: %s", exc)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------


def _call_claude(user_prompt: str) -> tuple[str, anthropic.types.Usage | None]:
    """
    Call the Claude API with the outreach generation prompt.

    Args:
        user_prompt: Formatted segment context for the user turn.

    Returns:
        Tuple of (response_text, usage). On API error, returns a JSON
        fallback string and None usage.

    Never raises — errors are caught and logged so the caller can return
    a graceful fallback response to the UI.
    """
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text
        logger.debug("Claude response: %s", text[:200])
        return text, response.usage
    except anthropic.APIError as exc:
        logger.error("Claude API error in outreach: %s", exc)
        return None, None


def _parse_response(text: str) -> dict:
    """
    Parse Claude's JSON response into the outreach dict.

    Args:
        text: Raw text returned by Claude.

    Returns:
        Dict with keys: subject, body, cta.

    Raises:
        OutreachError: If the response cannot be parsed as valid JSON or
                       is missing required keys.
    """
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise OutreachError(
            f"Claude returned non-JSON response: {text[:120]!r}"
        ) from exc

    required = {"subject", "body", "cta"}
    missing = required - data.keys()
    if missing:
        raise OutreachError(
            f"Claude response missing keys {missing}. Got: {list(data.keys())}"
        )

    return {k: str(data[k]).strip() for k in required}


def _build_user_prompt(
    segment: dict,
    product_context: str,
    tone: str,
) -> str:
    """
    Format segment data into the user turn of the Claude conversation.

    Args:
        segment:         Segment dict with company_size, industry, channel,
                         funnel_stage, cate_estimate keys.
        product_context: Short description of what the product does.
        tone:            One of "warm", "direct", "technical".

    Returns:
        Formatted prompt string.
    """
    return f"""\
Product context: {product_context}

Target segment:
- Company size: {segment.get("company_size", "unknown")}
- Industry: {segment.get("industry", "unknown")}
- Acquisition channel: {segment.get("channel", "unknown")}
- Funnel stage dropped off: {segment.get("funnel_stage", "activation")}
- Predicted activation lift (CATE): {segment.get("cate_estimate", 0):.1%}

Tone: {tone}

Generate the outreach message now."""


# ---------------------------------------------------------------------------
# Uplift group classification
# ---------------------------------------------------------------------------


def _classify_uplift_group(cate_estimate: float, threshold: float) -> str:
    """
    Classify a segment into an uplift group based on its CATE estimate.

    Args:
        cate_estimate: Predicted treatment effect for this segment.
        threshold:     The CATE_UPLIFT_THRESHOLD value in use.

    Returns:
        "high" if clearly above threshold, "marginal" if within 2pp, else "low".
    """
    if cate_estimate >= threshold + 0.02:
        return "high"
    if cate_estimate >= threshold:
        return "marginal"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_outreach(
    segment: dict,
    product_context: str,
    tone: Literal["warm", "direct", "technical"] = "direct",
    cate_threshold: float | None = None,
    holdout_fraction: float | None = None,
    user_id: str = "",
    log_to_db: bool = True,
    tenant_id: str = "demo",
) -> dict:
    """
    Generate a personalised outreach message for a high-uplift segment.

    Only segments with cate_estimate above cate_threshold receive messages.
    This is a correctness requirement: generating outreach for low-uplift
    segments wastes budget and dilutes the measured lift signal.

    Holdout assignment is deterministic via hash(segment_id, user_id).
    Holdout users are logged but the message is flagged — callers must NOT
    send to holdout users, they are the control group for lift measurement.

    Args:
        segment:          Dict with keys:
                              company_size, industry, channel,
                              funnel_stage, cate_estimate,
                              segment_id (optional, defaults to company_size+channel)
        product_context:  Short product description included in the prompt.
        tone:             Message tone — "warm", "direct", or "technical".
        cate_threshold:   Minimum CATE to qualify for outreach. Reads
                          CATE_UPLIFT_THRESHOLD env if None.
        holdout_fraction: Fraction to hold out. Reads HOLDOUT_FRACTION env if None.
        user_id:          User identifier for deterministic holdout assignment.
        log_to_db:        Whether to write to the outreach_log SQLite table.
                          Set False in tests.

    Returns:
        Dict with keys:
            subject               — email subject line
            body                  — email body (≤120 words)
            cta                   — call-to-action text
            predicted_uplift_group — "high" | "marginal" | "low"
            holdout_flag          — True if this user should NOT receive the message
            segment_id            — identifier used for holdout hash and logging

    Raises:
        OutreachError: If segment is below CATE threshold (correctness guard).
        ValueError:    If required segment keys are missing.
    """
    required_keys = {"cate_estimate"}
    missing = required_keys - segment.keys()
    if missing:
        raise ValueError(f"segment dict is missing required keys: {missing}")

    threshold = cate_threshold if cate_threshold is not None else _cate_threshold()
    cate_est = float(segment["cate_estimate"])

    if cate_est < threshold:
        raise OutreachError(
            f"Segment CATE estimate {cate_est:.3f} is below threshold {threshold:.3f}. "
            f"Do not generate outreach for low-uplift segments — "
            f"this wastes budget and dilutes lift measurement."
        )

    segment_id = segment.get(
        "segment_id",
        f"{segment.get('company_size', 'unknown')}_{segment.get('channel', 'unknown')}",
    )

    holdout = _is_holdout(segment_id, user_id, holdout_fraction)
    uplift_group = _classify_uplift_group(cate_est, threshold)

    user_prompt = _build_user_prompt(segment, product_context, tone)
    raw_text, usage = _call_claude(user_prompt)

    if raw_text is None:
        # API call failed — return graceful fallback, never crash
        return {**_FALLBACK_RESPONSE, "segment_id": segment_id, "holdout_flag": holdout}

    try:
        parsed = _parse_response(raw_text)
    except OutreachError:
        logger.error("Failed to parse Claude response for segment %s", segment_id)
        return {**_FALLBACK_RESPONSE, "segment_id": segment_id, "holdout_flag": holdout}

    if usage is not None:
        _log_api_usage("outreach", _MODEL, usage)

    if log_to_db:
        _log_send(
            segment_id=segment_id,
            company_size=segment.get("company_size", ""),
            industry=segment.get("industry", ""),
            channel=segment.get("channel", ""),
            cate_estimate=cate_est,
            tone=tone,
            subject=parsed["subject"],
            body=parsed["body"],
            is_holdout=holdout,
            tenant_id=tenant_id,
        )

    logger.info(
        "generate_outreach | segment=%s | uplift_group=%s | holdout=%s | tone=%s",
        segment_id, uplift_group, holdout, tone,
    )

    return {
        "subject":               parsed["subject"],
        "body":                  parsed["body"],
        "cta":                   parsed["cta"],
        "predicted_uplift_group": uplift_group,
        "holdout_flag":          holdout,
        "segment_id":            segment_id,
    }
