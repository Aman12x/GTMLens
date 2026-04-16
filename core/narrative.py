"""
Result narrative generation via Claude API.

Produces a structured, PM-ready experiment summary that follows the
NSM → primary metric → guardrail → recommendation hierarchy.

Key rules (CLAUDE.md):
    - Output must contain SHIP / ITERATE / ABORT recommendation
    - A guardrail section must always be present
    - Graceful fallback on Claude API error — never crash the app
    - Token usage logged to SQLite api_usage table
    - Raw API errors are never exposed to the UI
"""

import json
import logging
import os
from typing import Literal

import anthropic

from core.outreach import _log_api_usage  # shared SQLite logging helper

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-5"
_MAX_TOKENS = 512

_SYSTEM_PROMPT = """\
You are a data scientist presenting A/B experiment results to a product manager.

Structure your response EXACTLY as follows (do not add headers or markdown):

1. OUTCOME: What happened to the primary metric (one sentence, include magnitude and direction).
2. DRIVER: Why — which segment or channel drove the effect (one sentence).
3. GUARDRAILS: For each guardrail metric listed, state "held" or "breached" and the magnitude.
4. RECOMMENDATION: Exactly one word from {SHIP, ITERATE, ABORT}, followed by one sentence of rationale.

Rules:
- Do not editorialize or use hedging language ("might", "could", "potentially")
- Do not repeat the raw numbers from the input — synthesise them
- Max 150 words total
- Be direct. PMs read 50 of these a week.

Output ONLY valid JSON with this exact schema:
{
  "outcome": "<sentence>",
  "driver": "<sentence>",
  "guardrails": {"<metric_name>": "<held|breached: magnitude>"},
  "recommendation": "<SHIP|ITERATE|ABORT>",
  "rationale": "<one sentence>"
}"""

_FALLBACK_NARRATIVE = {
    "outcome":        "Narrative generation is temporarily unavailable.",
    "driver":         "Please retry or check the API key configuration.",
    "guardrails":     {},
    "recommendation": "ITERATE",
    "rationale":      "Cannot determine recommendation without a valid narrative.",
    "_error":         True,
}


class NarrativeError(ValueError):
    """Raised when a narrative cannot be generated from the provided inputs."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_user_prompt(
    experiment_result: dict,
    metric_hierarchy: dict,
    recommendation: str,
) -> str:
    """
    Format experiment result data into the Claude user turn.

    Args:
        experiment_result: Output of cuped_adjustment() or similar — must
                           contain 'ate', 'p_value', 'ci_lower', 'ci_upper'.
                           May contain 'segment_breakdown' list.
        metric_hierarchy:  Dict describing the metric stack:
                               nsm, primary_metric, secondary_metrics, guardrail_metrics
        recommendation:    Caller-provided recommendation hint ("ship"/"iterate"/"abort").
                           Claude may override this based on the evidence.

    Returns:
        Formatted prompt string.
    """
    ate        = experiment_result.get("ate", 0)
    p_value    = experiment_result.get("p_value", 1.0)
    ci_lower   = experiment_result.get("ci_lower", 0)
    ci_upper   = experiment_result.get("ci_upper", 0)
    n_t        = experiment_result.get("n_treatment", "?")
    n_c        = experiment_result.get("n_control", "?")
    var_red    = experiment_result.get("variance_reduction_pct", 0)

    segments   = experiment_result.get("segment_breakdown", [])
    guardrails = metric_hierarchy.get("guardrail_metrics", [])
    guardrail_results = experiment_result.get("guardrail_results", {})

    seg_text = ""
    if segments:
        top = sorted(segments, key=lambda s: abs(s.get("ate", 0)), reverse=True)[:3]
        lines = [
            f"  - {s.get('segment', '?')}: ATE={s.get('ate', 0):.3f} p={s.get('p_value', 1):.3f}"
            for s in top
        ]
        seg_text = "Top segments by effect:\n" + "\n".join(lines)

    guardrail_text = ""
    if guardrail_results:
        lines = [f"  - {k}: {v}" for k, v in guardrail_results.items()]
        guardrail_text = "Guardrail metric results:\n" + "\n".join(lines)
    elif guardrails:
        guardrail_text = "Guardrail metrics to assess: " + ", ".join(guardrails)

    return f"""\
Experiment results:
- Primary metric (ATE): {ate:+.4f} ({ate * 100:+.2f}pp)
- 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
- p-value: {p_value:.4f}
- N treatment: {n_t} | N control: {n_c}
- CUPED variance reduction: {var_red:.1f}%

Metric hierarchy:
- North Star Metric (NSM): {metric_hierarchy.get("nsm", "not specified")}
- Primary metric: {metric_hierarchy.get("primary_metric", "activation_rate")}
- Secondary metrics: {", ".join(metric_hierarchy.get("secondary_metrics", []))}
- Guardrail metrics: {", ".join(guardrails)}

{seg_text}
{guardrail_text}

Caller's recommended action: {recommendation.upper()}

Generate the structured narrative now."""


def _call_claude(user_prompt: str) -> tuple[str, anthropic.types.Usage | None]:
    """
    Call Claude API for narrative generation.

    Returns:
        Tuple of (response_text, usage). On API error returns a fallback
        JSON string and None — never raises.
    """
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text, response.usage
    except anthropic.APIError as exc:
        logger.error("Claude API error in narrative: %s", exc)
        return None, None


def _parse_response(text: str) -> dict:
    """
    Parse Claude's JSON narrative response.

    Args:
        text: Raw response text from Claude.

    Returns:
        Dict with keys: outcome, driver, guardrails, recommendation, rationale.

    Raises:
        NarrativeError: If JSON is invalid or required keys are missing.
    """
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise NarrativeError(
            f"Claude returned non-JSON narrative: {text[:120]!r}"
        ) from exc

    required = {"outcome", "driver", "guardrails", "recommendation", "rationale"}
    missing = required - data.keys()
    if missing:
        raise NarrativeError(f"Narrative response missing keys: {missing}")

    rec = str(data["recommendation"]).strip().upper()
    if rec not in {"SHIP", "ITERATE", "ABORT"}:
        raise NarrativeError(
            f"recommendation must be SHIP, ITERATE, or ABORT. Got: {rec!r}"
        )

    return {
        "outcome":        str(data["outcome"]).strip(),
        "driver":         str(data["driver"]).strip(),
        "guardrails":     dict(data["guardrails"]),
        "recommendation": rec,
        "rationale":      str(data["rationale"]).strip(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_result_narrative(
    experiment_result: dict,
    metric_hierarchy: dict,
    recommendation: Literal["ship", "iterate", "abort"] = "iterate",
    log_to_db: bool = True,
) -> dict:
    """
    Generate a PM-ready experiment result narrative via Claude.

    Enforces the NSM → primary metric → guardrail → recommendation structure
    that the CLAUDE.md requires. If the Claude API fails, returns a graceful
    fallback that is clearly marked as an error so the UI can handle it.

    Args:
        experiment_result: Output dict from cuped_adjustment() (or equivalent).
                           Required keys: ate, p_value, ci_lower, ci_upper.
                           Optional: segment_breakdown, guardrail_results,
                           n_treatment, n_control, variance_reduction_pct.
        metric_hierarchy:  Dict describing the metric stack:
                               nsm              — North Star Metric name
                               primary_metric   — what the experiment measures
                               secondary_metrics — list of secondary metric names
                               guardrail_metrics — list of guardrail metric names
        recommendation:    Caller's suggested recommendation ("ship"/"iterate"/"abort").
                           Claude considers this but may override based on evidence.
        log_to_db:         Write token usage to api_usage table. Set False in tests.

    Returns:
        Dict with keys:
            outcome        — what happened to the primary metric (1 sentence)
            driver         — causal mechanism / segment breakdown (1 sentence)
            guardrails     — dict of metric_name → "held|breached: magnitude"
            recommendation — "SHIP" | "ITERATE" | "ABORT"
            rationale      — one-sentence reasoning for the recommendation
            _error         — True only if fallback was used (API or parse failure)

    Raises:
        ValueError: If experiment_result is missing required keys.
    """
    required = {"ate", "p_value"}
    missing = required - experiment_result.keys()
    if missing:
        raise ValueError(f"experiment_result is missing required keys: {missing}")

    user_prompt = _build_user_prompt(experiment_result, metric_hierarchy, recommendation)
    raw_text, usage = _call_claude(user_prompt)

    if raw_text is None:
        # API call failed — return graceful fallback, never crash
        return {**_FALLBACK_NARRATIVE}

    if usage is not None and log_to_db:
        _log_api_usage("narrative", _MODEL, usage)

    try:
        parsed = _parse_response(raw_text)
    except NarrativeError as exc:
        logger.error("Narrative parse failed: %s", exc)
        return {**_FALLBACK_NARRATIVE}

    logger.info(
        "generate_result_narrative | recommendation=%s | p_value=%.4f",
        parsed["recommendation"],
        experiment_result.get("p_value", 1.0),
    )

    return parsed
