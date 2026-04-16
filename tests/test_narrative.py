"""
Tests for core/narrative.py.

Required coverage (CLAUDE.md):
    - Output contains SHIP / ITERATE / ABORT recommendation
    - Guardrail section is always present in the output
    - Graceful fallback on Claude API error
    - Missing required keys in experiment_result raise ValueError
    - JSON parse failure returns fallback (not an exception)
    - Claude API is mocked — no real calls in tests
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.narrative import (
    NarrativeError,
    _parse_response,
    generate_result_narrative,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_experiment_result() -> dict:
    """Realistic output from cuped_adjustment() for a significant experiment."""
    return {
        "ate":                    0.172,
        "ate_se":                 0.018,
        "p_value":                0.0003,
        "ci_lower":               0.137,
        "ci_upper":               0.207,
        "variance_reduction_pct": 23.4,
        "n_treatment":            4200,
        "n_control":              4180,
        "guardrail_results": {
            "unsubscribe_rate":    "held (+0.001, p=0.61)",
            "spam_complaint_rate": "held (+0.000, p=0.88)",
        },
        "segment_breakdown": [
            {"segment": "enterprise_paid_search", "ate": 0.18, "p_value": 0.001},
            {"segment": "mid_market_organic",     "ate": 0.09, "p_value": 0.031},
            {"segment": "smb_organic",            "ate": 0.02, "p_value": 0.54},
        ],
    }


@pytest.fixture()
def metric_hierarchy() -> dict:
    return {
        "nsm":               "weekly_active_accounts",
        "primary_metric":    "activation_rate",
        "secondary_metrics": ["feature_adoption_rate", "time_to_value"],
        "guardrail_metrics": ["unsubscribe_rate", "spam_complaint_rate"],
    }


def _make_narrative_response(
    recommendation: str = "SHIP",
    outcome: str = "Activation rate increased 17.2pp for enterprise accounts.",
    driver: str = "Effect concentrated in enterprise paid_search segment.",
    guardrails: dict | None = None,
    rationale: str = "Effect is large, significant, and guardrails held flat.",
) -> str:
    """Build a valid Claude narrative JSON response."""
    return json.dumps({
        "outcome":        outcome,
        "driver":         driver,
        "guardrails":     guardrails or {"unsubscribe_rate": "held", "spam_complaint_rate": "held"},
        "recommendation": recommendation,
        "rationale":      rationale,
    })


def _make_mock_client(response_text: str):
    mock_usage = MagicMock()
    mock_usage.input_tokens  = 350
    mock_usage.output_tokens = 120

    mock_content = MagicMock()
    mock_content.text = response_text

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage   = mock_usage

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Recommendation is always SHIP / ITERATE / ABORT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rec", ["SHIP", "ITERATE", "ABORT"])
def test_valid_recommendations_pass(
    rec: str,
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    response = _make_narrative_response(recommendation=rec)
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result,
            metric_hierarchy,
            log_to_db=False,
        )
    assert result["recommendation"] == rec


def test_recommendation_is_uppercase_in_output(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    response = _make_narrative_response(recommendation="SHIP")
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert result["recommendation"] in {"SHIP", "ITERATE", "ABORT"}
    assert result["recommendation"] == result["recommendation"].upper()


def test_invalid_recommendation_from_claude_returns_fallback(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    """If Claude returns a recommendation not in {SHIP, ITERATE, ABORT}, use fallback."""
    bad_response = json.dumps({
        "outcome": "x", "driver": "y",
        "guardrails": {}, "recommendation": "WAIT", "rationale": "z",
    })
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(bad_response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert result["_error"] is True


# ---------------------------------------------------------------------------
# Guardrail section always present
# ---------------------------------------------------------------------------


def test_guardrail_section_present_in_output(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    response = _make_narrative_response(
        guardrails={"unsubscribe_rate": "held", "spam_complaint_rate": "held"}
    )
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert "guardrails" in result
    assert isinstance(result["guardrails"], dict)


def test_guardrail_section_present_even_when_empty(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    """Even when Claude returns empty guardrails dict, key must be present."""
    response = _make_narrative_response(guardrails={})
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert "guardrails" in result


def test_fallback_has_guardrail_section() -> None:
    """Fallback response (on error) must also contain guardrails key."""
    from core.narrative import _FALLBACK_NARRATIVE
    assert "guardrails" in _FALLBACK_NARRATIVE


# ---------------------------------------------------------------------------
# Graceful fallback on API error
# ---------------------------------------------------------------------------


def test_api_error_returns_fallback(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    import anthropic as _anthropic
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _anthropic.APIError(
        message="Rate limit exceeded",
        request=MagicMock(),
        body=None,
    )
    with patch("core.narrative.anthropic.Anthropic", return_value=mock_client):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert result["_error"] is True
    assert "guardrails" in result
    assert result["recommendation"] in {"SHIP", "ITERATE", "ABORT"}


def test_malformed_json_returns_fallback(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    """Non-JSON Claude response must return fallback, not raise."""
    with patch("core.narrative.anthropic.Anthropic",
               return_value=_make_mock_client("Here is the summary: blah blah")):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert result["_error"] is True


def test_fallback_is_not_an_exception(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    """generate_result_narrative must never raise — always return a dict."""
    import anthropic as _anthropic
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _anthropic.APIError(
        message="Internal server error",
        request=MagicMock(),
        body=None,
    )
    with patch("core.narrative.anthropic.Anthropic", return_value=mock_client):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_missing_ate_raises_value_error(metric_hierarchy: dict) -> None:
    with pytest.raises(ValueError, match="ate"):
        generate_result_narrative(
            {"p_value": 0.05},  # missing ate
            metric_hierarchy,
            log_to_db=False,
        )


def test_missing_p_value_raises_value_error(metric_hierarchy: dict) -> None:
    with pytest.raises(ValueError, match="p_value"):
        generate_result_narrative(
            {"ate": 0.05},  # missing p_value
            metric_hierarchy,
            log_to_db=False,
        )


# ---------------------------------------------------------------------------
# _parse_response unit tests
# ---------------------------------------------------------------------------


def test_parse_response_valid() -> None:
    text = _make_narrative_response()
    result = _parse_response(text)
    assert result["recommendation"] in {"SHIP", "ITERATE", "ABORT"}
    assert "outcome" in result
    assert "driver" in result
    assert "guardrails" in result
    assert "rationale" in result


def test_parse_response_invalid_json_raises() -> None:
    with pytest.raises(NarrativeError, match="non-JSON"):
        _parse_response("not valid json {{")


def test_parse_response_missing_key_raises() -> None:
    incomplete = json.dumps({
        "outcome": "x", "driver": "y", "guardrails": {},
        # missing recommendation and rationale
    })
    with pytest.raises(NarrativeError, match="missing keys"):
        _parse_response(incomplete)


def test_parse_response_invalid_recommendation_raises() -> None:
    bad = json.dumps({
        "outcome": "x", "driver": "y", "guardrails": {},
        "recommendation": "DELAY", "rationale": "z",
    })
    with pytest.raises(NarrativeError, match="SHIP.*ITERATE.*ABORT"):
        _parse_response(bad)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_contains_all_expected_keys(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    response = _make_narrative_response()
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    expected = {"outcome", "driver", "guardrails", "recommendation", "rationale"}
    assert expected.issubset(result.keys())


def test_output_strings_are_non_empty(
    valid_experiment_result: dict,
    metric_hierarchy: dict,
) -> None:
    response = _make_narrative_response()
    with patch("core.narrative.anthropic.Anthropic", return_value=_make_mock_client(response)):
        result = generate_result_narrative(
            valid_experiment_result, metric_hierarchy, log_to_db=False
        )
    assert len(result["outcome"]) > 0
    assert len(result["driver"]) > 0
    assert len(result["rationale"]) > 0
