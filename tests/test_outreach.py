"""
Tests for core/outreach.py.

Required coverage (CLAUDE.md):
    - Only high-uplift segments get messages (threshold guard)
    - Holdout fraction is correct (~20% of segment calls)
    - JSON parse succeeds on valid Claude response
    - Graceful fallback returned on Claude API error
    - Holdout assignment is deterministic (same input → same result)
    - Input segment missing required keys raises ValueError
    - Claude API is mocked — no real calls in tests
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from core.outreach import (
    OutreachError,
    _is_holdout,
    _parse_response,
    generate_outreach,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def high_uplift_segment() -> dict:
    return {
        "company_size": "enterprise",
        "industry":     "SaaS",
        "channel":      "paid_search",
        "funnel_stage": "activation",
        "cate_estimate": 0.18,
        "segment_id":   "enterprise_paid_search",
    }


@pytest.fixture()
def low_uplift_segment() -> dict:
    return {
        "company_size": "SMB",
        "industry":     "Logistics",
        "channel":      "social",
        "funnel_stage": "activation",
        "cate_estimate": 0.01,
        "segment_id":   "smb_social",
    }


@pytest.fixture()
def mock_claude_response() -> str:
    """Valid JSON response as Claude would return it."""
    return json.dumps({
        "subject": "How enterprise teams cut activation time by 30%",
        "body": (
            "Enterprise SaaS buyers on paid search are evaluating fast — "
            "and dropping off at activation because setup complexity kills momentum. "
            "Our new onboarding flow removes the 3-step configuration wall. "
            "Teams on your plan now reach their first value moment in under 10 minutes."
        ),
        "cta": "See the new onboarding flow",
    })


def _make_mock_client(response_text: str):
    """Build a mock anthropic.Anthropic() client that returns response_text."""
    mock_usage = MagicMock()
    mock_usage.input_tokens  = 250
    mock_usage.output_tokens = 80

    mock_content = MagicMock()
    mock_content.text = response_text

    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_response.usage   = mock_usage

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    return mock_client


# ---------------------------------------------------------------------------
# Threshold guard — only high-uplift segments get messages
# ---------------------------------------------------------------------------


def test_below_threshold_raises_outreach_error(low_uplift_segment: dict) -> None:
    """Segment with CATE below threshold must raise OutreachError, not proceed."""
    with pytest.raises(OutreachError, match="below threshold"):
        generate_outreach(
            low_uplift_segment,
            product_context="GTM analytics platform",
            cate_threshold=0.10,
            log_to_db=False,
        )


def test_above_threshold_generates_message(
    high_uplift_segment: dict,
    mock_claude_response: str,
) -> None:
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            high_uplift_segment,
            product_context="GTM analytics platform",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert "subject" in result
    assert "body" in result
    assert "cta" in result


def test_exactly_at_threshold_generates_message(mock_claude_response: str) -> None:
    """cate_estimate == threshold should pass (>= semantics)."""
    segment = {
        "company_size": "mid_market",
        "industry": "FinTech",
        "channel": "organic",
        "funnel_stage": "activation",
        "cate_estimate": 0.10,
    }
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            segment,
            product_context="GTM analytics",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert result.get("_error") is not True


def test_missing_cate_estimate_raises() -> None:
    with pytest.raises(ValueError, match="cate_estimate"):
        generate_outreach(
            {"company_size": "SMB"},  # no cate_estimate
            product_context="test",
            log_to_db=False,
        )


# ---------------------------------------------------------------------------
# Holdout fraction correctness
# ---------------------------------------------------------------------------


def test_holdout_fraction_approximately_correct() -> None:
    """
    With 1000 distinct user_ids, ~20% should be assigned to holdout.
    Deterministic hash, so exact reproducibility expected at seed=0.
    """
    holdout_count = sum(
        _is_holdout("enterprise_paid_search", f"user_{i}", holdout_fraction=0.20)
        for i in range(1000)
    )
    # Expect 200 ± 30 (3σ tolerance for a deterministic hash)
    assert 170 <= holdout_count <= 230, (
        f"Expected ~200 holdout assignments, got {holdout_count}"
    )


def test_holdout_assignment_is_deterministic() -> None:
    """Same (segment_id, user_id) must always return the same holdout assignment."""
    result_a = _is_holdout("seg_x", "user_42")
    result_b = _is_holdout("seg_x", "user_42")
    assert result_a == result_b


def test_different_users_get_different_assignments() -> None:
    """Hash should distribute users — not all True or all False."""
    results = [_is_holdout("seg_y", f"u_{i}") for i in range(50)]
    assert any(results), "Expected some holdout assignments"
    assert not all(results), "Expected some non-holdout assignments"


def test_holdout_flag_in_output(
    high_uplift_segment: dict,
    mock_claude_response: str,
) -> None:
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert "holdout_flag" in result
    assert isinstance(result["holdout_flag"], bool)


# ---------------------------------------------------------------------------
# JSON parse correctness
# ---------------------------------------------------------------------------


def test_parse_response_valid_json() -> None:
    text = json.dumps({"subject": "s", "body": "b", "cta": "c"})
    result = _parse_response(text)
    assert result["subject"] == "s"
    assert result["body"] == "b"
    assert result["cta"] == "c"


def test_parse_response_missing_key_raises() -> None:
    text = json.dumps({"subject": "s", "body": "b"})  # missing cta
    with pytest.raises(OutreachError, match="missing keys"):
        _parse_response(text)


def test_parse_response_invalid_json_raises() -> None:
    with pytest.raises(OutreachError, match="non-JSON"):
        _parse_response("this is not json at all")


def test_parse_response_whitespace_stripped() -> None:
    text = json.dumps({"subject": "  hello  ", "body": "  world  ", "cta": "  go  "})
    result = _parse_response(text)
    assert result["subject"] == "hello"
    assert result["body"] == "world"
    assert result["cta"] == "go"


# ---------------------------------------------------------------------------
# Graceful fallback on API error
# ---------------------------------------------------------------------------


def test_api_error_returns_fallback(high_uplift_segment: dict) -> None:
    """Claude API errors must return a fallback dict, never crash."""
    import anthropic as _anthropic
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = _anthropic.APIError(
        message="Service unavailable",
        request=MagicMock(),
        body=None,
    )
    with patch("core.outreach.anthropic.Anthropic", return_value=mock_client):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert result["_error"] is True


def test_malformed_json_from_claude_returns_fallback(high_uplift_segment: dict) -> None:
    """If Claude returns non-JSON, return fallback (don't crash)."""
    with patch("core.outreach.anthropic.Anthropic",
               return_value=_make_mock_client("Here is your email: blah blah")):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert result["_error"] is True


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_output_contains_all_expected_keys(
    high_uplift_segment: dict,
    mock_claude_response: str,
) -> None:
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    expected = {"subject", "body", "cta", "predicted_uplift_group", "holdout_flag", "segment_id"}
    assert expected.issubset(result.keys())


def test_uplift_group_high_for_large_cate(
    high_uplift_segment: dict,
    mock_claude_response: str,
) -> None:
    """Enterprise + paid_search with CATE=0.18 should be classified as 'high'."""
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert result["predicted_uplift_group"] == "high"


def test_segment_id_echoed_in_output(
    high_uplift_segment: dict,
    mock_claude_response: str,
) -> None:
    with patch("core.outreach.anthropic.Anthropic", return_value=_make_mock_client(mock_claude_response)):
        result = generate_outreach(
            high_uplift_segment,
            product_context="test",
            cate_threshold=0.10,
            log_to_db=False,
        )
    assert result["segment_id"] == "enterprise_paid_search"
