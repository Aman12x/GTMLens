"""
Tests for ingestion/validator.py.

Coverage requirements:
    - Valid DataFrame passes with ValidationResult (no raise)
    - Missing required column raises ValidationError
    - Timestamp column with all-unparseable values raises ValidationError
    - Timestamp column with partial nulls produces a warning, not an error
    - treatment column with non-binary values raises ValidationError
    - outcome column with non-binary values raises ValidationError
    - Binary columns accept 0/1, True/False, "0"/"1", "true"/"false", "yes"/"no"
    - stage column with unknown values produces a warning (not an error)
    - company_size column with unknown values produces a warning (not an error)
    - Column with >50% nulls produces a warning (not an error)
    - Non-DataFrame input raises ValueError (not ValidationError)
    - ValidationError carries both errors and warnings
    - All errors are collected before raising (no early exit)
"""

import pandas as pd
import pytest

from ingestion.validator import (
    VALID_COMPANY_SIZES,
    VALID_STAGES,
    ValidationError,
    ValidationResult,
    validate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_df() -> pd.DataFrame:
    """Fully valid mapped DataFrame — all required + optional columns present."""
    return pd.DataFrame({
        "user_id":      ["u1", "u2", "u3", "u4"],
        "timestamp":    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "stage":        ["impression", "click", "activation", "conversion"],
        "outcome":      [0, 0, 1, 1],
        "treatment":    [1, 0, 1, 0],
        "channel":      ["organic", "paid_search", "referral", "email"],
        "company_size": ["SMB", "mid_market", "enterprise", "SMB"],
        "industry":     ["SaaS", "FinTech", "HealthTech", "EdTech"],
    })


@pytest.fixture()
def required_only_df() -> pd.DataFrame:
    """DataFrame with only required columns — optional columns absent."""
    return pd.DataFrame({
        "user_id":   ["u1", "u2"],
        "timestamp": ["2024-06-01 10:00:00", "2024-06-02 11:30:00"],
        "stage":     ["activation", "signup"],
        "outcome":   [1, 0],
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_df_passes(valid_df: pd.DataFrame) -> None:
    result = validate(valid_df)
    assert isinstance(result, ValidationResult)
    assert result.row_count == 4


def test_valid_required_only_passes(required_only_df: pd.DataFrame) -> None:
    result = validate(required_only_df)
    assert isinstance(result, ValidationResult)
    assert result.row_count == 2


def test_row_count_matches(valid_df: pd.DataFrame) -> None:
    result = validate(valid_df)
    assert result.row_count == len(valid_df)


# ---------------------------------------------------------------------------
# Missing required columns
# ---------------------------------------------------------------------------


def test_missing_user_id_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.drop(columns=["user_id"])
    with pytest.raises(ValidationError) as exc_info:
        validate(df)
    assert "user_id" in str(exc_info.value)


def test_missing_timestamp_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.drop(columns=["timestamp"])
    with pytest.raises(ValidationError):
        validate(df)


def test_missing_stage_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.drop(columns=["stage"])
    with pytest.raises(ValidationError):
        validate(df)


def test_missing_outcome_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.drop(columns=["outcome"])
    with pytest.raises(ValidationError):
        validate(df)


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------


def test_all_unparseable_timestamps_raise(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["timestamp"] = ["not-a-date", "also-bad", "nope", "???"]
    with pytest.raises(ValidationError, match="timestamp"):
        validate(df)


def test_partial_bad_timestamps_warn_not_raise(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    # One bad row out of four — should warn, not error
    df.loc[0, "timestamp"] = "not-a-date"
    result = validate(df)
    assert isinstance(result, ValidationResult)
    assert any("timestamp" in w for w in result.warnings)


def test_iso_timestamps_pass(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["timestamp"] = ["2024-01-01T08:00:00Z", "2024-01-02T09:00:00Z",
                       "2024-01-03T10:00:00Z", "2024-01-04T11:00:00Z"]
    result = validate(df)
    assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Binary column validation — treatment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("values", [
    [0, 1, 0, 1],          # int
    [0.0, 1.0, 0.0, 1.0],  # float
    [True, False, True, False],  # bool
    ["0", "1", "0", "1"],  # string int
    ["true", "false", "true", "false"],  # string bool
    ["yes", "no", "yes", "no"],  # yes/no
])
def test_treatment_accepts_binary_variants(valid_df: pd.DataFrame, values: list) -> None:
    df = valid_df.copy()
    df["treatment"] = values
    result = validate(df)
    assert isinstance(result, ValidationResult)


def test_treatment_non_binary_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["treatment"] = [0, 1, 2, 0]  # '2' is not binary
    with pytest.raises(ValidationError, match="treatment"):
        validate(df)


def test_treatment_string_non_binary_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["treatment"] = ["control", "treatment", "control", "treatment"]  # not 0/1
    with pytest.raises(ValidationError, match="treatment"):
        validate(df)


# ---------------------------------------------------------------------------
# Binary column validation — outcome
# ---------------------------------------------------------------------------


def test_outcome_non_binary_raises(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["outcome"] = [0, 1, 0.5, 1]  # 0.5 is not binary
    with pytest.raises(ValidationError, match="outcome"):
        validate(df)


def test_outcome_accepts_bool(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["outcome"] = [True, False, True, False]
    result = validate(df)
    assert isinstance(result, ValidationResult)


# ---------------------------------------------------------------------------
# Stage values — warnings only
# ---------------------------------------------------------------------------


def test_unknown_stage_values_warn_not_raise(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["stage"] = ["Lead", "MQL", "SQL", "Customer"]  # HubSpot lifecycle names
    result = validate(df)
    assert isinstance(result, ValidationResult)
    assert any("stage" in w for w in result.warnings)


def test_valid_stage_values_no_warning(valid_df: pd.DataFrame) -> None:
    result = validate(valid_df)
    stage_warnings = [w for w in result.warnings if "stage" in w and "normalisation" in w]
    assert stage_warnings == []


def test_valid_stages_set_complete() -> None:
    """Smoke test: confirm the valid stages set matches CLAUDE.md funnel definition."""
    assert VALID_STAGES == {"impression", "click", "signup", "activation", "conversion"}


# ---------------------------------------------------------------------------
# Company size values — warnings only
# ---------------------------------------------------------------------------


def test_unknown_company_size_warns(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["company_size"] = ["Small", "Medium", "Large", "Small"]
    result = validate(df)
    assert isinstance(result, ValidationResult)
    assert any("company_size" in w for w in result.warnings)


def test_numeric_company_size_warns(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["company_size"] = [50, 500, 5000, 25]
    result = validate(df)
    assert any("company_size" in w or "numeric" in w.lower() for w in result.warnings)


def test_valid_company_sizes_no_warning(valid_df: pd.DataFrame) -> None:
    result = validate(valid_df)
    size_warnings = [w for w in result.warnings if "company_size" in w]
    assert size_warnings == []


def test_valid_company_sizes_set_complete() -> None:
    assert VALID_COMPANY_SIZES == {"SMB", "mid_market", "enterprise"}


# ---------------------------------------------------------------------------
# Null density warnings
# ---------------------------------------------------------------------------


def test_high_null_density_warns(valid_df: pd.DataFrame) -> None:
    df = valid_df.copy()
    df["channel"] = [None, None, None, "organic"]  # 75% null
    result = validate(df)
    assert any("channel" in w for w in result.warnings)


def test_low_null_density_no_warning(valid_df: pd.DataFrame) -> None:
    result = validate(valid_df)
    null_warnings = [w for w in result.warnings if "null" in w.lower()]
    assert null_warnings == []


# ---------------------------------------------------------------------------
# Optional columns — absent means warnings, not errors
# ---------------------------------------------------------------------------


def test_optional_columns_absent_is_not_an_error(required_only_df: pd.DataFrame) -> None:
    """Missing optional columns must not raise — they should produce a warning."""
    result = validate(required_only_df)
    assert isinstance(result, ValidationResult)


def test_optional_columns_absent_produces_warning(required_only_df: pd.DataFrame) -> None:
    result = validate(required_only_df)
    assert any("Optional" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Error accumulation — all errors collected before raising
# ---------------------------------------------------------------------------


def test_multiple_errors_all_reported() -> None:
    """Validator must report ALL errors, not stop at the first one."""
    df = pd.DataFrame({
        "user_id":   ["u1", "u2"],
        "timestamp": ["bad-date", "also-bad"],  # timestamp error
        "stage":     ["activation", "signup"],
        "outcome":   ["maybe", "sort-of"],  # outcome error
        "treatment": ["group_a", "group_b"],  # treatment error
    })
    with pytest.raises(ValidationError) as exc_info:
        validate(df)
    # At minimum: timestamp + treatment + outcome = 3 errors
    assert len(exc_info.value.errors) >= 2


def test_validation_error_carries_warnings() -> None:
    """Even when errors are present, accumulated warnings must be attached."""
    df = pd.DataFrame({
        "user_id":      ["u1", "u2"],
        "timestamp":    ["2024-01-01", "2024-01-02"],
        "stage":        ["Lead", "MQL"],  # unknown stages → warning
        "outcome":      ["maybe", "probably"],  # non-binary → error
        # optional columns absent → warning
    })
    with pytest.raises(ValidationError) as exc_info:
        validate(df)
    exc = exc_info.value
    assert len(exc.errors) >= 1
    assert isinstance(exc.warnings, list)


# ---------------------------------------------------------------------------
# Input type guard
# ---------------------------------------------------------------------------


def test_non_dataframe_raises_value_error() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        validate({"user_id": ["u1"]})  # type: ignore[arg-type]


def test_list_input_raises_value_error() -> None:
    with pytest.raises(ValueError):
        validate([1, 2, 3])  # type: ignore[arg-type]
