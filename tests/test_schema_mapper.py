"""
Tests for ingestion/schema_mapper.py.

Coverage requirements:
    - HubSpot columns map to correct internal names
    - Salesforce columns map to correct internal names
    - Segment columns map to correct internal names
    - Custom overrides take precedence over source-specific map
    - Source auto-detected correctly from signature columns
    - Unknown columns trigger generic fuzzy fallback
    - Missing required column raises SchemaMapperError (not a crash)
    - Invalid override target raises SchemaMapperError
    - Empty DataFrame raises SchemaMapperError
    - Input DataFrame is never mutated
    - Only internal schema columns are present in the output DataFrame
    - Optional columns absent from source appear in unmapped_optional
"""

import pandas as pd
import pytest

from ingestion.schema_mapper import (
    ALL_COLUMNS,
    REQUIRED_COLUMNS,
    MappingResult,
    SchemaMapperError,
    map_schema,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal sample DataFrames mimicking real CRM exports
# ---------------------------------------------------------------------------


@pytest.fixture()
def hubspot_df() -> pd.DataFrame:
    """Minimal HubSpot contacts export (typical column set)."""
    return pd.DataFrame({
        "Record ID":          ["001", "002", "003"],
        "Create Date":        ["2024-01-10", "2024-01-11", "2024-01-12"],
        "Lifecycle Stage":    ["subscriber", "lead", "customer"],
        "Original Source":    ["ORGANIC_SEARCH", "PAID_SEARCH", "EMAIL"],
        "Company Size":       ["SMB", "mid_market", "enterprise"],
        "Industry":           ["SaaS", "FinTech", "HealthTech"],
        "Converted":          [0, 1, 1],
        "Treatment Group":    [0, 1, 0],
        "First Name":         ["Alice", "Bob", "Carol"],  # unmapped noise column
    })


@pytest.fixture()
def salesforce_df() -> pd.DataFrame:
    """Minimal Salesforce Lead export (typical column set)."""
    return pd.DataFrame({
        "Id":                 ["sf001", "sf002"],
        "CreatedDate":        ["2024-02-01T08:00:00Z", "2024-02-02T09:30:00Z"],
        "LeadStatus":         ["Open", "Qualified"],
        "LeadSource":         ["Web", "Paid Search"],
        "NumberOfEmployees":  [25, 800],
        "Industry":           ["Manufacturing", "FinTech"],
        "IsConverted":        [0, 1],
        "Treatment__c":       [1, 0],
        "AnnualRevenue":      [500000, 2000000],  # unmapped noise column
    })


@pytest.fixture()
def segment_df() -> pd.DataFrame:
    """Minimal Segment event export."""
    return pd.DataFrame({
        "userId":                   ["u_001", "u_002", "u_003"],
        "timestamp":                ["2024-03-01 10:00:00", "2024-03-01 11:00:00", "2024-03-02 09:00:00"],
        "event":                    ["activation", "signup", "conversion"],
        "channel":                  ["organic", "paid_search", "referral"],
        "properties_company_size":  ["SMB", "enterprise", "mid_market"],
        "properties_industry":      ["SaaS", "EdTech", "Logistics"],
        "properties_treatment":     [1, 0, 1],
        "properties_outcome":       [1, 0, 1],
    })


@pytest.fixture()
def minimal_valid_df() -> pd.DataFrame:
    """DataFrame with exactly the required internal column names already set."""
    return pd.DataFrame({
        "user_id":   ["u1", "u2"],
        "timestamp": ["2024-01-01", "2024-01-02"],
        "stage":     ["activation", "conversion"],
        "outcome":   [1, 0],
    })


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------


def test_hubspot_source_detected(hubspot_df: pd.DataFrame) -> None:
    result = map_schema(hubspot_df)
    assert result.source_detected == "hubspot"


def test_salesforce_source_detected(salesforce_df: pd.DataFrame) -> None:
    result = map_schema(salesforce_df)
    assert result.source_detected == "salesforce"


def test_segment_source_detected(segment_df: pd.DataFrame) -> None:
    result = map_schema(segment_df)
    assert result.source_detected == "segment"


def test_source_hint_overrides_detection(hubspot_df: pd.DataFrame) -> None:
    """Explicit source_hint must be used even when detection would pick something else."""
    # Force generic — mapping will rely on fuzzy fallback
    result = map_schema(hubspot_df, source_hint="generic")
    assert result.source_detected == "generic"


# ---------------------------------------------------------------------------
# HubSpot mapping
# ---------------------------------------------------------------------------


def test_hubspot_required_columns_mapped(hubspot_df: pd.DataFrame) -> None:
    result = map_schema(hubspot_df)
    for col in REQUIRED_COLUMNS:
        assert col in result.df.columns, f"Required column '{col}' missing from mapped output"


def test_hubspot_column_map_entries(hubspot_df: pd.DataFrame) -> None:
    result = map_schema(hubspot_df)
    # Record ID → user_id
    assert result.column_map.get("Record ID") == "user_id"
    # Create Date → timestamp
    assert result.column_map.get("Create Date") == "timestamp"
    # Lifecycle Stage → stage
    assert result.column_map.get("Lifecycle Stage") == "stage"
    # Original Source → channel
    assert result.column_map.get("Original Source") == "channel"
    # Converted → outcome
    assert result.column_map.get("Converted") == "outcome"


def test_hubspot_noise_columns_dropped(hubspot_df: pd.DataFrame) -> None:
    """Columns that don't map to an internal schema column must be dropped."""
    result = map_schema(hubspot_df)
    assert "First Name" not in result.df.columns


def test_hubspot_only_internal_columns_in_output(hubspot_df: pd.DataFrame) -> None:
    result = map_schema(hubspot_df)
    for col in result.df.columns:
        assert col in ALL_COLUMNS, f"Non-internal column '{col}' found in output"


# ---------------------------------------------------------------------------
# Salesforce mapping
# ---------------------------------------------------------------------------


def test_salesforce_required_columns_mapped(salesforce_df: pd.DataFrame) -> None:
    result = map_schema(salesforce_df)
    for col in REQUIRED_COLUMNS:
        assert col in result.df.columns


def test_salesforce_column_map_entries(salesforce_df: pd.DataFrame) -> None:
    result = map_schema(salesforce_df)
    assert result.column_map.get("Id") == "user_id"
    assert result.column_map.get("CreatedDate") == "timestamp"
    assert result.column_map.get("LeadStatus") == "stage"
    assert result.column_map.get("LeadSource") == "channel"
    assert result.column_map.get("IsConverted") == "outcome"


def test_salesforce_noise_columns_dropped(salesforce_df: pd.DataFrame) -> None:
    result = map_schema(salesforce_df)
    assert "AnnualRevenue" not in result.df.columns


# ---------------------------------------------------------------------------
# Segment mapping
# ---------------------------------------------------------------------------


def test_segment_required_columns_mapped(segment_df: pd.DataFrame) -> None:
    result = map_schema(segment_df)
    for col in REQUIRED_COLUMNS:
        assert col in result.df.columns


def test_segment_column_map_entries(segment_df: pd.DataFrame) -> None:
    result = map_schema(segment_df)
    assert result.column_map.get("userId") == "user_id"
    assert result.column_map.get("timestamp") == "timestamp"
    assert result.column_map.get("event") == "stage"
    assert result.column_map.get("channel") == "channel"


# ---------------------------------------------------------------------------
# Override behaviour
# ---------------------------------------------------------------------------


def test_override_takes_precedence_over_source_map() -> None:
    """An override for 'email' → 'user_id' must win over the source map."""
    df = pd.DataFrame({
        "email":         ["a@x.com", "b@x.com"],
        "created_at":    ["2024-01-01", "2024-01-02"],
        "funnel_stage":  ["activation", "signup"],
        "converted_flag": [1, 0],
    })
    result = map_schema(
        df,
        source_hint="generic",
        overrides={"email": "user_id"},
    )
    assert result.column_map.get("email") == "user_id"
    assert "user_id" in result.df.columns


def test_override_maps_non_obvious_column_name() -> None:
    """Overrides should handle completely arbitrary source column names."""
    df = pd.DataFrame({
        "contact_uuid":    ["abc", "def"],
        "event_occurred":  ["2024-06-01", "2024-06-02"],
        "pipeline_step":   ["activation", "signup"],
        "goal_reached":    [1, 0],
    })
    result = map_schema(
        df,
        source_hint="generic",
        overrides={
            "contact_uuid":   "user_id",
            "event_occurred": "timestamp",
            "pipeline_step":  "stage",
            "goal_reached":   "outcome",
        },
    )
    for col in REQUIRED_COLUMNS:
        assert col in result.df.columns


def test_override_invalid_target_raises() -> None:
    """An override pointing to a non-existent internal column must raise SchemaMapperError."""
    df = pd.DataFrame({
        "id": ["u1"],
        "ts": ["2024-01-01"],
        "st": ["activation"],
        "out": [1],
    })
    with pytest.raises(SchemaMapperError, match="not a valid internal column"):
        map_schema(df, overrides={"id": "nonexistent_column"})


# ---------------------------------------------------------------------------
# Generic fuzzy fallback
# ---------------------------------------------------------------------------


def test_generic_fuzzy_matches_on_substring() -> None:
    """Fuzzy matching should handle column names that contain known terms."""
    df = pd.DataFrame({
        "customer_user_id":   ["u1", "u2"],
        "event_timestamp":    ["2024-01-01", "2024-01-02"],
        "current_stage":      ["activation", "signup"],
        "outcome_flag":       [1, 0],
    })
    result = map_schema(df, source_hint="generic")
    assert "user_id" in result.df.columns
    assert "timestamp" in result.df.columns


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises() -> None:
    with pytest.raises(SchemaMapperError, match="empty"):
        map_schema(pd.DataFrame())


def test_no_columns_raises() -> None:
    with pytest.raises(SchemaMapperError):
        map_schema(pd.DataFrame(index=range(3)))


def test_missing_required_column_raises() -> None:
    """DataFrame with no mappable outcome column must raise SchemaMapperError."""
    df = pd.DataFrame({
        "user_id":   ["u1"],
        "timestamp": ["2024-01-01"],
        "stage":     ["activation"],
        # 'outcome' is absent and unmappable
    })
    with pytest.raises(SchemaMapperError, match="outcome"):
        map_schema(df, source_hint="generic")


def test_non_dataframe_input_raises() -> None:
    with pytest.raises(ValueError, match="pandas DataFrame"):
        map_schema({"col": [1, 2, 3]})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_input_dataframe_not_mutated(hubspot_df: pd.DataFrame) -> None:
    original_columns = list(hubspot_df.columns)
    original_shape = hubspot_df.shape
    map_schema(hubspot_df)
    assert list(hubspot_df.columns) == original_columns
    assert hubspot_df.shape == original_shape


# ---------------------------------------------------------------------------
# Optional column handling
# ---------------------------------------------------------------------------


def test_missing_optional_columns_reported(minimal_valid_df: pd.DataFrame) -> None:
    """When optional columns are absent, they appear in unmapped_optional."""
    result = map_schema(minimal_valid_df, source_hint="generic")
    # All optional cols should be unmapped since none are in minimal_valid_df
    for col in ["channel", "company_size", "industry", "treatment"]:
        assert col in result.unmapped_optional


def test_missing_optional_columns_produce_warning(minimal_valid_df: pd.DataFrame) -> None:
    result = map_schema(minimal_valid_df, source_hint="generic")
    assert len(result.warnings) > 0
    assert any("Optional" in w for w in result.warnings)


def test_all_optional_columns_present_no_warning(segment_df: pd.DataFrame) -> None:
    """When all optional columns map successfully, unmapped_optional is empty."""
    result = map_schema(segment_df)
    assert result.unmapped_optional == []
    assert not any("Optional" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_returns_mapping_result_type(hubspot_df: pd.DataFrame) -> None:
    result = map_schema(hubspot_df)
    assert isinstance(result, MappingResult)
    assert isinstance(result.df, pd.DataFrame)
    assert isinstance(result.column_map, dict)
    assert isinstance(result.warnings, list)
    assert isinstance(result.unmapped_optional, list)
