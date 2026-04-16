"""
Schema mapper for customer-uploaded CSVs.

Maps source column names from HubSpot, Salesforce, Segment, or generic
exports to the GTMLens internal schema:

    user_id      — unique user/contact identifier
    timestamp    — event datetime
    stage        — funnel stage (impression|click|signup|activation|conversion)
    channel      — acquisition channel
    company_size — SMB | mid_market | enterprise
    industry     — customer's industry vertical
    treatment    — binary experiment assignment (0 = control, 1 = treatment)
    outcome      — binary outcome metric (0 = did not occur, 1 = occurred)

Mapping is column-name only. Value normalization (e.g. "Lead" → "signup")
is handled downstream in ingestion/normalizer.py (Phase 3).

Source detection heuristic:
    1. Explicit source_hint bypasses detection.
    2. Signature columns identify the source (see _SIGNATURES).
    3. Falls back to "generic" fuzzy matching if no source is detected.

Override semantics:
    overrides maps source_column_name → target_column_name.
    Overrides take precedence over source-specific and generic maps.
"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal schema definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: frozenset[str] = frozenset({"user_id", "timestamp", "stage", "outcome"})
OPTIONAL_COLUMNS: frozenset[str] = frozenset({"channel", "company_size", "industry", "treatment"})
ALL_COLUMNS: frozenset[str] = REQUIRED_COLUMNS | OPTIONAL_COLUMNS

SourceName = Literal["hubspot", "salesforce", "segment", "generic"]

# ---------------------------------------------------------------------------
# Source-specific column maps
# Keys are normalised source column names (lowercase, whitespace→underscore).
# Values are internal schema column names.
# ---------------------------------------------------------------------------

_HUBSPOT_MAP: dict[str, str] = {
    # identifiers
    "record_id":               "user_id",
    "contact_id":              "user_id",
    "vid":                     "user_id",
    "email":                   "user_id",
    # timestamps
    "create_date":             "timestamp",
    "createdate":              "timestamp",
    "last_activity_date":      "timestamp",
    "close_date":              "timestamp",
    # stage / funnel
    "lifecycle_stage":         "stage",
    "lifecyclestage":          "stage",
    "lead_status":             "stage",
    "dealstage":               "stage",
    # channel
    "original_source":         "channel",
    "hs_analytics_source":     "channel",
    "lead_source":             "channel",
    # company attributes
    "company_size":            "company_size",
    "numberofemployees":       "company_size",
    "num_employees":           "company_size",
    "industry":                "industry",
    # experiment
    "treatment_group":         "treatment",
    "ab_variant":              "treatment",
    "experiment_group":        "treatment",
    "converted":               "outcome",
    "hs_deal_is_closed_won":   "outcome",
    "activated":               "outcome",
}

_SALESFORCE_MAP: dict[str, str] = {
    # identifiers
    "id":                      "user_id",
    "contactid":               "user_id",
    "leadid":                  "user_id",
    "email":                   "user_id",
    # timestamps
    "createddate":             "timestamp",
    "lastactivitydate":        "timestamp",
    "closedate":               "timestamp",
    "converteddate":           "timestamp",
    # stage / funnel
    "leadstatus":              "stage",
    "lead_status":             "stage",
    "stagename":               "stage",
    "opportunity_stage":       "stage",
    "status":                  "stage",
    # channel
    "leadsource":              "channel",
    "lead_source":             "channel",
    "campaign_source":         "channel",
    # company attributes
    "numberofemployees":       "company_size",
    "company_size__c":         "company_size",
    "industry":                "industry",
    # experiment
    "treatment__c":            "treatment",
    "ab_group__c":             "treatment",
    "experiment_variant__c":   "treatment",
    "isconverted":             "outcome",
    "iswon":                   "outcome",
    "activated__c":            "outcome",
}

_SEGMENT_MAP: dict[str, str] = {
    # identifiers
    "userid":                  "user_id",
    "user_id":                 "user_id",
    "anonymousid":             "user_id",
    # timestamps
    "timestamp":               "timestamp",
    "sentat":                  "timestamp",
    "received_at":             "timestamp",
    "original_timestamp":      "timestamp",
    # stage / funnel
    "event":                   "stage",
    "event_name":              "stage",
    "name":                    "stage",
    # channel
    "channel":                 "channel",
    "properties_channel":      "channel",
    "context_campaign_source": "channel",
    "utm_source":              "channel",
    # company attributes
    "properties_company_size": "company_size",
    "traits_company_size":     "company_size",
    "properties_industry":     "industry",
    "traits_industry":         "industry",
    # experiment
    "properties_treatment":    "treatment",
    "properties_experiment_group": "treatment",
    "traits_ab_group":         "treatment",
    "properties_outcome":      "outcome",
    "properties_converted":    "outcome",
    "properties_activated":    "outcome",
}

# Generic fuzzy terms — substring match on normalised column name.
# Ordered from most specific to least specific within each target.
_GENERIC_TERMS: list[tuple[str, str]] = [
    # user_id
    ("user_id",       "user_id"),
    ("userid",        "user_id"),
    ("contact_id",    "user_id"),
    ("record_id",     "user_id"),
    ("customer_id",   "user_id"),
    ("person_id",     "user_id"),
    ("email",         "user_id"),
    ("id",            "user_id"),
    # timestamp
    ("timestamp",     "timestamp"),
    ("created_at",    "timestamp"),
    ("event_time",    "timestamp"),
    ("date",          "timestamp"),
    # stage
    ("stage",         "stage"),
    ("lifecycle",     "stage"),
    ("status",        "stage"),
    ("event",         "stage"),
    ("funnel",        "stage"),
    # channel
    ("channel",       "channel"),
    ("source",        "channel"),
    ("medium",        "channel"),
    ("utm_source",    "channel"),
    # company_size
    ("company_size",  "company_size"),
    ("employees",     "company_size"),
    ("headcount",     "company_size"),
    ("segment_size",  "company_size"),
    # industry
    ("industry",      "industry"),
    ("vertical",      "industry"),
    ("sector",        "industry"),
    # treatment
    ("treatment",     "treatment"),
    ("variant",       "treatment"),
    ("ab_group",      "treatment"),
    ("experiment",    "treatment"),
    ("test_group",    "treatment"),
    # outcome
    ("outcome",       "outcome"),
    ("converted",     "outcome"),
    ("activated",     "outcome"),
    ("result",        "outcome"),
]

# Columns whose presence strongly signals a particular source.
# Keys are normalised column names (_normalise applied).
# Include both camelCase-normalised forms (e.g. "lifecyclestage") AND
# space-normalised forms (e.g. "lifecycle_stage") to handle exports where
# column names use spaces vs. no separator.
_SIGNATURES: dict[SourceName, set[str]] = {
    "hubspot": {
        "lifecyclestage", "lifecycle_stage",      # Lifecycle Stage / lifecyclestage
        "hs_analytics_source",                     # internal HubSpot analytics col
        "vid",                                     # HubSpot internal visitor ID
        "hs_deal_is_closed_won",
        "record_id",                               # "Record ID" export header
    },
    "salesforce": {
        "leadsource", "lead_source",
        "stagename", "stage_name",
        "isconverted", "is_converted",
        "iswon", "is_won",
        "createddate",                             # Salesforce datetime (no space)
    },
    "segment": {
        "anonymousid", "anonymous_id",
        "sentat", "sent_at",
        "received_at",
        "context_campaign_source",
        "userid",                                  # Segment userId normalises to userid
        "properties_company_size",                 # Segment nested property columns
    },
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SchemaMapperError(ValueError):
    """Raised when a CSV cannot be mapped to the GTMLens internal schema."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class MappingResult:
    """
    Output of map_schema().

    Attributes:
        df:               Renamed DataFrame with internal schema column names.
        source_detected:  Which source system was identified (or "generic").
        column_map:       Applied mapping of original_col → internal_col.
        unmapped_optional: Internal optional columns with no source match.
        warnings:         Non-fatal issues the caller should surface.
    """

    df: pd.DataFrame
    source_detected: SourceName
    column_map: dict[str, str]
    unmapped_optional: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    """
    Normalise a column name for comparison.

    Lowercases, strips whitespace, replaces spaces and hyphens with underscores,
    and removes trailing/leading underscores.

    Args:
        name: Raw column name from the customer's CSV.

    Returns:
        Normalised string safe for dict key lookup.
    """
    return name.lower().strip().replace(" ", "_").replace("-", "_").strip("_")


def _detect_source(normalised_cols: set[str]) -> SourceName:
    """
    Identify the CRM/CDP source from column name signatures.

    Args:
        normalised_cols: Set of normalised column names from the uploaded CSV.

    Returns:
        Detected source name, or "generic" if no signature matched.
    """
    for source, sig_cols in _SIGNATURES.items():
        if sig_cols & normalised_cols:
            logger.debug("Detected source '%s' via signature columns %s", source, sig_cols & normalised_cols)
            return source
    return "generic"


def _source_map(source: SourceName) -> dict[str, str]:
    """Return the column map dict for the given source."""
    return {
        "hubspot":    _HUBSPOT_MAP,
        "salesforce": _SALESFORCE_MAP,
        "segment":    _SEGMENT_MAP,
        "generic":    {},
    }[source]


def _apply_generic_fuzzy(
    normalised_col: str,
    already_mapped_targets: set[str],
) -> str | None:
    """
    Find an internal target column via substring matching on generic terms.

    Only matches each target once to avoid duplicate mappings.

    Args:
        normalised_col:        Normalised source column name.
        already_mapped_targets: Internal columns already claimed by prior mappings.

    Returns:
        Internal column name if a match is found, else None.
    """
    for term, target in _GENERIC_TERMS:
        if target in already_mapped_targets:
            continue
        if term in normalised_col or normalised_col in term:
            return target
    return None


def _build_column_map(
    original_cols: list[str],
    source: SourceName,
    overrides: dict[str, str],
) -> dict[str, str]:
    """
    Construct the original_col → internal_col mapping.

    Priority: overrides > source-specific map > generic fuzzy.

    Args:
        original_cols: Column names exactly as they appear in the uploaded CSV.
        source:        Detected or user-specified source.
        overrides:     Caller-supplied mapping taking top priority.
                       Keys are original column names; values are internal names.

    Returns:
        Dict mapping original column name → internal schema column name.
        Only columns with a match are included.

    Raises:
        SchemaMapperError: If an override target is not a recognised internal column.
    """
    for target in overrides.values():
        if target not in ALL_COLUMNS:
            raise SchemaMapperError(
                f"Override target '{target}' is not a valid internal column. "
                f"Valid columns: {sorted(ALL_COLUMNS)}"
            )

    src_map = _source_map(source)
    result: dict[str, str] = {}
    claimed_targets: set[str] = set()

    # Pass 1: overrides (original col name → internal col, exact match)
    override_norm = {_normalise(k): v for k, v in overrides.items()}
    for orig in original_cols:
        norm = _normalise(orig)
        if norm in override_norm:
            target = override_norm[norm]
            result[orig] = target
            claimed_targets.add(target)

    # Pass 2: source-specific map
    for orig in original_cols:
        if orig in result:
            continue
        norm = _normalise(orig)
        if norm in src_map:
            target = src_map[norm]
            if target not in claimed_targets:
                result[orig] = target
                claimed_targets.add(target)

    # Pass 3: generic fuzzy fallback
    for orig in original_cols:
        if orig in result:
            continue
        norm = _normalise(orig)
        target = _apply_generic_fuzzy(norm, claimed_targets)
        if target is not None:
            result[orig] = target
            claimed_targets.add(target)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_schema(
    df: pd.DataFrame,
    source_hint: SourceName | None = None,
    overrides: dict[str, str] | None = None,
) -> MappingResult:
    """
    Map a customer CSV DataFrame to the GTMLens internal schema.

    Detects the source system from column signatures (unless source_hint is
    given), builds a column mapping, renames the DataFrame's columns, and
    returns a MappingResult.  The input DataFrame is never mutated.

    Args:
        df:          DataFrame loaded from the customer's CSV.
        source_hint: Optional override for source detection.
                     One of "hubspot", "salesforce", "segment", "generic".
        overrides:   Optional dict of original_col → internal_col mappings
                     that take precedence over all automatic mapping.

    Returns:
        MappingResult with the renamed DataFrame and mapping metadata.

    Raises:
        SchemaMapperError: If the DataFrame is empty, has no columns, any
                           required internal column cannot be mapped, or an
                           override specifies an invalid internal column name.
        ValueError:        If df is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    if df.empty:
        raise SchemaMapperError("Uploaded CSV is empty — no rows to process.")
    if len(df.columns) == 0:
        raise SchemaMapperError("Uploaded CSV has no columns.")

    overrides = overrides or {}
    original_cols = list(df.columns)
    norm_cols = {_normalise(c) for c in original_cols}

    source: SourceName = source_hint if source_hint is not None else _detect_source(norm_cols)
    logger.info("Mapping schema | source=%s | columns=%s", source, original_cols)

    col_map = _build_column_map(original_cols, source, overrides)

    # Check that all required internal columns are covered
    mapped_targets = set(col_map.values())
    missing_required = REQUIRED_COLUMNS - mapped_targets
    if missing_required:
        raise SchemaMapperError(
            f"Cannot map required columns: {sorted(missing_required)}. "
            f"Source columns available: {original_cols}. "
            f"Use the 'overrides' parameter to specify the mapping manually."
        )

    # Identify optional columns with no mapping (non-fatal)
    unmapped_optional = sorted(OPTIONAL_COLUMNS - mapped_targets)
    warnings: list[str] = []
    if unmapped_optional:
        warnings.append(
            f"Optional columns not found in source: {unmapped_optional}. "
            f"Downstream analysis may have reduced segmentation capability."
        )
        logger.warning("Unmapped optional columns: %s", unmapped_optional)

    # Rename without mutating the input
    renamed_df = df.rename(columns=col_map)

    # Drop source columns that didn't map to anything (keep internal schema only)
    keep_cols = [c for c in renamed_df.columns if c in ALL_COLUMNS]
    renamed_df = renamed_df[keep_cols].copy()

    logger.info(
        "Mapping complete | source=%s | mapped=%s | unmapped_optional=%s",
        source, sorted(mapped_targets), unmapped_optional,
    )

    return MappingResult(
        df=renamed_df,
        source_detected=source,
        column_map=col_map,
        unmapped_optional=unmapped_optional,
        warnings=warnings,
    )
