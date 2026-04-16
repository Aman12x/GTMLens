"""
DataFrame validator for the GTMLens internal schema.

Validates a DataFrame that has already been through schema_mapper.map_schema().
Checks column presence, data types, and domain constraints.

Validation rules:
    Errors (raise ValidationError — upload must be rejected):
        - A required column is missing
        - timestamp column cannot be parsed as datetime
        - treatment column contains values outside {0, 1} after coercion
        - outcome column contains values outside {0, 1} after coercion

    Warnings (non-fatal — surfaced to the caller, upload proceeds):
        - Optional columns are absent
        - stage column contains values not in VALID_STAGES
        - company_size contains values not in VALID_COMPANY_SIZES
        - More than 50% of rows have null values in any column

Callers should pass the result of map_schema().df directly.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from ingestion.schema_mapper import OPTIONAL_COLUMNS, REQUIRED_COLUMNS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

VALID_STAGES: frozenset[str] = frozenset(
    {"impression", "click", "signup", "activation", "conversion"}
)

VALID_COMPANY_SIZES: frozenset[str] = frozenset({"SMB", "mid_market", "enterprise"})

_BINARY_TRUTHY: set = {1, 1.0, "1", "true", "yes", True}
_BINARY_FALSY: set = {0, 0.0, "0", "false", "no", False}

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """
    Raised when the uploaded data has errors that make it unusable.

    Attributes:
        errors:   List of error messages (blocking).
        warnings: List of warning messages (non-blocking).
    """

    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        self.errors = errors
        self.warnings = warnings or []
        combined = "; ".join(errors)
        super().__init__(f"Validation failed ({len(errors)} error(s)): {combined}")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """
    Output of validate() when the upload passes (no blocking errors).

    Attributes:
        row_count: Number of rows in the validated DataFrame.
        warnings:  Non-fatal issues the caller should surface in the UI.
    """

    row_count: int
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal validators — each returns (errors, warnings)
# ---------------------------------------------------------------------------


def _check_required_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Verify all required columns are present.

    Args:
        df: DataFrame after schema mapping.

    Returns:
        Tuple of (errors, warnings).
    """
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        return [f"Missing required columns: {missing}"], []
    return [], []


def _check_optional_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Warn when optional columns are absent (non-blocking).

    Args:
        df: DataFrame after schema mapping.

    Returns:
        Tuple of (errors, warnings).
    """
    missing_opt = sorted(OPTIONAL_COLUMNS - set(df.columns))
    if missing_opt:
        return [], [f"Optional columns absent: {missing_opt} — segmentation will be limited."]
    return [], []


def _check_timestamp(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Attempt to coerce the timestamp column to datetime.

    Args:
        df: DataFrame with a 'timestamp' column.

    Returns:
        Tuple of (errors, warnings). Error if no rows can be parsed.
    """
    if "timestamp" not in df.columns:
        return [], []  # caught by _check_required_columns

    try:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    except Exception as exc:
        return [f"timestamp column cannot be parsed as datetime: {exc}"], []

    null_count = int(parsed.isna().sum())
    total = len(df)

    if null_count == total:
        return ["timestamp column: no rows could be parsed as a valid datetime."], []

    warnings: list[str] = []
    if null_count > 0:
        pct = null_count / total * 100
        warnings.append(
            f"timestamp: {null_count} of {total} rows ({pct:.1f}%) could not be "
            f"parsed as datetime and will be treated as null."
        )

    return [], warnings


def _check_binary_column(
    df: pd.DataFrame,
    col: str,
    label: str,
) -> tuple[list[str], list[str]]:
    """
    Validate that a column contains only binary values (0/1 or equivalent).

    Accepts: 0, 1, True, False, "0", "1", "true", "false", "yes", "no" (case-insensitive).
    Rejects: any other value.

    Args:
        df:    DataFrame containing the column.
        col:   Column name to check.
        label: Human-readable column label for error messages.

    Returns:
        Tuple of (errors, warnings).
    """
    if col not in df.columns:
        return [], []

    series = df[col].dropna()
    if series.empty:
        return [f"{label} column is entirely null."], []

    normalised = series.astype(str).str.lower().str.strip()
    # Strip trailing ".0" so float representations (0.0, 1.0) compare equal to "0"/"1"
    normalised = normalised.str.replace(r"\.0+$", "", regex=True)
    valid_str = {"0", "1", "true", "false", "yes", "no"}
    invalid_mask = ~normalised.isin(valid_str)
    invalid_vals = series[invalid_mask.values].unique().tolist()[:5]  # show at most 5

    if invalid_vals:
        return [
            f"{label} column must be binary (0/1). "
            f"Found non-binary values (showing up to 5): {invalid_vals}"
        ], []

    return [], []


def _check_stage_values(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Warn when stage column contains values outside the known valid set.

    This is a warning (not an error) because customers may use their own
    stage names that require normalisation before analysis.

    Args:
        df: DataFrame with a 'stage' column.

    Returns:
        Tuple of (errors, warnings).
    """
    if "stage" not in df.columns:
        return [], []

    unique_vals = set(df["stage"].dropna().astype(str).str.strip())
    unknown = unique_vals - VALID_STAGES
    if unknown:
        return [], [
            f"stage column contains values not in {sorted(VALID_STAGES)}: "
            f"{sorted(unknown)}. These will need normalisation before analysis."
        ]
    return [], []


def _check_company_size_values(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Warn when company_size contains values outside SMB/mid_market/enterprise.

    Args:
        df: DataFrame with a 'company_size' column.

    Returns:
        Tuple of (errors, warnings).
    """
    if "company_size" not in df.columns:
        return [], []

    unique_vals = set(df["company_size"].dropna().astype(str).str.strip())
    unknown = unique_vals - VALID_COMPANY_SIZES
    if unknown:
        return [], [
            f"company_size contains non-standard values: {sorted(unknown)}. "
            f"Expected: {sorted(VALID_COMPANY_SIZES)}. "
            f"Numeric employee counts must be bucketed before analysis."
        ]
    return [], []


def _check_null_density(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Warn when any column has more than 50% null values.

    Args:
        df: Full DataFrame to check.

    Returns:
        Tuple of (errors, warnings).
    """
    warnings: list[str] = []
    for col in df.columns:
        null_pct = df[col].isna().mean()
        if null_pct > 0.50:
            warnings.append(
                f"Column '{col}' is {null_pct:.0%} null — data quality may affect "
                f"analysis reliability."
            )
    return [], warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CHECKS = [
    _check_required_columns,
    _check_optional_columns,
    _check_timestamp,
    _check_stage_values,
    _check_company_size_values,
    _check_null_density,
]


def validate(df: pd.DataFrame) -> ValidationResult:
    """
    Validate a schema-mapped DataFrame against GTMLens data requirements.

    Runs all checks. Collects all errors and warnings before raising so the
    caller receives the full picture in one pass.

    Args:
        df: DataFrame returned by map_schema().df (internal schema column names).

    Returns:
        ValidationResult on success (no blocking errors).

    Raises:
        ValueError:       If df is not a pandas DataFrame.
        ValidationError:  If any blocking errors are found. The exception
                          carries both errors and non-blocking warnings.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}")

    all_errors: list[str] = []
    all_warnings: list[str] = []

    for check_fn in _CHECKS:
        errors, warnings = check_fn(df)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    # Binary columns checked separately because they need a label argument
    for col, label in [("treatment", "treatment"), ("outcome", "outcome")]:
        errors, warnings = _check_binary_column(df, col, label)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if all_errors:
        logger.warning(
            "Validation failed | errors=%d | warnings=%d | errors=%s",
            len(all_errors), len(all_warnings), all_errors,
        )
        raise ValidationError(errors=all_errors, warnings=all_warnings)

    logger.info(
        "Validation passed | rows=%d | warnings=%d",
        len(df), len(all_warnings),
    )
    return ValidationResult(row_count=len(df), warnings=all_warnings)
