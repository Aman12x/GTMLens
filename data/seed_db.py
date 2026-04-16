"""
Seed script: generates synthetic funnel data and populates DuckDB + SQLite.

Run once before starting the API:
    uv run python -m data.seed_db

Environment variables (with defaults):
    DATABASE_URL   → path to DuckDB file  (default: <project_root>/data/gtmlens.duckdb)
    SQLITE_PATH    → path to SQLite file  (default: <project_root>/data/logs.db)
    N_USERS        → number of users to generate (default: 50000)
    SEED           → random seed (default: 42)

Schema created:
    DuckDB:
        events          — one row per user per stage reached
        users           — one row per user (wide format, all columns)
        daily_summary   — pre-aggregated daily funnel metrics
    SQLite:
        experiment_logs — experiment run records
        outreach_log    — outreach message send records
        api_usage       — Claude API token usage log
"""

import logging
import os
import sqlite3
from pathlib import Path

import duckdb
import pandas as pd

from data.synthetic import generate_funnel_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve paths relative to this file so the script works from any cwd
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_DB_URL = str(_PROJECT_ROOT / "data" / "gtmlens.duckdb")
_DEFAULT_SQLITE = str(_PROJECT_ROOT / "data" / "logs.db")

STAGES = ["impression", "click", "signup", "activation", "conversion"]


def _build_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot wide user DataFrame into a long-format events table.

    Each row = one (user_id, stage) pair, only for stages the user reached.
    Vectorised implementation — no iterrows().

    Args:
        df: Wide-format user DataFrame from generate_funnel_data().

    Returns:
        Long-format DataFrame with columns:
            user_id, stage, event_date, company_size, industry,
            channel, treatment, revenue
    """
    base = ["user_id", "company_size", "industry", "channel", "treatment"]

    frames: list[pd.DataFrame] = []

    # Impression — all users
    impr = df[base].copy()
    impr["stage"] = "impression"
    impr["event_date"] = df["impression_date"]
    impr["revenue"] = 0.0
    frames.append(impr)

    # Click — users who clicked
    clicked = df.loc[df["clicked"] == 1, base].copy()
    clicked["stage"] = "click"
    clicked["event_date"] = df.loc[df["clicked"] == 1, "click_date"]
    clicked["revenue"] = 0.0
    frames.append(clicked)

    # Signup — users who signed up
    signed = df.loc[df["signed_up"] == 1, base].copy()
    signed["stage"] = "signup"
    signed["event_date"] = df.loc[df["signed_up"] == 1, "signup_date"]
    signed["revenue"] = 0.0
    frames.append(signed)

    # Activation
    activated = df.loc[df["activated"] == 1, base].copy()
    activated["stage"] = "activation"
    activated["event_date"] = df.loc[df["activated"] == 1, "activation_date"]
    activated["revenue"] = 0.0
    frames.append(activated)

    # Conversion — carries actual revenue
    converted = df.loc[df["converted"] == 1, base].copy()
    converted["stage"] = "conversion"
    converted["event_date"] = df.loc[df["converted"] == 1, "conversion_date"]
    converted["revenue"] = df.loc[df["converted"] == 1, "revenue"].values
    frames.append(converted)

    result = pd.concat(frames, ignore_index=True)
    result = result[
        ["user_id", "stage", "event_date", "company_size", "industry", "channel", "treatment", "revenue"]
    ]
    # Drop rows where event_date is null (shouldn't happen but defensive)
    return result.dropna(subset=["event_date"]).reset_index(drop=True)


def _build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-aggregate funnel metrics by day for fast dashboard queries.

    Args:
        df: Wide-format user DataFrame.

    Returns:
        DataFrame with columns:
            date, impressions, clicks, signups, activations, conversions,
            treatment_activations, control_activations,
            activation_rate, treatment_activation_rate, control_activation_rate
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["impression_date"]).dt.date

    agg = df.groupby("date").agg(
        impressions=("user_id", "count"),
        clicks=("clicked", "sum"),
        signups=("signed_up", "sum"),
        activations=("activated", "sum"),
        conversions=("converted", "sum"),
    ).reset_index()

    treatment_agg = (
        df[df["treatment"] == 1]
        .groupby("date")
        .agg(
            treatment_signups=("signed_up", "sum"),
            treatment_activations=("activated", "sum"),
        )
        .reset_index()
    )
    control_agg = (
        df[df["treatment"] == 0]
        .groupby("date")
        .agg(
            control_signups=("signed_up", "sum"),
            control_activations=("activated", "sum"),
        )
        .reset_index()
    )

    summary = agg.merge(treatment_agg, on="date", how="left")
    summary = summary.merge(control_agg, on="date", how="left")

    summary["activation_rate"] = summary["activations"] / summary["signups"].clip(lower=1)
    summary["treatment_activation_rate"] = (
        summary["treatment_activations"] / summary["treatment_signups"].clip(lower=1)
    )
    summary["control_activation_rate"] = (
        summary["control_activations"] / summary["control_signups"].clip(lower=1)
    )

    return summary.fillna(0)


def seed_duckdb(df: pd.DataFrame, db_path: str) -> None:
    """
    Write users, events, and daily_summary tables to DuckDB.

    Args:
        df: Wide-format user DataFrame.
        db_path: Path to DuckDB file. Created if it doesn't exist.
    """
    logger.info("Seeding DuckDB at %s", db_path)
    events_df = _build_events(df)
    daily_df = _build_daily_summary(df)

    with duckdb.connect(db_path) as conn:
        # Drop and recreate tables for idempotent re-seeding
        conn.execute("DROP TABLE IF EXISTS users")
        conn.execute("DROP TABLE IF EXISTS events")
        conn.execute("DROP TABLE IF EXISTS daily_summary")

        # users table — wide format, all columns
        conn.execute("""
            CREATE TABLE users AS
            SELECT
                user_id,
                CAST(impression_date AS DATE)  AS impression_date,
                company_size,
                industry,
                channel,
                CAST(treatment AS INTEGER)     AS treatment,
                CAST(clicked AS INTEGER)        AS clicked,
                CAST(signed_up AS INTEGER)      AS signed_up,
                CAST(activated AS INTEGER)      AS activated,
                CAST(converted AS INTEGER)      AS converted,
                CAST(activation_prob AS DOUBLE) AS activation_prob,
                CAST(pre_activation_rate AS DOUBLE) AS pre_activation_rate,
                CAST(revenue AS DOUBLE)         AS revenue,
                CAST(click_date AS DATE)        AS click_date,
                CAST(signup_date AS DATE)       AS signup_date,
                CAST(activation_date AS DATE)   AS activation_date,
                CAST(conversion_date AS DATE)   AS conversion_date,
                days_to_click,
                days_to_signup,
                days_to_activation,
                days_to_conversion
            FROM df
        """)

        # events table — long format
        conn.execute("""
            CREATE TABLE events AS
            SELECT
                user_id,
                stage,
                CAST(event_date AS DATE) AS event_date,
                company_size,
                industry,
                channel,
                CAST(treatment AS INTEGER) AS treatment,
                CAST(revenue AS DOUBLE)    AS revenue
            FROM events_df
        """)

        # daily_summary — pre-aggregated
        conn.execute("""
            CREATE TABLE daily_summary AS
            SELECT
                CAST(date AS DATE)              AS date,
                CAST(impressions AS INTEGER)    AS impressions,
                CAST(clicks AS INTEGER)         AS clicks,
                CAST(signups AS INTEGER)        AS signups,
                CAST(activations AS INTEGER)    AS activations,
                CAST(conversions AS INTEGER)    AS conversions,
                CAST(treatment_activations AS INTEGER)  AS treatment_activations,
                CAST(control_activations AS INTEGER)    AS control_activations,
                CAST(activation_rate AS DOUBLE)         AS activation_rate,
                CAST(treatment_activation_rate AS DOUBLE) AS treatment_activation_rate,
                CAST(control_activation_rate AS DOUBLE)   AS control_activation_rate
            FROM daily_df
        """)

        n_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        n_days = conn.execute("SELECT COUNT(*) FROM daily_summary").fetchone()[0]
        logger.info(
            "DuckDB seeded — users: %d | events: %d | daily_summary rows: %d",
            n_users, n_events, n_days,
        )


def seed_sqlite(sqlite_path: str) -> None:
    """
    Create SQLite log tables (experiment_logs, outreach_log, api_usage).

    Idempotent — safe to re-run; existing tables are preserved.

    Args:
        sqlite_path: Path to SQLite file. Created if it doesn't exist.
    """
    logger.info("Seeding SQLite at %s", sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiment_logs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          TEXT NOT NULL,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                company_size    TEXT,
                channel         TEXT,
                n_treatment     INTEGER,
                n_control       INTEGER,
                baseline_rate   REAL,
                ate             REAL,
                ate_se          REAL,
                p_value         REAL,
                ci_lower        REAL,
                ci_upper        REAL,
                variance_reduction_pct REAL,
                srm_detected    INTEGER,
                srm_p_value     REAL,
                method          TEXT,
                notes           TEXT
            );

            CREATE TABLE IF NOT EXISTS outreach_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL DEFAULT 'demo',
                sent_at         TEXT NOT NULL DEFAULT (datetime('now')),
                segment_id      TEXT NOT NULL,
                company_size    TEXT,
                industry        TEXT,
                channel         TEXT,
                cate_estimate   REAL,
                tone            TEXT,
                subject_hash    TEXT,
                body_hash       TEXT,
                is_holdout      INTEGER DEFAULT 0,
                reply           INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS api_usage (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                called_at       TEXT NOT NULL DEFAULT (datetime('now')),
                endpoint        TEXT NOT NULL,
                model           TEXT,
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                cost_usd        REAL
            );

            CREATE TABLE IF NOT EXISTS contacts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id       TEXT NOT NULL DEFAULT 'demo',
                email           TEXT NOT NULL,
                first_name      TEXT,
                company         TEXT,
                company_size    TEXT,
                channel         TEXT,
                industry        TEXT,
                uploaded_at     TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(tenant_id, email)
            );

            CREATE TABLE IF NOT EXISTS contact_sends (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id           TEXT NOT NULL DEFAULT 'demo',
                sent_at             TEXT NOT NULL DEFAULT (datetime('now')),
                contact_id          INTEGER NOT NULL,
                email               TEXT NOT NULL,
                segment_id          TEXT NOT NULL,
                company_size        TEXT,
                channel             TEXT,
                cate_estimate       REAL,
                tone                TEXT,
                subject             TEXT NOT NULL,
                body                TEXT NOT NULL,
                cta                 TEXT,
                is_holdout          INTEGER NOT NULL DEFAULT 0,
                provider_message_id TEXT,
                status              TEXT NOT NULL DEFAULT 'sent',
                activated_at        TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_contacts_tenant     ON contacts(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_contact_sends_tenant ON contact_sends(tenant_id);
            CREATE INDEX IF NOT EXISTS idx_outreach_log_tenant  ON outreach_log(tenant_id);
        """)
        conn.commit()
    finally:
        conn.close()
    logger.info("SQLite schema ready")


def migrate_sqlite(sqlite_path: str) -> None:
    """
    Idempotent migration: add tenant_id and activated_at columns to existing
    tables that predate multi-tenancy support.

    Safe to run on every startup — ALTER TABLE is a no-op if the column
    already exists (OperationalError is caught and ignored).

    Args:
        sqlite_path: Path to the SQLite file to migrate.
    """
    migrations = [
        ("contacts",      "tenant_id TEXT NOT NULL DEFAULT 'demo'"),
        ("contacts",      "UNIQUE(tenant_id, email)"),  # constraints can't be added via ALTER
        ("contact_sends", "tenant_id TEXT NOT NULL DEFAULT 'demo'"),
        ("contact_sends", "activated_at TEXT"),
        ("outreach_log",  "tenant_id TEXT NOT NULL DEFAULT 'demo'"),
        ("experiment_logs", "tenant_id TEXT NOT NULL DEFAULT 'demo'"),
        ("api_usage",     "tenant_id TEXT NOT NULL DEFAULT 'demo'"),
    ]
    conn = sqlite3.connect(sqlite_path)
    try:
        # Ensure contact_sends exists — it may be missing on DBs created before
        # multi-tenancy was added. init_sqlite_db() creates it on fresh installs;
        # this covers stale installs that predate the table.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS contact_sends (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id           TEXT    NOT NULL DEFAULT 'demo',
                email               TEXT    NOT NULL,
                contact_id          INTEGER,
                segment_id          TEXT,
                company_size        TEXT,
                channel             TEXT,
                cate_estimate       REAL,
                tone                TEXT,
                subject             TEXT,
                body                TEXT,
                cta                 TEXT,
                is_holdout          INTEGER NOT NULL DEFAULT 0,
                provider_message_id TEXT,
                status              TEXT    NOT NULL DEFAULT 'sent',
                activated_at        TEXT,
                sent_at             TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

        for table, col_def in migrations:
            if "UNIQUE" in col_def:
                continue  # SQLite can't add constraints after creation
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists — idempotent
        # Add indexes if missing — table may not exist on a fresh install
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_contacts_tenant ON contacts(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_contact_sends_tenant ON contact_sends(tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_outreach_log_tenant ON outreach_log(tenant_id)",
        ]:
            try:
                conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass  # table doesn't exist yet — seed_db will create it
        conn.commit()
    finally:
        conn.close()
    logger.info("SQLite migration complete")


# Common column name aliases → canonical name.
# Keys are canonical; values are accepted alternates (case-insensitive).
_COLUMN_ALIASES: dict[str, list[str]] = {
    "user_id":      ["user", "id", "userid", "uid", "customer_id", "contact_id", "record_id"],
    "treatment":    ["is_treatment", "in_treatment", "group", "variant", "treated",
                     "sent", "outreach", "experiment_group", "arm"],
    "activated":    ["converted", "outcome", "activation", "is_activated", "is_converted",
                     "churned", "purchased", "subscribed"],
    "company_size": ["company_type", "size", "account_size", "tier", "segment"],
    "channel":      ["source", "acquisition_channel", "utm_source", "traffic_source",
                     "marketing_channel"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename recognised column aliases to their canonical names.

    Only renames when the canonical column is absent — never overwrites
    an existing canonical column.

    Args:
        df: Raw DataFrame from the user's upload.

    Returns:
        DataFrame with aliased columns renamed; others unchanged.
    """
    lower_to_original = {c.lower().strip(): c for c in df.columns}
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower_to_original:
                rename_map[lower_to_original[alias.lower()]] = canonical
                break
    return df.rename(columns=rename_map) if rename_map else df


def seed_tenant_duckdb(df: pd.DataFrame, db_path: str) -> dict:
    """
    Seed a tenant-specific DuckDB from a user-uploaded wide-format DataFrame.

    Accepts the minimum viable columns needed for CATE estimation:
        Required: user_id (str), treatment (0/1), activated (0/1)
        Optional: company_size, channel, industry, impression_date,
                  signup_date, activation_date, revenue, clicked, signed_up

    Common column aliases are normalised automatically (e.g. "converted"
    becomes "activated", "variant" becomes "treatment").

    Missing optional columns are filled with sensible defaults so the
    existing seed_duckdb() function can process the data without changes.

    Args:
        df:       Wide-format user DataFrame with at least user_id, treatment, activated.
        db_path:  Path to the tenant's DuckDB file.

    Returns:
        Summary dict with n_users, n_treatment, n_control, activation_rate.

    Raises:
        ValueError: If required columns are missing, data is empty, or there
                    is no control group.
    """
    df = _normalize_columns(df.copy())

    required = {"user_id", "treatment", "activated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Upload is missing required columns: {sorted(missing)}. "
            "Expected: user_id (unique user identifier), treatment (1=treated / 0=control), "
            "activated (1=activated / 0=not). Common aliases are accepted automatically."
        )
    if len(df) == 0:
        raise ValueError("Uploaded file contains no data rows.")

    df = df.copy()

    # Coerce required columns
    df["treatment"] = pd.to_numeric(df["treatment"], errors="coerce").fillna(0).astype(int)
    df["activated"] = pd.to_numeric(df["activated"], errors="coerce").fillna(0).astype(int)

    # Verify both arms exist — CATE estimation requires treated AND control users
    n_treated  = int(df["treatment"].sum())
    n_control  = int((df["treatment"] == 0).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            f"Upload needs both treated (treatment=1) and control (treatment=0) users. "
            f"Found {n_treated} treated and {n_control} control. "
            "If you haven't run an A/B test, set treatment=1 for users who received "
            "outreach and treatment=0 for a random holdout (~20%)."
        )

    # Fill optional segment columns
    for col, default in [("company_size", "unknown"), ("channel", "unknown"), ("industry", "unknown")]:
        if col not in df.columns:
            df[col] = default

    # Fill optional funnel columns
    today = pd.Timestamp.now().normalize()
    for col in ["impression_date", "signup_date", "click_date", "activation_date", "conversion_date"]:
        if col not in df.columns:
            df[col] = today

    for col in ["clicked", "signed_up"]:
        if col not in df.columns:
            df[col] = 1  # assume all users reached signup for CUPED purposes

    if "converted" not in df.columns:
        df["converted"] = 0
    if "revenue" not in df.columns:
        df["revenue"] = 0.0

    # pre_activation_rate: use the dataset mean as a rough historical proxy.
    # In a real product, this would be computed from a pre-experiment window.
    # As a constant covariate it gives zero CUPED variance reduction, which
    # is correct behaviour — CUPED requires genuine pre-period data to help.
    mean_act = float(df["activated"].mean())
    df["pre_activation_rate"] = mean_act
    df["activation_prob"] = df["activated"].astype(float)

    # Add timing columns expected by seed_duckdb
    for col in ["days_to_click", "days_to_signup", "days_to_activation", "days_to_conversion"]:
        if col not in df.columns:
            df[col] = 0

    seed_duckdb(df, db_path)

    return {
        "n_users":          len(df),
        "n_treatment":      int(df["treatment"].sum()),
        "n_control":        int((df["treatment"] == 0).sum()),
        "activation_rate":  round(mean_act, 4),
    }


def validate_seeded_data(db_path: str) -> None:
    """
    Run basic sanity checks on the seeded DuckDB data.

    Logs warnings for any distribution anomalies.

    Args:
        db_path: Path to DuckDB file.
    """
    with duckdb.connect(db_path) as conn:
        rates = conn.execute("""
            SELECT
                AVG(clicked)    AS click_rate,
                AVG(CASE WHEN clicked    = 1 THEN CAST(signed_up  AS FLOAT) END) AS signup_rate,
                AVG(CASE WHEN signed_up  = 1 THEN CAST(activated  AS FLOAT) END) AS activation_rate,
                AVG(CASE WHEN activated  = 1 THEN CAST(converted  AS FLOAT) END) AS conversion_rate
            FROM users
        """).fetchone()

        click_r, signup_r, activation_r, conversion_r = rates

        if not (0.50 <= click_r <= 0.85):
            logger.warning("Click rate %.3f outside expected range [0.50, 0.85]", click_r)
        if not (0.45 <= signup_r <= 0.80):
            logger.warning("Setup completion rate %.3f outside expected range [0.45, 0.80]", signup_r)
        if not (0.28 <= activation_r <= 0.65):
            logger.warning("Activation rate %.3f outside expected range [0.28, 0.65]", activation_r)
        if not (0.15 <= conversion_r <= 0.50):
            logger.warning("Conversion rate %.3f outside expected range [0.15, 0.50]", conversion_r)

        logger.info(
            "Funnel conditional rates — click: %.3f | signup|click: %.3f | "
            "activation|signup: %.3f | conversion|activation: %.3f",
            click_r, signup_r, activation_r, conversion_r,
        )

        balance = conn.execute("""
            SELECT
                AVG(treatment)                         AS overall_split,
                AVG(CASE WHEN company_size = 'enterprise' THEN treatment END) AS enterprise_split,
                AVG(CASE WHEN company_size = 'mid_market'  THEN treatment END) AS midmarket_split,
                AVG(CASE WHEN company_size = 'SMB'         THEN treatment END) AS smb_split
            FROM users
        """).fetchone()

        for label, val in zip(["overall", "enterprise", "mid_market", "SMB"], balance):
            if val is not None and not (0.45 <= val <= 0.55):
                logger.warning("Treatment split %s=%.3f deviates from 0.50", label, val)

        logger.info(
            "Treatment splits — overall: %.3f | enterprise: %.3f | mid_market: %.3f | SMB: %.3f",
            *balance,
        )

        seg_rates = conn.execute("""
            SELECT
                company_size,
                channel,
                AVG(CASE WHEN treatment = 1 THEN CAST(activated AS FLOAT) END) AS treatment_act,
                AVG(CASE WHEN treatment = 0 THEN CAST(activated AS FLOAT) END) AS control_act,
                COUNT(*) AS n
            FROM users
            WHERE signed_up = 1
            GROUP BY company_size, channel
            ORDER BY company_size, channel
        """).fetchall()

        logger.info("Segment-level activation rates (treatment vs. control):")
        for sz, ch, t_rate, c_rate, n in seg_rates:
            if t_rate is not None and c_rate is not None:
                observed_lift = t_rate - c_rate
                logger.info(
                    "  %-12s %-12s | T=%.3f C=%.3f | lift=%.3f | n=%d",
                    sz, ch, t_rate, c_rate, observed_lift, n,
                )


def main() -> None:
    """Entry point: generate data, seed DuckDB and SQLite, validate."""
    db_path = os.getenv("DATABASE_URL", _DEFAULT_DB_URL)
    sqlite_path = os.getenv("SQLITE_PATH", _DEFAULT_SQLITE)
    n_users = int(os.getenv("N_USERS", "50000"))
    seed = int(os.getenv("SEED", "42"))

    # Ensure directories exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d users (seed=%d)...", n_users, seed)
    df = generate_funnel_data(n_users=n_users, seed=seed)

    seed_duckdb(df, db_path)
    seed_sqlite(sqlite_path)
    validate_seeded_data(db_path)

    logger.info("Seeding complete. Ready to start the API.")


if __name__ == "__main__":
    main()
