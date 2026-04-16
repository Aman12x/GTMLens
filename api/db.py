"""
Shared DuckDB connection helper for API route handlers.

All routes use this context manager so the connection lifetime is
tied to the request — no global connection state.

Path resolution: DATABASE_URL env var takes precedence; the default is
resolved relative to this file so the path is correct regardless of
which directory uvicorn is started from.
"""

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import duckdb

# Resolve default path relative to this file so it is correct no matter
# what the current working directory is when uvicorn starts.
_DEFAULT_DB = str(Path(__file__).parent.parent / "data" / "gtmlens.duckdb")
_TENANTS_DIR = Path(__file__).parent.parent / "data" / "tenants"


def _db_url() -> str:
    return os.getenv("DATABASE_URL", _DEFAULT_DB)


def _tenant_db_path(tenant_id: str) -> str:
    """
    Return the DuckDB file path for a given tenant.

    The demo tenant always uses the shared synthetic database so the
    portfolio demo continues to work without authentication.

    Real tenants get an isolated file under data/tenants/ keyed by a
    SHA-256 hash of their email — avoids filesystem-unsafe characters.
    """
    if tenant_id == "demo":
        return _db_url()
    safe = hashlib.sha256(tenant_id.encode()).hexdigest()[:24]
    _TENANTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(_TENANTS_DIR / f"{safe}.duckdb")


@contextmanager
def get_conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Open a read-write DuckDB connection scoped to a single request.

    Usage:
        with get_conn() as conn:
            df = conn.execute("SELECT ...", [param]).df()
    """
    conn = duckdb.connect(_db_url())
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_tenant_conn(tenant_id: str) -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """
    Open a DuckDB connection scoped to a specific tenant.

    Routes that support both demo and real tenants should use this instead
    of get_conn(). The demo tenant resolves to the shared synthetic database;
    real tenants get their own isolated file.

    Args:
        tenant_id: "demo" or a user email string.

    Usage:
        with get_tenant_conn(tenant_id) as conn:
            df = conn.execute("SELECT ...").df()
    """
    path = _tenant_db_path(tenant_id)
    conn = duckdb.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def tenant_has_data(tenant_id: str) -> bool:
    """
    Return True if the tenant's DuckDB exists and has a users table with rows.

    Used by the frontend status endpoint to show "upload your data" vs
    "using demo data" messaging.
    """
    if tenant_id == "demo":
        return True
    path = _tenant_db_path(tenant_id)
    if not Path(path).exists():
        return False
    try:
        conn = duckdb.connect(path)
        n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        return n > 0
    except Exception:
        return False
