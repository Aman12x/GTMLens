"""
Authentication core: password hashing, JWT creation/verification, user persistence.

All business logic lives here. The API layer (api/routes/auth.py) delegates to
these functions and catches AuthError to return appropriate HTTP responses.

Storage: SQLite at AUTH_DB_PATH (env var, default data/auth.db).
Schema: single 'users' table — id, email, hashed_password, created_at, is_active.

JWT: HS256 signed access tokens. Secret read from JWT_SECRET_KEY env var.
     Tokens carry {sub: email, exp: unix timestamp}.
     No refresh tokens in this phase — access tokens expire after
     ACCESS_TOKEN_EXPIRE_MINUTES (default 60).
"""

import hashlib
import logging
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import bcrypt
from jose import JWTError, jwt

# Resolve default path relative to this file so auth.db is found regardless
# of the working directory uvicorn is started from.
_PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all from environment — no hardcoded values)
# ---------------------------------------------------------------------------

def _require_env(name: str, default: str | None = None) -> str:
    """
    Read an env var, raising a clear error if it is absent and no default given.

    Args:
        name:    Environment variable name.
        default: Fallback value (should only be used for non-secret config).

    Returns:
        The env var value or default.

    Raises:
        RuntimeError: If the var is unset and no default was supplied.
    """
    val = os.getenv(name, default)
    if val is None:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            f"Add it to your .env file before starting the server."
        )
    return val


def _auth_db_path() -> str:
    return os.getenv(
        "AUTH_DB_PATH",
        str(_PROJECT_ROOT / "data" / "auth.db"),
    )


def _jwt_secret() -> str:
    secret = _require_env("JWT_SECRET_KEY")
    if len(secret) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY is too short — minimum 32 characters required. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return secret


def _access_token_ttl_minutes() -> int:
    return int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


ALGORITHM = "HS256"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AuthError(ValueError):
    """
    Raised for all authentication domain errors.

    Attributes:
        message: Human-readable description safe to surface in the API response.
        status_code: Suggested HTTP status code (401, 409, etc.).
    """

    def __init__(self, message: str, status_code: int = 401) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


@contextmanager
def _db_conn() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that opens, commits, and closes a SQLite connection.

    On exception the transaction is rolled back automatically by sqlite3.
    """
    path = _auth_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_auth_db() -> None:
    """
    Create the users table if it does not exist.

    Idempotent — safe to call on every startup.
    """
    with _db_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                email           TEXT    NOT NULL UNIQUE COLLATE NOCASE,
                hashed_password TEXT    NOT NULL,
                created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active       INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT    NOT NULL COLLATE NOCASE,
                key_hash   TEXT    NOT NULL UNIQUE,
                key_prefix TEXT    NOT NULL,
                label      TEXT    NOT NULL DEFAULT '',
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                is_active  INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_api_keys_email ON api_keys (user_email)"
        )
    logger.info("Auth DB ready at %s", _auth_db_path())


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """
    Hash a plaintext password using bcrypt.

    Args:
        plain: Plaintext password from the registration form.

    Returns:
        bcrypt hash string safe to store in the database.
    """
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.

    Args:
        plain:  Plaintext password from the login form.
        hashed: Hash retrieved from the database.

    Returns:
        True if the password matches, False otherwise.
    """
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_access_token(email: str) -> str:
    """
    Create a signed JWT access token for the given email address.

    Args:
        email: The authenticated user's email (becomes the token subject).

    Returns:
        Signed JWT string.
    """
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=_access_token_ttl_minutes())
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """
    Decode and validate a JWT access token, returning the subject (email).

    Args:
        token: JWT string from the Authorization header.

    Returns:
        Email address stored in the token subject claim.

    Raises:
        AuthError: If the token is expired, malformed, or missing the subject claim.
    """
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[ALGORITHM])
    except JWTError as exc:
        raise AuthError(f"Invalid or expired token: {exc}") from exc

    email: str | None = payload.get("sub")
    if not email:
        raise AuthError("Token is missing subject claim.")
    return email


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------


def register_user(email: str, plain_password: str) -> dict:
    """
    Create a new user account.

    Args:
        email:          User's email address (case-insensitive unique key).
        plain_password: Plaintext password; will be hashed before storage.

    Returns:
        Dict with id, email, created_at fields for the new user.

    Raises:
        AuthError: If the email is already registered (status_code=409).
        ValueError: If email or password is empty.
    """
    if not email or not plain_password:
        raise ValueError("Email and password must not be empty.")

    hashed = hash_password(plain_password)

    try:
        with _db_conn() as conn:
            cursor = conn.execute(
                "INSERT INTO users (email, hashed_password) VALUES (?, ?)",
                (email.lower().strip(), hashed),
            )
            user_id = cursor.lastrowid
            row = conn.execute(
                "SELECT id, email, created_at, is_active FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
    except sqlite3.IntegrityError:
        raise AuthError(
            f"Email '{email}' is already registered.",
            status_code=409,
        )

    logger.info("Registered new user id=%d email=%s", row["id"], row["email"])
    return dict(row)


def authenticate_user(email: str, plain_password: str) -> dict:
    """
    Verify credentials and return the user record.

    Args:
        email:          Submitted email address.
        plain_password: Submitted plaintext password.

    Returns:
        Dict with id, email, created_at, is_active fields.

    Raises:
        AuthError: If email is not found or password is wrong (status_code=401).
                   Uses the same message for both cases to avoid user enumeration.
    """
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, email, hashed_password, created_at, is_active "
            "FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip(),),
        ).fetchone()

    _INVALID_MSG = "Invalid email or password."

    if row is None:
        raise AuthError(_INVALID_MSG)

    if not verify_password(plain_password, row["hashed_password"]):
        raise AuthError(_INVALID_MSG)

    if not row["is_active"]:
        raise AuthError("This account has been deactivated.", status_code=403)

    return {k: row[k] for k in ("id", "email", "created_at", "is_active")}


def get_user_by_email(email: str) -> dict | None:
    """
    Fetch a user record by email address.

    Args:
        email: Email to look up (case-insensitive).

    Returns:
        Dict with id, email, created_at, is_active — or None if not found.
    """
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT id, email, created_at, is_active FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip(),),
        ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# API keys (machine-to-machine auth — e.g. Clay HTTP API enrichment columns)
#
# Keys are 256-bit random tokens with a "gtml_" prefix. We store only a
# SHA-256 hash: the token has full entropy, so a fast hash is sufficient
# (bcrypt is for low-entropy passwords) and keeps per-request lookup cheap —
# Clay fires one request per table row.
# ---------------------------------------------------------------------------

_API_KEY_PREFIX = "gtml_"


def _hash_api_key(key: str) -> str:
    """Return the SHA-256 hex digest of an API key string."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_api_key(email: str, label: str = "") -> dict:
    """
    Mint a new API key for an existing user.

    The plaintext key is returned exactly once and never stored — only its
    SHA-256 hash is persisted.

    Args:
        email: Email of the key owner (must be a registered, active user).
        label: Optional human-readable label (e.g. "clay-outbound-table").

    Returns:
        Dict with keys:
            api_key    — plaintext key (show once, then discard)
            key_prefix — first 12 chars, for identifying the key later
            label      — echoed label
            created_at — creation timestamp

    Raises:
        AuthError: If the email does not belong to a registered active user.
    """
    user = get_user_by_email(email)
    if user is None or not user["is_active"]:
        raise AuthError("No active account for this email.", status_code=401)

    key = _API_KEY_PREFIX + secrets.token_urlsafe(32)
    key_prefix = key[:12]

    with _db_conn() as conn:
        conn.execute(
            "INSERT INTO api_keys (user_email, key_hash, key_prefix, label) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), _hash_api_key(key), key_prefix, label),
        )
        row = conn.execute(
            "SELECT created_at FROM api_keys WHERE key_hash = ?",
            (_hash_api_key(key),),
        ).fetchone()

    logger.info("Minted API key prefix=%s for email=%s label=%s", key_prefix, email, label)
    return {
        "api_key":    key,
        "key_prefix": key_prefix,
        "label":      label,
        "created_at": row["created_at"],
    }


def get_user_for_api_key(key: str) -> dict | None:
    """
    Resolve an API key to its owning user record.

    Args:
        key: Plaintext API key from the X-API-Key header.

    Returns:
        User record dict (id, email, created_at, is_active) if the key is
        valid, active, and belongs to an active user — otherwise None.
        Never raises: invalid keys degrade to unauthenticated (demo tenant).
    """
    if not key or not key.startswith(_API_KEY_PREFIX):
        return None

    with _db_conn() as conn:
        row = conn.execute(
            "SELECT user_email FROM api_keys WHERE key_hash = ? AND is_active = 1",
            (_hash_api_key(key),),
        ).fetchone()

    if row is None:
        return None

    user = get_user_by_email(row["user_email"])
    if user is None or not user["is_active"]:
        return None
    return user


def list_api_keys(email: str) -> list[dict]:
    """
    List a user's API keys (prefixes only — plaintext is never recoverable).

    Args:
        email: Key owner's email.

    Returns:
        List of dicts: key_prefix, label, created_at, is_active.
    """
    with _db_conn() as conn:
        rows = conn.execute(
            "SELECT key_prefix, label, created_at, is_active "
            "FROM api_keys WHERE user_email = ? COLLATE NOCASE ORDER BY created_at DESC",
            (email.strip(),),
        ).fetchall()
    return [dict(r) for r in rows]


def revoke_api_key(email: str, key_prefix: str) -> bool:
    """
    Deactivate an API key identified by its prefix, scoped to the owner.

    Args:
        email:      Key owner's email (prevents cross-tenant revocation).
        key_prefix: The 12-char prefix returned at mint time.

    Returns:
        True if a key was revoked, False if no matching active key exists.
    """
    with _db_conn() as conn:
        cursor = conn.execute(
            "UPDATE api_keys SET is_active = 0 "
            "WHERE user_email = ? COLLATE NOCASE AND key_prefix = ? AND is_active = 1",
            (email.strip(), key_prefix),
        )
        revoked = cursor.rowcount > 0
    if revoked:
        logger.info("Revoked API key prefix=%s for email=%s", key_prefix, email)
    return revoked
