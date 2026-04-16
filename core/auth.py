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

import logging
import os
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
