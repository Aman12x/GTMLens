"""
FastAPI shared dependencies.

Import these with Depends() in route handlers:

    from api.deps import CurrentUser

    @router.get("/me")
    def me(user: CurrentUser) -> dict:
        return user
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from core.auth import AuthError, decode_access_token, get_user_by_email

# Tokenurl must match the login route exactly
_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/alpha/auth/login")

# Optional variant — returns None instead of 401 when no token is present.
# Used by routes that serve unauthenticated demo traffic AND real tenants.
_oauth2_optional = OAuth2PasswordBearer(tokenUrl="/api/alpha/auth/login", auto_error=False)


def get_current_user(token: Annotated[str, Depends(_oauth2_scheme)]) -> dict:
    """
    FastAPI dependency that extracts and validates the Bearer JWT.

    Raises HTTP 401 if the token is missing, expired, or invalid.
    Raises HTTP 401 if the user no longer exists in the database.
    Raises HTTP 403 if the account is deactivated.

    Returns:
        User record dict: id, email, created_at, is_active.
    """
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        email = decode_access_token(token)
    except AuthError as exc:
        if exc.status_code == 403:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message)
        raise credentials_exc from exc

    user = get_user_by_email(email)
    if user is None:
        raise credentials_exc

    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    return user


def get_optional_user(token: str | None = Depends(_oauth2_optional)) -> dict | None:
    """
    FastAPI dependency that returns the authenticated user or None.

    Routes that use this dependency serve both unauthenticated demo traffic
    (tenant_id = "demo", synthetic data) and real tenants (tenant_id = email).

    Never raises — invalid or missing tokens silently return None.
    """
    if not token:
        return None
    try:
        email = decode_access_token(token)
        user = get_user_by_email(email)
        if user and user["is_active"]:
            return user
    except Exception:
        pass
    return None


def tenant_id_from(user: dict | None) -> str:
    """
    Derive the tenant identifier from an optional user record.

    Unauthenticated requests (user=None) land on the "demo" tenant which
    always uses the synthetic DuckDB seeded by data/seed_db.py.
    Authenticated requests get their own isolated DuckDB and SQLite rows.

    Args:
        user: User record dict from get_optional_user(), or None.

    Returns:
        Tenant identifier string — the user's email, or "demo".
    """
    return user["email"] if user else "demo"


# Annotated aliases — use these in route signatures for clean type hints
CurrentUser = Annotated[dict, Depends(get_current_user)]
OptionalUser = Annotated[dict | None, Depends(get_optional_user)]
