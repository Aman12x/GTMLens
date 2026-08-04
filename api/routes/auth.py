"""
Auth routes — all mounted under /api/alpha/auth/

POST   /api/alpha/auth/register             — create account
POST   /api/alpha/auth/login                — exchange credentials for JWT (OAuth2 password flow)
GET    /api/alpha/auth/me                   — return the authenticated user's profile
POST   /api/alpha/auth/api-key              — mint an API key for machine callers (e.g. Clay)
GET    /api/alpha/auth/api-keys             — list the caller's API keys (prefixes only)
DELETE /api/alpha/auth/api-key/{key_prefix} — revoke an API key
"""

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Annotated
from fastapi import Depends

from api.deps import CurrentUser
from core.auth import (
    AuthError,
    authenticate_user,
    create_access_token,
    create_api_key,
    list_api_keys,
    register_user,
    revoke_api_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str
    created_at: str
    is_active: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new GTMLens account",
)
def register(body: RegisterRequest) -> dict:
    """
    Register a new user with email and password.

    Returns the created user profile. Does not issue a token — clients
    must call /login after registration.

    Raises:
        409 if the email is already taken.
        422 if the request body fails validation (Pydantic handles this).
    """
    try:
        user = register_user(email=str(body.email), plain_password=body.password)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    logger.info("POST /register | email=%s", body.email)
    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Obtain a Bearer token (OAuth2 password flow)",
)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> dict:
    """
    Exchange email + password for a JWT access token.

    Accepts standard OAuth2 password form fields (username = email, password).
    The returned token must be sent as `Authorization: Bearer <token>` on
    all protected /api/alpha/ endpoints.

    Raises:
        401 if credentials are invalid.
        403 if the account is deactivated.
    """
    try:
        authenticate_user(email=form.username, plain_password=form.password)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    token = create_access_token(email=form.username.lower().strip())
    logger.info("POST /login | email=%s", form.username)
    return {"access_token": token, "token_type": "bearer"}


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Return the authenticated user's profile",
)
def me(user: CurrentUser) -> dict:
    """
    Return the profile of the currently authenticated user.

    Requires a valid Bearer token in the Authorization header.
    """
    return user


# ---------------------------------------------------------------------------
# API keys — machine-to-machine auth for Clay HTTP columns and similar callers
# ---------------------------------------------------------------------------


class ApiKeyMintRequest(BaseModel):
    label: str = Field("", max_length=100, description="Human-readable label, e.g. 'clay-outbound-table'")


class ApiKeyMintResponse(BaseModel):
    api_key: str = Field(..., description="Plaintext key — shown once, store it now")
    key_prefix: str
    label: str
    created_at: str


class ApiKeyInfo(BaseModel):
    key_prefix: str
    label: str
    created_at: str
    is_active: bool


class ApiKeyListResponse(BaseModel):
    keys: list[ApiKeyInfo]


@router.post(
    "/api-key",
    response_model=ApiKeyMintResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Mint an API key for machine callers (e.g. Clay HTTP API columns)",
)
def mint_api_key(body: ApiKeyMintRequest, user: CurrentUser) -> dict:
    """
    Create a new API key tied to the authenticated user's tenant.

    The plaintext key is returned once and never stored — configure it as the
    X-API-Key header in Clay's HTTP API column and discard it from the UI.
    """
    try:
        result = create_api_key(email=user["email"], label=body.label)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    logger.info("POST /api-key | email=%s prefix=%s", user["email"], result["key_prefix"])
    return result


@router.get(
    "/api-keys",
    response_model=ApiKeyListResponse,
    summary="List the caller's API keys (prefixes only)",
)
def get_api_keys(user: CurrentUser) -> dict:
    """List all API keys owned by the authenticated user. Plaintext is never recoverable."""
    return {"keys": list_api_keys(user["email"])}


@router.delete(
    "/api-key/{key_prefix}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke an API key by its prefix",
)
def delete_api_key(key_prefix: str, user: CurrentUser) -> None:
    """
    Deactivate an API key. Scoped to the caller's own keys — revoking another
    tenant's prefix returns 404.
    """
    if not revoke_api_key(email=user["email"], key_prefix=key_prefix):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Key not found", "detail": f"No active key with prefix '{key_prefix}'."},
        )
