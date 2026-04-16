"""
Tests for core/auth.py and api/routes/auth.py.

Coverage:
    core/auth.py:
        - hash_password produces a non-plaintext hash
        - verify_password: correct password passes, wrong password fails
        - create_access_token returns a decodable JWT with correct subject
        - decode_access_token returns email for a valid token
        - decode_access_token raises AuthError on tampered token
        - decode_access_token raises AuthError on expired token
        - register_user creates a user and returns id/email/created_at
        - register_user raises AuthError(409) on duplicate email
        - register_user raises ValueError on empty email/password
        - authenticate_user returns user dict on correct credentials
        - authenticate_user raises AuthError on wrong password
        - authenticate_user raises AuthError on unknown email
        - get_user_by_email returns dict for existing user
        - get_user_by_email returns None for unknown email

    api/routes/auth.py (via TestClient):
        - POST /register 201 on valid payload
        - POST /register 409 on duplicate email
        - POST /register 422 on short password
        - POST /register 422 on invalid email
        - POST /login 200 returns access_token on valid credentials
        - POST /login 401 on wrong password
        - POST /login 401 on unknown email
        - GET /me 200 returns user profile with valid token
        - GET /me 401 with no token
        - GET /me 401 with tampered token
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt

# ---------------------------------------------------------------------------
# Fixtures — isolated temp DB per test session so tests don't interfere
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def isolated_auth_db(tmp_path_factory):
    """
    Point AUTH_DB_PATH at a fresh temp file for the entire test module.
    Also sets JWT_SECRET_KEY so core/auth.py doesn't raise RuntimeError.
    Calls init_auth_db() so the users table exists before any unit test runs.
    """
    db_file = tmp_path_factory.mktemp("auth") / "test_auth.db"
    os.environ["AUTH_DB_PATH"] = str(db_file)
    os.environ["JWT_SECRET_KEY"] = "test-secret-do-not-use-in-production"
    os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"

    # Must import after env vars are set so _auth_db_path() reads the right value
    from core.auth import init_auth_db
    init_auth_db()

    yield

    # Cleanup env vars after module
    for key in ("AUTH_DB_PATH", "JWT_SECRET_KEY", "ACCESS_TOKEN_EXPIRE_MINUTES"):
        os.environ.pop(key, None)


@pytest.fixture(scope="module")
def client(isolated_auth_db):
    """FastAPI TestClient with startup event fired (initialises DB tables)."""
    from api.main import app
    from core.auth import init_auth_db
    init_auth_db()
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# core/auth.py — password helpers
# ---------------------------------------------------------------------------


def test_hash_password_not_plaintext():
    from core.auth import hash_password
    hashed = hash_password("my-secret-pass")
    assert hashed != "my-secret-pass"
    assert hashed.startswith("$2b$")


def test_verify_password_correct():
    from core.auth import hash_password, verify_password
    hashed = hash_password("correct-horse-battery")
    assert verify_password("correct-horse-battery", hashed) is True


def test_verify_password_wrong():
    from core.auth import hash_password, verify_password
    hashed = hash_password("correct-horse-battery")
    assert verify_password("wrong-password", hashed) is False


# ---------------------------------------------------------------------------
# core/auth.py — JWT helpers
# ---------------------------------------------------------------------------


def test_create_access_token_is_string():
    from core.auth import create_access_token
    token = create_access_token("user@example.com")
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_access_token_returns_email():
    from core.auth import create_access_token, decode_access_token
    token = create_access_token("alice@example.com")
    email = decode_access_token(token)
    assert email == "alice@example.com"


def test_decode_access_token_tampered_raises():
    from core.auth import create_access_token, decode_access_token, AuthError
    token = create_access_token("alice@example.com")
    tampered = token[:-4] + "xxxx"
    with pytest.raises(AuthError):
        decode_access_token(tampered)


def test_decode_access_token_expired_raises():
    from core.auth import AuthError, ALGORITHM
    secret = os.environ["JWT_SECRET_KEY"]
    # Forge a token that is already expired
    expired_payload = {
        "sub": "alice@example.com",
        "exp": datetime.now(tz=timezone.utc) - timedelta(seconds=1),
    }
    expired_token = jwt.encode(expired_payload, secret, algorithm=ALGORITHM)
    from core.auth import decode_access_token
    with pytest.raises(AuthError, match="expired"):
        decode_access_token(expired_token)


def test_decode_access_token_missing_sub_raises():
    from core.auth import AuthError, ALGORITHM
    secret = os.environ["JWT_SECRET_KEY"]
    no_sub = jwt.encode(
        {"exp": datetime.now(tz=timezone.utc) + timedelta(hours=1)},
        secret,
        algorithm=ALGORITHM,
    )
    from core.auth import decode_access_token
    with pytest.raises(AuthError, match="subject"):
        decode_access_token(no_sub)


# ---------------------------------------------------------------------------
# core/auth.py — user operations
# ---------------------------------------------------------------------------


def test_register_user_returns_user_dict():
    from core.auth import register_user
    user = register_user("bob@example.com", "strongpass1")
    assert user["email"] == "bob@example.com"
    assert "id" in user
    assert "created_at" in user
    assert "hashed_password" not in user  # must not leak


def test_register_user_duplicate_raises_409():
    from core.auth import register_user, AuthError
    register_user("dup@example.com", "strongpass1")
    with pytest.raises(AuthError) as exc_info:
        register_user("dup@example.com", "otherpass1")
    assert exc_info.value.status_code == 409


def test_register_user_case_insensitive_email():
    from core.auth import register_user, AuthError
    register_user("CaseTest@example.com", "strongpass1")
    with pytest.raises(AuthError) as exc_info:
        register_user("casetest@example.com", "otherpass1")
    assert exc_info.value.status_code == 409


def test_register_user_empty_email_raises():
    from core.auth import register_user
    with pytest.raises(ValueError):
        register_user("", "strongpass1")


def test_register_user_empty_password_raises():
    from core.auth import register_user
    with pytest.raises(ValueError):
        register_user("empty@example.com", "")


def test_authenticate_user_correct_credentials():
    from core.auth import authenticate_user, register_user
    register_user("carol@example.com", "mypassword1")
    user = authenticate_user("carol@example.com", "mypassword1")
    assert user["email"] == "carol@example.com"
    assert "hashed_password" not in user


def test_authenticate_user_wrong_password():
    from core.auth import authenticate_user, register_user, AuthError
    register_user("dave@example.com", "rightpassword")
    with pytest.raises(AuthError) as exc_info:
        authenticate_user("dave@example.com", "wrongpassword")
    assert exc_info.value.status_code == 401


def test_authenticate_user_unknown_email():
    from core.auth import authenticate_user, AuthError
    with pytest.raises(AuthError) as exc_info:
        authenticate_user("nobody@example.com", "anypassword")
    assert exc_info.value.status_code == 401


def test_authenticate_wrong_and_unknown_same_message():
    """Must not leak whether the email exists (prevent user enumeration)."""
    from core.auth import authenticate_user, register_user, AuthError
    register_user("enumtest@example.com", "password123")
    # Python 3: exception variable is deleted after the except block exits,
    # so capture to a regular variable inside the block.
    wrong_pass_exc: AuthError | None = None
    unknown_exc: AuthError | None = None
    try:
        authenticate_user("enumtest@example.com", "wrongpass")
    except AuthError as exc:
        wrong_pass_exc = exc
    try:
        authenticate_user("notregistered@example.com", "wrongpass")
    except AuthError as exc:
        unknown_exc = exc
    assert wrong_pass_exc is not None, "Expected AuthError for wrong password"
    assert unknown_exc is not None, "Expected AuthError for unknown email"
    assert wrong_pass_exc.message == unknown_exc.message


def test_get_user_by_email_found():
    from core.auth import get_user_by_email, register_user
    register_user("eve@example.com", "password123")
    user = get_user_by_email("eve@example.com")
    assert user is not None
    assert user["email"] == "eve@example.com"


def test_get_user_by_email_not_found():
    from core.auth import get_user_by_email
    result = get_user_by_email("ghost@example.com")
    assert result is None


# ---------------------------------------------------------------------------
# api/routes/auth.py — POST /register
# ---------------------------------------------------------------------------


def test_register_endpoint_201(client: TestClient):
    resp = client.post(
        "/api/alpha/auth/register",
        json={"email": "frank@example.com", "password": "securepass1"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "frank@example.com"
    assert "id" in body
    assert "access_token" not in body  # register must not return a token


def test_register_endpoint_409_duplicate(client: TestClient):
    client.post(
        "/api/alpha/auth/register",
        json={"email": "grace@example.com", "password": "securepass1"},
    )
    resp = client.post(
        "/api/alpha/auth/register",
        json={"email": "grace@example.com", "password": "securepass1"},
    )
    assert resp.status_code == 409


def test_register_endpoint_422_short_password(client: TestClient):
    resp = client.post(
        "/api/alpha/auth/register",
        json={"email": "henry@example.com", "password": "short"},
    )
    assert resp.status_code == 422


def test_register_endpoint_422_invalid_email(client: TestClient):
    resp = client.post(
        "/api/alpha/auth/register",
        json={"email": "not-an-email", "password": "securepass1"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# api/routes/auth.py — POST /login
# ---------------------------------------------------------------------------


def _register_and_login(client: TestClient, email: str, password: str) -> str:
    """Helper: register a user and return their access token."""
    client.post(
        "/api/alpha/auth/register",
        json={"email": email, "password": password},
    )
    resp = client.post(
        "/api/alpha/auth/login",
        data={"username": email, "password": password},
    )
    assert resp.status_code == 200
    return resp.json()["access_token"]


def test_login_endpoint_200_returns_token(client: TestClient):
    client.post(
        "/api/alpha/auth/register",
        json={"email": "iris@example.com", "password": "loginpass1"},
    )
    resp = client.post(
        "/api/alpha/auth/login",
        data={"username": "iris@example.com", "password": "loginpass1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


def test_login_endpoint_401_wrong_password(client: TestClient):
    client.post(
        "/api/alpha/auth/register",
        json={"email": "jack@example.com", "password": "correctpass1"},
    )
    resp = client.post(
        "/api/alpha/auth/login",
        data={"username": "jack@example.com", "password": "wrongpass"},
    )
    assert resp.status_code == 401


def test_login_endpoint_401_unknown_email(client: TestClient):
    resp = client.post(
        "/api/alpha/auth/login",
        data={"username": "unknown@example.com", "password": "anypass"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# api/routes/auth.py — GET /me
# ---------------------------------------------------------------------------


def test_me_endpoint_200_with_valid_token(client: TestClient):
    token = _register_and_login(client, "kate@example.com", "katepass12")
    resp = client.get(
        "/api/alpha/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "kate@example.com"


def test_me_endpoint_401_no_token(client: TestClient):
    resp = client.get("/api/alpha/auth/me")
    assert resp.status_code == 401


def test_me_endpoint_401_tampered_token(client: TestClient):
    token = _register_and_login(client, "liam@example.com", "liampass12")
    bad_token = token[:-6] + "xxxxxx"
    resp = client.get(
        "/api/alpha/auth/me",
        headers={"Authorization": f"Bearer {bad_token}"},
    )
    assert resp.status_code == 401


def test_me_endpoint_401_garbage_token(client: TestClient):
    resp = client.get(
        "/api/alpha/auth/me",
        headers={"Authorization": "Bearer not.a.real.token"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Health check — verify demo path is untouched
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
