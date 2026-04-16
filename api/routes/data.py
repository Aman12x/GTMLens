"""
POST /api/data/upload  — upload real event CSV, seed tenant DuckDB.
GET  /api/data/status  — check if tenant has real data or is using demo dataset.

CSV format for upload:
    Required columns: user_id, treatment (0/1), activated (0/1)
    Optional columns: company_size, channel, industry, impression_date,
                      signup_date, activation_date, revenue

company_size: SMB | mid_market | enterprise
channel:      organic | paid_search | social | referral | email

Authenticated requests operate on the caller's isolated tenant DuckDB.
Unauthenticated requests are rejected — demo data cannot be overwritten
via this endpoint (use POST /api/demo/reset for that).
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from api.db import _tenant_db_path, tenant_has_data
from api.deps import CurrentUser, get_optional_user, tenant_id_from
from data.seed_db import seed_tenant_duckdb

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])

_MAX_ROWS = 500_000


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DataStatus(BaseModel):
    tenant_id: str
    has_real_data: bool
    is_demo: bool
    message: str


class UploadDataResult(BaseModel):
    n_users: int
    n_treatment: int
    n_control: int
    activation_rate: float
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status", response_model=DataStatus)
def data_status(user: dict | None = Depends(get_optional_user)) -> dict:
    """
    Return whether the current tenant has real data or is using the demo dataset.

    Used by the frontend to show "upload your data" vs. "N users loaded" messaging.
    Unauthenticated requests always report is_demo=True.
    """
    tid = tenant_id_from(user)
    is_demo = tid == "demo"
    has_data = tenant_has_data(tid)

    if is_demo:
        msg = "Using synthetic demo data. Sign in and upload a CSV to analyse your own funnel."
    elif has_data:
        msg = "Using your uploaded data."
    else:
        msg = "No data uploaded yet. Upload a CSV to get started."

    return {
        "tenant_id":     tid,
        "has_real_data": has_data and not is_demo,
        "is_demo":       is_demo,
        "message":       msg,
    }


@router.post("/upload", response_model=UploadDataResult)
async def upload_data(file: UploadFile, user: CurrentUser) -> dict:
    """
    Upload a CSV of user-level funnel data and seed the tenant's DuckDB.

    Required columns: user_id, treatment, activated
    Optional columns: company_size, channel, industry, impression_date,
                      signup_date, activation_date, revenue

    This replaces any previously uploaded data for this tenant. The
    operation is idempotent — re-uploading a corrected CSV is safe.

    Returns summary stats for the uploaded dataset.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "File must be a .csv"},
        )

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "CSV must be UTF-8 encoded"},
        )

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Could not parse CSV: {exc}"},
        )

    if len(df) > _MAX_ROWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Upload exceeds {_MAX_ROWS:,} row limit"},
        )

    tenant_id = user["email"]
    db_path   = _tenant_db_path(tenant_id)

    try:
        summary = seed_tenant_duckdb(df, db_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": str(exc)},
        )
    except Exception as exc:
        logger.error("data/upload: seeding error for tenant %s: %s", tenant_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to seed database", "detail": str(exc)},
        )

    logger.info(
        "data/upload | tenant=%s | n_users=%d | n_treatment=%d",
        tenant_id, summary["n_users"], summary["n_treatment"],
    )

    return {
        **summary,
        "message": (
            f"Loaded {summary['n_users']:,} users "
            f"({summary['n_treatment']:,} treatment, {summary['n_control']:,} control). "
            f"Overall activation rate: {summary['activation_rate']:.1%}."
        ),
    }
