"""
POST /api/contacts/upload — CSV upload of contact list.
GET  /api/contacts         — list contacts with segment filter.
DELETE /api/contacts/{id}  — remove a single contact.

CSV format expected:
    Required columns: email
    Optional columns: first_name, company, company_size, channel, industry

company_size must be one of: SMB, mid_market, enterprise
channel must be one of: organic, paid_search, social, referral, email

Duplicate emails are upserted (UPDATE on conflict) so re-uploading a
cleaned CSV doesn't create duplicate rows.
"""

import csv
import io
import logging
import sqlite3

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from pydantic import BaseModel

from api.deps import CurrentUser, OptionalUser, tenant_id_from
from core.outreach import _sqlite_path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/contacts", tags=["contacts"])

_VALID_COMPANY_SIZES = {"SMB", "mid_market", "enterprise"}
_VALID_CHANNELS = {"organic", "paid_search", "social", "referral", "email"}
_MAX_UPLOAD_ROWS = 10_000


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Contact(BaseModel):
    id: int
    email: str
    first_name: str | None
    company: str | None
    company_size: str | None
    channel: str | None
    industry: str | None
    uploaded_at: str


class ContactsListResponse(BaseModel):
    contacts: list[Contact]
    total: int


class UploadResult(BaseModel):
    inserted: int
    updated: int
    skipped: int
    errors: list[str]


class ActivateRequest(BaseModel):
    emails: list[str]
    activated_at: str | None = None   # ISO datetime; defaults to now if omitted


class ActivateResult(BaseModel):
    updated: int
    not_found: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResult)
async def upload_contacts(file: UploadFile, user: OptionalUser) -> dict:
    """
    Upload a CSV of contacts to the contacts table.

    Accepts: email (required), first_name, company, company_size, channel, industry.
    Duplicate emails are updated in-place rather than rejected.
    Rows with invalid emails or unrecognised segment values are skipped with
    a descriptive error message returned in the response body.

    Maximum: 10,000 rows per upload.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "File must be a .csv"},
        )

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")  # strip BOM if present (common in Excel exports)
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "CSV must be UTF-8 encoded"},
        )

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or "email" not in reader.fieldnames:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "CSV must have an 'email' column"},
        )

    rows = list(reader)
    if len(rows) > _MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Upload exceeds {_MAX_UPLOAD_ROWS} row limit"},
        )

    tenant_id = tenant_id_from(user)
    inserted = updated = skipped = 0
    errors: list[str] = []

    conn = sqlite3.connect(_sqlite_path())
    try:
        for i, row in enumerate(rows, start=2):  # 2 = first data row (1 = header)
            email = row.get("email", "").strip().lower()
            if not email or "@" not in email or "." not in email.split("@")[-1]:
                errors.append(f"Row {i}: invalid or missing email '{email}'")
                skipped += 1
                continue

            company_size = row.get("company_size", "").strip() or None
            if company_size and company_size not in _VALID_COMPANY_SIZES:
                errors.append(
                    f"Row {i} ({email}): unknown company_size '{company_size}' "
                    f"— expected one of {sorted(_VALID_COMPANY_SIZES)}"
                )
                skipped += 1
                continue

            channel = row.get("channel", "").strip() or None
            if channel and channel not in _VALID_CHANNELS:
                errors.append(
                    f"Row {i} ({email}): unknown channel '{channel}' "
                    f"— expected one of {sorted(_VALID_CHANNELS)}"
                )
                skipped += 1
                continue

            first_name = row.get("first_name", "").strip() or None
            company    = row.get("company", "").strip() or None
            industry   = row.get("industry", "").strip() or None

            # Upsert — update segment fields if (tenant_id, email) already exists
            existing = conn.execute(
                "SELECT id FROM contacts WHERE tenant_id = ? AND email = ?", (tenant_id, email)
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE contacts
                    SET first_name = ?, company = ?, company_size = ?,
                        channel = ?, industry = ?, uploaded_at = datetime('now')
                    WHERE tenant_id = ? AND email = ?
                    """,
                    (first_name, company, company_size, channel, industry, tenant_id, email),
                )
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO contacts (tenant_id, email, first_name, company, company_size, channel, industry)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tenant_id, email, first_name, company, company_size, channel, industry),
                )
                inserted += 1

        conn.commit()
    except Exception as exc:
        logger.error("contacts upload error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Database error during upload", "detail": str(exc)},
        )
    finally:
        conn.close()

    logger.info("contacts/upload | inserted=%d | updated=%d | skipped=%d", inserted, updated, skipped)
    return {"inserted": inserted, "updated": updated, "skipped": skipped, "errors": errors[:20]}


@router.get("", response_model=ContactsListResponse)
def list_contacts(
    user: OptionalUser,
    company_size: str | None = None,
    channel: str | None = None,
    limit: int = 100,
) -> dict:
    """
    Return contacts for the current tenant, optionally filtered by segment.
    """
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "limit must be between 1 and 1000"},
        )

    tenant_id = tenant_id_from(user)
    clauses: list[str] = ["tenant_id = ?"]
    params: list = [tenant_id]
    if company_size:
        clauses.append("company_size = ?")
        params.append(company_size)
    if channel:
        clauses.append("channel = ?")
        params.append(channel)

    where = "WHERE " + " AND ".join(clauses)

    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM contacts {where} ORDER BY uploaded_at DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) FROM contacts {where}", params
        ).fetchone()[0]
    except Exception as exc:
        logger.error("contacts list error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Failed to fetch contacts", "detail": str(exc)},
        )
    finally:
        conn.close()

    contacts = [
        {
            "id":           row["id"],
            "email":        row["email"],
            "first_name":   row["first_name"],
            "company":      row["company"],
            "company_size": row["company_size"],
            "channel":      row["channel"],
            "industry":     row["industry"],
            "uploaded_at":  row["uploaded_at"],
        }
        for row in rows
    ]
    return {"contacts": contacts, "total": total}


@router.delete("/{contact_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_contact(contact_id: int, user: OptionalUser) -> None:
    """Remove a single contact by ID, scoped to the current tenant."""
    tenant_id = tenant_id_from(user)
    conn = sqlite3.connect(_sqlite_path())
    try:
        result = conn.execute(
            "DELETE FROM contacts WHERE id = ? AND tenant_id = ?", (contact_id, tenant_id)
        )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"Contact {contact_id} not found"},
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("contacts delete error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Delete failed", "detail": str(exc)},
        )
    finally:
        conn.close()


@router.post("/activate", response_model=ActivateResult)
def activate_contacts(body: ActivateRequest, user: CurrentUser) -> dict:
    """
    Mark a list of contacts as activated in contact_sends.

    Call this after importing activation data from your CRM or product
    analytics — e.g. after you see who signed up or activated following
    the outreach campaign. The Results tab uses this data to compute
    real campaign lift instead of the historical segment baseline.

    activated_at: ISO datetime string. Defaults to now if omitted.
    """
    from datetime import datetime, timezone

    tenant_id    = user["email"]
    activated_at = body.activated_at or datetime.now(tz=timezone.utc).isoformat()
    updated      = 0
    not_found:  list[str] = []

    conn = sqlite3.connect(_sqlite_path())
    try:
        for email in body.emails:
            email = email.strip().lower()
            result = conn.execute(
                """
                UPDATE contact_sends
                SET activated_at = ?, status = 'activated'
                WHERE email = ? AND tenant_id = ? AND is_holdout = 0
                """,
                (activated_at, email, tenant_id),
            )
            if result.rowcount == 0:
                not_found.append(email)
            else:
                updated += result.rowcount
        conn.commit()
    except Exception as exc:
        logger.error("contacts/activate error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "Activation update failed", "detail": str(exc)},
        )
    finally:
        conn.close()

    logger.info("contacts/activate | tenant=%s | updated=%d | not_found=%d", tenant_id, updated, len(not_found))
    return {"updated": updated, "not_found": not_found}
