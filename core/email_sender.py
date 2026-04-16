"""
Email delivery via Resend API.

Resend is used over SendGrid/Postmark for its simpler API surface and
3,000 email/month free tier — sufficient for SMB demo and early customers.

Required env vars:
    RESEND_API_KEY    — Resend API key (sk_...)
    RESEND_FROM_EMAIL — verified sender address (e.g. outreach@yourdomain.com)

The send_email() function is the only public symbol. It never silently
swallows errors — callers are expected to catch EmailDeliveryError and
decide whether to log-and-continue or surface to the UI.
"""

import logging
import os

import resend

logger = logging.getLogger(__name__)


class EmailDeliveryError(Exception):
    """
    Raised when an email cannot be delivered via Resend.

    Wraps Resend API errors, missing configuration, and unexpected
    exceptions so callers only need to handle one exception type.
    """


def _unsubscribe_footer(from_email: str) -> str:
    """
    CAN-SPAM / GDPR compliant footer appended to every outgoing email.

    Deployers must also add a physical mailing address per CAN-SPAM §7(a)(5)(A).
    Set PHYSICAL_ADDRESS env var to include it; omitting it does not prevent
    delivery but may affect compliance in some jurisdictions.
    """
    domain = from_email.split("@")[-1] if "@" in from_email else from_email
    address = os.getenv("PHYSICAL_ADDRESS", "")
    lines = [
        "",
        "---",
        f"You received this because you were identified as a strong fit by {domain}.",
        "To stop receiving emails, reply with \"unsubscribe\" in the subject line.",
    ]
    if address:
        lines.append(address)
    return "\n".join(lines)


def send_email(to: str, subject: str, body: str) -> str:
    """
    Send a plain-text email via Resend, appending a CAN-SPAM/GDPR footer.

    Args:
        to:      Recipient email address.
        subject: Email subject line.
        body:    Plain-text email body (footer is appended automatically).

    Returns:
        Resend message ID (e.g. "re_abc123") for delivery tracking.

    Raises:
        EmailDeliveryError: If RESEND_API_KEY is not set, or if the
                            Resend API returns an error.
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    if not api_key:
        raise EmailDeliveryError(
            "RESEND_API_KEY is not configured. "
            "Set it in your .env file to enable email delivery."
        )

    from_email = os.getenv("RESEND_FROM_EMAIL", "")
    if not from_email:
        raise EmailDeliveryError(
            "RESEND_FROM_EMAIL is not configured. "
            "Set it to a Resend-verified sender address (e.g. outreach@yourdomain.com)."
        )

    resend.api_key = api_key
    full_body = body + _unsubscribe_footer(from_email)

    try:
        params: resend.Emails.SendParams = {
            "from": from_email,
            "to": [to],
            "subject": subject,
            "text": full_body,
        }
        response = resend.Emails.send(params)
        message_id: str = response["id"]
        logger.info("Email sent | to=%s | message_id=%s", to, message_id)
        return message_id
    except Exception as exc:
        logger.error("Resend API error sending to %s: %s", to, exc)
        raise EmailDeliveryError(f"Resend API error: {exc}") from exc
