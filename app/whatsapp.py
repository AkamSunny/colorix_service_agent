"""
whatsapp.py — Twilio WhatsApp Sandbox
"""
from __future__ import annotations

import logging
from twilio.rest import Client
from twilio.request_validator import RequestValidator
from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


def get_twilio_client() -> Client:
    return Client(settings.twilio_account_sid, settings.twilio_auth_token)


async def send_whatsapp_message(to: str, body: str) -> None:
    """Send a WhatsApp message via Twilio. `to` must include whatsapp: prefix."""
    client = get_twilio_client()

    # Ensure proper format
    if not to.startswith("whatsapp:"):
        to = f"whatsapp:{to}"
    from_number = settings.twilio_whatsapp_number
    if not from_number.startswith("whatsapp:"):
        from_number = f"whatsapp:{from_number}"

    msg = client.messages.create(
        body = body[:1600],
        from_ = from_number,
        to = to,
    )
    logger.info(f"Sent to {to} | sid={msg.sid}")


def validate_twilio_request(
    url: str,
    params: dict,
    signature: str,
) -> bool:
    """Validate that the request genuinely came from Twilio."""
    validator = RequestValidator(settings.twilio_auth_token)
    return validator.validate(url, params, signature)


def parse_twilio_webhook(form_data: dict) -> dict | None:
    """
    Parse Twilio's form-encoded webhook body.
    Returns a dict with from, body, message_sid or None if not a message.
    """
    msg_body = form_data.get("Body", "").strip()
    from_num = form_data.get("From", "")
    msg_sid  = form_data.get("MessageSid", "")

    if not from_num:
        return None

    # Strip whatsapp: prefix for storage
    phone = from_num.replace("whatsapp:", "").replace("+", "")

    return {
        "from":        phone,
        "from_raw":    from_num,
        "text":        msg_body,
        "message_sid": msg_sid,
        "num_media":   int(form_data.get("NumMedia", 0)),
    }


def normalize_phone(phone: str) -> str:
    return phone.replace("+", "").replace(" ", "").replace("-", "")


def phone_to_session_id(phone: str) -> str:
    return f"wa_{normalize_phone(phone)}"