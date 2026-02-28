"""
main.py — FastAPI with Twilio WhatsApp webhook
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi           import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response

from app.config   import get_settings
from app.agent    import process_message
from app.whatsapp import (
    send_whatsapp_message,
    parse_twilio_webhook,
    phone_to_session_id,
)
from app.language import get_greeting, detect_language

settings = get_settings()

logging.basicConfig(
    stream  = sys.stdout,
    level   = getattr(logging, settings.log_level.upper(), logging.INFO),
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Colorix WhatsApp Agent starting...")

    from app.embeddings import get_embeddings
    get_embeddings()
    logger.info("✅ Embedding model ready")

    from app.database import document_count
    count = document_count()
    if count == 0:
        logger.warning("⚠️  No chunks in Supabase! Run: python -m scripts.ingest")
    else:
        logger.info(f"📦 Supabase: {count} document chunks ready")

    yield
    logger.info("🛑 Shutting down")


app = FastAPI(
    title    = "Colorix Groupe — WhatsApp RAG Agent",
    version  = "1.0.0",
    lifespan = lifespan,
)


@app.get("/health")
async def health():
    from app.database import get_stats
    return {"status": "ok", "stats": get_stats()}


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Twilio sends form-encoded POST requests.
    Always return 200 immediately — process in background.
    """
    form_data = dict(await request.form())
    msg = parse_twilio_webhook(form_data)

    if not msg:
        return Response(content="", media_type="text/xml")

    phone      = msg["from"]
    text       = msg["text"]
    num_media  = msg["num_media"]
    session_id = phone_to_session_id(phone)
    from_raw   = msg["from_raw"]

    logger.info(f"📨 {phone}: {text[:80]!r}")

    # Media messages
    if num_media > 0:
        lang  = detect_language(text or "hello")
        reply = {
            "en": (
                "Thank you for the file! 📎\n\n"
                "To submit design files please upload at colorixgroupe.com "
                "or call: *+237 696 26 26 56* 😊"
            ),
            "fr": (
                "Merci pour le fichier ! 📎\n\n"
                "Pour soumettre des fichiers, déposez-les sur "
                "colorixgroupe.com ou appelez : *+237 696 26 26 56* 😊"
            ),
        }.get(lang, "Please visit colorixgroupe.com to upload your file.")
        background_tasks.add_task(send_whatsapp_message, to=from_raw, body=reply)
        return Response(content="", media_type="text/xml")

    # Greetings
    if not text or text.lower() in {
        "hi", "hello", "bonjour", "salut", "hey",
        "start", "menu", "help", "aide"
    }:
        lang = detect_language(text or "hello")
        background_tasks.add_task(
            send_whatsapp_message,
            to   = from_raw,
            body = get_greeting(lang),
        )
        return Response(content="", media_type="text/xml")

    # Process with agent
    background_tasks.add_task(
        _process_and_reply,
        phone_number = phone,
        from_raw     = from_raw,
        session_id   = session_id,
        user_message = text,
    )

    return Response(content="", media_type="text/xml")


async def _process_and_reply(
    phone_number: str,
    from_raw:     str,
    session_id:   str,
    user_message: str,
) -> None:
    try:
        response = await process_message(
            session_id   = session_id,
            phone_number = phone_number,
            user_message = user_message,
        )
        await send_whatsapp_message(to=from_raw, body=response)
        logger.info(f"✅ Replied to {phone_number}: {response[:60]!r}")
    except Exception as e:
        logger.error(f"❌ Error for {phone_number}: {e}", exc_info=True)
        try:
            await send_whatsapp_message(
                to   = from_raw,
                body = "⚠️ Something went wrong. Please call: *+237 696 26 26 56*",
            )
        except Exception:
            pass


@app.post("/webhook/staff-reply")
async def staff_reply(
    escalation_id:   int,
    staff_response:  str,
    customer_number: str,
    secret:          str,
):
    from fastapi import HTTPException
    if secret != settings.app_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    from app.database import resolve_escalation
    resolve_escalation(escalation_id, staff_response)
    await send_whatsapp_message(
        to   = f"whatsapp:+{customer_number}",
        body = staff_response,
    )
    return {"status": "resolved", "escalation_id": escalation_id}


@app.post("/admin/ingest")
async def admin_ingest(background_tasks: BackgroundTasks, secret: str):
    from fastapi import HTTPException
    if secret != settings.app_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    background_tasks.add_task(_run_ingestion)
    return {"status": "ingestion_started"}


async def _run_ingestion():
    import asyncio
    from pathlib import Path
    from scripts.ingest import run_ingestion
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, run_ingestion, Path(settings.knowledge_base_path))


@app.get("/admin/stats")
async def admin_stats(secret: str):
    from fastapi import HTTPException
    if secret != settings.app_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")
    from app.database import get_stats
    return get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host      = settings.app_host,
        port      = settings.app_port,
        reload    = False,
        log_level = settings.log_level.lower(),
    )