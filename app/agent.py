"""
agent.py — Optimized single-LLM-call agent for Colorix WhatsApp Bot
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage

from app.config   import get_settings
from app.database import (
    get_conversation_history,
    save_message,
    upsert_session,
    create_escalation,
    similarity_search,
)
from app.embeddings import embed_text
from app.llm        import invoke_with_fallback
from app.whatsapp   import send_whatsapp_message

logger   = logging.getLogger(__name__)
settings = get_settings()


async def process_message(
    session_id:   str,
    phone_number: str,
    user_message: str,
) -> str:
    """
    Optimized pipeline — 1 LLM call per message.

    Flow:
    1. Load history + embed query (parallel)
    2. Similarity search Supabase
    3. Single LLM call → language detection + response generation
    4. Handle escalation if needed
    5. Save memory in background (non-blocking)
    """

    # ── Step 1: Parallel history load + embedding ─────────────────────
    history_task = asyncio.create_task(
        asyncio.to_thread(get_conversation_history, session_id, 8)
    )
    embed_task = asyncio.create_task(
        asyncio.to_thread(embed_text, user_message)
    )
    history, embedding = await asyncio.gather(history_task, embed_task)

    # ── Step 2: Vector search ──────────────────────────────────────────
    raw_chunks = await asyncio.to_thread(
        similarity_search, embedding, 5, 0.20
    )

    # ── Step 3: Build context ──────────────────────────────────────────
    context = "\n\n".join(
        f"[{i+1}] {r['content']}"
        for i, r in enumerate(raw_chunks)
    ) if raw_chunks else "No relevant information found in knowledge base."

    # ── Step 4: Build history string ───────────────────────────────────
    history_str = "\n".join(
        f"{'Customer' if m['role'] == 'user' else 'ColorixBot'}: {m['content']}"
        for m in history[-6:]
    ) or "No previous conversation."

    # ── Step 5: Get last language from history ─────────────────────────
    last_lang = "en"
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_lang = msg.get("language", "en")
            break

    # ── Step 6: Single LLM call ────────────────────────────────────────
    system_prompt = f"""You are ColorixBot, the WhatsApp assistant for Colorix Groupe — a professional printing company in Yaoundé, Cameroon.

LANGUAGE RULE:
- Detect the customer's language from their message
- Respond in the SAME language they used
- Default to English unless the message is clearly French
- If message is very short (ok, yes, oui, merci), use the last conversation language which was: {last_lang}

ESCALATION RULE:
- If the customer explicitly asks to speak to a human, staff, agent, manager, or real person → respond ONLY with the word: ESCALATE

KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{history_str}

RULES:
1. ALWAYS try to answer the question using the knowledge base above
2. If the knowledge base has relevant info → use it to answer directly
3. If the knowledge base has NO info BUT you know the answer as a printing company assistant → answer confidently using your general knowledge about Colorix's products
4. For example: if asked about envelopes, business cards, brochures — you KNOW Colorix prints these, answer yes and give details
5. Only say "I don't have that info" for questions completely unrelated to printing or Colorix (e.g. weather, sports, unrelated topics)
6. Never invent prices — all pricing is quote-based, direct to +237 696 26 26 56
7. Keep replies short and clear — this is WhatsApp
8. Do not include full URLs with https:// — just mention colorixgroupe.com
9. Complaints or reprints → hotline +237 699 88 85 77

COMPANY INFO:
- Name: Colorix Groupe
- Address: Rue de la Province, DGSN, Yaoundé, Cameroon
- Phone: +237 696 26 26 56
- Hotline: +237 699 88 85 77 (satisfaction and urgent orders)
- Website: colorixgroupe.com
- Guarantee: Satisfied or Reprinted — unhappy with quality? We reprint free!
- Products: posters, banners, flyers, business cards, brochures, envelopes, stamps, folders, signage, roll-ups, kakemono, mugs, pens, key rings, diaries, calendars, photo books, invitation cards, event packs (concert, wedding, birth, funeral)
"""

    response = await invoke_with_fallback([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ])
    response = (response or "").strip()

    # ── Step 7: Check escalation ───────────────────────────────────────
    explicit_human_keywords = [
        "speak to", "talk to", "real person", "human agent", "speak with",
        "talk with", "staff", "manager", "personnel", "representative",
        "parler à", "un agent", "responsable", "une personne",
    ]

    # Check if bot offered human last turn and user said yes
    last_bot_msg = ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_bot_msg = msg["content"].lower()
            break

    bot_offered = any(p in last_bot_msg for p in [
        "would you like to speak", "reply *yes*", "connect you",
        "souhaitez-vous", "répondez *oui*",
    ])
    user_said_yes = any(w in user_message.lower() for w in [
        "yes", "oui", "yeah", "sure", "please", "ok"
    ])

    needs_escalation = (
        response.upper().startswith("ESCALATE")
        or any(kw in user_message.lower() for kw in explicit_human_keywords)
        or (bot_offered and user_said_yes)
    )

    # ── Step 8: Handle escalation ──────────────────────────────────────
    if needs_escalation:
        escalation_id = create_escalation(
            session_id      = session_id,
            customer_number = phone_number,
            trigger_reason  = "user_requested_human",
            last_user_msg   = user_message,
            bot_draft       = response if not response.upper().startswith("ESCALATE") else "",
        )

        staff_number = settings.human_review_whatsapp
        staff_msg = (
            f"🔔 *New Customer Request #{escalation_id}*\n\n"
            f"📱 Number: +{phone_number}\n"
            f"💬 Message: _{user_message}_\n\n"
            f"Please reply to them directly on WhatsApp."
        )
        try:
            await send_whatsapp_message(to=staff_number, body=staff_msg)
            logger.info(f"[escalation] #{escalation_id} — staff notified")
        except Exception as e:
            logger.error(f"[escalation] Staff notify failed: {e}")

        response = (
            "I'm connecting you with a Colorix team member right now. 🙏\n\n"
            "Our staff will reach out to you shortly on WhatsApp.\n"
            "You can also call us directly: *+237 696 26 26 56*"
            if last_lang == "en" else
            "Je vous connecte avec un membre de l'équipe Colorix. 🙏\n\n"
            "Notre équipe vous contactera bientôt sur WhatsApp.\n"
            "Vous pouvez aussi appeler : *+237 696 26 26 56*"
        )

    # ── Step 9: Detect language for storage ───────────────────────────
    lang = "fr" if any(w in user_message.lower() for w in [
        "bonjour", "salut", "merci", "oui", "non", "je", "nous",
        "vous", "comment", "combien", "quel", "pour", "avec", "est-ce",
        "pouvez", "avez", "voulez", "souhaitez", "livraison", "commande",
    ]) else "en"

    # ── Step 10: Save memory in background ────────────────────────────
    asyncio.create_task(_save_memory(
        session_id   = session_id,
        phone_number = phone_number,
        user_message = user_message,
        response     = response,
        lang         = lang,
        escalated    = needs_escalation,
    ))

    logger.info(f"[agent] {phone_number} | lang={lang} | chunks={len(raw_chunks)} | escalated={needs_escalation}")
    return response


async def _save_memory(
    session_id:   str,
    phone_number: str,
    user_message: str,
    response:     str,
    lang:         str,
    escalated:    bool,
) -> None:
    """Save conversation to Supabase in background — does not block response."""
    try:
        save_message(
            session_id = session_id,
            role       = "user",
            content    = user_message,
            language   = lang,
            metadata   = {"phone": phone_number},
        )
        save_message(
            session_id = session_id,
            role       = "assistant",
            content    = response,
            language   = lang,
            metadata   = {"escalated": escalated, "timestamp": datetime.utcnow().isoformat()},
        )
        upsert_session(
            session_id   = session_id,
            phone_number = phone_number,
            language     = lang,
            state        = {
                "last_message":  user_message,
                "last_response": response,
            },
        )
        logger.info(f"[memory] session={session_id} saved")
    except Exception as e:
        logger.error(f"[memory] Save failed: {e}")