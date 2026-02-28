from __future__ import annotations
import logging
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@lru_cache(maxsize=1)
def get_gemini() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model = settings.gemini_model,
        google_api_key = settings.google_api_key,
        temperature = 0.3,
        max_tokens = 300,
        convert_system_message_to_human = True
    )

@lru_cache(maxsize=1)
def get_groq() -> ChatGroq:
    return ChatGroq(
        model = settings.groq_model,
        groq_api_key = settings.groq_api_key,
        temperature = 0.3,
        max_tokens= 300
    )


async def invoke_with_fallback(messages: list, prefer: str | None=None) -> str:
    """
    Try primary LLm. On any failure, automatically switch to the other.
    Return plain text response.
    """

    use_groq = (prefer or settings.primary_llm) == "groq"
    primary = get_groq() if use_groq else get_gemini()
    fallback = get_gemini if use_groq else get_groq()

    try:
        resp = await primary.ainvoke(messages)
        return resp.content
    except Exception as e:
        logger.warning(f"Primary LLM failed ({e}) - switching to fallback")
        resp = await fallback.ainvoke(messages)
        return resp.content