"""
embeddings.py — Lightweight embeddings via Google GenAI SDK
"""
from __future__ import annotations

import logging
from app.config import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


def embed_text(text: str) -> list[float]:
    """Embed a single string using Google Gemini embedding."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        result = genai.embed_content(
            model   = "models/text-embedding-004",
            content = text,
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return [0.0] * 768


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple strings."""
    return [embed_text(t) for t in texts]
