from __future__ import annotations
from functools import lru_cache
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    logger.info(f"Loading HF model: {settings.hf_embedding_model}")

    return HuggingFaceEmbeddings(
        model_name = settings.hf_embedding_model,
        model_kwargs = {"device": "cpu"},
        encode_kwargs = {
            "normalize_embeddings": True,
            }
    )


def embed_text(text: str) -> list[float]:
    """embed a single string -> 382-dim vector"""
    return get_embeddings().embed_query(text)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list a strings -> list of 382-dim vectors."""
    return get_embeddings().embed_documents(texts)
