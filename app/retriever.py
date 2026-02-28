"""
retriever.py — RAG retrieval with HyDE query expansion
Uses the LLM to rephrase queries before embedding for better semantic matching.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.embeddings import embed_text
from app.database   import similarity_search
from app.config     import get_settings

logger   = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    content:    str
    metadata:   dict
    similarity: float


async def expand_query(query: str) -> str:
    """
    Use LLM to rephrase query into a more retrieval-friendly form.
    This dramatically improves semantic search recall.
    """
    from app.llm import invoke_with_fallback
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = (
        "You are a search query optimizer for a printing company knowledge base.\n"
        "Rephrase the following customer question into a clear, keyword-rich search query.\n"
        "Include relevant synonyms and related terms.\n"
        "Return ONLY the rephrased query, nothing else.\n\n"
        f"Customer question: {query}"
    )

    try:
        rephrased = await invoke_with_fallback([
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ])
        rephrased = rephrased.strip()
        if rephrased and len(rephrased) < 300:
            logger.info(f"[query expansion] '{query}' → '{rephrased}'")
            return rephrased
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")

    return query


async def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """
    RAG retrieval with LLM-based query expansion.
    1. Expand query using LLM
    2. Embed both original and expanded query
    3. Merge results, deduplicate, return top k
    """
    k = top_k or settings.vector_top_k

    # Get LLM-expanded query
    expanded = await expand_query(query)

    # Deduplicate queries
    queries = list(dict.fromkeys([query, expanded]))

    all_results: dict[int, dict] = {}
    for q in queries:
        embedding = embed_text(q)
        results   = similarity_search(
            query_embedding = embedding,
            top_k           = k,
            threshold       = 0.15,
        )
        for r in results:
            chunk_id = r["id"]
            if chunk_id not in all_results or r["similarity"] > all_results[chunk_id]["similarity"]:
                all_results[chunk_id] = r

    sorted_results = sorted(
        all_results.values(),
        key     = lambda x: x["similarity"],
        reverse = True,
    )[:k]

    chunks = [
        RetrievedChunk(
            content    = r["content"],
            metadata   = r.get("metadata", {}),
            similarity = round(float(r.get("similarity", 0.0)), 4),
        )
        for r in sorted_results
    ]

    if chunks:
        logger.info(f"Retrieved {len(chunks)} chunks | best={chunks[0].similarity:.3f}")
    else:
        logger.warning(f"No chunks found for: {query[:50]!r}")

    return chunks


def format_context(chunks: list[RetrievedChunk]) -> tuple[str, float]:
    if not chunks:
        return "No relevant information found in the knowledge base.", 0.0

    parts = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.metadata.get("section", "Knowledge Base")
        parts.append(f"[Source {i} — {section}]\n{chunk.content}")

    context = "\n\n---\n\n".join(parts)
    avg_sim = round(sum(c.similarity for c in chunks) / len(chunks), 4)
    return context, avg_sim