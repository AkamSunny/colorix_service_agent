from __future__ import annotations
import logging
from datetime import datetime
from supabase import create_client, Client
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_supabase: Client | None = None

def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            settings.supabase_url,
            settings.supabase_service_key
        )
    return _supabase


def insert_document_chunks(rows: list[dict]) -> None:
    """
    Bulk insert document chunks into colorix_documents.
    Each row: {content, metadata, embedding}
    """

    db = get_supabase()
    db.table("colorix_documents").insert(rows).execute()


def similarity_search(query_embedding: list[float],
                      top_k: int = 5,
                      threshold: float = 0.40) -> list[dict]:
    """
    Call the match_colorix_documents Postgres function.
    Return list of {id, content, metadata, similarity}.
    """

    db = get_supabase()
    try:
        result = db.rpc("match_colorix_documents",
        {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": top_k
        },).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []
    
def clear_documents() -> None:
    """Delete all existing document chunks (used before re-ingestion)."""
    db = get_supabase()
    db.table("colorix_documents").delete().neq("id", 0).execute()


def document_count() -> int:
    """Return the number of chunks currently stored."""
    db = get_supabase()
    result = db.table("colorix_documents").select("id", count="exact").execute()
    return result.count or 0


# Conversation Memory
def save_message(session_id: str,
                 role: str,
                 content: str,
                 language: str = "en",
                 metadata: dict | None = None
                 ) -> None:
    db = get_supabase()
    db.table("colorix_conversations").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "language": language,
        "metadata": metadata or {},
    }).execute()


def get_conversation_history(session_id: str, limit: int = 12) -> list[dict]:
    """Return last N messages for a session, oldesr-first"""
    db = get_supabase()
    result = (db.table("colorix_conversations")
              .select("role, content, created_at")
              .eq("session_id", session_id)
              .order("created_at", desc=False)
              .limit(limit)
              .execute()
              )
    return result.data or []


# Session State

def get_session(session_id: str) -> dict | None:
    db = get_supabase()
    result = (
        db.table("colorix_sessions")
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    return result.data

def upsert_session(
    session_id:   str,
    phone_number: str,
    language:     str,
    state:        dict,
) -> None:
    db = get_supabase()
    db.table("colorix_sessions").upsert({
        "session_id":   session_id,
        "phone_number": phone_number,
        "language":     language,
        "state":        state,
        "last_active":  datetime.utcnow().isoformat(),
    }).execute()

    
# Escalation logging

def create_escalation(
        session_id: str,
        customer_number: str,
        trigger_reason: str,
        last_user_msg: str,
        bot_draft: str | None = None
 ) -> int:
    
    db = get_supabase()
    result = (
        db.table("colorix_escalations").insert({
            "session_id": session_id,
            "customer_number": customer_number,
            "trigger_reason": trigger_reason,
            "last_user_msg": last_user_msg,
            "bot_draft": bot_draft 
        }).execute()
    )
    return result.data[0]["id"] if result.data else -1


def resolve_escalation(escalation_id: int, staff_response:str) -> None:
    db = get_supabase()
    db.table("colorix_escalations").update({
        "resolved": True,
        "staff_response": staff_response,
        "resolved_at": datetime.utcnow().isoformat(),
    }).eq("id", escalation_id).execute()


def get_stats() -> dict:
    db = get_supabase()
    msgs = db.table("colorix_conversations").select("id", count="exact").execute()
    escs = db.table("colorix_escalations").select("id", count="exact").execute()
    sessions = db.table("colorix_sessions").select("session_id", count="exact").execute()
    docs = db.table("colorix_documents").select("id", count="exact").execute()

    return {
        "total_messages": msgs.count or 0,
        "total_escalations": escs.count or 0,
        "total_sessions": sessions.count or 0,
        "total_chunks": docs.count or 0
    }