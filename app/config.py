from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Load .env explicitly — required on Windows
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # Supabase
    supabase_url:         str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")

    # Knowledge base
    knowledge_base_path: str = Field(
        "./knowledge_base/Colorix_Knowledge_Base_.txt",
        env="KNOWLEDGE_BASE_PATH",
    )

    # LLMs
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model:   str = Field("gemini-1.5-flash",        env="GEMINI_MODEL")
    groq_api_key:   str = Field(..., env="GROQ_API_KEY")
    groq_model:     str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    primary_llm:    str = Field("gemini",                  env="PRIMARY_LLM")

    # HuggingFace
    hf_embedding_model: str = Field(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="HF_EMBEDDING_MODEL",
    )

    twilio_account_sid:     str = Field(..., env="TWILIO_ACCOUNT_SID")
    twilio_auth_token:      str = Field(..., env="TWILIO_AUTH_TOKEN")
    twilio_whatsapp_number: str = Field(..., env="TWILIO_WHATSAPP_NUMBER")

    # HITL
    human_review_whatsapp:     str   = Field(...,  env="HUMAN_REVIEW_WHATSAPP")
    hitl_confidence_threshold: float = Field(0.70, env="HITL_CONFIDENCE_THRESHOLD")
    hitl_keywords: str = Field(
        "complaint,urgent,refund,reprint,legal,manager,problem,issue",
        env="HITL_KEYWORDS",
    )

    # App
    app_host:   str = Field("0.0.0.0", env="APP_HOST")
    app_port:   int = Field(8000,      env="APP_PORT")
    app_secret: str = Field("changeme",env="APP_SECRET")
    log_level:  str = Field("INFO",    env="LOG_LEVEL")

    # RAG
    vector_top_k:  int = Field(5,   env="VECTOR_TOP_K")
    chunk_size:    int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64,  env="CHUNK_OVERLAP")

    @property
    def hitl_keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.hitl_keywords.split(",")]

    model_config = {
        "env_file":          ".env",
        "env_file_encoding": "utf-8",
        "extra":             "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
