
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_CHAT_MODEL: str = "llama3"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    CHROMA_DB_PATH: str = "./knowledge_base/chroma_db"
    CHROMA_COLLECTION_NAME: str = "mediassist_stg"

    BM25_INDEX_PATH: str = "./knowledge_base/bm25.pkl"

    PORT: int = 8000
    HOST: str = "0.0.0.0"

    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_TOKEN_BUDGET: int = 4000

    MAX_DIAGNOSIS_PROBABILITY: int = 99

    @property
    def allowed_origins_list(self) -> List[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"openai", "ollama"}
        if v.lower() not in allowed:
            raise ValueError(
                f"LLM_PROVIDER must be one of {allowed}, got '{v}'"
            )
        return v.lower()

    @field_validator("RETRIEVAL_TOP_K")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("RETRIEVAL_TOP_K must be between 1 and 20")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()