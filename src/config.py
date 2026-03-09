"""Configuration management for Hellobooks AI."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    anthropic_api_key: str = Field(default="")
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    llm_max_tokens: int = Field(default=1024, ge=256, le=8096)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.0)

    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)

    top_k: int = Field(default=3, ge=1, le=10)
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=256)

    knowledge_base_dir: Path = Field(default=BASE_DIR / "knowledge_base")
    vector_store_dir: Path = Field(default=BASE_DIR / "vector_store")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("vector_store_dir", "knowledge_base_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        return Path(v)

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_dir / "index.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.vector_store_dir / "metadata.json"

settings = Settings()
