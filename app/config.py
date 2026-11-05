"""Application settings, read from the environment and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed app configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    data_dir: Path = Path("./data")

    llm_provider: Literal["gemini", "fake"] = "gemini"
    # SecretStr keeps the key out of logs and tracebacks.
    google_api_key: SecretStr = SecretStr("")
    chat_model: str = ""
    embedding_provider: Literal["gemini", "local", "fake"] = "local"
    # Empty means the default local model. Gemini requires a value.
    embedding_model: str = ""
    # Local embeddings only. auto uses CUDA if it works and otherwise logs a line and uses the CPU.
    embedding_device: Literal["auto", "cuda", "cpu"] = "auto"
    # Retries after the first attempt, for temporary errors such as 429 and 503.
    max_retries: int = Field(default=3, ge=0)

    # Gemini embeddings only: small batches with a pause between them, to stay under rate limits.
    embed_batch_size: int = Field(default=20, ge=1, le=100)
    embed_delay_seconds: float = Field(default=1.0, ge=0)
    # How long to wait after a rate limit when Gemini doesn't suggest a delay.
    embed_rate_limit_wait_seconds: float = Field(default=60.0, ge=0)
    # The most one batch may spend waiting before indexing fails.
    embed_max_wait_seconds: float = Field(default=600.0, ge=0)

    chunk_size: int = 1500
    chunk_overlap: int = 150
    retrieval_k: int = 6
    max_files: int = 1500
    max_file_bytes: int = 200_000

    @property
    def fastembed_dir(self) -> Path:
        """Where downloaded local embedding models are stored."""
        return self.data_dir / "fastembed"

    @property
    def repos_dir(self) -> Path:
        """Where cloned repos are stored."""
        return self.data_dir / "repos"

    @property
    def chroma_dir(self) -> Path:
        """Where the Chroma index is stored."""
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance, creating its data directories."""
    settings = Settings()
    settings.repos_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return settings
