"""Application settings, read from the environment and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


ThinkingLevel = Literal["default", "minimal", "low", "medium", "high"]


class Settings(BaseSettings):
    """Typed app configuration."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # A small pool suits Neon's connection limits, and hosted Postgres can take a while to wake up.
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_connect_timeout_seconds: int = Field(default=5, ge=1)
    data_dir: Path = Path("./data")

    llm_provider: Literal["gemini", "fake"] = "gemini"
    # SecretStr keeps the key out of logs and tracebacks.
    google_api_key: SecretStr = SecretStr("")
    # Optional. Without it, GitHub allows far fewer requests.
    github_token: SecretStr = SecretStr("")
    # Models to try in order, as comma-separated lists. NoDecode stops pydantic expecting JSON here.
    chat_models: Annotated[list[str], NoDecode] = []
    guide_models: Annotated[list[str], NoDecode] = []
    # How much models reason before answering. default sends nothing, leaving it to the model.
    chat_thinking_level: ThinkingLevel = "minimal"
    guide_thinking_level: ThinkingLevel = "low"
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

    # The public demo. Off by default, and turned on by the Dockerfile. See app/demo.py.
    demo_mode: bool = False
    # Needed in demo mode to add, re-index or delete repos. Empty turns those off entirely.
    index_password: SecretStr = SecretStr("")
    max_repo_size_mb: int = Field(default=20, ge=1)
    # Gemini's free tier allows 1,000 embedded texts a day, so one repo can't be allowed to use them all.
    max_index_chunks: int = Field(default=400, ge=1)
    rate_limit_search_per_hour: int = Field(default=10, ge=1)
    rate_limit_ask_per_hour: int = Field(default=30, ge=1)
    rate_limit_guide_per_hour: int = Field(default=5, ge=1)
    rate_limit_index_per_hour: int = Field(default=10, ge=1)
    daily_request_cap: int = Field(default=100, ge=1)
    # The header a reverse proxy puts the real client IP in. Empty means there's no proxy.
    client_ip_header: str = ""

    @field_validator("database_url")
    @classmethod
    def use_the_psycopg_driver(cls, value: str) -> str:
        """Accept the URLs that hosts hand out, such as Neon's postgresql://, and select psycopg 3."""
        for prefix in ("postgresql://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql+psycopg://" + value.removeprefix(prefix)
        return value

    @field_validator("chat_models", "guide_models", mode="before")
    @classmethod
    def split_model_list(cls, value: Any) -> Any:
        """Turn `a, b,a` into `["a", "b"]`. Trying a model twice in one chain can't help."""
        if not isinstance(value, str):
            return value
        return list(dict.fromkeys(name.strip() for name in value.split(",") if name.strip()))

    @property
    def fastembed_dir(self) -> Path:
        """Where downloaded local embedding models are stored."""
        return self.data_dir / "fastembed"

    @property
    def repos_dir(self) -> Path:
        """Where cloned repos are stored."""
        return self.data_dir / "repos"


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance, creating its data directories."""
    settings = Settings()
    settings.repos_dir.mkdir(parents=True, exist_ok=True)
    return settings
