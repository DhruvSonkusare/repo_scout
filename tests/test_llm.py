"""Tests for the model factory. Nothing here calls the network."""

from collections.abc import Iterator

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.config import Settings
from app.llm import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    FAKE_CHAT_RESPONSES,
    FAKE_EMBEDDING_SIZE,
    _local_embeddings,
    get_chat_model,
    get_embeddings,
)

pytestmark = pytest.mark.usefixtures("fresh_settings")


class StubFastEmbed:
    """Stands in for FastEmbedEmbeddings, so tests don't download a 640 MB model."""

    def __init__(self, model_name: str, cache_dir: str) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Select local embeddings, with the stub instead of the real model."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setattr("app.llm.FastEmbedEmbeddings", StubFastEmbed)
    _local_embeddings.cache_clear()
    yield monkeypatch
    _local_embeddings.cache_clear()


@pytest.fixture
def gemini_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Select the Gemini provider with placeholder values."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODEL", "chat-model-name")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model-name")
    return monkeypatch


def test_tests_run_on_the_fake_provider() -> None:
    assert isinstance(get_chat_model(), FakeListChatModel)


def test_fake_chat_model_works_in_an_lcel_chain() -> None:
    chain = ChatPromptTemplate.from_template("{question}") | get_chat_model() | StrOutputParser()

    assert chain.invoke({"question": "Hello?"}) == FAKE_CHAT_RESPONSES[0]


def test_fake_chat_model_streams() -> None:
    tokens = [chunk.text for chunk in get_chat_model().stream("Hello?")]

    assert "".join(tokens) == FAKE_CHAT_RESPONSES[0]


def test_fake_embeddings_are_deterministic_and_sized() -> None:
    embeddings = get_embeddings()

    vector = embeddings.embed_query("same text")

    assert len(vector) == FAKE_EMBEDDING_SIZE
    assert vector == embeddings.embed_query("same text")
    assert vector != embeddings.embed_query("different text")


def test_local_is_the_default_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_PROVIDER")

    assert Settings(_env_file=None).embedding_provider == "local"


def test_embedding_provider_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "other")

    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER|embedding_provider"):
        Settings(_env_file=None)


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_default_to_the_code_model_in_the_data_folder() -> None:
    embeddings = get_embeddings()

    assert embeddings.model_name == DEFAULT_LOCAL_EMBEDDING_MODEL
    assert embeddings.cache_dir == str(get_settings().data_dir / "fastembed")


def test_embedding_model_setting_chooses_the_local_model(local_env: pytest.MonkeyPatch) -> None:
    local_env.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    assert get_embeddings().model_name == "BAAI/bge-small-en-v1.5"


@pytest.mark.usefixtures("local_env")
def test_local_model_is_only_loaded_once() -> None:
    assert get_embeddings() is get_embeddings()


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_need_no_gemini_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    assert isinstance(get_embeddings(), StubFastEmbed)


def test_gemini_model_name_on_the_local_provider_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # This is a real FastEmbed check: it rejects the name without downloading anything.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    _local_embeddings.cache_clear()

    with pytest.raises(ValueError, match="isn't a FastEmbed model"):
        get_embeddings()


@pytest.mark.usefixtures("gemini_env")
def test_gemini_chat_model_uses_settings() -> None:
    model = get_chat_model()

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("chat-model-name")
    assert model.temperature == 0.2
    assert model.google_api_key.get_secret_value() == "placeholder-key"
    assert model.max_retries == get_settings().max_retries + 1


@pytest.mark.usefixtures("gemini_env")
def test_gemini_embeddings_use_settings() -> None:
    embeddings = get_embeddings()

    assert isinstance(embeddings, GoogleGenerativeAIEmbeddings)
    assert embeddings.model.endswith("embedding-model-name")
    assert embeddings.google_api_key.get_secret_value() == "placeholder-key"


@pytest.mark.parametrize(
    ("missing", "factory"),
    [
        ("GOOGLE_API_KEY", get_chat_model),
        ("CHAT_MODEL", get_chat_model),
        ("GOOGLE_API_KEY", get_embeddings),
        ("EMBEDDING_MODEL", get_embeddings),
    ],
)
def test_missing_gemini_setting_names_the_env_variable(gemini_env, missing, factory) -> None:
    # An empty value overrides whatever the developer's own .env file contains.
    gemini_env.setenv(missing, "")

    with pytest.raises(ValueError, match=missing):
        factory()
