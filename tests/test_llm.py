"""Tests for the model factory. Nothing here calls the network."""

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.llm import FAKE_CHAT_RESPONSES, FAKE_EMBEDDING_SIZE, get_chat_model, get_embeddings

pytestmark = pytest.mark.usefixtures("fresh_settings")


@pytest.fixture
def gemini_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Select the Gemini provider with placeholder values."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
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
