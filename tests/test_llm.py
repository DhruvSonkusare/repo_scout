"""Tests for the model factory. Nothing here calls the network."""

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest
from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.config import Settings
from app.llm import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    FAKE_CHAT_RESPONSES,
    FAKE_EMBEDDING_SIZE,
    _build_chat_model,
    _local_embeddings,
    embedding_identity,
    get_chat_model,
    get_embeddings,
    get_guide_model,
)

pytestmark = pytest.mark.usefixtures("fresh_settings")


class StubTextEmbedding:
    """Stands in for FastEmbed's TextEmbedding, so tests need no model download or GPU."""

    session_provider = CUDA_PROVIDER  # the provider the loaded session reports
    load_error: Exception | None = None  # raised when loading on CUDA
    loads: list[list[str]] = []  # the providers requested for each load

    def __init__(self, model_name: str, cache_dir: str, providers: list[str]) -> None:
        StubTextEmbedding.loads.append(providers)
        if StubTextEmbedding.load_error is not None and CUDA_PROVIDER in providers:
            raise StubTextEmbedding.load_error
        self.model_name = model_name
        self.cache_dir = cache_dir
        active = StubTextEmbedding.session_provider if CUDA_PROVIDER in providers else CPU_PROVIDER
        self.model = SimpleNamespace(model=SimpleNamespace(get_providers=lambda: [active]))

    @staticmethod
    def list_supported_models() -> list[dict[str, str]]:
        return [{"model": DEFAULT_LOCAL_EMBEDDING_MODEL}, {"model": "BAAI/bge-small-en-v1.5"}]

    def embed(self, texts: list[str], **kwargs: object) -> list[np.ndarray]:
        return [np.array([0.5, 0.5]) for _ in texts]

    def query_embed(self, text: str, **kwargs: object) -> Iterator[np.ndarray]:
        yield np.array([0.5, 0.5])


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    """Select local embeddings on a machine with working CUDA, using the stub instead of a real model."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("EMBEDDING_DEVICE", "auto")
    monkeypatch.setattr("app.llm.TextEmbedding", StubTextEmbedding)
    monkeypatch.setattr("app.llm.ort.get_available_providers", lambda: [CUDA_PROVIDER, CPU_PROVIDER])
    monkeypatch.setattr(StubTextEmbedding, "session_provider", CUDA_PROVIDER)
    monkeypatch.setattr(StubTextEmbedding, "load_error", None)
    monkeypatch.setattr(StubTextEmbedding, "loads", [])
    _local_embeddings.cache_clear()
    yield monkeypatch
    _local_embeddings.cache_clear()


@pytest.fixture
def gemini_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Select the Gemini provider with placeholder values."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "placeholder-key")
    monkeypatch.setenv("CHAT_MODELS", "chat-model-name")
    monkeypatch.setenv("GUIDE_MODELS", "guide-model-name")
    monkeypatch.setenv("EMBEDDING_MODEL", "embedding-model-name")
    return monkeypatch


def test_tests_run_on_the_fake_provider() -> None:
    assert isinstance(_build_chat_model("any-name", True, "minimal"), FakeListChatModel)


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
    assert embeddings.model.cache_dir == str(get_settings().data_dir / "fastembed")


def test_embedding_model_setting_chooses_the_local_model(local_env: pytest.MonkeyPatch) -> None:
    local_env.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    assert get_embeddings().model_name == "BAAI/bge-small-en-v1.5"


@pytest.mark.usefixtures("local_env")
def test_local_model_is_only_loaded_once() -> None:
    assert get_embeddings() is get_embeddings()


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_need_no_gemini_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "")

    assert isinstance(get_embeddings().model, StubTextEmbedding)


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_embed_through_the_langchain_interface() -> None:
    embeddings = get_embeddings()

    assert isinstance(embeddings, Embeddings)
    assert embeddings.embed_documents(["one", "two"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert embeddings.embed_query("one") == [0.5, 0.5]


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_use_the_query_method_for_searches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(StubTextEmbedding, "query_embed", lambda self, text: iter([np.array([0.9, 0.1])]))
    embeddings = get_embeddings()

    assert embeddings.embed_query("one") == pytest.approx([0.9, 0.1])
    assert embeddings.embed_documents(["one"]) == [[0.5, 0.5]]


@pytest.mark.usefixtures("local_env")
def test_local_embeddings_handle_no_documents() -> None:
    assert get_embeddings().embed_documents([]) == []


def test_gemini_model_name_on_the_local_provider_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # This uses FastEmbed's real model list, and rejects the name without downloading anything.
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    _local_embeddings.cache_clear()

    with pytest.raises(ValueError, match="isn't a FastEmbed model"):
        get_embeddings()


def test_default_local_model_is_in_fastembeds_real_list() -> None:
    assert DEFAULT_LOCAL_EMBEDDING_MODEL in {model["model"] for model in TextEmbedding.list_supported_models()}


# --- EMBEDDING_DEVICE


def active_provider() -> str:
    """Return the provider of the session behind get_embeddings()."""
    return get_embeddings().model.model.model.get_providers()[0]


def test_auto_is_the_default_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)

    assert Settings(_env_file=None).embedding_device == "auto"


def test_embedding_device_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_DEVICE", "tpu")

    with pytest.raises(ValueError, match="embedding_device"):
        Settings(_env_file=None)


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_cuda_is_used_when_it_works_and_is_logged(
    local_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, device: str
) -> None:
    local_env.setenv("EMBEDDING_DEVICE", device)

    with caplog.at_level(logging.INFO, logger="app.llm"):
        assert active_provider() == CUDA_PROVIDER

    assert StubTextEmbedding.loads == [[CUDA_PROVIDER, CPU_PROVIDER]]
    assert "running on CUDA" in caplog.text


def test_cpu_device_never_tries_cuda(local_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    local_env.setenv("EMBEDDING_DEVICE", "cpu")

    with caplog.at_level(logging.INFO, logger="app.llm"):
        assert active_provider() == CPU_PROVIDER

    assert StubTextEmbedding.loads == [[CPU_PROVIDER]]
    assert "running on the CPU" in caplog.text


def break_cuda(env: pytest.MonkeyPatch, how: str) -> None:
    """Make CUDA unusable in one of the ways it can fail."""
    if how == "not_in_this_install":
        env.setattr("app.llm.ort.get_available_providers", lambda: [CPU_PROVIDER])
    elif how == "silently_fell_back_to_cpu":
        env.setattr(StubTextEmbedding, "session_provider", CPU_PROVIDER)
    else:
        env.setattr(StubTextEmbedding, "load_error", RuntimeError("out of GPU memory"))


CUDA_FAILURES = ["not_in_this_install", "silently_fell_back_to_cpu", "raises_on_load"]


@pytest.mark.parametrize("how", CUDA_FAILURES)
def test_auto_falls_back_to_the_cpu_with_a_log_line(
    local_env: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    break_cuda(local_env, how)

    assert active_provider() == CPU_PROVIDER
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "falling back to the CPU" in warnings[0]


@pytest.mark.parametrize("how", CUDA_FAILURES)
def test_explicit_cuda_fails_clearly_instead_of_using_the_cpu(local_env: pytest.MonkeyPatch, how: str) -> None:
    local_env.setenv("EMBEDDING_DEVICE", "cuda")
    break_cuda(local_env, how)

    with pytest.raises(ValueError, match="EMBEDDING_DEVICE=cuda, but CUDA isn't usable"):
        get_embeddings()


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [
        ("fake", "", ("fake", "deterministic-256")),
        ("local", "", ("local", DEFAULT_LOCAL_EMBEDDING_MODEL)),
        ("local", "BAAI/bge-small-en-v1.5", ("local", "BAAI/bge-small-en-v1.5")),
        ("gemini", "models/gemini-embedding-001", ("gemini", "models/gemini-embedding-001")),
    ],
)
def test_embedding_identity(
    monkeypatch: pytest.MonkeyPatch, provider: str, model: str, expected: tuple[str, str]
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", provider)
    monkeypatch.setenv("EMBEDDING_MODEL", model)

    assert embedding_identity() == expected


def test_embedding_identity_ignores_the_device(local_env: pytest.MonkeyPatch) -> None:
    # CUDA and CPU run the same model, so switching device mustn't force a re-index.
    identities = set()
    for device in ["cpu", "cuda", "auto"]:
        local_env.setenv("EMBEDDING_DEVICE", device)
        get_settings.cache_clear()
        identities.add(embedding_identity())

    assert len(identities) == 1


@pytest.mark.usefixtures("gemini_env")
def test_gemini_chat_model_uses_settings() -> None:
    model = _build_chat_model("chat-model-name", True, "minimal")

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model.endswith("chat-model-name")
    assert model.temperature == 0.2
    assert model.google_api_key.get_secret_value() == "placeholder-key"
    assert model.max_retries == get_settings().max_retries + 1


def test_minimal_is_the_default_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHAT_THINKING_LEVEL", raising=False)

    assert Settings(_env_file=None).chat_thinking_level == "minimal"


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high"])
@pytest.mark.usefixtures("gemini_env")
def test_thinking_level_reaches_the_gemini_model(level: str) -> None:
    assert _build_chat_model("gemini-3-flash", True, level).thinking_level == level


@pytest.mark.usefixtures("gemini_env")
def test_default_thinking_level_sends_nothing() -> None:
    assert _build_chat_model("gemini-3-flash", True, "default").thinking_level is None


def test_low_is_the_default_guide_thinking_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GUIDE_THINKING_LEVEL", raising=False)

    assert Settings(_env_file=None).guide_thinking_level == "low"


@pytest.mark.parametrize("setting", ["chat_thinking_level", "guide_thinking_level"])
def test_thinking_levels_reject_unknown_values(monkeypatch: pytest.MonkeyPatch, setting: str) -> None:
    monkeypatch.setenv(setting.upper(), "extreme")

    with pytest.raises(ValueError, match=setting):
        Settings(_env_file=None)


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
        ("CHAT_MODELS", get_chat_model),
        ("GOOGLE_API_KEY", get_guide_model),
        ("GUIDE_MODELS", get_guide_model),
        ("GOOGLE_API_KEY", get_embeddings),
        ("EMBEDDING_MODEL", get_embeddings),
    ],
)
def test_missing_gemini_setting_names_the_env_variable(gemini_env, missing, factory) -> None:
    # An empty value overrides whatever the developer's own .env file contains.
    gemini_env.setenv(missing, "")

    with pytest.raises(ValueError, match=missing):
        factory()
