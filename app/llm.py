"""The only place chat and embedding models are created (gemini or fake)."""

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings, get_settings

FAKE_CHAT_RESPONSES = ["This is a fake answer from the offline test model [1]."]
FAKE_EMBEDDING_SIZE = 256


def _require_gemini_setting(value: str, env_name: str) -> str:
    """Return a Gemini setting, or explain which .env value is missing."""
    if not value:
        raise ValueError(f"{env_name} is not set. Add it to .env, or set LLM_PROVIDER=fake to run offline.")
    return value


def _gemini_api_key(settings: Settings) -> str:
    """Return the Gemini API key, or explain that it is missing."""
    return _require_gemini_setting(settings.google_api_key.get_secret_value(), "GOOGLE_API_KEY")


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model.

    Every LangChain chat model shares the BaseChatModel interface (invoke, stream, and use in
    LCEL pipes like `prompt | model | parser`). So chains never need to know which provider they
    got, and swapping Gemini for the fake only changes this function.
    """
    settings = get_settings()
    if settings.llm_provider == "fake":
        # FakeListChatModel replies with these strings in turn and never touches the network.
        return FakeListChatModel(responses=FAKE_CHAT_RESPONSES)
    return ChatGoogleGenerativeAI(
        model=_require_gemini_setting(settings.chat_model, "CHAT_MODEL"),
        google_api_key=_gemini_api_key(settings),
        temperature=0.2,
    )


def get_embeddings() -> Embeddings:
    """Return the configured embedding model.

    Embeddings turn text into a list of numbers, so a vector store can find similar text by
    comparing the numbers. A vector store must use one model for both indexing and searching,
    because vectors from different models can't be compared.
    """
    settings = get_settings()
    if settings.llm_provider == "fake":
        # The same text always gets the same vector, but the numbers don't capture meaning.
        return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE)
    return GoogleGenerativeAIEmbeddings(
        model=_require_gemini_setting(settings.embedding_model, "EMBEDDING_MODEL"),
        google_api_key=_gemini_api_key(settings),
    )
