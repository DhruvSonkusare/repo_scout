"""Manual check that the configured chat and embedding models respond."""

import sys
from pathlib import Path

import httpx
from google.genai.errors import APIError
from langchain_google_genai._common import GoogleGenerativeAIError

# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import get_chat_model, get_embeddings  # noqa: E402

GEMINI_ERROR_MESSAGES = {
    400: "Gemini rejected the request. Check GOOGLE_API_KEY and the model names in .env.",
    401: "Gemini rejected the API key. Check GOOGLE_API_KEY in .env.",
    403: "This API key isn't allowed to use that model. Check GOOGLE_API_KEY in .env.",
    404: "Gemini can't find that model. Run scripts/list_models.py and check the names in .env.",
    429: "Gemini's rate limit or quota was reached. Wait a minute and try again.",
}
GEMINI_BUSY_MESSAGE = "Gemini is busy right now. Try again shortly, or use a different CHAT_MODEL."


def describe_gemini_error(error: Exception) -> str:
    """Turn a Gemini or network error into a short message.

    Chat errors are themselves APIErrors, while embedding errors wrap one in __cause__.
    """
    api_error = error if isinstance(error, APIError) else error.__cause__
    if isinstance(api_error, APIError):
        if api_error.code >= 500:
            return f"{GEMINI_BUSY_MESSAGE} (HTTP {api_error.code})"
        known = GEMINI_ERROR_MESSAGES.get(api_error.code, "Gemini returned an error.")
        return f"{known} (HTTP {api_error.code})"
    if isinstance(error, httpx.HTTPError) or isinstance(error.__cause__, httpx.HTTPError):
        return "Couldn't reach Gemini. Check your internet connection."
    return "Gemini returned an unexpected error."


def main() -> None:
    """Send one chat message, embed one sentence and print the results."""
    settings = get_settings()
    print(f"Provider: {settings.llm_provider}")

    try:
        # .text gives the reply as plain text, whichever content format the provider returns.
        reply = get_chat_model().invoke("Say hello in 5 words")
        print(f"Chat reply: {reply.text}")

        vector = get_embeddings().embed_query("RepoGuide finds open-source repos for your problem.")
        print(f"Embedding vector length: {len(vector)}")
    except ValueError as error:
        # The model factory raises this when a setting is missing from .env.
        sys.exit(f"Setup problem: {error}")
    except (APIError, GoogleGenerativeAIError, httpx.HTTPError) as error:
        sys.exit(f"{describe_gemini_error(error)} Tried {settings.max_retries + 1} time(s).")


if __name__ == "__main__":
    main()
