"""Manual check that the configured chat and embedding models respond."""

import sys
from pathlib import Path

# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import GEMINI_ERRORS, describe_gemini_error, get_chat_model, get_embeddings  # noqa: E402


def main() -> None:
    """Send one chat message, embed one sentence and print the results."""
    settings = get_settings()
    print(f"Chat provider: {settings.llm_provider}, embedding provider: {settings.embedding_provider}")

    try:
        # .text gives the reply as plain text, whichever content format the provider returns.
        reply = get_chat_model().invoke("Say hello in 5 words")
        print(f"Chat reply: {reply.text}")

        vector = get_embeddings().embed_query("RepoGuide finds open-source repos for your problem.")
        print(f"Embedding vector length: {len(vector)}")
    except ValueError as error:
        # The model factory raises this when a setting is missing from .env.
        sys.exit(f"Setup problem: {error}")
    except GEMINI_ERRORS as error:
        sys.exit(describe_gemini_error(error))


if __name__ == "__main__":
    main()
