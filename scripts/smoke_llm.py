"""Manual check that the configured chat and embedding models respond."""

import sys
from pathlib import Path

# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.llm import get_chat_model, get_embeddings  # noqa: E402


def main() -> None:
    """Send one chat message, embed one sentence and print the results."""
    print(f"Provider: {get_settings().llm_provider}")

    # .text gives the reply as plain text, whichever content format the provider returns.
    reply = get_chat_model().invoke("Say hello in 5 words")
    print(f"Chat reply: {reply.text}")

    vector = get_embeddings().embed_query("RepoGuide finds open-source repos for your problem.")
    print(f"Embedding vector length: {len(vector)}")


if __name__ == "__main__":
    main()
