"""Chroma helpers: one persistent client, and one collection per repo."""

from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma

from app.config import get_settings
from app.llm import get_embeddings


@lru_cache
def _client_for_path(path: str) -> ClientAPI:
    """Return the one Chroma client for a storage folder."""
    return chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))


def _get_client() -> ClientAPI:
    """Return the Chroma client for the configured data directory."""
    return _client_for_path(str(get_settings().chroma_dir))


def _collection_name(repo_id: int) -> str:
    """Return the name of a repo's collection."""
    return f"repo_{repo_id}"


def get_vectorstore(repo_id: int) -> Chroma:
    """Return a repo's vector store, creating its collection if needed.

    A vector store saves each chunk with its embedding, and searches by comparing embeddings.
    LangChain's Chroma wrapper calls the embedding model for us when documents are added or
    searched. Cosine similarity compares a vector's direction, not its length.
    """
    return Chroma(
        client=_get_client(),
        collection_name=_collection_name(repo_id),
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def delete_collection(repo_id: int) -> None:
    """Delete a repo's collection if it exists. Needs no embedding model or API key."""
    try:
        _get_client().delete_collection(_collection_name(repo_id))
    except NotFoundError:
        pass
