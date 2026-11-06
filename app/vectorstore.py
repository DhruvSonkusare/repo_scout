"""pgvector helpers: vectors live in Postgres next to the rest of the data, with one collection per repo."""

from langchain_postgres import PGVector
from sqlalchemy import text

from app.db import engine
from app.llm import get_embeddings


def _collection_name(repo_id: int) -> str:
    """Return the name of a repo's collection."""
    return f"repo_{repo_id}"


def get_vectorstore(repo_id: int) -> PGVector:
    """Return a repo's vector store, creating its collection if needed.

    A vector store saves each chunk with its embedding, and searches by comparing embeddings.
    LangChain's PGVector calls the embedding model for us when documents are added or searched,
    and compares vectors by cosine similarity, which looks at a vector's direction, not its length.
    It shares the app's engine, so one small connection pool serves everything. Alembic owns
    the tables and the vector extension, so the check PGVector makes on startup finds nothing to do.
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=_collection_name(repo_id),
        connection=engine,
        use_jsonb=True,
        create_extension=False,
    )


def delete_collection(repo_id: int) -> None:
    """Delete a repo's collection if it exists, and its vectors with it. Needs no embedding model or API key."""
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM langchain_pg_collection WHERE name = :name"), {"name": _collection_name(repo_id)})
