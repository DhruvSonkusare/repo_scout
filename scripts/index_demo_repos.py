"""Index the demo repos into Neon with the model the deployed app uses. Run this from your own machine.

See how many embeddings it would use, with no Gemini calls or database writes:
    python scripts/index_demo_repos.py --plan
Index them. This creates the tables if needed, and uses your GOOGLE_API_KEY's daily embedding quota:
    python scripts/index_demo_repos.py

It reads NEON_DATABASE_URL and GOOGLE_API_KEY from .env, and your own DATABASE_URL stays untouched. Repos
that are already indexed with this model are skipped, so running it again the next day picks up where
a used-up quota stopped it.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
# Running this file directly puts scripts/ on the import path, not the project root.
sys.path.insert(0, str(ROOT))

DEPLOYED_EMBEDDING_PROVIDER = "gemini"
DEPLOYED_EMBEDDING_MODEL = "gemini-embedding-001"
# Small repos, chosen by counting their chunks with --plan.
DEMO_REPOS = ["pallets/itsdangerous", "pallets/markupsafe", "pallets-eco/blinker"]
# Gemini's free tier allows 1,000 embedded texts a day. Questions and guides need some of those too.
MAX_TOTAL_CHUNKS = 700
INDEX_POLL_NOTE = "This pauses between batches to stay under Gemini's rate limits."


def configure_environment() -> str:
    """Point the app at Neon and the deployed model. This must run before anything imports the app."""
    url = os.environ.get("NEON_DATABASE_URL") or dotenv_values(ROOT / ".env").get("NEON_DATABASE_URL")
    if not url:
        raise SystemExit("Add NEON_DATABASE_URL to .env first. See docs/DEPLOY.md.")
    os.environ.update(
        {
            "DATABASE_URL": url,
            "LLM_PROVIDER": "fake",  # indexing never needs a chat model
            "EMBEDDING_PROVIDER": DEPLOYED_EMBEDDING_PROVIDER,
            "EMBEDDING_MODEL": DEPLOYED_EMBEDDING_MODEL,
            # A separate folder, because Neon's repo ids would overwrite the local clones with the same ids.
            "DATA_DIR": tempfile.mkdtemp(prefix="reposcout-demo-"),
        }
    )
    return urlsplit(url).hostname or "unknown host"


def count_chunks(repos: list[str]) -> dict[str, int]:
    """Clone each repo and count the chunks it would embed. This calls no model."""
    from app.ingest import clone_repo, collect_chunks, remove_clone

    counts: dict[str, int] = {}
    for offset, name in enumerate(repos, start=1):
        # High ids, so these clones can't be mistaken for real repos in the temporary folder.
        repo_id = 1_000_000 + offset
        chunks, _ = collect_chunks(clone_repo(f"https://github.com/{name}", repo_id))
        counts[name] = len(chunks)
        remove_clone(repo_id)
    return counts


def index_all(repos: list[str]) -> bool:
    """Index the repos into the database, and return True if all of them ended up ready."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import select

    from app.db import Repo, SessionLocal
    from app.ingest import index_repo
    from app.llm import embedding_identity

    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    all_ready = True
    with SessionLocal() as db:
        for name in repos:
            url = f"https://github.com/{name}"
            repo = db.scalar(select(Repo).where(Repo.url == url))
            if repo is not None and repo.status == "ready" and (repo.embedding_provider, repo.embedding_model) == embedding_identity():
                print(f"{name}: already indexed, skipping")
                continue
            if repo is None:
                repo = Repo(url=url, name=name)
                db.add(repo)
                db.commit()
            start = time.perf_counter()
            print(f"{name}: indexing. {INDEX_POLL_NOTE}", flush=True)
            index_repo(repo.id)
            db.refresh(repo)
            print(f"{name}: {repo.status} in {time.perf_counter() - start:.0f}s, {repo.chunks_embedded} chunks embedded")
            if repo.status != "ready":
                print(f"  {repo.error}")
                all_ready = False
    return all_ready


def main() -> int:
    """Check the plan fits the quota, then index. Exit with 0 only if every repo ended up ready."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="only count chunks, with no Gemini calls or database writes")
    parser.add_argument("--repo", action="append", metavar="OWNER/NAME", help="index this repo instead of the defaults")
    args = parser.parse_args()
    repos = args.repo or DEMO_REPOS

    host = configure_environment()
    from app.ingest import CloneError

    try:
        try:
            counts = count_chunks(repos)
        except CloneError as error:
            print(error)
            return 1
        for name, count in counts.items():
            print(f"{name}: {count} chunks")
        total = sum(counts.values())
        print(f"Total: {total} chunks, against a limit of {MAX_TOTAL_CHUNKS} (Gemini allows 1,000 embeddings a day).")
        if total > MAX_TOTAL_CHUNKS:
            print("That's too many, because questions and guides need embeddings too. Choose smaller repos.")
            return 1
        if args.plan:
            return 0
        print(f"Indexing into the database at {host}.")
        return 0 if index_all(repos) else 1
    finally:
        shutil.rmtree(os.environ["DATA_DIR"], ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
