"""FastAPI application entry point."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db import engine
from app.ingest import fail_interrupted_repos
from app.llm import close_models, model_chain_problem, validate_model_chains
from app.routers import chat, finder, guide, repos

logger = logging.getLogger(__name__)

# Uvicorn only configures its own loggers, so without this our INFO lines wouldn't appear.
logging.basicConfig(format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Clean up after the previous run and check the model names at startup, and close clients at shutdown."""
    try:
        fail_interrupted_repos()
    except SQLAlchemyError:
        # Still start, so /api/health can report the database problem.
        logger.warning("Could not check for interrupted repos: the database is unavailable.")
    await asyncio.to_thread(validate_model_chains)
    yield
    await close_models()


app = FastAPI(title="RepoGuide", lifespan=lifespan)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(guide.router)
app.include_router(finder.router)


@app.get("/api/health")
def health() -> JSONResponse:
    """Report whether the database is reachable and the model chains can run, and which LLM provider is in use."""
    db = "ok"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        db = str(getattr(exc, "orig", exc)).strip()
    models = model_chain_problem() or "ok"
    is_ok = db == "ok" and models == "ok"
    return JSONResponse(
        {"status": "ok" if is_ok else "error", "db": db, "models": models, "llm_provider": get_settings().llm_provider},
        status_code=200 if is_ok else 503,
    )
