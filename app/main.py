"""FastAPI application entry point."""

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
from app.routers import chat, guide, repos

logger = logging.getLogger(__name__)

# Uvicorn only configures its own loggers, so without this our INFO lines wouldn't appear.
logging.basicConfig(format="%(levelname)s:     %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Clean up after the previous run when the server starts."""
    try:
        fail_interrupted_repos()
    except SQLAlchemyError:
        # Still start, so /api/health can report the database problem.
        logger.warning("Could not check for interrupted repos: the database is unavailable.")
    yield


app = FastAPI(title="RepoGuide", lifespan=lifespan)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(guide.router)


@app.get("/api/health")
def health() -> JSONResponse:
    """Report whether the app can reach the database, and which LLM provider it uses."""
    llm_provider = get_settings().llm_provider
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        error = str(getattr(exc, "orig", exc)).strip()
        return JSONResponse({"status": "error", "db": error, "llm_provider": llm_provider}, status_code=503)
    return JSONResponse({"status": "ok", "db": "ok", "llm_provider": llm_provider})
