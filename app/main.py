"""FastAPI application entry point."""

import asyncio
import logging
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db import engine
from app.ingest import fail_interrupted_repos
from app.llm import close_models, model_chain_problem, validate_model_chains
from app.routers import auth, chat, finder, guide, repos

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def ignore_known_warnings() -> None:
    """Keep a harmless warning from a dependency out of the server log. pytest.ini has the same filter.

    langchain-core's _generate_response_from_error calls response.json() without await, for error
    metadata it only passes to callbacks. The Google client's async errors carry an aiohttp response,
    whose json() is a coroutine, so that coroutine is created and never run. It does no I/O and our
    error handling doesn't use it. langchain-core 1.7.6 and google-genai 2.34.0 have no fix to pin.
    Draft upstream report: docs/upstream-langchain-issue.md.
    """
    warnings.filterwarnings(
        "ignore", message="coroutine 'ClientResponse.json' was never awaited", category=RuntimeWarning
    )


ignore_known_warnings()

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


app = FastAPI(title="RepoScout", lifespan=lifespan)
app.include_router(auth.router)
app.include_router(repos.router)
app.include_router(chat.router)
app.include_router(guide.router)
app.include_router(finder.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def public_config() -> dict[str, bool]:
    """Tell the frontend what it needs to know about this deployment."""
    return {"demo_mode": get_settings().demo_mode}


@app.get("/api/health")
def health() -> JSONResponse:
    """Report whether the database is reachable and the model chains can run, and which LLM provider is in use."""
    db = "ok"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        # A connection error can name the database host and user, which a public demo shouldn't show.
        db = "unavailable" if get_settings().demo_mode else str(getattr(exc, "orig", exc)).strip()
    models = model_chain_problem() or "ok"
    is_ok = db == "ok" and models == "ok"
    return JSONResponse(
        {"status": "ok" if is_ok else "error", "db": db, "models": models, "llm_provider": get_settings().llm_provider},
        status_code=200 if is_ok else 503,
    )
