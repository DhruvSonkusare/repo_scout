"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.config import get_settings
from app.db import engine

app = FastAPI(title="RepoGuide")


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
