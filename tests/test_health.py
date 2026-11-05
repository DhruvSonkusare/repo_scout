"""Tests for GET /api/health against real Postgres."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.db import engine
from app.main import app

client = TestClient(app)


def test_health_ok_when_database_responds() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "ok", "models": "ok", "llm_provider": "fake"}


def test_health_reports_error_when_database_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # Same server, but a database that doesn't exist, so the failure comes straight from Postgres.
    missing_database = engine.url.set(database="repoguide_missing_database")
    monkeypatch.setattr("app.main.engine", create_engine(missing_database))

    response = client.get("/api/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    assert "repoguide_missing_database" in body["db"]
