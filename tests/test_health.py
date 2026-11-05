"""Tests for GET /api/health against real Postgres."""

import warnings

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from app.db import engine
from app.main import app, ignore_known_warnings

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


def test_the_known_dependency_warning_is_ignored_and_others_are_not() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ignore_known_warnings()
        warnings.warn("coroutine 'ClientResponse.json' was never awaited", RuntimeWarning, stacklevel=1)
        warnings.warn("coroutine 'our_own_function' was never awaited", RuntimeWarning, stacklevel=1)

    assert [str(warning.message) for warning in caught] == ["coroutine 'our_own_function' was never awaited"]
