"""Shared test setup: every test runs against the real repoguide_test Postgres database."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# This must run before anything imports the app, because app.db builds its engine on import.
load_dotenv(PROJECT_ROOT / ".env")
test_database_url = os.environ.get("TEST_DATABASE_URL")
if not test_database_url:
    raise RuntimeError("Set TEST_DATABASE_URL in .env or the environment to run the tests.")
# Tests truncate every table, so refuse anything that isn't clearly a test database.
if not (make_url(test_database_url).database or "").endswith("_test"):
    raise RuntimeError("TEST_DATABASE_URL must point at a database whose name ends in _test.")
os.environ["DATABASE_URL"] = test_database_url
# Tests never call a real model, so they need no API key or quota.
os.environ["LLM_PROVIDER"] = "fake"

from app.config import get_settings  # noqa: E402
from app.db import Base, SessionLocal, engine  # noqa: E402


@pytest.fixture(scope="session")
def alembic_config() -> Config:
    """Alembic configuration for this project."""
    return Config(str(PROJECT_ROOT / "alembic.ini"))


@pytest.fixture(scope="session", autouse=True)
def migrated_database(alembic_config: Config) -> None:
    """Build the schema with the real migrations once per test session."""
    command.upgrade(alembic_config, "head")


@pytest.fixture(autouse=True)
def clean_tables(migrated_database: None) -> None:
    """Start each test with empty tables and ids counting from 1."""
    tables = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """Let a test change settings through the environment without affecting other tests."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def db_session() -> Iterator[Session]:
    """A session on the test database."""
    with SessionLocal() as session:
        yield session
