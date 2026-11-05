"""Tests for application settings."""

from pathlib import Path

import pytest

from app.config import get_settings

pytestmark = pytest.mark.usefixtures("fresh_settings")


def test_get_settings_creates_data_directories(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    settings = get_settings()

    assert settings.repos_dir == tmp_path / "data" / "repos"
    assert settings.repos_dir.is_dir()
    assert settings.chroma_dir.is_dir()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_indexing_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    settings = get_settings()

    assert settings.chunk_size == 1500
    assert settings.chunk_overlap == 150
    assert settings.retrieval_k == 6
    assert settings.max_files == 1500
    assert settings.max_file_bytes == 200_000
