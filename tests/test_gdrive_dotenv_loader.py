"""Tests for the tiny .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.gdrive.dotenv_loader import load_dotenv


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any test-relevant variables from the environment for each test."""
    for var in (
        "FACE_LOCAL_GOOGLE_CLIENT_ID",
        "FACE_LOCAL_GOOGLE_CLIENT_SECRET",
        "FACE_LOCAL_CONFIG",
    ):
        monkeypatch.delenv(var, raising=False)


class TestLoadDotenv:
    def test_missing_file_is_silent(self, tmp_path: Path) -> None:
        n = load_dotenv(tmp_path / "no-such.env")
        assert n == 0

    def test_loads_whitelisted_vars(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "FACE_LOCAL_GOOGLE_CLIENT_ID=id-from-file.apps.googleusercontent.com\n"
            "FACE_LOCAL_GOOGLE_CLIENT_SECRET=secret-from-file\n",
            encoding="utf-8",
        )
        n = load_dotenv(env)
        assert n == 2
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_ID"].startswith("id-from-file")
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_SECRET"] == "secret-from-file"

    def test_ignores_unknown_vars(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "FACE_LOCAL_GOOGLE_CLIENT_ID=ok\n"
            "AWS_SECRET_ACCESS_KEY=should-not-leak\n"
            "SOME_API_KEY=also-ignored\n",
            encoding="utf-8",
        )
        load_dotenv(env)
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_ID"] == "ok"
        assert "AWS_SECRET_ACCESS_KEY" not in os.environ
        assert "SOME_API_KEY" not in os.environ

    def test_does_not_overwrite_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACE_LOCAL_GOOGLE_CLIENT_SECRET", "from-shell")
        env = tmp_path / ".env"
        env.write_text("FACE_LOCAL_GOOGLE_CLIENT_SECRET=from-file\n", encoding="utf-8")
        load_dotenv(env)
        # Shell wins.
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_SECRET"] == "from-shell"

    def test_strips_quotes_and_export_prefix(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            'export FACE_LOCAL_GOOGLE_CLIENT_ID="quoted-id"\n'
            "export FACE_LOCAL_GOOGLE_CLIENT_SECRET='single-quoted'\n",
            encoding="utf-8",
        )
        load_dotenv(env)
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_ID"] == "quoted-id"
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_SECRET"] == "single-quoted"

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "\n"
            "# This is a comment\n"
            "FACE_LOCAL_GOOGLE_CLIENT_ID=value\n"
            "  \n"
            "# Another comment\n",
            encoding="utf-8",
        )
        n = load_dotenv(env)
        assert n == 1
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_ID"] == "value"

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            "this-is-not-key-value\n"
            "FACE_LOCAL_GOOGLE_CLIENT_ID=ok\n",
            encoding="utf-8",
        )
        n = load_dotenv(env)
        assert n == 1
        assert os.environ["FACE_LOCAL_GOOGLE_CLIENT_ID"] == "ok"
