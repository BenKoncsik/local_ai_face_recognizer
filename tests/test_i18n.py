"""Tests for the minimal i18n module."""

from __future__ import annotations

import json

import pytest

import app.ui.i18n as i18n


@pytest.fixture(autouse=True)
def _reset_language(monkeypatch, tmp_path):
    """Keep tests isolated from the user's real language prefs."""
    i18n._lang = "en"
    prefs = tmp_path / "language_prefs.json"
    monkeypatch.setattr(i18n, "_prefs_file", lambda: prefs)
    yield
    i18n._lang = "en"


class TestTranslation:
    def test_t_english_default(self):
        assert i18n.t("stop") == "Stop"

    def test_t_hungarian(self):
        i18n.set_language("hu")
        assert i18n.t("stop") == "Leállítás"

    def test_t_with_format_kwargs(self):
        text = i18n.t("force_rescan_msg", n=12)
        assert "12" in text

    def test_unknown_key_returns_key(self):
        assert i18n.t("totally_missing_key_xyz") == "totally_missing_key_xyz"

    def test_current_language(self):
        i18n.set_language("hu")
        assert i18n.current_language() == "hu"


class TestLanguagePrefs:
    def test_set_language_persists(self):
        i18n.set_language("hu")
        assert i18n._prefs_file().exists()
        data = json.loads(i18n._prefs_file().read_text(encoding="utf-8"))
        assert data["language"] == "hu"

    def test_set_language_rejects_unknown(self):
        i18n.set_language("fr")
        assert i18n.current_language() == "en"

    def test_load_prefs_reads_file(self):
        i18n._prefs_file().write_text(
            json.dumps({"language": "hu"}), encoding="utf-8"
        )
        i18n.load_prefs()
        assert i18n.current_language() == "hu"
        assert i18n.t("settings") == "Beállítások"

    def test_load_prefs_ignores_invalid_language(self):
        i18n._prefs_file().write_text(
            json.dumps({"language": "xx"}), encoding="utf-8"
        )
        i18n.load_prefs()
        assert i18n.current_language() == "en"
