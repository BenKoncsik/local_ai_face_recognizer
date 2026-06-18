"""Tests for Google Drive OAuth client configuration."""

from __future__ import annotations

import importlib

import pytest

import app.gdrive.oauth_config as oauth_config


class TestBundledConstants:
    def test_scopes_include_drive_and_profile(self) -> None:
        assert "https://www.googleapis.com/auth/drive" in oauth_config.SCOPES
        assert "openid" in oauth_config.SCOPES
        assert "https://www.googleapis.com/auth/userinfo.email" in oauth_config.SCOPES

    def test_redirect_host_is_localhost(self) -> None:
        assert oauth_config.REDIRECT_HOST == "localhost"

    def test_client_id_is_non_empty(self) -> None:
        assert oauth_config.CLIENT_ID
        assert "apps.googleusercontent.com" in oauth_config.CLIENT_ID


class TestIsConfigured:
    def test_false_when_secret_is_placeholder(self) -> None:
        assert oauth_config.is_configured() is False

    def test_true_when_both_values_are_real(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FACE_LOCAL_GOOGLE_CLIENT_SECRET", "real-secret-value")
        importlib.reload(oauth_config)
        try:
            assert oauth_config.is_configured() is True
        finally:
            monkeypatch.delenv("FACE_LOCAL_GOOGLE_CLIENT_SECRET", raising=False)
            importlib.reload(oauth_config)


class TestClientConfig:
    def test_installed_client_shape(self) -> None:
        cfg = oauth_config.client_config()
        assert "installed" in cfg
        installed = cfg["installed"]
        assert installed["client_id"] == oauth_config.CLIENT_ID
        assert installed["client_secret"] == oauth_config.CLIENT_SECRET
        assert installed["auth_uri"] == "https://accounts.google.com/o/oauth2/auth"
        assert installed["token_uri"] == "https://oauth2.googleapis.com/token"
        assert installed["redirect_uris"] == [f"http://{oauth_config.REDIRECT_HOST}"]

    def test_env_override_for_client_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override = "override-id.apps.googleusercontent.com"
        monkeypatch.setenv("FACE_LOCAL_GOOGLE_CLIENT_ID", override)
        importlib.reload(oauth_config)
        try:
            assert oauth_config.CLIENT_ID == override
            assert oauth_config.client_config()["installed"]["client_id"] == override
        finally:
            monkeypatch.delenv("FACE_LOCAL_GOOGLE_CLIENT_ID", raising=False)
            importlib.reload(oauth_config)
