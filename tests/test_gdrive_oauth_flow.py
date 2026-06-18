"""Tests for the Google OAuth installed-application flow."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.gdrive.connectivity import GDriveOfflineError
from app.gdrive.credential_store import StoredCredential
from app.gdrive.oauth_flow import (
    OAuthCancelled,
    OAuthConfigError,
    restore_credentials_object,
    run_login_flow,
)


@pytest.fixture()
def configured_oauth(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.gdrive.oauth_flow.oauth_config.is_configured", lambda: True)
    monkeypatch.setattr("app.gdrive.oauth_flow.require_online", lambda: None)


class TestRunLoginFlow:
    def test_raises_config_error_when_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.gdrive.oauth_flow.oauth_config.is_configured", lambda: False)
        with pytest.raises(OAuthConfigError, match="not configured"):
            run_login_flow()

    def test_raises_offline_error_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.gdrive.oauth_flow.oauth_config.is_configured", lambda: True)

        def _offline() -> None:
            raise GDriveOfflineError("offline")

        monkeypatch.setattr("app.gdrive.oauth_flow.require_online", _offline)
        with pytest.raises(GDriveOfflineError, match="offline"):
            run_login_flow()

    def test_raises_config_error_when_oauthlib_missing(
        self, configured_oauth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def _import(name, *args, **kwargs):
            if name == "google_auth_oauthlib.flow":
                raise ImportError("no oauthlib")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import)
        with pytest.raises(OAuthConfigError, match="google-auth-oauthlib"):
            run_login_flow()

    def test_success_returns_stored_credential(
        self, configured_oauth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        creds = SimpleNamespace(
            refresh_token="refresh",
            token="access",
            expiry=datetime(2030, 1, 1, tzinfo=timezone.utc),
            client_id="cid",
            client_secret="csecret",
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["openid", "drive"],
        )
        flow = MagicMock()
        flow.run_local_server.return_value = creds
        monkeypatch.setattr(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_config",
            lambda *_a, **_k: flow,
        )
        monkeypatch.setattr(
            "app.gdrive.oauth_flow._fetch_account_email",
            lambda _c: "alice@example.com",
        )

        stored = run_login_flow(timeout_seconds=30)
        assert isinstance(stored, StoredCredential)
        assert stored.account_email == "alice@example.com"
        assert stored.refresh_token == "refresh"
        assert stored.access_token == "access"
        flow.run_local_server.assert_called_once()

    def test_cancelled_flow_raises_oauth_cancelled(
        self, configured_oauth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = MagicMock()
        flow.run_local_server.side_effect = RuntimeError("User cancelled login")
        monkeypatch.setattr(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_config",
            lambda *_a, **_k: flow,
        )
        with pytest.raises(OAuthCancelled, match="cancelled"):
            run_login_flow()

    def test_generic_failure_raises_runtime_error(
        self, configured_oauth, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = MagicMock()
        flow.run_local_server.side_effect = RuntimeError("unexpected boom")
        monkeypatch.setattr(
            "google_auth_oauthlib.flow.InstalledAppFlow.from_client_config",
            lambda *_a, **_k: flow,
        )
        with pytest.raises(RuntimeError, match="OAuth flow failed"):
            run_login_flow()


class TestRestoreCredentialsObject:
    def test_rebuilds_google_credentials(self) -> None:
        stored = StoredCredential(
            account_email="alice@example.com",
            refresh_token="rt",
            access_token="at",
            token_expiry="2030-01-01T00:00:00",
            client_id="cid",
            client_secret="csecret",
            scopes=["openid"],
        )
        creds = restore_credentials_object(stored)
        assert creds.token == "at"
        assert creds.refresh_token == "rt"
        assert creds.client_id == "cid"
        assert creds.expiry is not None

    def test_invalid_expiry_is_ignored(self) -> None:
        stored = StoredCredential(
            account_email="alice@example.com",
            refresh_token="rt",
            token_expiry="not-a-date",
        )
        creds = restore_credentials_object(stored)
        assert creds.expiry is None


class TestFetchAccountEmail:
    def test_returns_email_from_userinfo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = MagicMock()
        service.userinfo.return_value.get.return_value.execute.return_value = {
            "email": "bob@example.com"
        }
        monkeypatch.setattr(
            "googleapiclient.discovery.build",
            lambda *_a, **_k: service,
        )
        from app.gdrive.oauth_flow import _fetch_account_email

        email = _fetch_account_email(MagicMock())
        assert email == "bob@example.com"

    def test_falls_back_when_userinfo_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "googleapiclient.discovery.build",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("api down")),
        )
        from app.gdrive.oauth_flow import _fetch_account_email

        assert _fetch_account_email(MagicMock()) == "unknown@gdrive"
