"""Tests for the secure credential store."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.gdrive.credential_store import (
    GoogleCredentialStore,
    InMemoryCredentialStore,
    StoredCredential,
)


@pytest.fixture
def cred() -> StoredCredential:
    return StoredCredential(
        account_email="alice@example.com",
        refresh_token="rt-secret-123",
        access_token="at-456",
        token_expiry="2030-01-01T00:00:00",
        client_id="cid",
        client_secret="csecret",
        scopes=["openid", "drive.file"],
    )


# ---------------------------------------------------------------------------
# In-memory store (used by the rest of the test suite as a fake)
# ---------------------------------------------------------------------------

class TestInMemoryStore:
    def test_roundtrip(self, cred: StoredCredential) -> None:
        store = InMemoryCredentialStore()
        store.save(cred)
        loaded = store.load("alice@example.com")
        assert loaded is not None
        assert loaded.refresh_token == cred.refresh_token

    def test_list_accounts(self, cred: StoredCredential) -> None:
        store = InMemoryCredentialStore()
        store.save(cred)
        accounts = list(store.list_accounts())
        assert accounts == ["alice@example.com"]

    def test_delete_is_idempotent(self) -> None:
        store = InMemoryCredentialStore()
        # No error on missing account
        store.delete("nonexistent@example.com")
        assert list(store.list_accounts()) == []


# ---------------------------------------------------------------------------
# Production store with file fallback
# ---------------------------------------------------------------------------

class TestGoogleCredentialStoreFallback:
    """All tests force fallback mode (no keyring) for determinism."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> GoogleCredentialStore:
        s = GoogleCredentialStore(data_dir=tmp_path)
        s._keyring_ok = False  # force fallback path
        return s

    def test_save_and_load(self, store: GoogleCredentialStore, cred: StoredCredential) -> None:
        store.save(cred)
        loaded = store.load(cred.account_email)
        assert loaded is not None
        assert loaded.refresh_token == cred.refresh_token
        assert loaded.scopes == cred.scopes

    def test_secret_not_in_plaintext_file(
        self, store: GoogleCredentialStore, cred: StoredCredential, tmp_path: Path
    ) -> None:
        store.save(cred)
        enc = (tmp_path / "gdrive_credentials.enc").read_bytes()
        assert b"rt-secret-123" not in enc

    def test_account_index_lists_email(
        self, store: GoogleCredentialStore, cred: StoredCredential
    ) -> None:
        store.save(cred)
        assert list(store.list_accounts()) == ["alice@example.com"]

    def test_delete_removes_credential(
        self, store: GoogleCredentialStore, cred: StoredCredential
    ) -> None:
        store.save(cred)
        store.delete(cred.account_email)
        assert store.load(cred.account_email) is None
        assert list(store.list_accounts()) == []

    def test_roundtrip_survives_new_instance(
        self, tmp_path: Path, cred: StoredCredential
    ) -> None:
        store1 = GoogleCredentialStore(data_dir=tmp_path)
        store1._keyring_ok = False
        store1.save(cred)

        # Brand-new instance — data must survive the process restart.
        store2 = GoogleCredentialStore(data_dir=tmp_path)
        store2._keyring_ok = False
        loaded = store2.load(cred.account_email)
        assert loaded is not None
        assert loaded.refresh_token == cred.refresh_token
