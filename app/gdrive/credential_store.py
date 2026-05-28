"""Secure credential storage for Google Drive tokens.

Design goals
------------

* Tokens **must not** land in plain-text config files or QSettings.
* The UI layer never sees a raw refresh-token; it talks to the
  :class:`CredentialService` interface only.
* On platforms with a system keychain (macOS Keychain, Windows Credential
  Manager, Linux Secret Service / KWallet), use it via the ``keyring``
  package.
* When ``keyring`` cannot reach a backend (headless Linux, sandboxed
  environments), fall back to an **encrypted file** in the OS data dir
  using Fernet symmetric encryption.  The encryption key is itself stored
  in the keyring whenever possible — if even that fails, the key lives
  next to the file with file-mode ``0o600`` and a clear warning is logged.

The service is intentionally storage-agnostic: anything that satisfies the
:class:`CredentialService` interface can be plugged in (e.g. an in-memory
store for tests).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger(__name__)

_KEYRING_SERVICE = "face-local-gdrive"
_KEYRING_ENC_KEY_ITEM = "__fernet_key__"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class StoredCredential:
    """Serialisable Google OAuth credential record.

    Attributes mirror the fields ``google.oauth2.credentials.Credentials``
    uses, plus a few application-specific identifiers.
    """

    account_email: str           # Stable user identifier (also display name)
    refresh_token: str            # Long-lived; can mint new access tokens
    access_token: Optional[str] = None
    token_expiry: Optional[str] = None  # ISO-8601, UTC
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    token_uri: str = "https://oauth2.googleapis.com/token"
    scopes: List[str] = field(default_factory=list)
    saved_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "StoredCredential":
        return cls(**json.loads(raw))


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class CredentialService(ABC):
    """Abstract token store used by the Drive client and the UI.

    Implementations must be **thread-safe** because the OAuth refresh path
    is called from background workers.
    """

    @abstractmethod
    def save(self, cred: StoredCredential) -> None:
        """Persist *cred*, replacing any existing entry with the same email."""

    @abstractmethod
    def load(self, account_email: str) -> Optional[StoredCredential]:
        """Return the stored credential for *account_email*, or ``None``."""

    @abstractmethod
    def list_accounts(self) -> Iterable[str]:
        """Return the e-mail addresses of all stored accounts."""

    @abstractmethod
    def delete(self, account_email: str) -> None:
        """Remove the stored credential for *account_email* (idempotent)."""


# ---------------------------------------------------------------------------
# In-memory store (for tests)
# ---------------------------------------------------------------------------

class InMemoryCredentialStore(CredentialService):
    """Volatile store useful for tests; never persists anything to disk."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, StoredCredential] = {}

    def save(self, cred: StoredCredential) -> None:
        with self._lock:
            self._data[cred.account_email] = cred

    def load(self, account_email: str) -> Optional[StoredCredential]:
        with self._lock:
            return self._data.get(account_email)

    def list_accounts(self) -> Iterable[str]:
        with self._lock:
            return list(self._data.keys())

    def delete(self, account_email: str) -> None:
        with self._lock:
            self._data.pop(account_email, None)


# ---------------------------------------------------------------------------
# Production store: keyring + encrypted-file fallback
# ---------------------------------------------------------------------------

class GoogleCredentialStore(CredentialService):
    """Production credential service.

    Strategy:

    1. **Primary:** ``keyring`` (macOS Keychain / Windows Credential Manager
       / Linux Secret Service).  Each account email becomes a keyring item
       under the service name ``face-local-gdrive``.
    2. **Fallback:** Encrypted JSON file at
       :func:`app.paths.user_data_dir` / ``gdrive_credentials.enc``.
       The Fernet key is stored in the keyring (preferred) or alongside
       the file with mode 0600 (last-resort).

    The fallback is selected *transparently* when keyring is unavailable
    or its backend raises an error.  Accounts written to the keyring while
    available are mirrored to the fallback file on save, so a user who
    later loses keyring access can still recover sign-ins (the inverse is
    also true).
    """

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        from app.paths import user_data_dir
        self._lock = threading.Lock()
        self._data_dir = (data_dir or user_data_dir()).resolve()
        self._index_file = self._data_dir / "gdrive_accounts.json"
        self._enc_file = self._data_dir / "gdrive_credentials.enc"
        self._enc_key_file = self._data_dir / "gdrive_credentials.key"
        self._keyring_ok: Optional[bool] = None  # Lazily probed
        self._fernet = None  # Lazily created when fallback is used

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, cred: StoredCredential) -> None:
        with self._lock:
            self._ensure_dir()
            written_to_keyring = False
            if self._keyring_available():
                try:
                    self._keyring_set(cred.account_email, cred.to_json())
                    written_to_keyring = True
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Could not save credentials to keyring (%s) — "
                        "falling back to encrypted file.", exc,
                    )
                    self._keyring_ok = False
            # Mirror to encrypted file as a fallback / recovery path.
            self._fallback_save(cred)
            self._update_index(cred.account_email, add=True)
            log.info(
                "Saved credentials for %s (keyring=%s)",
                cred.account_email, written_to_keyring,
            )

    def load(self, account_email: str) -> Optional[StoredCredential]:
        with self._lock:
            if self._keyring_available():
                try:
                    raw = self._keyring_get(account_email)
                    if raw:
                        return StoredCredential.from_json(raw)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Keyring load failed for %s: %s", account_email, exc)
            return self._fallback_load(account_email)

    def list_accounts(self) -> Iterable[str]:
        with self._lock:
            try:
                if self._index_file.exists():
                    data = json.loads(self._index_file.read_text(encoding="utf-8"))
                    return list(data.get("accounts", []))
            except Exception as exc:  # noqa: BLE001
                log.debug("Could not read account index: %s", exc)
            return []

    def delete(self, account_email: str) -> None:
        with self._lock:
            if self._keyring_available():
                try:
                    self._keyring_delete(account_email)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Keyring delete failed for %s: %s", account_email, exc)
            self._fallback_delete(account_email)
            self._update_index(account_email, add=False)
            log.info("Removed credentials for %s", account_email)

    # ------------------------------------------------------------------
    # Keyring helpers
    # ------------------------------------------------------------------

    def _keyring_available(self) -> bool:
        if self._keyring_ok is not None:
            return self._keyring_ok
        try:
            import keyring  # noqa: F401
            from keyring.errors import KeyringError  # noqa: F401
            # Probe: try writing & reading a sentinel under a private name.
            import keyring as _kr
            sentinel_name = "__probe__"
            try:
                _kr.set_password(_KEYRING_SERVICE, sentinel_name, "ok")
                ok = _kr.get_password(_KEYRING_SERVICE, sentinel_name) == "ok"
                _kr.delete_password(_KEYRING_SERVICE, sentinel_name)
            except Exception as exc:  # noqa: BLE001
                log.debug("Keyring probe failed: %s", exc)
                ok = False
            self._keyring_ok = ok
        except ImportError:
            log.warning("`keyring` package not installed — using encrypted-file fallback.")
            self._keyring_ok = False
        if not self._keyring_ok:
            log.info("Using encrypted-file token store (keyring unavailable).")
        return self._keyring_ok

    def _keyring_set(self, account: str, value: str) -> None:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, account, value)

    def _keyring_get(self, account: str) -> Optional[str]:
        import keyring
        return keyring.get_password(_KEYRING_SERVICE, account)

    def _keyring_delete(self, account: str) -> None:
        import keyring
        from keyring.errors import PasswordDeleteError
        try:
            keyring.delete_password(_KEYRING_SERVICE, account)
        except PasswordDeleteError:
            pass

    # ------------------------------------------------------------------
    # Encrypted-file fallback
    # ------------------------------------------------------------------

    def _get_fernet(self):  # noqa: ANN001 — local import for optional dep
        if self._fernet is not None:
            return self._fernet
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:
            raise RuntimeError(
                "cryptography package required for fallback credential storage "
                "but not installed."
            ) from exc

        key = self._load_or_create_fernet_key()
        self._fernet = Fernet(key)
        return self._fernet

    def _load_or_create_fernet_key(self) -> bytes:
        from cryptography.fernet import Fernet

        # First preference: keyring-stored Fernet key, even when we can't
        # store the full token there (e.g. some keyring backends choke on
        # large values; a 44-byte key always fits).
        if self._keyring_available():
            try:
                import keyring
                existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ENC_KEY_ITEM)
                if existing:
                    return existing.encode("ascii")
                new_key = Fernet.generate_key()
                keyring.set_password(
                    _KEYRING_SERVICE, _KEYRING_ENC_KEY_ITEM, new_key.decode("ascii")
                )
                return new_key
            except Exception as exc:  # noqa: BLE001
                log.debug("Could not stash Fernet key in keyring: %s", exc)

        # Second preference: a 0600-mode file next to the encrypted blob.
        # This is the LEAST secure option; we log a warning so the user knows.
        if self._enc_key_file.exists():
            return self._enc_key_file.read_bytes()
        key = Fernet.generate_key()
        self._enc_key_file.write_bytes(key)
        try:
            os.chmod(self._enc_key_file, 0o600)
        except OSError:
            pass
        log.warning(
            "Stored Fernet key in plain file %s — install/enable a keyring "
            "backend for stronger protection.",
            self._enc_key_file,
        )
        return key

    def _fallback_load_all(self) -> dict[str, dict]:
        if not self._enc_file.exists():
            return {}
        try:
            fernet = self._get_fernet()
            data = fernet.decrypt(self._enc_file.read_bytes())
            return json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not decrypt fallback credentials: %s", exc)
            return {}

    def _fallback_write_all(self, store: dict[str, dict]) -> None:
        fernet = self._get_fernet()
        encoded = fernet.encrypt(json.dumps(store, ensure_ascii=False).encode("utf-8"))
        self._enc_file.write_bytes(encoded)
        try:
            os.chmod(self._enc_file, 0o600)
        except OSError:
            pass

    def _fallback_save(self, cred: StoredCredential) -> None:
        try:
            store = self._fallback_load_all()
            store[cred.account_email] = asdict(cred)
            self._fallback_write_all(store)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fallback save failed for %s: %s", cred.account_email, exc)

    def _fallback_load(self, account_email: str) -> Optional[StoredCredential]:
        store = self._fallback_load_all()
        raw = store.get(account_email)
        if raw is None:
            return None
        return StoredCredential(**raw)

    def _fallback_delete(self, account_email: str) -> None:
        try:
            store = self._fallback_load_all()
            if account_email in store:
                store.pop(account_email)
                self._fallback_write_all(store)
        except Exception as exc:  # noqa: BLE001
            log.warning("Fallback delete failed for %s: %s", account_email, exc)

    # ------------------------------------------------------------------
    # Account index (plain JSON — only e-mails, no secrets)
    # ------------------------------------------------------------------

    def _update_index(self, account_email: str, *, add: bool) -> None:
        try:
            data: dict = {}
            if self._index_file.exists():
                data = json.loads(self._index_file.read_text(encoding="utf-8"))
            accounts = set(data.get("accounts", []))
            if add:
                accounts.add(account_email)
            else:
                accounts.discard(account_email)
            data["accounts"] = sorted(accounts)
            self._index_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not update account index: %s", exc)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        self._data_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Optional[CredentialService] = None
_store_lock = threading.Lock()


def get_credential_store() -> CredentialService:
    """Return the application-wide credential store."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = GoogleCredentialStore()
    return _store


def set_credential_store(store: CredentialService) -> None:
    """Override the global store (used in tests)."""
    global _store
    with _store_lock:
        _store = store


# Suppress unused-import warning for the optional base64 import in case
# downstream modules want raw bytes access in the future.
_ = base64
