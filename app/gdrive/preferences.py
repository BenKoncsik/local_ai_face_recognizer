"""Persistent Google Drive mode preferences.

Stored in :class:`QSettings` under the ``gdrive/`` namespace.  Only
non-secret values live here — actual OAuth tokens are owned by the
:class:`CredentialService`.  This module is intentionally tiny so the UI
layer never has to remember Qt keys directly.

Settings stored
---------------

* ``gdrive/enabled``        — Drive work mode is currently active.
* ``gdrive/account_email``  — Selected (signed-in) Google account.
* ``gdrive/folder_id``      — Drive folder id selected as project root.
* ``gdrive/folder_name``    — Cached display name for the folder.
* ``gdrive/selected_at``    — When the folder was picked (ISO timestamp).
* ``gdrive/db_sync``        — Auto-sync the DB on close (default: on).
* ``gdrive/last_sync_at``   — Last successful DB upload (ISO timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


def _qs():  # noqa: ANN202
    from app.app_settings import app_qsettings
    return app_qsettings()


@dataclass
class GDrivePrefs:
    """Snapshot of all persisted Drive preferences."""

    enabled: bool = False
    account_email: Optional[str] = None
    folder_id: Optional[str] = None
    folder_name: Optional[str] = None
    selected_at: Optional[str] = None
    db_sync_enabled: bool = True
    last_sync_at: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        """True when both an account and a folder are picked."""
        return bool(self.account_email) and bool(self.folder_id)


def load() -> GDrivePrefs:
    qs = _qs()
    return GDrivePrefs(
        enabled=bool(qs.value("gdrive/enabled", False, type=bool)),
        account_email=qs.value("gdrive/account_email", "", type=str) or None,
        folder_id=qs.value("gdrive/folder_id", "", type=str) or None,
        folder_name=qs.value("gdrive/folder_name", "", type=str) or None,
        selected_at=qs.value("gdrive/selected_at", "", type=str) or None,
        db_sync_enabled=bool(qs.value("gdrive/db_sync", True, type=bool)),
        last_sync_at=qs.value("gdrive/last_sync_at", "", type=str) or None,
    )


def save(prefs: GDrivePrefs) -> None:
    qs = _qs()
    qs.setValue("gdrive/enabled", prefs.enabled)
    qs.setValue("gdrive/account_email", prefs.account_email or "")
    qs.setValue("gdrive/folder_id", prefs.folder_id or "")
    qs.setValue("gdrive/folder_name", prefs.folder_name or "")
    qs.setValue("gdrive/selected_at", prefs.selected_at or "")
    qs.setValue("gdrive/db_sync", prefs.db_sync_enabled)
    qs.setValue("gdrive/last_sync_at", prefs.last_sync_at or "")


def update_last_sync(now_iso: Optional[str] = None) -> None:
    qs = _qs()
    qs.setValue(
        "gdrive/last_sync_at",
        now_iso or datetime.utcnow().isoformat(),
    )


def clear_account() -> None:
    """Forget the selected account + folder.  Leaves enabled flag alone."""
    qs = _qs()
    qs.setValue("gdrive/account_email", "")
    qs.setValue("gdrive/folder_id", "")
    qs.setValue("gdrive/folder_name", "")
    qs.setValue("gdrive/selected_at", "")
