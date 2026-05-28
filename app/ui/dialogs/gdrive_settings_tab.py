"""Google Drive tab inside the Settings dialog.

Keeps all Drive-specific UI in one place so :file:`settings_dialog.py`
stays focused on the general / pairing / quality tabs.

Architecture rule
-----------------

This widget **never touches the Google API directly**.  It calls into
:mod:`app.gdrive.oauth_flow`, :mod:`app.gdrive.drive_client`,
:mod:`app.gdrive.preferences`, :mod:`app.gdrive.cache` and
:mod:`app.gdrive.credential_store`.  The OAuth dance and any network I/O
are off-loaded to background ``QThread`` workers, so the dialog never
freezes while the browser is open.
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gdrive import oauth_config, preferences
from app.gdrive.cache import get_cache_manager
from app.gdrive.connectivity import GDriveOfflineError, is_online
from app.gdrive.credential_store import StoredCredential, get_credential_store
from app.gdrive.folder_url import parse_folder_input
from app.gdrive.oauth_flow import OAuthCancelled, OAuthConfigError, run_login_flow
from app.ui.i18n import t

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _SignInThread(QThread):
    """Runs the blocking OAuth flow off the UI thread."""

    succeeded = Signal(object)  # StoredCredential
    failed = Signal(str)
    cancelled = Signal()

    def run(self) -> None:
        try:
            cred = run_login_flow()
            self.succeeded.emit(cred)
        except OAuthCancelled:
            self.cancelled.emit()
        except OAuthConfigError as exc:
            self.failed.emit(str(exc))
        except GDriveOfflineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Sign-in flow crashed")
            self.failed.emit(str(exc))


class _FolderProbeThread(QThread):
    """Verifies a Drive folder ID against the API, returning its name."""

    succeeded = Signal(str, str)  # (folder_id, folder_name)
    failed = Signal(str)

    def __init__(self, account_email: str, folder_id: str, parent=None) -> None:
        super().__init__(parent)
        self._account_email = account_email
        self._folder_id = folder_id

    def run(self) -> None:
        try:
            from app.gdrive.drive_client import build_drive_client
            client = build_drive_client(self._account_email)
            meta = client.get_metadata(self._folder_id)
            if not meta.is_folder:
                self.failed.emit(t("gdrive_folder_invalid"))
                return
            self.succeeded.emit(meta.file_id, meta.name)
        except GDriveOfflineError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Folder probe failed")
            self.failed.emit(t("gdrive_folder_unreachable", error=str(exc)))


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class GDriveSettingsTab(QWidget):
    """A self-contained widget plugged into :class:`SettingsDialog`."""

    # Emitted whenever the persisted Drive prefs change so the main window
    # can update its status bar in real time.
    prefs_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._prefs = preferences.load()
        self._signin_thread: Optional[_SignInThread] = None
        self._probe_thread: Optional[_FolderProbeThread] = None
        self._build()
        self._refresh_state()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(14)

        layout.addWidget(self._build_account_group())
        layout.addWidget(self._build_folder_group())
        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_status_group())
        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _build_account_group(self) -> QGroupBox:
        group = QGroupBox(t("gdrive_account_group"))
        v = QVBoxLayout(group)

        self._account_label = QLabel()
        self._account_label.setWordWrap(True)
        v.addWidget(self._account_label)

        row = QHBoxLayout()
        self._add_account_btn = QPushButton(t("gdrive_add_account_btn"))
        self._add_account_btn.clicked.connect(self._on_add_account)
        row.addWidget(self._add_account_btn)

        self._signout_btn = QPushButton(t("gdrive_signout_btn"))
        self._signout_btn.clicked.connect(self._on_sign_out)
        row.addWidget(self._signout_btn)
        row.addStretch()
        v.addLayout(row)
        return group

    def _build_folder_group(self) -> QGroupBox:
        group = QGroupBox(t("gdrive_folder_group"))
        v = QVBoxLayout(group)

        self._folder_label = QLabel()
        self._folder_label.setWordWrap(True)
        v.addWidget(self._folder_label)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel(t("gdrive_folder_id_hint")))
        self._folder_id_edit = QLineEdit()
        self._folder_id_edit.setReadOnly(True)
        self._folder_id_edit.setStyleSheet("color: #aaa;")
        id_row.addWidget(self._folder_id_edit, 1)
        v.addLayout(id_row)

        btn_row = QHBoxLayout()
        self._pick_folder_btn = QPushButton(t("gdrive_pick_folder_btn"))
        self._pick_folder_btn.clicked.connect(self._on_pick_folder)
        btn_row.addWidget(self._pick_folder_btn)

        self._clear_folder_btn = QPushButton(t("gdrive_clear_folder_btn"))
        self._clear_folder_btn.clicked.connect(self._on_clear_folder)
        btn_row.addWidget(self._clear_folder_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)
        return group

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox(t("gdrive_mode_group"))
        v = QVBoxLayout(group)

        self._mode_toggle = QCheckBox(t("gdrive_mode_toggle"))
        self._mode_toggle.setToolTip(t("gdrive_mode_tip"))
        self._mode_toggle.toggled.connect(self._on_toggle_mode)
        v.addWidget(self._mode_toggle)

        self._db_sync_toggle = QCheckBox(t("gdrive_db_sync_toggle"))
        self._db_sync_toggle.toggled.connect(self._on_toggle_db_sync)
        v.addWidget(self._db_sync_toggle)

        tip = QLabel(t("gdrive_mode_tip"))
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #888; font-size: 11px;")
        v.addWidget(tip)
        return group

    def _build_status_group(self) -> QGroupBox:
        group = QGroupBox(t("gdrive_status_group"))
        v = QVBoxLayout(group)

        self._online_label = QLabel()
        v.addWidget(self._online_label)

        self._summary_label = QLabel()
        self._summary_label.setWordWrap(True)
        v.addWidget(self._summary_label)

        self._last_sync_label = QLabel()
        v.addWidget(self._last_sync_label)

        btn_row = QHBoxLayout()
        self._sync_now_btn = QPushButton(t("gdrive_sync_now_btn"))
        self._sync_now_btn.setEnabled(False)  # wired up in main window later
        btn_row.addWidget(self._sync_now_btn)

        self._clear_cache_btn = QPushButton(t("gdrive_cache_clear_btn"))
        self._clear_cache_btn.clicked.connect(self._on_clear_cache)
        btn_row.addWidget(self._clear_cache_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)
        return group

    # ------------------------------------------------------------------
    # State synchronisation
    # ------------------------------------------------------------------

    def _refresh_state(self) -> None:
        prefs = preferences.load()
        self._prefs = prefs

        # Account row
        if prefs.account_email:
            self._account_label.setText(
                t("gdrive_account_signed_in", email=prefs.account_email)
            )
            self._signout_btn.setEnabled(True)
            self._add_account_btn.setText(t("gdrive_add_account_btn"))
        else:
            self._account_label.setText(t("gdrive_account_none"))
            self._signout_btn.setEnabled(False)

        # Folder row
        if prefs.folder_id:
            display = prefs.folder_name or prefs.folder_id
            self._folder_label.setText(t("gdrive_folder_selected", name=display))
            self._folder_id_edit.setText(prefs.folder_id)
            self._clear_folder_btn.setEnabled(True)
        else:
            self._folder_label.setText(t("gdrive_folder_none"))
            self._folder_id_edit.setText("")
            self._clear_folder_btn.setEnabled(False)

        self._pick_folder_btn.setEnabled(bool(prefs.account_email))

        # Mode toggles
        self._mode_toggle.blockSignals(True)
        self._db_sync_toggle.blockSignals(True)
        self._mode_toggle.setChecked(prefs.enabled)
        self._mode_toggle.setEnabled(prefs.is_ready)
        self._db_sync_toggle.setChecked(prefs.db_sync_enabled)
        self._db_sync_toggle.setEnabled(True)
        self._mode_toggle.blockSignals(False)
        self._db_sync_toggle.blockSignals(False)

        # Status
        online = is_online()
        self._online_label.setText(
            t("gdrive_status_online") if online else t("gdrive_status_offline")
        )
        self._online_label.setStyleSheet(
            "color: #4caf50;" if online else "color: #f57c00;"
        )
        if not prefs.is_ready:
            self._summary_label.setText(t("gdrive_status_incomplete"))
        else:
            self._summary_label.setText("")

        if prefs.last_sync_at:
            self._last_sync_label.setText(
                t("gdrive_last_sync", when=prefs.last_sync_at)
            )
        else:
            self._last_sync_label.setText(t("gdrive_last_sync_never"))

    # ------------------------------------------------------------------
    # Account handlers
    # ------------------------------------------------------------------

    def _on_add_account(self) -> None:
        if not oauth_config.is_configured():
            QMessageBox.warning(
                self,
                t("gdrive_oauth_not_configured_title"),
                t("gdrive_oauth_not_configured_msg"),
            )
            return
        if not is_online():
            QMessageBox.warning(
                self, t("gdrive_error_title"), t("gdrive_offline_error")
            )
            return

        self._add_account_btn.setEnabled(False)
        self._account_label.setText(t("gdrive_signin_in_progress"))

        thread = _SignInThread(self)
        thread.succeeded.connect(self._on_sign_in_ok)
        thread.failed.connect(self._on_sign_in_failed)
        thread.cancelled.connect(self._on_sign_in_cancelled)
        thread.finished.connect(lambda: self._add_account_btn.setEnabled(True))
        self._signin_thread = thread
        thread.start()

    def _on_sign_in_ok(self, cred: StoredCredential) -> None:
        get_credential_store().save(cred)
        prefs = preferences.load()
        prefs.account_email = cred.account_email
        preferences.save(prefs)
        log.info("Signed in: %s", cred.account_email)
        QMessageBox.information(
            self, t("gdrive_info_title"),
            t("gdrive_signin_ok", email=cred.account_email),
        )
        self._refresh_state()
        self.prefs_changed.emit()

    def _on_sign_in_failed(self, error: str) -> None:
        log.warning("Sign-in failed: %s", error)
        QMessageBox.warning(
            self, t("gdrive_error_title"),
            t("gdrive_signin_failed", error=error),
        )
        self._refresh_state()

    def _on_sign_in_cancelled(self) -> None:
        QMessageBox.information(
            self, t("gdrive_info_title"), t("gdrive_signin_cancelled")
        )
        self._refresh_state()

    def _on_sign_out(self) -> None:
        prefs = preferences.load()
        if not prefs.account_email:
            return
        reply = QMessageBox.question(
            self,
            t("gdrive_signout_confirm_title"),
            t("gdrive_signout_confirm_msg", email=prefs.account_email),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            get_credential_store().delete(prefs.account_email)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not delete stored credential: %s", exc)
        preferences.clear_account()
        # Drive mode requires both account + folder — switch off automatically.
        new_prefs = preferences.load()
        if new_prefs.enabled:
            new_prefs.enabled = False
            preferences.save(new_prefs)
        self._refresh_state()
        self.prefs_changed.emit()

    # ------------------------------------------------------------------
    # Folder handlers
    # ------------------------------------------------------------------

    def _on_pick_folder(self) -> None:
        prefs = preferences.load()
        if not prefs.account_email:
            return
        if not is_online():
            QMessageBox.warning(
                self, t("gdrive_error_title"), t("gdrive_offline_error")
            )
            return

        text, ok = QInputDialog.getText(
            self,
            t("gdrive_pick_folder_title"),
            t("gdrive_pick_folder_intro"),
            QLineEdit.Normal,
            "",
        )
        if not ok:
            return
        folder_id = parse_folder_input(text)
        if folder_id is None:
            QMessageBox.warning(
                self, t("gdrive_error_title"), t("gdrive_folder_invalid")
            )
            return

        self._pick_folder_btn.setEnabled(False)
        self._folder_label.setText(t("gdrive_signin_in_progress"))

        thread = _FolderProbeThread(prefs.account_email, folder_id, self)
        thread.succeeded.connect(self._on_folder_probe_ok)
        thread.failed.connect(self._on_folder_probe_failed)
        thread.finished.connect(lambda: self._pick_folder_btn.setEnabled(True))
        self._probe_thread = thread
        thread.start()

    def _on_folder_probe_ok(self, folder_id: str, folder_name: str) -> None:
        from datetime import datetime

        prefs = preferences.load()
        prefs.folder_id = folder_id
        prefs.folder_name = folder_name
        prefs.selected_at = datetime.utcnow().isoformat()
        preferences.save(prefs)
        log.info("Drive folder selected: %s (%s)", folder_name, folder_id)
        QMessageBox.information(
            self, t("gdrive_info_title"),
            t("gdrive_folder_selected", name=folder_name),
        )
        self._refresh_state()
        self.prefs_changed.emit()

    def _on_folder_probe_failed(self, error: str) -> None:
        log.warning("Drive folder probe failed: %s", error)
        QMessageBox.warning(self, t("gdrive_error_title"), error)
        self._refresh_state()

    def _on_clear_folder(self) -> None:
        prefs = preferences.load()
        prefs.folder_id = None
        prefs.folder_name = None
        prefs.selected_at = None
        if prefs.enabled:
            prefs.enabled = False  # mode needs a folder
        preferences.save(prefs)
        self._refresh_state()
        self.prefs_changed.emit()

    # ------------------------------------------------------------------
    # Mode toggles
    # ------------------------------------------------------------------

    def _on_toggle_mode(self, checked: bool) -> None:
        prefs = preferences.load()
        # The checkbox is only enabled when prefs.is_ready, but defend anyway.
        if checked and not prefs.is_ready:
            self._mode_toggle.setChecked(False)
            return
        prefs.enabled = bool(checked)
        preferences.save(prefs)
        log.info("Drive mode %s", "enabled" if checked else "disabled")
        self.prefs_changed.emit()

    def _on_toggle_db_sync(self, checked: bool) -> None:
        prefs = preferences.load()
        prefs.db_sync_enabled = bool(checked)
        preferences.save(prefs)

    # ------------------------------------------------------------------
    # Cache control
    # ------------------------------------------------------------------

    def _on_clear_cache(self) -> None:
        n = get_cache_manager().shutdown_cleanup()
        QMessageBox.information(
            self, t("gdrive_info_title"),
            t("gdrive_cache_cleared", n=n),
        )
