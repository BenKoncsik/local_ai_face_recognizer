"""Dialogs for saving and loading server export profiles.

SaveDialog  — choose a name + optional PIN (non-macOS only), press Save.
LoadDialog  — pick an existing profile, authenticate, press Load.

On macOS with native Keychain helper:
  - Save: no PIN fields; Touch ID / Face ID is enforced by the OS on every read.
  - Load: clicking "Load" triggers the native OS auth prompt automatically.

On Windows / Linux / macOS fallback:
  - Save: optional app-level PIN (PBKDF2-hashed, stored in OS keyring).
  - Load: Windows Hello button (optional) or PIN field.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Optional

log = logging.getLogger(__name__)

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.services.server_settings_store import (
    ProfileMeta,
    ServerProfile,
    UserCancelledError,
    get_server_settings_store,
)
from app.ui.i18n import t


# ---------------------------------------------------------------------------
# Save dialog
# ---------------------------------------------------------------------------

class ServerProfileSaveDialog(QDialog):
    """Let the user name a profile and (on non-macOS) set an optional PIN."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("sps_title"))
        self.setMinimumWidth(400)

        self._store = get_server_settings_store()
        self._native_kc = self._store.uses_native_secure_keychain()
        self._existing = [p.name for p in self._store.list_profiles()]

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._name_combo = QComboBox()
        self._name_combo.setEditable(True)
        self._name_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for name in self._existing:
            self._name_combo.addItem(name)
        self._name_combo.setCurrentIndex(-1)
        self._name_combo.lineEdit().setPlaceholderText(t("sps_name_placeholder"))
        form.addRow(t("sps_name_label"), self._name_combo)
        layout.addLayout(form)

        if self._native_kc:
            # macOS with Secure Enclave Keychain: OS handles auth, no PIN needed
            info = QLabel(t("sps_macos_secure_info"))
            info.setWordWrap(True)
            info.setStyleSheet("color: #7fc; font-size: 11px; margin-top: 6px;")
            layout.addWidget(info)
        else:
            # Non-macOS: optional app-level PIN
            pin_form = QFormLayout()
            pin_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

            self._pin = QLineEdit()
            self._pin.setEchoMode(QLineEdit.EchoMode.Password)
            self._pin.setPlaceholderText(t("sps_pin_placeholder"))
            pin_form.addRow(t("sps_pin_label"), self._pin)

            self._pin_confirm = QLineEdit()
            self._pin_confirm.setEchoMode(QLineEdit.EchoMode.Password)
            self._pin_confirm.setPlaceholderText(t("sps_pin_confirm_placeholder"))
            pin_form.addRow(t("sps_pin_confirm_label"), self._pin_confirm)
            layout.addLayout(pin_form)

            info = QLabel(t("sps_pin_info"))
            info.setWordWrap(True)
            info.setStyleSheet("color: #aaa; font-size: 11px;")
            layout.addWidget(info)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        name = self._name_combo.currentText().strip()
        if not name:
            QMessageBox.warning(self, t("sps_title"), t("sps_err_no_name"))
            return
        if not self._native_kc:
            pin = self._pin.text()
            confirm = self._pin_confirm.text()
            if pin and pin != confirm:
                QMessageBox.warning(self, t("sps_title"), t("sps_err_pin_mismatch"))
                return
            if pin and len(pin) < 4:
                QMessageBox.warning(self, t("sps_title"), t("sps_err_pin_short"))
                return
        if name in self._existing:
            reply = QMessageBox.question(
                self, t("sps_title"), t("sps_overwrite_confirm", name=name),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.accept()

    @property
    def profile_name(self) -> str:
        return self._name_combo.currentText().strip()

    @property
    def pin(self) -> Optional[str]:
        if self._native_kc:
            return None
        p = self._pin.text()
        return p if p else None


# ---------------------------------------------------------------------------
# Load dialog
# ---------------------------------------------------------------------------

class ServerProfileLoadDialog(QDialog):
    """Profile selector with platform-appropriate auth."""

    profile_loaded = Signal(object)  # ServerProfile
    _load_result = Signal(object)    # ServerProfile | None | str("cancelled")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("spl_title"))
        self.setMinimumWidth(440)

        self._store = get_server_settings_store()
        self._native_kc = self._store.uses_native_secure_keychain()
        self._profiles: list[ProfileMeta] = self._store.list_profiles()
        self._loaded_profile: Optional[ServerProfile] = None

        # Connect the thread-safe result signal to the main-thread handler
        self._load_result.connect(self._on_load_result)

        layout = QVBoxLayout(self)

        if not self._profiles:
            layout.addWidget(QLabel(t("spl_no_profiles")))
            cancel = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            cancel.rejected.connect(self.reject)
            layout.addWidget(cancel)
            return

        # Profile selector
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._profile_combo = QComboBox()
        for p in self._profiles:
            label = f"{p.name}  🔒" if (p.has_pin and not self._native_kc) else p.name
            self._profile_combo.addItem(label, p.name)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        form.addRow(t("spl_select_label"), self._profile_combo)
        layout.addLayout(form)

        if self._native_kc:
            # macOS signed app: Touch ID is enforced by the OS on every keychain read
            info = QLabel(t("spl_macos_touchid_info"))
            info.setWordWrap(True)
            info.setStyleSheet("color: #7fc; font-size: 11px; margin-top: 6px;")
            layout.addWidget(info)
        else:
            # macOS (unsigned/dev), Windows, Linux: show biometric button + PIN field
            self._bio_widget = QWidget()
            bio_layout = QVBoxLayout(self._bio_widget)
            bio_layout.setContentsMargins(0, 8, 0, 0)
            if self._store.biometric_available():
                self._bio_btn = QPushButton(t("spl_biometric_btn"))
                self._bio_btn.setMinimumHeight(36)
                if sys.platform == "darwin":
                    self._bio_btn.clicked.connect(self._try_macos_touch_id)
                else:
                    self._bio_btn.clicked.connect(self._try_windows_hello)
                bio_layout.addWidget(self._bio_btn)
            self._bio_status = QLabel()
            self._bio_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._bio_status.setStyleSheet("color: #aaa; font-size: 11px;")
            bio_layout.addWidget(self._bio_status)
            layout.addWidget(self._bio_widget)

            sep = QLabel(t("spl_or_pin"))
            sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sep.setStyleSheet("color: #888; margin: 4px 0;")
            layout.addWidget(sep)
            self._sep_label = sep

            self._pin_widget = QWidget()
            pin_layout = QHBoxLayout(self._pin_widget)
            pin_layout.setContentsMargins(0, 0, 0, 0)
            self._pin_field = QLineEdit()
            self._pin_field.setEchoMode(QLineEdit.EchoMode.Password)
            self._pin_field.setPlaceholderText(t("spl_pin_placeholder"))
            self._pin_field.returnPressed.connect(self._verify_pin_and_load)
            pin_layout.addWidget(self._pin_field)
            layout.addWidget(self._pin_widget)

        # Button row
        btn_row = QHBoxLayout()
        self._delete_btn = QPushButton(t("spl_delete_btn"))
        self._delete_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()

        self._load_btn = QPushButton(t("spl_load_btn"))
        self._load_btn.setDefault(True)
        self._load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self._load_btn)

        cancel_btn = QPushButton(t("spl_cancel_btn"))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._on_profile_changed()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _current_meta(self) -> Optional[ProfileMeta]:
        idx = self._profile_combo.currentIndex()
        if 0 <= idx < len(self._profiles):
            return self._profiles[idx]
        return None

    def _on_profile_changed(self) -> None:
        meta = self._current_meta()
        if meta is None or self._native_kc:
            return
        has_pin = meta.has_pin
        self._bio_widget.setVisible(has_pin)
        self._sep_label.setVisible(has_pin)
        self._pin_widget.setVisible(has_pin)
        self._load_btn.setText(t("spl_load_pin_btn") if has_pin else t("spl_load_btn"))
        self._bio_status.clear()
        self._pin_field.clear()

    def _on_load_clicked(self) -> None:
        meta = self._current_meta()
        if meta is None:
            return
        if self._native_kc:
            self._load_with_native_keychain(meta.name)
        else:
            self._verify_pin_and_load()

    # ------------------------------------------------------------------
    # macOS native Keychain path (Touch ID triggered inside subprocess)
    # ------------------------------------------------------------------

    def _load_with_native_keychain(self, name: str) -> None:
        self._load_btn.setEnabled(False)
        self._load_btn.setText(t("spl_bio_trying"))
        QApplication.processEvents()

        reason = t("spl_bio_reason")

        def _worker() -> None:
            try:
                profile = self._store.load_profile(name, reason=reason)
                self._load_result.emit(profile)
            except UserCancelledError:
                self._load_result.emit("cancelled")
            except Exception as e:
                log.warning("Load profile error: %s", e)
                self._load_result.emit(None)

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_result(self, result: object) -> None:
        self._load_btn.setEnabled(True)
        self._load_btn.setText(t("spl_load_btn"))
        if result == "cancelled":
            if not self._native_kc:
                bio_status = getattr(self, "_bio_status", None)
                if bio_status:
                    bio_status.setText(t("spl_bio_failed"))
            # macOS: user explicitly cancelled Touch ID — don't show error, just stay open
            return
        if result is None:
            QMessageBox.critical(self, t("spl_title"), t("spl_err_load_failed"))
            return
        self._loaded_profile = result  # type: ignore[assignment]
        self.profile_loaded.emit(result)
        self.accept()

    # ------------------------------------------------------------------
    # macOS app-level Touch ID (unsigned / dev fallback)
    # ------------------------------------------------------------------

    def _try_macos_touch_id(self) -> None:
        self._bio_status.setText(t("spl_bio_trying"))
        self._bio_btn.setEnabled(False)
        QApplication.processEvents()

        def _run() -> None:
            ok = self._store.try_macos_touch_id(t("spl_bio_reason"))
            if ok:
                meta = self._current_meta()
                if meta:
                    profile = self._store.load_profile(meta.name)
                    self._load_result.emit(profile)
                else:
                    self._bio_btn.setEnabled(True)
            else:
                self._bio_status.setText(t("spl_bio_failed"))
                self._bio_btn.setEnabled(True)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Windows Hello path
    # ------------------------------------------------------------------

    def _try_windows_hello(self) -> None:
        self._bio_status.setText(t("spl_bio_trying"))
        self._bio_btn.setEnabled(False)
        QApplication.processEvents()

        def _run() -> None:
            ok = self._store.try_windows_hello(t("spl_bio_reason"))
            if ok:
                meta = self._current_meta()
                if meta:
                    profile = self._store.load_profile(meta.name)
                    self._load_result.emit(profile)
            else:
                self._bio_status.setText(t("spl_bio_failed"))
                self._bio_btn.setEnabled(True)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # PIN path
    # ------------------------------------------------------------------

    def _verify_pin_and_load(self) -> None:
        meta = self._current_meta()
        if meta is None:
            return
        if not meta.has_pin:
            profile = self._store.load_profile(meta.name)
            self._load_result.emit(profile)
            return
        pin = self._pin_field.text()
        if not pin:
            QMessageBox.warning(self, t("spl_title"), t("spl_err_enter_pin"))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            ok = self._store.verify_pin(meta.name, pin)
        finally:
            QApplication.restoreOverrideCursor()
        if ok:
            profile = self._store.load_profile(meta.name)
            self._load_result.emit(profile)
        else:
            QMessageBox.warning(self, t("spl_title"), t("spl_err_wrong_pin"))
            self._pin_field.selectAll()
            self._pin_field.setFocus()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete_selected(self) -> None:
        meta = self._current_meta()
        if meta is None:
            return
        reply = QMessageBox.question(
            self, t("spl_title"), t("spl_delete_confirm", name=meta.name),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._store.delete_profile(meta.name)
        idx = self._profile_combo.currentIndex()
        self._profile_combo.removeItem(idx)
        self._profiles.pop(idx)
        if not self._profiles:
            self.reject()

    @property
    def loaded_profile(self) -> Optional[ServerProfile]:
        return self._loaded_profile
