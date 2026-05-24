"""Settings dialog — language, database management, TPU status and updates."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSettings, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from app import __version__
from app.ui.i18n import SUPPORTED, current_language, set_language, t


def _qsettings() -> QSettings:
    return QSettings("FaceLocal", "FaceLocal")


class _TpuProbeThread(QThread):
    """Runs probe_tpu() in background so the dialog doesn't freeze."""

    result_ready = Signal(dict)

    def run(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import probe_tpu
        self.result_ready.emit(probe_tpu())


class SettingsDialog(QDialog):
    """Settings dialog: language, database management, and TPU status."""

    def __init__(self, current_db_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(520)
        self._current_db_path = current_db_path
        self._new_db_path: Optional[str] = None
        self._language_changed = False
        self._probe_thread: Optional[_TpuProbeThread] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        # ── Language ──────────────────────────────────────────────────────
        lang_group = QGroupBox(t("lang_label").rstrip(":"))
        lang_layout = QFormLayout(lang_group)

        self._lang_combo = QComboBox()
        for code, name in SUPPORTED.items():
            self._lang_combo.addItem(name, userData=code)
        idx = self._lang_combo.findData(current_language())
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        lang_layout.addRow(t("lang_label"), self._lang_combo)
        layout.addWidget(lang_group)

        # ── Database ──────────────────────────────────────────────────────
        db_group = QGroupBox(t("db_group"))
        db_layout = QVBoxLayout(db_group)

        cur_row = QHBoxLayout()
        cur_row.addWidget(QLabel(t("current_db")))
        self._db_label = QLineEdit(self._current_db_path)
        self._db_label.setReadOnly(True)
        self._db_label.setStyleSheet("color: #aaa;")
        cur_row.addWidget(self._db_label)
        db_layout.addLayout(cur_row)

        btn_row = QHBoxLayout()
        new_btn = QPushButton(t("new_db"))
        new_btn.clicked.connect(self._on_new_db)
        btn_row.addWidget(new_btn)

        open_btn = QPushButton(t("open_db"))
        open_btn.clicked.connect(self._on_open_db)
        btn_row.addWidget(open_btn)
        btn_row.addStretch()
        db_layout.addLayout(btn_row)
        layout.addWidget(db_group)

        # ── Updates ───────────────────────────────────────────────────────
        upd_group = QGroupBox(t("updates_group"))
        upd_layout = QVBoxLayout(upd_group)

        self._update_status_label = QLabel(t("current_version", version=__version__))
        self._update_status_label.setTextFormat(Qt.TextFormat.RichText)
        upd_layout.addWidget(self._update_status_label)

        self._notify_check = QCheckBox(t("notify_updates"))
        notify_enabled = _qsettings().value("updates/notify", True, type=bool)
        self._notify_check.setChecked(notify_enabled)
        upd_layout.addWidget(self._notify_check)

        upd_btn_row = QHBoxLayout()
        self._check_upd_btn = QPushButton(f"🔄  {t('check_updates_btn')}")
        self._check_upd_btn.clicked.connect(self._on_check_update)
        upd_btn_row.addWidget(self._check_upd_btn)
        upd_btn_row.addStretch()
        upd_layout.addLayout(upd_btn_row)
        layout.addWidget(upd_group)

        # ── TPU status ────────────────────────────────────────────────────
        tpu_group = QGroupBox(t("tpu_title"))
        tpu_layout = QVBoxLayout(tpu_group)

        self._tpu_summary_label = QLabel()
        tpu_layout.addWidget(self._tpu_summary_label)

        tpu_btn_row = QHBoxLayout()
        check_btn = QPushButton(t("tpu_status"))
        check_btn.clicked.connect(self._on_tpu_check)
        tpu_btn_row.addWidget(check_btn)

        self._tpu_fix_btn = QPushButton(f"🔧 {t('fix')}")
        self._tpu_fix_btn.clicked.connect(self._on_tpu_fix)
        self._tpu_fix_btn.setVisible(False)
        tpu_btn_row.addWidget(self._tpu_fix_btn)

        tpu_btn_row.addStretch()
        tpu_layout.addLayout(tpu_btn_row)
        layout.addWidget(tpu_group)

        self._start_tpu_probe()

        # ── Buttons ───────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ------------------------------------------------------------------
    # TPU
    # ------------------------------------------------------------------

    def _start_tpu_probe(self) -> None:
        """Launch TPU probe in background — dialog opens immediately."""
        self._tpu_summary_label.setText(f"⏳ {t('tpu_checking')}")
        self._tpu_summary_label.setStyleSheet("color: #888;")
        self._probe_thread = _TpuProbeThread(self)
        self._probe_thread.result_ready.connect(self._on_tpu_probe_done)
        self._probe_thread.start()

    def _on_tpu_probe_done(self, info: dict) -> None:
        ok = info["delegate_ok"]
        if ok:
            self._tpu_summary_label.setText(f"✓ {t('tpu_ok_label')}")
            self._tpu_summary_label.setStyleSheet("color: #4caf50;")
            self._tpu_fix_btn.setVisible(False)
        else:
            parts = []
            if not info["ai_edge_litert"]:
                parts.append(t("ai_edge_missing"))
            if not info["libedgetpu"]:
                parts.append(t("libedgetpu_missing_short"))
            if info["error"]:
                parts.append(info["error"])
            detail = "; ".join(parts) if parts else t("tpu_none")
            self._tpu_summary_label.setText(f"✗ {t('tpu_warn_label')} — {detail}")
            self._tpu_summary_label.setStyleSheet("color: #f57c00;")
            self._tpu_fix_btn.setVisible(True)

    def _on_tpu_check(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import TpuStatusDialog
        dlg = TpuStatusDialog(parent=self)
        dlg.exec()
        self._start_tpu_probe()

    def _on_tpu_fix(self) -> None:
        from app.ui.dialogs.tpu_status_dialog import TpuStatusDialog
        dlg = TpuStatusDialog(parent=self)
        dlg.exec()
        self._start_tpu_probe()

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    def _on_new_db(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, t("db_new_title"), str(Path.home() / "faces.db"),
            "SQLite (*.db *.sqlite)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    def _on_open_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t("db_open_title"), str(Path.home()),
            "SQLite (*.db *.sqlite);;All files (*)",
        )
        if path:
            self._new_db_path = path
            self._db_label.setText(path)

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _on_check_update(self) -> None:
        from app.services.update_service import fetch_latest_release, is_newer
        from app.ui.dialogs.update_dialog import UpdateDialog

        self._check_upd_btn.setEnabled(False)
        self._update_status_label.setText(f"⏳ {t('checking')}")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

        release = fetch_latest_release()
        self._check_upd_btn.setEnabled(True)

        if release is None:
            self._update_status_label.setText(
                f"⚠ {t('could_not_reach_github')}"
            )
            return

        if not is_newer(release.version, __version__):
            self._update_status_label.setText(
                f"✓ {t('up_to_date_version', version=__version__)}"
            )
            self._update_status_label.setStyleSheet("color: #4caf50;")
            return

        self._update_status_label.setText(
            f"🆕 {t('new_version_status', new=release.version, current=__version__)}"
        )
        self._update_status_label.setStyleSheet("color: #ffcc00;")
        dlg = UpdateDialog(release, parent=self)
        dlg.exec()

    def _on_accept(self) -> None:
        _qsettings().setValue("updates/notify", self._notify_check.isChecked())
        selected_lang = self._lang_combo.currentData()
        if selected_lang != current_language():
            set_language(selected_lang)
            self._language_changed = True
        self.accept()

    def selected_db_path(self) -> Optional[str]:
        return self._new_db_path

    def language_changed(self) -> bool:
        return self._language_changed
