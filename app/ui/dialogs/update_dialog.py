"""Update dialog — shows available release and handles download + apply."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import __version__
from app.services.update_service import ReleaseInfo, apply_update, download_asset
from app.ui.i18n import t

log = logging.getLogger(__name__)


class _DownloadThread(QThread):
    progress = Signal(int, int)       # downloaded_bytes, total_bytes
    download_complete = Signal(str)   # local file path — NOT named 'finished' to
                                      # avoid shadowing QThread.finished, which Qt
                                      # uses internally for thread lifecycle management.
                                      # Shadowing it prevents proper cleanup and causes
                                      # QThread::~QThread() to abort() on a live thread.
    error = Signal(str)

    def __init__(self, release: ReleaseInfo) -> None:
        super().__init__()
        self._release = release

    def run(self) -> None:
        try:
            log.info(
                "Update download thread started: version=%s asset=%s",
                self._release.version,
                self._release.asset_name,
            )
            path = download_asset(self._release, self.progress.emit)
            self.download_complete.emit(str(path))
        except Exception as exc:
            log.exception("Update download failed")
            self.error.emit(str(exc))


class UpdateDialog(QDialog):
    """Shows release info and handles download + apply.

    When ``auto_start=True`` the download begins immediately after the dialog
    is shown and the update is applied automatically when the download finishes
    — no extra button clicks needed.  This is the mode used when the user
    clicks the status-bar notification chip.

    When ``auto_start=False`` (default) the user initiates the download
    manually, which is the mode used from the Settings dialog.
    """

    def __init__(
        self,
        release: ReleaseInfo,
        parent: Optional[QWidget] = None,
        auto_start: bool = False,
    ) -> None:
        super().__init__(parent)
        self._release = release
        self._downloaded_path: Optional[Path] = None
        self._thread: Optional[_DownloadThread] = None
        self._auto_start = auto_start

        self.setWindowTitle(t("update_available_short", version=release.version))
        self.setMinimumWidth(460)
        self._build_ui()

        if auto_start:
            # Hide the manual download button — we start immediately.
            self._download_btn.setVisible(False)
            self._progress.setVisible(True)
            self._status_label.setVisible(True)
            self._status_label.setText(t("downloading"))
            # Defer one event-loop tick so the dialog is fully painted first.
            QTimer.singleShot(0, self._on_download)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Version info ---
        info = QLabel(
            f"<b>{t('update_available')}</b><br><br>"
            f"{t('current')}: &nbsp;<code>{__version__}</code><br>"
            f"{t('latest')}: &nbsp;&nbsp;<code>{self._release.version}</code><br><br>"
            f"{t('package')}: <code>{self._release.asset_name}</code>"
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        layout.addWidget(info)

        platform_map = {"darwin": "macOS", "win32": "Windows"}
        platform_name = platform_map.get(sys.platform, "Linux")
        note = QLabel(f"Platform: <b>{platform_name}</b>")
        note.setTextFormat(Qt.RichText)
        note.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(note)

        # --- Progress bar ---
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # --- Buttons ---
        btn_row = QHBoxLayout()

        self._download_btn = QPushButton(f"⬇  {t('download_install')}")
        self._download_btn.setDefault(True)
        self._download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(self._download_btn)

        is_macos_dmg = sys.platform == "darwin"
        apply_label = (f"▶  {t('update_restart')}"
                       if is_macos_dmg else
                       f"▶  {t('open_installer')}")
        self._apply_btn = QPushButton(apply_label)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(self._apply_btn)

        close_btn = QPushButton(t("skip"))
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _on_download(self) -> None:
        log.info(
            "Update download requested: version=%s asset=%s size=%d",
            self._release.version,
            self._release.asset_name,
            self._release.asset_size,
        )
        self._download_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._status_label.setVisible(True)
        self._status_label.setText(t("downloading"))

        self._thread = _DownloadThread(self._release)
        self._thread.progress.connect(self._on_progress)
        self._thread.download_complete.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        # Let Qt manage cleanup: delete the C++ object after the OS thread exits.
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self._progress.setValue(int(downloaded / total * 100))
        mb = downloaded / 1_048_576
        total_mb = total / 1_048_576 if total else 0
        if total_mb:
            self._status_label.setText(f"{mb:.1f} / {total_mb:.1f} MB")
        else:
            self._status_label.setText(f"{mb:.1f} MB")

    def _on_done(self, path: str) -> None:
        self._downloaded_path = Path(path)
        log.info("Update download ready for install: %s", self._downloaded_path)
        self._progress.setValue(100)
        self._apply_btn.setEnabled(True)

        # Wait for the OS thread to fully exit before we close the dialog or
        # call sys.exit().  download_complete is emitted from inside run(),
        # so the Qt thread may still be alive at the OS level when this slot
        # runs.  Without wait(), Python GC can destroy the QThread wrapper
        # while the thread is still winding down → QThread::~QThread() abort.
        if self._thread is not None and self._thread.isRunning():
            self._thread.wait(3000)

        if self._auto_start or sys.platform == "darwin":
            # Auto-apply: either the user clicked the notification chip
            # (auto_start) or we are on macOS where the shell-script installer
            # always runs without UAC prompts.
            self._status_label.setText(f"✓ {t('downloaded_updating')}")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 11px;")
            QApplication.processEvents()
            self._on_apply()
        else:
            self._status_label.setText(f"✓ {t('downloaded_click_install')}")
            self._status_label.setStyleSheet("color: #4caf50; font-size: 11px;")

    def _on_error(self, msg: str) -> None:
        log.error("Update dialog error: %s", msg)
        self._status_label.setText(f"✗ {t('error')}: {msg}")
        self._status_label.setStyleSheet("color: #f44336; font-size: 11px;")
        self._download_btn.setEnabled(True)

    def _on_apply(self) -> None:
        if self._downloaded_path:
            log.info("Update install/apply requested: %s", self._downloaded_path)
            try:
                apply_update(self._downloaded_path)
            except Exception as exc:
                log.exception("Update install/apply failed")
                self._status_label.setVisible(True)
                self._status_label.setText(f"✗ {t('error')}: {exc}")
                self._status_label.setStyleSheet("color: #f44336; font-size: 11px;")
                self._apply_btn.setEnabled(True)
                return
            self.accept()
