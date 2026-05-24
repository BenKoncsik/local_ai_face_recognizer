"""Image Library dialogs.

* :class:`ImageLibraryMissingDialog`  — startup warning when the library
  root is configured but no longer exists on disk.
* :class:`MigrateLibraryDialog`       — progress dialog for the path
  migration that converts absolute paths to relative paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from app.ui.i18n import t

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Missing library root dialog
# ---------------------------------------------------------------------------

class ImageLibraryMissingDialog(QDialog):
    """Warn the user that the configured library root no longer exists.

    The user can pick a new root folder or skip (continue without resolution).
    """

    def __init__(
        self,
        missing_path: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("img_lib_missing_title"))
        self.setMinimumWidth(480)
        self._new_root: Optional[str] = None
        self._build_ui(missing_path)

    def _build_ui(self, missing_path: str) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        icon_row = QHBoxLayout()
        warn_icon = QLabel("⚠")
        warn_icon.setStyleSheet("font-size: 32px; color: #f57c00;")
        icon_row.addWidget(warn_icon)

        msg_label = QLabel(t("img_lib_missing_msg", path=missing_path))
        msg_label.setWordWrap(True)
        icon_row.addWidget(msg_label, 1)
        layout.addLayout(icon_row)

        btn_row = QHBoxLayout()
        find_btn = QPushButton(t("img_lib_find_btn"))
        find_btn.clicked.connect(self._on_find)
        btn_row.addWidget(find_btn)

        skip_btn = QPushButton(t("img_lib_skip_btn"))
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(skip_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _on_find(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            t("img_lib_select_title"),
            str(Path.home()),
        )
        if folder:
            self._new_root = folder
            self.accept()

    def new_root(self) -> Optional[str]:
        """Return the selected root path, or ``None`` if the user skipped."""
        return self._new_root


# ---------------------------------------------------------------------------
# Migration worker thread
# ---------------------------------------------------------------------------

class _MigrationThread(QThread):
    """Runs path migration in a background thread."""

    progress = Signal(int, int, str)   # current, total, path
    finished = Signal(object)          # MigrationResult

    def __init__(self, db_path: str, library_root: str, parent=None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._library_root = library_root

    def run(self) -> None:
        from app.db.database import session_scope
        from app.services.image_library_service import ImageLibraryService

        svc = ImageLibraryService(self._db_path)
        with session_scope() as session:
            result = svc.migrate_to_relative_paths(
                session=session,
                library_root=self._library_root,
                validate_files=True,
                progress_cb=lambda c, t, p: self.progress.emit(c, t, p),
            )
        self.finished.emit(result)


# ---------------------------------------------------------------------------
# Migration dialog
# ---------------------------------------------------------------------------

class MigrateLibraryDialog(QDialog):
    """Runs the path migration and shows progress.

    The dialog runs migration in a background thread so the GUI stays
    responsive, then displays a summary on completion.
    """

    def __init__(
        self,
        db_path: str,
        library_root: str,
        total_images: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("img_lib_migrate_title"))
        self.setMinimumWidth(500)
        self._db_path = db_path
        self._library_root = library_root
        self._total_images = total_images
        self._thread: Optional[_MigrationThread] = None
        self._build_ui()
        self._start_migration()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._status_label = QLabel(t("img_lib_migrating"))
        layout.addWidget(self._status_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, max(1, self._total_images))
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._detail_label = QLabel()
        self._detail_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._detail_label)

        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(140)
        self._log_box.setVisible(False)
        layout.addWidget(self._log_box)

        self._btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        self._btn_box.rejected.connect(self.reject)
        self._btn_box.setEnabled(False)
        layout.addWidget(self._btn_box)

    def _start_migration(self) -> None:
        self._thread = _MigrationThread(
            db_path=self._db_path,
            library_root=self._library_root,
            parent=self,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, current: int, total: int, path: str) -> None:
        self._progress_bar.setValue(current)
        if path:
            self._detail_label.setText(Path(path).name)

    def _on_finished(self, result) -> None:
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._btn_box.setEnabled(True)

        summary = t(
            "img_lib_migrate_done",
            migrated=result.migrated,
            skipped=result.skipped,
        )
        self._status_label.setText(summary)
        self._detail_label.clear()

        if result.errors:
            self._log_box.setVisible(True)
            self._log_box.setPlainText(
                t("img_lib_migrate_errors", n=len(result.errors))
                + "\n\n"
                + "\n".join(result.errors[:50])
            )
