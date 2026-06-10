"""Manager dialog for the permanent face ignore list.

Lists every ignored face embedding with its thumbnail (when the crop file
still exists), the source person name snapshot and the ignore date, and lets
the user revoke entries so the face can be recognised again on the next run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.db.database import session_scope
from app.services.ignored_face_service import IgnoredFaceService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_THUMB_SIZE = 96


class IgnoredFacesDialog(QDialog):
    """Modal manager for permanently ignored faces."""

    def __init__(
        self,
        config: AppConfig,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._changed = False

        self.setWindowTitle(t("ignored_faces_title"))
        self.setMinimumSize(520, 420)
        self.resize(640, 560)
        self._build_ui()
        self._reload()

    def changed(self) -> bool:
        """True when at least one entry was revoked."""
        return self._changed

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._info_lbl = QLabel(t("ignored_faces_info"))
        self._info_lbl.setWordWrap(True)
        self._info_lbl.setStyleSheet("color: #A6ADC8;")
        layout.addWidget(self._info_lbl)

        self._list = QListWidget()
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.setSelectionMode(QListWidget.ExtendedSelection)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list, stretch=1)

        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet("color: #888;")
        layout.addWidget(self._count_lbl)

        btn_row = QHBoxLayout()
        self._unignore_btn = QPushButton(t("ignored_faces_unignore"))
        self._unignore_btn.setEnabled(False)
        self._unignore_btn.clicked.connect(self._on_unignore)
        btn_row.addWidget(self._unignore_btn)
        btn_row.addStretch()

        close_btn = QPushButton(t("scanModes.close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _reload(self) -> None:
        self._list.clear()
        with session_scope() as session:
            entries = IgnoredFaceService(session).list_ignored()
            for entry in entries:
                label_parts = [entry.source_person_name or t("ignored_faces_unknown_source")]
                if entry.created_at is not None:
                    label_parts.append(entry.created_at.strftime("%Y-%m-%d %H:%M"))
                if entry.note:
                    label_parts.append(entry.note)
                item = QListWidgetItem(" — ".join(label_parts))
                item.setData(Qt.UserRole, entry.id)
                if entry.thumbnail_path and Path(entry.thumbnail_path).exists():
                    pixmap = QPixmap(entry.thumbnail_path)
                    if not pixmap.isNull():
                        item.setIcon(QIcon(pixmap.scaled(
                            _THUMB_SIZE, _THUMB_SIZE,
                            Qt.KeepAspectRatio, Qt.SmoothTransformation,
                        )))
                self._list.addItem(item)

        n = self._list.count()
        self._count_lbl.setText(t("ignored_faces_count", n=n))
        if n == 0:
            empty = QListWidgetItem(t("ignored_faces_empty"))
            empty.setFlags(Qt.NoItemFlags)
            self._list.addItem(empty)
        self._unignore_btn.setEnabled(False)

    def _on_selection_changed(self) -> None:
        self._unignore_btn.setEnabled(bool(self._selected_ids()))

    def _selected_ids(self) -> list[int]:
        ids = []
        for item in self._list.selectedItems():
            entry_id = item.data(Qt.UserRole)
            if entry_id is not None:
                ids.append(int(entry_id))
        return ids

    def _on_unignore(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self,
            t("ignored_faces_title"),
            t("ignored_faces_unignore_confirm", n=len(ids)),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        with session_scope() as session:
            svc = IgnoredFaceService(session)
            for entry_id in ids:
                if svc.unignore(entry_id):
                    self._changed = True
        self._reload()
