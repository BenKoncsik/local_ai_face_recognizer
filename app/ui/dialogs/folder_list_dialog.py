"""Dialog for managing a list of source folders to scan."""

from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from app.ui.i18n import t


class FolderListDialog(QDialog):
    """Dialog that lets the user add / remove source folders."""

    def __init__(self, folders: List[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("folder_list_title"))
        self.setMinimumWidth(520)
        self.setMinimumHeight(320)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        for folder in folders:
            self._add_item(folder)
        self._list.currentRowChanged.connect(self._on_selection_changed)

        self._hint = QLabel(t("folder_list_hint"))
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignLeft)

        self._btn_add = QPushButton(t("folder_list_add"))
        self._btn_remove = QPushButton(t("folder_list_remove"))
        self._btn_remove.setEnabled(False)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_remove.clicked.connect(self._on_remove)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_remove)
        btn_row.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._hint)
        layout.addWidget(self._list)
        layout.addLayout(btn_row)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------

    def folders(self) -> List[str]:
        """Return the current list of folder paths."""
        return [
            self._list.item(i).text()
            for i in range(self._list.count())
        ]

    # ------------------------------------------------------------------

    def _add_item(self, folder: str) -> None:
        item = QListWidgetItem(folder)
        item.setToolTip(folder)
        self._list.addItem(item)

    def _on_add(self) -> None:
        start = (
            self._list.item(self._list.count() - 1).text()
            if self._list.count()
            else str(Path.home())
        )
        folder = QFileDialog.getExistingDirectory(
            self, t("select_folder"), start
        )
        if not folder:
            return
        # Avoid duplicates
        existing = [self._list.item(i).text() for i in range(self._list.count())]
        if folder not in existing:
            self._add_item(folder)

    def _on_remove(self) -> None:
        row = self._list.currentRow()
        if row >= 0:
            self._list.takeItem(row)

    def _on_selection_changed(self, row: int) -> None:
        self._btn_remove.setEnabled(row >= 0)
