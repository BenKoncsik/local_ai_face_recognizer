"""Rename person dialog."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from app.ui.i18n import t


class RenameDialog(QDialog):
    """Simple dialog to rename a person / cluster."""

    def __init__(
        self,
        current_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("rename_person"))
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(t("rename_person_to", name=current_name)))

        self._input = QLineEdit(current_name)
        self._input.selectAll()
        layout.addWidget(self._input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def new_name(self) -> str:
        return self._input.text().strip()
