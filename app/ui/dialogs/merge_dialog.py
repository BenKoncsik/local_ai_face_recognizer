"""Merge persons dialog."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t


class MergeDialog(QDialog):
    """Dialog to merge the currently selected person into another person."""

    def __init__(
        self,
        source_person: Person,
        all_persons: List[Person],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("merge_into"))
        self.setMinimumWidth(380)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                t("merge_source_into", name=source_person.name)
            )
        )

        self._combo = QComboBox()
        for person in all_persons:
            if person.id == source_person.id:
                continue
            face_count = len(person.faces)
            self._combo.addItem(
                t("merge_faces_count", name=person.name, n=face_count), userData=person.id
            )

        layout.addWidget(self._combo)
        layout.addWidget(
            QLabel(
                f"<small>{t('merge_source_deleted')}</small>"
            )
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def target_person_id(self) -> Optional[int]:
        """Return the selected target person ID, or ``None`` if empty."""
        data = self._combo.currentData()
        return int(data) if data is not None else None
