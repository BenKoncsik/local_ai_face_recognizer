"""Merge persons dialog."""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect
from app.utils.person_search import PersonEntry


class MergeDialog(QDialog):
    """Dialog to merge the currently selected person into another person."""

    def __init__(
        self,
        source_person: Person,
        all_persons: List[Person],
        parent: Optional[QWidget] = None,
        match_scores: Optional[Dict[int, float]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("merge_into"))
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(t("merge_source_into", name=source_person.name))
        )

        entries = [
            PersonEntry(
                person_id=p.id,
                name=p.name,
                display_text=t("merge_faces_count", name=p.name, n=len(p.faces)),
            )
            for p in all_persons
            if p.id != source_person.id
        ]

        self._selector = PersonSearchSelect()
        self._selector.set_entries(entries)
        if match_scores:
            self._selector.set_match_scores(match_scores)

        layout.addWidget(self._selector)
        layout.addWidget(
            QLabel(f"<small>{t('merge_source_deleted')}</small>")
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._selector.focus_search)

    def target_person_id(self) -> Optional[int]:
        """Return the selected target person ID, or ``None`` if empty."""
        return self._selector.current_person_id()
