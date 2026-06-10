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
from app.utils.person_search import PersonEntry, person_is_unknown


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
                is_unknown=person_is_unknown(p),
            )
            for p in all_persons
            if p.id != source_person.id
        ]

        self._selector = PersonSearchSelect()
        self._selector.set_entries(entries)
        if match_scores:
            # Open best-match-first: the most likely target persons rise to the
            # top of the list so the user does not have to hunt through hundreds
            # of names.  default_sort only flips this selector's checkbox; the
            # user can still uncheck it to fall back to alphabetical order.
            self._selector.set_match_scores(match_scores, default_sort=True)
        # Accept the dialog when the user double-clicks a person in the list.
        # PersonSearchSelect emits person_double_clicked after the selection
        # has already been committed, so accepting here behaves like clicking
        # OK on a highlighted entry.
        self._selector.person_double_clicked.connect(self.accept)

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
