"""Dialog for choosing the destination of a batch face move.

The user can either pick an existing person (searchable list, reusing
:class:`PersonSearchSelect`) or type a name to create a brand-new person for the
selected faces.  The source person is excluded from the list so faces cannot be
"moved" onto themselves.  When the caller supplies face-match scores the selector
exposes the shared "order by similarity" toggle (see
:mod:`app.services.match_scoring`).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect


class MoveFacesDialog(QDialog):
    """Pick an existing person or name a new one as the move destination."""

    def __init__(
        self,
        face_count: int,
        persons: List[Person],
        exclude_person_id: Optional[int] = None,
        match_scores: Optional[Dict[int, float]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("move_faces_title"))
        self.setMinimumWidth(380)
        self._selected_person_id: Optional[int] = None
        self._new_person_name: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header = QLabel(t("move_faces_header", n=face_count))
        header.setWordWrap(True)
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        # --- Existing person selector ---
        layout.addWidget(QLabel(t("move_faces_pick_existing")))
        self._selector = PersonSearchSelect(self)
        candidates = [p for p in persons if p.id != exclude_person_id]
        self._selector.set_persons(candidates)
        # Offer the shared "order by face-match similarity" toggle when the
        # caller scored the moved faces against known people.
        if match_scores:
            self._selector.set_match_scores(match_scores)
        self._selector.person_selected.connect(self._on_person_selected)
        self._selector.person_double_clicked.connect(self._on_person_double_clicked)
        layout.addWidget(self._selector)

        # --- Separator ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # --- New person ---
        layout.addWidget(QLabel(t("move_faces_create_new")))
        self._new_name = QLineEdit()
        self._new_name.setPlaceholderText(t("move_faces_new_placeholder"))
        self._new_name.textChanged.connect(self._on_new_name_changed)
        layout.addWidget(self._new_name)

        # --- Buttons ---
        self._buttons = QDialogButtonBox()
        self._move_btn = self._buttons.addButton(
            t("move_faces_confirm"), QDialogButtonBox.AcceptRole
        )
        self._buttons.addButton(t("cancel"), QDialogButtonBox.RejectRole)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._move_btn.setEnabled(False)
        self._selector.focus_search()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _on_person_selected(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self._refresh_enabled()

    def _on_person_double_clicked(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self._on_accept()

    def _on_new_name_changed(self, text: str) -> None:
        # Typing a new name takes precedence and clears any list selection.
        if text.strip():
            self._selector.clear_selection()
            self._selected_person_id = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        has_target = bool(self._new_name.text().strip()) or (
            self._selected_person_id is not None
        )
        self._move_btn.setEnabled(has_target)

    def _on_accept(self) -> None:
        new_name = self._new_name.text().strip()
        if new_name:
            self._new_person_name = new_name
            self._selected_person_id = None
        elif self._selected_person_id is None:
            return  # nothing chosen; ignore the click
        self.accept()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def selected_person_id(self) -> Optional[int]:
        """The chosen existing person, or ``None`` if creating a new one."""
        return self._selected_person_id

    def new_person_name(self) -> Optional[str]:
        """The name for a new person, or ``None`` if an existing one was picked."""
        return self._new_person_name
