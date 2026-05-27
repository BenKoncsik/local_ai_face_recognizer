"""Person info dialog — edit structured personal data for a recognised person."""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCompleter,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.ui.i18n import t

log = logging.getLogger(__name__)


class PersonInfoDialog(QDialog):
    """Edit structured personal data for a person."""

    def __init__(self, person: Person, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person.id
        self.setWindowTitle(t("person_info_title", name=person.name))
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{person.name}</b>")
        title.setStyleSheet("font-size: 14px; margin-bottom: 6px;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(form.labelAlignment())
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._gender = QComboBox()
        self._gender.addItem(t("gender_unknown"), None)
        self._gender.addItem(t("gender_male"), "male")
        self._gender.addItem(t("gender_female"), "female")
        idx = self._gender.findData(person.gender)
        self._gender.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow(t("gender"), self._gender)

        self._family_code = QLineEdit(person.family_code or "")
        self._family_code.setPlaceholderText(t("example_family_code"))
        self._family_code.setToolTip(t("family_code_help"))
        form.addRow(t("family_code"), self._family_code)

        family_help = QLabel(t("family_code_help"))
        family_help.setWordWrap(True)
        family_help.setStyleSheet("color: #888; font-size: 11px;")
        form.addRow("", family_help)

        self._last_name = QLineEdit(person.last_name or "")
        self._last_name.setPlaceholderText(t("example_last_name"))
        form.addRow(t("last_name"), self._last_name)

        self._first_name = QLineEdit(person.first_name or "")
        self._first_name.setPlaceholderText(t("example_first_name"))
        form.addRow(t("first_name"), self._first_name)

        self._second_name = QLineEdit(person.second_name or "")
        self._second_name.setPlaceholderText(t("example_second_name"))
        form.addRow(t("second_name"), self._second_name)

        self._nickname = QLineEdit(person.nickname or "")
        self._nickname.setPlaceholderText(t("example_nickname"))
        form.addRow(t("nickname"), self._nickname)

        self._married_name = QLineEdit(person.married_name or "")
        self._married_name.setPlaceholderText(t("example_married_name"))
        form.addRow(t("married_name"), self._married_name)

        self._birth_place = QLineEdit(person.birth_place or "")
        self._birth_place.setPlaceholderText(t("example_birth_place"))
        form.addRow(t("birth_place"), self._birth_place)

        self._birth_date = QLineEdit(person.birth_date or "")
        self._birth_date.setPlaceholderText(t("example_birth_date"))
        form.addRow(t("birth_date"), self._birth_date)

        separator = QLabel(" ")
        separator.setFixedHeight(4)
        form.addRow(separator)

        self._death_date = QLineEdit(person.death_date or "")
        self._death_date.setPlaceholderText(t("example_death_date"))
        form.addRow(t("death_date"), self._death_date)

        self._death_place = QLineEdit(person.death_place or "")
        self._death_place.setPlaceholderText(t("example_death_place"))
        form.addRow(t("death_place"), self._death_place)

        layout.addLayout(form)

        notes_label = QLabel(t("notes"))
        notes_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(notes_label)

        self._notes = QTextEdit(person.notes or "")
        self._notes.setPlaceholderText(t("free_notes"))
        self._notes.setMaximumHeight(100)
        layout.addWidget(self._notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._setup_completers()

    def accept(self) -> None:
        try:
            with session_scope() as session:
                canonical = FamilyService(session).ensure_unique_family_code(
                    self.family_code(),
                    current_person_id=self._person_id,
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("family_code_invalid_title"), str(exc))
            return
        self._family_code.setText(canonical or "")
        super().accept()

    # ------------------------------------------------------------------
    # Autocomplete
    # ------------------------------------------------------------------

    def _fetch_person_values(self, *fields: str) -> List[str]:
        from sqlalchemy import select, distinct
        from app.db.database import session_scope
        values: set[str] = set()
        try:
            with session_scope() as session:
                for field in fields:
                    col = getattr(Person, field)
                    rows = session.execute(
                        select(distinct(col)).where(col.isnot(None)).where(col != "")
                    ).scalars().all()
                    values.update(rows)
        except Exception:
            log.exception("Failed to load autocomplete values for person fields %s", fields)
        return sorted(values)

    def _make_completer(self, values: List[str]) -> QCompleter:
        completer = QCompleter(values, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        return completer

    def _setup_completers(self) -> None:
        places = self._fetch_person_values("birth_place", "death_place")
        place_completer = self._make_completer(places)
        self._birth_place.setCompleter(place_completer)
        self._death_place.setCompleter(self._make_completer(places))

        self._last_name.setCompleter(
            self._make_completer(self._fetch_person_values("last_name"))
        )
        self._first_name.setCompleter(
            self._make_completer(self._fetch_person_values("first_name"))
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def last_name(self) -> str:
        return self._last_name.text().strip()

    def gender(self) -> Optional[str]:
        value = self._gender.currentData()
        return str(value) if value else None

    def family_code(self) -> str:
        return self._family_code.text().strip()

    def first_name(self) -> str:
        return self._first_name.text().strip()

    def second_name(self) -> str:
        return self._second_name.text().strip()

    def nickname(self) -> str:
        return self._nickname.text().strip()

    def married_name(self) -> str:
        return self._married_name.text().strip()

    def birth_place(self) -> str:
        return self._birth_place.text().strip()

    def birth_date(self) -> str:
        return self._birth_date.text().strip()

    def death_date(self) -> str:
        return self._death_date.text().strip()

    def death_place(self) -> str:
        return self._death_place.text().strip()

    def notes(self) -> str:
        return self._notes.toPlainText().strip()
