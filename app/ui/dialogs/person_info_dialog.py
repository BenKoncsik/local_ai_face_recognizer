"""Person info dialog — edit structured personal data for a recognised person."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person


class PersonInfoDialog(QDialog):
    """Edit Vezetéknév / Keresztnév / Születési adatok / Megjegyzés for a person."""

    def __init__(self, person: Person, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Személyadatok — {person.name}")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        title = QLabel(f"<b>{person.name}</b>")
        title.setStyleSheet("font-size: 14px; margin-bottom: 6px;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setLabelAlignment(form.labelAlignment())
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self._last_name = QLineEdit(person.last_name or "")
        self._last_name.setPlaceholderText("pl. Kovács")
        form.addRow("Vezetéknév:", self._last_name)

        self._first_name = QLineEdit(person.first_name or "")
        self._first_name.setPlaceholderText("pl. János")
        form.addRow("Keresztnév:", self._first_name)

        self._second_name = QLineEdit(person.second_name or "")
        self._second_name.setPlaceholderText("pl. István  (opcionális)")
        form.addRow("Keresztnév 2:", self._second_name)

        self._birth_place = QLineEdit(person.birth_place or "")
        self._birth_place.setPlaceholderText("pl. Budapest")
        form.addRow("Születési hely:", self._birth_place)

        self._birth_date = QLineEdit(person.birth_date or "")
        self._birth_date.setPlaceholderText("pl. 1954  vagy  1954.03.12  vagy  1930-as évek")
        form.addRow("Születési idő:", self._birth_date)

        layout.addLayout(form)

        notes_label = QLabel("Egyéb megjegyzés:")
        notes_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(notes_label)

        self._notes = QTextEdit(person.notes or "")
        self._notes.setPlaceholderText("Szabad szöveges megjegyzések…")
        self._notes.setMaximumHeight(100)
        layout.addWidget(self._notes)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def last_name(self) -> str:
        return self._last_name.text().strip()

    def first_name(self) -> str:
        return self._first_name.text().strip()

    def second_name(self) -> str:
        return self._second_name.text().strip()

    def birth_place(self) -> str:
        return self._birth_place.text().strip()

    def birth_date(self) -> str:
        return self._birth_date.text().strip()

    def notes(self) -> str:
        return self._notes.toPlainText().strip()
