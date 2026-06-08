"""Person info dialog — edit structured personal data for a recognised person."""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QCompleter,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QStyle,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.person_group_service import PersonGroupService
from app.ui.i18n import t
from app.ui.widgets.group_chip_select import GroupChipSelect

log = logging.getLogger(__name__)


class PersonInfoDialog(QDialog):
    """Edit structured personal data for a person."""

    def __init__(self, person: Person, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person.id
        self._is_protected = person.is_protected
        self.setWindowTitle(t("person_info_title", name=person.name))
        self.setMinimumWidth(460)
        self.setMinimumHeight(380)
        self.resize(460, 600)

        # Outer layout: scroll area fills space, buttons pinned at bottom
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 8)
        outer.setSpacing(0)

        # --- Scroll area wraps all editable content ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(4)
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

        # --- Title ---
        title = QLabel(f"<b>{person.name}</b>")
        title.setStyleSheet("font-size: 14px; margin-bottom: 6px;")
        layout.addWidget(title)

        # --- Form fields ---
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
        self._add_help_action(self._family_code, t("family_code_help"))
        form.addRow(t("family_code"), self._family_code)

        self._external_family_code = QLineEdit(person.external_family_code or "")
        self._external_family_code.setPlaceholderText(t("example_external_family_code"))
        self._add_help_action(
            self._external_family_code, t("external_family_code_help")
        )
        form.addRow(t("external_family_code"), self._external_family_code)

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

        # --- Notes ---
        notes_label = QLabel(t("notes"))
        notes_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(notes_label)

        self._notes = QTextEdit(person.notes or "")
        self._notes.setPlaceholderText(t("free_notes"))
        self._notes.setFixedHeight(80)
        layout.addWidget(self._notes)

        # --- Groups ---
        groups_label = QLabel(t("person_groups"))
        groups_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(groups_label)

        self._groups = GroupChipSelect()
        if self._is_protected:
            self._groups.setEnabled(False)
            self._groups.setToolTip(t("person_groups_protected_tip"))
        layout.addWidget(self._groups)

        # --- Related objects (read-only; managed from the Objects tab) ---
        objects_label = QLabel(t("person_related_objects"))
        objects_label.setStyleSheet("margin-top: 8px;")
        layout.addWidget(objects_label)

        self._objects_view = QLabel()
        self._objects_view.setWordWrap(True)
        self._objects_view.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(self._objects_view)
        layout.addStretch()

        # --- Buttons pinned outside the scroll area ---
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line)

        btn_row = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btn_row.accepted.connect(self.accept)
        btn_row.rejected.connect(self.reject)
        btn_row.setContentsMargins(12, 4, 12, 4)
        outer.addWidget(btn_row)

        self._setup_completers()
        self._load_groups()
        self._load_objects()

    def _add_help_action(self, field: QLineEdit, help_text: str) -> None:
        """Attach a trailing ``?`` help icon inside *field*.

        The icon lives inside the line edit (so editing/clicking the field is
        never blocked by a wrapping container) and reveals the full *help_text*
        as a multi-line tooltip on hover. The detail is kept out of the layout,
        so the dialog stays compact.
        """
        # The <html> wrapper guarantees Qt treats the tooltip as rich text on
        # every platform; each example then sits on its own wrapped line.
        tooltip_html = "<html>" + help_text.replace(", ", ",<br>") + "</html>"
        field.setToolTip(tooltip_html)

        icon = self.style().standardIcon(QStyle.SP_TitleBarContextHelpButton)
        action = field.addAction(icon, QLineEdit.TrailingPosition)
        action.setToolTip(tooltip_html)
        # Clicking the icon should reveal the same help, not just hovering it.
        action.triggered.connect(
            lambda _=False, w=field, html=tooltip_html: QToolTip.showText(
                QCursor.pos(), html, w
            )
        )

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

        try:
            with session_scope() as session:
                external_canonical = FamilyService(session).ensure_unique_external_family_code(
                    self.external_family_code(),
                    current_person_id=self._person_id,
                )
        except ValueError as exc:
            QMessageBox.warning(self, t("external_family_code_invalid_title"), str(exc))
            return
        self._external_family_code.setText(external_canonical or "")

        # Save group memberships (skipped for protected persons)
        if not self._is_protected:
            try:
                with session_scope() as session:
                    PersonGroupService(session).set_person_groups(
                        self._person_id, self._groups.selected_group_ids()
                    )
            except Exception:
                log.exception("Failed to save groups for person id=%d", self._person_id)

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

    def _load_groups(self) -> None:
        """Populate the GroupChipSelect with all existing groups and this person's groups."""
        from app.db.models import PersonGroup as _PG
        try:
            with session_scope() as session:
                svc = PersonGroupService(session)
                all_groups = [
                    (g.id, g.name)
                    for g in session.query(_PG).order_by(_PG.name).all()
                ]
                person_groups = [
                    (g.id, g.name) for g in svc.get_person_groups(self._person_id)
                ]
        except Exception:
            log.exception("Failed to load groups for person id=%d", self._person_id)
            return
        self._groups.set_available_groups(all_groups)
        self._groups.set_selected_groups(person_groups)

    def _load_objects(self) -> None:
        """Show the objects this person is linked to (read-only)."""
        from app.services.object_service import ObjectService

        try:
            with session_scope() as session:
                links = ObjectService(session).get_objects_for_person(self._person_id)
        except Exception:
            log.exception("Failed to load objects for person id=%d", self._person_id)
            return
        if not links:
            self._objects_view.setText(t("person_no_objects"))
            return
        lines = []
        for link in links:
            role = t(f"object_role_{link.role}")
            lines.append(f"• {link.name} — {role}")
        self._objects_view.setText("\n".join(lines))

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

    def external_family_code(self) -> str:
        return self._external_family_code.text().strip()

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
