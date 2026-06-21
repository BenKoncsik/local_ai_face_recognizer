"""Family-tree relationship editor.

Lets the user manage one person's family relationships (parents, children,
spouses), toggle whether they appear in the family tree, and delete existing
links. All relationship mutations go through :class:`FamilyService` so cycle
prevention and spouse-ordering rules are enforced centrally; this dialog only
drives the UI and surfaces validation errors.

The module also exposes :class:`PersonPickerDialog`, a small reusable picker
built on :class:`PersonSearchSelect` (used here and by the family-tree panel to
choose a root).
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sqlalchemy import or_

from app.db.database import session_scope
from app.db.models import Person, Relationship
from app.services.family_service import (
    REL_PARENT_CHILD,
    REL_SPOUSE,
    FamilyService,
    validate_family_code,
)
from app.services.family_tree_service import FamilyTreeService
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect

# Internal row-role markers (kept distinct from the stored relationship type so
# parent vs. child direction is preserved in the relationship list).
REL_PARENT_CHILD_PARENT = "parent_of"
REL_PARENT_CHILD_CHILD = "child_of"
REL_SPOUSE_ROW = "spouse_of"

# Qt user-data role for stashing (role, other_person_id) on a list item.
_ROLE_DATA = int(Qt.UserRole)
_SELECTABLE = Qt.ItemIsSelectable


class PersonPickerDialog(QDialog):
    """Modal person picker. Returns the chosen person id via :meth:`selected_id`."""

    def __init__(
        self,
        persons: List[Person],
        title: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or t("ft_pick_person_title"))
        self.setMinimumWidth(360)
        self._selected_id: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if not persons:
            layout.addWidget(QLabel(t("ft_no_persons")))

        self._selector = PersonSearchSelect(self)
        self._selector.set_persons(persons)
        self._selector.person_selected.connect(self._on_selected)
        self._selector.person_double_clicked.connect(self._on_double_clicked)
        layout.addWidget(self._selector)

        self._buttons = QDialogButtonBox()
        self._ok_btn = self._buttons.addButton(
            t("ft_select_label"), QDialogButtonBox.AcceptRole
        )
        self._buttons.addButton(t("cancel"), QDialogButtonBox.RejectRole)
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._ok_btn.setEnabled(False)
        self._selector.focus_search()

    def _on_selected(self, person_id: int) -> None:
        self._selected_id = person_id
        self._ok_btn.setEnabled(True)

    def _on_double_clicked(self, person_id: int) -> None:
        self._selected_id = person_id
        self.accept()

    def selected_id(self) -> Optional[int]:
        return self._selected_id


def _role_label(kind: str) -> str:
    return {
        "parent": t("ft_rel_parent"),
        "child": t("ft_rel_child"),
        "spouse": t("ft_rel_spouse"),
    }.get(kind, kind)


def _opposite_gender(gender: Optional[str]) -> Optional[str]:
    if gender == "male":
        return "female"
    if gender == "female":
        return "male"
    return None


def _period_text(start: Optional[str], end: Optional[str]) -> str:
    """Compact marriage-period text, e.g. '1954–1990', '1954–', '–1990'."""
    if not start and not end:
        return ""
    return t("ft_period_sep", start=start or "", end=end or "")


class AddRelativeDialog(QDialog):
    """Add a parent / child / spouse, picking an existing person or creating one.

    The family code is pre-filled from the subject's code (next free child
    digit, ``H``-numbered spouse for further marriages, derived parent code) and
    stays editable. Returns the chosen existing id *or* a new name plus the
    final family code.
    """

    def __init__(
        self, subject_id: int, kind: str, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._subject_id = subject_id
        self._kind = kind
        self._existing_id: Optional[int] = None
        self._new_name: Optional[str] = None
        self._family_code: str = ""
        self._new_gender: Optional[str] = None
        self._marriage_start: str = ""
        self._marriage_end: str = ""
        self._marriage_place: str = ""

        with session_scope() as session:
            fam = FamilyService(session)
            if kind == "spouse":
                suggestion = fam.suggest_spouse_code(subject_id)
            elif kind == "child":
                suggestion = fam.suggest_child_code(subject_id)
            else:
                suggestion = fam.suggest_parent_code(subject_id)
            subject = session.get(Person, subject_id)
            subject_name = subject.name if subject is not None else ""
            subject_gender = subject.gender if subject is not None else None
            candidates = [
                p
                for p in session.query(Person).order_by(Person.name).all()
                if p.id != subject_id
            ]
            self._codes = {p.id: (p.family_code or "") for p in candidates}
            self._genders = {p.id: p.gender for p in candidates}
            session.expunge_all()

        # A spouse must be of the opposite biological sex: a man is offered women
        # only, a woman is offered men only. Persons without a known biological
        # sex are excluded from the candidates list and cannot be added as a
        # spouse. New spouses default to the opposite sex of the subject.
        self._subject_gender = subject_gender
        opposite = _opposite_gender(subject_gender)
        if kind == "spouse" and subject_gender in ("male", "female"):
            candidates = [
                p for p in candidates if self._genders.get(p.id) == opposite
            ]
            self._new_gender = opposite

        self._suggestion = suggestion or ""
        self._build_ui(subject_name, candidates)

    def _build_ui(self, subject_name: str, candidates: list) -> None:
        self.setWindowTitle(
            t("ft_add_title", role=_role_label(self._kind), name=subject_name)
        )
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QLabel(t("ft_add_existing")))
        self._selector = PersonSearchSelect(self)
        self._selector.set_persons(candidates)
        self._selector.person_selected.connect(self._on_existing_selected)
        layout.addWidget(self._selector)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        layout.addWidget(QLabel(t("ft_add_new")))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText(t("ft_add_new_placeholder"))
        self._name_edit.textChanged.connect(self._on_name_changed)
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel(t("ft_add_code_label")))
        self._code_edit = QLineEdit(self._suggestion)
        layout.addWidget(self._code_edit)
        hint = QLabel(t("ft_add_code_hint"))
        hint.setStyleSheet("color:#888;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Marriage period + place — only relevant when adding a spouse.
        self._start_edit: Optional[QLineEdit] = None
        self._end_edit: Optional[QLineEdit] = None
        self._place_edit: Optional[QLineEdit] = None
        if self._kind == "spouse":
            period_row = QHBoxLayout()
            period_row.addWidget(QLabel(t("ft_married_from")))
            self._start_edit = QLineEdit()
            self._start_edit.setPlaceholderText(t("ft_date_placeholder"))
            period_row.addWidget(self._start_edit)
            period_row.addWidget(QLabel(t("ft_married_until")))
            self._end_edit = QLineEdit()
            self._end_edit.setPlaceholderText(t("ft_date_placeholder"))
            period_row.addWidget(self._end_edit)
            layout.addLayout(period_row)

            place_row = QHBoxLayout()
            place_row.addWidget(QLabel(t("ft_marriage_place")))
            self._place_edit = QLineEdit()
            self._place_edit.setPlaceholderText(t("ft_marriage_place_placeholder"))
            place_row.addWidget(self._place_edit)
            layout.addLayout(place_row)

        buttons = QDialogButtonBox()
        self._ok_btn = buttons.addButton(
            t("ft_add_confirm"), QDialogButtonBox.AcceptRole
        )
        buttons.addButton(t("cancel"), QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._ok_btn.setEnabled(False)
        self._selector.focus_search()

    def _on_existing_selected(self, person_id: int) -> None:
        self._existing_id = person_id
        self._name_edit.blockSignals(True)
        self._name_edit.clear()
        self._name_edit.blockSignals(False)
        # Show the existing person's own code if they have one, else the
        # suggestion (so an uncoded person can adopt the derived code).
        existing_code = self._codes.get(person_id, "")
        self._code_edit.setText(existing_code or self._suggestion)
        self._ok_btn.setEnabled(True)

    def _on_name_changed(self, text: str) -> None:
        if text.strip():
            self._existing_id = None
            self._selector.clear_selection()
            if not self._code_edit.text().strip():
                self._code_edit.setText(self._suggestion)
        self._ok_btn.setEnabled(
            bool(text.strip()) or self._existing_id is not None
        )

    def _on_accept(self) -> None:
        code_raw = self._code_edit.text().strip()
        if code_raw:
            try:
                validate_family_code(code_raw)
            except ValueError as exc:
                QMessageBox.warning(self, t("ft_invalid_code_title"), str(exc))
                return
        new_name = self._name_edit.text().strip()
        if new_name:
            self._new_name = new_name
            self._existing_id = None
        elif self._existing_id is None:
            return

        # Guard: when adding a spouse, the selected existing person must have
        # the opposite biological sex. (New persons inherit it automatically.)
        if self._kind == "spouse" and self._existing_id is not None:
            opposite = _opposite_gender(self._subject_gender)
            selected_gender = self._genders.get(self._existing_id)
            if selected_gender != opposite:
                QMessageBox.warning(
                    self,
                    t("ft_invalid_title"),
                    t("ft_spouse_opposite_sex_required", name=""),
                )
                return

        self._family_code = code_raw
        if self._start_edit is not None:
            self._marriage_start = self._start_edit.text().strip()
        if self._end_edit is not None:
            self._marriage_end = self._end_edit.text().strip()
        if self._place_edit is not None:
            self._marriage_place = self._place_edit.text().strip()
        self.accept()

    def existing_id(self) -> Optional[int]:
        return self._existing_id

    def new_name(self) -> Optional[str]:
        return self._new_name

    def family_code(self) -> str:
        return self._family_code

    def new_person_gender(self) -> Optional[str]:
        return self._new_gender

    def marriage_start(self) -> str:
        return self._marriage_start

    def marriage_end(self) -> str:
        return self._marriage_end

    def marriage_place_text(self) -> str:
        return self._marriage_place


class MarriagePeriodDialog(QDialog):
    """Edit the marriage start/end dates and place of an existing spouse relationship."""

    def __init__(
        self,
        spouse_name: str,
        start: Optional[str],
        end: Optional[str],
        place: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("ft_marriage_edit_title", name=spouse_name))
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        date_row = QHBoxLayout()
        date_row.addWidget(QLabel(t("ft_married_from")))
        self._start_edit = QLineEdit(start or "")
        self._start_edit.setPlaceholderText(t("ft_date_placeholder"))
        date_row.addWidget(self._start_edit)
        date_row.addWidget(QLabel(t("ft_married_until")))
        self._end_edit = QLineEdit(end or "")
        self._end_edit.setPlaceholderText(t("ft_date_placeholder"))
        date_row.addWidget(self._end_edit)
        layout.addLayout(date_row)

        place_row = QHBoxLayout()
        place_row.addWidget(QLabel(t("ft_marriage_place")))
        self._place_edit = QLineEdit(place or "")
        self._place_edit.setPlaceholderText(t("ft_marriage_place_placeholder"))
        place_row.addWidget(self._place_edit)
        layout.addLayout(place_row)

        buttons = QDialogButtonBox()
        buttons.addButton(t("ft_save_code"), QDialogButtonBox.AcceptRole)
        buttons.addButton(t("cancel"), QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._start_edit.setFocus()

    def start_date(self) -> str:
        return self._start_edit.text().strip()

    def end_date(self) -> str:
        return self._end_edit.text().strip()

    def marriage_place(self) -> str:
        return self._place_edit.text().strip()


class FamilyTreeEditorDialog(QDialog):
    """Edit one person's relationships and family-tree membership."""

    relationships_changed = Signal()

    def __init__(self, person_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person_id
        self._person_name = ""
        self._build_ui()
        self._reload()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._title = QLabel()
        self._title.setStyleSheet("font-weight: bold;")
        self._title.setWordWrap(True)
        layout.addWidget(self._title)

        self._member_check = QCheckBox(t("ft_is_member"))
        self._member_check.toggled.connect(self._on_member_toggled)
        layout.addWidget(self._member_check)

        # The subject's own family code is editable here too — editing or
        # extending it re-derives links on the next "Regenerate".
        code_row = QHBoxLayout()
        code_row.addWidget(QLabel(t("ft_own_code")))
        self._own_code_edit = QLineEdit()
        self._own_code_edit.returnPressed.connect(self._on_save_own_code)
        code_row.addWidget(self._own_code_edit, 1)
        self._save_code_btn = QPushButton(t("ft_save_code"))
        self._save_code_btn.clicked.connect(self._on_save_own_code)
        code_row.addWidget(self._save_code_btn)
        layout.addLayout(code_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        layout.addWidget(QLabel(t("ft_current_relationships")))
        self._rel_list = QListWidget()
        self._rel_list.setMaximumHeight(180)
        self._rel_list.itemDoubleClicked.connect(self._on_edit_marriage)
        layout.addWidget(self._rel_list)

        self._marriage_hint = QLabel(t("ft_marriage_edit_hint"))
        self._marriage_hint.setStyleSheet("color:#888; font-size: 11px;")
        self._marriage_hint.setWordWrap(True)
        layout.addWidget(self._marriage_hint)

        self._remove_btn = QPushButton(t("ft_remove_selected"))
        self._remove_btn.clicked.connect(self._on_remove)
        layout.addWidget(self._remove_btn)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line2)

        layout.addWidget(QLabel(t("ft_add_relationship")))
        add_row = QHBoxLayout()
        self._kind_combo = QComboBox()
        self._kind_combo.addItem(t("ft_rel_parent"), "parent")
        self._kind_combo.addItem(t("ft_rel_child"), "child")
        self._kind_combo.addItem(t("ft_rel_spouse"), "spouse")
        add_row.addWidget(self._kind_combo, 1)
        self._add_btn = QPushButton(t("ft_add_btn"))
        self._add_btn.clicked.connect(self._on_add)
        add_row.addWidget(self._add_btn)
        layout.addLayout(add_row)

        buttons = QDialogButtonBox()
        buttons.addButton(t("close"), QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        """Re-read the person's relationships and repopulate the list."""
        self._rel_list.clear()
        with session_scope() as session:
            person = session.get(Person, self._person_id)
            if person is None:
                self.reject()
                return
            self._person_name = person.name
            self._member_check.blockSignals(True)
            self._member_check.setChecked(bool(person.is_family_tree_member))
            self._member_check.blockSignals(False)
            self._own_code_edit.setText(person.family_code or "")

            fam = FamilyService(session)
            tree = FamilyTreeService(session)
            graph = tree.build_tree(self._person_id)
            node = graph.node(self._person_id)
            names = {pid: n.name for pid, n in graph.nodes.items()}

            # (role, name, pid, period_text) — period only for spouse rows.
            rows: list[tuple[str, str, int, str]] = []
            if node is not None:
                for pid in node.parent_ids:
                    rows.append((REL_PARENT_CHILD_PARENT, names.get(pid, str(pid)), pid, ""))
                for pid in node.child_ids:
                    rows.append((REL_PARENT_CHILD_CHILD, names.get(pid, str(pid)), pid, ""))
                for pid in node.spouse_ids:
                    start, end, place = fam.marriage_period(self._person_id, pid)
                    rows.append(
                        (REL_SPOUSE_ROW, names.get(pid, str(pid)), pid, _period_text(start, end))
                    )

        self._title.setText(t("ft_editor_title", name=self._person_name))

        if not rows:
            placeholder = QListWidgetItem(t("ft_no_relationships"))
            placeholder.setFlags(placeholder.flags() & ~ _SELECTABLE)
            self._rel_list.addItem(placeholder)
            self._remove_btn.setEnabled(False)
            return

        self._remove_btn.setEnabled(True)
        role_labels = {
            REL_PARENT_CHILD_PARENT: t("ft_rel_parent"),
            REL_PARENT_CHILD_CHILD: t("ft_rel_child"),
            REL_SPOUSE_ROW: t("ft_rel_spouse"),
        }
        for role, name, pid, period in rows:
            if period:
                text = t("ft_rel_row_period", role=role_labels[role], name=name, period=period)
            else:
                text = t("ft_rel_row", role=role_labels[role], name=name)
            item = QListWidgetItem(text)
            item.setData(_ROLE_DATA, (role, pid))
            self._rel_list.addItem(item)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_member_toggled(self, checked: bool) -> None:
        with session_scope() as session:
            FamilyTreeService(session).set_family_tree_member(
                self._person_id, checked
            )
        self.relationships_changed.emit()

    def _on_save_own_code(self) -> None:
        code = self._own_code_edit.text().strip()
        if code:
            try:
                validate_family_code(code)
            except ValueError as exc:
                QMessageBox.warning(self, t("ft_invalid_code_title"), str(exc))
                return
        with session_scope() as session:
            person = session.get(Person, self._person_id)
            if person is not None:
                person.family_code = code or None
        self._reload()
        self.relationships_changed.emit()

    def _on_remove(self) -> None:
        item = self._rel_list.currentItem()
        if item is None:
            return
        data = item.data(_ROLE_DATA)
        if not data:
            return
        role, other_id = data
        with session_scope() as session:
            fam = FamilyService(session)
            if role == REL_PARENT_CHILD_PARENT:
                fam.remove_relationship(other_id, self._person_id, REL_PARENT_CHILD)
            elif role == REL_PARENT_CHILD_CHILD:
                fam.remove_relationship(self._person_id, other_id, REL_PARENT_CHILD)
            else:
                fam.remove_relationship(self._person_id, other_id, REL_SPOUSE)
        self._reload()
        self.relationships_changed.emit()

    def _on_edit_marriage(self, item: QListWidgetItem) -> None:
        """Double-clicking a spouse row edits that marriage's start/end/place."""
        data = item.data(_ROLE_DATA)
        if not data:
            return
        role, other_id = data
        if role != REL_SPOUSE_ROW:
            return
        with session_scope() as session:
            fam = FamilyService(session)
            spouse = session.get(Person, other_id)
            spouse_name = spouse.name if spouse is not None else str(other_id)
            start, end, place = fam.marriage_period(self._person_id, other_id)

        dialog = MarriagePeriodDialog(spouse_name, start, end, place, parent=self)
        if dialog.exec() != QDialog.Accepted:
            return
        with session_scope() as session:
            FamilyService(session).set_marriage_period(
                self._person_id, other_id,
                dialog.start_date(), dialog.end_date(), dialog.marriage_place(),
            )
        self._reload()
        self.relationships_changed.emit()

    def _on_add(self) -> None:
        if add_relative(self._person_id, self._kind_combo.currentData(), self):
            self._reload()
            self.relationships_changed.emit()


# ----------------------------------------------------------------------------
# Shared add-relative flow (used by the editor dialog and the panel quick-add
# buttons). Opens AddRelativeDialog, creates/links the relative, returns True
# when something was added so callers can refresh.
# ----------------------------------------------------------------------------

def add_relative(subject_id: int, kind: str, parent: QWidget) -> bool:
    if kind == "spouse":
        with session_scope() as session:
            subject = session.get(Person, subject_id)
            subject_name = subject.name if subject else ""
            subject_gender = subject.gender if subject else None

            if subject_gender not in ("male", "female"):
                QMessageBox.warning(
                    parent,
                    t("ft_invalid_title"),
                    t("ft_spouse_gender_unknown", name=subject_name),
                )
                return False

            living_spouse_name = _find_living_spouse_name(session, subject_id)
            if living_spouse_name is not None:
                QMessageBox.warning(
                    parent,
                    t("ft_invalid_title"),
                    t("ft_spouse_living", name=subject_name, spouse=living_spouse_name),
                )
                return False

    dialog = AddRelativeDialog(subject_id, kind, parent=parent)
    if dialog.exec() != QDialog.Accepted:
        return False

    new_name = dialog.new_name()
    existing_id = dialog.existing_id()
    code = dialog.family_code()  # already validated (or empty)
    gender = dialog.new_person_gender()

    try:
        with session_scope() as session:
            fam = FamilyService(session)
            target_id = _resolve_target(
                session, new_name, existing_id, code, gender
            )
            if target_id is None:
                return False

            if kind == "parent":
                fam.add_parent_child(target_id, subject_id)
            elif kind == "spouse":
                fam.add_spouse(
                    subject_id,
                    target_id,
                    start_date=dialog.marriage_start(),
                    end_date=dialog.marriage_end(),
                    marriage_place=dialog.marriage_place_text(),
                )
            else:  # child
                fam.add_parent_child(subject_id, target_id)
                # When the subject has exactly one spouse, the child belongs to
                # both — link the spouse as the second parent too.
                spouse_id = _single_spouse_id(session, subject_id)
                if spouse_id is not None and spouse_id != target_id:
                    try:
                        fam.add_parent_child(spouse_id, target_id)
                    except ValueError:
                        pass  # cycle / already linked — keep the primary link
    except ValueError as exc:
        QMessageBox.warning(parent, t("ft_invalid_title"), str(exc))
        return False
    return True


def _resolve_target(
    session,
    new_name: Optional[str],
    existing_id: Optional[int],
    code: str,
    gender: Optional[str] = None,
) -> Optional[int]:
    """Create the new person (if any) or adopt the code for an existing one."""
    if new_name:
        person = Person(
            name=new_name,
            is_auto_named=False,
            family_code=code or None,
            gender=gender,
        )
        session.add(person)
        session.flush()
        return person.id
    if existing_id is None:
        return None
    if code:
        person = session.get(Person, existing_id)
        # Only fill an empty code — never overwrite an existing identity.
        if person is not None and not (person.family_code or "").strip():
            person.family_code = code
    return existing_id


def _find_living_spouse_name(session, person_id: int) -> Optional[str]:
    """Return the name of a living current spouse, or None if there is none."""
    rows = (
        session.query(Relationship)
        .filter(
            Relationship.relationship_type == REL_SPOUSE,
            or_(
                Relationship.person_a_id == person_id,
                Relationship.person_b_id == person_id,
            ),
        )
        .all()
    )
    for rel in rows:
        spouse_id = rel.person_a_id if rel.person_b_id == person_id else rel.person_b_id
        spouse = session.get(Person, spouse_id)
        if spouse is not None and not spouse.death_date:
            return spouse.name
    return None


def _single_spouse_id(session, person_id: int) -> Optional[int]:
    rows = (
        session.query(Relationship)
        .filter(
            Relationship.relationship_type == REL_SPOUSE,
            or_(
                Relationship.person_a_id == person_id,
                Relationship.person_b_id == person_id,
            ),
        )
        .all()
    )
    spouse_ids = {
        r.person_a_id if r.person_b_id == person_id else r.person_b_id
        for r in rows
    }
    return next(iter(spouse_ids)) if len(spouse_ids) == 1 else None
