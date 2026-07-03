"""Widget tests for the family-tree panel and relationship editor dialog."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from app.db.database import init_db, session_scope
from app.db.models import Person, Relationship
from app.services.family_service import (
    REL_PARENT_CHILD,
    REL_SPOUSE,
    FamilyService,
)
from app.ui.dialogs.family_tree_editor_dialog import (
    AddRelativeDialog,
    FamilyTreeEditorDialog,
)
from app.ui.panels.family_tree_panel import FamilyTreePanel


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "tree_ui.db")


def _person(session, name: str) -> int:
    p = Person(name=name, is_auto_named=False)
    session.add(p)
    session.flush()
    return p.id


def _seed(session) -> dict[str, int]:
    ids = {n: _person(session, n) for n in ("GpA", "P1", "C1", "C2")}
    fam = FamilyService(session)
    fam.add_parent_child(ids["GpA"], ids["P1"])
    fam.add_parent_child(ids["P1"], ids["C1"])
    fam.add_parent_child(ids["P1"], ids["C2"])
    return ids


def _rendered_ids(panel) -> set[int]:
    """Person ids currently drawn as boxes in the graphical view."""
    return set(panel._tree._node_items)


# ── panel ───────────────────────────────────────────────────────────────────────

def test_panel_renders_hierarchy(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._root_id = ids["GpA"]
    panel.refresh()

    # The whole family is drawn as boxes.
    rendered = _rendered_ids(panel)
    assert ids["GpA"] in rendered
    assert ids["C1"] in rendered
    assert ids["C2"] in rendered


def test_panel_non_member_filter(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)
        # Drop C2 out of the tree.
        session.get(Person, ids["C2"]).is_family_tree_member = False

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._root_id = ids["GpA"]

    panel._non_members_check.setChecked(False)
    panel.refresh()
    assert ids["C2"] not in _rendered_ids(panel)

    panel._non_members_check.setChecked(True)
    panel.refresh()
    assert ids["C2"] in _rendered_ids(panel)


# ── editor dialog ─────────────────────────────────────────────────────────────────

def test_editor_lists_relationships(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)

    dlg = FamilyTreeEditorDialog(ids["P1"])
    qtbot.addWidget(dlg)
    rows = [dlg._rel_list.item(i).text() for i in range(dlg._rel_list.count())]
    assert len(rows) == 3
    assert any("GpA" in r for r in rows)


def test_editor_remove_relationship(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)

    dlg = FamilyTreeEditorDialog(ids["P1"])
    qtbot.addWidget(dlg)

    # Select the C1 child row and remove it.
    target_row = next(
        i
        for i in range(dlg._rel_list.count())
        if dlg._rel_list.item(i).data(int(Qt.UserRole))[1] == ids["C1"]
    )
    dlg._rel_list.setCurrentRow(target_row)
    dlg._on_remove()

    with session_scope() as session:
        remaining = (
            session.query(Relationship)
            .filter(
                Relationship.relationship_type == REL_PARENT_CHILD,
                Relationship.person_a_id == ids["P1"],
                Relationship.person_b_id == ids["C1"],
            )
            .count()
        )
        assert remaining == 0


def test_editor_member_toggle(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)

    dlg = FamilyTreeEditorDialog(ids["C1"])
    qtbot.addWidget(dlg)
    assert dlg._member_check.isChecked()
    dlg._member_check.setChecked(False)

    with session_scope() as session:
        assert session.get(Person, ids["C1"]).is_family_tree_member is False


def test_editor_add_spouse_via_service(db, qtbot):
    """The add flow delegates to FamilyService; verify the wiring + refresh."""
    with session_scope() as session:
        ids = _seed(session)
        extra = _person(session, "Spouse")

    dlg = FamilyTreeEditorDialog(ids["P1"])
    qtbot.addWidget(dlg)

    # Simulate the picker returning a person and the spouse add path.
    with session_scope() as session:
        FamilyService(session).add_spouse(ids["P1"], extra)
    dlg._reload()

    rows = [dlg._rel_list.item(i).text() for i in range(dlg._rel_list.count())]
    assert any("Spouse" in r for r in rows)

    with session_scope() as session:
        assert (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_SPOUSE)
            .count()
            == 1
        )


# ── add-relative dialog (auto-filled family code) ─────────────────────────────────

def _coded_person(session, name: str, code: str, gender: str | None = None) -> int:
    p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
    session.add(p)
    session.flush()
    return p.id


def test_add_dialog_prefills_spouse_code(db, qtbot):
    with session_scope() as session:
        base = _coded_person(session, "Base", "C8")

    dlg = AddRelativeDialog(base, "spouse")
    qtbot.addWidget(dlg)
    assert dlg._code_edit.text() == "C80"


def _autofill_exec(name: str):
    """Build a fake exec() that types a new name and accepts the dialog."""
    from PySide6.QtWidgets import QDialog

    def fake_exec(self):
        self._name_edit.setText(name)
        self._on_accept()
        return QDialog.Accepted

    return fake_exec


def test_editor_add_new_spouse_creates_coded_person(db, qtbot, monkeypatch):
    with session_scope() as session:
        # A spouse can only be added when the subject's gender is known;
        # otherwise add_relative() shows a modal warning (see add_relative()).
        base = _coded_person(session, "Base", "C8", gender="male")

    dlg = FamilyTreeEditorDialog(base)
    qtbot.addWidget(dlg)
    dlg._kind_combo.setCurrentIndex(2)  # spouse
    monkeypatch.setattr(AddRelativeDialog, "exec", _autofill_exec("Wife"))
    dlg._on_add()

    with session_scope() as session:
        wife = (
            session.query(Person).filter(Person.name == "Wife").one()
        )
        assert wife.family_code == "C80"
        assert (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_SPOUSE)
            .count()
            == 1
        )


def test_editor_add_child_links_both_parents(db, qtbot, monkeypatch):
    with session_scope() as session:
        base = _coded_person(session, "Base", "C8")
        spouse = _coded_person(session, "Wife", "C80")
        FamilyService(session).add_spouse(base, spouse)

    dlg = FamilyTreeEditorDialog(base)
    qtbot.addWidget(dlg)
    dlg._kind_combo.setCurrentIndex(1)  # child
    monkeypatch.setattr(AddRelativeDialog, "exec", _autofill_exec("Kid"))
    dlg._on_add()

    with session_scope() as session:
        kid = session.query(Person).filter(Person.name == "Kid").one()
        assert kid.family_code == "C81"
        # Linked to both base and spouse as parents.
        parent_links = (
            session.query(Relationship)
            .filter(
                Relationship.relationship_type == REL_PARENT_CHILD,
                Relationship.person_b_id == kid.id,
            )
            .count()
        )
        assert parent_links == 2


def test_editor_save_own_code(db, qtbot):
    with session_scope() as session:
        base = _coded_person(session, "Base", "C8")

    dlg = FamilyTreeEditorDialog(base)
    qtbot.addWidget(dlg)
    assert dlg._own_code_edit.text() == "C8"
    dlg._own_code_edit.setText("C9")
    dlg._on_save_own_code()

    with session_scope() as session:
        assert session.get(Person, base).family_code == "C9"


# ── panel: whole-tree default, regenerate, quick-add ──────────────────────────────

def test_panel_shows_whole_forest_by_default(db, qtbot):
    with session_scope() as session:
        a = _coded_person(session, "A base", "A")
        _coded_person(session, "A child", "A1")
        b = _coded_person(session, "B base", "B")
        _coded_person(session, "B child", "B1")
        fam = FamilyService(session)
        fam.add_parent_child(a, session.query(Person).filter(Person.name == "A child").one().id)
        fam.add_parent_child(b, session.query(Person).filter(Person.name == "B child").one().id)

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    # No root chosen — every person in both families is drawn.
    assert panel._root_id is None
    assert len(_rendered_ids(panel)) == 4


def test_panel_regenerate_button(db, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    with session_scope() as session:
        _coded_person(session, "Base", "C8")
        _coded_person(session, "Child", "C81")

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    panel._on_regenerate()

    with session_scope() as session:
        assert (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_PARENT_CHILD)
            .count()
            == 1
        )


def _coded_gendered(session, name, code, gender):
    p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
    session.add(p)
    session.flush()
    return p.id


def test_add_dialog_offers_only_opposite_gender(db, qtbot):
    with session_scope() as session:
        man = _coded_gendered(session, "Man", "C8", "male")
        _coded_gendered(session, "Woman", None, "female")
        _coded_gendered(session, "Other man", None, "male")

    dlg = AddRelativeDialog(man, "spouse")
    qtbot.addWidget(dlg)
    offered = {e.person_id for e in dlg._selector._entries}
    with session_scope() as session:
        genders = {p.id: p.gender for p in session.query(Person).all()}
    assert all(genders[pid] != "male" for pid in offered)
    assert dlg.new_person_gender() == "female"


def test_quick_add_spouse_stores_dates_and_gender(db, qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    with session_scope() as session:
        man = _coded_gendered(session, "Man", "C8", "male")

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._selected_id = man

    def fake_exec(self):
        self._name_edit.setText("New Wife")
        self._start_edit.setText("1955")
        self._end_edit.setText("2000")
        self._on_accept()
        return QDialog.Accepted

    monkeypatch.setattr(AddRelativeDialog, "exec", fake_exec)
    panel._on_quick_add("spouse")

    with session_scope() as session:
        wife = session.query(Person).filter(Person.name == "New Wife").one()
        assert wife.gender == "female"
        assert wife.family_code == "C80"
        rel = (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_SPOUSE)
            .one()
        )
        assert (rel.start_date, rel.end_date) == ("1955", "2000")


def test_panel_focus_shows_lineage_only(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)  # GpA - P1 - C1, C2

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._root_id = ids["P1"]
    panel._selected_id = ids["P1"]
    panel.refresh()

    rendered = _rendered_ids(panel)
    assert ids["GpA"] in rendered   # ancestor
    assert ids["C1"] in rendered    # descendant
    # The detail relation row is shown in focus mode.
    assert panel._relation_row.isVisible() or panel._relation_lbl is not None


def test_panel_selection_populates_detail(db, qtbot):
    with session_scope() as session:
        man = _coded_gendered(session, "Detail Guy", "C8", "male")

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._on_node_selected(man)
    assert panel._name.text() == "Detail Guy"
    assert panel._code_lbl.text() == "C8"
    assert panel._edit_person_btn.isEnabled()
    assert panel._images_btn.isEnabled()


def test_panel_quick_add_child(db, qtbot, monkeypatch):
    with session_scope() as session:
        base = _coded_person(session, "Base", "C8")

    panel = FamilyTreePanel()
    qtbot.addWidget(panel)
    panel._selected_id = base
    monkeypatch.setattr(AddRelativeDialog, "exec", _autofill_exec("Kiddo"))
    panel._on_quick_add("child")

    with session_scope() as session:
        kid = session.query(Person).filter(Person.name == "Kiddo").one()
        assert kid.family_code == "C81"
