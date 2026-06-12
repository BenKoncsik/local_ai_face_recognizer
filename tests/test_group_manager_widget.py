"""UI tests for GroupManagerWidget: member editing + auto-save.

Covers the Társaságok tab editor: selecting a group lists its members, members
can be added/removed inline, and name/note edits persist both via the Save
button and automatically (when focus leaves a field, before member ops, etc.).
"""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person
from app.services.person_group_service import PersonGroupService
from app.ui.dialogs import group_manager_dialog
from app.ui.dialogs import person_info_dialog
from app.ui.dialogs.group_manager_dialog import _ROLE_ID, GroupManagerWidget


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "groups_ui.db")


def _add_person(name: str) -> int:
    with session_scope() as s:
        p = Person(name=name, is_auto_named=False)
        s.add(p)
        s.flush()
        return p.id


def _make_group(name: str) -> int:
    with session_scope() as s:
        return PersonGroupService(s).create_group(name).id


def _add_image_with_face(person_id: int, path: str) -> None:
    with session_scope() as s:
        img = Image(file_path=path, file_hash="h" + path, file_mtime=0.0)
        s.add(img)
        s.flush()
        s.add(
            Face(
                image_id=img.id,
                person_id=person_id,
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                confidence=0.9,
                crop_path=path,  # arbitrary; not loaded from disk in these tests
            )
        )


def _members(gid: int) -> set[str]:
    with session_scope() as s:
        return {p.name for p in PersonGroupService(s).get_persons_in_group(gid)}


def _select_group(widget: GroupManagerWidget, gid: int) -> None:
    for row in range(widget._list.count()):
        if widget._list.item(row).data(_ROLE_ID) == gid:
            widget._list.setCurrentRow(row)
            return
    raise AssertionError(f"group {gid} not in list")


def test_selecting_group_lists_members(db, qtbot):
    alice = _add_person("Alice")
    _add_person("Bob")
    gid = _make_group("Kórus")
    with session_scope() as s:
        PersonGroupService(s).add_person_to_group(alice, gid)

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)

    names = {
        widget._members_list.item(r).text()
        for r in range(widget._members_list.count())
    }
    assert names == {"Alice"}


def test_remove_member_updates_db(db, qtbot):
    alice = _add_person("Alice")
    bob = _add_person("Bob")
    gid = _make_group("Kórus")
    with session_scope() as s:
        svc = PersonGroupService(s)
        svc.add_person_to_group(alice, gid)
        svc.add_person_to_group(bob, gid)

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)

    # Select Alice in the member list and remove her.
    for r in range(widget._members_list.count()):
        if widget._members_list.item(r).data(_ROLE_ID) == alice:
            widget._members_list.setCurrentRow(r)
    widget._on_remove_member()

    assert _members(gid) == {"Bob"}


def test_name_edit_auto_saves_on_focus_out(db, qtbot):
    gid = _make_group("Kórus")

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)

    widget._name_edit.setText("Kamarakórus")
    # editingFinished is what fires on real focus loss / Enter.
    widget._name_edit.editingFinished.emit()

    with session_scope() as s:
        info = next(g for g in PersonGroupService(s).list_groups() if g.id == gid)
    assert info.name == "Kamarakórus"
    # Left-list row label reflects the new name.
    assert "Kamarakórus" in widget._list.currentItem().text()


def test_switching_group_persists_pending_edit(db, qtbot):
    g1 = _make_group("Alpha")
    g2 = _make_group("Beta")

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, g1)

    # Type a new note but do NOT press Save; switch to the other group.
    widget._note_edit.setPlainText("rehearsal Tuesdays")
    _select_group(widget, g2)

    with session_scope() as s:
        info = next(g for g in PersonGroupService(s).list_groups() if g.id == g1)
    assert info.description == "rehearsal Tuesdays"


def test_save_button_persists(db, qtbot):
    gid = _make_group("Kórus")

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)

    widget._name_edit.setText("Vegyeskar")
    widget._note_edit.setPlainText("note")
    widget._on_save()

    with session_scope() as s:
        info = next(g for g in PersonGroupService(s).list_groups() if g.id == gid)
    assert info.name == "Vegyeskar"
    assert info.description == "note"


def test_invalid_name_not_persisted_silently_on_switch(db, qtbot):
    g1 = _make_group("Alpha")
    g2 = _make_group("Beta")  # name collision target

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, g1)

    # Rename Alpha → "Beta" (duplicate). Switching groups triggers the silent
    # leave-save, which must reject the duplicate without corrupting Alpha.
    widget._name_edit.setText("Beta")
    _select_group(widget, g2)

    with session_scope() as s:
        info = next(g for g in PersonGroupService(s).list_groups() if g.id == g1)
    assert info.name == "Alpha"  # unchanged — duplicate rejected


def test_member_images_passes_person_image_paths(db, qtbot, monkeypatch):
    alice = _add_person("Alice")
    gid = _make_group("Kórus")
    with session_scope() as s:
        PersonGroupService(s).add_person_to_group(alice, gid)
    _add_image_with_face(alice, "/tmp/a1.jpg")
    _add_image_with_face(alice, "/tmp/a2.jpg")

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)
    widget._members_list.setCurrentRow(0)

    captured = {}

    class _FakeDialog:
        def __init__(self, name, paths, parent=None):
            captured["name"] = name
            captured["paths"] = list(paths)

        def exec(self):
            return 0

    monkeypatch.setattr(group_manager_dialog, "MemberImagesDialog", _FakeDialog)
    widget._on_member_images()

    assert captured["name"] == "Alice"
    assert set(captured["paths"]) == {"/tmp/a1.jpg", "/tmp/a2.jpg"}


def test_member_detail_opens_editor_and_refreshes(db, qtbot, monkeypatch):
    alice = _add_person("Alice")
    gid = _make_group("Kórus")
    with session_scope() as s:
        PersonGroupService(s).add_person_to_group(alice, gid)

    widget = GroupManagerWidget()
    qtbot.addWidget(widget)
    _select_group(widget, gid)
    widget._members_list.setCurrentRow(0)

    calls = {}

    def _fake_edit(person_id, parent=None):
        calls["person_id"] = person_id
        # Simulate the user removing this person from the group via the profile.
        with session_scope() as s:
            PersonGroupService(s).remove_person_from_group(person_id, gid)
        return True

    monkeypatch.setattr(person_info_dialog, "edit_person_dialog", _fake_edit)
    widget._on_member_detail()

    assert calls["person_id"] == alice
    # The members list refreshed and now reflects the removal.
    assert widget._members_list.count() == 0
    assert _members(gid) == set()
