"""Persons-tab table stays in sync with edits made elsewhere.

Regression test for the bug where a family code entered via the image
browser / sidebar PersonInfoDialog committed to the DB but the standalone
Persons table did not reload, so the change only surfaced the next time a
different person was saved through the Persons tab itself.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.person_service import PersonService
from app.ui.panels.persons_panel import COL_FAMILY_CODE, COL_ID, PersonsPanel


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "persons.db")


def _add_person(session, name: str) -> int:
    person = Person(name=name, is_auto_named=False)
    session.add(person)
    session.flush()
    return person.id


def _family_code_cell(panel: PersonsPanel, person_id: int) -> str:
    """Return the Family code cell text for *person_id* (id-based, not row index)."""
    table = panel._table
    for row in range(table.rowCount()):
        id_item = table.item(row, COL_ID)
        if id_item is not None and id_item.data(Qt.UserRole) == person_id:
            return table.item(row, COL_FAMILY_CODE).text()
    raise AssertionError(f"person {person_id} not in table")


def test_mark_stale_refreshes_when_visible(db, qtbot):
    with session_scope() as session:
        magdi = _add_person(session, "Abonyi Magdi")

    panel = PersonsPanel()
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)

    assert _family_code_cell(panel, magdi) == ""

    # Simulate an edit committed elsewhere (image browser / sidebar dialog).
    with session_scope() as session:
        PersonService(session).update_person(magdi, family_code="12/H")

    # The cross-tab refresh hook fires.  Because the panel is visible it must
    # reload immediately — no second save of another person required.
    panel.mark_stale()

    assert _family_code_cell(panel, magdi) == "12/H"


def test_mark_stale_defers_until_shown_when_hidden(db, qtbot):
    with session_scope() as session:
        gyurka = _add_person(session, "Barta Gyurka")

    panel = PersonsPanel()
    qtbot.addWidget(panel)
    # Never shown → hidden.  A stale edit should defer, not rebuild eagerly.
    assert not panel.isVisible()

    with session_scope() as session:
        PersonService(session).update_person(gyurka, family_code="7/H")

    panel.mark_stale()
    assert panel._needs_refresh is True
    assert _family_code_cell(panel, gyurka) == ""  # not reloaded yet

    # Showing the tab consumes the pending refresh.
    panel.show()
    qtbot.waitExposed(panel)
    assert panel._needs_refresh is False
    assert _family_code_cell(panel, gyurka) == "7/H"
