"""Widget tests for PlaceSearchSelect — Enter behaviour with many/no matches."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

from app.db.models import Place
from app.ui.widgets.place_search_select import PlaceSearchSelect


def _place(pid: int, name: str) -> Place:
    """Transient Place row carrying just the fields the widget reads."""
    p = Place(name=name)
    p.id = pid
    return p


@pytest.fixture()
def widget(qtbot) -> PlaceSearchSelect:
    w = PlaceSearchSelect()
    qtbot.addWidget(w)
    w.set_places([_place(1, "Balaton"), _place(2, "Szeged")])
    return w


def test_enter_on_unmatched_query_requests_create(widget, qtbot):
    """Typing a brand-new name + Enter asks the caller to create it."""
    created: list[str] = []
    widget.create_requested.connect(created.append)

    widget._search.setText("Visegrád")  # matches no existing place
    qtbot.keyClick(widget._search, Qt.Key_Return)

    assert created == ["Visegrád"]


def test_enter_on_matching_query_selects_existing(widget, qtbot):
    """Enter on a query that matches an existing place selects it, not create."""
    created: list[str] = []
    selected: list[int] = []
    widget.create_requested.connect(created.append)
    widget.place_selected.connect(selected.append)

    widget._search.setText("Bala")  # matches "Balaton"
    qtbot.keyClick(widget._search, Qt.Key_Return)

    assert created == []          # did not ask to create
    assert selected == [1]        # selected the matching place id


def test_enter_on_empty_query_does_nothing(widget, qtbot):
    """Enter with no text must not emit a create request."""
    created: list[str] = []
    widget.create_requested.connect(created.append)

    widget._search.setText("")
    qtbot.keyClick(widget._search, Qt.Key_Return)

    assert created == []
