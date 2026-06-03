"""Reproduce the 'Create typed place' button flow from the image browser."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Image, Place
from app.ui.panels.image_browser_panel import ImageBrowserPanel


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "browser.db")


def _add_image(session, path="/tmp/p.jpg") -> Image:
    img = Image(file_path=path, file_hash=f"h_{path}", file_mtime=0.0)
    session.add(img)
    session.flush()
    return img


def test_create_typed_place_button(db, qtbot):
    with session_scope() as session:
        image_id = _add_image(session).id

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    # Simulate an image being open and a new place name typed into the search.
    panel._current_image_id = image_id
    panel._current_path = "/tmp/p.jpg"
    panel._place_search._search.setText("Visegrád")

    panel._create_or_assign_place_by_text()

    with session_scope() as session:
        place = session.query(Place).filter(Place.name == "Visegrád").one_or_none()
        assert place is not None, "place was not created"
        img = session.get(Image, image_id)
        assert img.place_id == place.id, "place not assigned to image"


def test_create_typed_place_without_image(db, qtbot):
    """Create must work even when no image is selected (used to do nothing)."""
    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._current_image_id = None
    panel._place_search._search.setText("Eger")

    panel._create_or_assign_place_by_text()

    with session_scope() as session:
        assert (
            session.query(Place).filter(Place.name == "Eger").one_or_none() is not None
        ), "place was not created without an image"


def test_create_typed_place_from_rename_field(db, qtbot):
    """A name typed into the rename field is also accepted by Create."""
    with session_scope() as session:
        image_id = _add_image(session).id

    panel = ImageBrowserPanel(config=None)
    qtbot.addWidget(panel)

    panel._current_image_id = image_id
    panel._current_path = "/tmp/p.jpg"
    panel._place_search._search.setText("")          # search box empty
    panel._place_rename_edit.setText("Pécs")          # typed in the other field

    panel._create_or_assign_place_by_text()

    with session_scope() as session:
        assert (
            session.query(Place).filter(Place.name == "Pécs").one_or_none() is not None
        ), "place from rename field was not created"
