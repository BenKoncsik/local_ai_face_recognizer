"""Tests for the FaceTimelineView scene construction and bounds."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.face_grouping_service import FaceGroup
from app.ui.widgets.face_timeline_view import _MarkerItem, FaceTimelineView
from app.utils.fuzzy_date import FuzzyDate


def _face(fid, photo_date):
    image = SimpleNamespace(id=fid, photo_date=photo_date, estimated_date=None)
    return SimpleNamespace(id=fid, image=image, crop_path=None, quality_score=0.8)


def _group(fid, photo_date, members=1):
    fs = [_face(fid * 10 + i, photo_date) for i in range(members)]
    g = FaceGroup(members=fs, representative=fs[0])
    return g


@pytest.fixture
def view(qtbot):
    v = FaceTimelineView()
    qtbot.addWidget(v)
    return v


def _markers(view):
    return [it for it in view._scene.items() if isinstance(it, _MarkerItem)]


def test_empty_when_no_dates(view):
    view.set_data([_group(1, None)])
    assert _markers(view) == []
    # The empty-state message is present.
    texts = [it for it in view._scene.items() if it.__class__.__name__ == "QGraphicsTextItem"]
    assert texts


def test_one_marker_per_dated_group(view):
    groups = [_group(1, "1960"), _group(2, "1985"), _group(3, None)]
    view.set_data(groups)
    assert len(_markers(view)) == 2          # the undated group is skipped


def test_bounds_use_birth_and_death(view):
    view.set_data(
        [_group(1, "1960")],
        birth=FuzzyDate.parse("1950"),
        death=FuzzyDate.parse("2000"),
    )
    # Axis should start near birth and end near death (with small padding).
    assert view._start_year <= 1950.5
    assert view._end_year >= 2000.0


def test_bounds_fall_back_to_photo_range(view):
    view.set_data([_group(1, "1960"), _group(2, "1990")])
    assert 1959 <= view._start_year <= 1960.5
    assert 1990 <= view._end_year <= 1991.5


def test_merged_group_marker_has_count_badge(view):
    # A group of 3 → marker pixmap is the padded/stacked variant (wider).
    view.set_data([_group(1, "1970", members=3)])
    markers = _markers(view)
    assert len(markers) == 1
    pm = markers[0].pixmap()
    assert pm.width() > 48        # stacked canvas is _THUMB + 6
