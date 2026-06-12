"""Tests for the date/quality sort ordering in ClusterPanel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.ui.panels.cluster_panel import (
    SORT_DATE_ASC,
    SORT_DATE_DESC,
    SORT_ORIGINAL,
    SORT_QUALITY,
    ClusterPanel,
)


def _face(fid: int, photo_date=None, quality=None):
    image = SimpleNamespace(id=fid, photo_date=photo_date, estimated_date=None)
    return SimpleNamespace(id=fid, image=image, quality_score=quality)


@pytest.fixture
def panel(qtbot):
    p = ClusterPanel()
    qtbot.addWidget(p)
    return p


def test_date_ascending_orders_oldest_first_unknown_last(panel):
    panel._faces = [
        _face(1, "1980"),
        _face(2, "1954.03.12"),
        _face(3, None),            # unknown → last
        _face(4, "1970-es évek"),
    ]
    panel._sort_mode = SORT_DATE_ASC
    order = [f.id for f in panel._sorted_faces()]
    assert order == [2, 4, 1, 3]


def test_date_descending_orders_newest_first_unknown_last(panel):
    panel._faces = [
        _face(1, "1980"),
        _face(2, "1954.03.12"),
        _face(3, None),
        _face(4, "1970-es évek"),
    ]
    panel._sort_mode = SORT_DATE_DESC
    order = [f.id for f in panel._sorted_faces()]
    assert order == [1, 4, 2, 3]


def test_original_order_preserved(panel):
    panel._faces = [_face(3), _face(1), _face(2)]
    panel._sort_mode = SORT_ORIGINAL
    assert [f.id for f in panel._sorted_faces()] == [3, 1, 2]


def test_quality_orders_best_first_unscored_as_usable(panel):
    panel._faces = [
        _face(1, quality=0.3),
        _face(2, quality=0.9),
        _face(3, quality=None),    # None == usable (1.0) → near the top
    ]
    panel._sort_mode = SORT_QUALITY
    order = [f.id for f in panel._sorted_faces()]
    assert order[0] == 3        # 1.0
    assert order[-1] == 1       # 0.3
