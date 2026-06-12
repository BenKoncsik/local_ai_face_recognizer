"""End-to-end smoke test: show_person → sort → merge toggle without errors."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.ui.panels.cluster_panel import (
    SORT_DATE_ASC,
    ClusterPanel,
    FaceGroupTile,
)


def _face(fid, photo_date, vec=None):
    arr = None if vec is None else np.asarray(vec, dtype=np.float32)
    image = SimpleNamespace(
        id=fid, file_path=f"/tmp/img{fid}.jpg",
        photo_date=photo_date, estimated_date=None,
    )
    return SimpleNamespace(
        id=fid, person_id=1, person=SimpleNamespace(name="Anna"),
        image=image, crop_path=None,
        bbox_x=0, bbox_y=0, bbox_w=10, bbox_h=10,
        confidence=0.9, detector_backend="cpu",
        is_low_quality=False, auto_merge_review_status=None,
        is_uncertain_identification=False, identification_note=None,
        quality_score=0.8,
        get_embedding=lambda a=arr: a,
    )


@pytest.fixture
def panel(qtbot):
    p = ClusterPanel()
    qtbot.addWidget(p)
    return p


def test_show_sort_and_merge_roundtrip(panel):
    faces = [
        _face(1, "1980", vec=[1.0, 0.0]),
        _face(2, "1954", vec=[1.0, 0.01]),   # ~identical to face 1's vec
        _face(3, "1970-es évek", vec=[0.0, 1.0]),
    ]
    panel.show_person("Anna", faces)
    assert len(panel._ordered_thumbs) == 3

    # Date ascending.
    panel.set_merged_mode(False)
    panel._sort_combo.setCurrentIndex(panel._sort_combo.findData(SORT_DATE_ASC))
    ids = [w.face_id for w in panel._ordered_thumbs]
    assert ids[0] == 2          # 1954 oldest first

    # Merge on: faces 1 & 2 collapse into a stacked tile, face 3 stays.
    panel.set_merged_mode(True)
    tiles = [w for w in panel._ordered_thumbs if isinstance(w, FaceGroupTile)]
    assert len(tiles) == 1
    assert tiles[0]._count == 2

    # Expand the stack → its two members reappear.
    panel._on_group_toggle(tiles[0].rep_face_id)
    assert all(not isinstance(w, FaceGroupTile) for w in panel._ordered_thumbs)
    assert len(panel._ordered_thumbs) == 3

    # Merge off again → back to three plain thumbnails.
    panel.set_merged_mode(False)
    assert len(panel._ordered_thumbs) == 3
