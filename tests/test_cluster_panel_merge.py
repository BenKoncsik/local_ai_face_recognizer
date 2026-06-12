"""Tests for the merged ("stacked") view layout logic in ClusterPanel."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QLabel

from app.services.face_grouping_service import FaceGroup
from app.ui.panels.cluster_panel import SORT_ORIGINAL, ClusterPanel, FaceGroupTile


def _face(fid):
    return SimpleNamespace(
        id=fid, crop_path=None, quality_score=None, confidence=0.9,
        image=SimpleNamespace(id=fid, photo_date=None, estimated_date=None),
    )


@pytest.fixture
def panel(qtbot):
    p = ClusterPanel()
    qtbot.addWidget(p)
    return p


def _setup(panel, groups):
    """Wire a panel with pre-computed groups and dummy thumbnails."""
    faces = [m for g in groups for m in g.members]
    panel._faces = faces
    panel._faces_by_id = {f.id: f for f in faces}
    panel._thumbs = {f.id: QLabel() for f in faces}
    panel._groups = groups
    panel._sort_mode = SORT_ORIGINAL


def test_merged_collapses_multi_group_to_one_tile(panel):
    gA = FaceGroup(members=[_face(1), _face(2)], representative=_face(1))
    gA.representative = gA.members[0]
    gB = FaceGroup(members=[_face(3)], representative=_face(3))
    gB.representative = gB.members[0]
    _setup(panel, [gA, gB])

    panel._merged_mode = True
    widgets = panel._ordered_widgets()
    # Group A → one stacked tile; group B (singleton) → its thumbnail.
    assert len(widgets) == 2
    tiles = [w for w in widgets if isinstance(w, FaceGroupTile)]
    assert len(tiles) == 1
    assert tiles[0].rep_face_id == 1


def test_expanding_group_splices_members(panel):
    gA = FaceGroup(members=[_face(1), _face(2)], representative=_face(1))
    gA.representative = gA.members[0]
    _setup(panel, [gA])

    panel._merged_mode = True
    panel._expanded_reps = {1}            # group A expanded
    widgets = panel._ordered_widgets()
    # Both member thumbnails are shown, no tile.
    assert all(isinstance(w, QLabel) and not isinstance(w, FaceGroupTile) for w in widgets)
    assert len(widgets) == 2


def test_unmerged_mode_shows_every_face(panel):
    gA = FaceGroup(members=[_face(1), _face(2)], representative=_face(1))
    gA.representative = gA.members[0]
    gB = FaceGroup(members=[_face(3)], representative=_face(3))
    gB.representative = gB.members[0]
    _setup(panel, [gA, gB])

    panel._merged_mode = False
    widgets = panel._ordered_widgets()
    assert len(widgets) == 3
    assert not any(isinstance(w, FaceGroupTile) for w in widgets)


def test_toggle_handler_flips_expansion(panel):
    gA = FaceGroup(members=[_face(1), _face(2)], representative=_face(1))
    gA.representative = gA.members[0]
    _setup(panel, [gA])
    panel._merged_mode = True

    panel._on_group_toggle(1)
    assert 1 in panel._expanded_reps
    panel._on_group_toggle(1)
    assert 1 not in panel._expanded_reps
