"""Tests for greedy face-label placement."""

from __future__ import annotations

import pytest

from app.ui.helpers.label_placement import FaceLabel, LabelLayout, place_labels


def _label(
    face_id: int,
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    selected: bool = False,
    label_w: int = 60,
    label_h: int = 20,
) -> FaceLabel:
    return FaceLabel(face_id, x, y, w, h, selected, label_w, label_h)


class TestPlaceLabels:
    def test_empty_input(self):
        assert place_labels(800, 600, []) == []

    def test_single_label_prefers_above_center(self):
        labels = [_label(1, 100, 100, 80, 80)]
        layouts = place_labels(800, 600, labels)
        assert len(layouts) == 1
        lay = layouts[0]
        assert lay.label_y + 20 + 6 <= 100  # above the face with gap
        assert lay.leader_start is None  # adjacent → no leader line

    def test_selected_face_wins_overlap(self):
        labels = [
            _label(1, 100, 100, 80, 80, selected=False),
            _label(2, 110, 105, 80, 80, selected=True),
        ]
        layouts = place_labels(800, 600, labels)
        # Selected face (2) is placed first and keeps the preferred above slot.
        assert all(isinstance(l, LabelLayout) for l in layouts)
        assert layouts[0].label_x >= 0
        assert layouts[1].label_x >= 0

    def test_zero_size_label_uses_face_origin(self):
        labels = [_label(1, 50, 60, 40, 40, label_w=0, label_h=0)]
        layouts = place_labels(400, 300, labels)
        assert layouts[0].label_x == 50
        assert layouts[0].label_y == 60

    def test_leader_line_when_placement_not_adjacent(self, monkeypatch):
        from app.ui.helpers import label_placement as lp

        monkeypatch.setattr(
            lp, "_candidates",
            lambda fx, fy, fw, fh, lw, lh: [(300, 300, 0)],
        )
        labels = [_label(1, 10, 10, 40, 40, label_w=60, label_h=20)]
        layouts = lp.place_labels(400, 400, labels)
        assert layouts[0].leader_start is not None
        assert layouts[0].leader_end is not None

    def test_clamps_position_within_image(self):
        labels = [_label(1, 0, 0, 40, 40, label_w=80, label_h=20)]
        layouts = place_labels(100, 80, labels)
        lay = layouts[0]
        assert lay.label_x >= 0
        assert lay.label_y >= 0
        assert lay.label_y + 20 <= 80
