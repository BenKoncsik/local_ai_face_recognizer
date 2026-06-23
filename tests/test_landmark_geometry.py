"""Unit tests for the landmark-geometry false-positive gate."""

from __future__ import annotations

import math

from app.detectors.base import Detection
from app.detectors.landmark_geometry import (
    geometry_reject_reason,
    validate_face_landmarks,
)

# Canonical frontal face inside a 100×100 box at the origin, in the YuNet /
# ArcFace order: right-eye, left-eye, nose, right-mouth, left-mouth.
_GOOD = [[32, 40], [68, 40], [50, 58], [38, 75], [62, 75]]


def _det(landmarks, x=0, y=0, w=100, h=100, conf=0.5) -> Detection:
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, landmarks=landmarks)


def _rotate(pts, deg, cx=50.0, cy=50.0):
    """Rotate landmark points about (cx, cy) by *deg* degrees."""
    rad = math.radians(deg)
    cos, sin = math.cos(rad), math.sin(rad)
    out = []
    for px, py in pts:
        dx, dy = px - cx, py - cy
        out.append([cx + dx * cos - dy * sin, cy + dx * sin + dy * cos])
    return out


class TestPlausibleGeometryPasses:
    def test_canonical_frontal_face(self):
        assert validate_face_landmarks(_det(_GOOD)) is True
        assert geometry_reject_reason(_det(_GOOD)) is None

    def test_mildly_tilted_face(self):
        # 30° in-plane roll — within the 45° tolerance, must still pass.
        assert validate_face_landmarks(_det(_rotate(_GOOD, 30))) is True

    def test_small_face_same_ratios(self):
        # Same constellation scaled into a 40×40 box keeps all the ratios.
        small = [[px * 0.4, py * 0.4] for px, py in _GOOD]
        assert validate_face_landmarks(_det(small, w=40, h=40)) is True

    # --- False-negative protection: real but non-frontal faces must pass. ---
    # Rejecting these would silently delete genuine faces (the worst failure
    # mode of the gate), so they are locked in as must-pass.

    def test_profile_face_bunched_eyes(self):
        # Strong profile: eyes close together (iod ≈ 0.20 of box width).
        lms = [[40, 40], [60, 40], [46, 58], [40, 75], [58, 75]]
        assert validate_face_landmarks(_det(lms)) is True

    def test_three_quarter_view_nose_toward_eye(self):
        lms = [[32, 40], [68, 40], [40, 57], [36, 74], [60, 74]]
        assert validate_face_landmarks(_det(lms)) is True

    def test_wide_set_eyes_baby_face(self):
        lms = [[25, 42], [75, 42], [50, 58], [38, 72], [62, 72]]
        assert validate_face_landmarks(_det(lms)) is True

    def test_profile_face_with_tilt(self):
        # Profile constellation plus a 15° in-plane roll — both real-world
        # variations at once must still pass.
        lms = [[40, 40], [60, 40], [46, 58], [40, 75], [58, 75]]
        assert validate_face_landmarks(_det(_rotate(lms, 15))) is True


class TestImplausibleGeometryRejected:
    def test_vertical_eye_line(self):
        lms = [[50, 32], [50, 68], _GOOD[2], _GOOD[3], _GOOD[4]]
        assert validate_face_landmarks(_det(lms)) is False

    def test_nose_above_eyes(self):
        lms = [_GOOD[0], _GOOD[1], [50, 20], _GOOD[3], _GOOD[4]]
        assert geometry_reject_reason(_det(lms)) == "nose_above_eyes"

    def test_mouth_collapsed_onto_eyes(self):
        lms = [_GOOD[0], _GOOD[1], _GOOD[2], [38, 50], [62, 50]]
        assert geometry_reject_reason(_det(lms)) == "mouth_eye_distance"

    def test_inter_ocular_too_narrow(self):
        lms = [[48, 40], [52, 40], _GOOD[2], _GOOD[3], _GOOD[4]]
        assert geometry_reject_reason(_det(lms)) == "iod_box_ratio"

    def test_inter_ocular_too_wide(self):
        lms = [[2, 40], [98, 40], _GOOD[2], _GOOD[3], _GOOD[4]]
        assert geometry_reject_reason(_det(lms)) == "iod_box_ratio"

    def test_nose_off_to_the_side(self):
        lms = [_GOOD[0], _GOOD[1], [95, 58], _GOOD[3], _GOOD[4]]
        assert geometry_reject_reason(_det(lms)) == "nose_off_center"

    def test_mouth_corners_swapped(self):
        lms = [_GOOD[0], _GOOD[1], _GOOD[2], [62, 75], [38, 75]]
        assert geometry_reject_reason(_det(lms)) == "mouth_corners_swapped"

    def test_scrambled_collinear_landmarks(self):
        # The kind of constellation a tree knot / hand produces.
        lms = [[40, 10], [40, 20], [40, 30], [40, 40], [40, 50]]
        assert validate_face_landmarks(_det(lms)) is False


class TestFailOpen:
    def test_none_landmarks_pass(self):
        assert validate_face_landmarks(_det(None)) is True

    def test_wrong_count_passes(self):
        assert validate_face_landmarks(_det([[10, 10], [20, 20]])) is True

    def test_non_finite_passes(self):
        lms = [[float("nan"), 40], [68, 40], [50, 58], [38, 75], [62, 75]]
        assert validate_face_landmarks(_det(lms)) is True

    def test_coincident_eyes_degenerate_passes(self):
        lms = [[50, 40], [50, 40], [50, 58], [38, 75], [62, 75]]
        assert validate_face_landmarks(_det(lms)) is True

    def test_zero_size_box_passes(self):
        assert validate_face_landmarks(_det(_GOOD, w=0, h=0)) is True
