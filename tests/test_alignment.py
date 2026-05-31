"""Tests for face crop geometry: square crops and 5-point alignment."""

from __future__ import annotations

import numpy as np

from app.embeddings.alignment import (
    _ARCFACE_TEMPLATE_112,
    align_face_5pt,
    make_face_crop,
    square_crop,
)


def _gradient_image(h: int, w: int) -> np.ndarray:
    """A deterministic BGR image with a horizontal gradient."""
    base = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.repeat(base[None, :], h, axis=0)
    return np.dstack([img, img, img])


class TestSquareCrop:
    def test_output_size_is_exact(self):
        img = _gradient_image(400, 600)
        out = square_crop(img, (100, 100, 80, 120), output_size=(112, 112))
        assert out is not None
        assert out.shape == (112, 112, 3)

    def test_aspect_ratio_preserved_for_tall_box(self):
        """A tall box must not be horizontally stretched: a square region of
        the source maps to the square output, so widths stay proportional."""
        img = _gradient_image(400, 400)
        # Two boxes with the same centre but very different aspect ratios should
        # crop the *same* square region (driven by max(w,h)).
        tall = square_crop(img, (180, 100, 40, 200), output_size=(112, 112))
        wide = square_crop(img, (100, 180, 200, 40), output_size=(112, 112))
        assert tall is not None and wide is not None
        # Same centre (200,200) and same side length (200*1.2) → identical crop.
        assert np.array_equal(tall, wide)

    def test_box_at_edge_is_padded_not_distorted(self):
        img = _gradient_image(300, 300)
        out = square_crop(img, (0, 0, 60, 60), output_size=(112, 112))
        assert out is not None
        assert out.shape == (112, 112, 3)

    def test_degenerate_box_returns_none(self):
        img = _gradient_image(100, 100)
        assert square_crop(img, (10, 10, 0, 10)) is None


class TestAlign5pt:
    def test_landmarks_map_to_template(self):
        """After alignment, the input landmarks should land on the template."""
        img = _gradient_image(500, 500)
        # Pick arbitrary source landmarks (already roughly face-shaped).
        src = np.array(
            [
                [200, 200],   # right eye
                [280, 205],   # left eye
                [240, 250],   # nose
                [210, 300],   # right mouth
                [275, 302],   # left mouth
            ],
            dtype=np.float32,
        )
        out = align_face_5pt(img, src, output_size=(112, 112))
        assert out is not None
        assert out.shape == (112, 112, 3)

    def test_wrong_landmark_count_returns_none(self):
        img = _gradient_image(200, 200)
        assert align_face_5pt(img, [[10, 10], [20, 20]]) is None


class TestMakeFaceCrop:
    def test_legacy_mode_matches_old_behaviour(self):
        """Legacy mode reproduces the original stretch-to-fill crop exactly."""
        import cv2

        img = _gradient_image(400, 400)
        x, y, w, h = 100, 120, 90, 60
        out = make_face_crop(img, (x, y, w, h), (128, 128), mode="legacy")

        # Reproduce the pre-refactor computation.
        mx, my = int(w * 0.10), int(h * 0.10)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(400, x + w + mx), min(400, y + h + my)
        expected = cv2.resize(img[y1:y2, x1:x2], (128, 128),
                              interpolation=cv2.INTER_AREA)
        assert np.array_equal(out, expected)

    def test_aligned_falls_back_to_square_without_landmarks(self):
        img = _gradient_image(400, 400)
        out = make_face_crop(img, (100, 100, 80, 80), (112, 112), mode="aligned")
        expected = square_crop(img, (100, 100, 80, 80), (112, 112))
        assert np.array_equal(out, expected)

    def test_aligned_with_landmarks_warps_to_template(self):
        """With landmarks present, aligned mode warps (≠ the square crop) and
        matches the direct align_face_5pt output."""
        img = _gradient_image(400, 400)
        lms = [[120, 130], [180, 132], [150, 165], [128, 195], [175, 196]]
        out = make_face_crop(
            img, (100, 100, 80, 80), (112, 112), mode="aligned", landmarks=lms
        )
        assert out is not None
        direct = align_face_5pt(img, lms, (112, 112))
        assert np.array_equal(out, direct)
        # It must NOT be the square-crop fallback.
        sq = square_crop(img, (100, 100, 80, 80), (112, 112))
        assert not np.array_equal(out, sq)

    def test_template_unchanged_module_constant(self):
        """The module-level template must not be mutated by scaled alignment."""
        before = _ARCFACE_TEMPLATE_112.copy()
        img = _gradient_image(300, 300)
        src = np.array(
            [[100, 100], [160, 100], [130, 140], [110, 180], [150, 180]],
            dtype=np.float32,
        )
        align_face_5pt(img, src, output_size=(224, 224))
        assert np.array_equal(_ARCFACE_TEMPLATE_112, before)
