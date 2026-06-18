"""Tests for SFace embedder helpers and optional live model."""

from __future__ import annotations

import numpy as np
import pytest

from app.embeddings.sface_embedder import (
    SFaceEmbedder,
    _enhance_grayscale,
    _is_grayscale,
)
from app.paths import resource_path

_MODEL = resource_path("models/sface.onnx")
_model_available = _MODEL.exists()


class TestGrayscaleHelpers:
    def test_is_grayscale_detects_bw(self):
        gray = np.full((32, 32, 3), 128, dtype=np.uint8)
        assert _is_grayscale(gray) is True

    def test_is_grayscale_rejects_color(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[:, :16, 2] = 200
        img[:, 16:, 0] = 180
        assert _is_grayscale(img) is False

    def test_enhance_grayscale_returns_bgr(self):
        gray = np.full((40, 40, 3), 100, dtype=np.uint8)
        out = _enhance_grayscale(gray)
        assert out.shape == (40, 40, 3)
        assert out.dtype == np.uint8


class TestSFaceEmbedder:
    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SFace model not found"):
            SFaceEmbedder(model_path=str(tmp_path / "missing.onnx"))

    @pytest.mark.skipif(not _model_available, reason="SFace model not on disk")
    def test_embed_returns_unit_vector(self):
        embedder = SFaceEmbedder(model_path=str(_MODEL))
        assert embedder.embedding_dim == 128
        assert embedder._backend == "sface"
        face = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        vec = embedder.embed(face)
        assert vec.shape == (128,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-4

    @pytest.mark.skipif(not _model_available, reason="SFace model not on disk")
    def test_grayscale_input_enhanced(self, monkeypatch):
        embedder = SFaceEmbedder(model_path=str(_MODEL))
        called = {"enhance": False}
        orig = _enhance_grayscale

        def _spy(img):
            called["enhance"] = True
            return orig(img)

        monkeypatch.setattr(
            "app.embeddings.sface_embedder._enhance_grayscale", _spy
        )
        gray = np.full((64, 64, 3), 120, dtype=np.uint8)
        embedder.embed(gray)
        assert called["enhance"] is True
