"""Unit tests for TFLite backend selection in TFLiteEmbedder._load_tflite().

These tests mock all three import paths so no real model file or runtime is needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Pre-import so cv2 and the embedder module are fully cached in sys.modules
# before any patch.dict context manipulates sys.modules.  Without this, the
# first patch.dict would add these modules and then remove them on exit,
# causing cv2 to be re-imported inside subsequent patches (which triggers a
# cv2.dnn initialisation bug on some platforms).
import cv2  # noqa: F401
import app.embeddings.tflite_embedder  # noqa: F401
from app.embeddings.tflite_embedder import TFLiteEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_interpreter_cls():
    """Return a minimal mock Interpreter class that satisfies TFLiteEmbedder."""
    interp = MagicMock()
    interp.get_input_details.return_value = [{"index": 0, "shape": [1, 112, 112, 3]}]
    interp.get_output_details.return_value = [{"index": 1, "shape": [1, 192]}]

    cls = MagicMock(return_value=interp)
    return cls


def _fake_model_path(tmp_path: Path) -> Path:
    """Write a zero-byte placeholder so Path.exists() returns True."""
    p = tmp_path / "mobilefacenet.tflite"
    p.write_bytes(b"")
    return p


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------

class TestLoadTfliteBackendSelection:
    """_load_tflite() must pick the first available backend."""

    def _load(self, model_path: Path):
        embedder = TFLiteEmbedder.__new__(TFLiteEmbedder)
        embedder._embedding_dim = 192
        embedder._input_w = 112
        embedder._input_h = 112
        embedder._interpreter = None
        embedder._input_index = 0
        embedder._output_index = 0
        embedder._sface = None
        embedder._load_tflite(model_path)
        return embedder

    def test_ai_edge_litert_preferred(self, tmp_path):
        """When ai_edge_litert is available it is used and logged."""
        interp_cls = _make_fake_interpreter_cls()
        fake_module = types.ModuleType("ai_edge_litert")
        fake_module.interpreter = types.ModuleType("ai_edge_litert.interpreter")
        fake_module.interpreter.Interpreter = interp_cls

        path = _fake_model_path(tmp_path)

        with patch.dict(sys.modules, {
            "ai_edge_litert": fake_module,
            "ai_edge_litert.interpreter": fake_module.interpreter,
        }):
            embedder = self._load(path)

        assert embedder._backend == "tflite"
        interp_cls.assert_called_once_with(model_path=str(path))

    def test_tflite_runtime_fallback(self, tmp_path):
        """When ai_edge_litert is absent, tflite_runtime is used."""
        interp_cls = _make_fake_interpreter_cls()
        fake_tflite = types.ModuleType("tflite_runtime")
        fake_tflite.interpreter = types.ModuleType("tflite_runtime.interpreter")
        fake_tflite.interpreter.Interpreter = interp_cls

        path = _fake_model_path(tmp_path)

        with patch.dict(sys.modules, {
            "ai_edge_litert": None,
            "ai_edge_litert.interpreter": None,
            "tflite_runtime": fake_tflite,
            "tflite_runtime.interpreter": fake_tflite.interpreter,
        }):
            embedder = self._load(path)

        assert embedder._backend == "tflite"
        interp_cls.assert_called_once_with(model_path=str(path))

    def test_tensorflow_fallback(self, tmp_path):
        """When both litert and tflite_runtime are absent, tensorflow is used."""
        interp_cls = _make_fake_interpreter_cls()
        fake_tf = types.ModuleType("tensorflow")
        fake_tf.lite = MagicMock()
        fake_tf.lite.Interpreter = interp_cls

        path = _fake_model_path(tmp_path)

        with patch.dict(sys.modules, {
            "ai_edge_litert": None,
            "ai_edge_litert.interpreter": None,
            "tflite_runtime": None,
            "tflite_runtime.interpreter": None,
            "tensorflow": fake_tf,
        }):
            embedder = self._load(path)

        assert embedder._backend == "tflite"
        interp_cls.assert_called_once_with(model_path=str(path))

    def test_no_backend_raises_import_error(self, tmp_path):
        """When no backend is available, ImportError with diagnostics is raised."""
        path = _fake_model_path(tmp_path)

        with patch.dict(sys.modules, {
            "ai_edge_litert": None,
            "ai_edge_litert.interpreter": None,
            "tflite_runtime": None,
            "tflite_runtime.interpreter": None,
            "tensorflow": None,
        }):
            embedder = TFLiteEmbedder.__new__(TFLiteEmbedder)
            embedder._embedding_dim = 192
            embedder._input_w = 112
            embedder._input_h = 112
            embedder._interpreter = None
            embedder._input_index = 0
            embedder._output_index = 0
            embedder._sface = None

            with pytest.raises(ImportError) as exc_info:
                embedder._load_tflite(path)

        msg = str(exc_info.value)
        assert "No TFLite backend found" in msg
        assert "Python" in msg
        assert "Platform" in msg
        assert "pip install ai-edge-litert" in msg

    def test_error_message_includes_attempted_backends(self, tmp_path):
        """The error message lists all attempted backends."""
        path = _fake_model_path(tmp_path)

        with patch.dict(sys.modules, {
            "ai_edge_litert": None,
            "ai_edge_litert.interpreter": None,
            "tflite_runtime": None,
            "tflite_runtime.interpreter": None,
            "tensorflow": None,
        }):
            embedder = TFLiteEmbedder.__new__(TFLiteEmbedder)
            embedder._embedding_dim = 192
            embedder._input_w = 112
            embedder._input_h = 112
            embedder._interpreter = None
            embedder._input_index = 0
            embedder._output_index = 0
            embedder._sface = None

            with pytest.raises(ImportError) as exc_info:
                embedder._load_tflite(path)

        msg = str(exc_info.value)
        assert "ai_edge_litert" in msg
        assert "tflite_runtime" in msg
        assert "tensorflow" in msg


# ---------------------------------------------------------------------------
# Batched inference equivalence
# ---------------------------------------------------------------------------

class TestEmbedBatch:
    """embed_batch must produce the same vectors as per-crop embed."""

    def test_batch_matches_serial(self):
        """A real TFLite model embeds a batch identically to one-by-one."""
        from app.config import AppConfig
        from app.services.embedding_service import build_embedder

        embedder = build_embedder(AppConfig())
        if getattr(embedder, "_backend", None) != "tflite":
            pytest.skip("real MobileFaceNet TFLite model not available")

        rng = np.random.default_rng(7)
        faces = [
            rng.integers(0, 255, size=(80, 70, 3), dtype=np.uint8)
            for _ in range(5)
        ]

        serial = [embedder.embed(f) for f in faces]
        batched = embedder.embed_batch(faces)

        assert len(batched) == len(serial)
        for got, want in zip(batched, serial):
            assert got.shape == want.shape
            assert np.allclose(got, want, atol=1e-4)

    def test_empty_batch_returns_empty(self):
        from app.config import AppConfig
        from app.services.embedding_service import build_embedder

        embedder = build_embedder(AppConfig())
        assert embedder.embed_batch([]) == []


# ---------------------------------------------------------------------------
# HOG stub fallback (no model file, no TFLite backend)
# ---------------------------------------------------------------------------

class TestHOGFallback:
    """When no model file exists, embedder falls back to SFace/HOG stub."""

    def test_embed_returns_normalized_vector(self):
        """HOG stub _embed_hog_stub() returns a unit vector of correct dimension."""
        embedder = TFLiteEmbedder.__new__(TFLiteEmbedder)
        embedder._embedding_dim = 192
        embedder._input_w = 112
        embedder._input_h = 112
        embedder._interpreter = None
        embedder._input_index = 0
        embedder._output_index = 0
        embedder._sface = None
        embedder._backend = "hog_stub"

        # Use a noisy image so HOG produces a non-zero descriptor
        rng = np.random.default_rng(42)
        face = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        vec = embedder._embed_hog_stub(face)
        assert vec.shape == (192,)
        assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5
