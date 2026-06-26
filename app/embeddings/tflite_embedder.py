"""TFLite-based face embedder (CPU).

Uses a MobileFaceNet TFLite model to produce 192-dimensional L2-normalised
face embeddings.  This runs entirely on CPU — Coral is NOT used here.

Why CPU for embeddings?
-----------------------
The Edge TPU excels at accelerating the first few layers of quantised
models, but the full embedding pipeline (detection → alignment → embedding)
benefits more from throughput on CPU at batch level.  More importantly,
no practical, freely-redistributable ArcFace / MobileFaceNet model compiled
for Edge TPU is currently available.  The CPU TFLite path is transparent,
easy to swap, and fast enough for batch processing.

Model download
--------------
MobileFaceNet TFLite (float32, ~2 MB):

    Option A — use the model from the insightface project:
        https://github.com/deepinsight/insightface/tree/master/model_zoo

    Option B — use the ONNX → TFLite conversion from:
        https://github.com/sirius-ai/MobileFaceNet_TF

    Option C — use any compatible 112×112 input ArcFace TFLite model.

Place the model file at ``models/mobilefacenet.tflite`` (or set
``embedding.model_path`` in config.yaml).

Fallback
--------
If the TFLite model file is missing, the embedder falls back to a
deterministic HOG + PCA stub so the rest of the pipeline can still be
exercised.  The stub is clearly documented and produces lower-quality
embeddings.  It is NOT suitable for production.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.embeddings.base import FaceEmbedder
from app.paths import resource_path

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = "models/mobilefacenet.tflite"


class TFLiteEmbedder(FaceEmbedder):
    """Face embedder backed by a MobileFaceNet TFLite model.

    NOTE: Runs on CPU only.  No Edge TPU usage.

    Args:
        model_path:    Path to ``.tflite`` model file.
        embedding_dim: Expected output dimension (must match the model).
        input_size:    (width, height) expected by the model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 192,
        input_size: tuple[int, int] = (112, 112),
    ) -> None:
        self._embedding_dim = embedding_dim
        self._input_w, self._input_h = input_size
        self._interpreter = None
        self._input_index: int = 0
        self._output_index: int = 0
        # Batch dimension currently allocated on the interpreter's input tensor.
        # The model is loaded with a fixed batch of 1; ``_ensure_batch`` resizes
        # it on demand so single-crop and batched inference can share one
        # interpreter without re-allocating every call.
        self._allocated_n: Optional[int] = None
        # Tri-state: None = batch support not probed yet, True/False = known.
        # Many MobileFaceNet/ArcFace TFLite graphs are exported with a fixed
        # batch of 1 and the XNNPACK delegate refuses to reshape them, so true
        # batching is a best-effort optimisation that disables itself on the
        # first failure and falls back to per-crop inference.
        self._batch_supported: Optional[bool] = None
        self._sface = None

        resolved = Path(model_path) if model_path else resource_path(_DEFAULT_MODEL_PATH)

        if resolved.exists():
            self._load_tflite(resolved)
        else:
            log.warning(
                "Embedding model not found at %s — trying SFace fallback.",
                resolved,
            )
            self._try_sface_fallback()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _try_sface_fallback(self) -> None:
        try:
            from app.embeddings.sface_embedder import SFaceEmbedder

            self._sface = SFaceEmbedder()
            self._embedding_dim = self._sface.embedding_dim
            self._backend = "sface"
            log.info("Using SFace embedder as fallback (backend: sface, dim=128)")
        except FileNotFoundError:
            log.warning(
                "SFace model not found at models/sface.onnx — using HOG stub fallback. "
                "Run build_and_run.sh or download manually: "
                "curl -L https://github.com/opencv/opencv_zoo/raw/main/models/"
                "face_recognition_sface/face_recognition_sface_2021dec.onnx "
                "-o models/sface.onnx",
            )
            self._backend = "hog_stub"
        except Exception as exc:
            log.warning("SFace fallback failed (%s) — using HOG stub", exc)
            self._backend = "hog_stub"

    def _load_tflite(self, model_path: Path) -> None:
        """Load the TFLite model via ai-edge-litert, tflite-runtime, or tensorflow."""
        import platform
        import sys

        Interpreter = None
        attempted: list[str] = []

        try:
            # Primary backend: ai-edge-litert (replaces tflite-runtime since 2024)
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import]
            log.info("Embedding: using ai_edge_litert backend")
        except ImportError as e1:
            attempted.append(f"ai_edge_litert: {e1}")

        if Interpreter is None:
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore[import]
                Interpreter = tflite.Interpreter
                log.info("Embedding: using tflite_runtime backend")
            except ImportError as e2:
                attempted.append(f"tflite_runtime: {e2}")

        if Interpreter is None:
            try:
                import tensorflow as tf  # type: ignore[import]
                Interpreter = tf.lite.Interpreter
                log.info("Embedding: using tensorflow.lite backend")
            except ImportError as e3:
                attempted.append(f"tensorflow: {e3}")

        if Interpreter is None:
            raise ImportError(
                "No TFLite backend found.\n"
                f"  Python:   {sys.version}\n"
                f"  Platform: {sys.platform} / {platform.machine()}\n"
                f"  Attempted: {'; '.join(attempted)}\n"
                "  Fix: pip install ai-edge-litert\n"
                "  (Supports Python 3.9-3.12+ on Linux x64/ARM64, macOS, Windows x64)\n"
                "  Alternative: pip install tensorflow"
            )

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]

        # Validate expected input shape
        _, h, w, c = input_details[0]["shape"]
        if (int(w), int(h)) != (self._input_w, self._input_h):
            log.warning(
                "Model input size %dx%d differs from configured %dx%d — "
                "updating input size to match model.",
                w, h, self._input_w, self._input_h,
            )
            self._input_w, self._input_h = int(w), int(h)

        # Update embedding dim from model output
        out_shape = output_details[0]["shape"]
        if len(out_shape) >= 2:
            self._embedding_dim = int(out_shape[-1])

        # The model is allocated with a batch of 1 here; _ensure_batch tracks
        # and resizes this lazily for batched inference.
        self._allocated_n = 1
        self._backend = "tflite"
        # TFLite on macOS uses XNNPACK (CPU) — never the ANE — so report CPU.
        from app import accel
        accel.report("embedding", accel.ACC_CPU, "MobileFaceNet TFLite (XNNPACK)")
        log.info(
            "Embedding model loaded: %s  (input=%dx%d, dim=%d)",
            model_path.name, self._input_w, self._input_h, self._embedding_dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        if self._interpreter is not None:
            return self._embed_tflite(face_bgr)
        if self._sface is not None:
            return self._sface.embed(face_bgr)
        return self._embed_hog_stub(face_bgr)

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> List[np.ndarray]:
        """Embed many crops in a single TFLite invocation when the model allows.

        Only the real TFLite path attempts batching; the SFace/HOG fallbacks
        have no batch API, so they reuse the per-crop default from the base
        class.  If the model's graph cannot be reshaped to a batch > 1 (common
        for fixed-batch exports), batching disables itself and every call falls
        back to fast per-crop inference on this interpreter.
        """
        if self._interpreter is None:
            return super().embed_batch(faces_bgr)
        if not faces_bgr:
            return []
        if self._batch_supported is False or len(faces_bgr) == 1:
            return [self._embed_tflite(face_bgr) for face_bgr in faces_bgr]

        batch = np.concatenate(
            [self._preprocess(face_bgr) for face_bgr in faces_bgr], axis=0
        )
        n = batch.shape[0]
        try:
            self._ensure_batch(n)
            self._interpreter.set_tensor(self._input_index, batch)
            self._interpreter.invoke()
            out = self._interpreter.get_tensor(self._output_index)
        except (RuntimeError, ValueError) as exc:
            # The model graph won't reshape to a batch — disable batching for
            # this interpreter and continue per-crop (still benefits from the
            # caller's parallel crop loading).
            log.info(
                "TFLite model does not support batched inference (%s) — "
                "using per-crop embedding.",
                exc,
            )
            self._batch_supported = False
            self._restore_single_batch()
            return [self._embed_tflite(face_bgr) for face_bgr in faces_bgr]

        self._batch_supported = True
        return [self._l2_normalise(out[i].astype(np.float32)) for i in range(n)]

    def _restore_single_batch(self) -> None:
        """Reset the interpreter to a batch of 1 after a failed batch resize."""
        try:
            self._allocated_n = None
            self._ensure_batch(1)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not restore single-batch interpreter state: %s", exc)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """Resize and normalise face crop for model input."""
        from app.embeddings.sface_embedder import _is_grayscale, _enhance_grayscale
        if _is_grayscale(face_bgr):
            face_bgr = _enhance_grayscale(face_bgr)
        resized = cv2.resize(
            face_bgr, (self._input_w, self._input_h), interpolation=cv2.INTER_LINEAR
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalise to [-1, 1] (standard for MobileFaceNet / ArcFace models)
        normalised = (rgb.astype(np.float32) - 127.5) / 128.0
        return np.expand_dims(normalised, axis=0)  # (1, H, W, 3)

    def _ensure_batch(self, n: int) -> None:
        """Resize the interpreter's input tensor to a batch of *n* if needed.

        Re-allocating tensors is only done when the requested batch size differs
        from the one currently allocated, so a steady-size embedding loop pays
        the cost once rather than every call.
        """
        if self._allocated_n == n:
            return
        self._interpreter.resize_tensor_input(
            self._input_index, [n, self._input_h, self._input_w, 3]
        )
        self._interpreter.allocate_tensors()
        self._allocated_n = n

    def _embed_tflite(self, face_bgr: np.ndarray) -> np.ndarray:
        """Run TFLite inference for a single crop."""
        inp = self._preprocess(face_bgr)
        self._ensure_batch(1)
        self._interpreter.set_tensor(self._input_index, inp)
        self._interpreter.invoke()
        embedding = self._interpreter.get_tensor(self._output_index)[0]
        return self._l2_normalise(embedding.astype(np.float32))

    def _embed_hog_stub(self, face_bgr: np.ndarray) -> np.ndarray:
        """HOG-based stub embedding — low quality, for development only.

        WARNING: This produces embeddings that are NOT comparable to
        TFLite model outputs.  Only use when the real model is unavailable
        and you are testing pipeline plumbing, not face recognition quality.
        """
        resized = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )
        descriptor = hog.compute(gray).flatten()

        # Trim or pad to match configured embedding_dim
        if len(descriptor) >= self._embedding_dim:
            vec = descriptor[: self._embedding_dim]
        else:
            vec = np.pad(descriptor, (0, self._embedding_dim - len(descriptor)))

        return self._l2_normalise(vec.astype(np.float32))

    @staticmethod
    def _l2_normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return vec
        return vec / norm
