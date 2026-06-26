"""ArcFace ResNet-50 face embedder via InsightFace.

Reuses the recognition model bundled in InsightFace's ``buffalo_l`` model pack
(``w600k_r50`` — an ArcFace ResNet-50) which is *already downloaded* for the
SCRFD co-detector.  No extra model download is required: if the detection
co-detector works, this embedder works.

Embeddings:
    512-dimensional, L2-normalised.  Higher quality (and higher resource use)
    than the lightweight MobileFaceNet TFLite default — recommended only when
    the insightface + onnxruntime packages are installed.

The two backends are NOT interchangeable on an existing library: their vectors
live in different spaces and have different dimensionality (192 vs 512), so
switching requires a full re-embed + AI-model rebuild.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from app.embeddings.base import FaceEmbedder
from app.embeddings.sface_embedder import _enhance_grayscale, _is_grayscale

log = logging.getLogger(__name__)

_INPUT_SIZE = (112, 112)
_EMBED_DIM = 512


class InsightFaceEmbedder(FaceEmbedder):
    """Face embedder backed by InsightFace's ArcFace ResNet-50 recognition model.

    Args:
        model_pack: InsightFace model pack name (default ``"buffalo_l"``).
        gpu:        Force GPU (True), CPU (False) or auto-detect (None).

    Raises:
        ImportError:  ``insightface`` / ``onnxruntime`` not installed.
        RuntimeError: the recognition model could not be loaded from the pack.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        gpu: Optional[bool] = None,
    ) -> None:
        self._backend = "arcface"
        self._rec = self._load_model(model_pack, gpu)

    @property
    def embedding_dim(self) -> int:
        return _EMBED_DIM

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Generate an L2-normalised ArcFace embedding for *face_bgr*.

        The crop is resized to 112×112; alignment quality therefore follows the
        configured ``embedding.crop_mode`` (use ``"aligned"`` for best results).
        """
        if _is_grayscale(face_bgr):
            face_bgr = _enhance_grayscale(face_bgr)
        resized = cv2.resize(face_bgr, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        # ArcFaceONNX.get_feat handles the model's own normalisation/blob prep
        # and returns a single (1, 512) feature for a single input image.
        feat = self._rec.get_feat(resized)
        vec = np.asarray(feat, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(vec))
        return vec if norm < 1e-8 else vec / norm

    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_pack: str, gpu: Optional[bool]):
        # We deliberately do NOT use FaceAnalysis here: it asserts a detection
        # model is present and would load the whole pack (det + landmark +
        # genderage) into memory.  We only need the recognition model, so we
        # load that ONNX directly via the model zoo.
        try:
            from insightface.model_zoo import get_model
            from insightface.utils import storage as if_storage
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "insightface is not installed. Run: "
                "pip install insightface onnxruntime"
            ) from exc

        import glob
        import os

        from app import accel

        ctx_id = InsightFaceEmbedder._resolve_ctx(gpu)
        # Platform-safe provider list: CUDA stays the priority where present
        # (Windows/Linux GPU path unchanged); CoreML is added only on macOS so
        # ArcFace runs on the Apple Neural Engine / GPU.  CPU is the fallback.
        providers = accel.onnx_providers()

        try:
            # Downloads the pack on first use (same pack the SCRFD co-detector
            # uses), then returns the local directory.
            pack_dir = if_storage.ensure_available(
                "models", model_pack, root="~/.insightface"
            )
            rec = None
            for onnx_path in sorted(glob.glob(os.path.join(pack_dir, "*.onnx"))):
                model = get_model(onnx_path, providers=providers)
                if getattr(model, "taskname", None) == "recognition":
                    rec = model
                    break
            if rec is None:
                raise RuntimeError("no recognition model in pack")
            rec.prepare(ctx_id=ctx_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load InsightFace recognition model from "
                f"pack {model_pack!r}: {exc}"
            ) from exc

        # Report the accelerator the session actually bound to (CoreML/ANE,
        # CUDA or CPU) so the Performance tab can show a truthful status.
        try:
            active_providers = rec.session.get_providers()
        except Exception:  # noqa: BLE001
            active_providers = providers
        accel.report_onnx("embedding", active_providers, detail="ArcFace r50")

        log.info(
            "InsightFace ArcFace embedder loaded: pack=%s ctx_id=%d dim=%d providers=%s",
            model_pack, ctx_id, _EMBED_DIM, active_providers,
        )
        return rec

    @staticmethod
    def _resolve_ctx(gpu: Optional[bool]) -> int:
        if gpu is False:
            return -1
        if gpu is True:
            return 0
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                return 0
        except Exception:  # noqa: BLE001
            pass
        return -1
