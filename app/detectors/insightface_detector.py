"""InsightFace (SCRFD via the buffalo_l pack) face detector.

An *optional* detector used by the multi-stage verification ensemble as one
independent opinion on whether a crop is a real face.  InsightFace is a heavy
dependency (``insightface`` + ``onnxruntime``) and downloads its model pack on
first use, so it is never required: when the package or the model is missing
the constructor raises, the caller catches it, and the ensemble simply abstains
on this technology.

GPU is used when available (ctx_id=0); otherwise CPU (ctx_id=-1).

Adapted from the reference implementation in
``tools/face_recognition/face_detector.py`` to the project's
:class:`~app.detectors.base.FaceDetector` interface (BGR input,
``(x, y, w, h)`` boxes, ArcFace-ordered 5-point landmarks).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.detectors.base import Detection, FaceDetector

log = logging.getLogger(__name__)


class InsightFaceDetector(FaceDetector):
    """SCRFD face detector backed by InsightFace's ``buffalo_l`` model pack.

    Args:
        model_pack: InsightFace model pack name (default ``"buffalo_l"``).
        gpu:        Force GPU (True), CPU (False) or auto-detect (None).
        det_size:   Detection network input resolution.
        det_thresh: Detector confidence threshold [0.0 – 1.0].

    Raises:
        ImportError:  ``insightface`` / ``onnxruntime`` not installed.
        RuntimeError: the model pack could not be loaded.
    """

    def __init__(
        self,
        model_pack: str = "buffalo_l",
        gpu: Optional[bool] = None,
        det_size: tuple = (640, 640),
        det_thresh: float = 0.5,
    ) -> None:
        self._det_thresh = det_thresh
        self._det_size = det_size
        self._app = self._load_model(model_pack, gpu)

    @property
    def backend_name(self) -> str:
        return "insightface_scrfd"

    # ------------------------------------------------------------------

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 0,
    ) -> List[Detection]:
        """Detect faces in *image_bgr* (BGR uint8); strongest first."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        try:
            raw_faces = self._app.get(image_bgr, max_num=0)
        except Exception as exc:  # noqa: BLE001
            log.warning("InsightFace detection failed: %s", exc)
            return []

        img_h, img_w = image_bgr.shape[:2]
        results: List[Detection] = []
        for f in raw_faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < confidence_threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.bbox.astype(int))
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            if min_face_size and (w < min_face_size or h < min_face_size):
                continue

            landmarks = None
            kps = getattr(f, "kps", None)
            if kps is not None:
                # InsightFace 5-point order is [right-eye, left-eye, nose,
                # right-mouth, left-mouth] — the same ArcFace order this project
                # uses, so no reordering is needed.
                landmarks = [[float(px), float(py)] for px, py in kps]

            results.append(
                Detection(
                    x=x1, y=y1, w=w, h=h,
                    confidence=score,
                    landmarks=landmarks,
                ).clamp(img_w, img_h)
            )

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    # ------------------------------------------------------------------

    def _load_model(self, model_pack: str, gpu: Optional[bool]):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "insightface is not installed. Run: "
                "pip install insightface onnxruntime"
            ) from exc

        from app import accel

        ctx_id = self._resolve_ctx(gpu)
        # Platform-safe provider list: CUDA priority where present (unchanged
        # Windows/Linux path), CoreML only on macOS, CPU always as fallback.
        providers = accel.onnx_providers()
        try:
            app = FaceAnalysis(name=model_pack, providers=providers)
            app.prepare(
                ctx_id=ctx_id,
                det_thresh=self._det_thresh,
                det_size=self._det_size,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load InsightFace pack {model_pack!r}: {exc}"
            ) from exc

        # Report which accelerator the detection session actually bound to.  The
        # detection graph (SCRFD) has dynamic input shapes, so CoreML often only
        # partially offloads — classify_providers still reports CoreML when its
        # provider is in the active list.
        active_providers = self._session_providers(app) or providers
        accel.report_onnx("detection", active_providers, detail="SCRFD")

        log.info(
            "InsightFace loaded: pack=%s ctx_id=%d det_size=%s thresh=%.2f providers=%s",
            model_pack, ctx_id, self._det_size, self._det_thresh, active_providers,
        )
        return app

    @staticmethod
    def _session_providers(app) -> Optional[list]:
        """Active onnxruntime providers from any sub-model session, if exposed."""
        try:
            for model in getattr(app, "models", {}).values():
                session = getattr(model, "session", None)
                if session is not None:
                    return session.get_providers()
        except Exception:  # noqa: BLE001
            pass
        return None

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
