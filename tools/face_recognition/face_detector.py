"""Face detection using InsightFace (SCRFD / RetinaFace via buffalo_l).

GPU is used when available (ctx_id=0); falls back to CPU (ctx_id=-1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """One face detected in an image."""

    bbox: tuple                           # (x, y, w, h) in pixels
    confidence: float
    landmarks: Optional[np.ndarray]       # 5-point landmarks, shape (5, 2)
    kps: Optional[np.ndarray]             # same as landmarks (InsightFace convention)
    pose: Optional[tuple]                 # (yaw, pitch, roll) in degrees or None
    embedding: Optional[np.ndarray] = None  # filled in by FaceEmbedder


class FaceDetector:
    """InsightFace-backed face detector using the buffalo_l pack.

    The buffalo_l model bundle includes SCRFD for detection and ArcFace for
    embedding.  Detection and embedding are deliberately separated here so
    quality filtering can happen between the two steps.

    Args:
        model_pack:   InsightFace model pack name (default ``"buffalo_l"``).
        gpu:          Force GPU (True), CPU (False), or auto-detect (None).
        det_size:     Input resolution for the detection network.  Larger =
                      slower but catches smaller faces.
        det_thresh:   Detection confidence threshold [0.0 – 1.0].
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

    # ------------------------------------------------------------------

    def detect(self, img_rgb: np.ndarray) -> List[DetectedFace]:
        """Run detection on *img_rgb* (H×W×3, uint8, RGB).

        Returns a list of :class:`DetectedFace` objects sorted by descending
        confidence.  Faces are aligned by InsightFace internally before
        embedding (alignment is always applied during :meth:`FaceEmbedder.embed`).
        """
        if img_rgb is None or img_rgb.size == 0:
            return []

        # InsightFace expects BGR.
        import cv2
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        try:
            raw_faces = self._app.get(img_bgr, max_num=0)
        except Exception as exc:  # noqa: BLE001
            log.error("InsightFace detection failed: %s", exc)
            return []

        result: List[DetectedFace] = []
        for f in raw_faces:
            bbox_xyxy = f.bbox.astype(int)
            x1, y1, x2, y2 = bbox_xyxy
            w, h = max(1, x2 - x1), max(1, y2 - y1)

            pose = None
            if hasattr(f, "pose") and f.pose is not None:
                pose = tuple(float(v) for v in f.pose)

            result.append(DetectedFace(
                bbox=(int(x1), int(y1), int(w), int(h)),
                confidence=float(f.det_score),
                landmarks=f.kps if hasattr(f, "kps") else None,
                kps=f.kps if hasattr(f, "kps") else None,
                pose=pose,
            ))

        result.sort(key=lambda d: d.confidence, reverse=True)
        return result

    # ------------------------------------------------------------------

    def _load_model(self, model_pack: str, gpu: Optional[bool]):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            ctx_id = self._resolve_ctx(gpu)
            app = FaceAnalysis(
                name=model_pack,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if ctx_id >= 0
                else ["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=ctx_id, det_thresh=self._det_thresh, det_size=self._det_size)
            log.info(
                "InsightFace loaded: pack=%s ctx_id=%d det_size=%s thresh=%.2f",
                model_pack, ctx_id, self._det_size, self._det_thresh,
            )
            return app
        except ImportError as exc:
            raise ImportError(
                "insightface is not installed. Run: pip install insightface onnxruntime-gpu"
            ) from exc

    @staticmethod
    def _resolve_ctx(gpu: Optional[bool]) -> int:
        if gpu is False:
            return -1
        if gpu is True:
            return 0
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                log.info("CUDA available — using GPU (ctx_id=0)")
                return 0
        except Exception:  # noqa: BLE001
            pass
        log.info("No CUDA — falling back to CPU (ctx_id=-1)")
        return -1
