"""Multi-stage, multi-technology face-verification gate.

The single-technology :class:`~app.services.face_verification_service.FaceVerifier`
(crop-level YuNet re-detection + landmark geometry) removes most false positives,
but one detector can still be fooled: ears, hands, card/wood textures and other
objects are occasionally re-found "as a face" in their own crop.

This service asks several *independent* technologies about the same enlarged
crop and keeps a detection only when a quorum of them agree it is a face:

* **YuNet** crop re-detection — the existing :class:`FaceVerifier` (DNN #1 +
  landmark geometry).  Always present (it is the base verifier).
* **Caffe SSD (res10)** — OpenCV DNN, an independent deep detector
  (:class:`~app.detectors.cpu_detector.CpuDetector`, only when the Caffe model
  is present — when it would fall back to Haar it abstains, the Haar technology
  below covers that).
* **Haar cascade** — classic Viola–Jones frontal + profile face cascades,
  algorithmically unrelated to the DNNs.
* **Eye cascade** — Haar eye detection in the upper half of the candidate box.
  A real face contains an eye pair; an ear / object / texture does not, so this
  is the strongest discriminator against the exact false positives reported.
* **InsightFace / SCRFD** — optional heavy detector
  (:class:`~app.detectors.insightface_detector.InsightFaceDetector`); abstains
  when the library or model pack is unavailable.

Decision (config-driven, fail-safe):

* only technologies that could give an opinion vote;
* the detection survives when at least
  ``detection.multistage_min_confirmations`` technologies confirm it;
* if fewer than ``detection.multistage_min_available`` technologies are actually
  available, we fall back to the legacy single-YuNet verifier — a stripped-down
  install never deletes more aggressively than before.

The public surface — :meth:`verify` returning the (possibly refined) detection
or ``None``, and the :attr:`available` property — matches :class:`FaceVerifier`
exactly, so every existing gate call site swaps in this validator unchanged.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from app.config import DetectionConfig
from app.detectors.base import Detection, FaceDetector
from app.services.face_verification_service import FaceVerifier
from app.utils.image_utils import apply_clahe

log = logging.getLogger(__name__)

# A re-detected box on the crop confirms the candidate when it overlaps it this
# much (IoU) or has its centre inside it — same rule as FaceVerifier so a face
# elsewhere in the generous margin never confirms a non-face.
_MIN_OVERLAP_IOU = 0.25

# Vote outcomes.
CONFIRM = "confirm"
REJECT = "reject"


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _box_confirms(
    local: Tuple[float, float, float, float],
    boxes: List[Tuple[float, float, float, float]],
) -> bool:
    """True when any of *boxes* sits on the candidate box *local*."""
    for b in boxes:
        if _iou(local, b) >= _MIN_OVERLAP_IOU:
            return True
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        if local[0] <= cx <= local[0] + local[2] and local[1] <= cy <= local[1] + local[3]:
            return True
    return False


class MultiStageFaceValidator:
    """Ensemble false-positive gate; ``verify``/``available`` mirror FaceVerifier.

    Args:
        config:          The detection configuration block (``verification_*`` +
                         ``multistage_*`` fields).
        yunet_detector:  Optional pre-built YuNet detector reused for the base
                         verifier (no second ONNX load).
        verifier:        Optional pre-built :class:`FaceVerifier` (tests / reuse).
    """

    def __init__(
        self,
        config: DetectionConfig,
        yunet_detector: Optional[FaceDetector] = None,
        verifier: Optional[FaceVerifier] = None,
        insightface_detector: Optional[FaceDetector] = None,
    ) -> None:
        self._config = config
        self._verifier = verifier or FaceVerifier(config, detector=yunet_detector)
        # Reuse an already-loaded InsightFace (e.g. the co-detector) instead of
        # loading the ~300 MB model a second time for the gate.
        self._insightface = insightface_detector
        # Each entry: (name, weight, vote_fn(crop, local, min_face) -> vote).
        # A face survives when the CONFIRM weight reaches min_confirmations, so a
        # strong detector (InsightFace, weight 2) can carry a hard profile face
        # the frontal-biased technologies miss, while a lone YuNet hit (weight 1)
        # on hair/texture still needs a second opinion.
        self._techs: List[Tuple[str, int, Callable]] = []
        if config.multistage_enabled:
            self._build_techs()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True when at least the base YuNet verifier can run."""
        return self._verifier.available

    def active_technologies(self) -> List[str]:
        """Names of the technologies that will actually vote (YuNet first)."""
        names: List[str] = []
        if self._verifier.available:
            names.append("yunet")
        names.extend(name for name, _, _ in self._techs)
        return names

    @property
    def _multistage_live(self) -> bool:
        """True when voting is on AND enough technologies are available."""
        if not self._config.multistage_enabled:
            return False
        return len(self.active_technologies()) >= self._config.multistage_min_available

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify(self, image_bgr: np.ndarray, det: Detection) -> Optional[Detection]:
        """Confirm or reject *det* by ensemble vote.

        Returns the (possibly refined) detection when a quorum of technologies
        confirm it, else ``None``.  Falls back to the legacy single-YuNet
        verifier when multi-stage voting is disabled or too few technologies are
        available — so behaviour never regresses below today's gate.
        """
        # YuNet first: it both votes and supplies the refined box/landmarks.
        yunet_det = self._verifier.verify(image_bgr, det)

        if not self._multistage_live:
            return yunet_det

        crop, local, min_face = self._prepare_crop(image_bgr, det)
        if crop is None:
            # Degenerate geometry — do not reject on our own failure; defer to
            # the base verifier's verdict.
            return yunet_det

        confirm_weight = 1 if yunet_det is not None else 0
        votes = {"yunet": CONFIRM if yunet_det is not None else REJECT}
        for name, weight, vote_fn in self._techs:
            try:
                vote = vote_fn(crop, local, min_face)
            except Exception as exc:  # noqa: BLE001
                log.debug("Validator %s errored (%s) — abstaining", name, exc)
                continue
            votes[name] = vote
            if vote == CONFIRM:
                confirm_weight += weight

        keep = confirm_weight >= self._config.multistage_min_confirmations
        if not keep:
            log.debug(
                "MULTISTAGE drop: bbox=(%d,%d,%d,%d) conf=%.2f weight=%d/%d votes=%s",
                det.x, det.y, det.w, det.h, det.confidence,
                confirm_weight, self._config.multistage_min_confirmations, votes,
            )
            return None

        # Prefer YuNet's refined geometry; otherwise keep the original box.
        return yunet_det if yunet_det is not None else det

    # ------------------------------------------------------------------
    # Crop preparation (mirrors FaceVerifier's margin/upscale)
    # ------------------------------------------------------------------

    def _prepare_crop(
        self, image_bgr: np.ndarray, det: Detection
    ) -> Tuple[Optional[np.ndarray], Tuple[float, float, float, float], int]:
        """Return (crop, local candidate box in crop coords, min_face_size)."""
        import cv2

        if det.w <= 1 or det.h <= 1:
            return None, (0, 0, 0, 0), 0

        img_h, img_w = image_bgr.shape[:2]
        margin = self._config.verification_margin
        x1 = max(0, int(det.x - det.w * margin))
        y1 = max(0, int(det.y - det.h * margin))
        x2 = min(img_w, int(det.x + det.w * (1 + margin)))
        y2 = min(img_h, int(det.y + det.h * (1 + margin)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, (0, 0, 0, 0), 0

        crop = image_bgr[y1:y2, x1:x2]
        short_side = min(crop.shape[:2])
        scale = 1.0
        min_crop = self._config.verification_min_crop
        if short_side < min_crop:
            scale = min_crop / short_side
            crop = cv2.resize(
                crop,
                (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
                interpolation=cv2.INTER_CUBIC,
            )

        local = (
            (det.x - x1) * scale,
            (det.y - y1) * scale,
            det.w * scale,
            det.h * scale,
        )
        min_face = max(12, int(min(det.w, det.h) * scale * 0.3))
        return crop, local, min_face

    # ------------------------------------------------------------------
    # Technology builders
    # ------------------------------------------------------------------

    def _build_techs(self) -> None:
        if self._config.multistage_use_caffe:
            self._add_caffe()
        if self._config.multistage_use_haar:
            self._add_haar()
        if self._config.multistage_use_eyes:
            self._add_eyes()
        if self._config.multistage_use_insightface:
            self._add_insightface()

    def _add_caffe(self) -> None:
        """OpenCV DNN Caffe SSD — only when a real Caffe model is present."""
        try:
            from app.detectors.cpu_detector import CpuDetector

            detector = CpuDetector(model_path=self._config.cpu_model_path)
        except Exception as exc:  # noqa: BLE001
            log.info("Multi-stage: Caffe SSD validator unavailable (%s)", exc)
            return
        # When CpuDetector fell back to Haar it is not an independent DNN — skip
        # it so the dedicated Haar technology is not double-counted.
        if not detector.backend_name.endswith("caffe_ssd"):
            log.info("Multi-stage: Caffe model absent — Caffe validator skipped")
            return

        threshold = self._config.verification_threshold

        def vote(crop, local, min_face) -> str:
            found = detector.detect(
                crop, confidence_threshold=threshold, min_face_size=min_face
            )
            boxes = [(d.x, d.y, d.w, d.h) for d in found]
            return CONFIRM if _box_confirms(local, boxes) else REJECT

        self._techs.append(("caffe_ssd", 1, vote))

    def _add_haar(self) -> None:
        """Classic Haar frontal + profile face cascades (always available)."""
        import cv2

        frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        if frontal.empty() and profile.empty():
            log.info("Multi-stage: Haar cascades unavailable — face validator skipped")
            return

        def vote(crop, local, min_face) -> str:
            import cv2 as _cv2

            gray = _cv2.cvtColor(crop, _cv2.COLOR_BGR2GRAY)
            _cv2.equalizeHist(gray, gray)
            floor = max(int(min_face), 20)
            boxes: List[Tuple[float, float, float, float]] = []
            for cascade in (frontal, profile):
                if cascade.empty():
                    continue
                for (x, y, w, h) in cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(floor, floor)
                ):
                    boxes.append((float(x), float(y), float(w), float(h)))
            return CONFIRM if _box_confirms(local, boxes) else REJECT

        self._techs.append(("haar", 1, vote))

    def _add_eyes(self) -> None:
        """Haar eye cascade in the upper face region — the ear/object gate."""
        import cv2

        eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        glasses = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        )
        if eye.empty() and glasses.empty():
            log.info("Multi-stage: eye cascade unavailable — eye validator skipped")
            return

        def vote(crop, local, min_face) -> str:
            import cv2 as _cv2

            lx, ly, lw, lh = local
            ch, cw = crop.shape[:2]
            # Eyes live in the upper ~60% of the face box.
            x1 = max(0, int(lx))
            y1 = max(0, int(ly))
            x2 = min(cw, int(lx + lw))
            y2 = min(ch, int(ly + lh * 0.6))
            if x2 - x1 < 12 or y2 - y1 < 8:
                return REJECT
            region = crop[y1:y2, x1:x2]
            gray = _cv2.cvtColor(region, _cv2.COLOR_BGR2GRAY)
            _cv2.equalizeHist(gray, gray)
            floor = max(8, int(min_face * 0.25))
            for cascade in (eye, glasses):
                if cascade.empty():
                    continue
                eyes = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=4, minSize=(floor, floor)
                )
                if len(eyes) >= 1:
                    return CONFIRM
            return REJECT

        self._techs.append(("eyes", 1, vote))

    def _add_insightface(self) -> None:
        """InsightFace / SCRFD — a strong DNN; reused from the co-detector when
        provided, otherwise loaded once.  Its CONFIRM carries
        ``multistage_insightface_weight`` so it can keep a hard profile face on
        its own (the frontal-biased techs miss those), while never resurrecting a
        hair/texture box it does not detect.  Abstains if unavailable."""
        detector = self._insightface
        if detector is None:
            try:
                from app.detectors.insightface_detector import InsightFaceDetector

                detector = InsightFaceDetector(
                    model_pack=self._config.insightface_model_pack,
                    det_thresh=self._config.verification_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                log.info("Multi-stage: InsightFace validator unavailable (%s)", exc)
                return

        threshold = self._config.verification_threshold
        weight = max(1, int(self._config.multistage_insightface_weight))

        def vote(crop, local, min_face) -> str:
            for variant in (crop, apply_clahe(crop)):
                found = detector.detect(
                    variant, confidence_threshold=threshold, min_face_size=0
                )
                boxes = [(d.x, d.y, d.w, d.h) for d in found]
                if _box_confirms(local, boxes):
                    return CONFIRM
            return REJECT

        self._techs.append(("insightface", weight, vote))
