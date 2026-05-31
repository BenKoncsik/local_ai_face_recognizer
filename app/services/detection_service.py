"""Face detection service.

Processes a list of image IDs, runs the face detector, saves face records
and crop thumbnails to disk and database.

Detection modes
---------------
* Fast (default): single pass, normal confidence threshold.
* High accuracy: multiple preprocessing variants per image (original,
  CLAHE, gamma-brightened), lower confidence threshold, IoU-based merge of
  duplicate bounding boxes from the different variants.

Manual faces (detector_backend == "manual") are never deleted by this
service — they are preserved across re-detection runs.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import AppConfig
from app.db.models import Face, Image
from app.detectors.base import Detection, FaceDetector
from app.detectors.cpu_detector import CpuDetector
from app.services.face_crop_service import face_debug_state
from app.services.face_quality_service import FaceQualityEvaluator
from app.services.image_library_service import resolve_image_path
from app.utils.image_utils import (
    apply_bilateral,
    apply_clahe,
    apply_gamma,
    apply_histeq,
    compute_dhash,
    get_exif_orientation,
    load_image_bgr_normalized,
    save_face_crop,
)

log = logging.getLogger(__name__)

# Diagnostic logger — separate so it can be directed to its own handler.
diag_log = logging.getLogger(__name__ + ".diag")


def _iou(a: Detection, b: Detection) -> float:
    """Compute Intersection-over-Union for two detections."""
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x, b.y, b.x2, b.y2
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(
    detections: List[Detection], iou_threshold: float = 0.35
) -> List[Detection]:
    """Greedy NMS: keep the highest-confidence box; suppress overlapping boxes.

    Unlike ``cv2.dnn.NMSBoxes`` this works on plain :class:`Detection` objects
    and does not require OpenCV DNN to be present.
    """
    if len(detections) <= 1:
        return list(detections)

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: List[Detection] = []
    for det in sorted_dets:
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return kept


class DetectionService:
    """Runs face detection on images that haven't been processed yet.

    Args:
        session:       SQLAlchemy session.
        detector:      A :class:`~app.detectors.base.FaceDetector` instance.
        config:        Full application configuration.
        progress_cb:   Optional ``(current, total, path)`` progress callback.
        high_accuracy: When ``True`` the service runs multiple preprocessing
                       variants per image and uses a lower confidence threshold.
    """

    def __init__(
        self,
        session: Session,
        detector: FaceDetector,
        config: AppConfig,
        progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
        high_accuracy: bool = False,
    ) -> None:
        self._session = session
        self._detector = detector
        self._config = config
        self._crops_dir = config.crops_dir_resolved
        self._crops_dir.mkdir(parents=True, exist_ok=True)
        self._progress_cb = progress_cb or (lambda *_: None)
        self._high_accuracy = high_accuracy
        self._quality_evaluator = FaceQualityEvaluator()
        # In-memory dhash registry for duplicate-image diagnostics
        self._dhash_registry: dict[int, tuple[str, int]] = {}  # dhash → (path, face_count)

    def process(self, image_ids: List[int]) -> int:
        """Detect faces in the given image IDs.

        Args:
            image_ids: Primary keys from the ``images`` table.

        Returns:
            Total number of faces detected across all images.
        """
        total = len(image_ids)
        total_faces = 0

        for idx, image_id in enumerate(image_ids, start=1):
            image: Optional[Image] = self._session.get(Image, image_id)
            if image is None:
                log.warning("Image id=%d not found in DB — skipping", image_id)
                continue

            display_path = image.file_path or f"image id={image.id}"
            self._progress_cb(idx, total, display_path)

            try:
                n = self._process_image(image)
                total_faces += n
            except Exception as exc:  # noqa: BLE001
                log.error("Detection failed for %s: %s", display_path, exc)
            finally:
                image.detection_done = True
                self._session.commit()

        log.info("Detection complete: %d face(s) across %d image(s)", total_faces, total)
        return total_faces

    def _process_image(self, image: Image) -> int:
        """Detect and persist faces in a single image.

        Manual faces (detector_backend == "manual") are kept intact.

        Returns the number of newly detected faces.
        """
        path = resolve_image_path(image)
        if path is None:
            log.warning("Cannot resolve path for image id=%d", image.id)
            return 0
        if not path.exists():
            log.warning("Image file missing on disk: %s", path)
            return 0

        # Always load with EXIF orientation correction applied.
        img_bgr = load_image_bgr_normalized(str(path))
        if img_bgr is None:
            log.warning("OpenCV could not read: %s", path)
            return 0

        h, w = img_bgr.shape[:2]
        image.width = w
        image.height = h

        # --- Diagnostic: EXIF orientation ---
        exif_orient = get_exif_orientation(str(path))

        # --- Compute perceptual hash for duplicate diagnostics ---
        dhash = compute_dhash(img_bgr)

        try:
            detections = self._run_detection(img_bgr)
        except RuntimeError as exc:
            if "EdgeTPU" in str(exc) or "Falling back to CPU" in str(exc):
                log.warning(
                    "Coral TPU inference failed (%s) — switching to CPU detector.", exc
                )
                self._detector = CpuDetector(
                    model_path=self._config.detection.cpu_model_path
                )
                detections = self._run_detection(img_bgr)
            else:
                raise

        # --- Diagnostic log ---
        mode_label = "high_accuracy" if self._high_accuracy else "fast"
        threshold = (
            self._config.detection.high_accuracy_confidence_threshold
            if self._high_accuracy
            else self._config.detection.confidence_threshold
        )
        diag_log.debug(
            "DETECT | path=%s | dhash=%016x | exif_orient=%d | size=%dx%d | "
            "mode=%s | threshold=%.2f | faces=%d | bboxes=%s | confidences=%s",
            path,
            dhash,
            exif_orient,
            w, h,
            mode_label,
            threshold,
            len(detections),
            [(d.x, d.y, d.w, d.h) for d in detections],
            [round(d.confidence, 3) for d in detections],
        )

        # --- Duplicate-image diagnostic ---
        hamming_closest = None
        for stored_hash, (stored_path, stored_faces) in self._dhash_registry.items():
            hamming = bin(dhash ^ stored_hash).count("1")
            if hamming <= 10:  # ~10/64 bits differ → visually similar
                hamming_closest = (stored_path, stored_faces, hamming)
                break

        if hamming_closest is not None:
            similar_path, similar_faces, hamming = hamming_closest
            if len(detections) != similar_faces:
                diag_log.warning(
                    "DUPLICATE_MISMATCH | path=%s | similar=%s | "
                    "faces_here=%d | faces_there=%d | hamming=%d | exif_orient=%d",
                    path, similar_path, len(detections), similar_faces, hamming, exif_orient,
                )

        self._dhash_registry[dhash] = (str(path), len(detections))

        # --- Remove stale AUTO-detected faces; keep manual ones AND named ones ---
        # Collect kept named faces BEFORE deletion so we can dedup new detections.
        kept_named: List[Detection] = [
            Detection(x=f.bbox_x, y=f.bbox_y, w=f.bbox_w, h=f.bbox_h, confidence=1.0)
            for f in self._session.query(Face).filter(
                Face.image_id == image.id,
                Face.person_id.isnot(None),
                Face.detector_backend != "manual",
            ).all()
        ]

        # Only delete unnamed auto-detected faces; preserve named training examples.
        self._session.query(Face).filter(
            Face.image_id == image.id,
            Face.detector_backend != "manual",
            Face.person_id.is_(None),
        ).delete(synchronize_session="fetch")

        # Skip any new detection that substantially overlaps an already-named face
        # (prevents duplicate face records for the same person).
        if kept_named:
            iou_thresh = self._config.detection.iou_merge_threshold
            detections = [
                det for det in detections
                if all(_iou(det, nd) < iou_thresh for nd in kept_named)
            ]
            diag_log.debug(
                "Named-face dedup: %d kept named face(s); %d new detection(s) after dedup",
                len(kept_named), len(detections),
            )

        for det in detections:
            face = Face(
                image_id=image.id,
                bbox_x=det.x,
                bbox_y=det.y,
                bbox_w=det.w,
                bbox_h=det.h,
                confidence=det.confidence,
                detector_backend=self._detector.backend_name,
                crop_path=None,
            )
            if det.landmarks is not None:
                face.set_landmarks(det.landmarks)
            self._session.add(face)
            self._session.flush()

            crop_path = save_face_crop(
                img_bgr=img_bgr,
                detection=det,
                crops_dir=self._crops_dir,
                image_id=image.id,
                thumbnail_size=self._config.scan.thumbnail_size,
                face_index=face.id,
                crop_mode=self._config.embedding.crop_mode,
                landmarks=det.landmarks,
            )
            if crop_path is not None:
                face.crop_path = str(crop_path)
            # Evaluate quality now that bbox and crop are set.
            self._quality_evaluator.evaluate_and_update(face)
            log.debug(
                "Detected face entity: %s",
                face_debug_state(face, str(crop_path) if crop_path else None),
            )

        log.debug("Image %s: %d face(s) detected", path.name, len(detections))
        return len(detections)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _run_detection(self, img_bgr: np.ndarray) -> List[Detection]:
        """Dispatch to fast or high-accuracy detection."""
        if self._high_accuracy:
            return self._detect_high_accuracy(img_bgr)
        return self._detector.detect(
            img_bgr,
            confidence_threshold=self._config.detection.confidence_threshold,
            min_face_size=self._config.detection.min_face_size,
        )

    def _detect_high_accuracy(self, img_bgr: np.ndarray) -> List[Detection]:
        """Multi-variant detection with IoU-based deduplication.

        Preprocessing variants tried:
        1. Original (EXIF-normalised) image
        2. CLAHE contrast-enhanced
        3. Gamma-brightened (γ=0.7)

        All detections are merged using greedy IoU NMS.
        """
        threshold = self._config.detection.high_accuracy_confidence_threshold
        min_size = self._config.detection.min_face_size
        iou_merge = self._config.detection.iou_merge_threshold

        all_dets: List[Detection] = []

        variants = [
            ("original",    img_bgr),
            ("clahe",       apply_clahe(img_bgr)),
            ("gamma_bright", apply_gamma(img_bgr, gamma=0.7)),
            ("histeq",      apply_histeq(img_bgr)),
            ("bilateral",   apply_bilateral(img_bgr)),
        ]

        for name, variant_img in variants:
            try:
                dets = self._detector.detect(
                    variant_img,
                    confidence_threshold=threshold,
                    min_face_size=min_size,
                )
                diag_log.debug(
                    "HA variant=%s: %d detection(s) at threshold=%.2f",
                    name, len(dets), threshold,
                )
                all_dets.extend(dets)
            except Exception as exc:  # noqa: BLE001
                log.warning("HA variant=%s failed: %s", name, exc)

        merged = _merge_detections(all_dets, iou_threshold=iou_merge)
        diag_log.debug(
            "HA merge: %d raw → %d unique (iou_threshold=%.2f)",
            len(all_dets), len(merged), iou_merge,
        )
        return merged
