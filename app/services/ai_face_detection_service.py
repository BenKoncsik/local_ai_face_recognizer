"""AI (deep learning) face-detection service — analysis only.

Part of the AI pipeline, but deliberately separate from the classic
detection/recognition path: it answers *whether* there are faces on an image,
*how many*, *where* (bounding boxes) and with what confidence, using a
pretrained deep-learning detector (YuNet ONNX).  It never creates ``Face``
rows, never assigns identities and never modifies the classic results — its
findings live in the dedicated ``ai_face_detections`` table, keyed by image,
detector, run id and timestamp so they can be queried later.

Failure policy (the AI model may legitimately be missing): the service never
raises out of :meth:`AiFaceDetectionService.detect_images` for a missing or
broken model — it logs the error and reports it on the returned stats, so the
callers (deep pipeline stage, re-recognition flow) keep working unaffected.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import AiFaceDetectionConfig
from app.db.models import AiFaceDetection, Image
from app.detectors.base import FaceDetector
from app.services.image_library_service import resolve_image_path
from app.utils.image_utils import load_image_bgr_normalized

log = logging.getLogger(__name__)

SOURCE_MANUAL = "manual"
SOURCE_PIPELINE = "pipeline"
SOURCE_RERECOGNITION = "rerecognition"


@dataclass
class AiFaceDetectionStats:
    """Outcome of one AI face-detection run."""

    run_id: str = ""
    detector_name: str = ""
    images_processed: int = 0
    images_failed: int = 0
    faces_found: int = 0
    # Set when the detector could not be created (missing model etc.); the
    # run is then a no-op and the caller should treat it as "AI unavailable".
    error: Optional[str] = None

    @property
    def available(self) -> bool:
        return self.error is None


class AiFaceDetectionService:
    """Runs the pretrained deep-learning face detector and stores its findings.

    Args:
        session:  SQLAlchemy session.
        config:   :class:`~app.config.AiFaceDetectionConfig`.
        detector: Optional pre-built detector (tests / reuse).  ``None`` →
                  a YuNet detector is created lazily on the first run; a
                  creation failure is reported on the stats, never raised.
    """

    def __init__(
        self,
        session: Session,
        config: AiFaceDetectionConfig,
        detector: Optional[FaceDetector] = None,
    ) -> None:
        self._session = session
        self._config = config
        self._detector = detector
        self._detector_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Detector handling
    # ------------------------------------------------------------------

    def _get_detector(self) -> Optional[FaceDetector]:
        """Return the deep-learning detector, or ``None`` if unavailable."""
        if self._detector is not None:
            return self._detector
        if self._detector_error is not None:
            return None
        try:
            from app.detectors.yunet_detector import YuNetDetector

            self._detector = YuNetDetector(model_path=self._config.model_path)
        except Exception as exc:  # noqa: BLE001 — missing model, broken cv2…
            self._detector_error = str(exc)
            log.error("AI face detector unavailable: %s", exc)
            return None
        return self._detector

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_images(
        self,
        image_ids: Sequence[int],
        source: str = SOURCE_MANUAL,
        progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> AiFaceDetectionStats:
        """Run AI face detection on the given images and persist the results.

        Per image the previous ``ai_face_detections`` rows are replaced, so a
        later query always sees the latest run.  A single failing image is
        logged and counted, never aborts the batch.

        Args:
            image_ids:    Primary keys from the ``images`` table.
            source:       What triggered the run (stored on each row).
            progress_cb:  Optional ``(current, total, path)`` callback.
            cancel_check: Optional callable; return ``True`` to stop early.

        Returns:
            :class:`AiFaceDetectionStats` — with ``error`` set (and nothing
            written) when the AI model is unavailable.
        """
        stats = AiFaceDetectionStats(run_id=uuid.uuid4().hex)
        if not self._config.enabled:
            stats.error = "AI face detection is disabled in the configuration"
            return stats

        detector = self._get_detector()
        if detector is None:
            stats.error = self._detector_error or "AI face detector unavailable"
            return stats
        stats.detector_name = detector.backend_name

        total = len(image_ids)
        cb = progress_cb or (lambda *_: None)

        for idx, image_id in enumerate(image_ids, start=1):
            if cancel_check is not None and cancel_check():
                log.info("AI face detection cancelled after %d image(s)", idx - 1)
                break

            image: Optional[Image] = self._session.get(Image, image_id)
            if image is None:
                log.warning("AI detect: image id=%d not found — skipping", image_id)
                stats.images_failed += 1
                continue

            display_path = image.file_path or f"image id={image.id}"
            cb(idx, total, display_path)

            try:
                stats.faces_found += self._process_image(
                    image, detector, stats.run_id, source
                )
                stats.images_processed += 1
            except Exception as exc:  # noqa: BLE001
                log.error("AI face detection failed for %s: %s", display_path, exc)
                stats.images_failed += 1
            finally:
                self._session.commit()

        log.info(
            "AI face detection (%s): %d face(s) on %d image(s), %d failed "
            "[detector=%s run=%s]",
            source, stats.faces_found, stats.images_processed,
            stats.images_failed, stats.detector_name, stats.run_id,
        )
        return stats

    def _process_image(
        self,
        image: Image,
        detector: FaceDetector,
        run_id: str,
        source: str,
    ) -> int:
        """Detect faces on one image and replace its stored AI results."""
        path = resolve_image_path(image)
        if path is None or not path.exists():
            raise FileNotFoundError(f"image file not found: {path or image.file_path}")

        # EXIF-corrected load, same as the classic path, so the boxes match
        # what the preview shows.
        img_bgr = load_image_bgr_normalized(str(path))
        if img_bgr is None:
            raise ValueError(f"OpenCV could not read: {path}")

        detections = detector.detect(
            img_bgr,
            confidence_threshold=self._config.confidence_threshold,
            min_face_size=self._config.min_face_size,
        )

        self._session.query(AiFaceDetection).filter(
            AiFaceDetection.image_id == image.id
        ).delete(synchronize_session="fetch")

        for det in detections:
            self._session.add(
                AiFaceDetection(
                    image_id=image.id,
                    image_path=str(path),
                    bbox_x=det.x,
                    bbox_y=det.y,
                    bbox_w=det.w,
                    bbox_h=det.h,
                    confidence=float(det.confidence),
                    detector_name=detector.backend_name,
                    run_id=run_id,
                    source=source,
                )
            )

        log.debug("AI detect %s: %d face(s)", path.name, len(detections))
        return len(detections)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all_image_ids(self) -> List[int]:
        """Every image id in the database (the standalone full-sweep scope)."""
        return [r[0] for r in self._session.query(Image.id).order_by(Image.id).all()]

    def list_for_image(self, image_id: int) -> List[AiFaceDetection]:
        """Stored AI detections of one image (latest run), strongest first."""
        return (
            self._session.query(AiFaceDetection)
            .filter(AiFaceDetection.image_id == image_id)
            .order_by(AiFaceDetection.confidence.desc())
            .all()
        )
