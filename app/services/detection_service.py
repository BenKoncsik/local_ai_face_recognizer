"""Face detection service.

Processes a list of image IDs, runs the face detector, saves face records
and crop thumbnails to disk and database.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

from sqlalchemy.orm import Session

from app.config import AppConfig
from app.db.models import Face, Image
from app.detectors.base import Detection, FaceDetector
from app.detectors.cpu_detector import CpuDetector
from app.utils.image_utils import load_image_bgr, save_face_crop

log = logging.getLogger(__name__)


class DetectionService:
    """Runs face detection on images that haven't been processed yet.

    Args:
        session:     SQLAlchemy session.
        detector:    A :class:`~app.detectors.base.FaceDetector` instance.
        config:      Full application configuration.
        progress_cb: Optional ``(current, total, path)`` progress callback.
    """

    def __init__(
        self,
        session: Session,
        detector: FaceDetector,
        config: AppConfig,
        progress_cb: Optional[Callable[[int, Optional[int], str], None]] = None,
    ) -> None:
        self._session = session
        self._detector = detector
        self._config = config
        self._crops_dir = config.crops_dir_resolved
        self._crops_dir.mkdir(parents=True, exist_ok=True)
        self._progress_cb = progress_cb or (lambda *_: None)

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

            self._progress_cb(idx, total, image.file_path)

            try:
                n = self._process_image(image)
                total_faces += n
            except Exception as exc:  # noqa: BLE001
                log.error("Detection failed for %s: %s", image.file_path, exc)
            finally:
                image.detection_done = True
                self._session.commit()

        log.info("Detection complete: %d face(s) across %d image(s)", total_faces, total)
        return total_faces

    def _process_image(self, image: Image) -> int:
        """Detect and persist faces in a single image.

        Returns the number of detected faces.
        """
        path = Path(image.file_path)
        if not path.exists():
            log.warning("Image file missing on disk: %s", path)
            return 0

        img_bgr = load_image_bgr(str(path))
        if img_bgr is None:
            log.warning("OpenCV could not read: %s", path)
            return 0

        h, w = img_bgr.shape[:2]
        image.width = w
        image.height = h

        try:
            detections: List[Detection] = self._detector.detect(
                img_bgr,
                confidence_threshold=self._config.detection.confidence_threshold,
                min_face_size=self._config.detection.min_face_size,
            )
        except RuntimeError as exc:
            if "EdgeTPU" in str(exc) or "Falling back to CPU" in str(exc):
                log.warning(
                    "Coral TPU inference failed (%s) — switching to CPU detector for remaining images.",
                    exc,
                )
                self._detector = CpuDetector(
                    model_path=self._config.detection.cpu_model_path
                )
                detections = self._detector.detect(
                    img_bgr,
                    confidence_threshold=self._config.detection.confidence_threshold,
                    min_face_size=self._config.detection.min_face_size,
                )
            else:
                raise

        # Remove stale face records for this image before adding new ones
        self._session.query(Face).filter(Face.image_id == image.id).delete()

        for face_index, det in enumerate(detections):
            crop_path = save_face_crop(
                img_bgr=img_bgr,
                detection=det,
                crops_dir=self._crops_dir,
                image_id=image.id,
                thumbnail_size=self._config.scan.thumbnail_size,
                face_index=face_index,
            )

            face = Face(
                image_id=image.id,
                bbox_x=det.x,
                bbox_y=det.y,
                bbox_w=det.w,
                bbox_h=det.h,
                confidence=det.confidence,
                detector_backend=self._detector.backend_name,
                crop_path=str(crop_path) if crop_path else None,
            )
            self._session.add(face)

        log.debug("Image %s: %d face(s) detected", path.name, len(detections))
        return len(detections)
