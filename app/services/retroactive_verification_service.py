"""Retroactive crop-level verification of already-stored face detections.

New detections go through FaceVerifier at detection time, but faces stored
*before* the verifier was added still sit in the database.  This service
re-examines every such face — loading the original image and running the same
crop-level re-detection used during normal scanning — and identifies those that
cannot be confirmed as real faces.

Conservative rules (mirrors FaceGeometryCleanupService):
* ``detector_backend == "manual"`` faces (hand-drawn boxes) are never touched.
* Faces the user explicitly assigned (manual / manual_merge / suggestion_approved)
  are *flagged* for review but never auto-deleted.
* All other non-manual faces that fail verification are *droppable*.

Two-phase API: :meth:`scan` (read-only) → :meth:`delete_faces` (destructive,
call only after user confirmation).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import AppConfig
from app.db.models import Face, Image
from app.db.query_utils import in_chunks
from app.detectors.base import Detection
from app.services.image_library_service import resolve_image_path
from app.utils.image_utils import load_image_bgr_normalized

log = logging.getLogger(__name__)

# Sources set only when a human confirmed the face.
TRUSTED_MANUAL_SOURCES = frozenset({"manual", "manual_merge", "suggestion_approved"})


@dataclass
class VerificationCandidate:
    """One stored face that failed crop-level re-detection."""

    face_id: int
    image_id: int
    image_path: str
    person_id: Optional[int]
    assignment_source: Optional[str]
    detector_backend: str
    bbox: tuple  # (x, y, w, h)
    confidence: float
    crop_path: Optional[str]


@dataclass
class RetroactiveVerificationReport:
    """Outcome of a :meth:`RetroactiveVerificationService.scan`."""

    scanned: int = 0          # non-manual faces actually verified
    images_loaded: int = 0    # original images loaded from disk
    images_missing: int = 0   # images skipped (file not found)
    droppable: List[VerificationCandidate] = field(default_factory=list)
    flagged: List[VerificationCandidate] = field(default_factory=list)

    @property
    def droppable_count(self) -> int:
        return len(self.droppable)

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def droppable_ids(self) -> List[int]:
        return [c.face_id for c in self.droppable]


class RetroactiveVerificationService:
    """Re-verifies stored face detections using crop-level re-detection.

    Args:
        session:     SQLAlchemy session.
        config:      Full application config (provides verification settings).
        progress_cb: Optional ``(current, total, path)`` callable.
    """

    def __init__(
        self,
        session: Session,
        config: AppConfig,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self._session = session
        self._config = config
        self._progress_cb = progress_cb or (lambda *_: None)
        self._verifier = self._build_verifier()

    # ------------------------------------------------------------------
    # Scan (read-only)
    # ------------------------------------------------------------------

    def scan(self) -> RetroactiveVerificationReport:
        """Verify every non-manual stored face against the original image.

        Loads each original image once (groups faces by image_id) and runs
        the crop-level FaceVerifier on each stored bbox.  Returns a report
        that separates droppable faces (no human confirmation) from flagged
        ones (manually assigned but still failing).

        Read-only — nothing is deleted.
        """
        if self._verifier is None:
            log.warning(
                "RetroactiveVerificationService: verifier unavailable — "
                "verification gate is disabled, scan aborted."
            )
            return RetroactiveVerificationReport()

        report = RetroactiveVerificationReport()

        # Only re-verify *uncertain* faces.  Detections at/above the exemption
        # confidence are trusted as-is by the live detection gate
        # (DetectionService._filter_verified), so re-checking them here would
        # make the retroactive sweep stricter than detection itself and could
        # delete genuine high-confidence faces a lone YuNet hit cannot re-confirm
        # in a margin-padded crop.  Skipping them also avoids loading images that
        # have nothing to check — a large speed-up on big libraries.
        #
        # With "verify all" the exemption is disabled so EVERY stored non-manual
        # face is re-checked (catches high-confidence false positives too).
        query = self._session.query(Face).filter(Face.detector_backend != "manual")
        if self._config.detection.verification_verify_all:
            exempt = 2.0
        else:
            exempt = self._config.detection.verification_confidence_exempt
            query = query.filter(Face.confidence < exempt)

        # Load all matching non-manual faces grouped by image.
        image_face_map: Dict[int, List[Face]] = defaultdict(list)
        all_non_manual = query.all()
        for face in all_non_manual:
            image_face_map[face.image_id].append(face)

        total_images = len(image_face_map)
        log.info(
            "Retroactive verification: %d uncertain non-manual face(s) "
            "(confidence < %.2f) across %d image(s)",
            len(all_non_manual), exempt, total_images,
        )

        for img_idx, (image_id, faces) in enumerate(image_face_map.items(), start=1):
            image: Optional[Image] = self._session.get(Image, image_id)
            if image is None:
                log.warning("Image id=%d not found in DB — skipping %d face(s)", image_id, len(faces))
                report.images_missing += 1
                continue

            path = resolve_image_path(image)
            display = str(path) if path else f"image id={image_id}"
            self._progress_cb(img_idx, total_images, display)

            if path is None or not path.exists():
                log.warning("Image file missing: %s — skipping %d face(s)", display, len(faces))
                report.images_missing += 1
                continue

            img_bgr = load_image_bgr_normalized(str(path))
            if img_bgr is None:
                log.warning("Could not load image: %s", display)
                report.images_missing += 1
                continue

            report.images_loaded += 1

            for face in faces:
                if face.bbox_w <= 0 or face.bbox_h <= 0:
                    continue

                det = Detection(
                    x=face.bbox_x,
                    y=face.bbox_y,
                    w=face.bbox_w,
                    h=face.bbox_h,
                    confidence=face.confidence,
                )

                try:
                    result = self._verifier.verify(img_bgr, det)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Verifier error for face id=%d in %s: %s — keeping face",
                        face.id, display, exc,
                    )
                    continue

                report.scanned += 1

                if result is not None:
                    # Face re-confirmed — keep it.
                    continue

                # Verification failed: decide droppable vs flagged.
                candidate = VerificationCandidate(
                    face_id=face.id,
                    image_id=face.image_id,
                    image_path=display,
                    person_id=face.person_id,
                    assignment_source=face.assignment_source,
                    detector_backend=face.detector_backend,
                    bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    confidence=face.confidence,
                    crop_path=face.crop_path,
                )
                if self._is_droppable(face):
                    report.droppable.append(candidate)
                else:
                    report.flagged.append(candidate)

        log.info(
            "Retroactive verification scan complete: %d face(s) verified across "
            "%d image(s) — %d droppable, %d flagged, %d image(s) missing",
            report.scanned, report.images_loaded,
            report.droppable_count, report.flagged_count, report.images_missing,
        )
        return report

    # ------------------------------------------------------------------
    # Delete (destructive — call only after user confirmation)
    # ------------------------------------------------------------------

    def delete_faces(self, face_ids: Sequence[int]) -> int:
        """Hard-delete the given faces and their crop files.

        Returns the number of face rows removed.  Pass only ids from
        ``report.droppable_ids`` after the user has confirmed the preview.
        """
        ids = list(face_ids)
        if not ids:
            return 0
        faces = []
        for chunk in in_chunks(ids):
            faces.extend(self._session.query(Face).filter(Face.id.in_(chunk)).all())
        removed = 0
        crops_removed = 0
        for face in faces:
            if self._unlink_crop(face.crop_path):
                crops_removed += 1
            self._session.delete(face)
            removed += 1
        self._session.commit()
        log.info(
            "Retroactive verification: hard-deleted %d non-face(s), %d crop file(s) removed",
            removed, crops_removed,
        )
        return removed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_verifier(self):
        """Build the crop-level verification gate; returns None when unavailable.

        Uses the multi-technology :class:`MultiStageFaceValidator` ensemble when
        ``detection.multistage_enabled`` (so the retroactive sweep applies the
        same several-technology vote as live detection), otherwise the legacy
        single :class:`FaceVerifier`.  Both share the ``verify``/``available``
        surface, so :meth:`scan` is unchanged.
        """
        if not self._config.detection.verification_enabled:
            log.warning(
                "RetroactiveVerificationService: verification_enabled=False in config — "
                "the service will not run."
            )
            return None
        try:
            if self._config.detection.multistage_enabled:
                from app.services.multi_stage_face_validator import (
                    MultiStageFaceValidator,
                )

                verifier = MultiStageFaceValidator(self._config.detection)
                log.info(
                    "RetroactiveVerificationService: multi-stage gate, technologies=%s",
                    verifier.active_technologies(),
                )
            else:
                from app.services.face_verification_service import FaceVerifier

                verifier = FaceVerifier(self._config.detection)
            if not verifier.available:
                log.warning(
                    "RetroactiveVerificationService: verifier not available "
                    "(model missing?) — scan aborted."
                )
                return None
            return verifier
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not build verification gate: %s", exc)
            return None

    @staticmethod
    def _is_droppable(face: Face) -> bool:
        """True when a verification-failing face may be auto-deleted.

        Manually drawn boxes and faces a human explicitly assigned are
        never auto-deleted — only flagged for review.
        """
        if face.detector_backend == "manual":
            return False
        if face.person_id is None:
            return True  # unassigned — no one confirmed it
        return face.assignment_source not in TRUSTED_MANUAL_SOURCES

    @staticmethod
    def _unlink_crop(crop_path: Optional[str]) -> bool:
        if not crop_path:
            return False
        try:
            p = Path(crop_path)
            if p.exists():
                p.unlink()
                return True
        except OSError as exc:
            log.warning("Could not delete crop file %r: %s", crop_path, exc)
        return False
