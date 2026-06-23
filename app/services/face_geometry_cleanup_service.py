"""Non-destructive cleanup of already-stored false faces.

The landmark-geometry gate (:mod:`app.detectors.landmark_geometry`) only stops
*new* false positives — cards, hands, feet, tree knots — from being recorded.
Faces detected **before** the gate existed are still sitting in the ``faces``
table.  Re-running detection would clean them, but the only existing full
re-scan also wipes manual names.

This service re-validates the **persisted** 5-point landmark geometry of stored
faces (no image reload needed — the landmarks are already in the database) and
identifies the geometrically-impossible ones.  It is deliberately conservative:

* Only faces that actually carry landmarks are examined.  Landmark-less
  detectors (Coral / Caffe SSD / Haar) fail open and are never swept up.
* ``detector_backend == "manual"`` faces (boxes the user drew by hand) are
  skipped entirely.
* A geometrically-impossible face is **droppable** only when nobody confirmed
  it: it is unassigned, or its ``person_id`` was set by an *automatic* source
  (clustering / recognition / auto-merge …).  Impossible faces that a human (or
  trusted legacy data) assigned are **flagged** for review, never auto-deleted.

The two-phase API (:meth:`scan` → :meth:`delete_faces`) lets the UI show a
preview and require confirmation before anything is removed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from app.config import DetectionConfig
from app.db.models import Face, FaceBlob
from app.detectors.base import Detection
from app.detectors.landmark_geometry import geometry_reject_reason

log = logging.getLogger(__name__)

# Assignment sources that mean a human (or trusted legacy data) made the call.
# Mirrors ``vector_scoring._TRUSTED_MANUAL_SOURCES``: these are never
# auto-deleted even when the geometry looks impossible — only reported.
TRUSTED_MANUAL_SOURCES = {None, "manual", "manual_merge", "suggestion_approved"}


@dataclass
class GeometryCleanupCandidate:
    """One stored face whose landmark geometry is not a plausible face."""

    face_id: int
    image_id: int
    person_id: Optional[int]
    assignment_source: Optional[str]
    detector_backend: str
    reason: str  # which geometry check failed (see geometry_reject_reason)
    bbox: tuple
    crop_path: Optional[str]


@dataclass
class GeometryCleanupReport:
    """Outcome of a :meth:`FaceGeometryCleanupService.scan`."""

    scanned: int = 0  # faces with landmarks that were geometry-checked
    droppable: List[GeometryCleanupCandidate] = field(default_factory=list)
    # Geometrically impossible but human/manually assigned — review, not delete.
    flagged: List[GeometryCleanupCandidate] = field(default_factory=list)

    @property
    def droppable_count(self) -> int:
        return len(self.droppable)

    @property
    def flagged_count(self) -> int:
        return len(self.flagged)

    @property
    def droppable_ids(self) -> List[int]:
        return [c.face_id for c in self.droppable]


class FaceGeometryCleanupService:
    """Re-validate stored faces against landmark geometry and drop impossibles."""

    def __init__(
        self,
        session: Session,
        detection_config: Optional[DetectionConfig] = None,
    ) -> None:
        self._session = session
        self._config = detection_config

    # ------------------------------------------------------------------
    # Scan (read-only)
    # ------------------------------------------------------------------

    def scan(self) -> GeometryCleanupReport:
        """Examine every stored face that carries landmarks; classify the
        geometrically-impossible ones into *droppable* and *flagged*.

        Read-only — nothing is deleted.  Returns the full report so a UI can
        preview counts and require confirmation before calling
        :meth:`delete_faces`.
        """
        report = GeometryCleanupReport()
        # Only faces with stored landmarks can be geometry-checked; the EXISTS
        # filter keeps landmark-less rows (which fail open) out of the scan.
        query = self._session.query(Face).filter(
            Face.blob.has(FaceBlob.landmarks.isnot(None))
        )
        for face in query:
            lms = face.get_landmarks()
            if lms is None:
                continue
            report.scanned += 1
            det = Detection(
                x=face.bbox_x,
                y=face.bbox_y,
                w=face.bbox_w,
                h=face.bbox_h,
                confidence=face.confidence,
                landmarks=lms.tolist(),
            )
            reason = geometry_reject_reason(det)
            if reason is None:
                continue  # plausible geometry — keep
            candidate = GeometryCleanupCandidate(
                face_id=face.id,
                image_id=face.image_id,
                person_id=face.person_id,
                assignment_source=face.assignment_source,
                detector_backend=face.detector_backend,
                reason=reason,
                bbox=(face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                crop_path=face.crop_path,
            )
            if self._is_droppable(face):
                report.droppable.append(candidate)
            else:
                report.flagged.append(candidate)

        log.info(
            "Geometry cleanup scan: %d face(s) with landmarks, "
            "%d droppable, %d flagged for review",
            report.scanned, report.droppable_count, report.flagged_count,
        )
        return report

    @staticmethod
    def _is_droppable(face: Face) -> bool:
        """True when a geometrically-impossible face can be auto-deleted.

        Conservative: hand-drawn boxes are never touched, and faces a human (or
        trusted legacy data) assigned are preserved — only unassigned or
        automatically-assigned impossibles are droppable.
        """
        if face.detector_backend == "manual":
            return False
        if face.person_id is None:
            return True  # unassigned auto-detection — nobody confirmed it
        return face.assignment_source not in TRUSTED_MANUAL_SOURCES

    # ------------------------------------------------------------------
    # Delete (destructive — call only after user confirmation)
    # ------------------------------------------------------------------

    def delete_faces(self, face_ids: Sequence[int]) -> int:
        """Hard-delete the given faces and their crop files.

        Returns the number of face rows removed.  The caller is responsible for
        passing only ids it intends to delete (typically
        ``report.droppable_ids`` after the user confirmed the preview).
        """
        ids = list(face_ids)
        if not ids:
            return 0
        faces = self._session.query(Face).filter(Face.id.in_(ids)).all()
        removed = 0
        crops_removed = 0
        for face in faces:
            if self._unlink_crop(face.crop_path):
                crops_removed += 1
            # Hard delete: cascades FaceBlob / FaceCorrection with the row.
            self._session.delete(face)
            removed += 1
        self._session.commit()
        log.info(
            "Geometry cleanup: hard-deleted %d false face(s), %d crop file(s)",
            removed, crops_removed,
        )
        return removed

    def scan_and_delete(self) -> tuple[GeometryCleanupReport, int]:
        """Convenience: scan, then delete every droppable face. Returns
        ``(report, deleted_count)``.  Use the two-phase API instead when a
        confirmation preview is wanted."""
        report = self.scan()
        deleted = self.delete_faces(report.droppable_ids)
        return report, deleted

    # ------------------------------------------------------------------

    @staticmethod
    def _unlink_crop(crop_path: Optional[str]) -> bool:
        """Best-effort delete of a crop thumbnail file; never raises."""
        if not crop_path:
            return False
        try:
            path = Path(crop_path)
            if path.exists():
                path.unlink()
                return True
        except OSError as exc:
            log.warning("Could not delete crop file %r: %s", crop_path, exc)
        return False
