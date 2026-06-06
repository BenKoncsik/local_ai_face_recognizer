"""Export recognised/assigned person data into image metadata.

This service writes the people detected in an image — their database id, name,
face bounding box, assignment provenance and optional notes — into the image's
own metadata (XMP, with EXIF / sidecar JSON fallbacks) so the information
travels with the file when it is copied, re-imported, or opened by other tools.

The payload follows the versioned ``facelocal.faces.v1`` schema (see
:data:`app.utils.image_metadata.FACELOCAL_SCHEMA`).

The service is intentionally UI-agnostic: it takes a SQLAlchemy session, builds
payloads, performs the writes through :mod:`app.utils.image_metadata`, and
returns structured results the UI can present.  It never writes automatically —
callers decide when (and after what confirmation) an export runs.

Privacy / safety notes
----------------------
* Person names and face positions are sensitive.  Callers can disable the name
  (``include_person_name=False``), the box (``include_face_box=False``) or the
  notes (``include_notes=False``), or force a sidecar-only write
  (``prefer_sidecar_only=True``) so the image file itself is never modified.
* Unknown faces never receive a fabricated name; they are emitted as
  ``{"unknown": true, ...}`` entries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person
from app.services.image_library_service import resolve_image_path
from app.utils import image_metadata as meta

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class FaceMetadataExportOptions:
    """Per-export configuration.

    The defaults are the safe, privacy-conscious baseline: everything useful is
    included, the single ``facelocal`` block is kept up to date, and embedding
    is attempted (with automatic sidecar fallback) rather than forced.
    """

    include_person_name: bool = True
    include_person_id: bool = True
    include_face_box: bool = True
    include_notes: bool = True
    # When True, never modify the image file — always write a sidecar JSON.
    prefer_sidecar_only: bool = False
    # When True (default), JPEGs get the JSON in the EXIF UserComment field (the
    # image's "comment", visible among the standard EXIF data); when False, an
    # XMP block is used instead. PNGs always use XMP (they have no EXIF).
    embed_in_exif_comment: bool = True
    # When True (default) and on macOS, also set the Finder/Spotlight comment so
    # the JSON shows up in Finder → Get Info → "Comments". No-op elsewhere.
    set_macos_finder_comment: bool = True
    # When True (default), a re-export refreshes the existing facelocal block.
    # When False and a block already exists, the image is left untouched.
    overwrite_existing_facelocal_metadata: bool = True


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass
class FaceMetadataExportResult:
    """Outcome for a single image."""

    image_id: Optional[int]
    image_path: Optional[str]
    success: bool
    write_mode: str  # one of app.utils.image_metadata.WRITE_MODE_*
    sidecar_path: Optional[str] = None
    error_message: Optional[str] = None
    exported_face_count: int = 0


@dataclass
class FaceMetadataExportSummary:
    """Aggregate outcome for a batch export — what the UI summarises."""

    results: list[FaceMetadataExportResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def embedded_count(self) -> int:
        return sum(1 for r in self.results if r.write_mode in _EMBED_MODES)

    @property
    def sidecar_count(self) -> int:
        return sum(1 for r in self.results if r.write_mode == meta.WRITE_MODE_SIDECAR)

    @property
    def skipped_count(self) -> int:
        return sum(1 for r in self.results if r.write_mode == meta.WRITE_MODE_SKIPPED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def errors(self) -> list[str]:
        out: list[str] = []
        for r in self.results:
            if not r.success and r.error_message:
                name = Path(r.image_path).name if r.image_path else f"id={r.image_id}"
                out.append(f"{name}: {r.error_message}")
        return out


_EMBED_MODES = {
    meta.WRITE_MODE_XMP,
    meta.WRITE_MODE_EXIF_USER_COMMENT,
    meta.WRITE_MODE_EXIF_IMAGE_DESCRIPTION,
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class FaceMetadataExportService:
    """Builds face metadata payloads and writes them into image files.

    Args:
        session: An open SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_image(
        self,
        image_id: int,
        options: Optional[FaceMetadataExportOptions] = None,
    ) -> FaceMetadataExportResult:
        """Export metadata for a single image id."""
        options = options or FaceMetadataExportOptions()
        image = self._session.get(Image, image_id)
        if image is None:
            return FaceMetadataExportResult(
                image_id=image_id,
                image_path=None,
                success=False,
                write_mode=meta.WRITE_MODE_FAILED,
                error_message="image not found in database",
            )
        return self._export_one(image, options)

    def export_images(
        self,
        image_ids: Iterable[int],
        options: Optional[FaceMetadataExportOptions] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> FaceMetadataExportSummary:
        """Export metadata for many image ids, returning an aggregate summary."""
        options = options or FaceMetadataExportOptions()
        ids = list(image_ids)
        total = len(ids)
        summary = FaceMetadataExportSummary()
        for idx, image_id in enumerate(ids):
            summary.results.append(self.export_image(image_id, options))
            if progress_cb:
                progress_cb(idx + 1, total)
        return summary

    def export_all(
        self,
        options: Optional[FaceMetadataExportOptions] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> FaceMetadataExportSummary:
        """Export metadata for every image in the database."""
        ids = [row[0] for row in self._session.query(Image.id).order_by(Image.id).all()]
        return self.export_images(ids, options, progress_cb)

    def build_payload(
        self,
        image: Image,
        options: Optional[FaceMetadataExportOptions] = None,
    ) -> dict:
        """Build the ``facelocal.faces.v1`` payload dict for *image*.

        Exposed separately from the write path so it can be unit-tested without
        touching the filesystem.
        """
        options = options or FaceMetadataExportOptions()

        faces = (
            self._session.query(Face)
            .filter(Face.image_id == image.id, Face.is_excluded.is_(False))
            .order_by(Face.id)
            .all()
        )

        face_entries = [self._face_entry(f, options) for f in faces]

        payload: dict = {
            "schema": meta.FACELOCAL_SCHEMA,
            "exported_at": _now_local_iso(),
            "image_id": image.id,
            # Boxes are stored/displayed in the EXIF-oriented (upright) pixel
            # space — the same space the detector ran in (see
            # image_utils.load_image_bgr_normalized). We record the orientation
            # so external tools can map the boxes back onto the raw pixels if
            # they read the file without applying EXIF orientation.
            "coordinate_space": "exif_oriented",
            "exif_orientation": self._orientation(image),
            "faces": face_entries,
        }
        if image.relative_path:
            payload["relative_path"] = image.relative_path
        return payload

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _export_one(
        self,
        image: Image,
        options: FaceMetadataExportOptions,
    ) -> FaceMetadataExportResult:
        abs_path = resolve_image_path(image)
        path_str = str(abs_path) if abs_path else (image.file_path or None)

        payload = self.build_payload(image, options)
        face_count = len(payload["faces"])

        if abs_path is None:
            return FaceMetadataExportResult(
                image_id=image.id,
                image_path=path_str,
                success=False,
                write_mode=meta.WRITE_MODE_FAILED,
                error_message="could not resolve image path",
                exported_face_count=face_count,
            )

        # Respect "do not overwrite": if a block already exists and the caller
        # asked not to overwrite, skip without touching the file.
        if not options.overwrite_existing_facelocal_metadata:
            try:
                if meta.read_face_metadata(abs_path) is not None:
                    return FaceMetadataExportResult(
                        image_id=image.id,
                        image_path=path_str,
                        success=True,
                        write_mode=meta.WRITE_MODE_SKIPPED,
                        exported_face_count=face_count,
                    )
            except Exception:  # noqa: BLE001
                log.debug("Existing-metadata check failed for %s", abs_path, exc_info=True)

        try:
            write_result = meta.write_face_metadata(
                abs_path,
                payload,
                prefer_sidecar_only=options.prefer_sidecar_only,
                prefer_exif_comment=options.embed_in_exif_comment,
                set_finder_comment=options.set_macos_finder_comment,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Face metadata export failed for image id=%s", image.id)
            return FaceMetadataExportResult(
                image_id=image.id,
                image_path=path_str,
                success=False,
                write_mode=meta.WRITE_MODE_FAILED,
                error_message=str(exc),
                exported_face_count=face_count,
            )

        return FaceMetadataExportResult(
            image_id=image.id,
            image_path=path_str,
            success=write_result.ok,
            write_mode=write_result.write_mode,
            sidecar_path=str(write_result.sidecar_path) if write_result.sidecar_path else None,
            error_message=write_result.error_message,
            exported_face_count=face_count,
        )

    def _orientation(self, image: Image) -> int:
        """EXIF orientation tag of the source file (1 when absent/unreadable)."""
        try:
            from app.utils.image_utils import get_exif_orientation

            abs_path = resolve_image_path(image)
            if abs_path is not None and Path(abs_path).exists():
                return int(get_exif_orientation(str(abs_path)))
        except Exception:  # noqa: BLE001
            log.debug("Orientation read failed for image id=%s", image.id, exc_info=True)
        return 1

    def _face_entry(self, face: Face, options: FaceMetadataExportOptions) -> dict:
        person = self._session.get(Person, face.person_id) if face.person_id else None
        is_unknown = person is None or bool(person.is_protected)

        entry: dict = {}

        if is_unknown:
            entry["unknown"] = True
        else:
            if options.include_person_id:
                entry["person_id"] = person.id
            if options.include_person_name:
                entry["person_name"] = person.name

        if options.include_face_box:
            entry["box"] = {
                "x": int(face.bbox_x),
                "y": int(face.bbox_y),
                "width": int(face.bbox_w),
                "height": int(face.bbox_h),
            }

        source = face.assignment_source or "unknown"
        entry["assignment_source"] = _normalise_source(source, is_unknown)

        conf = face.assignment_confidence
        if conf is None:
            conf = face.confidence
        if conf is not None:
            entry["confidence"] = round(float(conf), 4)

        if face.assigned_at is not None:
            entry["assigned_at"] = _stored_to_local_iso(face.assigned_at)

        if options.include_notes and not is_unknown and person is not None:
            note = (person.notes or "").strip()
            if note:
                entry["notes"] = note

        return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical assignment-source vocabulary surfaced in the payload.
_SOURCE_MAP = {
    "manual": "manual",
    "manual_merge": "manual",
    "recognition": "auto",
    "clustering": "auto",
    "auto": "auto",
    "suggestion": "suggestion",
}


def _normalise_source(source: str, is_unknown: bool) -> str:
    if is_unknown:
        return "unknown"
    return _SOURCE_MAP.get(source, "unknown")


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _stored_to_local_iso(dt: datetime) -> str:
    """Render a DB datetime (stored as naive UTC) as a local ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().isoformat(timespec="seconds")
