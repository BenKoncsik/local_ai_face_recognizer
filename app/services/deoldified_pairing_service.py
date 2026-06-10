"""Deoldified image pairing — filename parsing and DB lookup.

Naming convention:
    Original:   some_name.jpg
    Colorized:  some_name-deoldified (artistic).jpg
                some_name-deoldified (stable).jpg
                some_name-deoldified.jpg

The '-deoldified' token (case-insensitive) marks the separation point.
Everything from '-deoldified' onwards is stripped; the original extension
is re-appended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.db.models import Image

_RE_DEOLDIFIED = re.compile(r"-deoldified.*$", re.IGNORECASE)
# Captures the text *after* the '-deoldified' token (e.g. " (artistic)"),
# used to label one colorized variant apart from its siblings.
_RE_VARIANT_LABEL = re.compile(r"-deoldified\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class ComparisonMember:
    """One image in a comparison group (the B&W original or a colorized variant).

    ``label`` is the human-facing variant tag — empty for the B&W original,
    ``"(artistic)"`` / ``"(stable)"`` / ``"deoldified"`` for colorized images.
    ``file_path`` is the on-disk path (already resolved when possible).
    """

    image_id: int
    file_path: str
    label: str
    is_bw: bool


def extract_original_stem(stem: str) -> Optional[str]:
    """Strip the '-deoldified' suffix and everything after it from a stem.

    Returns the original stem, or None if '-deoldified' is not present.

    >>> extract_original_stem("photo-deoldified (artistic)")
    'photo'
    >>> extract_original_stem("photo-deoldified")
    'photo'
    >>> extract_original_stem("normal_photo") is None
    True
    """
    if not _RE_DEOLDIFIED.search(stem):
        return None
    result = _RE_DEOLDIFIED.sub("", stem)
    return result or None


def extract_original_filename(filename: str) -> Optional[str]:
    """Return the expected original filename for a deoldified image, or None.

    Examples:
        "photo-deoldified (artistic).JPG" → "photo.JPG"
        "photo-deoldified (stable).jpg"   → "photo.jpg"
        "photo-deoldified.jpg"            → "photo.jpg"
        "normal_photo.jpg"                → None
    """
    p = Path(filename)
    original_stem = extract_original_stem(p.stem)
    if original_stem is None:
        return None
    return original_stem + p.suffix


def extract_variant_label(stem: str) -> str:
    """Return the colorized-variant label from a stem, or '' if not deoldified.

    The label is the text following the '-deoldified' token, with surrounding
    whitespace trimmed.  A plain '-deoldified' suffix yields ``"deoldified"`` so
    every colorized member has a non-empty, distinguishable label.

    >>> extract_variant_label("photo-deoldified (artistic)")
    '(artistic)'
    >>> extract_variant_label("photo-deoldified (stable)")
    '(stable)'
    >>> extract_variant_label("photo-deoldified")
    'deoldified'
    >>> extract_variant_label("normal_photo")
    ''
    """
    m = _RE_VARIANT_LABEL.search(stem)
    if m is None:
        return ""
    return m.group(1).strip() or "deoldified"


def is_deoldified_path(path: str) -> bool:
    """Return True if the filename contains '-deoldified' (case-insensitive)."""
    return extract_original_stem(Path(path).stem) is not None


def _basename(path: str) -> str:
    """Return the final path component, handling both '/' and '\\' separators."""
    # PurePosix/Windows both store backslashes literally, so normalise first.
    return Path(path.replace("\\", "/")).name


def _like_escape(value: str) -> str:
    """Escape SQL-LIKE wildcards ('%', '_') and the escape char itself."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class DeoldifiedPairingService:
    """Finds paired images in the database for the deoldified feature.

    Pairing is purely filename-based and ignores the containing folder: a
    colorized image and its black-and-white original are considered the same
    photo whenever their filenames match (after stripping '-deoldified ...'),
    even if they live in different directories.
    """

    def __init__(self, session: "Session") -> None:
        self._session = session

    def find_original_for_deoldified(
        self, deoldified_image: "Image"
    ) -> Optional["Image"]:
        """Return the original Image record for a deoldified image, or None.

        Matches purely by filename across all folders. Tries the exact
        extension first, then lowercase/uppercase variants.
        """
        resolved_name = _basename(deoldified_image.file_path)
        if deoldified_image.relative_path:
            resolved_name = _basename(deoldified_image.relative_path)
        # Prefer the on-disk name when resolvable (portability), else file_path.
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(deoldified_image)
        name = _basename(str(resolved)) if resolved else resolved_name

        original_name = extract_original_filename(name)
        if original_name is None:
            return None

        ext = Path(original_name).suffix
        stem_orig = Path(original_name).stem

        # Build candidate names: exact extension + case variants
        candidates: list[str] = [original_name]
        for alt_ext in (ext.lower(), ext.upper()):
            alt = stem_orig + alt_ext
            if alt not in candidates:
                candidates.append(alt)

        return self._find_by_basenames(candidates, exclude_id=deoldified_image.id)

    def find_deoldified_for_original(
        self, original_image: "Image"
    ) -> Optional["Image"]:
        """Return the first deoldified Image for an original image, or None.

        Convenience wrapper over :meth:`find_all_deoldified_for_original` that
        keeps the original single-pair callers working.
        """
        variants = self.find_all_deoldified_for_original(original_image)
        return variants[0] if variants else None

    def find_all_deoldified_for_original(
        self, original_image: "Image"
    ) -> List["Image"]:
        """Return every deoldified Image for an original, ordered by label.

        Matches any image — in any folder — whose filename is
        '{original_stem}-deoldified...' (case-insensitive).  A single B&W
        original can have several colorized siblings ('(artistic)', '(stable)',
        plain); all are returned, sorted by their variant label for stability.
        """
        import sqlalchemy as sa

        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(original_image)
        name = (
            _basename(str(resolved))
            if resolved
            else _basename(
                original_image.relative_path or original_image.file_path
            )
        )
        stem = Path(name).stem

        needle = f"{stem}-deoldified".lower()
        safe = _like_escape(needle)

        # Pre-filter in SQL (substring), then verify the basename in Python so a
        # match like 'other/{stem}-deoldified.jpg' counts but 'x{stem}...' does not.
        rows = (
            self._session.query(Image)
            .filter(
                sa.or_(
                    sa.func.lower(Image.file_path).like(
                        f"%{safe}%", escape="\\"
                    ),
                    sa.func.lower(Image.relative_path).like(
                        f"%{safe}%", escape="\\"
                    ),
                )
            )
            .all()
        )
        matches: List["Image"] = []
        for row in rows:
            if row.id == original_image.id:
                continue
            row_stem = Path(_basename(row.file_path)).stem
            orig_stem = extract_original_stem(row_stem)
            if orig_stem is not None and orig_stem.lower() == stem.lower():
                matches.append(row)
        matches.sort(
            key=lambda r: extract_variant_label(Path(_basename(r.file_path)).stem)
        )
        return matches

    def get_comparison_group(self, image: "Image") -> List[ComparisonMember]:
        """Return the full comparison group for an image: B&W original + variants.

        Resolves the B&W original (``image`` itself when it is the original, or
        its parent when ``image`` is colorized), then lists every colorized
        sibling.  The B&W original is always first (``is_bw=True``), followed by
        the colorized variants in label order.  Returns an empty list when no
        group exists (no original found, or an original with no colorized
        variants), so callers can fall back to single-image behaviour.
        """
        from app.services.image_library_service import resolve_image_path

        if is_deoldified_path(image.file_path):
            original = self.find_original_for_deoldified(image)
            if original is None:
                return []
        else:
            original = image

        variants = self.find_all_deoldified_for_original(original)
        if not variants:
            return []

        def _member(img: "Image", *, is_bw: bool) -> ComparisonMember:
            resolved = resolve_image_path(img)
            path = str(resolved) if resolved else img.file_path
            label = (
                "" if is_bw
                else extract_variant_label(Path(_basename(img.file_path)).stem)
            )
            return ComparisonMember(
                image_id=img.id, file_path=path, label=label, is_bw=is_bw
            )

        members = [_member(original, is_bw=True)]
        members.extend(_member(v, is_bw=False) for v in variants)
        return members

    def _find_by_basenames(
        self, candidates: list[str], *, exclude_id: Optional[int] = None
    ) -> Optional["Image"]:
        """Return the first Image whose basename matches any candidate, or None.

        Searches across all folders; matching is case-insensitive on the
        filename only. A SQL LIKE pre-filter narrows the scan, then the exact
        basename is verified in Python to avoid false suffix matches.
        """
        import sqlalchemy as sa

        from app.db.models import Image

        wanted = {c.lower() for c in candidates}
        clauses = []
        for c in candidates:
            safe = _like_escape(c.lower())
            clauses.append(
                sa.func.lower(Image.file_path).like(f"%{safe}", escape="\\")
            )
            clauses.append(
                sa.func.lower(Image.relative_path).like(f"%{safe}", escape="\\")
            )

        rows = self._session.query(Image).filter(sa.or_(*clauses)).all()
        for row in rows:
            if exclude_id is not None and row.id == exclude_id:
                continue
            if _basename(row.file_path).lower() in wanted:
                return row
        return None

    # ──────────────────────────────────────────────────────────────────
    # One-directional data sync between a pair
    # ──────────────────────────────────────────────────────────────────

    # Image-level metadata fields copied between paired images.  Person-group
    # memberships are intentionally absent: they live on Person, so copying a
    # Face's person_id carries them automatically.
    _META_FIELDS = (
        "photo_date",
        "note",
        "place_id",
        "image_latitude",
        "image_longitude",
        "exif_latitude",
        "exif_longitude",
    )

    @staticmethod
    def image_has_data(image: "Image") -> bool:
        """Return True if the image has any face or user-supplied metadata."""
        if image.faces:
            return True
        for field in DeoldifiedPairingService._META_FIELDS:
            value = getattr(image, field, None)
            if value not in (None, ""):
                return True
        return False

    def sync_pair_data(
        self,
        image_a: "Image",
        image_b: "Image",
        *,
        crops_dir: "Optional[Path]" = None,
        thumbnail_size: "Optional[tuple[int, int]]" = None,
        crop_mode: str = "legacy",
    ) -> Optional[dict]:
        """Copy annotations one-directionally between a deoldified pair.

        Copies **only when exactly one side is empty**: data flows from the
        side that has data into the side that has none.  When both sides are
        empty (nothing to copy) or both already hold data (avoid clobbering),
        nothing happens and None is returned.

        Faces are replicated with their person assignment, embedding and
        landmarks; crops are regenerated from the *target* image's own pixels
        when ``crops_dir``/``thumbnail_size`` are supplied.  Image-level
        metadata (date, note, place, GPS) is copied for empty target fields.

        Returns a summary dict, or None when no copy was performed.
        """
        a_has = self.image_has_data(image_a)
        b_has = self.image_has_data(image_b)
        if a_has == b_has:
            return None

        source, target = (image_a, image_b) if a_has else (image_b, image_a)

        faces_copied = self._copy_faces(
            source, target, crops_dir, thumbnail_size, crop_mode
        )
        meta_copied = self._copy_metadata(source, target)
        self._session.flush()
        return {
            "source_id": source.id,
            "target_id": target.id,
            "faces_copied": faces_copied,
            "metadata_fields": meta_copied,
        }

    def _copy_faces(
        self,
        source: "Image",
        target: "Image",
        crops_dir: "Optional[Path]",
        thumbnail_size: "Optional[tuple[int, int]]",
        crop_mode: str,
    ) -> int:
        """Replicate source faces onto target; return the number copied."""
        from app.db.models import Face

        if not source.faces:
            return 0

        new_faces: list[Face] = []
        for f in source.faces:
            nf = Face(
                image_id=target.id,
                person_id=f.person_id,
                bbox_x=f.bbox_x,
                bbox_y=f.bbox_y,
                bbox_w=f.bbox_w,
                bbox_h=f.bbox_h,
                confidence=f.confidence,
                detector_backend=f.detector_backend,
                is_excluded=f.is_excluded,
                assignment_source=f.assignment_source,
                assignment_confidence=f.assignment_confidence,
                assigned_at=f.assigned_at,
                quality_score=f.quality_score,
                quality_reasons=f.quality_reasons,
                is_low_quality=f.is_low_quality,
            )
            emb = f.get_embedding()
            if emb is not None:
                nf.set_embedding(emb)
            lm = f.get_landmarks()
            if lm is not None:
                nf.set_landmarks(lm)
            self._session.add(nf)
            new_faces.append(nf)

        self._session.flush()  # assign Face.id values

        # Regenerate crops from the target image's own (colorized/B&W) pixels.
        # The pair is pixel-aligned, so the source bbox is valid on the target.
        if crops_dir is not None and thumbnail_size is not None:
            from app.services.face_crop_service import save_crop_for_face
            from app.services.image_library_service import resolve_image_path
            from app.utils.image_utils import load_image_bgr

            resolved = resolve_image_path(target)
            img_bgr = load_image_bgr(
                str(resolved) if resolved else target.file_path
            )
            if img_bgr is not None:
                for nf in new_faces:
                    save_crop_for_face(
                        nf,
                        crops_dir=crops_dir,
                        thumbnail_size=thumbnail_size,
                        img_bgr=img_bgr,
                        crop_mode=crop_mode,
                    )

        target.detection_done = source.detection_done
        target.embedding_done = source.embedding_done
        return len(new_faces)

    def _copy_metadata(self, source: "Image", target: "Image") -> list[str]:
        """Copy image-level metadata into empty target fields; return names."""
        copied: list[str] = []
        for field in self._META_FIELDS:
            src_val = getattr(source, field, None)
            if src_val in (None, ""):
                continue
            if getattr(target, field, None) in (None, ""):
                setattr(target, field, src_val)
                copied.append(field)
        return copied
