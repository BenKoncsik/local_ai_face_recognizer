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
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.db.models import Image

_RE_DEOLDIFIED = re.compile(r"-deoldified.*$", re.IGNORECASE)


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


def is_deoldified_path(path: str) -> bool:
    """Return True if the filename contains '-deoldified' (case-insensitive)."""
    return extract_original_stem(Path(path).stem) is not None


class DeoldifiedPairingService:
    """Finds paired images in the database for the deoldified feature."""

    def __init__(self, session: "Session") -> None:
        self._session = session

    def find_original_for_deoldified(
        self, deoldified_image: "Image"
    ) -> Optional["Image"]:
        """Return the original Image record for a deoldified image, or None.

        Searches the same directory using the expected original filename.
        Tries exact extension first, then lowercase/uppercase variants.
        """
        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(deoldified_image)
        file_path = Path(resolved) if resolved else Path(deoldified_image.file_path)

        original_name = extract_original_filename(file_path.name)
        if original_name is None:
            return None

        folder = file_path.parent
        ext = Path(original_name).suffix
        stem_orig = Path(original_name).stem

        # Build candidate names: exact extension + case variants
        candidates: list[str] = [original_name]
        for alt_ext in (ext.lower(), ext.upper()):
            alt = stem_orig + alt_ext
            if alt not in candidates:
                candidates.append(alt)

        for name in candidates:
            img = (
                self._session.query(Image)
                .filter(Image.file_path == str(folder / name))
                .first()
            )
            if img is not None:
                return img

        # Relative path fallback
        if deoldified_image.relative_path is not None:
            rel = Path(deoldified_image.relative_path)
            for name in candidates:
                orig_rel = str(rel.parent / name)
                img = (
                    self._session.query(Image)
                    .filter(Image.relative_path == orig_rel)
                    .first()
                )
                if img is not None:
                    return img

        return None

    def find_deoldified_for_original(
        self, original_image: "Image"
    ) -> Optional["Image"]:
        """Return a deoldified Image for an original image, or None.

        Searches for an image in the same folder whose file_path contains
        '{original_stem}-deoldified' (case-insensitive).
        """
        import sqlalchemy as sa
        from app.db.models import Image
        from app.services.image_library_service import resolve_image_path

        resolved = resolve_image_path(original_image)
        file_path = Path(resolved) if resolved else Path(original_image.file_path)

        stem = file_path.stem
        folder = file_path.parent

        pattern_prefix = str(folder / f"{stem}-deoldified").lower()
        safe = (
            pattern_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

        img = (
            self._session.query(Image)
            .filter(
                sa.func.lower(Image.file_path).like(safe + "%", escape="\\")
            )
            .first()
        )
        if img is not None:
            return img

        if original_image.relative_path is not None:
            rel = Path(original_image.relative_path)
            pattern_rel = str(rel.parent / f"{rel.stem}-deoldified").lower()
            safe_rel = (
                pattern_rel
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            img = (
                self._session.query(Image)
                .filter(
                    sa.func.lower(Image.relative_path).like(
                        safe_rel + "%", escape="\\"
                    )
                )
                .first()
            )
            if img is not None:
                return img

        return None
