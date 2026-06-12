"""Resolve the effective (fuzzy) date of a face for sorting and the timeline.

A face has no date of its own — it inherits the date of the image it was
detected in.  The image's date may come from several sources; this module
applies a fixed priority order and returns a normalised :class:`FuzzyDate`:

1. manual / EXIF photo date  (``Image.photo_date``)
2. estimated date            (``Image.estimated_date`` — optional column)
3. family-tree derived date  (future: derived from relationships/events)

The result is cached per ``Image`` id within a single resolver instance so a
grid of hundreds of faces parses each distinct date string only once.
"""

from __future__ import annotations

from typing import Dict, Optional

from app.db.models import Face, Image
from app.utils.fuzzy_date import FuzzyDate


class FaceDateResolver:
    """Caching resolver for the effective :class:`FuzzyDate` of faces/images."""

    def __init__(self) -> None:
        self._cache: Dict[int, FuzzyDate] = {}

    # ------------------------------------------------------------------
    def for_image(self, image: Optional[Image]) -> FuzzyDate:
        if image is None:
            return FuzzyDate.unknown()
        cached = self._cache.get(image.id)
        if cached is not None:
            return cached
        fd = self._resolve(image)
        self._cache[image.id] = fd
        return fd

    def for_face(self, face: Face) -> FuzzyDate:
        return self.for_image(face.image)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(image: Image) -> FuzzyDate:
        # 1) manual / EXIF photo date
        fd = FuzzyDate.parse(image.photo_date)
        if fd.is_known:
            return fd
        # 2) estimated date — optional column, present only after migration.
        estimated = getattr(image, "estimated_date", None)
        if estimated:
            est = FuzzyDate.parse(estimated)
            if est.is_known:
                # Flag as estimated even if the string itself looked exact.
                if not est.is_estimated:
                    est = FuzzyDate(
                        est.earliest, est.latest, est.precision, True, est.raw
                    )
                return est
        # 3) family-tree derived date — reserved for a later phase.
        return fd  # UNKNOWN


# Module-level convenience for one-off lookups (no cross-call caching).
def effective_fuzzy_date(face: Face) -> FuzzyDate:
    """Return the effective :class:`FuzzyDate` for a single *face*."""
    return FaceDateResolver().for_face(face)
