"""Composite detector — a primary detector augmented by recall-boosting peers.

YuNet is the project's primary detector (it produces the ArcFace-ordered
landmarks the "aligned" embedding crop relies on), but it is frontal-biased and
misses strongly turned / profile faces.  This wrapper runs the primary detector
plus one or more secondary detectors (e.g. InsightFace SCRFD) on the same image
and keeps every primary box, then *adds* any secondary box that does not overlap
a primary one.

The primary's boxes are therefore never displaced — existing behaviour for faces
YuNet already finds is unchanged — while faces only a secondary detector sees are
recovered.  The downstream multi-stage verification gate still filters the merged
set, so the looser recall does not translate into false positives.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from app.detectors.base import Detection, FaceDetector, detection_iou

log = logging.getLogger(__name__)


class CompositeDetector(FaceDetector):
    """Run a primary detector and merge in non-overlapping secondary boxes.

    Args:
        primary:       The authoritative detector (its boxes always survive).
        secondaries:   Extra detectors whose *new* (non-overlapping) boxes are
                       added for recall.
        overlap_iou:   A secondary box is dropped when it overlaps any already
                       kept box at this IoU or above.
    """

    def __init__(
        self,
        primary: FaceDetector,
        secondaries: List[FaceDetector],
        overlap_iou: float = 0.3,
        dedup_mutual_iou: float = 0.2,
    ) -> None:
        self._primary = primary
        self._secondaries = [s for s in secondaries if s is not None]
        self._overlap_iou = overlap_iou
        self._dedup_mutual_iou = dedup_mutual_iou

    @property
    def primary(self) -> FaceDetector:
        return self._primary

    def get(self, backend_substring: str) -> Optional[FaceDetector]:
        """Return the first member detector whose backend name matches, or None."""
        for d in (self._primary, *self._secondaries):
            if backend_substring in d.backend_name:
                return d
        return None

    @property
    def backend_name(self) -> str:
        names = "+".join(d.backend_name for d in (self._primary, *self._secondaries))
        return names

    def detect(
        self,
        image_bgr: np.ndarray,
        confidence_threshold: float = 0.5,
        min_face_size: int = 50,
    ) -> List[Detection]:
        primary = list(
            self._primary.detect(
                image_bgr,
                confidence_threshold=confidence_threshold,
                min_face_size=min_face_size,
            )
        )
        secondary: List[Detection] = []
        for sec in self._secondaries:
            try:
                secondary.extend(
                    sec.detect(
                        image_bgr,
                        confidence_threshold=confidence_threshold,
                        min_face_size=min_face_size,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Composite secondary %s failed: %s", sec.backend_name, exc)

        # 1) Collapse primary near-duplicates: the primary detector sometimes
        #    fires two offset boxes on one face (IoU below its own NMS / the
        #    overlap-resolution threshold, so both would otherwise survive).  A
        #    secondary box that sees that face as ONE is an anchor: primary boxes
        #    overlapping the same anchor *and* each other are the same face seen
        #    twice — keep the highest-confidence one.
        kept = self._dedup_primary(primary, secondary)

        # 2) Add secondary boxes no primary covers — the recall win (profile /
        #    turned faces the primary missed entirely).
        added = 0
        for det in secondary:
            if not any(self._same_face(det, k) for k in kept):
                kept.append(det)
                added += 1
        if added:
            log.debug("Composite: secondary added %d face(s) the primary missed", added)
        return kept

    def _same_face(self, a: Detection, b: Detection) -> bool:
        """True when *a* and *b* are the same face: they overlap enough, or one
        box's centre sits inside the other (handles a detector boxing the same
        face much tighter/looser than its peer — IoU stays low but it is clearly
        one face)."""
        if detection_iou(a, b) >= self._overlap_iou:
            return True
        return self._center_in(a, b) or self._center_in(b, a)

    @staticmethod
    def _center_in(a: Detection, b: Detection) -> bool:
        cx, cy = a.x + a.w / 2.0, a.y + a.h / 2.0
        return b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h

    def _dedup_primary(
        self, primary: List[Detection], secondary: List[Detection]
    ) -> List[Detection]:
        """Drop primary boxes that duplicate another primary box on the same
        secondary-anchored face (keeping the higher-confidence one)."""
        if len(primary) < 2 or not secondary:
            return list(primary)
        drop: set[int] = set()
        for anchor in secondary:
            group = [
                i for i, p in enumerate(primary)
                if detection_iou(p, anchor) >= self._overlap_iou
            ]
            if len(group) < 2:
                continue
            group.sort(key=lambda i: primary[i].confidence, reverse=True)
            for a_pos, ia in enumerate(group):
                if ia in drop:
                    continue
                for ib in group[a_pos + 1:]:
                    if ib in drop:
                        continue
                    # Same face only when the two primary boxes also overlap each
                    # other — keeps two *distinct* faces near one anchor apart.
                    if detection_iou(primary[ia], primary[ib]) >= self._dedup_mutual_iou:
                        drop.add(ib)
        if drop:
            log.debug("Composite: collapsed %d duplicate primary box(es)", len(drop))
        return [p for i, p in enumerate(primary) if i not in drop]
