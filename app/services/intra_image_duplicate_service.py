"""Same-image duplicate-detection cleanup.

Removes a *freshly detected, still-unassigned* face when it is really the same
physical face as one the user already marked on the **same** photo.

Why a dedicated pass?
---------------------
Detection already skips a new box that overlaps a retained face by IoU
(:data:`app.config.DetectionConfig.iou_merge_threshold`).  But when a photo is
re-detected — especially across detector modes — the new box can land *shifted*
or differently *sized* enough that the IoU falls under that bar while still
covering the same face.  The geometric check then misses it and the duplicate
becomes a brand-new "Unknown".  This is exactly the "same image, slightly
different spot, not recognised as already-processed" symptom.

This pass runs after embeddings exist and combines two independent signals:

* **embedding identity** — two crops of one face score very high cosine
  similarity, whereas two different people score far lower; and
* **spatial overlap** — the boxes must still overlap somewhat.

Requiring *both* keeps it safe: a genuine second appearance of the same person
elsewhere in the frame does not overlap, so it is never deleted.

Only unassigned, auto-detected faces are ever removed.  Manually drawn faces,
assigned faces and protected faces are anchors, never victims — mirroring the
guarantees of the intra-image *consistency* pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import IntraImageDuplicateConfig
from app.db.models import Face

log = logging.getLogger(__name__)


@dataclass
class IntraImageDuplicateStats:
    """Summary of a same-image duplicate cleanup run."""

    images_examined: int = 0
    faces_removed: int = 0


def _iou(a: Face, b: Face) -> float:
    """Intersection-over-union of two faces' bounding boxes."""
    ax2, ay2 = a.bbox_x + a.bbox_w, a.bbox_y + a.bbox_h
    bx2, by2 = b.bbox_x + b.bbox_w, b.bbox_y + b.bbox_h
    ix1, iy1 = max(a.bbox_x, b.bbox_x), max(a.bbox_y, b.bbox_y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = a.bbox_w * a.bbox_h + b.bbox_w * b.bbox_h - inter
    return inter / union if union > 0 else 0.0


class IntraImageDuplicateService:
    """Deletes duplicate detections of an already-known face within one image."""

    def __init__(
        self,
        session: Session,
        config: Optional[IntraImageDuplicateConfig] = None,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config or IntraImageDuplicateConfig()
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------

    def remove_duplicates(self) -> IntraImageDuplicateStats:
        """Remove same-image duplicate detections of already-retained faces.

        A candidate face (unassigned, auto-detected) is deleted when, for some
        retained face on the same image, BOTH:

        * cosine similarity ≥ ``duplicate_similarity``, and
        * bounding-box IoU ≥ ``min_overlap``.

        Runs in a single transaction; rolls back and re-raises on error.

        Returns:
            Counts of images examined and faces removed.
        """
        stats = IntraImageDuplicateStats()
        if not self._config.enabled:
            return stats

        candidates_by_image = self._load_candidates_by_image()
        if not candidates_by_image:
            return stats

        to_delete: List[int] = []
        try:
            for image_id, candidates in candidates_by_image.items():
                anchors = self._load_anchor_faces(image_id)
                if not anchors:
                    continue
                if len(candidates) + len(anchors) > self._config.max_faces_per_image:
                    log.debug(
                        "Skipping image %d: %d face(s) exceeds max_faces_per_image",
                        image_id, len(candidates) + len(anchors),
                    )
                    continue
                stats.images_examined += 1

                anchor_vecs = {a.id: self._unit(a) for a in anchors}
                for cand in candidates:
                    cvec = self._unit(cand)
                    if cvec is None:
                        continue
                    for anchor in anchors:
                        avec = anchor_vecs.get(anchor.id)
                        if avec is None:
                            continue
                        sim = float(np.dot(cvec, avec))
                        if sim < self._config.duplicate_similarity:
                            continue
                        if _iou(cand, anchor) < self._config.min_overlap:
                            continue
                        to_delete.append(cand.id)
                        log.debug(
                            "Image %d: face %d is a duplicate of retained face %d "
                            "(sim=%.3f, iou=%.3f) — removing",
                            image_id, cand.id, anchor.id, sim, _iou(cand, anchor),
                        )
                        break

            if to_delete:
                self._session.query(Face).filter(Face.id.in_(to_delete)).delete(
                    synchronize_session=False
                )
                stats.faces_removed = len(to_delete)
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        if stats.faces_removed:
            log.info(
                "Same-image duplicate cleanup: removed %d duplicate face(s) "
                "across %d image(s)",
                stats.faces_removed, stats.images_examined,
            )
        return stats

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_candidates_by_image(self) -> Dict[int, List[Face]]:
        """Unassigned, auto-detected, embedded faces grouped by image."""
        query = self._session.query(Face).filter(
            Face.person_id.is_(None),
            Face.detector_backend != "manual",
            Face._embedding.isnot(None),
            Face.is_excluded == False,  # noqa: E712
        )
        if self._exclude_low_quality:
            query = query.filter(
                (Face.is_low_quality.is_(None)) | (Face.is_low_quality == False)  # noqa: E712
            )
        grouped: Dict[int, List[Face]] = {}
        for face in query.all():
            grouped.setdefault(face.image_id, []).append(face)
        return grouped

    def _load_anchor_faces(self, image_id: int) -> List[Face]:
        """Retained faces on *image_id*: manual boxes or assigned faces."""
        return (
            self._session.query(Face)
            .filter(
                Face.image_id == image_id,
                Face._embedding.isnot(None),
                (Face.detector_backend == "manual") | (Face.person_id.isnot(None)),
            )
            .all()
        )

    @staticmethod
    def _unit(face: Face) -> Optional[np.ndarray]:
        emb = face.get_embedding()
        if emb is None:
            return None
        vec = np.asarray(emb, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
