"""Single source of truth for face-match similarity scoring.

Every person-selector popup that offers the *"order by face-match similarity"*
option (the image-browser inline editor, the merge / reassign dialogs, the batch
*Move faces* dialog, …) feeds its :class:`~app.ui.widgets.person_search_select.PersonSearchSelect`
with a ``{person_id: similarity}`` mapping produced **here** — there is no other
implementation.

All functions are best-effort: any missing embedding, missing profile, or
unexpected error yields an empty mapping, which the selector treats as
*"no match data"* (default name ordering, checkbox hidden).  A missing embedding
must never break a selector.

Two scoring paths are combined:

* **Recognition profiles** — the same probabilistic profiles automatic
  recognition uses (:meth:`RecognitionService.score_persons`).  This is the
  primary path and matches what the pipeline would decide.
* **Per-person centroid fallback** — when the primary path yields nothing (e.g.
  the query is an auto-named / unknown cluster that has no recognition profile),
  a normalised centroid is built from *every* person's available face
  embeddings, so unknown / auto-named people are also ranked.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np

from app.config import RecognitionConfig
from app.db.models import Face, Person
from app.services.recognition_service import RecognitionService

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Low-level vector helpers
# ──────────────────────────────────────────────────────────────────────────


def _normalised(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return *vec* scaled to unit length, or ``None`` when missing/degenerate."""
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else None


def _person_centroid(person: Person) -> Optional[np.ndarray]:
    """Normalised centroid of *person*'s available face embeddings, or ``None``."""
    vecs = [
        v
        for v in (_normalised(f.get_embedding()) for f in person.faces)
        if v is not None
    ]
    if not vecs:
        return None
    return _normalised(np.mean(vecs, axis=0))


def _fallback_scores(session, query_centroid: np.ndarray) -> Dict[int, float]:
    """Cosine-score *query_centroid* against every person's centroid.

    Unlike recognition profiles this also includes auto-named / unknown people,
    so a popup can still rank them when no recognition profile exists.
    """
    out: Dict[int, float] = {}
    for person in session.query(Person).all():
        centroid = _person_centroid(person)
        if centroid is None:
            continue
        out[person.id] = float(np.dot(query_centroid, centroid))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Public API — one entry point per kind of query
# ──────────────────────────────────────────────────────────────────────────


def match_scores_for_embedding(
    session,
    embedding: Optional[np.ndarray],
    recognition_cfg: Optional[RecognitionConfig] = None,
) -> Dict[int, float]:
    """Score *embedding* against known people; ``{}`` when scoring is impossible.

    Tries recognition profiles first, then the per-person centroid fallback.
    """
    if embedding is None:
        return {}
    try:
        scores = (
            RecognitionService(session, recognition_cfg).score_persons(embedding) or {}
        )
    except Exception:  # noqa: BLE001 — scoring is best-effort, never fatal
        log.exception("Face-match scoring failed; using default order")
        scores = {}
    if scores:
        return scores

    query = _normalised(embedding)
    if query is None:
        return {}
    try:
        return _fallback_scores(session, query)
    except Exception:  # noqa: BLE001
        log.exception("Fallback face-match scoring failed; continuing without scores")
        return {}


def match_scores_for_face(
    session,
    face_id: int,
    recognition_cfg: Optional[RecognitionConfig] = None,
    config=None,
) -> Dict[int, float]:
    """Score the face *face_id* against known people; ``{}`` on any failure.

    When the face has no stored embedding (e.g. an older manual mark) and a full
    *config* is supplied, the embedding is computed on the fly so match-ordering
    stays available instead of silently disappearing.
    """
    try:
        face = session.get(Face, face_id)
        if face is None:
            return {}
        embedding = face.get_embedding()
        if embedding is None and config is not None and face.crop_path:
            from app.services.embedding_service import embed_manual_face

            if embed_manual_face(session, face, config):
                embedding = face.get_embedding()
        return match_scores_for_embedding(session, embedding, recognition_cfg)
    except Exception:  # noqa: BLE001
        log.exception("Face-match scoring failed; using default order")
        return {}


def match_scores_for_faces(
    session,
    face_ids: Iterable[int],
    recognition_cfg: Optional[RecognitionConfig] = None,
) -> Dict[int, float]:
    """Score a *batch* of faces (e.g. a multi-select move) against known people.

    The selected faces' embeddings are averaged into one normalised query
    centroid before scoring, so the ordering reflects the group as a whole.
    """
    try:
        vecs = []
        for face_id in face_ids:
            face = session.get(Face, face_id)
            if face is None:
                continue
            v = _normalised(face.get_embedding())
            if v is not None:
                vecs.append(v)
        if not vecs:
            return {}
        centroid = _normalised(np.mean(vecs, axis=0))
        if centroid is None:
            return {}
        return match_scores_for_embedding(session, centroid, recognition_cfg)
    except Exception:  # noqa: BLE001
        log.exception("Batch face-match scoring failed; using default order")
        return {}


def match_scores_for_person(
    session,
    person: Person,
    recognition_cfg: Optional[RecognitionConfig] = None,
) -> Dict[int, float]:
    """Score everyone against *person* (a merge source); ``{}`` on any failure.

    Prefers *person*'s recognition-profile centroid (matching what automatic
    recognition would compare), falling back to a centroid built from the
    person's own face embeddings when no profile exists (auto-named / unknown).
    """
    try:
        svc = RecognitionService(session, recognition_cfg)
        profile = svc.build_profiles().get(person.id)
        if profile is not None:
            scores = svc.score_persons(profile.centroid) or {}
            if scores:
                return scores
    except Exception:  # noqa: BLE001
        log.exception("Profile-based match scoring failed; trying centroid fallback")

    src_centroid = _person_centroid(person)
    if src_centroid is None:
        return {}
    try:
        return _fallback_scores(session, src_centroid)
    except Exception:  # noqa: BLE001
        log.exception("Fallback face-match scoring failed; continuing without scores")
        return {}
