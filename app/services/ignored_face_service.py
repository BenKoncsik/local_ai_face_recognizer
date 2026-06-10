"""Permanently-ignored faces service.

Implements the "ignore forever" feature for unwanted Unknown persons.  Plain
deletion only hides such a person until the next pipeline run re-clusters the
same embeddings into a fresh "Unknown N".  This service instead snapshots the
person's face embeddings into the persistent ``ignored_faces`` table; the
pipeline then suppresses any newly embedded, still-unassigned face whose
embedding is close enough to a stored vector.

Matching is embedding-based, never name-based: "Unknown 327"-style labels are
regenerated on every run and cannot identify a person across runs.

Pipeline placement matters: the filter runs *after* recognition and *before*
unknown clustering, so faces of known people (Kati, Zsuzsi, …) are claimed by
recognition first and are never ignore-filtered.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import IgnoredFaceConfig
from app.db.models import Face, IgnoredFace, Person

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class IgnoredFilterStats:
    """Counters from a single :meth:`IgnoredFaceService.suppress_matching_unassigned` run."""

    n_ignored_entries: int = 0   # embeddings on the ignore list
    n_candidates: int = 0        # unassigned embedded faces checked
    n_suppressed: int = 0        # faces excluded because they matched the list


class IgnoredFaceService:
    """Manages the persistent ignore list and applies it to new faces."""

    def __init__(
        self,
        session: Session,
        config: Optional[IgnoredFaceConfig] = None,
    ) -> None:
        self._session = session
        self._config = config or IgnoredFaceConfig()

    # ------------------------------------------------------------------
    # Ignore / un-ignore
    # ------------------------------------------------------------------

    def ignore_person_forever(self, person_id: int, note: Optional[str] = None) -> int:
        """Move every embedded face of an auto-named person to the ignore list.

        The person row is deleted and its faces are marked ``is_excluded`` so
        they disappear immediately; the stored embeddings keep them from
        resurfacing on later pipeline runs.

        Only auto-named ("Unknown N") persons may be ignored — manually named
        people must never be swallowed by the ignore list.

        Returns:
            Number of embeddings added to the ignore list.

        Raises:
            ValueError: when the person is missing, protected, or manually named.
        """
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found.")
        if person.is_protected:
            raise ValueError(f"'{person.name}' is protected and cannot be ignored.")
        if not person.is_auto_named:
            raise ValueError(
                f"'{person.name}' is a named person — only auto-named (Unknown) "
                "persons can be ignored forever."
            )

        n_added = 0
        for face in list(person.faces):
            if face.get_embedding() is not None:
                self._add_ignored_entry(face, person.name, note)
                n_added += 1
            # Hide the face either way: excluded faces are skipped by
            # recognition, clustering and the annotation overlays.
            face.is_excluded = True
            face.person_id = None
            face.assignment_source = "ignored"
            face.assignment_confidence = None
            face.assigned_at = _utcnow_naive()

        self._session.delete(person)
        self._session.commit()
        log.info(
            "Ignored forever: person %d (%r) — %d embedding(s) stored",
            person_id, person.name, n_added,
        )
        return n_added

    def snapshot_person_embeddings(
        self, person_id: int, note: Optional[str] = None
    ) -> int:
        """Copy a person's face embeddings onto the ignore list, no commit.

        Unlike :meth:`ignore_person_forever`, this neither deletes the person
        nor mutates its faces — it only records the embeddings so that the same
        physical person cannot resurface via re-detection.  The caller owns the
        transaction (used by :meth:`IdentityService.delete_person` when the user
        deletes a false person *and* ticks "exclude forever", so the snapshot and
        the hard delete commit together as one atomic operation).

        Returns:
            Number of embeddings added to the ignore list.

        Raises:
            ValueError: when the person is missing or protected.
        """
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found.")
        if person.is_protected:
            raise ValueError(f"'{person.name}' is protected and cannot be ignored.")

        n_added = 0
        for face in list(person.faces):
            if face.get_embedding() is not None:
                self._add_ignored_entry(face, person.name, note)
                n_added += 1
        log.info(
            "Snapshotted %d embedding(s) of person %d (%r) onto the ignore list",
            n_added, person_id, person.name,
        )
        return n_added

    def ignore_face_forever(self, face_id: int, note: Optional[str] = None) -> IgnoredFace:
        """Add a single face to the ignore list and exclude it.

        Refuses faces currently assigned to a manually named person.
        """
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError(f"Face id={face_id} not found.")
        if face.get_embedding() is None:
            raise ValueError(f"Face id={face_id} has no embedding.")
        if (
            face.person is not None
            and not face.person.is_auto_named
            and not face.person.is_protected
        ):
            raise ValueError(
                f"Face id={face_id} belongs to named person {face.person.name!r} "
                "— remove it from the person first."
            )

        source_person = face.person
        source_name = source_person.name if source_person is not None else None
        entry = self._add_ignored_entry(face, source_name, note)
        face.is_excluded = True
        face.person_id = None
        face.assignment_source = "ignored"
        face.assignment_confidence = None
        face.assigned_at = _utcnow_naive()

        # Ignoring the last face of an Unknown person would leave an empty
        # "?" placeholder behind — remove it (protected persons are kept).
        if (
            source_person is not None
            and source_person.is_auto_named
            and not source_person.is_protected
        ):
            remaining = (
                self._session.query(Face)
                .filter(Face.person_id == source_person.id)
                .count()
            )
            if remaining == 0:
                self._session.delete(source_person)

        self._session.commit()
        log.info("Ignored forever: face %d (from %r)", face_id, source_name)
        return entry

    def unignore(self, ignored_face_id: int) -> bool:
        """Remove an entry from the ignore list.

        When the source face still exists, it is un-excluded so the next
        pipeline run can recognise / cluster it again.

        Returns:
            True when an entry was removed.
        """
        entry = self._session.get(IgnoredFace, ignored_face_id)
        if entry is None:
            return False

        if entry.source_face_id is not None:
            face = self._session.get(Face, entry.source_face_id)
            if face is not None and face.assignment_source == "ignored":
                face.is_excluded = False
                face.assignment_source = None
                face.assignment_confidence = None
                face.assigned_at = None

        self._session.delete(entry)
        self._session.commit()
        log.info("Un-ignored entry %d", ignored_face_id)
        return True

    def list_ignored(self) -> List[IgnoredFace]:
        """Return every ignore-list entry, newest first."""
        return (
            self._session.query(IgnoredFace)
            .order_by(IgnoredFace.created_at.desc(), IgnoredFace.id.desc())
            .all()
        )

    def count_ignored(self) -> int:
        return self._session.query(IgnoredFace).count()

    # ------------------------------------------------------------------
    # Pipeline filter
    # ------------------------------------------------------------------

    def suppress_matching_unassigned(self) -> IgnoredFilterStats:
        """Exclude unassigned embedded faces that match the ignore list.

        Runs after recognition (so known people keep their faces) and before
        unknown clustering (so suppressed faces never spawn new Unknowns).
        A face is suppressed when its cosine similarity to *any* ignored
        embedding reaches ``ignore_similarity``.
        """
        stats = IgnoredFilterStats()
        if not self._config.enabled:
            return stats

        matrix = self._load_ignored_matrix()
        if matrix is None:
            return stats
        stats.n_ignored_entries = matrix.shape[0]

        candidates: List[Face] = (
            self._session.query(Face)
            .filter(Face._embedding.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.is_(None))
            .all()
        )
        stats.n_candidates = len(candidates)
        if not candidates:
            return stats

        threshold = self._config.ignore_similarity
        now = _utcnow_naive()
        for face in candidates:
            embedding = self._normalise(face.get_embedding())
            if embedding is None:
                continue
            similarity = float(np.max(matrix @ embedding))
            if similarity >= threshold:
                face.is_excluded = True
                face.assignment_source = "ignored"
                face.assignment_confidence = similarity
                face.assigned_at = now
                stats.n_suppressed += 1
                log.debug(
                    "Face id=%d suppressed by ignore list (similarity=%.3f)",
                    face.id, similarity,
                )

        self._session.commit()
        log.info(
            "Ignore filter: %d entr(ies), %d candidate(s), %d suppressed",
            stats.n_ignored_entries, stats.n_candidates, stats.n_suppressed,
        )
        return stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_ignored_entry(
        self, face: Face, source_name: Optional[str], note: Optional[str]
    ) -> IgnoredFace:
        entry = IgnoredFace(
            thumbnail_path=face.crop_path,
            source_face_id=face.id,
            source_person_name=source_name,
            note=note,
        )
        entry.set_embedding(face.get_embedding())
        self._session.add(entry)
        self._session.flush()
        return entry

    def _load_ignored_matrix(self) -> Optional[np.ndarray]:
        """Return all ignored embeddings as one L2-normalised matrix."""
        vectors: List[np.ndarray] = []
        for entry in self._session.query(IgnoredFace).all():
            vec = self._normalise(entry.get_embedding())
            if vec is not None:
                vectors.append(vec)
        if not vectors:
            return None
        return np.vstack(vectors).astype(np.float32)

    @staticmethod
    def _normalise(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        vec = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
