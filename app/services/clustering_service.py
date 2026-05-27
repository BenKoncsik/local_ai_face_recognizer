"""Clustering service.

Groups all faces that have embeddings into identity clusters and assigns
:class:`~app.db.models.Person` records.

Design
------
* Reads all embedded faces from DB.
* Runs DBSCAN clustering.
* Creates / reuses ``Person`` records (auto-named "Unknown N" for new ones).
* Assigns ``face.person_id`` for every face.
* Faces with label -1 (noise) are assigned to their own singleton Person.
* Re-clustering after user corrections is triggered by calling
  :meth:`recluster`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Dict, List, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.clustering.clusterer import cluster_embeddings
from app.config import ClusteringConfig
from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ClusteringService:
    """Assigns faces to Person clusters.

    Args:
        session: SQLAlchemy session.
        config:  Clustering configuration block.
    """

    def __init__(
        self,
        session: Session,
        config: ClusteringConfig,
        exclude_low_quality: bool = False,
    ) -> None:
        self._session = session
        self._config = config
        self._exclude_low_quality = exclude_low_quality

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Cluster all embedded faces and update person assignments.

        Returns:
            Number of distinct persons (clusters) created or updated.
        """
        face_ids, embeddings = self._load_embeddings()

        if not face_ids:
            log.info("No embedded faces to cluster")
            return 0

        same_pairs, diff_pairs = self._load_corrections(face_ids)

        label_map: Dict[int, int] = cluster_embeddings(
            face_ids=face_ids,
            embeddings=embeddings,
            epsilon=self._config.epsilon,
            min_samples=self._config.min_samples,
            metric=self._config.metric,
            same_pairs=same_pairs,
            diff_pairs=diff_pairs,
        )

        n_persons = self._assign_persons(face_ids, label_map)
        self._delete_orphan_auto_persons()
        self._session.commit()
        log.info("Clustering assigned %d person(s)", n_persons)
        return n_persons

    def recluster(self) -> int:
        """Re-run clustering with the latest corrections.

        Existing manually-set names are preserved for clusters that are
        stable (same face membership after re-clustering).
        """
        log.info("Re-clustering all faces with current corrections")
        return self.run()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_embeddings(self) -> Tuple[List[int], List[np.ndarray]]:
        """Load all embedded, non-excluded faces from DB."""
        query = (
            self._session.query(Face)
            .filter(Face._embedding.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
        )
        if self._exclude_low_quality:
            query = query.filter(
                or_(Face.is_low_quality.is_(None), Face.is_low_quality == False)  # noqa: E712
            )
        faces: List[Face] = query.all()

        face_ids = []
        embeddings = []
        for face in faces:
            emb = face.get_embedding()
            if emb is not None:
                face_ids.append(face.id)
                embeddings.append(emb)

        log.debug("Loaded %d embeddings for clustering", len(face_ids))
        return face_ids, embeddings

    def _load_corrections(
        self, face_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Load manual face-pair corrections from DB."""
        fid_set = set(face_ids)
        corrections: List[FaceCorrection] = self._session.query(FaceCorrection).all()

        same_pairs = [
            (c.face_id_a, c.face_id_b)
            for c in corrections
            if c.same_person
            and c.face_id_a in fid_set
            and c.face_id_b in fid_set
        ]
        diff_pairs = [
            (c.face_id_a, c.face_id_b)
            for c in corrections
            if not c.same_person
            and c.face_id_a in fid_set
            and c.face_id_b in fid_set
        ]

        log.debug("Corrections loaded: %d same, %d different", len(same_pairs), len(diff_pairs))
        return same_pairs, diff_pairs

    def _assign_persons(
        self, face_ids: List[int], label_map: Dict[int, int]
    ) -> int:
        """Map DBSCAN labels to Person rows and update Face.person_id."""

        # Get existing persons that are auto-named (we may reuse or extend them)
        auto_persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .all()
        )
        auto_counter = max((p.id for p in auto_persons), default=0)

        # Cache of label → Person.id (built as we go)
        label_to_person: Dict[int, int] = {}

        # Face lookup
        face_map: Dict[int, Face] = {
            f.id: f
            for f in self._session.query(Face)
            .filter(Face.id.in_(face_ids))
            .all()
        }

        for fid in face_ids:
            face = face_map.get(fid)
            if face is None:
                continue

            # If this face was manually assigned to a named person, skip it
            if face.person_id is not None:
                existing_person = self._session.get(Person, face.person_id)
                if existing_person and not existing_person.is_auto_named:
                    continue

            label = label_map.get(fid, -1)

            if label == -1:
                # Noise faces each get their own singleton person so different
                # people are not merged into a single "Unknown" catch-all.
                person = self._get_or_create_person(label, auto_counter)
                if person.id is None:
                    self._session.flush()
                face.person_id = person.id
                face.assignment_source = "clustering"
                face.assignment_confidence = None
                face.assigned_at = _utcnow_naive()
            else:
                if label not in label_to_person:
                    person = self._get_or_create_person(label, auto_counter)
                    if person.id is None:
                        self._session.flush()
                    label_to_person[label] = person.id  # type: ignore[assignment]
                face.person_id = label_to_person[label]
                face.assignment_source = "clustering"
                face.assignment_confidence = None
                face.assigned_at = _utcnow_naive()

        return len(label_to_person)

    def _delete_orphan_auto_persons(self) -> int:
        """Delete auto-named persons left with no faces after re-clustering.

        Every clustering run creates a fresh set of "Unknown N" persons and
        reassigns faces to them.  Persons from previous runs would otherwise
        accumulate as face-less rows that render as blank "?" entries in the
        UI.  Manually-named persons are never touched, even when face-less.

        Returns:
            Number of orphaned persons removed.
        """
        orphans: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .filter(~Person.faces.any())
            .all()
        )
        for person in orphans:
            self._session.delete(person)
        if orphans:
            log.info("Removed %d orphaned auto-named person(s)", len(orphans))
        return len(orphans)

    def _get_or_create_person(self, label: int, auto_counter: int) -> Person:
        """Return an existing auto Person for *label* or create a new one."""
        # For noise points (label == -1), each face gets its own singleton
        # person.  We never reuse -1 across calls to this method within a run,
        # so the caller increments the counter.
        name = self._next_auto_name()
        person = Person(name=name, is_auto_named=True)
        self._session.add(person)
        self._session.flush()
        return person

    def _next_auto_name(self) -> str:
        """Generate the next "Unknown N" name."""
        max_q = (
            self._session.query(Person)
            .filter(Person.name.like("Unknown %"))
            .filter(Person.is_auto_named == True)  # noqa: E712
            .count()
        )
        return f"Unknown {max_q + 1}"
