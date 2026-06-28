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
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.clustering.clusterer import cluster_embeddings
from app.config import ClusteringConfig
from app.db.models import Face, FaceCorrection, Person
from app.db.query_utils import in_chunks

log = logging.getLogger(__name__)


@dataclass
class ClusteringStats:
    """Counters returned by the incremental unknown-face clustering stage."""
    n_assigned_to_existing: int = 0   # faces added to pre-existing Unknown persons
    n_new_persons: int = 0             # new Unknown N persons created
    n_assigned_to_new: int = 0         # faces assigned to newly created Unknown persons
    n_singletons: int = 0              # noise / sub-threshold faces left unassigned


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
            .filter(Face.embedding_exists())
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

        # Face lookup (chunked to stay under SQLite's per-statement parameter limit)
        face_map: Dict[int, Face] = {}
        for chunk in in_chunks(face_ids):
            for f in self._session.query(Face).filter(Face.id.in_(chunk)).all():
                face_map[f.id] = f

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
        """Generate the next "Unknown N" name.

        Uses the highest existing number (not count) to avoid duplicates when
        some Unknown persons have been deleted, leaving gaps in the sequence.
        """
        rows = (
            self._session.query(Person.name)
            .filter(Person.name.like("Unknown %"))
            .filter(Person.is_auto_named == True)  # noqa: E712
            .all()
        )
        max_n = 0
        for (name,) in rows:
            m = re.match(r"^Unknown (\d+)$", name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
        return f"Unknown {max_n + 1}"

    # ------------------------------------------------------------------
    # Incremental unknown clustering (called from the pipeline)
    # ------------------------------------------------------------------

    def cluster_unassigned(self) -> ClusteringStats:
        """Incrementally cluster only faces not yet assigned to any person.

        Unlike :meth:`run`, this method **only** processes faces whose
        ``person_id`` is ``NULL`` (i.e. unassigned after the recognition
        stage).  Faces already linked to real or Unknown persons are never
        moved.

        Strategy
        --------
        1. Load truly unassigned (person_id IS NULL) embedded faces.
        2. For each face, check cosine distance against existing Unknown
           persons' centroids.  If close enough → assign to that Unknown.
        3. DBSCAN-cluster the remaining unassigned faces.
        4. Clusters ≥ min_cluster_size → create a new Unknown person.
        5. Smaller clusters / noise → leave unassigned (avoid singleton spam).

        Returns
        -------
        ClusteringStats with counts of what happened.
        """
        stats = ClusteringStats()

        face_ids, embeddings = self._load_null_assignment_embeddings()
        if not face_ids:
            log.info("Unknown clustering: no unassigned embedded faces")
            return stats

        log.info("Unknown clustering: %d unassigned face(s) to process", len(face_ids))

        # Normalise embeddings once up-front
        normed: List[np.ndarray] = []
        for emb in embeddings:
            norm = float(np.linalg.norm(emb))
            normed.append(emb / norm if norm > 1e-8 else emb.copy())

        existing_unknowns = self._load_unknown_centroids()
        face_map: Dict[int, Face] = {}
        for chunk in in_chunks(face_ids):
            for f in self._session.query(Face).filter(Face.id.in_(chunk)).all():
                face_map[f.id] = f
        now = _utcnow_naive()
        threshold = self._config.unknown_assign_threshold

        # --- Step 1: try to assign each face to an existing Unknown person ---
        still_ids: List[int] = []
        still_embs: List[np.ndarray] = []

        for fid, emb in zip(face_ids, normed):
            if existing_unknowns:
                best_pid, best_dist = self._closest_unknown(emb, existing_unknowns)
                if best_pid is not None and best_dist <= threshold:
                    face = face_map[fid]
                    face.person_id = best_pid
                    face.assignment_source = "clustering"
                    face.assigned_at = now
                    stats.n_assigned_to_existing += 1
                    continue
            still_ids.append(fid)
            still_embs.append(emb)

        log.info(
            "Unknown clustering: %d → existing Unknown persons, %d remaining for DBSCAN",
            stats.n_assigned_to_existing, len(still_ids),
        )

        if not still_ids:
            self._session.commit()
            return stats

        # --- Step 2: DBSCAN on remaining unassigned faces ---
        same_pairs, diff_pairs = self._load_corrections(still_ids)
        label_map = cluster_embeddings(
            face_ids=still_ids,
            embeddings=still_embs,
            epsilon=self._config.epsilon,
            min_samples=self._config.min_samples,
            metric=self._config.metric,
            same_pairs=same_pairs,
            diff_pairs=diff_pairs,
        )

        # Group face IDs by cluster label
        cluster_faces: Dict[int, List[int]] = defaultdict(list)
        for fid in still_ids:
            label = label_map.get(fid, -1)
            cluster_faces[label].append(fid)

        n_noise = len(cluster_faces.get(-1, []))
        n_clusters = sum(1 for lbl in cluster_faces if lbl != -1)
        log.info(
            "Unknown clustering DBSCAN: %d cluster(s), %d noise point(s)",
            n_clusters, n_noise,
        )

        # --- Step 3: assign clusters to (new) Unknown persons ---
        min_size = self._config.create_unknown_min_cluster_size

        for label, fids in cluster_faces.items():
            if label == -1 or len(fids) < min_size:
                # Noise or too small: leave unassigned
                stats.n_singletons += len(fids)
                continue

            person = Person(name=self._next_auto_name(), is_auto_named=True)
            self._session.add(person)
            self._session.flush()
            stats.n_new_persons += 1

            for fid in fids:
                face = face_map[fid]
                face.person_id = person.id
                face.assignment_source = "clustering"
                face.assigned_at = now
                stats.n_assigned_to_new += 1

            log.debug(
                "Unknown clustering: created %r (person_id=%d) with %d face(s)",
                person.name, person.id, len(fids),
            )

        self._session.commit()

        log.info(
            "Unknown clustering complete: %d new Unknown person(s), "
            "%d face(s) → existing, %d face(s) → new, %d face(s) unassigned",
            stats.n_new_persons,
            stats.n_assigned_to_existing,
            stats.n_assigned_to_new,
            stats.n_singletons,
        )
        return stats

    # ------------------------------------------------------------------
    # Helpers for incremental clustering
    # ------------------------------------------------------------------

    def _load_null_assignment_embeddings(
        self,
    ) -> Tuple[List[int], List[np.ndarray]]:
        """Load embedded faces that have no person assignment at all."""
        query = (
            self._session.query(Face)
            .filter(Face.embedding_exists())
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.is_(None))
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

        log.debug(
            "Loaded %d unassigned embeddings for incremental clustering", len(face_ids)
        )
        return face_ids, embeddings

    def _load_unknown_centroids(self) -> Dict[int, np.ndarray]:
        """Return ``{person_id: centroid}`` for all existing auto-named persons."""
        auto_persons: List[Person] = (
            self._session.query(Person)
            .filter(Person.is_auto_named == True)  # noqa: E712
            .all()
        )
        centroids: Dict[int, np.ndarray] = {}
        for person in auto_persons:
            vecs: List[np.ndarray] = []
            for face in person.faces:
                emb = face.get_embedding()
                if emb is not None:
                    norm = float(np.linalg.norm(emb))
                    if norm > 1e-8:
                        vecs.append(emb / norm)
            if vecs:
                centroid = np.mean(vecs, axis=0).astype(np.float32)
                norm = float(np.linalg.norm(centroid))
                if norm > 1e-8:
                    centroids[person.id] = centroid / norm

        log.debug(
            "Loaded centroids for %d existing Unknown person(s)", len(centroids)
        )
        return centroids

    def _closest_unknown(
        self,
        embedding: np.ndarray,
        centroids: Dict[int, np.ndarray],
    ) -> Tuple[Optional[int], float]:
        """Return ``(person_id, cosine_distance)`` of the nearest Unknown person."""
        best_pid: Optional[int] = None
        best_dist = float("inf")
        for pid, centroid in centroids.items():
            dist = float(1.0 - np.dot(embedding, centroid))
            if dist < best_dist:
                best_dist = dist
                best_pid = pid
        return best_pid, best_dist
