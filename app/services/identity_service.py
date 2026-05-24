"""Identity management service.

Handles user-driven operations on Person clusters:
* rename
* merge two clusters
* split (remove a face from a cluster)
* reassign a face to another cluster
* mark two faces as same / different
* delete a person (un-assign all faces)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Face, FaceCorrection, Person

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class IdentityService:
    """Provides atomic identity management operations.

    All methods commit on success.  On failure they roll back and re-raise.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Person operations
    # ------------------------------------------------------------------

    def rename_person(self, person_id: int, new_name: str) -> Person:
        """Rename a person / cluster.

        Args:
            person_id: DB primary key.
            new_name:  New display name.

        Returns:
            The updated :class:`Person` row.
        """
        person = self._require_person(person_id)
        if person.is_protected:
            raise ValueError(f"'{person.name}' is a protected person and cannot be renamed.")
        old_name = person.name
        was_auto_named = person.is_auto_named
        person.name = new_name.strip()
        person.is_auto_named = False
        if was_auto_named:
            now = _utcnow_naive()
            for face in person.faces:
                face.assignment_source = "manual"
                face.assignment_confidence = None
                face.assigned_at = now
        self._session.commit()
        log.info("Renamed person %d: %r → %r", person_id, old_name, person.name)
        return person

    def merge_persons(self, source_id: int, target_id: int) -> Person:
        """Move all faces from *source* into *target* and delete *source*.

        Args:
            source_id: Person to merge FROM (will be deleted).
            target_id: Person to merge INTO (will be kept).

        Returns:
            The surviving :class:`Person` row.
        """
        source = self._require_person(source_id)
        target = self._require_person(target_id)

        if source_id == target_id:
            raise ValueError("Cannot merge a person with itself")

        face_ids = [f.id for f in source.faces]
        log.info(
            "Merging person %d (%r) → %d (%r) — %d face(s)",
            source_id, source.name, target_id, target.name, len(face_ids),
        )

        # Re-assign faces
        self._session.query(Face).filter(Face.person_id == source_id).update(
            {
                Face.person_id: target_id,
                Face.assignment_source: "manual_merge",
                Face.assignment_confidence: None,
                Face.assigned_at: _utcnow_naive(),
            },
            synchronize_session="fetch",
        )

        # Migrate corrections
        self._session.query(FaceCorrection).filter(
            FaceCorrection.face_id_a.in_(face_ids)
        ).update(
            {FaceCorrection.face_id_a: FaceCorrection.face_id_a},
            synchronize_session=False,
        )

        # Record the merge as same-person pairs for future re-clustering
        target_faces = [f.id for f in target.faces if f.id not in face_ids]
        for a in face_ids:
            for b in target_faces[:5]:  # sample — avoid O(n²) explosion
                self._record_correction(a, b, same_person=True)

        # The bulk UPDATE above moved the faces in the DB, but ``source.faces``
        # is a stale loaded collection.  Expire it so deleting ``source`` does
        # not nullify the FK of faces that now belong to ``target``.
        self._session.expire(source, ["faces"])

        self._session.delete(source)
        self._session.commit()
        return target

    def delete_person(self, person_id: int) -> None:
        """Un-assign all faces from *person_id* and delete the person row."""
        person = self._require_person(person_id)
        if person.is_protected:
            raise ValueError(f"'{person.name}' is a protected person and cannot be deleted.")
        self._session.query(Face).filter(Face.person_id == person_id).update(
            {
                Face.person_id: None,
                Face.assignment_source: None,
                Face.assignment_confidence: None,
                Face.assigned_at: None,
            },
            synchronize_session="fetch",
        )
        self._session.delete(person)
        self._session.commit()
        log.info("Deleted person %d (%r)", person_id, person.name)

    # ------------------------------------------------------------------
    # Face operations
    # ------------------------------------------------------------------

    def reassign_face(self, face_id: int, target_person_id: int) -> Face:
        """Move a single face to a different person.

        Args:
            face_id:          Face to move.
            target_person_id: Person to move it to.

        Returns:
            The updated :class:`Face` row.
        """
        face = self._require_face(face_id)
        old_pid = face.person_id
        self._require_person(target_person_id)

        face.person_id = target_person_id
        face.assignment_source = "manual"
        face.assignment_confidence = None
        face.assigned_at = _utcnow_naive()
        self._session.commit()
        log.info(
            "Reassigned face %d: person %s → %d", face_id, old_pid, target_person_id
        )
        return face

    def remove_face_from_cluster(self, face_id: int) -> Face:
        """Un-assign a face from its current person (makes it unclustered)."""
        face = self._require_face(face_id)
        old_pid = face.person_id
        face.person_id = None
        face.is_excluded = True  # prevent re-clustering into same group
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        self._session.commit()
        log.info("Removed face %d from person %s", face_id, old_pid)
        return face

    def exclude_face(self, face_id: int) -> Face:
        """Mark a face as excluded from all future clustering."""
        face = self._require_face(face_id)
        face.person_id = None
        face.is_excluded = True
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        self._session.commit()
        log.info("Excluded face %d from clustering", face_id)
        return face

    # ------------------------------------------------------------------
    # Correction recording
    # ------------------------------------------------------------------

    def record_same(self, face_id_a: int, face_id_b: int) -> FaceCorrection:
        """Record that two faces are the same person."""
        return self._record_correction(face_id_a, face_id_b, same_person=True)

    def record_different(self, face_id_a: int, face_id_b: int) -> FaceCorrection:
        """Record that two faces are different people."""
        return self._record_correction(face_id_a, face_id_b, same_person=False)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_persons(
        self, named_only: bool = False, search: Optional[str] = None
    ) -> List[Person]:
        """Return all persons, optionally filtered."""
        q = self._session.query(Person)
        if named_only:
            q = q.filter(Person.is_auto_named == False)  # noqa: E712
        if search:
            q = q.filter(Person.name.ilike(f"%{search}%"))
        return q.order_by(Person.name).all()

    def get_faces_for_person(self, person_id: int) -> List[Face]:
        """Return all faces assigned to *person_id*."""
        return (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .all()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_person(self, person_id: int) -> Person:
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found")
        return person

    def _require_face(self, face_id: int) -> Face:
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError(f"Face id={face_id} not found")
        return face

    def _record_correction(
        self, face_id_a: int, face_id_b: int, same_person: bool
    ) -> FaceCorrection:
        """Insert or update a FaceCorrection record (idempotent)."""
        # Normalise order for uniqueness constraint
        a, b = sorted([face_id_a, face_id_b])

        existing: Optional[FaceCorrection] = (
            self._session.query(FaceCorrection)
            .filter(
                FaceCorrection.face_id_a == a,
                FaceCorrection.face_id_b == b,
            )
            .first()
        )

        if existing:
            existing.same_person = same_person
            self._session.commit()
            return existing

        correction = FaceCorrection(
            face_id_a=a, face_id_b=b, same_person=same_person
        )
        self._session.add(correction)
        self._session.commit()
        return correction
