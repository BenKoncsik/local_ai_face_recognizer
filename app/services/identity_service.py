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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.models import Face, FaceCorrection, Person
from app.services.face_crop_service import face_debug_state

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# Person columns captured when snapshotting a source person for undo.  Excludes
# relationships and the timestamps that the ORM manages on its own.
_PERSON_SNAPSHOT_COLUMNS = (
    "id",
    "name",
    "is_auto_named",
    "gender",
    "thumbnail_path",
    "thumbnail_is_manual",
    "family_code",
    "external_family_code",
    "last_name",
    "first_name",
    "second_name",
    "nickname",
    "married_name",
    "birth_place",
    "birth_date",
    "death_place",
    "death_date",
    "notes",
    "is_protected",
)


@dataclass(frozen=True)
class FaceAssignmentSnapshot:
    """Pre-move assignment state of a single face — enough to undo the move."""

    face_id: int
    person_id: Optional[int]
    assignment_source: Optional[str]
    assignment_confidence: Optional[float]
    assigned_at: Optional[datetime]


@dataclass
class BulkReassignResult:
    """Outcome of a :meth:`IdentityService.reassign_faces_bulk` call.

    Carries everything :meth:`IdentityService.restore_face_assignments` needs to
    undo the operation, plus a list of face ids that were skipped because they no
    longer exist (or already belonged to the target).
    """

    target_person_id: int
    snapshots: List[FaceAssignmentSnapshot] = field(default_factory=list)
    skipped_face_ids: List[int] = field(default_factory=list)
    removed_persons: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def moved_count(self) -> int:
        return len(self.snapshots)

    @property
    def moved_face_ids(self) -> List[int]:
        return [s.face_id for s in self.snapshots]


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

    def merge_persons(
        self,
        source_id: int,
        target_id: int,
        respect_merge_exclusions: bool = False,
    ) -> Person:
        """Move faces from *source* into *target*.

        Normally every face of *source* moves into *target* and the now-empty
        *source* person is deleted.

        When *respect_merge_exclusions* is True, faces flagged with
        :attr:`Face.is_merge_excluded` are held back: only the non-excluded
        faces move to *target*, and the excluded faces stay assigned to
        *source* (which therefore survives the merge instead of being deleted).
        This backs the "exclude from merge" workflow in the name-suggestion
        gallery, where a few stray faces of a different person must not be
        merged into the named identity.  The merge-exclusion flag is cleared on
        the faces that stay behind, so the leftover Unknown cluster is clean for
        any future suggestion.

        Args:
            source_id: Person to merge FROM.
            target_id: Person to merge INTO (will be kept).
            respect_merge_exclusions: Keep ``is_merge_excluded`` faces in
                *source* instead of moving them.

        Returns:
            The surviving *target* :class:`Person` row.

        Raises:
            ValueError: if merging a person with itself, or if
                *respect_merge_exclusions* is set and **every** source face is
                excluded (nothing left to merge).
        """
        source = self._require_person(source_id)
        target = self._require_person(target_id)

        if source_id == target_id:
            raise ValueError("Cannot merge a person with itself")

        if respect_merge_exclusions:
            move_ids = [f.id for f in source.faces if not f.is_merge_excluded]
            keep_ids = [f.id for f in source.faces if f.is_merge_excluded]
        else:
            move_ids = [f.id for f in source.faces]
            keep_ids = []

        if not move_ids:
            raise ValueError(
                "All faces are excluded from the merge — nothing to merge."
            )

        log.info(
            "Merging person %d (%r) → %d (%r) — moving %d face(s), keeping %d excluded",
            source_id, source.name, target_id, target.name,
            len(move_ids), len(keep_ids),
        )

        # Re-assign only the faces that are being moved.
        self._session.query(Face).filter(Face.id.in_(move_ids)).update(
            {
                Face.person_id: target_id,
                Face.assignment_source: "manual_merge",
                Face.assignment_confidence: None,
                Face.assigned_at: _utcnow_naive(),
            },
            synchronize_session="fetch",
        )

        # Record the merge as same-person pairs for future re-clustering
        target_faces = [f.id for f in target.faces if f.id not in move_ids]
        for a in move_ids:
            for b in target_faces[:5]:  # sample — avoid O(n²) explosion
                self._record_correction(a, b, same_person=True)

        # The bulk UPDATE above moved the faces in the DB, but ``source.faces``
        # is a stale loaded collection.  Expire it so the source state below
        # reflects what actually remains.
        self._session.expire(source, ["faces"])

        if keep_ids:
            # Excluded faces stay with the source: it survives as a (smaller)
            # cluster.  Clear the flag so the leftover faces are not treated as
            # excluded by any future merge of this person.
            self._session.query(Face).filter(Face.id.in_(keep_ids)).update(
                {Face.is_merge_excluded: False},
                synchronize_session="fetch",
            )
            self._session.commit()
            return target

        self._session.delete(source)
        self._session.commit()
        return target

    def cleanup_empty_unknown_persons(self) -> int:
        """Delete auto-generated "Unknown" persons that have no faces left.

        Reassigning, removing or excluding faces can drain an auto-named
        ("Unknown N") cluster until it holds no :class:`Face` rows.  Such
        face-less placeholders render as blank "?" entries and serve no
        purpose, so this maintenance pass removes them.

        Strict safety rules — a person is only deleted when ALL hold:

        * ``is_auto_named is True``  – it is a generated Unknown placeholder,
          never a manually named identity (those survive even when face-less);
        * ``is_protected is False``  – the canonical "Ismeretlen" person is
          always kept;
        * it has zero associated faces (checked with a DB-side ``NOT EXISTS``
          subquery, not an in-memory scan).

        The whole operation runs in a single transaction: on any error it
        rolls back and re-raises, leaving the database untouched.

        Returns:
            The number of empty Unknown persons that were deleted.
        """
        empty_ids = [
            pid
            for (pid,) in (
                self._session.query(Person.id)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .filter(~Person.faces.any())
                .all()
            )
        ]
        if not empty_ids:
            return 0

        try:
            deleted = (
                self._session.query(Person)
                .filter(Person.id.in_(empty_ids))
                .delete(synchronize_session=False)
            )
            self._session.commit()
        except Exception:
            self._session.rollback()
            raise

        log.info("Cleaned up %d empty Unknown person(s)", deleted)
        return deleted

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
        if face.image is not None:
            _ = face.image.file_path
        log.debug("Assignment before: %s", face_debug_state(face, face.crop_path))

        # Invalidate manual thumbnail on the *source* person before the move
        self._invalidate_manual_thumbnail_if_needed(face)

        face.person_id = target_person_id
        face.assignment_source = "manual"
        face.assignment_confidence = None
        face.assigned_at = _utcnow_naive()
        log.debug("Assignment after: %s", face_debug_state(face, face.crop_path))
        self._session.commit()
        log.info(
            "Reassigned face %d: person %s → %d", face_id, old_pid, target_person_id
        )
        # Moving the last face away may leave the source Unknown cluster empty.
        self.cleanup_empty_unknown_persons()
        return face

    def reassign_faces_bulk(
        self, face_ids: List[int], target_person_id: int
    ) -> BulkReassignResult:
        """Move several faces to *target_person_id* in a single transaction.

        Builds on the single-face :meth:`reassign_face` semantics but applies
        them atomically: either every still-existing face moves, or — on any
        error — nothing does (the caller's ``session_scope`` rolls back).

        Faces that no longer exist, or that already belong to the target, are
        skipped (recorded in :attr:`BulkReassignResult.skipped_face_ids`) rather
        than aborting the whole operation.  Source "Unknown" clusters left empty
        are cleaned up just like in the single-face path, but a snapshot of each
        deleted person is kept so :meth:`restore_face_assignments` can undo the
        move even when it emptied a cluster.

        Args:
            face_ids:          Faces to move (order is irrelevant; duplicates ok).
            target_person_id:  Person to move them to.

        Returns:
            A :class:`BulkReassignResult` describing what moved, what was
            skipped, and the data needed to undo the operation.
        """
        self._require_person(target_person_id)

        result = BulkReassignResult(target_person_id=target_person_id)
        now = _utcnow_naive()
        seen: set[int] = set()
        source_ids: set[int] = set()

        for face_id in face_ids:
            if face_id in seen:
                continue
            seen.add(face_id)

            face = self._session.get(Face, face_id)
            if face is None:
                result.skipped_face_ids.append(face_id)
                continue
            if face.person_id == target_person_id:
                # Already where the user wants it — nothing to move or undo.
                result.skipped_face_ids.append(face_id)
                continue

            result.snapshots.append(
                FaceAssignmentSnapshot(
                    face_id=face.id,
                    person_id=face.person_id,
                    assignment_source=face.assignment_source,
                    assignment_confidence=face.assignment_confidence,
                    assigned_at=face.assigned_at,
                )
            )
            if face.person_id is not None:
                source_ids.add(face.person_id)

            self._invalidate_manual_thumbnail_if_needed(face)
            face.person_id = target_person_id
            face.assignment_source = "manual"
            face.assignment_confidence = None
            face.assigned_at = now

        if not result.snapshots:
            # No-op move (every face skipped); avoid an empty cleanup pass.
            return result

        self._session.flush()

        # Snapshot the source clusters that the cleanup pass is about to delete,
        # so an undo can recreate them before restoring the faces.
        for pid in source_ids:
            person = self._session.get(Person, pid)
            if (
                person is not None
                and person.is_auto_named
                and not person.is_protected
                and not person.faces
            ):
                result.removed_persons.append(self._snapshot_person(person))

        self._session.commit()
        log.info(
            "Bulk reassigned %d face(s) → person %d (%d skipped, %d source cluster(s) removed)",
            result.moved_count,
            target_person_id,
            len(result.skipped_face_ids),
            len(result.removed_persons),
        )
        self.cleanup_empty_unknown_persons()
        return result

    def restore_face_assignments(
        self,
        snapshots: List[FaceAssignmentSnapshot],
        removed_persons: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Undo a bulk move by restoring each face to its snapshotted state.

        Source clusters that the move emptied (and the cleanup pass deleted) are
        recreated from *removed_persons* before the faces are reattached, so the
        undo is complete even in that edge case.  Faces that have since been
        deleted are silently skipped.

        Args:
            snapshots:        The per-face state captured by the bulk move.
            removed_persons:  Snapshots of source persons deleted by cleanup.

        Returns:
            The number of faces actually restored.
        """
        for snap in removed_persons or []:
            pid = snap.get("id")
            if pid is None or self._session.get(Person, pid) is not None:
                continue
            self._session.add(Person(**snap))
        self._session.flush()

        restored = 0
        for snap in snapshots:
            face = self._session.get(Face, snap.face_id)
            if face is None:
                continue
            # Only restore to a person that still exists; fall back to unassigned.
            target = snap.person_id
            if target is not None and self._session.get(Person, target) is None:
                target = None
            face.person_id = target
            face.assignment_source = snap.assignment_source
            face.assignment_confidence = snap.assignment_confidence
            face.assigned_at = snap.assigned_at
            restored += 1

        self._session.commit()
        log.info("Restored %d face assignment(s) (undo)", restored)
        return restored

    def _snapshot_person(self, person: Person) -> Dict[str, Any]:
        """Capture a person's column values for later recreation (undo)."""
        return {col: getattr(person, col) for col in _PERSON_SNAPSHOT_COLUMNS}

    def remove_face_from_cluster(self, face_id: int) -> Face:
        """Un-assign a face from its current person (makes it unclustered)."""
        face = self._require_face(face_id)
        old_pid = face.person_id
        self._invalidate_manual_thumbnail_if_needed(face)
        face.person_id = None
        face.is_excluded = True  # prevent re-clustering into same group
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        self._session.commit()
        log.info("Removed face %d from person %s", face_id, old_pid)
        # Un-assigning the last face may leave the source Unknown cluster empty.
        self.cleanup_empty_unknown_persons()
        return face

    def exclude_face(self, face_id: int) -> Face:
        """Mark a face as excluded from all future clustering."""
        face = self._require_face(face_id)
        self._invalidate_manual_thumbnail_if_needed(face)
        face.person_id = None
        face.is_excluded = True
        face.assignment_source = None
        face.assignment_confidence = None
        face.assigned_at = None
        self._session.commit()
        log.info("Excluded face %d from clustering", face_id)
        # Excluding the last face may leave the source Unknown cluster empty.
        self.cleanup_empty_unknown_persons()
        return face

    def set_face_merge_excluded(self, face_id: int, excluded: bool) -> Face:
        """Flag (or un-flag) a face as excluded from name-matching merges.

        The face stays assigned to its current person and visible everywhere;
        it is only held back when its person is merged into a named identity
        (see :meth:`merge_persons` with ``respect_merge_exclusions=True``) and
        is dropped from the candidate's matching profile so it never counts as a
        positive example for the merge.  The flag is persistent.

        Args:
            face_id:  Face to flag.
            excluded: ``True`` to exclude from the merge, ``False`` to restore.

        Returns:
            The updated :class:`Face` row.
        """
        face = self._require_face(face_id)
        face.is_merge_excluded = bool(excluded)
        self._session.commit()
        log.info(
            "Face %d %s merge", face_id,
            "excluded from" if excluded else "restored to",
        )
        return face

    # ------------------------------------------------------------------
    # Thumbnail management
    # ------------------------------------------------------------------

    def set_person_thumbnail(self, person_id: int, face_id: int) -> Person:
        """Set a specific face crop as the manual thumbnail for a person.

        Args:
            person_id: Person whose thumbnail is being set.
            face_id:   Face whose crop_path will become the thumbnail.

        Returns:
            The updated :class:`Person` row.
        """
        person = self._require_person(person_id)
        face = self._require_face(face_id)
        if face.person_id != person_id:
            raise ValueError(
                f"Face {face_id} does not belong to person {person_id}"
            )
        if not face.crop_path or not Path(face.crop_path).exists():
            raise ValueError(
                f"Face {face_id} has no valid crop file"
            )
        person.thumbnail_path = face.crop_path
        person.thumbnail_is_manual = True
        self._session.commit()
        log.info(
            "Set manual thumbnail for person %d → face %d (%s)",
            person_id, face_id, face.crop_path,
        )
        return person

    def clear_manual_person_thumbnail(self, person_id: int) -> Person:
        """Reset a person's thumbnail to automatic selection.

        Picks the first available face crop as the new fallback.

        Args:
            person_id: Person to reset.

        Returns:
            The updated :class:`Person` row.
        """
        person = self._require_person(person_id)
        person.thumbnail_is_manual = False
        faces_with_crop = [f for f in person.faces if f.crop_path]
        person.thumbnail_path = (
            faces_with_crop[0].crop_path if faces_with_crop else None
        )
        self._session.commit()
        log.info("Cleared manual thumbnail for person %d", person_id)
        return person

    # ------------------------------------------------------------------
    # Private thumbnail helpers
    # ------------------------------------------------------------------

    def _invalidate_manual_thumbnail_if_needed(self, face: Face) -> None:
        """If face is the manual thumbnail of its person, fall back to auto."""
        if face.person_id is None:
            return
        person = self._session.get(Person, face.person_id)
        if person is None:
            return
        if person.thumbnail_is_manual and person.thumbnail_path == face.crop_path:
            person.thumbnail_is_manual = False
            remaining = [
                f for f in person.faces
                if f.id != face.id and f.crop_path
            ]
            person.thumbnail_path = (
                remaining[0].crop_path if remaining else None
            )

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
