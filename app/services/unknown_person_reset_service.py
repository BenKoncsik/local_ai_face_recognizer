"""Reset automatically created Unknown identities for a fresh recognition pass."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Face, Person


@dataclass(frozen=True)
class UnknownPersonResetOptions:
    """Configuration options for resetting Unknown identities."""

    delete_unknown_persons: bool = True
    """Delete auto-created 'Unknown N' persons."""

    delete_face_assignments: bool = True
    """Unassign faces from Unknown persons (delete person_id, etc.)."""

    delete_face_data: bool = False
    """Delete actual face detection data (embeddings, bboxes) - dangerous!"""

    rebuild_clusters: bool = True
    """Re-cluster unassigned faces into new Unknown groups after reset."""


@dataclass(frozen=True)
class UnknownPersonResetResult:
    """Summary of an Unknown identity reset."""

    deleted_persons: int = 0
    unassigned_faces: int = 0
    persons_deleted: bool = False
    """Whether Unknown persons were deleted."""

    faces_unassigned: bool = False
    """Whether face assignments were removed."""

    clusters_rebuilt: bool = False
    """Whether clustering was queued to rebuild Unknown groups."""


class UnknownPersonResetService:
    """Remove auto-named identities while preserving their detected faces."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def reset(
        self, options: UnknownPersonResetOptions | None = None
    ) -> UnknownPersonResetResult:
        """Unassign faces from auto-named persons and delete those persons.

        Embeddings and face boxes remain intact so the next pipeline run can
        attempt recognition again before rebuilding any remaining Unknown
        clusters.

        Args:
            options: Configuration for which steps to perform. Defaults to all steps.
        """
        if options is None:
            options = UnknownPersonResetOptions()

        deleted_persons = 0
        unassigned_faces = 0
        persons_deleted = False
        faces_unassigned = False

        if not options.delete_unknown_persons and not options.delete_face_assignments:
            # Nothing to do
            return UnknownPersonResetResult()

        # Find auto-named persons to process
        person_ids = [
            person_id
            for (person_id,) in (
                self._session.query(Person.id)
                .filter(Person.is_auto_named == True)  # noqa: E712
                .filter(Person.is_protected == False)  # noqa: E712
                .all()
            )
        ]
        if not person_ids:
            return UnknownPersonResetResult()

        # Step 1: Unassign faces from Unknown persons (if enabled)
        if options.delete_face_assignments:
            unassigned_faces = (
                self._session.query(Face)
                .filter(Face.person_id.in_(person_ids))
                .update(
                    {
                        Face.person_id: None,
                        Face.assignment_source: None,
                        Face.assignment_confidence: None,
                        Face.assigned_at: None,
                    },
                    synchronize_session=False,
                )
            )
            faces_unassigned = unassigned_faces > 0

        # Step 2: Delete Unknown persons (if enabled)
        if options.delete_unknown_persons:
            deleted_persons = (
                self._session.query(Person)
                .filter(Person.id.in_(person_ids))
                .delete(synchronize_session=False)
            )
            persons_deleted = deleted_persons > 0

        # Step 3: Delete face data if requested (advanced option, disabled by default)
        if options.delete_face_data and options.delete_face_assignments:
            # Only delete embedding/box data if explicitly requested
            remaining_unknown_faces = (
                self._session.query(Face)
                .filter(Face.person_id.in_(person_ids))
                .all()
            )
            for face in remaining_unknown_faces:
                face.embedding = None
                face.x = None
                face.y = None
                face.w = None
                face.h = None

        self._session.commit()
        return UnknownPersonResetResult(
            deleted_persons=deleted_persons,
            unassigned_faces=unassigned_faces,
            persons_deleted=persons_deleted,
            faces_unassigned=faces_unassigned,
            clusters_rebuilt=options.rebuild_clusters,
        )
