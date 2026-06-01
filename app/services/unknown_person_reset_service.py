"""Reset automatically created Unknown identities for a fresh recognition pass."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Face, Person


@dataclass(frozen=True)
class UnknownPersonResetResult:
    """Summary of an Unknown identity reset."""

    deleted_persons: int = 0
    unassigned_faces: int = 0


class UnknownPersonResetService:
    """Remove auto-named identities while preserving their detected faces."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def reset(self) -> UnknownPersonResetResult:
        """Unassign faces from auto-named persons and delete those persons.

        Embeddings and face boxes remain intact so the next pipeline run can
        attempt recognition again before rebuilding any remaining Unknown
        clusters.
        """
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
        deleted_persons = (
            self._session.query(Person)
            .filter(Person.id.in_(person_ids))
            .delete(synchronize_session=False)
        )
        self._session.commit()
        return UnknownPersonResetResult(
            deleted_persons=deleted_persons,
            unassigned_faces=unassigned_faces,
        )
