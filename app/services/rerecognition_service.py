"""User-triggered re-recognition of Unknown faces in the image browser.

Given a set of images, this service re-examines the faces that are still
*unresolved* (assigned to an auto-named "Unknown N" cluster, to the protected
catch-all, or to no person) and tries to match each one against the embedding
profiles of the **named** people — exactly the same profiles and scoring the
automatic recognizer uses (:class:`app.services.recognition_service.RecognitionService`).

Nothing new is ever created: a face can only be merged into an *existing* named
person.  Matches are classified by score:

* ``score >= auto_threshold`` (with a sufficient margin)  → **AUTO** merge,
* ``suggest_threshold <= score < auto_threshold``         → **SUGGEST** (review),
* otherwise                                               → **NONE**.

The heavy scoring is pure (numpy) and operates on detached :class:`FaceData`
snapshots, so the background worker can run it off the UI thread / in a pool.
Database writes (the merge itself plus an audit/undo row in
``recognition_merge_log``) are serialised through :class:`IdentityService`, whose
:meth:`reassign_faces_bulk` already handles empty-Unknown cleanup, and whose
:meth:`restore_face_assignments` powers per-batch undo.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
from sqlalchemy.orm import Session

from app.config import RecognitionConfig
from app.db.models import Face, RecognitionMergeLog
from app.services.identity_service import FaceAssignmentSnapshot, IdentityService
from app.services.recognition_service import (
    PersonRecognitionProfile,
    RecognitionService,
)

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


# Classification outcomes.
KIND_AUTO = "auto"
KIND_SUGGEST = "suggest"
KIND_NONE = "none"


@dataclass
class FaceData:
    """Detached snapshot of one candidate Unknown face (thread-safe to score)."""

    face_id: int
    image_id: Optional[int]
    crop_path: Optional[str]
    embedding: Optional[np.ndarray]
    prev_person_id: Optional[int]
    prev_person_name: Optional[str]
    prev_person_was_auto: bool
    prev_assignment_source: Optional[str]
    prev_assignment_confidence: Optional[float]
    prev_assigned_at: Optional[datetime]


@dataclass
class Candidate:
    """One ranked person match for a face."""

    person_id: int
    name: str
    score: float


@dataclass
class AutoItem:
    """A face confidently matched to a person — to be merged without asking."""

    face: FaceData
    target_person_id: int
    target_name: str
    score: float


@dataclass
class SuggestItem:
    """A face with plausible but uncertain matches — needs user review."""

    face: FaceData
    candidates: List[Candidate]


@dataclass
class ReRecognitionResult:
    """Outcome of one re-recognition run, handed back to the UI."""

    batch_id: Optional[str] = None
    n_examined: int = 0
    n_auto: int = 0
    suggest_items: List[SuggestItem] = field(default_factory=list)
    n_none: int = 0
    cancelled: bool = False

    # Outcome of the optional AI face-detection pass that piggybacks on
    # re-recognition (analysis only — boxes + confidences, no assignments).
    # Both stay None when the pass did not run (disabled / model unavailable),
    # keeping older consumers of this result untouched.
    ai_images_scanned: Optional[int] = None
    ai_faces_found: Optional[int] = None

    @property
    def n_suggest(self) -> int:
        return len(self.suggest_items)


@dataclass
class BatchSummary:
    """One row of the re-recognition history list."""

    batch_id: str
    created_at: datetime
    n_faces: int
    target_names: List[str]
    is_undone: bool


class ReRecognitionService:
    """Re-recognize Unknown faces and merge/suggest matches with undo support."""

    def __init__(
        self,
        session: Session,
        recognition_config: Optional[RecognitionConfig] = None,
        *,
        auto_threshold: Optional[float] = None,
        suggest_threshold: Optional[float] = None,
        margin: Optional[float] = None,
        max_candidates: int = 3,
    ) -> None:
        self._session = session
        self._rec_cfg = recognition_config or RecognitionConfig()
        self.auto_threshold = (
            auto_threshold
            if auto_threshold is not None
            else self._rec_cfg.rerecognition_auto_threshold
        )
        self.suggest_threshold = (
            suggest_threshold
            if suggest_threshold is not None
            else self._rec_cfg.rerecognition_suggest_threshold
        )
        self.margin = margin if margin is not None else self._rec_cfg.min_margin
        self.max_candidates = max(1, max_candidates)
        # Scorer reused across all faces in a run; rank_persons only reads the
        # profiles passed to it, so a single instance is fine for the whole batch.
        self._rec = RecognitionService(session, self._rec_cfg)

    # ------------------------------------------------------------------
    # Loading (require a session)
    # ------------------------------------------------------------------

    def load_profiles(self) -> Dict[int, PersonRecognitionProfile]:
        """Build the named-person profiles used for scoring (one matmul source)."""
        return self._rec.build_profiles()

    def extract_candidates(self, image_ids: Sequence[int]) -> List[FaceData]:
        """Snapshot the unresolved, embedded faces on *image_ids*.

        "Unresolved" mirrors the automatic recognizer's candidate rule
        (:meth:`RecognitionService._load_candidate_faces`): a face is eligible
        when it is not excluded, has an embedding, and is either unassigned or
        belongs to an auto-named / protected person.  Faces already on a named
        person are never touched.
        """
        ids = [int(i) for i in image_ids if i is not None]
        if not ids:
            return []

        faces: List[Face] = (
            self._session.query(Face)
            .filter(Face.image_id.in_(ids))
            .filter(Face._embedding.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
            .all()
        )

        out: List[FaceData] = []
        for face in faces:
            person = face.person
            if person is not None and not person.is_auto_named and not person.is_protected:
                # Already a named identity — leave it alone.
                continue
            embedding = face.get_embedding()
            if embedding is None:
                continue
            out.append(
                FaceData(
                    face_id=face.id,
                    image_id=face.image_id,
                    crop_path=face.crop_path,
                    embedding=np.asarray(embedding, dtype=np.float32),
                    prev_person_id=face.person_id,
                    prev_person_name=person.name if person is not None else None,
                    prev_person_was_auto=bool(person.is_auto_named) if person else False,
                    prev_assignment_source=face.assignment_source,
                    prev_assignment_confidence=face.assignment_confidence,
                    prev_assigned_at=face.assigned_at,
                )
            )
        return out

    def load_rejected_targets(self, faces: Sequence[FaceData]) -> Dict[int, Set[int]]:
        """Person ids each face must not match (from negative FaceCorrections)."""
        if not faces:
            return {}
        # Reuse the recognizer's correction-aware loader; it expects ORM faces.
        face_ids = [f.face_id for f in faces]
        orm_faces = (
            self._session.query(Face).filter(Face.id.in_(face_ids)).all()
        )
        return self._rec._load_rejected_face_targets(orm_faces)

    # ------------------------------------------------------------------
    # Classification (pure — no session, safe to run in a thread pool)
    # ------------------------------------------------------------------

    def classify(
        self,
        face: FaceData,
        profiles: Dict[int, PersonRecognitionProfile],
        rejected_targets: Optional[Dict[int, Set[int]]] = None,
    ) -> "tuple[str, Optional[object]]":
        """Classify one face into AUTO / SUGGEST / NONE.

        Returns ``(kind, item)`` where *item* is an :class:`AutoItem` for
        ``KIND_AUTO``, a :class:`SuggestItem` for ``KIND_SUGGEST``, or ``None``
        for ``KIND_NONE``.
        """
        ranked = self._rec.rank_persons(face.embedding, profiles)
        blocked = (rejected_targets or {}).get(face.face_id, set())
        ranked = [(pid, name, score) for (pid, name, score) in ranked if pid not in blocked]

        if not ranked:
            return KIND_NONE, None

        best_pid, best_name, best_score = ranked[0]
        second_score = ranked[1][2] if len(ranked) > 1 else 0.0
        margin_ok = (best_score - second_score) >= self.margin

        if best_score >= self.auto_threshold and margin_ok:
            return KIND_AUTO, AutoItem(
                face=face,
                target_person_id=best_pid,
                target_name=best_name,
                score=best_score,
            )

        if best_score >= self.suggest_threshold:
            candidates = [
                Candidate(person_id=pid, name=name, score=score)
                for (pid, name, score) in ranked[: self.max_candidates]
            ]
            return KIND_SUGGEST, SuggestItem(face=face, candidates=candidates)

        return KIND_NONE, None

    # ------------------------------------------------------------------
    # Applying merges (require a session; commit on success)
    # ------------------------------------------------------------------

    def apply_auto_merges(
        self, auto_items: Sequence[AutoItem], batch_id: Optional[str] = None
    ) -> str:
        """Merge every AUTO item into its matched person and log each move.

        Faces are grouped by target person and moved with
        :meth:`IdentityService.reassign_faces_bulk` (which cleans up emptied
        Unknown clusters), then one :class:`RecognitionMergeLog` row is written
        per moved face so the run can be audited and undone.

        Returns the ``batch_id`` shared by all rows of this run.
        """
        batch_id = batch_id or uuid.uuid4().hex
        if not auto_items:
            return batch_id

        by_target: Dict[int, List[AutoItem]] = {}
        for item in auto_items:
            by_target.setdefault(item.target_person_id, []).append(item)

        identity = IdentityService(self._session)
        now = _utcnow_naive()

        for target_id, items in by_target.items():
            item_by_face = {it.face.face_id: it for it in items}
            result = identity.reassign_faces_bulk(
                list(item_by_face.keys()), target_id
            )
            moved = set(result.moved_face_ids)
            for face_id, item in item_by_face.items():
                if face_id not in moved:
                    continue  # skipped (already on target / vanished)
                self._session.add(
                    RecognitionMergeLog(
                        batch_id=batch_id,
                        created_at=now,
                        face_id=face_id,
                        image_id=item.face.image_id,
                        prev_person_id=item.face.prev_person_id,
                        prev_person_name=item.face.prev_person_name,
                        prev_assignment_source=item.face.prev_assignment_source,
                        prev_assignment_confidence=item.face.prev_assignment_confidence,
                        prev_assigned_at=item.face.prev_assigned_at,
                        prev_person_was_auto=item.face.prev_person_was_auto,
                        matched_person_id=target_id,
                        matched_person_name=item.target_name,
                        score=item.score,
                        decision=KIND_AUTO,
                    )
                )
                log.info(
                    "Re-recognition: face %d %r→%r score=%.3f (batch %s)",
                    face_id, item.face.prev_person_name, item.target_name,
                    item.score, batch_id,
                )
        self._session.commit()
        return batch_id

    def apply_user_decision(
        self,
        face: FaceData,
        target_person_id: int,
        target_name: str,
        score: float,
        batch_id: str,
    ) -> None:
        """Merge a single reviewed face into the user-chosen person and log it."""
        identity = IdentityService(self._session)
        result = identity.reassign_faces_bulk([face.face_id], target_person_id)
        if face.face_id not in set(result.moved_face_ids):
            return
        self._session.add(
            RecognitionMergeLog(
                batch_id=batch_id,
                created_at=_utcnow_naive(),
                face_id=face.face_id,
                image_id=face.image_id,
                prev_person_id=face.prev_person_id,
                prev_person_name=face.prev_person_name,
                prev_assignment_source=face.prev_assignment_source,
                prev_assignment_confidence=face.prev_assignment_confidence,
                prev_assigned_at=face.prev_assigned_at,
                prev_person_was_auto=face.prev_person_was_auto,
                matched_person_id=target_person_id,
                matched_person_name=target_name,
                score=score,
                decision="manual",
            )
        )
        self._session.commit()
        log.info(
            "Re-recognition (review): face %d %r→%r score=%.3f (batch %s)",
            face.face_id, face.prev_person_name, target_name, score, batch_id,
        )

    # ------------------------------------------------------------------
    # History + undo
    # ------------------------------------------------------------------

    def list_batches(self) -> List[BatchSummary]:
        """Summarise every re-recognition batch, newest first."""
        rows: List[RecognitionMergeLog] = (
            self._session.query(RecognitionMergeLog)
            .order_by(RecognitionMergeLog.created_at.desc())
            .all()
        )
        order: List[str] = []
        grouped: Dict[str, List[RecognitionMergeLog]] = {}
        for row in rows:
            if row.batch_id not in grouped:
                grouped[row.batch_id] = []
                order.append(row.batch_id)
            grouped[row.batch_id].append(row)

        summaries: List[BatchSummary] = []
        for batch_id in order:
            batch = grouped[batch_id]
            names: List[str] = []
            for r in batch:
                if r.matched_person_name and r.matched_person_name not in names:
                    names.append(r.matched_person_name)
            summaries.append(
                BatchSummary(
                    batch_id=batch_id,
                    created_at=batch[0].created_at,
                    n_faces=len(batch),
                    target_names=names,
                    is_undone=all(r.undone_at is not None for r in batch),
                )
            )
        return summaries

    def get_batch_rows(self, batch_id: str) -> List[RecognitionMergeLog]:
        """Return the audit rows of one batch, newest first."""
        return (
            self._session.query(RecognitionMergeLog)
            .filter(RecognitionMergeLog.batch_id == batch_id)
            .order_by(RecognitionMergeLog.created_at.desc())
            .all()
        )

    def latest_undoable_batch(self) -> Optional[str]:
        """The most recent batch that still has un-undone rows, if any."""
        row: Optional[RecognitionMergeLog] = (
            self._session.query(RecognitionMergeLog)
            .filter(RecognitionMergeLog.undone_at.is_(None))
            .order_by(RecognitionMergeLog.created_at.desc())
            .first()
        )
        return row.batch_id if row is not None else None

    def undo_batch(self, batch_id: str) -> int:
        """Undo every still-applied merge in *batch_id*.

        Restores each face to its logged previous assignment and recreates any
        auto-named Unknown cluster that the merge emptied (and cleanup deleted).

        Returns the number of faces restored.
        """
        rows: List[RecognitionMergeLog] = (
            self._session.query(RecognitionMergeLog)
            .filter(RecognitionMergeLog.batch_id == batch_id)
            .filter(RecognitionMergeLog.undone_at.is_(None))
            .all()
        )
        if not rows:
            return 0

        snapshots = [
            FaceAssignmentSnapshot(
                face_id=r.face_id,
                person_id=r.prev_person_id,
                assignment_source=r.prev_assignment_source,
                assignment_confidence=r.prev_assignment_confidence,
                assigned_at=r.prev_assigned_at,
            )
            for r in rows
            if r.face_id is not None
        ]

        # Recreate emptied Unknown clusters: only auto-named source persons that
        # no longer exist need rebuilding (restore_face_assignments skips ids
        # that still exist).
        removed_persons: List[dict] = []
        seen_pids: Set[int] = set()
        for r in rows:
            pid = r.prev_person_id
            if (
                pid is None
                or pid in seen_pids
                or not r.prev_person_was_auto
            ):
                continue
            seen_pids.add(pid)
            removed_persons.append(
                {
                    "id": pid,
                    "name": r.prev_person_name or f"Unknown {pid}",
                    "is_auto_named": True,
                    "is_protected": False,
                }
            )

        restored = IdentityService(self._session).restore_face_assignments(
            snapshots, removed_persons
        )

        now = _utcnow_naive()
        for r in rows:
            r.undone_at = now
        self._session.commit()
        log.info("Undid re-recognition batch %s (%d face(s))", batch_id, restored)
        return restored
