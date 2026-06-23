"""Reviewable auto-merge of an Unknown cluster when one of its faces is named.

When the user manually assigns **one** face of an auto-named ``Unknown N``
cluster to a real, named person, that single face is a confident, human
judgement — but the *rest* of the cluster may or may not be the same person
(small clusters usually are; large clusters often mix several people).

:class:`UnknownMergeService` therefore moves the whole cluster onto the target
person but flags every *dragged-along* face as **pending review** (see the
``auto_merge_*`` columns on :class:`app.db.models.Face`).  The manually picked
face is never flagged.  Pending faces can later be confirmed, re-assigned or
deleted — individually in the image browser / face view, or in bulk from the
dedicated review dialog.

Nothing here is a parallel implementation of move/merge/delete: the actual face
moves go through :class:`app.services.identity_service.IdentityService`
(``reassign_faces_bulk`` / ``reassign_face``, which already clean up emptied
Unknown clusters and clear the auto-merge markers on a manual override), and the
intelligent auto-confirm reuses
:class:`app.services.vector_scoring.FaceVectorScorer` scoring.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.config import RecognitionConfig
from app.db.models import Face, Person
from app.services.identity_service import (
    BulkReassignResult,
    IdentityService,
)
from app.services.vector_scoring import FaceVectorScorer

log = logging.getLogger(__name__)

# Marker value written to ``Face.auto_merge_review_status`` while a dragged-along
# face awaits review.
REVIEW_PENDING = "pending"
# ``assignment_source`` stamped on auto-moved (still-pending) sibling faces, so
# they are distinguishable from a deliberate manual assignment and are NOT used
# as recognition training examples (not in ``_TRUSTED_MANUAL_SOURCES``).
SOURCE_AUTO_MERGE = "auto_merge_unknown"

# Schema version + rule id of the serialized decision graph stored on
# ``Face.auto_merge_decision_json``.  Bump the version if the structure changes.
DECISION_VERSION = 1
RULE_UNKNOWN_SCATTER = "unknown_cluster_scatter"
# How many ranked persons to keep in the stored decision (keeps the JSON small).
_MAX_RANKED = 8


def _utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass
class UnknownMergeResult:
    """Outcome of :meth:`UnknownMergeService.assign_unknown_face`."""

    selected_face_id: int
    target_person_id: int
    target_name: str
    # Source ``Unknown N`` person id, or None when the selected face was not in
    # an auto-named cluster (then this was a plain single-face reassignment).
    source_person_id: Optional[int] = None
    source_person_name: Optional[str] = None
    was_unknown_cluster: bool = False
    # Sibling faces moved along with the selected one.
    n_moved_siblings: int = 0
    # Siblings still flagged "pending" after the optional auto-confirm pass.
    n_pending: int = 0
    # Siblings auto-confirmed (high similarity) — flag already removed.
    n_auto_confirmed: int = 0
    # Undo payload from the underlying bulk move (None for the simple path).
    bulk_result: Optional[BulkReassignResult] = None


@dataclass
class PendingAutoMerge:
    """One row of the auto-merge review list."""

    face_id: int
    crop_path: Optional[str]
    person_id: Optional[int]
    person_name: Optional[str]
    source_person_id: Optional[int]
    source_person_name: str
    image_id: Optional[int]
    image_path: Optional[str]
    assigned_at: Optional[datetime]
    # (x, y, w, h) of the face on the original image — used to highlight the
    # exact face the suggestion was based on in the full-image viewer.
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # Best similarity score the merge engine measured for this face against the
    # target (from the stored decision), or None when unknown.
    confidence: Optional[float] = None
    # Parsed decision graph (``Face.auto_merge_decision_json``) or None for rows
    # merged before the decision was recorded.
    decision: Optional[Dict[str, Any]] = None


class UnknownMergeService:
    """Scatter an Unknown cluster onto a named person with reviewable flags."""

    def __init__(
        self,
        session: Session,
        recognition_config: Optional[RecognitionConfig] = None,
    ) -> None:
        self._session = session
        self._rec_cfg = recognition_config or RecognitionConfig()
        self._identity = IdentityService(session)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def assign_unknown_face(
        self, selected_face_id: int, target_person_id: int
    ) -> UnknownMergeResult:
        """Assign *selected_face_id* to *target_person_id*, scattering its cluster.

        If the selected face belongs to an auto-named ``Unknown N`` cluster, the
        cluster's other faces are moved to the same target but flagged pending
        review.  The selected face is moved as a plain manual assignment.  When
        the selected face is unassigned, on the protected catch-all, or already
        on a named person, this degrades to a single :meth:`reassign_face` — the
        previous behaviour — with no scatter.

        Returns an :class:`UnknownMergeResult` describing what happened.
        """
        face = self._session.get(Face, selected_face_id)
        if face is None:
            raise ValueError(f"Face id={selected_face_id} not found")
        target = self._session.get(Person, target_person_id)
        if target is None:
            raise ValueError(f"Person id={target_person_id} not found")

        source = face.person
        # Only auto-named "Unknown N" clusters are scattered.  The protected
        # "Ismeretlen" catch-all deliberately mixes many people, and unassigned
        # faces have no cluster — both take the plain single-face path.
        is_unknown_cluster = (
            source is not None
            and source.is_auto_named
            and not source.is_protected
        )

        if not is_unknown_cluster:
            self._identity.reassign_face(selected_face_id, target_person_id)
            return UnknownMergeResult(
                selected_face_id=selected_face_id,
                target_person_id=target_person_id,
                target_name=target.name,
                was_unknown_cluster=False,
            )

        source_id = source.id
        source_name = source.name

        # Siblings = the cluster's other usable faces, minus merge-excluded ones
        # (those intentionally stay behind, mirroring merge_persons' exclusion).
        sibling_ids = [
            f.id
            for f in self._identity.get_faces_for_person(source_id)
            if f.id != selected_face_id and not f.is_merge_excluded
        ]

        # One transactional move for the whole cluster.  This sets every moved
        # face to assignment_source="manual" and clears any prior auto markers,
        # and cleans up the emptied Unknown cluster.
        bulk = self._identity.reassign_faces_bulk(
            [selected_face_id] + sibling_ids, target_person_id
        )
        moved = set(bulk.moved_face_ids)

        # Stamp the dragged-along siblings (everything moved except the face the
        # user positively identified) as pending review.
        now = _utcnow_naive()
        flagged_ids: List[int] = []
        for fid in moved:
            if fid == selected_face_id:
                continue
            sibling = self._session.get(Face, fid)
            if sibling is None:
                continue
            sibling.auto_merged_from_unknown = True
            sibling.auto_merge_review_status = REVIEW_PENDING
            sibling.auto_merge_source_person_id = source_id
            sibling.auto_merge_confirmed_at = None
            sibling.auto_merge_confirmed_by_user = False
            sibling.assignment_source = SOURCE_AUTO_MERGE
            sibling.assignment_confidence = None
            sibling.assigned_at = now
            flagged_ids.append(fid)
        self._session.commit()

        # Always record a decision graph for every dragged-along face — even
        # when auto-merge is disabled or the target has no profile — so the
        # review dialog can later explain *why* each face was suggested.  The
        # same pass auto-confirms the confident ones when enabled.
        n_auto = 0
        if flagged_ids:
            n_auto = self._evaluate_and_record(
                flagged_ids,
                target_person_id,
                context={
                    "selected_face_id": selected_face_id,
                    "source_person_id": source_id,
                    "source_person_name": source_name,
                    "cluster_size": len(flagged_ids) + 1,
                },
            )

        result = UnknownMergeResult(
            selected_face_id=selected_face_id,
            target_person_id=target_person_id,
            target_name=target.name,
            source_person_id=source_id,
            source_person_name=source_name,
            was_unknown_cluster=True,
            n_moved_siblings=len(flagged_ids),
            n_pending=len(flagged_ids) - n_auto,
            n_auto_confirmed=n_auto,
            bulk_result=bulk,
        )
        log.info(
            "Unknown-merge: face %d + %d sibling(s) %r→%r (%d auto-confirmed, "
            "%d pending)",
            selected_face_id, len(flagged_ids), source_name, target.name,
            n_auto, result.n_pending,
        )
        return result

    # ------------------------------------------------------------------
    # Intelligent auto-confirm
    # ------------------------------------------------------------------

    def maybe_auto_confirm(
        self, face_ids: List[int], target_person_id: int
    ) -> int:
        """Record decision graphs and auto-confirm the confident pending faces.

        A pending face is auto-confirmed only when **all** hold:

        * auto-merge is enabled in the config;
        * the target person has at least one confirmed reference face (i.e. it
          has a recognition profile — pending faces are excluded from profiles);
        * the target is the *best* match for the face;
        * the similarity is ``>= unknown_auto_confirm_threshold``;
        * the margin to the second-best person is ``>= min_margin`` (no strong
          conflict with another known identity).

        Every evaluated face also gets its decision graph stored on
        ``Face.auto_merge_decision_json``.  Returns the number auto-confirmed.
        """
        return self._evaluate_and_record(face_ids, target_person_id)

    def _evaluate_and_record(
        self,
        face_ids: List[int],
        target_person_id: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Score each pending face against the target, persist its decision
        graph, and auto-confirm the confident matches.

        Decisions are recorded for **all** evaluated faces — including when
        auto-merge is disabled or the target has no profile (those simply get
        ``outcome="pending"``).  Returns the number of faces auto-confirmed.
        """
        if not face_ids:
            return 0

        enabled = self._rec_cfg.unknown_auto_merge_enabled
        threshold = self._rec_cfg.unknown_auto_confirm_threshold
        margin = self._rec_cfg.min_margin

        rec = FaceVectorScorer(self._session, self._rec_cfg)
        profiles = rec.build_profiles()
        target_has_profile = target_person_id in profiles
        target = self._session.get(Person, target_person_id)
        target_name = target.name if target is not None else None

        confirmed = 0
        dirty = False
        for fid in face_ids:
            face = self._session.get(Face, fid)
            if face is None or face.auto_merge_review_status != REVIEW_PENDING:
                continue
            embedding = face.get_embedding()
            ranked = (
                rec.rank_persons(embedding, profiles)
                if embedding is not None
                else []
            )

            best_pid = ranked[0][0] if ranked else None
            best_score = ranked[0][2] if ranked else None
            second_score = ranked[1][2] if len(ranked) > 1 else 0.0

            passes = bool(
                enabled
                and target_has_profile
                and best_pid == target_person_id
                and best_score is not None
                and best_score >= threshold
                and (best_score - second_score) >= margin
            )
            outcome = "auto_confirmed" if passes else "pending"

            decision = self._build_decision(
                face=face,
                target_person_id=target_person_id,
                target_name=target_name,
                ranked=ranked,
                best_score=best_score,
                second_score=second_score,
                threshold=threshold,
                margin=margin,
                enabled=enabled,
                target_has_profile=target_has_profile,
                outcome=outcome,
                context=context,
            )
            face.auto_merge_decision_json = json.dumps(decision, ensure_ascii=False)
            dirty = True

            if passes:
                self._apply_confirm(face, by_user=False, score=best_score)
                confirmed += 1

        if dirty:
            self._session.commit()
        if confirmed:
            log.info(
                "Unknown-merge auto-confirmed %d face(s) for person %d",
                confirmed, target_person_id,
            )
        return confirmed

    def _build_decision(
        self,
        *,
        face: Face,
        target_person_id: int,
        target_name: Optional[str],
        ranked: List[Tuple[int, str, float]],
        best_score: Optional[float],
        second_score: float,
        threshold: float,
        margin: float,
        enabled: bool,
        target_has_profile: bool,
        outcome: str,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the serializable decision graph for one pending face.

        The ``gates`` list mirrors the open-set rejection gates of the AI debug
        view (``app.deep.debug_info.GateResult``) so the review dialog can
        render the exact same flow graph: *is the target the best match → is it
        above the confirm threshold → is the margin to the runner-up wide
        enough*.  The first failing gate is where auto-confirmation stopped.
        """
        ctx = context or {}
        best_pid = ranked[0][0] if ranked else None
        target_is_best = best_pid == target_person_id and best_pid is not None

        gates: List[Dict[str, Any]] = [
            {
                # Target is the single best match in embedding space.
                "name": "best_match",
                "passed": bool(target_is_best),
                "value": float(best_score) if (target_is_best and best_score is not None) else 0.0,
                "threshold": float(second_score),
            },
            {
                # Similarity to the target clears the confirm threshold.
                "name": "sim_floor",
                "passed": bool(best_score is not None and best_score >= threshold),
                "value": float(best_score) if best_score is not None else 0.0,
                "threshold": float(threshold),
            },
            {
                # Margin to the runner-up is wide enough (no rival identity).
                "name": "margin",
                "passed": bool(
                    best_score is not None and (best_score - second_score) >= margin
                ),
                "value": float(best_score - second_score) if best_score is not None else 0.0,
                "threshold": float(margin),
            },
        ]

        return {
            "version": DECISION_VERSION,
            "rule": RULE_UNKNOWN_SCATTER,
            "face_id": face.id,
            "selected_face_id": ctx.get("selected_face_id"),
            "source_person_id": (
                ctx.get("source_person_id")
                if ctx.get("source_person_id") is not None
                else face.auto_merge_source_person_id
            ),
            "source_person_name": ctx.get("source_person_name"),
            "cluster_size": ctx.get("cluster_size"),
            "target_person_id": target_person_id,
            "target_person_name": target_name,
            "ranked": [
                [int(pid), str(name), float(score)]
                for pid, name, score in ranked[:_MAX_RANKED]
            ],
            "best_person_id": int(best_pid) if best_pid is not None else None,
            "best_score": float(best_score) if best_score is not None else None,
            "second_score": float(second_score),
            "threshold": float(threshold),
            "min_margin": float(margin),
            "auto_merge_enabled": bool(enabled),
            "target_has_profile": bool(target_has_profile),
            "gates": gates,
            "outcome": outcome,
        }

    # ------------------------------------------------------------------
    # Per-face review operations
    # ------------------------------------------------------------------

    def confirm_auto_merge(self, face_id: int) -> Face:
        """User confirms a pending face — it becomes a normal manual assignment."""
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError(f"Face id={face_id} not found")
        self._apply_confirm(face, by_user=True, score=None)
        self._session.commit()
        log.info("Unknown-merge: face %d confirmed by user", face_id)
        return face

    def move_auto_merge(self, face_id: int, new_person_id: int) -> Face:
        """Re-assign a pending face to another person (clears the pending flag)."""
        # reassign_face stamps "manual" and clears the auto-merge markers.
        return self._identity.reassign_face(face_id, new_person_id)

    def delete_face(self, face_id: int) -> bool:
        """Hard-delete a face row together with its bbox (matches the existing
        single-face delete used by the preview / image browser).

        Returns True when a row was removed.
        """
        face = self._session.get(Face, face_id)
        if face is None:
            return False
        self._session.delete(face)
        self._session.commit()
        self._identity.cleanup_empty_unknown_persons()
        log.info("Unknown-merge: face %d deleted (pending review)", face_id)
        return True

    def _apply_confirm(
        self, face: Face, *, by_user: bool, score: Optional[float]
    ) -> None:
        """Clear the pending markers and turn *face* into a confirmed assignment."""
        face.auto_merged_from_unknown = False
        face.auto_merge_review_status = None
        face.auto_merge_confirmed_at = _utcnow_naive()
        face.auto_merge_confirmed_by_user = by_user
        # A user confirmation is a deliberate manual assignment; an auto-confirm
        # is treated like a merge — both are trusted recognition references.
        face.assignment_source = "manual" if by_user else "manual_merge"
        face.assignment_confidence = None if by_user else score
        face.assigned_at = _utcnow_naive()

    # ------------------------------------------------------------------
    # Listing / counting (review dialog + badges)
    # ------------------------------------------------------------------

    def count_pending(self) -> int:
        """Number of faces currently flagged pending review."""
        return (
            self._session.query(Face)
            .filter(Face.auto_merge_review_status == REVIEW_PENDING)
            .count()
        )

    def pending_face_ids(self, image_id: Optional[int] = None) -> List[int]:
        """Ids of pending faces, optionally restricted to one image."""
        q = self._session.query(Face.id).filter(
            Face.auto_merge_review_status == REVIEW_PENDING
        )
        if image_id is not None:
            q = q.filter(Face.image_id == image_id)
        return [row[0] for row in q.all()]

    def list_pending(self) -> List[PendingAutoMerge]:
        """All faces awaiting review, newest first, with display context."""
        faces: List[Face] = (
            self._session.query(Face)
            .filter(Face.auto_merge_review_status == REVIEW_PENDING)
            .order_by(Face.assigned_at.desc(), Face.id.desc())
            .all()
        )
        # Resolve original-Unknown names in one pass; emptied clusters are gone,
        # so fall back to a synthetic "Unknown #<id>" label.
        src_ids = {
            f.auto_merge_source_person_id
            for f in faces
            if f.auto_merge_source_person_id is not None
        }
        src_names = {
            p.id: p.name
            for p in (
                self._session.query(Person).filter(Person.id.in_(src_ids)).all()
                if src_ids
                else []
            )
        }

        rows: List[PendingAutoMerge] = []
        for f in faces:
            sid = f.auto_merge_source_person_id
            src_name = (
                src_names.get(sid)
                if sid is not None
                else None
            ) or (f"Unknown #{sid}" if sid is not None else "Unknown")
            image_path = None
            if f.image is not None:
                image_path = f.image.file_path
            decision = self._parse_decision(f.auto_merge_decision_json)
            confidence = (
                decision.get("best_score") if decision is not None else None
            )
            rows.append(
                PendingAutoMerge(
                    face_id=f.id,
                    crop_path=f.crop_path,
                    person_id=f.person_id,
                    person_name=f.person.name if f.person is not None else None,
                    source_person_id=sid,
                    source_person_name=src_name,
                    image_id=f.image_id,
                    image_path=image_path,
                    assigned_at=f.assigned_at,
                    bbox=(f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h),
                    confidence=confidence,
                    decision=decision,
                )
            )
        return rows

    @staticmethod
    def _parse_decision(raw: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse a stored decision JSON, tolerating missing/corrupt values."""
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            log.warning("Unparseable auto_merge_decision_json: %r", raw[:120])
            return None
        return data if isinstance(data, dict) else None
