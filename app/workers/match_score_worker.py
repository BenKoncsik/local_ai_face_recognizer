"""Background face-match scoring using QRunnable + QThreadPool.

Computing match scores deserialises every trusted training embedding, which can
take seconds on a large database.  Running that on the GUI thread freezes the
whole app the moment the user selects a face, so this runnable does the work in
a thread-pool worker and emits the result for the GUI thread to apply.

Usage
-----
    worker = MatchScoreRunnable(face_id, recognition_cfg, config)
    worker.signals.ready.connect(my_slot)     # (face_id, {person_id: score})
    worker.signals.failed.connect(my_err)     # (face_id,)
    QThreadPool.globalInstance().start(worker)

The DB engine is opened with ``check_same_thread=False`` and WAL mode, so the
worker safely opens its own read-only ``session_scope`` on the pool thread.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

log = logging.getLogger(__name__)


class _MatchScoreSignals(QObject):
    """Signal carrier — must be a QObject, created before the runnable starts."""

    ready = Signal(int, object)   # face_id, {person_id: score}
    failed = Signal(int)          # face_id


class _BatchMatchScoreSignals(QObject):
    """Signal carrier for the batch runnable."""

    ready = Signal(object)  # {person_id: score} — {} on failure


class BatchMatchScoreRunnable(QRunnable):
    """Score one or more faces (averaged) against known people off-thread.

    Used by modal person pickers: the dialog opens instantly with a
    "computing…" hint and re-ranks itself when this emits.  Always emits
    ``ready`` (with ``{}`` on any failure) so the pending hint never sticks.
    """

    def __init__(self, face_ids, recognition_cfg, config) -> None:
        super().__init__()
        self.signals = _BatchMatchScoreSignals()
        self._face_ids = list(face_ids)
        self._recognition_cfg = recognition_cfg
        self._config = config
        self.setAutoDelete(True)

    def run(self) -> None:
        scores = {}
        try:
            from app.db.database import session_scope
            from app.services.match_scoring import (
                match_scores_for_face,
                match_scores_for_faces,
            )

            with session_scope() as session:
                if len(self._face_ids) == 1:
                    scores = match_scores_for_face(
                        session,
                        self._face_ids[0],
                        self._recognition_cfg,
                        config=self._config,
                    )
                elif self._face_ids:
                    scores = match_scores_for_faces(
                        session,
                        self._face_ids,
                        self._recognition_cfg,
                        config=self._config,
                    )
        except Exception as exc:  # noqa: BLE001 — ranking is best-effort
            log.warning(
                "Batch match scoring failed for faces %s: %s", self._face_ids, exc
            )
        self.signals.ready.emit(scores or {})


class PersonMatchScoreRunnable(QRunnable):
    """Score everyone against a merge-source person off the GUI thread.

    Same contract as :class:`BatchMatchScoreRunnable`: always emits ``ready``
    (``{}`` on failure) so a picker's "computing…" hint never sticks.
    """

    def __init__(self, person_id: int, recognition_cfg, config) -> None:
        super().__init__()
        self.signals = _BatchMatchScoreSignals()
        self._person_id = person_id
        self._recognition_cfg = recognition_cfg
        self._config = config
        self.setAutoDelete(True)

    def run(self) -> None:
        scores = {}
        try:
            from app.db.database import session_scope
            from app.db.models import Person
            from app.services.match_scoring import match_scores_for_person

            with session_scope() as session:
                person = session.get(Person, self._person_id)
                if person is not None:
                    scores = match_scores_for_person(
                        session,
                        person,
                        self._recognition_cfg,
                        config=self._config,
                    )
        except Exception as exc:  # noqa: BLE001 — ranking is best-effort
            log.warning(
                "Person match scoring failed for person %s: %s",
                self._person_id, exc,
            )
        self.signals.ready.emit(scores or {})


class MatchScoreRunnable(QRunnable):
    """Score one face against known people off the GUI thread."""

    def __init__(self, face_id: int, recognition_cfg, config) -> None:
        super().__init__()
        self.signals = _MatchScoreSignals()
        self._face_id = face_id
        self._recognition_cfg = recognition_cfg
        self._config = config
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            from app.db.database import session_scope
            from app.services.match_scoring import match_scores_for_face

            with session_scope() as session:
                scores = match_scores_for_face(
                    session,
                    self._face_id,
                    self._recognition_cfg,
                    config=self._config,
                )
            self.signals.ready.emit(self._face_id, scores or {})
        except Exception as exc:  # noqa: BLE001 — ranking is best-effort, never fatal
            log.warning("Match scoring failed for face %s: %s", self._face_id, exc)
            self.signals.failed.emit(self._face_id)
