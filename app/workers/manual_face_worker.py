"""Background embedding for manually-marked faces.

Computing an embedding needs a loaded model and an inference pass — far too
slow for the GUI thread.  Marking a face used to block until the model loaded
and the vector was computed; now the face row + crop are saved instantly and
this runnable fills in the embedding afterwards, so the marked box appears the
moment the user releases the mouse.

The match-scoring path (`match_scores_for_face`) computes a missing embedding
on the fly, so opening the person picker before this worker finishes still
gets correct percentages — both paths share one embedder lock, so the vector
is only computed once.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

log = logging.getLogger(__name__)


class _ManualFaceEmbedSignals(QObject):
    """Signal carrier — must be a QObject, created before the runnable starts."""

    done = Signal(int, bool)  # face_id, success


class ManualFaceEmbedRunnable(QRunnable):
    """Compute and store one manually-marked face's embedding off-thread."""

    def __init__(self, face_id: int, config) -> None:
        super().__init__()
        self.signals = _ManualFaceEmbedSignals()
        self._face_id = face_id
        self._config = config
        self.setAutoDelete(True)

    def run(self) -> None:
        ok = False
        try:
            from app.db.database import session_scope
            from app.db.models import Face
            from app.services.embedding_service import embed_manual_face

            with session_scope() as session:
                face = session.get(Face, self._face_id)
                # Skip when already embedded (e.g. the person picker's lazy
                # embedding won the race) — no need to run inference twice.
                if face is not None and face.get_embedding() is None:
                    ok = embed_manual_face(session, face, self._config)
                elif face is not None:
                    ok = True
        except Exception as exc:  # noqa: BLE001 — embedding is best-effort
            log.warning(
                "Background embedding failed for manual face %s: %s",
                self._face_id, exc,
            )
        self.signals.done.emit(self._face_id, ok)
