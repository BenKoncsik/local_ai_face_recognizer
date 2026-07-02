"""Merge persons dialog.

Built on the shared :class:`PersonSearchSelect`.  Match scores can be supplied
pre-computed (*match_scores*) or — preferred on large databases — computed in
the background by passing *score_person_id* (score everyone against the merge
source) or *score_face_ids* (score a face being reassigned): the dialog opens
instantly with a "computing similarity…" hint and re-ranks itself when scoring
finishes.

Face counts next to the names come from *face_counts* (one aggregate COUNT
query at the call site).  When omitted, ``len(p.faces)`` is used, which lazy
loads every person's faces — avoid that on large databases.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect
from app.utils.person_search import PersonEntry, person_is_unknown


class MergeDialog(QDialog):
    """Dialog to merge the currently selected person into another person."""

    def __init__(
        self,
        source_person: Person,
        all_persons: List[Person],
        parent: Optional[QWidget] = None,
        match_scores: Optional[Dict[int, float]] = None,
        face_counts: Optional[Dict[int, int]] = None,
        score_person_id: Optional[int] = None,
        score_face_ids: Optional[Sequence[int]] = None,
        recognition_cfg=None,
        config=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("merge_into"))
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(t("merge_source_into", name=source_person.name))
        )

        def _count(p: Person) -> int:
            if face_counts is not None:
                return face_counts.get(p.id, 0)
            return len(p.faces)

        entries = [
            PersonEntry(
                person_id=p.id,
                name=p.name,
                display_text=t("merge_faces_count", name=p.name, n=_count(p)),
                is_unknown=person_is_unknown(p),
            )
            for p in all_persons
            if p.id != source_person.id
        ]

        self._selector = PersonSearchSelect()
        self._selector.set_entries(entries)
        if match_scores:
            # Open best-match-first: the most likely target persons rise to the
            # top of the list so the user does not have to hunt through hundreds
            # of names.  default_sort only flips this selector's checkbox; the
            # user can still uncheck it to fall back to alphabetical order.
            self._selector.set_match_scores(match_scores, default_sort=True)
        elif score_person_id is not None or score_face_ids:
            # Background scoring: open instantly with a "computing…" hint and
            # re-rank when the worker finishes (exec()'s event loop delivers it).
            self._selector.set_scores_pending()
            self._start_background_scoring(
                score_person_id, score_face_ids, recognition_cfg, config
            )
        # Accept the dialog when the user double-clicks a person in the list.
        # PersonSearchSelect emits person_double_clicked after the selection
        # has already been committed, so accepting here behaves like clicking
        # OK on a highlighted entry.
        self._selector.person_double_clicked.connect(self.accept)

        layout.addWidget(self._selector)
        layout.addWidget(
            QLabel(f"<small>{t('merge_source_deleted')}</small>")
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _start_background_scoring(
        self,
        score_person_id: Optional[int],
        score_face_ids: Optional[Sequence[int]],
        recognition_cfg,
        config,
    ) -> None:
        from app.workers.match_score_worker import (
            BatchMatchScoreRunnable,
            PersonMatchScoreRunnable,
        )

        if score_person_id is not None:
            worker = PersonMatchScoreRunnable(
                score_person_id, recognition_cfg, config
            )
        else:
            worker = BatchMatchScoreRunnable(
                list(score_face_ids or []), recognition_cfg, config
            )
        worker.signals.ready.connect(self._on_scores_ready)
        QThreadPool.globalInstance().start(worker)

    def _on_scores_ready(self, scores: object) -> None:
        try:
            self._selector.set_match_scores(scores or {}, default_sort=True)
        except RuntimeError:
            # Dialog already closed/deleted before scoring finished — fine.
            pass

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._selector.focus_search)

    def target_person_id(self) -> Optional[int]:
        """Return the selected target person ID, or ``None`` if empty."""
        return self._selector.current_person_id()
