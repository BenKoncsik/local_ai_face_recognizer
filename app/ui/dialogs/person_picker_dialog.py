"""Shared modal person picker used by every "assign/move to person" flow.

One dialog to replace the near-identical private pickers that used to live in
the auto-merge review, re-recognition review and family-tree editors (and the
body of the move-faces dialog).  Features — each optional, so every call site
gets exactly the surface it needs:

* searchable person list (:class:`PersonSearchSelect`) with keyboard support,
* face-match percentages, either supplied pre-computed (*match_scores*) or
  computed in the background (*score_face_ids*): the dialog opens instantly
  with a "computing similarity…" hint and re-ranks itself when scoring
  finishes — it never blocks the GUI thread on a large database,
* an optional "create new person" name field (*allow_create_new*),
* optional header / footer texts.

Results are read back via :meth:`selected_person_id`,
:meth:`selected_person_name` and :meth:`new_person_name`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person
from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect
from app.utils.person_search import PersonEntry

log = logging.getLogger(__name__)


class PersonPickerDialog(QDialog):
    """Modal person picker — the shared "assign to person" dialog."""

    def __init__(
        self,
        persons: Optional[List[Person]] = None,
        *,
        entries: Optional[List[PersonEntry]] = None,
        title: Optional[str] = None,
        header: Optional[str] = None,
        footer: Optional[str] = None,
        exclude_person_id: Optional[int] = None,
        match_scores: Optional[Dict[int, float]] = None,
        default_sort: bool = False,
        score_face_ids: Optional[Sequence[int]] = None,
        recognition_cfg=None,
        config=None,
        allow_create_new: bool = False,
        ok_label: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or t("ppd_title"))
        self.setMinimumWidth(380)

        self._selected_person_id: Optional[int] = None
        self._new_person_name: Optional[str] = None
        self._names: Dict[int, str] = {}
        self._default_sort = default_sort

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        if header:
            header_lbl = QLabel(header)
            header_lbl.setWordWrap(True)
            header_lbl.setStyleSheet("font-weight: bold;")
            layout.addWidget(header_lbl)

        self._selector = PersonSearchSelect(self)
        if entries is not None:
            self._selector.set_entries(entries)
            self._names = {e.person_id: e.name for e in entries}
        else:
            candidates = [
                p for p in (persons or []) if p.id != exclude_person_id
            ]
            self._selector.set_persons(candidates)
            self._names = {p.id: p.name for p in candidates}
        self._selector.person_selected.connect(self._on_person_selected)
        self._selector.person_double_clicked.connect(self._on_person_double_clicked)
        layout.addWidget(self._selector)

        if match_scores:
            self._selector.set_match_scores(match_scores, default_sort=default_sort)
        elif score_face_ids:
            # Background scoring: open instantly, show "computing…", re-rank on
            # arrival.  The modal exec() loop pumps Qt events, so the worker's
            # signal is delivered while the dialog is open.
            self._selector.set_scores_pending()
            self._start_background_scoring(
                list(score_face_ids), recognition_cfg, config
            )

        self._new_name: Optional[QLineEdit] = None
        if allow_create_new:
            line = QFrame()
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line)

            layout.addWidget(QLabel(t("ppd_create_new")))
            self._new_name = QLineEdit()
            self._new_name.setPlaceholderText(t("ppd_new_placeholder"))
            self._new_name.textChanged.connect(self._on_new_name_changed)
            layout.addWidget(self._new_name)

        if footer:
            layout.addWidget(QLabel(f"<small>{footer}</small>"))

        self._buttons = QDialogButtonBox()
        self._ok_btn = self._buttons.addButton(
            ok_label or t("ppd_select_btn"), QDialogButtonBox.AcceptRole
        )
        self._buttons.addButton(t("cancel"), QDialogButtonBox.RejectRole)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        self._ok_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Background scoring
    # ------------------------------------------------------------------

    def _start_background_scoring(
        self, face_ids: List[int], recognition_cfg, config
    ) -> None:
        from app.workers.match_score_worker import BatchMatchScoreRunnable

        worker = BatchMatchScoreRunnable(face_ids, recognition_cfg, config)
        worker.signals.ready.connect(self._on_scores_ready)
        QThreadPool.globalInstance().start(worker)

    def _on_scores_ready(self, scores: object) -> None:
        try:
            self._selector.set_match_scores(
                scores or {}, default_sort=self._default_sort
            )
        except RuntimeError:
            # The dialog (and its C++ widgets) may already be deleted when the
            # user closed it before scoring finished — a stale result is fine.
            pass

    # ------------------------------------------------------------------
    # Qt plumbing
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._selector.focus_search)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _on_person_selected(self, person_id: int) -> None:
        self._selected_person_id = person_id
        if self._new_name is not None and self._new_name.text().strip():
            self._new_name.clear()
        self._refresh_enabled()

    def _on_person_double_clicked(self, person_id: int) -> None:
        self._selected_person_id = person_id
        self._on_accept()

    def _on_new_name_changed(self, text: str) -> None:
        # Typing a new name takes precedence and clears any list selection.
        if text.strip():
            self._selector.clear_selection()
            self._selected_person_id = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        has_new = self._new_name is not None and bool(self._new_name.text().strip())
        self._ok_btn.setEnabled(has_new or self._selected_person_id is not None)

    def _on_accept(self) -> None:
        new_name = self._new_name.text().strip() if self._new_name is not None else ""
        if new_name:
            self._new_person_name = new_name
            self._selected_person_id = None
        elif self._selected_person_id is None:
            return  # nothing chosen; ignore the click
        self.accept()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def selected_person_id(self) -> Optional[int]:
        """The chosen existing person, or ``None`` (e.g. when creating a new one)."""
        return self._selected_person_id

    def selected_person_name(self) -> Optional[str]:
        """The chosen existing person's name, or ``None``."""
        if self._selected_person_id is None:
            return None
        return self._names.get(self._selected_person_id)

    def new_person_name(self) -> Optional[str]:
        """The name typed for a new person, or ``None`` if an existing one was picked."""
        return self._new_person_name
