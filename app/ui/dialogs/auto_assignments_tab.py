"""Review tab for the deep engine's automatic groupings.

Lists every face the last AI run placed with a person.  Each row shows the
face crop, where it came from, where it was put and the model's confidence.
The user can:

* **confirm** — the grouping is right; the face becomes trusted training data;
* **correct** — pick the right person (or create a new one); the engine
  records the mistake as a correction and learns from it;
* **revert** — send the face back to where it was before the run.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import (
    AUTO_ASSIGN_STATUS_CONFIRMED,
    AUTO_ASSIGN_STATUS_CORRECTED,
    AUTO_ASSIGN_STATUS_REVERTED,
    Person,
)
from app.services.deep_recognition_service import (
    AutoAssignmentDTO,
    DeepRecognitionService,
)
from app.ui.i18n import t

log = logging.getLogger(__name__)

_ROW_STYLE = "QFrame { background: #232323; border-radius: 6px; }"


class _DecidedAssignmentRow(QFrame):
    """One already-reviewed automatic grouping — display only."""

    def __init__(
        self, dto: AutoAssignmentDTO, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background: #1d1d1d; border-radius: 6px; }")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(8)

        from app.ui.dialogs.suggestion_dialog import _crop_pixmap

        crop = QLabel()
        crop.setFixedSize(36, 36)
        crop.setAlignment(Qt.AlignCenter)
        pixmap = _crop_pixmap(dto.crop_path)
        if pixmap is not None:
            crop.setPixmap(
                pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            crop.setStyleSheet("border: 1px solid #444; border-radius: 4px;")
        else:
            crop.setText("?")
            crop.setStyleSheet(
                "border: 1px solid #444; border-radius: 4px; color: #666;"
            )
        row.addWidget(crop)

        name = QLabel(f"→ {dto.person_name}")
        name.setStyleSheet("color: #ccc; border: none;")
        name.setWordWrap(True)
        row.addWidget(name, stretch=1)

        if dto.status == AUTO_ASSIGN_STATUS_CONFIRMED:
            status_text, color = t("autoAssign.status_confirmed"), "#88ee88"
        elif dto.status == AUTO_ASSIGN_STATUS_CORRECTED:
            status_text = t(
                "autoAssign.status_corrected",
                name=dto.corrected_person_name or "?",
            )
            color = "#88aaff"
        elif dto.status == AUTO_ASSIGN_STATUS_REVERTED:
            status_text, color = t("autoAssign.status_reverted"), "#ff8888"
        else:
            status_text, color = dto.status, "#aaaaaa"
        status = QLabel(status_text)
        status.setStyleSheet(
            f"color: {color}; font-weight: bold; border: none; font-size: 11px;"
        )
        row.addWidget(status)

        if dto.decided_at is not None:
            when = QLabel(dto.decided_at.strftime("%Y-%m-%d %H:%M"))
            when.setStyleSheet("color: #666; border: none; font-size: 10px;")
            row.addWidget(when)


class _AutoAssignmentRow(QFrame):
    """One reviewable automatic grouping."""

    confirmed = Signal(int)   # assignment_id
    corrected = Signal(int)   # assignment_id
    reverted = Signal(int)    # assignment_id

    def __init__(
        self, dto: AutoAssignmentDTO, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._dto = dto
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(_ROW_STYLE)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(8)

        # Reuse the suggestion dialog's clickable crop (hover zoom + full image).
        from app.ui.dialogs.suggestion_dialog import _ClickableCrop

        row.addWidget(
            _ClickableCrop(dto.crop_path, dto.image_path, dto.bbox, dto.face_id)
        )

        info = QVBoxLayout()
        info.setSpacing(2)

        target = QLabel(dto.person_name)
        target.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #eee; border: none;"
        )
        target.setWordWrap(True)
        info.addWidget(target)

        if dto.previous_person_name:
            origin = QLabel(
                t("autoAssign.from_group", name=dto.previous_person_name)
            )
        else:
            origin = QLabel(t("autoAssign.from_unassigned"))
        origin.setStyleSheet("font-size: 10px; color: #888; border: none;")
        origin.setWordWrap(True)
        info.addWidget(origin)
        row.addLayout(info, stretch=1)

        pct = int(round(max(0.0, min(1.0, dto.score)) * 100))
        score = QLabel(t("suggestions_similarity", pct=pct))
        score.setAlignment(Qt.AlignCenter)
        score.setFixedWidth(70)
        score.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #88ee88; border: none;"
        )
        row.addWidget(score)

        confirm_btn = QPushButton(t("autoAssign.confirm"))
        confirm_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        confirm_btn.setToolTip(t("autoAssign.confirm_tooltip"))
        confirm_btn.clicked.connect(lambda: self.confirmed.emit(dto.assignment_id))
        row.addWidget(confirm_btn)

        correct_btn = QPushButton(t("autoAssign.correct"))
        correct_btn.setStyleSheet("QPushButton { color: #88aaff; }")
        correct_btn.setToolTip(t("autoAssign.correct_tooltip"))
        correct_btn.clicked.connect(lambda: self.corrected.emit(dto.assignment_id))
        row.addWidget(correct_btn)

        revert_btn = QPushButton(t("autoAssign.revert"))
        revert_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        revert_btn.setToolTip(t("autoAssign.revert_tooltip"))
        revert_btn.clicked.connect(lambda: self.reverted.emit(dto.assignment_id))
        row.addWidget(revert_btn)


class AutoAssignmentsTab(QWidget):
    """Tab widget listing the last run's automatic groupings for review.

    Signals:
        data_changed: emitted after any confirm / correct / revert.
        count_changed: ``(n_open: int)`` — for the tab-title badge.
    """

    data_changed = Signal()
    count_changed = Signal(int)

    def __init__(self, deep_config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._deep_config = deep_config
        self._loaded = False
        self._build_ui()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        intro = QLabel(t("autoAssign.intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        root.addWidget(intro)

        header = QHBoxLayout()
        self._run_label = QLabel("")
        self._run_label.setStyleSheet("color: #888;")
        header.addWidget(self._run_label)
        header.addStretch()
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #88aaff; font-weight: bold;")
        header.addWidget(self._count_label)
        root.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

        self._empty_label = QLabel(t("autoAssign.empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 20px;"
        )
        root.addWidget(self._empty_label)

        # ── "Already reviewed" history (collapsed by default) ─────────────
        from app.ui.widgets.collapsible_section import CollapsibleSection

        decided_scroll = QScrollArea()
        decided_scroll.setWidgetResizable(True)
        decided_scroll.setMaximumHeight(240)
        self._decided_widget = QWidget()
        self._decided_layout = QVBoxLayout(self._decided_widget)
        self._decided_layout.setSpacing(4)
        self._decided_layout.addStretch()
        decided_scroll.setWidget(self._decided_widget)

        self._decided_section = CollapsibleSection(
            t("autoAssign.decided_header", n=0), decided_scroll
        )
        self._decided_section.toggled.connect(self._on_decided_toggled)
        self._decided_loaded = False
        root.addWidget(self._decided_section)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load on first show (the tab is cheap until the user opens it)."""
        if not self._loaded:
            self.reload()

    def reload(self) -> None:
        self._loaded = True
        dtos: List[AutoAssignmentDTO] = []
        try:
            with session_scope() as session:
                svc = DeepRecognitionService(session, self._deep_config)
                dtos = svc.list_auto_assignments()
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load automatic groupings")
            QMessageBox.critical(self, t("error"), str(exc))

        # Clear old rows (keep trailing stretch).
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for dto in dtos:
            row = _AutoAssignmentRow(dto)
            row.confirmed.connect(self._on_confirm)
            row.corrected.connect(self._on_correct)
            row.reverted.connect(self._on_revert)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

        if dtos and dtos[0].run_finished_at is not None:
            stamp = dtos[0].run_finished_at.strftime("%Y-%m-%d %H:%M")
            self._run_label.setText(t("autoAssign.run_info", date=stamp))
        else:
            self._run_label.setText("")

        self._count_label.setText(t("autoAssign.count", n=len(dtos)))
        self._empty_label.setVisible(not dtos)
        self.count_changed.emit(len(dtos))
        self._refresh_decided()

    def _refresh_decided(self) -> None:
        """Update the "already reviewed" header; reload its rows if open."""
        from app.db.models import AUTO_ASSIGN_STATUS_AUTO, AutoAssignment

        try:
            with session_scope() as session:
                count = (
                    session.query(AutoAssignment)
                    .filter(AutoAssignment.status != AUTO_ASSIGN_STATUS_AUTO)
                    .count()
                )
                decided = (
                    DeepRecognitionService(
                        session, self._deep_config
                    ).list_decided_assignments()
                    if self._decided_section.is_expanded()
                    else None
                )
        except Exception:  # noqa: BLE001 — history must never break the tab
            log.exception("Failed to load reviewed groupings")
            return

        self._decided_section.set_title(t("autoAssign.decided_header", n=count))
        if decided is None:
            self._decided_loaded = False
            return
        self._decided_loaded = True

        while self._decided_layout.count() > 1:
            item = self._decided_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        if not decided:
            empty = QLabel(t("autoAssign.decided_empty"))
            empty.setStyleSheet("color: #666; font-style: italic; border: none;")
            empty.setAlignment(Qt.AlignCenter)
            self._decided_layout.insertWidget(0, empty)
        for i, dto in enumerate(decided):
            self._decided_layout.insertWidget(i, _DecidedAssignmentRow(dto))

    def _on_decided_toggled(self, expanded: bool) -> None:
        if expanded and not self._decided_loaded:
            self._refresh_decided()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_confirm(self, assignment_id: int) -> None:
        self._run_action(
            lambda svc: svc.confirm_assignment(assignment_id), "confirm"
        )

    def _on_correct(self, assignment_id: int) -> None:
        from app.ui.dialogs.move_faces_dialog import MoveFacesDialog

        try:
            with session_scope() as session:
                persons = (
                    session.query(Person)
                    .filter(Person.is_auto_named == False)  # noqa: E712
                    .filter(Person.is_protected == False)  # noqa: E712
                    .order_by(Person.name)
                    .all()
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load persons for correction")
            QMessageBox.critical(self, t("error"), str(exc))
            return

        dlg = MoveFacesDialog(face_count=1, persons=persons, parent=self.window())
        if dlg.exec() != MoveFacesDialog.Accepted:
            return

        new_name = dlg.new_person_name()
        person_id = dlg.selected_person_id()

        def action(svc: DeepRecognitionService):
            if new_name:
                svc.correct_assignment_to_new_person(assignment_id, new_name)
            elif person_id is not None:
                svc.correct_assignment(assignment_id, person_id)

        self._run_action(action, "correct")

    def _on_revert(self, assignment_id: int) -> None:
        self._run_action(
            lambda svc: svc.revert_assignment(assignment_id), "revert"
        )

    def _run_action(self, action, label: str) -> None:
        try:
            with session_scope() as session:
                action(DeepRecognitionService(session, self._deep_config))
        except Exception as exc:  # noqa: BLE001
            log.exception("Automatic-grouping %s failed", label)
            QMessageBox.warning(self, t("error"), str(exc))
            return
        self.reload()
        self.data_changed.emit()
