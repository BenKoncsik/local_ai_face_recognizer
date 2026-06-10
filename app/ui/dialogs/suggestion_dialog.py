"""Name-suggestion review dialog.

Shows proposed Unknown → Known matches.  Each row pairs the unknown
person's representative face with the suggested named person's face and
lets the user approve (merge) or reject (record a 'different person'
correction) the suggestion.

New in this version:
- Crop thumbnails are clickable → opens FullImageDialog for the source photo
- Hover over a crop thumbnail shows an enlarged zoom popup
- Each person block has an "All Images" button → FaceGalleryDialog
- A "Compare" button opens a side-by-side CompareDialog
- Keyboard navigation: ↑↓ focus, Enter approve, Del reject, Space compare
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.db.database import session_scope
from app.services.merge_suggestion_service import (
    MergeSuggestionDTO,
    MergeSuggestionService,
)
from app.services.suggestion_service import Suggestion
from app.ui.dialogs.suggestion_viewer import (
    CompareDialog,
    FaceGalleryDialog,
    FullImageDialog,
    get_other_bboxes,
)
from app.ui.i18n import t
from app.ui.workers.suggestion_worker import SuggestionWorker

log = logging.getLogger(__name__)

_CROP_SIZE = 72
_ZOOM_POPUP_SIZE = 220

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_pixmap(path: Optional[str]) -> Optional[QPixmap]:
    """Load a face crop scaled to the row thumbnail size, or None."""
    if path and Path(path).exists():
        from app.utils.image_utils import load_pixmap_exif
        pix = load_pixmap_exif(path)
        if not pix.isNull():
            return pix.scaled(
                _CROP_SIZE, _CROP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
    return None


# Singleton zoom popup shared across all suggestion rows
_zoom_popup: Optional[QLabel] = None


def _get_zoom_popup() -> QLabel:
    global _zoom_popup
    if _zoom_popup is None:
        lbl = QLabel(None, Qt.ToolTip | Qt.FramelessWindowHint)
        lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px solid #88aaff; "
            "border-radius: 6px; padding: 4px; }"
        )
        lbl.setFixedSize(_ZOOM_POPUP_SIZE + 8, _ZOOM_POPUP_SIZE + 8)
        _zoom_popup = lbl
    return _zoom_popup


def _show_zoom_popup(crop_path: Optional[str], global_pos) -> None:
    popup = _get_zoom_popup()
    if crop_path and Path(crop_path).exists():
        from app.utils.image_utils import load_pixmap_exif
        pix = load_pixmap_exif(crop_path).scaled(
            _ZOOM_POPUP_SIZE, _ZOOM_POPUP_SIZE,
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        popup.setPixmap(pix)
    else:
        popup.setText("?")

    screen = QApplication.primaryScreen().geometry()
    x = global_pos.x() + 16
    y = global_pos.y() - popup.height() // 2
    if x + popup.width() > screen.right():
        x = global_pos.x() - popup.width() - 16
    if y < screen.top():
        y = screen.top() + 4
    if y + popup.height() > screen.bottom():
        y = screen.bottom() - popup.height() - 4
    popup.move(x, y)
    popup.show()


def _hide_zoom_popup() -> None:
    _get_zoom_popup().hide()


# ---------------------------------------------------------------------------
# Clickable crop label
# ---------------------------------------------------------------------------

class _ClickableCrop(QLabel):
    """Suggestion thumbnail: shows a face crop, supports hover zoom + click to open full image."""

    def __init__(
        self,
        crop_path: Optional[str],
        image_path: Optional[str],
        bbox: Optional[tuple],
        face_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._crop_path = crop_path
        self._image_path = image_path
        self._bbox = bbox
        self._face_id = face_id

        self.setFixedSize(_CROP_SIZE, _CROP_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        can_open = bool(image_path and bbox)
        if can_open:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(t("suggestions_click_full_image"))

        pixmap = _crop_pixmap(crop_path)
        if pixmap is not None:
            self.setPixmap(pixmap)
            border = "#88aaff" if can_open else "#555"
            self.setStyleSheet(
                f"QLabel {{ border: 1px solid {border}; border-radius: 4px; }}"
                f"QLabel:hover {{ border: 2px solid #88aaff; }}"
            )
        else:
            self.setText("?")
            self.setStyleSheet(
                "border: 1px solid #555; border-radius: 4px; "
                "color: #888; font-size: 20px;"
            )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        _show_zoom_popup(self._crop_path, self.mapToGlobal(self.rect().center()))

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        _hide_zoom_popup()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton and self._image_path and self._bbox:
            _hide_zoom_popup()
            other = get_other_bboxes(self._image_path, self._face_id)
            dlg = FullImageDialog(
                self._image_path, self._bbox, other, parent=self.window()
            )
            dlg.exec()


# ---------------------------------------------------------------------------
# Suggestion row
# ---------------------------------------------------------------------------

_ROW_STYLE_NORMAL  = "QFrame { background: #232323; border-radius: 6px; }"
_ROW_STYLE_FOCUSED = (
    "QFrame { background: #2a2a3a; border: 2px solid #88aaff; border-radius: 6px; }"
)


class _SuggestionRow(QFrame):
    """One suggestion: unknown face → suggested named face, with actions.

    Signals (all carry the persisted ``suggestion_id``):
        approved / rejected / deferred: ``(suggestion_id: int)``
        focused: emitted when any part of this row is interacted with
    """

    approved  = Signal(int)
    rejected  = Signal(int)
    deferred  = Signal(int)
    focused   = Signal()
    exclusions_changed = Signal()

    def __init__(
        self,
        suggestion: Suggestion,
        suggestion_id: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._s = suggestion
        self._sid = suggestion_id
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(_ROW_STYLE_NORMAL)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Main content row ──────────────────────────────────────────────
        main_row = QHBoxLayout()
        main_row.setSpacing(8)

        main_row.addWidget(
            _ClickableCrop(
                suggestion.candidate_crop_path,
                suggestion.candidate_image_path,
                suggestion.candidate_bbox,
                suggestion.candidate_face_id,
            )
        )
        main_row.addWidget(
            self._person_block(
                suggestion.candidate_name,
                suggestion.candidate_face_count,
                suggestion.candidate_person_id,
                allow_merge_exclusion=True,
            )
        )

        arrow = QLabel("→")
        arrow.setStyleSheet("font-size: 22px; color: #88aaff; border: none;")
        main_row.addWidget(arrow)

        main_row.addWidget(
            _ClickableCrop(
                suggestion.target_crop_path,
                suggestion.target_image_path,
                suggestion.target_bbox,
                suggestion.target_face_id,
            )
        )
        main_row.addWidget(
            self._person_block(
                suggestion.target_name,
                suggestion.target_face_count,
                suggestion.target_person_id,
            )
        )

        pct = int(round(max(0.0, min(1.0, suggestion.similarity)) * 100))
        sim_label = QLabel(t("suggestions_similarity", pct=pct))
        sim_label.setAlignment(Qt.AlignCenter)
        sim_label.setFixedWidth(80)
        sim_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #88ee88; border: none;"
        )
        main_row.addWidget(sim_label)
        main_row.addStretch()

        # action buttons
        compare_btn = QPushButton(t("suggestions_compare"))
        compare_btn.setStyleSheet("QPushButton { color: #88aaff; }")
        compare_btn.setToolTip(t("suggestions_compare"))
        compare_btn.clicked.connect(self._trigger_compare)
        main_row.addWidget(compare_btn)

        approve_btn = QPushButton(t("suggestions_approve"))
        approve_btn.setStyleSheet("QPushButton { color: #88ee88; }")
        approve_btn.clicked.connect(self._trigger_approve)
        main_row.addWidget(approve_btn)

        reject_btn = QPushButton(t("suggestions_reject"))
        reject_btn.setStyleSheet("QPushButton { color: #ff8888; }")
        reject_btn.clicked.connect(self._trigger_reject)
        main_row.addWidget(reject_btn)

        defer_btn = QPushButton(t("suggestions_defer"))
        defer_btn.setStyleSheet("QPushButton { color: #e0c080; }")
        defer_btn.clicked.connect(self._trigger_defer)
        main_row.addWidget(defer_btn)

        root.addLayout(main_row)

    # ------------------------------------------------------------------
    # Widget builders
    # ------------------------------------------------------------------

    def _person_block(
        self,
        name: str,
        face_count: int,
        person_id: int,
        allow_merge_exclusion: bool = False,
    ) -> QWidget:
        widget = QWidget()
        widget.setFixedWidth(155)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        name_label = QLabel(name)
        name_label.setWordWrap(True)
        name_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #eee; border: none;"
        )
        count_label = QLabel(t("suggestions_faces", n=face_count))
        count_label.setStyleSheet("font-size: 10px; color: #888; border: none;")

        all_btn = QPushButton(t("suggestions_all_images"))
        all_btn.setStyleSheet("font-size: 10px; padding: 1px 4px; color: #88aaff;")
        all_btn.setFixedHeight(20)
        all_btn.setToolTip(t("suggestions_all_images"))
        _name = name
        _pid = person_id
        all_btn.clicked.connect(
            lambda: self._open_gallery(_pid, _name, allow_merge_exclusion)
        )

        layout.addWidget(name_label)
        layout.addWidget(count_label)
        layout.addWidget(all_btn)
        return widget

    def _open_gallery(
        self, person_id: int, name: str, allow_merge_exclusion: bool
    ) -> None:
        self.focused.emit()
        dlg = FaceGalleryDialog(
            person_id,
            name,
            parent=self.window(),
            allow_merge_exclusion=allow_merge_exclusion,
        )
        dlg.exec()
        if allow_merge_exclusion and getattr(dlg, "changed", False):
            # Exclusions changed the candidate's faces — ask the dialog to
            # reload so counts/representatives reflect the new state.
            self.exclusions_changed.emit()

    # ------------------------------------------------------------------
    # Focus
    # ------------------------------------------------------------------

    def set_focused(self, focused: bool) -> None:
        self.setStyleSheet(_ROW_STYLE_FOCUSED if focused else _ROW_STYLE_NORMAL)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        self.focused.emit()

    # ------------------------------------------------------------------
    # Trigger methods (callable from keyboard shortcuts in parent dialog)
    # ------------------------------------------------------------------

    def _trigger_approve(self) -> None:
        self.focused.emit()
        self.approved.emit(self._sid)

    def _trigger_reject(self) -> None:
        self.focused.emit()
        self.rejected.emit(self._sid)

    def _trigger_defer(self) -> None:
        self.focused.emit()
        self.deferred.emit(self._sid)

    def trigger_defer(self) -> None:
        """Programmatic defer (e.g. from keyboard shortcut)."""
        self._trigger_defer()

    def _trigger_compare(self) -> None:
        self.focused.emit()
        CompareDialog(
            self._s.candidate_person_id, self._s.candidate_name,
            self._s.target_person_id, self._s.target_name,
            parent=self.window(),
        ).exec()

    def trigger_approve(self) -> None:
        """Programmatic approve (e.g. from keyboard shortcut)."""
        self._trigger_approve()

    def trigger_reject(self) -> None:
        """Programmatic reject (e.g. from keyboard shortcut)."""
        self._trigger_reject()

    def trigger_compare(self) -> None:
        """Programmatic compare (e.g. from keyboard shortcut)."""
        self._trigger_compare()


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class SuggestionDialog(QDialog):
    """Reviewable list of Unknown → Known name suggestions.

    Signals:
        data_changed: emitted after an approval changes person assignments.
    """

    data_changed = Signal()

    def __init__(
        self, config, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        # Accept either a full AppConfig or a bare SuggestionConfig (back-compat).
        self._app_config = config if isinstance(config, AppConfig) else None
        self._matching = (
            self._app_config.matching if self._app_config is not None
            else getattr(config, "matching", None)
        )
        if self._matching is None:
            from app.config import MatchingConfig
            self._matching = MatchingConfig()
        self._suggestions: List[Suggestion] = []
        self._suggestion_ids: List[int] = []
        self._rows: List[_SuggestionRow] = []
        self._focus_idx: int = -1

        # Create worker for background merge decisions (allows concurrent runs)
        self._worker = SuggestionWorker(self._matching)
        self._worker.decision_finished.connect(self._on_decision_finished)
        self._worker.error_occurred.connect(self._on_decision_error)
        # Track running decisions by suggestion_id
        self._worker_in_progress = set()

        self.setWindowTitle(t("suggestions_title"))
        self.setMinimumSize(900, 560)
        self._build_ui()
        self._reload()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        dialog_root = QVBoxLayout(self)

        self._tabs = QTabWidget()
        dialog_root.addWidget(self._tabs, stretch=1)

        # ── Tab 1: Unknown → Known merge suggestions (the original UI) ────
        suggestions_tab = QWidget()
        root = QVBoxLayout(suggestions_tab)

        intro = QLabel(t("suggestions_intro"))
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        root.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel(t("suggestions_threshold")))

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(float(self._matching.min_confidence))
        self._threshold_spin.valueChanged.connect(lambda *_: self._reload())
        controls.addWidget(self._threshold_spin)

        self._show_deferred = QCheckBox(t("suggestions_show_deferred"))
        self._show_deferred.toggled.connect(lambda *_: self._reload())
        controls.addWidget(self._show_deferred)

        # Auto-merge section
        auto_merge_section = QHBoxLayout()
        auto_merge_section.setSpacing(2)

        self._auto_merge_btn = QPushButton(t("suggestions_auto_merge"))
        self._auto_merge_btn.clicked.connect(self._on_auto_merge)
        auto_merge_section.addWidget(self._auto_merge_btn)

        auto_merge_section.addWidget(QLabel(t("suggestions_auto_merge_max_faces")))
        self._auto_merge_max_faces = QDoubleSpinBox()
        self._auto_merge_max_faces.setRange(1, 100)
        self._auto_merge_max_faces.setSingleStep(1)
        self._auto_merge_max_faces.setDecimals(0)
        self._auto_merge_max_faces.setValue(float(self._matching.auto_merge_max_unknown_faces))
        auto_merge_section.addWidget(self._auto_merge_max_faces)

        auto_merge_section.addWidget(QLabel(t("suggestions_auto_merge_min_confidence")))
        self._auto_merge_min_confidence = QDoubleSpinBox()
        self._auto_merge_min_confidence.setRange(0.0, 1.0)
        self._auto_merge_min_confidence.setSingleStep(0.05)
        self._auto_merge_min_confidence.setDecimals(2)
        self._auto_merge_min_confidence.setValue(float(self._matching.auto_merge_min_confidence))
        auto_merge_section.addWidget(self._auto_merge_min_confidence)

        controls.addLayout(auto_merge_section)

        controls.addStretch()

        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #88aaff; font-weight: bold;")
        controls.addWidget(self._count_label)
        root.addLayout(controls)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()
        scroll.setWidget(self._list_widget)
        root.addWidget(scroll, stretch=1)

        self._empty_label = QLabel(t("suggestions_empty"))
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet(
            "color: #888; font-style: italic; padding: 20px;"
        )
        root.addWidget(self._empty_label)

        # keyboard hint
        kb_hint = QLabel(t("suggestions_keyboard_hint"))
        kb_hint.setStyleSheet("color: #666; font-size: 10px;")
        kb_hint.setAlignment(Qt.AlignCenter)
        root.addWidget(kb_hint)

        self._tabs.addTab(suggestions_tab, t("suggestions_tab_matches"))

        # ── Tab 2: automatic groupings of the last AI run ─────────────────
        from app.ui.dialogs.auto_assignments_tab import AutoAssignmentsTab

        deep_config = (
            self._app_config.deep_recognition
            if self._app_config is not None
            else None
        )
        self._auto_tab = AutoAssignmentsTab(deep_config, parent=self)
        self._auto_tab.data_changed.connect(self.data_changed)
        self._auto_tab.count_changed.connect(self._on_auto_count_changed)
        self._tabs.addTab(self._auto_tab, t("suggestions_tab_auto"))
        self._tabs.currentChanged.connect(self._on_tab_changed)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        dialog_root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------

    def show_auto_assignments_tab(self) -> None:
        """Open the dialog focused on the automatic-groupings review tab."""
        self._tabs.setCurrentWidget(self._auto_tab)

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._auto_tab:
            self._auto_tab.ensure_loaded()

    @Slot(int)
    def _on_auto_count_changed(self, count: int) -> None:
        label = t("suggestions_tab_auto")
        if count > 0:
            label = f"{label} ({count})"
        self._tabs.setTabText(self._tabs.indexOf(self._auto_tab), label)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _reload(self) -> None:
        """Load persisted background suggestions and rebuild rows.

        Reading from the persisted :class:`MergeSuggestion` store is a cheap,
        indexed query — the heavy matching happens in the background worker, so
        opening this dialog never blocks the UI with a full archive scan.
        """
        threshold = self._threshold_spin.value()
        include_deferred = self._show_deferred.isChecked()
        self._suggestions = []
        self._suggestion_ids = []
        try:
            with session_scope() as session:
                dtos = MergeSuggestionService(session, self._matching).list_open(
                    include_deferred=include_deferred
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to load suggestions")
            QMessageBox.critical(self, t("error"), str(exc))
            dtos = []

        for dto in dtos:
            if dto.confidence < threshold:
                continue
            self._suggestions.append(self._to_display(dto))
            self._suggestion_ids.append(dto.suggestion_id)
        self._rebuild_rows()

    @staticmethod
    def _to_display(dto: MergeSuggestionDTO) -> Suggestion:
        """Adapt a persisted DTO to the row's display struct."""
        return Suggestion(
            candidate_person_id=dto.candidate_person_id,
            candidate_name=dto.candidate_name,
            candidate_face_id=0,
            candidate_crop_path=dto.candidate_crop_path,
            candidate_face_count=dto.candidate_face_count,
            target_person_id=dto.target_person_id,
            target_name=dto.target_name,
            target_face_id=0,
            target_crop_path=dto.target_crop_path,
            target_face_count=dto.target_face_count,
            similarity=dto.confidence,
            candidate_image_path=dto.candidate_image_path,
            candidate_bbox=dto.candidate_bbox,
            target_image_path=dto.target_image_path,
            target_bbox=dto.target_bbox,
        )

    def _rebuild_rows(self) -> None:
        # Remove existing rows, keeping the trailing stretch item.
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self._rows = []
        self._focus_idx = -1

        for i, suggestion in enumerate(self._suggestions):
            row = _SuggestionRow(suggestion, self._suggestion_ids[i])
            row.approved.connect(self._on_approve)
            row.rejected.connect(self._on_reject)
            row.deferred.connect(self._on_defer)
            row.exclusions_changed.connect(self._reload)
            row.focused.connect(lambda idx=i: self._set_focus(idx))
            # If a decision for this suggestion is currently running, disable the row
            if self._suggestion_ids[i] in self._worker_in_progress:
                row.setDisabled(True)
            self._rows.append(row)
            self._list_layout.insertWidget(self._list_layout.count() - 1, row)

        has_rows = bool(self._suggestions)
        self._empty_label.setVisible(not has_rows)
        self._count_label.setText(t("suggestions_count", n=len(self._suggestions)))

        if has_rows:
            self._set_focus(0)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------

    def keyPressEvent(self, event) -> None:
        # Row navigation shortcuts only apply on the suggestions tab.
        if self._tabs.currentWidget() is self._auto_tab:
            super().keyPressEvent(event)
            return

        key = event.key()
        n = len(self._rows)

        if key in (Qt.Key_Down, Qt.Key_Tab) and not (
            key == Qt.Key_Tab and event.modifiers() & Qt.ShiftModifier
        ):
            self._set_focus(min(self._focus_idx + 1, n - 1))
            event.accept()
            return

        if key == Qt.Key_Up or (
            key == Qt.Key_Tab and event.modifiers() & Qt.ShiftModifier
        ):
            self._set_focus(max(self._focus_idx - 1, 0))
            event.accept()
            return

        focused = self._focused_row()
        if focused is None:
            super().keyPressEvent(event)
            return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            focused.trigger_approve()
            event.accept()
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            focused.trigger_reject()
            event.accept()
        elif key == Qt.Key_L:
            focused.trigger_defer()
            event.accept()
        elif key == Qt.Key_Space:
            focused.trigger_compare()
            event.accept()
        else:
            super().keyPressEvent(event)

    def _focused_row(self) -> Optional[_SuggestionRow]:
        if 0 <= self._focus_idx < len(self._rows):
            return self._rows[self._focus_idx]
        return None

    def _set_focus(self, idx: int) -> None:
        if 0 <= self._focus_idx < len(self._rows):
            self._rows[self._focus_idx].set_focused(False)
        self._focus_idx = idx
        if 0 <= idx < len(self._rows):
            self._rows[idx].set_focused(True)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _display_for(self, suggestion_id: int) -> Optional[Suggestion]:
        for sid, s in zip(self._suggestion_ids, self._suggestions):
            if sid == suggestion_id:
                return s
        return None

    @Slot(int)
    def _on_approve(self, suggestion_id: int) -> None:
        s = self._display_for(suggestion_id)
        cand_name = s.candidate_name if s else str(suggestion_id)
        target_name = s.target_name if s else ""
        reply = QMessageBox.question(
            self,
            t("suggestions_approve"),
            t("suggestions_approve_confirm", cand=cand_name, target=target_name),
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_decision_async(
            lambda svc: svc.accept(suggestion_id), "approval", suggestion_id
        )

    @Slot(int)
    def _on_reject(self, suggestion_id: int) -> None:
        self._run_decision_async(
            lambda svc: svc.reject(suggestion_id), "rejection", suggestion_id
        )

    @Slot(int)
    def _on_defer(self, suggestion_id: int) -> None:
        self._run_decision_async(
            lambda svc: svc.defer(suggestion_id), "defer", suggestion_id
        )

    def _run_decision(self, action, label: str) -> bool:
        """Execute *action(service)* inside a session; report failures."""
        try:
            with session_scope() as session:
                action(MergeSuggestionService(session, self._matching))
            return True
        except Exception as exc:  # noqa: BLE001
            log.exception("Suggestion %s failed", label)
            QMessageBox.warning(self, t("error"), str(exc))
            return False

    def _run_decision_async(self, action, label: str, suggestion_id: int) -> None:
        """Queue a decision to run in the thread pool (non-blocking)."""
        if suggestion_id in self._worker_in_progress:
            log.warning("Decision for suggestion %d already in progress, ignoring new request", suggestion_id)
            return
        self._worker_in_progress.add(suggestion_id)
        # Disable the corresponding row UI while processing
        try:
            idx = self._suggestion_ids.index(suggestion_id)
        except ValueError:
            idx = None
        if idx is not None:
            self._rows[idx].setDisabled(True)
        self._worker.run_decision(action, label, suggestion_id)

    def _on_auto_merge(self) -> None:
        """Trigger automatic merging of high-confidence suggestions."""
        min_conf = self._auto_merge_min_confidence.value()
        max_faces = int(self._auto_merge_max_faces.value())

        def run_auto_merge(svc):
            return svc.auto_merge_suggestions(
                min_confidence=min_conf,
                max_unknown_faces=max_faces,
            )

        try:
            with session_scope() as session:
                merged_count = run_auto_merge(MergeSuggestionService(session, self._matching))
            if merged_count > 0:
                QMessageBox.information(
                    self,
                    t("suggestions_auto_merge"),
                    t("suggestions_auto_merge_result", n=merged_count),
                )
                self._reload()
            else:
                QMessageBox.information(
                    self,
                    t("suggestions_auto_merge"),
                    t("suggestions_auto_merge_none"),
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("Auto-merge failed")
            QMessageBox.warning(self, t("error"), str(exc))

    @Slot(int, str, bool)
    def _on_decision_finished(self, suggestion_id: int, label: str, success: bool) -> None:
        """Handle worker completion for a specific suggestion.

        Remove the processed row from the UI immediately when possible. On
        success the row is removed; on failure the row is re-enabled so the
        user can try again.
        """
        # Mark done
        self._worker_in_progress.discard(suggestion_id)

        if not success:
            # Re-enable the row if present; error message is delivered via
            # _on_decision_error when available, but handle the generic case.
            try:
                idx = self._suggestion_ids.index(suggestion_id)
                self._rows[idx].setDisabled(False)
            except ValueError:
                pass
            return

        # success path
        log.info("Merge suggestion %s completed for %d", label, suggestion_id)
        if label == "approval":
            self.data_changed.emit()

        # Try to remove the specific row from the list without a full reload.
        try:
            idx = self._suggestion_ids.index(suggestion_id)
        except ValueError:
            # If we can't find it locally, fall back to a full reload.
            self._reload()
            return

        # Remove widget from layout and internal lists
        row = self._rows.pop(idx)
        self._suggestions.pop(idx)
        self._suggestion_ids.pop(idx)
        # Remove from layout and delete
        try:
            self._list_layout.removeWidget(row)
            row.deleteLater()
        except Exception:
            log.debug("Failed to remove row widget cleanly", exc_info=True)

        # Update count label
        self._count_label.setText(t("suggestions_count", n=len(self._suggestions)))

        # Adjust focus index
        if self._focus_idx >= len(self._rows):
            self._focus_idx = len(self._rows) - 1
        if self._focus_idx >= 0:
            # Ensure one row has focused styling
            for i, r in enumerate(self._rows):
                r.set_focused(i == self._focus_idx)
        else:
            # No rows left -> show empty label
            self._empty_label.setVisible(len(self._rows) == 0)

    @Slot(int, str)
    def _on_decision_error(self, suggestion_id: int, error_msg: str) -> None:
        """Handle worker error for a specific suggestion."""
        self._worker_in_progress.discard(suggestion_id)
        log.exception("Suggestion %d failed: %s", suggestion_id, error_msg)
        # Re-enable row if present
        try:
            idx = self._suggestion_ids.index(suggestion_id)
            self._rows[idx].setDisabled(False)
        except ValueError:
            pass
        QMessageBox.warning(self, t("error"), error_msg)
