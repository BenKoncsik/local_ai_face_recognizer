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

log = logging.getLogger(__name__)

_CROP_SIZE = 72
_ZOOM_POPUP_SIZE = 220

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_pixmap(path: Optional[str]) -> Optional[QPixmap]:
    """Load a face crop scaled to the row thumbnail size, or None."""
    if path and Path(path).exists():
        return QPixmap(path).scaled(
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
        pix = QPixmap(crop_path).scaled(
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

    def _person_block(self, name: str, face_count: int, person_id: int) -> QWidget:
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
            lambda: (
                self.focused.emit(),
                FaceGalleryDialog(_pid, _name, parent=self.window()).exec(),
            )
        )

        layout.addWidget(name_label)
        layout.addWidget(count_label)
        layout.addWidget(all_btn)
        return widget

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

        self.setWindowTitle(t("suggestions_title"))
        self.setMinimumSize(900, 560)
        self._build_ui()
        self._reload()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

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

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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
            row.focused.connect(lambda idx=i: self._set_focus(idx))
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
        if self._run_decision(lambda svc: svc.accept(suggestion_id), "approval"):
            log.info("Merge suggestion %d approved", suggestion_id)
            self.data_changed.emit()
            self._reload()

    @Slot(int)
    def _on_reject(self, suggestion_id: int) -> None:
        if self._run_decision(lambda svc: svc.reject(suggestion_id), "rejection"):
            log.info("Merge suggestion %d rejected", suggestion_id)
            self._reload()

    @Slot(int)
    def _on_defer(self, suggestion_id: int) -> None:
        if self._run_decision(lambda svc: svc.defer(suggestion_id), "defer"):
            log.info("Merge suggestion %d deferred", suggestion_id)
            self._reload()

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
