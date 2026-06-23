"""Name-suggestion review dialog.

Hosts the deep engine's automatic groupings for review.  Every face the last
AI run placed with a person is listed; the user confirms (positive training
example), corrects, or reverts (negative example) each grouping.  Those
decisions teach the neural network — see :class:`AutoAssignmentsTab` and the
background retrain triggered by the main window when the dialog closes.

The legacy cosine "Name Suggestions (classic)" tab and its background
merge-suggestion engine were removed; the AI engine is the only path now.

This module also exposes a few small widgets reused elsewhere (the clickable
crop thumbnail with hover-zoom + full-image popup).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.ui.dialogs.suggestion_viewer import FullImageDialog, get_other_bboxes
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
        from app.utils.image_utils import load_pixmap_exif
        pix = load_pixmap_exif(path)
        if not pix.isNull():
            return pix.scaled(
                _CROP_SIZE, _CROP_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
    return None


# Singleton zoom popup shared across all suggestion rows
class _ZoomPopup(QLabel):
    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px solid #88aaff; "
            "border-radius: 6px; padding: 4px; }"
        )
        self.setFixedSize(_ZOOM_POPUP_SIZE + 8, _ZOOM_POPUP_SIZE + 8)
        self._source: Optional[QWidget] = None
        self._filter_active: bool = False

    def show_for(self, crop_path: Optional[str], global_pos, source: Optional[QWidget] = None) -> None:
        if crop_path and Path(crop_path).exists():
            from app.utils.image_utils import load_pixmap_exif
            pix = load_pixmap_exif(crop_path).scaled(
                _ZOOM_POPUP_SIZE, _ZOOM_POPUP_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            self.setPixmap(pix)
        else:
            self.setText("?")

        screen = QApplication.primaryScreen().geometry()
        x = global_pos.x() + 16
        y = global_pos.y() - self.height() // 2
        if x + self.width() > screen.right():
            x = global_pos.x() - self.width() - 16
        if y < screen.top():
            y = screen.top() + 4
        if y + self.height() > screen.bottom():
            y = screen.bottom() - self.height() - 4
        self.move(x, y)
        self._source = source
        self.show()
        if source is not None and not self._filter_active:
            QApplication.instance().installEventFilter(self)
            self._filter_active = True

    def hide(self) -> None:
        if self._filter_active:
            QApplication.instance().removeEventFilter(self)
            self._filter_active = False
        self._source = None
        super().hide()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseMove and self._source is not None:
            try:
                tl = self._source.mapToGlobal(QPoint(0, 0))
                if not QRect(tl, self._source.size()).contains(QCursor.pos()):
                    self.hide()
            except RuntimeError:
                self.hide()
        return False


_zoom_popup: Optional[_ZoomPopup] = None


def _get_zoom_popup() -> _ZoomPopup:
    global _zoom_popup
    if _zoom_popup is None:
        _zoom_popup = _ZoomPopup()
    return _zoom_popup


def _show_zoom_popup(crop_path: Optional[str], global_pos, source: Optional[QWidget] = None) -> None:
    _get_zoom_popup().show_for(crop_path, global_pos, source)


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
        _show_zoom_popup(self._crop_path, self.mapToGlobal(self.rect().center()), self)

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
# Main dialog
# ---------------------------------------------------------------------------

class SuggestionDialog(QDialog):
    """Reviewable list of the deep engine's automatic name groupings.

    Confirming a grouping turns the face into a positive training example;
    reverting it records a negative ("different person") judgement.  The dialog
    tracks whether any such decision was made so the caller can retrain the
    neural network in the background after it closes.

    Signals:
        data_changed: emitted after a confirm / correct / revert changes
            person assignments.
    """

    data_changed = Signal()

    def __init__(self, config, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Accept either a full AppConfig or a bare config (back-compat).
        self._app_config = config if isinstance(config, AppConfig) else None
        self._deep_config = (
            self._app_config.deep_recognition
            if self._app_config is not None
            else getattr(config, "deep_recognition", None)
        )
        # True once the user makes any learning-relevant decision; the caller
        # uses this to decide whether to retrain the model on close.
        self._made_decisions = False

        self.setWindowTitle(t("suggestions_title"))
        self.setMinimumSize(900, 560)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        dialog_root = QVBoxLayout(self)

        from app.ui.dialogs.auto_assignments_tab import AutoAssignmentsTab

        self._auto_tab = AutoAssignmentsTab(self._deep_config, parent=self)
        self._auto_tab.data_changed.connect(self._on_data_changed)
        dialog_root.addWidget(self._auto_tab, stretch=1)
        self._auto_tab.ensure_loaded()

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        dialog_root.addWidget(buttons)

    # ------------------------------------------------------------------

    def show_auto_assignments_tab(self) -> None:
        """Kept for back-compat: the auto-groupings view is the only view now."""
        self._auto_tab.ensure_loaded()

    def _on_data_changed(self) -> None:
        self._made_decisions = True
        self.data_changed.emit()

    @property
    def made_decisions(self) -> bool:
        """Whether the user confirmed/corrected/reverted at least one grouping."""
        return self._made_decisions
