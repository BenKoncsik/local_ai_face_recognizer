"""Multi-select grid of a person's face crops.

Backed by a :class:`QListWidget` in icon mode so selection is virtualised and
survives scrolling for free.  ``MultiSelection`` means a plain click toggles a
single face in/out of the selection — no modifier keys required — which matches
the "click to select / deselect" behaviour the batch-move feature needs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, QSize, QThreadPool, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from app.services.person_service import PersonFaceCrop
from app.workers.thumbnail_worker import ThumbnailRunnable

log = logging.getLogger(__name__)

_ROLE_FACE_ID        = Qt.UserRole
_ROLE_IS_UNCERTAIN   = Qt.UserRole + 1
_THUMB = 88
_BADGE_SIZE = 20   # px — orange "?" badge diameter


class SelectableFaceGrid(QListWidget):
    """Grid of face crops with toggle-style multi-selection.

    Signals
    -------
    selection_changed(int)
        Emitted whenever the number of selected faces changes; carries the new
        selection count.
    """

    selection_changed = Signal(int)

    def __init__(self, parent=None) -> None:  # noqa: ANN001
        super().__init__(parent)
        self.setViewMode(QListWidget.IconMode)
        self.setSelectionMode(QListWidget.MultiSelection)
        self.setMovement(QListWidget.Static)
        self.setResizeMode(QListWidget.Adjust)
        self.setUniformItemSizes(True)
        self.setSpacing(6)
        self.setIconSize(QSize(_THUMB, _THUMB))
        self.setGridSize(QSize(_THUMB + 14, _THUMB + 14))
        self.setWordWrap(False)
        self.setMinimumHeight(_THUMB * 2 + 40)
        # Dark-theme selection styling consistent with the rest of the app.
        self.setStyleSheet(
            "QListWidget { background:#181825; border:1px solid #313244; border-radius:4px; }"
            "QListWidget::item { border:2px solid transparent; border-radius:4px; }"
            "QListWidget::item:selected { border:2px solid #89B4FA; background:#1E2030; }"
        )
        self._items: Dict[int, QListWidgetItem] = {}  # face_id → item
        self.itemSelectionChanged.connect(self._emit_selection_count)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_faces(self, crops: List[PersonFaceCrop]) -> None:
        """Replace the grid with *crops* (selection is reset)."""
        self.clear()
        self._items.clear()
        self._uncertain: set[int] = set()
        for crop in crops:
            item = QListWidgetItem()
            item.setData(_ROLE_FACE_ID, crop.face_id)
            item.setData(_ROLE_IS_UNCERTAIN, crop.is_uncertain)
            item.setSizeHint(QSize(_THUMB + 12, _THUMB + 12))
            item.setIcon(self._placeholder_icon())
            if crop.is_uncertain:
                self._uncertain.add(crop.face_id)
                from app.ui.i18n import t
                note = crop.identification_note or ""
                tip = t("face_uncertain_tooltip")
                if note:
                    tip = f"{tip}\n{note}"
                item.setToolTip(tip)
            self.addItem(item)
            self._items[crop.face_id] = item

            if crop.crop_path and Path(crop.crop_path).exists():
                worker = ThumbnailRunnable(crop.crop_path, str(crop.face_id), size=_THUMB)
                worker.signals.ready.connect(self._on_thumb_ready)
                QThreadPool.globalInstance().start(worker)
        self._emit_selection_count()

    def selected_face_ids(self) -> List[int]:
        """Return the face ids of the currently selected crops."""
        return [
            int(item.data(_ROLE_FACE_ID)) for item in self.selectedItems()
        ]

    def clear_selection(self) -> None:
        """Deselect everything without rebuilding the grid."""
        self.clearSelection()
        self._emit_selection_count()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _placeholder_icon(self) -> QIcon:
        pix = QPixmap(_THUMB, _THUMB)
        pix.fill(Qt.transparent)
        return QIcon(pix)

    def _add_uncertain_badge(self, pix: QPixmap) -> QPixmap:
        """Overlay an orange '?' badge in the top-right corner of *pix*."""
        result = pix.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = _BADGE_SIZE // 2
        cx = result.width() - r - 2
        cy = r + 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 140, 0))
        painter.drawEllipse(cx - r, cy - r, _BADGE_SIZE, _BADGE_SIZE)
        font = QFont()
        font.setPixelSize(max(9, _BADGE_SIZE - 6))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(cx - r, cy - r, _BADGE_SIZE, _BADGE_SIZE, Qt.AlignCenter, "?")
        painter.end()
        return result

    def _on_thumb_ready(self, key: str, img: QImage) -> None:
        try:
            face_id = int(key)
        except (TypeError, ValueError):
            return
        item = self._items.get(face_id)
        if item is not None:
            pix = QPixmap.fromImage(img)
            if face_id in getattr(self, "_uncertain", set()):
                pix = self._add_uncertain_badge(pix)
            item.setIcon(QIcon(pix))

    def _emit_selection_count(self) -> None:
        self.selection_changed.emit(len(self.selectedItems()))
