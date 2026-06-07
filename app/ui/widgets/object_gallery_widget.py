"""Gallery widget for a tagged object: large preview + thumbnail strip.

Unlike :class:`PlaceGalleryWidget`, each image is drawn with the object's
bounding-box *frame* (cyan) so the user can see exactly what was tagged, and a
right-click lets them pick which occurrence is the object's thumbnail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t

log = logging.getLogger(__name__)

_THUMB_SIZE = 84
_PREVIEW_MAX_H = 260
_PREVIEW_MAX_W = 460
_OBJ_COLOR = QColor(80, 200, 255)

# (occurrence_id, image_path, bbox_or_None)  bbox = (x, y, w, h) in image px
OccItem = Tuple[int, str, Optional[Tuple[int, int, int, int]]]


def _draw_bbox(pix: QPixmap, bbox: Optional[Tuple[int, int, int, int]]) -> QPixmap:
    """Return a copy of *pix* with the object's bbox drawn in cyan."""
    if bbox is None or pix.isNull():
        return pix
    out = QPixmap(pix)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(_OBJ_COLOR, max(2, round(pix.width() / 250)))
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    x, y, w, h = bbox
    painter.drawRect(x, y, w, h)
    painter.end()
    return out


def crop_pixmap(
    image_path: str, bbox: Optional[Tuple[int, int, int, int]], size: int = 256
) -> Optional[QPixmap]:
    """Load *image_path* and return a scaled crop of *bbox* (or the whole image)."""
    from app.utils.image_utils import load_pixmap_exif

    pix = load_pixmap_exif(image_path)
    if pix.isNull():
        return None
    if bbox is not None:
        x, y, w, h = bbox
        x = max(0, min(x, pix.width() - 1))
        y = max(0, min(y, pix.height() - 1))
        w = max(1, min(w, pix.width() - x))
        h = max(1, min(h, pix.height() - y))
        pix = pix.copy(x, y, w, h)
    return pix.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _ObjThumb(QLabel):
    """Clickable thumbnail showing an image with the object's frame drawn."""

    clicked = Signal(int)        # occurrence_id
    right_clicked = Signal(int)  # occurrence_id

    def __init__(self, occ_id: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._occ_id = occ_id
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setText("…")
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        border = "#89B4FA" if selected else "#45475A"
        self.setStyleSheet(
            f"border:2px solid {border};background:#181825;color:#6C7086;font-size:10px;"
        )

    def set_pixmap(self, pix: QPixmap) -> None:
        self.setPixmap(
            pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        self.setText("")

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._occ_id)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: ANN001
        self.right_clicked.emit(self._occ_id)


class ObjectGalleryWidget(QWidget):
    """Large preview (with bbox frame) + thumbnail strip for a tagged object."""

    set_thumbnail_requested = Signal(int)   # occurrence_id
    clear_thumbnail_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._items: List[OccItem] = []
        self._thumbs: Dict[int, _ObjThumb] = {}
        self._current: Optional[int] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setFixedHeight(_PREVIEW_MAX_H)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._preview.setStyleSheet("background:#181825;color:#6C7086;border-radius:4px;")
        layout.addWidget(self._preview)

        scroll = QScrollArea()
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(_THUMB_SIZE + 18)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:#181825;border:none;")
        self._strip = QWidget()
        self._strip_layout = QHBoxLayout(self._strip)
        self._strip_layout.setContentsMargins(4, 4, 4, 4)
        self._strip_layout.setSpacing(4)
        self._strip_layout.addStretch()
        scroll.setWidget(self._strip)
        layout.addWidget(scroll)

        self._clear_preview()

    def set_occurrences(self, items: List[OccItem]) -> None:
        """Replace the gallery with the given object occurrences."""
        self._items = list(items)
        self._thumbs.clear()
        self._current = None
        while self._strip_layout.count() > 1:
            it = self._strip_layout.takeAt(0)
            if it and it.widget():
                it.widget().deleteLater()

        if not items:
            self._clear_preview()
            return

        for occ_id, image_path, bbox in items:
            thumb = _ObjThumb(occ_id)
            thumb.clicked.connect(self._on_thumb_click)
            thumb.right_clicked.connect(self._on_thumb_right_clicked)
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, thumb)
            self._thumbs[occ_id] = thumb
            pix = crop_pixmap_full_with_frame(image_path, bbox, _THUMB_SIZE)
            if pix is not None:
                thumb.set_pixmap(pix)
            else:
                thumb.setText("✗")

        self._select(items[0][0])

    def _select(self, occ_id: int) -> None:
        if self._current is not None and self._current in self._thumbs:
            self._thumbs[self._current].set_selected(False)
        self._current = occ_id
        if occ_id in self._thumbs:
            self._thumbs[occ_id].set_selected(True)
        item = next((it for it in self._items if it[0] == occ_id), None)
        if item is None:
            self._clear_preview()
            return
        _oid, image_path, bbox = item
        pix = crop_pixmap_full_with_frame(image_path, bbox, _PREVIEW_MAX_W, _PREVIEW_MAX_H)
        if pix is None:
            self._clear_preview()
            return
        self._preview.setPixmap(pix)
        self._preview.setText("")

    def _clear_preview(self) -> None:
        self._preview.setPixmap(QPixmap())
        self._preview.setText(t("object_no_comments"))

    def _on_thumb_click(self, occ_id: int) -> None:
        self._select(occ_id)

    def _on_thumb_right_clicked(self, occ_id: int) -> None:
        self._select(occ_id)
        menu = QMenu(self)
        set_action = menu.addAction(t("object_set_thumbnail"))
        clear_action = menu.addAction(t("object_clear_thumbnail"))
        chosen = menu.exec(QCursor.pos())
        if chosen == set_action:
            self.set_thumbnail_requested.emit(occ_id)
        elif chosen == clear_action:
            self.clear_thumbnail_requested.emit()

    def clear(self) -> None:
        self.set_occurrences([])


def crop_pixmap_full_with_frame(
    image_path: str,
    bbox: Optional[Tuple[int, int, int, int]],
    max_w: int,
    max_h: Optional[int] = None,
) -> Optional[QPixmap]:
    """Load the full image, draw the object's bbox frame, and scale to fit."""
    from app.utils.image_utils import load_pixmap_exif

    if not image_path or not Path(image_path).exists():
        return None
    pix = load_pixmap_exif(image_path)
    if pix.isNull():
        return None
    pix = _draw_bbox(pix, bbox)
    if max_h is None:
        max_h = max_w
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
