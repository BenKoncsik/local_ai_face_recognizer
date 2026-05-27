"""Cluster detail panel — face thumbnail grid for a selected person."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImageReader, QPainter, QPixmap, QPixmapCache
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face
from app.ui.i18n import t

# Warning badge drawn over thumbnails of low-quality faces
_BADGE_COLOR = "#f57c00"   # orange

log = logging.getLogger(__name__)

_THUMB_SIZE = 96
_THUMB_COLS = 5
_ZOOM_SIZE = 280


def _load_crop_pixmap(
    crop_path: Optional[str],
    size: int,
    face_id: Optional[int] = None,
) -> Optional[QPixmap]:
    """Load a crop without reusing Qt's filename pixmap cache."""
    if not crop_path or not Path(crop_path).exists():
        return None
    QPixmapCache.remove(crop_path)
    reader = QImageReader(crop_path)
    image = reader.read()
    if image.isNull():
        log.warning(
            "Crop preview load failed: FaceId=%s preview_source=%s error=%s",
            face_id,
            crop_path,
            reader.errorString(),
        )
        return None
    pixmap = QPixmap.fromImage(image).scaled(
        size,
        size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    log.debug(
        "Render crop preview: FaceId=%s preview_source=%s size=%dx%d",
        face_id,
        crop_path,
        pixmap.width(),
        pixmap.height(),
    )
    return pixmap


class _ZoomPopup(QLabel):
    """Floating popup that shows a larger version of a face crop."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px solid #88aaff; "
            "border-radius: 6px; padding: 4px; }"
        )
        self.setFixedSize(_ZOOM_SIZE + 8, _ZOOM_SIZE + 8)

    def show_for(self, crop_path: Optional[str], global_pos) -> None:
        pixmap = _load_crop_pixmap(crop_path, _ZOOM_SIZE)
        if pixmap is not None:
            self.setPixmap(pixmap)
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
        self.show()


_zoom_popup = None


def _get_zoom_popup() -> _ZoomPopup:
    global _zoom_popup
    if _zoom_popup is None:
        _zoom_popup = _ZoomPopup()
    return _zoom_popup


class FaceThumbnail(QLabel):
    """Clickable face thumbnail widget.

    Signals:
        clicked: ``(face_id: int)``
        right_clicked: ``(face_id: int, global_x: int, global_y: int)``
    """

    clicked = Signal(int)
    right_clicked = Signal(int, int, int)

    def __init__(self, face: Face, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face.id
        self._crop_path = face.crop_path
        self._is_low_quality: bool = bool(face.is_low_quality)
        self._person_name: Optional[str] = face.person.name if face.person else None
        self.setObjectName(f"face-thumb-{face.id}")
        self.setProperty("face_id", face.id)
        self.setProperty("crop_path", face.crop_path or "")
        self._load_pixmap(face.crop_path)
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        person_name = face.person.name if face.person else "—"
        quality_suffix = (
            f"\n⚠ {t('fq_low_quality_tip')}" if self._is_low_quality else ""
        )
        self.setToolTip(
            t(
                "face_tooltip",
                person=person_name,
                id=face.id,
                confidence=face.confidence,
                backend=face.detector_backend,
                file=Path(face.image.file_path).name if face.image else "?",
            ) + quality_suffix
        )
        border_color = "#f57c00" if self._is_low_quality else "#555"
        self.setStyleSheet(
            f"QLabel {{ border: 1px solid {border_color}; border-radius: 4px; }}"
            "QLabel:hover { border: 2px solid #88aaff; }"
        )
        self.setMouseTracking(True)

    def _load_pixmap(self, crop_path: Optional[str]) -> None:
        pixmap = _load_crop_pixmap(crop_path, _THUMB_SIZE, self.face_id)
        if pixmap is not None:
            if self._is_low_quality:
                # Draw a small orange ⚠ badge in the top-right corner.
                badged = QPixmap(pixmap.size())
                badged.fill(Qt.transparent)
                painter = QPainter(badged)
                painter.drawPixmap(0, 0, pixmap)
                painter.setPen(Qt.NoPen)
                # badge background circle
                from PySide6.QtGui import QBrush, QColor, QFont
                badge_size = 18
                margin = 2
                bx = badged.width() - badge_size - margin
                by = margin
                painter.setBrush(QBrush(QColor(_BADGE_COLOR)))
                painter.drawEllipse(bx, by, badge_size, badge_size)
                # "!" glyph
                font = QFont()
                font.setPixelSize(12)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(bx, by, badge_size, badge_size, Qt.AlignCenter, "!")
                painter.end()
                self.setPixmap(badged)
            else:
                self.setPixmap(pixmap)
        else:
            self.setText("?")
            self.setStyleSheet(
                "QLabel { background: #333; color: #888; "
                "border: 1px solid #555; font-size: 20px; "
                "border-radius: 4px; }"
            )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        popup = _get_zoom_popup()
        popup.show_for(self._crop_path, self.mapToGlobal(self.rect().center()))

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        _get_zoom_popup().hide()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.face_id)

    def contextMenuEvent(self, event) -> None:
        gp = self.mapToGlobal(event.pos())
        self.right_clicked.emit(self.face_id, gp.x(), gp.y())


class ClusterPanel(QWidget):
    """Scrollable grid of face thumbnails for the selected person.

    Signals:
        face_selected: ``(face_id: int)``
        face_right_clicked: ``(face_id: int, global_x: int, global_y: int)``
    """

    face_selected = Signal(int)
    face_right_clicked = Signal(int, int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._face_names: dict[int, Optional[str]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel(t("select_person_sidebar"))
        self._header.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setSpacing(6)
        scroll.setWidget(self._grid_widget)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_person(self, person_name: str, faces: list[Face]) -> None:
        """Populate the grid with *faces* belonging to *person_name*."""
        self._clear_grid()
        self._face_names.clear()
        self._header.setText(t("face_count_header", name=person_name, n=len(faces)))

        for i, face in enumerate(faces):
            row, col = divmod(i, _THUMB_COLS)
            log.debug(
                "Cluster grid item: FaceId=%s PersonId=%s crop_path=%r "
                "image_path=%r bbox=(%s,%s,%s,%s) preview_source=%r row=%d col=%d",
                face.id,
                face.person_id,
                face.crop_path,
                face.image.file_path if face.image else None,
                face.bbox_x,
                face.bbox_y,
                face.bbox_w,
                face.bbox_h,
                face.crop_path,
                row,
                col,
            )
            thumb = FaceThumbnail(face)
            thumb.clicked.connect(self.face_selected.emit)
            thumb.right_clicked.connect(self.face_right_clicked.emit)
            self._face_names[face.id] = thumb._person_name
            self._grid.addWidget(thumb, row, col)

        # Fill remaining cells in last row
        if faces:
            remainder = len(faces) % _THUMB_COLS
            if remainder:
                for col in range(remainder, _THUMB_COLS):
                    spacer = QWidget()
                    spacer.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
                    self._grid.addWidget(spacer, len(faces) // _THUMB_COLS, col)

    def get_face_person_name(self, face_id: int) -> Optional[str]:
        """Return the person name for *face_id* as loaded in the current grid."""
        return self._face_names.get(face_id)

    def clear(self) -> None:
        self._clear_grid()
        self._face_names.clear()
        self._header.setText(t("select_person_sidebar"))

    # ------------------------------------------------------------------

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
