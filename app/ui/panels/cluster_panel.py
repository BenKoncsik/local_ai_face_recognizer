"""Cluster detail panel — face thumbnail grid for a selected person."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face
from app.ui.i18n import t

log = logging.getLogger(__name__)

_THUMB_SIZE = 96
_THUMB_COLS = 5
_ZOOM_SIZE = 280


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
        if crop_path and Path(crop_path).exists():
            pixmap = QPixmap(crop_path).scaled(
                _ZOOM_SIZE, _ZOOM_SIZE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
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
    """

    clicked = Signal(int)

    def __init__(self, face: Face, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face.id
        self._crop_path = face.crop_path
        self._load_pixmap(face.crop_path)
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        person_name = face.person.name if face.person else "—"
        self.setToolTip(
            t(
                "face_tooltip",
                person=person_name,
                id=face.id,
                confidence=face.confidence,
                backend=face.detector_backend,
                file=Path(face.image.file_path).name if face.image else "?",
            )
        )
        self.setStyleSheet(
            "QLabel { border: 1px solid #555; border-radius: 4px; }"
            "QLabel:hover { border: 2px solid #88aaff; }"
        )
        self.setMouseTracking(True)

    def _load_pixmap(self, crop_path: Optional[str]) -> None:
        if crop_path and Path(crop_path).exists():
            pixmap = QPixmap(crop_path).scaled(
                _THUMB_SIZE, _THUMB_SIZE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
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


class ClusterPanel(QWidget):
    """Scrollable grid of face thumbnails for the selected person.

    Signals:
        face_selected: ``(face_id: int)``
    """

    face_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
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
        self._header.setText(t("face_count_header", name=person_name, n=len(faces)))

        for i, face in enumerate(faces):
            row, col = divmod(i, _THUMB_COLS)
            thumb = FaceThumbnail(face)
            thumb.clicked.connect(self.face_selected.emit)
            self._grid.addWidget(thumb, row, col)

        # Fill remaining cells in last row
        if faces:
            remainder = len(faces) % _THUMB_COLS
            if remainder:
                for col in range(remainder, _THUMB_COLS):
                    spacer = QWidget()
                    spacer.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
                    self._grid.addWidget(spacer, len(faces) // _THUMB_COLS, col)

    def clear(self) -> None:
        self._clear_grid()
        self._header.setText(t("select_person_sidebar"))

    # ------------------------------------------------------------------

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
