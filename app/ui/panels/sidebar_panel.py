"""Sidebar panel — person / cluster list with search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Person

log = logging.getLogger(__name__)

_FACE_THUMB = 64
_FACE_COLS = 3
_POPUP_MAX = 380  # max width/height of hover popup


@dataclass
class _FaceData:
    """Plain-value snapshot of a face — safe to use after session closes."""
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h


def _build_face_data(person: Person) -> _FaceData:
    """Extract the representative face data from a Person (first face with a crop)."""
    # Prefer thumbnail_path on the person itself
    thumb = person.thumbnail_path

    # Fallback: first face that has a crop_path
    first_face = next((f for f in person.faces if f.crop_path), None)
    crop = thumb or (first_face.crop_path if first_face else None)

    image_path = None
    bbox = None
    if first_face and first_face.image:
        image_path = first_face.image.file_path
        bbox = (first_face.bbox_x, first_face.bbox_y,
                first_face.bbox_w, first_face.bbox_h)

    return _FaceData(crop_path=crop, image_path=image_path, bbox=bbox)


def _render_original_with_box(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    max_size: int,
) -> Optional[QPixmap]:
    """Load original image, draw bbox, return scaled QPixmap."""
    from app.utils.image_utils import load_image_bgr
    img = load_image_bgr(image_path)
    if img is None:
        return None

    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 200, 50), 3)

    ih, iw = img.shape[:2]
    ratio = min(max_size / iw, max_size / ih, 1.0)
    if ratio < 1.0:
        img = cv2.resize(img, (int(iw * ratio), int(ih * ratio)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h2, w2, ch = rgb.shape
    qimg = QImage(rgb.data, w2, h2, ch * w2, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _HoverPopup(QWidget):
    """Floating popup: original image with face box, and name label below."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "QWidget { background: #1a1a1a; border: 2px solid #88aaff; border-radius: 6px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("border: none;")
        layout.addWidget(self._img_label)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setStyleSheet(
            "color: #ffffff; font-size: 13px; font-weight: bold; border: none; padding: 2px;"
        )
        layout.addWidget(self._name_label)

    def show_for(self, face_data: _FaceData, name: str, global_pos) -> None:
        self._name_label.setText(name)

        pixmap = None
        if face_data.image_path and face_data.bbox:
            try:
                pixmap = _render_original_with_box(
                    face_data.image_path, face_data.bbox, _POPUP_MAX
                )
            except Exception:
                pass

        if pixmap:
            self._img_label.setPixmap(pixmap)
            self._img_label.setVisible(True)
        else:
            self._img_label.setVisible(False)

        self.adjustSize()

        screen = QApplication.primaryScreen().geometry()
        x = global_pos.x() + 16
        y = global_pos.y() - self.height() // 2
        if x + self.width() > screen.right():
            x = global_pos.x() - self.width() - 16
        y = max(screen.top() + 4, min(y, screen.bottom() - self.height() - 4))
        self.move(x, y)
        self.show()


_hover_popup: Optional[_HoverPopup] = None


def _get_hover_popup() -> _HoverPopup:
    global _hover_popup
    if _hover_popup is None:
        _hover_popup = _HoverPopup()
    return _hover_popup


class _PersonThumb(QLabel):
    """Small face thumbnail for the sidebar with hover popup showing original image."""

    clicked = Signal(int)

    def __init__(self, person: Person, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person.id
        self._person_name = person.name
        self._face_data = _build_face_data(person)
        self._load_pixmap()
        self.setFixedSize(_FACE_THUMB, _FACE_THUMB)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { border: 1px solid #555; border-radius: 4px; }"
            "QLabel:hover { border: 2px solid #88aaff; }"
        )
        self.setMouseTracking(True)

    def _load_pixmap(self) -> None:
        crop = self._face_data.crop_path
        if crop and Path(crop).exists():
            pixmap = QPixmap(crop).scaled(
                _FACE_THUMB, _FACE_THUMB,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.setPixmap(pixmap)
        else:
            self.setText("?")
            self.setStyleSheet(
                "QLabel { background: #333; color: #888; "
                "border: 1px solid #555; font-size: 14px; "
                "border-radius: 4px; }"
            )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        _get_hover_popup().show_for(
            self._face_data,
            self._person_name,
            self.mapToGlobal(self.rect().center()),
        )

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        _get_hover_popup().hide()

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._person_id)


class PersonListItem(QListWidgetItem):
    """List item that carries a Person reference."""

    def __init__(self, person: Person) -> None:
        face_count = len(person.faces)
        prefix = "🔒 " if person.is_protected else ""
        label = f"{prefix}{person.name}  ({face_count})"
        super().__init__(label)
        self.person_id = person.id
        self.person_name = person.name
        if person.is_protected:
            from PySide6.QtGui import QColor
            self.setForeground(QColor("#A6ADC8"))


class SidebarPanel(QWidget):
    """Left sidebar showing persons as face thumbnails + searchable list.

    Signals:
        person_selected: ``(person_id: int)``
    """

    person_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._all_persons: list[Person] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Face thumbnail grid at top ---
        face_box = QGroupBox("All Faces")
        face_box_layout = QVBoxLayout(face_box)
        face_box_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._thumb_container = QWidget()
        self._thumb_grid = QGridLayout(self._thumb_container)
        self._thumb_grid.setSpacing(4)
        scroll.setWidget(self._thumb_container)
        scroll.setMinimumHeight(200)
        face_box_layout.addWidget(scroll)
        layout.addWidget(face_box, stretch=1)

        # --- Searchable person list below ---
        search_box = QGroupBox("People")
        search_layout = QVBoxLayout(search_box)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search person name …")
        self._search_input.textChanged.connect(self._on_search_changed)
        search_layout.addWidget(self._search_input)

        self._person_list = QListWidget()
        self._person_list.setAlternatingRowColors(True)
        self._person_list.currentItemChanged.connect(self._on_selection_changed)
        search_layout.addWidget(self._person_list)

        self._count_label = QLabel("0 persons")
        self._count_label.setAlignment(Qt.AlignCenter)
        search_layout.addWidget(self._count_label)

        layout.addWidget(search_box, stretch=1)

        self._recluster_btn = QPushButton("Re-cluster All")
        self._recluster_btn.setToolTip("Re-run clustering with current corrections")
        layout.addWidget(self._recluster_btn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, persons: list[Person]) -> None:
        """Rebuild the list and thumbnail grid with the given persons."""
        self._all_persons = persons
        self._rebuild_thumb_grid(persons)
        self._apply_filter(self._search_input.text())

    def set_recluster_callback(self, cb: Callable) -> None:
        self._recluster_btn.clicked.connect(cb)

    def current_person_id(self) -> Optional[int]:
        item = self._person_list.currentItem()
        if isinstance(item, PersonListItem):
            return item.person_id
        return None

    # ------------------------------------------------------------------

    def _rebuild_thumb_grid(self, persons: list[Person]) -> None:
        while self._thumb_grid.count():
            item = self._thumb_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for i, person in enumerate(persons):
            row, col = divmod(i, _FACE_COLS)
            thumb = _PersonThumb(person)
            thumb.clicked.connect(self.person_selected.emit)
            self._thumb_grid.addWidget(thumb, row, col)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_search_changed(self, text: str) -> None:
        self._apply_filter(text)

    def _apply_filter(self, text: str) -> None:
        self._person_list.clear()
        persons = self._all_persons
        text = text.strip().lower()

        shown = 0
        for person in persons:
            if text and text not in person.name.lower():
                continue
            item = PersonListItem(person)
            self._person_list.addItem(item)
            shown += 1

        self._count_label.setText(f"{shown} person(s)")

    def _on_selection_changed(
        self, current: QListWidgetItem, previous: QListWidgetItem
    ) -> None:
        if isinstance(current, PersonListItem):
            self.person_selected.emit(current.person_id)
