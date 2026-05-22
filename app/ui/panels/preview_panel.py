"""Preview panel — shows the original image with all face bounding boxes and names."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face

log = logging.getLogger(__name__)


def _draw_faces(img_bgr: np.ndarray, faces, selected_id: int) -> np.ndarray:
    """Draw all face boxes with names; selected face in bright green, others in gray."""
    img = img_bgr.copy()
    for f in faces:
        x, y, w, h = f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h
        selected = f.id == selected_id
        color = (50, 200, 50) if selected else (160, 160, 160)
        thickness = 3 if selected else 2

        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        name = (f.person.name if f.person else "?") if hasattr(f, "person") else "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(1.1, w / 80))
        (tw, th), bl = cv2.getTextSize(name, font, scale, 2)
        ty = max(y - 6, th + 6)
        cv2.rectangle(img, (x, ty - th - bl - 4), (x + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(img, name, (x + 3, ty - bl), font, scale, color, 2)
    return img


def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _ZoomDialog(QDialog):
    """Fullscreen-ish dialog showing the image at full resolution with scroll."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nagyítás / Zoom")
        screen = parent.screen().availableGeometry() if parent else pixmap.rect()
        self.resize(min(pixmap.width() + 40, screen.width() - 60),
                    min(pixmap.height() + 80, screen.height() - 80))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        img_label = QLabel()
        img_label.setPixmap(pixmap)
        img_label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(img_label)
        layout.addWidget(scroll)

        close_btn = QPushButton("Bezárás / Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


class _ClickableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._on_click = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._on_click:
            self._on_click()


class PreviewPanel(QWidget):
    """Shows a full image preview with all faces highlighted and named."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image_path: Optional[str] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._image_label = _ClickableLabel()
        self._image_label.setText("Click a face thumbnail to preview")
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(300, 200)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet(
            "QLabel { background: #222; border: 1px solid #444; }"
        )
        self._image_label.setToolTip("Kattints a nagyításhoz / Click to zoom")
        self._image_label._on_click = self._open_zoom
        layout.addWidget(self._image_label)

        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; }")
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton("Open in File Manager")
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_in_filemanager)
        btn_row.addWidget(self._open_btn)
        self._zoom_btn = QPushButton("🔍 Nagyítás / Zoom")
        self._zoom_btn.setEnabled(False)
        self._zoom_btn.clicked.connect(self._open_zoom)
        btn_row.addWidget(self._zoom_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def show_face(self, face: Face) -> None:
        if face.image is None:
            self._image_label.setText("(no image reference)")
            return

        img_path = face.image.file_path
        self._current_image_path = img_path

        from app.utils.image_utils import load_image_bgr
        img_bgr = load_image_bgr(img_path)
        if img_bgr is None:
            self._image_label.setText(f"Cannot load:\n{img_path}")
            return

        annotated = _draw_faces(img_bgr, face.image.faces, selected_id=face.id)
        self._full_pixmap = _bgr_to_qpixmap(annotated)
        self._update_scaled_pixmap()
        self._path_label.setText(img_path)
        self._open_btn.setEnabled(True)
        self._zoom_btn.setEnabled(True)

    def clear(self) -> None:
        self._full_pixmap = None
        self._image_label.clear()
        self._image_label.setText("Click a face thumbnail to preview")
        self._path_label.setText("")
        self._open_btn.setEnabled(False)
        self._zoom_btn.setEnabled(False)
        self._current_image_path = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._full_pixmap is None:
            return
        w = self._image_label.width()
        h = self._image_label.height()
        scaled = self._full_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)

    def _open_zoom(self) -> None:
        if self._full_pixmap is None:
            return
        dlg = _ZoomDialog(self._full_pixmap, parent=self)
        dlg.exec()

    # ------------------------------------------------------------------

    def _open_in_filemanager(self) -> None:
        if not self._current_image_path:
            return
        path = Path(self._current_image_path)
        if not path.exists():
            log.warning("File not found: %s", path)
            return

        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path.parent)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(path)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", "/select,", str(path)])
        except OSError as exc:
            log.warning("Cannot open file manager: %s", exc)
