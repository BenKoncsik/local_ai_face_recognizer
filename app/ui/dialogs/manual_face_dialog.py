"""Dialogs for browsing images without detected faces and marking faces manually."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QSize, Signal
from PySide6.QtGui import QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.config import AppConfig
from app.db.database import session_scope
from app.db.models import Face, Image
from app.detectors.base import Detection
from app.ui.i18n import t
from app.utils.image_utils import load_image_bgr, save_face_crop

log = logging.getLogger(__name__)


# ── Marker widget ────────────────────────────────────────────────────────────

class _MarkerLabel(QLabel):
    """Label that lets the user draw a rectangle with the mouse."""

    rect_drawn = Signal(QRect)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._rect: Optional[QRect] = None
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton:
            self._start = ev.position().toPoint()
            self._end = self._start
            self._rect = None
            self.update()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._start is not None:
            self._end = ev.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._start is None:
            return
        end = ev.position().toPoint()
        rect = QRect(self._start, end).normalized()
        self._start = None
        self._end = None
        if rect.width() >= 10 and rect.height() >= 10:
            self._rect = rect
            self.rect_drawn.emit(rect)
        self.update()

    def current_rect(self) -> Optional[QRect]:
        return self._rect

    def clear_rect(self) -> None:
        self._rect = None
        self.update()

    def paintEvent(self, ev) -> None:  # noqa: ANN001
        super().paintEvent(ev)
        painter = QPainter(self)
        pen = QPen(Qt.green, 2, Qt.SolidLine)
        painter.setPen(pen)
        if self._start is not None and self._end is not None:
            painter.drawRect(QRect(self._start, self._end).normalized())
        elif self._rect is not None:
            painter.drawRect(self._rect)


# ── Manual-marking dialog ────────────────────────────────────────────────────

class ManualMarkDialog(QDialog):
    """Show an image and let the user drag a rectangle to mark a face."""

    _MAX_W = 900
    _MAX_H = 700

    def __init__(self, image_id: int, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._image_id = image_id
        self._config = config
        self._saved = False
        self._img_bgr: Optional[np.ndarray] = None
        self._display_scale: float = 1.0
        self._display_offset: Tuple[int, int] = (0, 0)

        with session_scope() as session:
            image = session.get(Image, image_id)
            if image is None:
                raise ValueError(f"Image {image_id} not found")
            self._image_path = image.file_path

        self.setWindowTitle(t("mark_face"))
        self._build_ui()
        self._load_image()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        hint = QLabel(t("mark_face_hint"))
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        self._marker = _MarkerLabel()
        self._marker.setAlignment(Qt.AlignCenter)
        self._marker.setStyleSheet("background: #222;")
        self._marker.rect_drawn.connect(self._on_rect_drawn)
        layout.addWidget(self._marker)

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._save_btn = btns.button(QDialogButtonBox.Save)
        self._save_btn.setEnabled(False)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _load_image(self) -> None:
        self._img_bgr = load_image_bgr(self._image_path)
        if self._img_bgr is None:
            QMessageBox.warning(self, t("error"), t("cannot_load", path=self._image_path))
            self.reject()
            return

        h, w = self._img_bgr.shape[:2]
        scale = min(self._MAX_W / w, self._MAX_H / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        self._display_scale = scale

        resized = cv2.resize(self._img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, disp_w, disp_h, 3 * disp_w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        self._marker.setPixmap(pixmap)
        self._marker.setFixedSize(disp_w, disp_h)

    def _on_rect_drawn(self, rect: QRect) -> None:
        self._save_btn.setEnabled(rect.width() >= 10 and rect.height() >= 10)

    def _on_save(self) -> None:
        rect = self._marker.current_rect()
        if rect is None or self._img_bgr is None:
            return

        scale = self._display_scale
        x = int(rect.x() / scale)
        y = int(rect.y() / scale)
        w = int(rect.width() / scale)
        h = int(rect.height() / scale)

        img_h, img_w = self._img_bgr.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = max(1, min(w, img_w - x))
        h = max(1, min(h, img_h - y))

        detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0)
        crops_dir = self._config.crops_dir_resolved
        crops_dir.mkdir(parents=True, exist_ok=True)

        with session_scope() as session:
            existing = session.query(Face).filter(Face.image_id == self._image_id).count()
            crop_path = save_face_crop(
                img_bgr=self._img_bgr,
                detection=detection,
                crops_dir=crops_dir,
                image_id=self._image_id,
                thumbnail_size=self._config.scan.thumbnail_size,
                face_index=existing,
            )
            face = Face(
                image_id=self._image_id,
                bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                confidence=1.0,
                detector_backend="manual",
                crop_path=str(crop_path) if crop_path else None,
            )
            session.add(face)

        log.info("Manual face added for image_id=%d at (%d,%d,%d,%d)", self._image_id, x, y, w, h)
        self._saved = True
        QMessageBox.information(self, t("mark_face"), t("mark_face_saved"))
        self.accept()

    def was_saved(self) -> bool:
        return self._saved


_THUMB_SIZE = 64


def _make_thumbnail_icon(file_path: str) -> Optional[QIcon]:
    """Load and scale an image to a small icon for list display."""
    img = load_image_bgr(file_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = _THUMB_SIZE / max(w, h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, nw, nh, 3 * nw, QImage.Format_RGB888)
    return QIcon(QPixmap.fromImage(qimg.copy()))


# ── No-face image list dialog ────────────────────────────────────────────────

class NoFaceImagesDialog(QDialog):
    """List images with no detected faces; double-click to mark manually."""

    changed = Signal()

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config
        self.setWindowTitle(t("view_no_face"))
        self.setMinimumSize(600, 500)
        self._build_ui()
        self._reload()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._count_label = QLabel("")
        layout.addWidget(self._count_label)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        self._list.itemDoubleClicked.connect(self._on_open_marker)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        self._mark_btn = QPushButton(t("mark_face"))
        self._mark_btn.setEnabled(False)
        self._mark_btn.clicked.connect(self._on_mark_selected)
        btn_row.addWidget(self._mark_btn)
        btn_row.addStretch()

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self._list.currentItemChanged.connect(
            lambda cur, _prev: self._mark_btn.setEnabled(cur is not None)
        )

    def _reload(self) -> None:
        self._list.clear()
        with session_scope() as session:
            no_face: List[Image] = (
                session.query(Image)
                .filter(Image.detection_done == True)  # noqa: E712
                .filter(~Image.faces.any())
                .order_by(Image.file_path)
                .all()
            )
            self._count_label.setText(t("n_images_no_face", n=len(no_face)))
            for img in no_face:
                item = QListWidgetItem(Path(img.file_path).name)
                item.setData(Qt.UserRole, img.id)
                item.setToolTip(img.file_path)
                icon = _make_thumbnail_icon(img.file_path)
                if icon:
                    item.setIcon(icon)
                self._list.addItem(item)

    def _on_open_marker(self, item: QListWidgetItem) -> None:
        image_id = item.data(Qt.UserRole)
        if image_id is None:
            return
        try:
            dlg = ManualMarkDialog(image_id=image_id, config=self._config, parent=self)
        except ValueError as exc:
            QMessageBox.warning(self, t("error"), str(exc))
            return
        if dlg.exec() == QDialog.Accepted and dlg.was_saved():
            self.changed.emit()
            self._reload()

    def _on_mark_selected(self) -> None:
        item = self._list.currentItem()
        if item is not None:
            self._on_open_marker(item)
