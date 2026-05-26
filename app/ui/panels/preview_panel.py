"""Preview panel — shows the original image with all face bounding boxes and names."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face
from app.ui.i18n import t

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None)
_FaceData = Tuple[int, int, int, int, int, Optional[str]]


def _get_pil_font(size: int):
    import sys

    from PIL import ImageFont

    if sys.platform == "darwin":
        candidates = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
        ]
    elif sys.platform == "win32":
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    from PIL import ImageFont
    return ImageFont.load_default()


def _draw_faces(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
) -> np.ndarray:
    """Draw all face boxes with names; selected face in bright green, others in gray."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    img = img_bgr.copy()

    for face_id, x, y, w, h, _ in faces:
        selected = face_id == selected_id
        color = (50, 220, 50) if selected else (180, 180, 180)
        thickness = 3 if selected else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    image_h, image_w = img.shape[:2]
    font_size = max(34, min(96, int(min(image_w, image_h) * 0.028)))

    for face_id, x, y, w, h, person_name in faces:
        selected = face_id == selected_id
        color_rgb = (50, 220, 50) if selected else (180, 180, 180)
        name = person_name or "?"
        label_font_size = font_size
        font = _get_pil_font(label_font_size)

        bbox = draw.textbbox((0, 0), name, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad_x = max(8, label_font_size // 4)
        pad_y = max(5, label_font_size // 7)
        max_label_w = max(80, image_w - 8)
        while tw + 2 * pad_x > max_label_w and label_font_size > 26:
            label_font_size -= 2
            font = _get_pil_font(label_font_size)
            bbox = draw.textbbox((0, 0), name, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            pad_x = max(8, label_font_size // 4)
            pad_y = max(5, label_font_size // 7)

        label_w = min(tw + 2 * pad_x, image_w)
        label_h = th + 2 * pad_y
        label_x = min(max(x, 0), max(0, image_w - label_w))
        label_y = y - label_h - 6
        if label_y < 0:
            label_y = y + h + 6
        label_y = min(max(label_y, 0), max(0, image_h - label_h))

        draw.rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            fill=(20, 20, 20),
        )
        draw.text(
            (label_x + pad_x, label_y + pad_y),
            name,
            font=font,
            fill=color_rgb,
        )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class _ZoomDialog(QDialog):
    """Fullscreen-ish dialog showing the image at full resolution with scroll."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("zoom"))
        screen = parent.screen().availableGeometry() if parent else pixmap.rect()
        self.resize(min(pixmap.width() + 40, screen.width() - 60),
                    min(pixmap.height() + 80, screen.height() - 80))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        scroll = _WheelZoomScrollArea(pixmap)
        scroll.setWidgetResizable(False)
        layout.addWidget(scroll)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


class _WheelZoomScrollArea(QScrollArea):
    """Scroll area that zooms the image with the mouse wheel."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source_pixmap = pixmap
        self._zoom = 1.0
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setPixmap(pixmap)
        self._image_label.resize(pixmap.size())
        self.setWidget(self._image_label)

    def wheelEvent(self, event) -> None:
        if self._source_pixmap.isNull():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return

        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(0.1, min(12.0, self._zoom * factor))
        if new_zoom == self._zoom:
            event.accept()
            return

        cursor_pos = event.position()
        old_x = self.horizontalScrollBar().value() + cursor_pos.x()
        old_y = self.verticalScrollBar().value() + cursor_pos.y()
        scale = new_zoom / self._zoom

        self._zoom = new_zoom
        scaled_size = self._source_pixmap.size() * self._zoom
        scaled = self._source_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.resize(scaled.size())

        self.horizontalScrollBar().setValue(int(old_x * scale - cursor_pos.x()))
        self.verticalScrollBar().setValue(int(old_y * scale - cursor_pos.y()))
        event.accept()


class _FaceImageLabel(QLabel):
    """QLabel that detects face clicks and can draw a manual face rectangle."""

    face_clicked = Signal(int)        # face_id clicked with left button
    canvas_clicked = Signal()         # left click outside all faces → open zoom
    face_right_clicked = Signal(int, int, int)  # face_id, global_x, global_y
    rect_drawn = Signal(QRect)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._face_data: List[Tuple[int, int, int, int, int]] = []  # (id, x, y, w, h)
        self._full_w: int = 0
        self._full_h: int = 0
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None

    def set_face_data(
        self,
        faces: List[Tuple[int, int, int, int, int]],
        full_w: int,
        full_h: int,
    ) -> None:
        self._face_data = faces
        self._full_w = full_w
        self._full_h = full_h

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._start = None
        self._end = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.PointingHandCursor)
        self.update()

    def _label_to_image(self, lx: float, ly: float) -> Tuple[int, int]:
        if self._full_w == 0 or self._full_h == 0:
            return -1, -1
        lw, lh = self.width(), self.height()
        if lw == 0 or lh == 0:
            return -1, -1
        scale = min(lw / self._full_w, lh / self._full_h)
        disp_w = self._full_w * scale
        disp_h = self._full_h * scale
        ox = (lw - disp_w) / 2
        oy = (lh - disp_h) / 2
        rx = lx - ox
        ry = ly - oy
        if rx < 0 or ry < 0 or rx >= disp_w or ry >= disp_h:
            return -1, -1
        return int(rx / scale), int(ry / scale)

    def _hit_test(self, lx: float, ly: float) -> Optional[int]:
        ix, iy = self._label_to_image(lx, ly)
        if ix < 0:
            return None
        for face_id, x, y, w, h in self._face_data:
            if x <= ix <= x + w and y <= iy <= y + h:
                return face_id
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._draw_mode:
                self._start = event.position().toPoint()
                self._end = self._start
                self.update()
                return
            pos = event.position()
            face_id = self._hit_test(pos.x(), pos.y())
            if face_id is not None:
                self.face_clicked.emit(face_id)
            else:
                self.canvas_clicked.emit()

    def mouseMoveEvent(self, event) -> None:
        if self._draw_mode and self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._draw_mode or self._start is None or event.button() != Qt.LeftButton:
            return
        rect = QRect(self._start, event.position().toPoint()).normalized()
        self._start = None
        self._end = None
        self.update()
        if rect.width() >= 8 and rect.height() >= 8:
            self.rect_drawn.emit(rect)

    def contextMenuEvent(self, event) -> None:
        face_id = self._hit_test(event.pos().x(), event.pos().y())
        if face_id is not None:
            gp = self.mapToGlobal(event.pos())
            self.face_right_clicked.emit(face_id, gp.x(), gp.y())

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._draw_mode and self._start is not None and self._end is not None:
            painter = QPainter(self)
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.drawRect(QRect(self._start, self._end).normalized())


class PreviewPanel(QWidget):
    """Shows a full image preview with all faces highlighted and named.

    Signals:
        face_selected: ``(face_id: int)`` — emitted when user clicks a face in the image.
        face_assign_requested: ``(face_id: int)`` — user chose "Személyhez adás" from context menu.
        face_delete_requested: ``(face_id: int)`` — user chose "Törlés" from context menu.
        face_create_requested: ``(image_id, x, y, w, h)`` — user drew a new face.
        face_bbox_update_requested: ``(face_id, x, y, w, h)`` — user redrew a face bbox.
    """

    face_selected = Signal(int)
    face_assign_requested = Signal(int)
    face_delete_requested = Signal(int)
    face_create_requested = Signal(int, int, int, int, int)
    face_bbox_update_requested = Signal(int, int, int, int, int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image_path: Optional[str] = None
        self._current_image_id: Optional[int] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._face_data: List[_FaceData] = []
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._image_label = _FaceImageLabel()
        self._image_label.setText(t("preview_empty"))
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(300, 200)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet(
            "QLabel { background: #222; border: 1px solid #444; }"
        )
        self._image_label.setToolTip(
            t("preview_tip")
        )
        self._image_label.face_clicked.connect(self._on_face_clicked)
        self._image_label.canvas_clicked.connect(self._open_zoom)
        self._image_label.face_right_clicked.connect(self._on_face_right_clicked)
        self._image_label.rect_drawn.connect(self._on_rect_drawn)
        layout.addWidget(self._image_label)

        self._draw_hint = QLabel(t("draw_face_hint"))
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        layout.addWidget(self._draw_hint)

        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setMinimumWidth(0)
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._path_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; }")
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        self._open_btn = QPushButton(t("open_file_manager"))
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_in_filemanager)
        btn_row.addWidget(self._open_btn)
        self._zoom_btn = QPushButton(f"🔍 {t('zoom')}")
        self._zoom_btn.setEnabled(False)
        self._zoom_btn.clicked.connect(self._open_zoom)
        btn_row.addWidget(self._zoom_btn)
        self._draw_btn = QPushButton(f"✏ {t('selection')}")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setEnabled(False)
        self._draw_btn.setToolTip(t("draw_face_hint"))
        self._draw_btn.toggled.connect(self._on_draw_mode_toggled)
        btn_row.addWidget(self._draw_btn)
        self._edit_btn = QPushButton(t("modify_selection"))
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._start_selected_face_edit)
        btn_row.addWidget(self._edit_btn)
        self._assign_btn = QPushButton(t("assign_to_person"))
        self._assign_btn.setEnabled(False)
        self._assign_btn.clicked.connect(self._assign_selected_face)
        btn_row.addWidget(self._assign_btn)
        self._delete_btn = QPushButton(t("delete_selection"))
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_selected_face)
        btn_row.addWidget(self._delete_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_face(self, face: Face) -> None:
        """Load and display the image for *face*, highlighting all faces.

        Draw mode is preserved when the same image is refreshed (e.g. after a
        manual face is created while draw mode was active), so the user can
        keep drawing additional faces without having to re-enable the mode.
        """
        if face.image is None:
            self._image_label.setText(t("no_recognized_face"))
            return

        img_path = face.image.file_path
        # Remember draw state BEFORE we reset it — restore if staying on same image
        same_image = (face.image.id == self._current_image_id)
        preserve_draw = (
            same_image
            and self._draw_btn.isChecked()
            and self._editing_face_id is None
        )

        self._current_image_path = img_path
        self._current_image_id = face.image.id

        from app.utils.image_utils import load_image_bgr
        img_bgr = load_image_bgr(img_path)
        if img_bgr is None:
            self._image_label.setText(t("cannot_load", path=img_path))
            return

        self._orig_img_bgr = img_bgr
        self._face_data = [
            (
                f.id,
                f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                f.person.name if f.person else None,
            )
            for f in face.image.faces
            if not f.is_excluded
        ]
        self._selected_face_id = face.id
        self._editing_face_id = None
        self._draw_btn.setChecked(False)

        self._image_label.set_face_data(
            [(fd[0], fd[1], fd[2], fd[3], fd[4]) for fd in self._face_data],
            img_bgr.shape[1],
            img_bgr.shape[0],
        )
        self._render()

        self._path_label.setText(img_path)
        self._open_btn.setEnabled(True)
        self._zoom_btn.setEnabled(True)
        self._draw_btn.setEnabled(True)
        self._update_action_buttons()

        # Restore draw mode so the user can continue adding faces without
        # having to click the draw button again after each creation.
        if preserve_draw:
            self._draw_btn.setChecked(True)

    def select_face(self, face_id: int) -> None:
        """Change the highlighted face without reloading the image from disk."""
        if self._selected_face_id == face_id:
            return
        self._selected_face_id = face_id
        self._render()
        self._update_action_buttons()

    def clear(self) -> None:
        self._full_pixmap = None
        self._orig_img_bgr = None
        self._face_data = []
        self._selected_face_id = None
        self._editing_face_id = None
        self._current_image_id = None
        self._image_label.set_face_data([], 0, 0)
        self._image_label.set_draw_mode(False)
        self._image_label.clear()
        self._image_label.setText(t("preview_empty"))
        self._path_label.setText("")
        self._open_btn.setEnabled(False)
        self._zoom_btn.setEnabled(False)
        self._draw_btn.setChecked(False)
        self._draw_btn.setEnabled(False)
        self._draw_hint.setVisible(False)
        self._edit_btn.setEnabled(False)
        self._assign_btn.setEnabled(False)
        self._delete_btn.setEnabled(False)
        self._current_image_path = None

    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _render(self) -> None:
        if self._orig_img_bgr is None:
            return
        annotated = _draw_faces(self._orig_img_bgr, self._face_data, self._selected_face_id)
        self._full_pixmap = _bgr_to_qpixmap(annotated)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._full_pixmap is None:
            return
        w = self._image_label.width()
        h = self._image_label.height()
        scaled = self._full_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)

    # ------------------------------------------------------------------
    # Face interaction
    # ------------------------------------------------------------------

    def _on_face_clicked(self, face_id: int) -> None:
        """Highlight the clicked face and notify parent."""
        self._selected_face_id = face_id
        self._render()
        self._update_action_buttons()
        self.face_selected.emit(face_id)

    def _on_face_right_clicked(self, face_id: int, gx: int, gy: int) -> None:
        """Show context menu for the right-clicked face."""
        # Select the face immediately so it's highlighted when the menu appears
        if self._selected_face_id != face_id:
            self._selected_face_id = face_id
            self._render()
            self._update_action_buttons()
            self.face_selected.emit(face_id)

        entry = next((f for f in self._face_data if f[0] == face_id), None)
        person_name = entry[5] if entry else None

        menu = QMenu(self)

        title = menu.addAction(f"👤  {person_name}" if person_name else f"👤  {t('unknown_face')}")
        title.setEnabled(False)
        menu.addSeparator()

        assign_action = menu.addAction(f"👤  {t('assign_to_person')}")
        edit_action = menu.addAction(f"✏  {t('modify_selection')}")
        delete_action = menu.addAction(f"🗑  {t('delete_selection')}")

        chosen = menu.exec(QPoint(gx, gy))

        if chosen == assign_action:
            self.face_assign_requested.emit(face_id)
        elif chosen == edit_action:
            self._start_face_edit(face_id)
        elif chosen == delete_action:
            self.face_delete_requested.emit(face_id)

    def _select_face_for_action(self, face_id: int) -> None:
        self._selected_face_id = face_id
        self._render()
        self._update_action_buttons()
        self.face_selected.emit(face_id)

    def _on_draw_mode_toggled(self, active: bool) -> None:
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if not active:
            self._editing_face_id = None
            self._draw_hint.setText(t("draw_face_hint"))

    def _on_rect_drawn(self, label_rect: QRect) -> None:
        coords = self._label_rect_to_image(label_rect)
        if coords is None or self._current_image_id is None:
            return
        x, y, w, h = coords
        if self._editing_face_id is not None:
            face_id = self._editing_face_id
            self._editing_face_id = None
            self._draw_hint.setText(t("draw_face_hint"))
            self._selected_face_id = face_id
            self.face_bbox_update_requested.emit(face_id, x, y, w, h)
        else:
            self.face_create_requested.emit(self._current_image_id, x, y, w, h)

    def _label_rect_to_image(self, rect: QRect) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        x2, y2 = self._label_to_image(rect.right(), rect.bottom())
        if x1 < 0 or y1 < 0:
            return None
        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
            x2 = min(max(x2, 0), iw - 1)
            y2 = min(max(y2, 0), ih - 1)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _label_to_image(self, lx: float, ly: float) -> Tuple[int, int]:
        if self._orig_img_bgr is None:
            return -1, -1
        full_h, full_w = self._orig_img_bgr.shape[:2]
        lw, lh = self._image_label.width(), self._image_label.height()
        if full_w == 0 or full_h == 0 or lw == 0 or lh == 0:
            return -1, -1
        scale = min(lw / full_w, lh / full_h)
        disp_w = full_w * scale
        disp_h = full_h * scale
        ox = (lw - disp_w) / 2
        oy = (lh - disp_h) / 2
        rx = lx - ox
        ry = ly - oy
        if rx < 0 or ry < 0 or rx >= disp_w or ry >= disp_h:
            return -1, -1
        return int(rx / scale), int(ry / scale)

    def _start_selected_face_edit(self) -> None:
        if self._selected_face_id is not None:
            self._start_face_edit(self._selected_face_id)

    def _start_face_edit(self, face_id: int) -> None:
        self._editing_face_id = face_id
        self._selected_face_id = face_id
        self._render()
        self._update_action_buttons()
        self._draw_hint.setText(t("redraw_face_hint"))
        self._draw_btn.setChecked(True)

    def _assign_selected_face(self) -> None:
        if self._selected_face_id is not None:
            self.face_assign_requested.emit(self._selected_face_id)

    def _delete_selected_face(self) -> None:
        if self._selected_face_id is not None:
            self.face_delete_requested.emit(self._selected_face_id)

    def _update_action_buttons(self) -> None:
        has_image = self._full_pixmap is not None
        has_face = self._selected_face_id is not None
        self._draw_btn.setEnabled(has_image)
        self._edit_btn.setEnabled(has_face)
        self._assign_btn.setEnabled(has_face)
        self._delete_btn.setEnabled(has_face)

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
