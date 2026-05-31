"""Preview panel — shows the original image with all face bounding boxes and names.

Overlay architecture:
  - The base image is stored as a clean QPixmap (no baked-in annotations).
  - _FaceImageLabel.paintEvent() draws bounding boxes and labels via QPainter
    on top of the clean image, using separately controllable opacity values.
  - Slider movement triggers only overlay.update() — no image reload.
  - Face selection change (select_face) also only triggers an overlay repaint.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face
from app.ui.i18n import t

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None)
_FaceData = Tuple[int, int, int, int, int, Optional[str]]

# Colours used for bounding-box / label rendering
_COLOR_SELECTED = QColor(50, 220, 50)
_COLOR_HOVER    = QColor(255, 200, 60)
_COLOR_NORMAL   = QColor(180, 180, 180)

_BG_SELECTED = QColor(10,  30, 10, 210)
_BG_HOVER    = QColor(30,  25,  5, 210)
_BG_NORMAL   = QColor(20,  20, 20, 200)

_BORDER_SELECTED = QColor(50,  220, 50,  180)
_BORDER_HOVER    = QColor(255, 200, 60,  180)
_BORDER_NORMAL   = QColor(100, 100, 100, 120)


# ---------------------------------------------------------------------------
# PIL-based renderer kept for the full-resolution Zoom dialog
# ---------------------------------------------------------------------------

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


def _draw_faces_pil(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
) -> np.ndarray:
    """Draw face boxes and labels using OpenCV + PIL (for the zoom dialog)."""
    from PIL import Image as PILImage
    from PIL import ImageDraw

    from app.ui.helpers.label_placement import FaceLabel, place_labels

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

    face_meta = []
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
        face_meta.append((name, font, pad_x, pad_y, label_w, label_h, color_rgb))

    face_labels = [
        FaceLabel(
            face_id=faces[i][0],
            x=faces[i][1], y=faces[i][2],
            w=faces[i][3], h=faces[i][4],
            selected=faces[i][0] == selected_id,
            label_w=face_meta[i][4],
            label_h=face_meta[i][5],
        )
        for i in range(len(faces))
    ]
    layouts = place_labels(image_w, image_h, face_labels)

    for i, layout in enumerate(layouts):
        name, font, pad_x, pad_y, label_w, label_h, color_rgb = face_meta[i]

        if layout.leader_start and layout.leader_end:
            draw.line([layout.leader_start, layout.leader_end],
                      fill=(120, 120, 120), width=2)

        draw.rectangle(
            [layout.label_x, layout.label_y,
             layout.label_x + label_w, layout.label_y + label_h],
            fill=(20, 20, 20),
        )
        draw.text(
            (layout.label_x + pad_x, layout.label_y + pad_y),
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


# ---------------------------------------------------------------------------
# Zoom dialog
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Face image label with QPainter overlay
# ---------------------------------------------------------------------------

class _FaceImageLabel(QLabel):
    """QLabel that detects face clicks, supports draw mode, and renders
    face bounding-box / label overlays directly via QPainter.

    The base image pixmap is kept clean (no baked-in annotations).  All
    overlay drawing happens in paintEvent, controlled by opacity sliders.
    """

    face_clicked = Signal(int)
    canvas_clicked = Signal()
    face_right_clicked = Signal(int, int, int)
    rect_drawn = Signal(QRect)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)

        # Hit-test data (image coordinates)
        self._face_data: List[Tuple[int, int, int, int, int]] = []  # (id, x, y, w, h)
        self._full_w: int = 0
        self._full_h: int = 0

        # Extended render data including names
        self._face_render_data: List[_FaceData] = []  # (id, x, y, w, h, name)

        # Draw-mode state
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None

        # Overlay settings
        self._selected_face_id: Optional[int] = None
        self._hover_face_id: Optional[int] = None
        self._show_bboxes: bool = True
        self._bbox_opacity: float = 0.7
        self._show_labels: bool = True
        self._label_opacity: float = 0.4

        # Layout cache: avoids re-running place_labels() on every repaint
        # when nothing has changed.
        self._layout_cache_key: Optional[tuple] = None
        self._layout_cache: Optional[tuple] = None  # (label_sizes, layouts)

    # ------------------------------------------------------------------
    # Public setters
    # ------------------------------------------------------------------

    def set_face_data(
        self,
        faces: List[_FaceData],
        full_w: int,
        full_h: int,
    ) -> None:
        self._face_render_data = faces
        self._face_data = [(fd[0], fd[1], fd[2], fd[3], fd[4]) for fd in faces]
        self._full_w = full_w
        self._full_h = full_h
        self._layout_cache_key = None
        self.update()

    def set_overlay_settings(
        self,
        show_bboxes: bool,
        bbox_opacity: float,
        show_labels: bool,
        label_opacity: float,
        selected_id: Optional[int],
    ) -> None:
        self._show_bboxes = show_bboxes
        self._bbox_opacity = bbox_opacity
        self._show_labels = show_labels
        self._label_opacity = label_opacity
        self._selected_face_id = selected_id
        self.update()

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._start = None
        self._end = None
        self.setCursor(Qt.CrossCursor if enabled else Qt.PointingHandCursor)
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _display_transform(self) -> Optional[Tuple[float, float, float]]:
        """Return (scale, offset_x, offset_y) for image→display mapping.

        Returns None when dimensions are unknown.
        """
        if self._full_w == 0 or self._full_h == 0:
            return None
        lw, lh = self.width(), self.height()
        if lw == 0 or lh == 0:
            return None
        scale = min(lw / self._full_w, lh / self._full_h)
        ox = (lw - self._full_w * scale) / 2
        oy = (lh - self._full_h * scale) / 2
        return scale, ox, oy

    def _label_to_image(self, lx: float, ly: float) -> Tuple[int, int]:
        t = self._display_transform()
        if t is None:
            return -1, -1
        scale, ox, oy = t
        rx = lx - ox
        ry = ly - oy
        disp_w = self._full_w * scale
        disp_h = self._full_h * scale
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

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            if self._draw_mode:
                self._start = event.position().toPoint()
                self._end = self._start
                log.debug("pointer down: draw start at (%d,%d)", self._start.x(), self._start.y())
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
            return
        # Hover tracking
        pos = event.position()
        new_hover = self._hit_test(pos.x(), pos.y())
        if new_hover != self._hover_face_id:
            self._hover_face_id = new_hover
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if not self._draw_mode or self._start is None or event.button() != Qt.LeftButton:
            return
        rect = QRect(self._start, event.position().toPoint()).normalized()
        log.debug(
            "pointer up: raw label_rect=(%d,%d,%d,%d) draw_mode=%s",
            rect.x(), rect.y(), rect.width(), rect.height(), self._draw_mode,
        )
        self._start = None
        self._end = None
        self.update()
        if rect.width() >= 8 and rect.height() >= 8:
            self.rect_drawn.emit(rect)
        else:
            log.debug("pointer up: rect too small (%dx%d) — ignored", rect.width(), rect.height())

    def leaveEvent(self, event) -> None:
        if self._hover_face_id is not None:
            self._hover_face_id = None
            self.update()

    def contextMenuEvent(self, event) -> None:
        face_id = self._hit_test(event.pos().x(), event.pos().y())
        if face_id is not None:
            gp = self.mapToGlobal(event.pos())
            self.face_right_clicked.emit(face_id, gp.x(), gp.y())

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        needs_overlay = bool(self._face_render_data)
        needs_rubber  = self._draw_mode and self._start is not None and self._end is not None
        if not needs_overlay and not needs_rubber:
            return
        t = self._display_transform()
        self._draw_overlay(t)  # handles both overlay and rubber band; t may be None

    def _get_label_layouts(
        self,
        disp_faces: List[Tuple],
        disp_w: int,
        disp_h: int,
        font: QFont,
    ) -> Tuple[List[Tuple], List]:
        """Return (label_sizes, layouts), using cache when possible."""
        from app.ui.helpers.label_placement import FaceLabel, place_labels

        cache_key = (
            disp_w, disp_h,
            self._selected_face_id,
            tuple((f[0], int(f[1]), int(f[2]), int(f[3]), int(f[4])) for f in disp_faces),
        )
        if self._layout_cache_key == cache_key and self._layout_cache is not None:
            return self._layout_cache

        metrics = QFontMetrics(font)
        pad_x = max(4, font.pixelSize() // 4) if font.pixelSize() > 0 else 6
        pad_y = max(3, font.pixelSize() // 7) if font.pixelSize() > 0 else 4

        label_sizes: List[Tuple[str, int, int, int, int]] = []
        for face_id, dx, dy, dw, dh, name in disp_faces:
            display_name = name or "?"
            tw = metrics.horizontalAdvance(display_name)
            th = metrics.height()
            label_sizes.append((display_name, tw + 2 * pad_x, th + 2 * pad_y, pad_x, pad_y))

        face_labels = [
            FaceLabel(
                face_id=disp_faces[i][0],
                x=int(disp_faces[i][1]),
                y=int(disp_faces[i][2]),
                w=max(1, int(disp_faces[i][3])),
                h=max(1, int(disp_faces[i][4])),
                selected=(disp_faces[i][0] == self._selected_face_id),
                label_w=label_sizes[i][1],
                label_h=label_sizes[i][2],
            )
            for i in range(len(disp_faces))
        ]
        layouts = place_labels(disp_w, disp_h, face_labels)

        result = (label_sizes, layouts)
        self._layout_cache_key = cache_key
        self._layout_cache = result
        return result

    def _draw_overlay(self, transform: Optional[Tuple[float, float, float]]) -> None:
        # Rubber band when in draw mode (may work without transform)
        if self._draw_mode and self._start is not None and self._end is not None:
            p = QPainter(self)
            p.setOpacity(1.0)
            p.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QRect(self._start, self._end).normalized())
            p.end()

        if transform is None or not self._face_render_data:
            return

        scale, ox, oy = transform
        disp_w = int(self._full_w * scale)
        disp_h = int(self._full_h * scale)

        # Build display-space face list
        disp_faces: List[Tuple] = []
        for face_id, ix, iy, iw, ih, name in self._face_render_data:
            dx = ox + ix * scale
            dy = oy + iy * scale
            dw = iw * scale
            dh = ih * scale
            disp_faces.append((face_id, dx, dy, dw, dh, name))

        # Font sized for display space
        img_font_size = max(34, min(96, int(min(self._full_w, self._full_h) * 0.028)))
        disp_font_px = max(9, int(img_font_size * scale))
        font = QFont()
        font.setPixelSize(disp_font_px)

        label_sizes, layouts = self._get_label_layouts(disp_faces, disp_w, disp_h, font)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setFont(font)

        # ── Bounding boxes ───────────────────────────────────────────────
        if self._show_bboxes:
            for i, (face_id, dx, dy, dw, dh, name) in enumerate(disp_faces):
                is_selected = face_id == self._selected_face_id
                is_hovered = face_id == self._hover_face_id

                if is_selected:
                    color = _COLOR_SELECTED
                    thickness = 3.0
                    opacity = 1.0
                elif is_hovered:
                    color = _COLOR_HOVER
                    thickness = 2.5
                    opacity = 1.0
                else:
                    color = _COLOR_NORMAL
                    thickness = 1.5
                    opacity = self._bbox_opacity

                painter.setOpacity(opacity)
                painter.setPen(QPen(color, thickness))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(dx, dy, dw, dh))

        # ── Labels ──────────────────────────────────────────────────────
        if self._show_labels:
            pad_x = max(4, disp_font_px // 4)
            pad_y = max(3, disp_font_px // 7)

            for i, (face_id, dx, dy, dw, dh, name) in enumerate(disp_faces):
                if layouts[i] is None:
                    continue
                layout = layouts[i]
                display_name, lw_label, lh_label, _, _ = label_sizes[i]

                is_selected = face_id == self._selected_face_id
                is_hovered = face_id == self._hover_face_id

                if is_selected:
                    text_color   = _COLOR_SELECTED
                    bg_color     = _BG_SELECTED
                    border_color = _BORDER_SELECTED
                    opacity      = 1.0
                elif is_hovered:
                    text_color   = _COLOR_HOVER
                    bg_color     = _BG_HOVER
                    border_color = _BORDER_HOVER
                    opacity      = 1.0
                else:
                    text_color   = _COLOR_NORMAL
                    bg_color     = _BG_NORMAL
                    border_color = _BORDER_NORMAL
                    opacity      = self._label_opacity

                painter.setOpacity(opacity)

                # Leader line (only when label moved far from face)
                if layout.leader_start and layout.leader_end:
                    painter.setPen(QPen(QColor(120, 120, 120, 160), 1))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawLine(
                        QPointF(*layout.leader_start),
                        QPointF(*layout.leader_end),
                    )

                # Label background
                lx, ly = float(layout.label_x), float(layout.label_y)
                painter.setPen(QPen(border_color, 1))
                painter.setBrush(QBrush(bg_color))
                painter.drawRoundedRect(QRectF(lx, ly, lw_label, lh_label), 3, 3)

                # Label text
                painter.setPen(text_color)
                painter.setBrush(Qt.NoBrush)
                text_rect = QRectF(
                    lx + pad_x, ly + pad_y,
                    lw_label - 2 * pad_x, lh_label - 2 * pad_y,
                )
                painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, display_name)

        painter.end()


# ---------------------------------------------------------------------------
# Preview panel
# ---------------------------------------------------------------------------

class PreviewPanel(QWidget):
    """Shows a full image preview with all faces highlighted and named.

    Signals:
        face_selected: ``(face_id: int)``
        face_assign_requested: ``(face_id: int)``
        face_delete_requested: ``(face_id: int)``
        face_create_requested: ``(image_id, x, y, w, h)``
        face_bbox_update_requested: ``(face_id, x, y, w, h)``
        prev_image_requested: emitted when the user clicks "← Előző"
        next_image_requested: emitted when the user clicks "Következő →"
    """

    face_selected              = Signal(int)
    face_assign_requested      = Signal(int)
    face_delete_requested      = Signal(int)
    face_create_requested      = Signal(int, int, int, int, int)
    face_bbox_update_requested = Signal(int, int, int, int, int)
    face_diagnostics_requested = Signal(int)
    prev_image_requested       = Signal()
    next_image_requested       = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_image_path: Optional[str] = None
        self._current_image_id: Optional[int] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._face_data: List[_FaceData] = []
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None

        # Overlay state
        self._show_bboxes: bool = True
        self._bbox_opacity: float = 0.7
        self._prev_bbox_opacity: float = 0.7
        self._show_labels: bool = True
        self._label_opacity: float = 0.4
        self._prev_label_opacity: float = 0.4

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Image area ───────────────────────────────────────────────────
        self._image_label = _FaceImageLabel()
        self._image_label.setText(t("preview_empty"))
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(300, 200)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet(
            "QLabel { background: #222; border: 1px solid #444; }"
        )
        self._image_label.setToolTip(t("preview_tip"))
        self._image_label.face_clicked.connect(self._on_face_clicked)
        self._image_label.canvas_clicked.connect(self._open_zoom)
        self._image_label.face_right_clicked.connect(self._on_face_right_clicked)
        self._image_label.rect_drawn.connect(self._on_rect_drawn)
        layout.addWidget(self._image_label)

        # ── Draw-mode hint ───────────────────────────────────────────────
        self._draw_hint = QLabel(t("draw_face_hint"))
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        layout.addWidget(self._draw_hint)

        # ── Path label ───────────────────────────────────────────────────
        self._path_label = QLabel("")
        self._path_label.setWordWrap(True)
        self._path_label.setMinimumWidth(0)
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._path_label.setStyleSheet("QLabel { color: #aaa; font-size: 10px; }")
        layout.addWidget(self._path_label)

        # ── Nav buttons + overlay controls ────────────────────────────────
        nav_row = QHBoxLayout()
        nav_row.setSpacing(4)

        _ov_chk_style = (
            "QCheckBox { color: #cdd6f4; font-size: 11px; spacing: 3px; }"
            "QCheckBox::indicator { width: 13px; height: 13px; }"
        )
        _ov_sld_style = (
            "QSlider::groove:horizontal { height: 4px; background: #45475a; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 10px; height: 10px; margin: -3px 0;"
            " background: #89b4fa; border-radius: 5px; }"
            "QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 2px; }"
            "QSlider:disabled { opacity: 0.4; }"
        )
        _ov_pct_style = "QLabel { color: #888; font-size: 11px; min-width: 30px; }"

        self._bbox_check = QCheckBox(t("overlay_bboxes"))
        self._bbox_check.setChecked(True)
        self._bbox_check.setToolTip(t("overlay_bbox_tip"))
        self._bbox_check.setStyleSheet(_ov_chk_style)
        nav_row.addWidget(self._bbox_check)

        self._bbox_slider = QSlider(Qt.Horizontal)
        self._bbox_slider.setRange(0, 100)
        self._bbox_slider.setValue(70)
        self._bbox_slider.setToolTip(t("overlay_bbox_tip"))
        self._bbox_slider.setMinimumWidth(50)
        self._bbox_slider.setMaximumWidth(90)
        self._bbox_slider.setStyleSheet(_ov_sld_style)
        nav_row.addWidget(self._bbox_slider)

        self._bbox_pct_label = QLabel("70%")
        self._bbox_pct_label.setStyleSheet(_ov_pct_style)
        self._bbox_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nav_row.addWidget(self._bbox_pct_label)

        nav_row.addSpacing(8)

        self._label_check = QCheckBox(t("overlay_labels"))
        self._label_check.setChecked(True)
        self._label_check.setToolTip(t("overlay_label_tip"))
        self._label_check.setStyleSheet(_ov_chk_style)
        nav_row.addWidget(self._label_check)

        self._label_slider = QSlider(Qt.Horizontal)
        self._label_slider.setRange(0, 100)
        self._label_slider.setValue(40)
        self._label_slider.setToolTip(t("overlay_label_tip"))
        self._label_slider.setMinimumWidth(50)
        self._label_slider.setMaximumWidth(90)
        self._label_slider.setStyleSheet(_ov_sld_style)
        nav_row.addWidget(self._label_slider)

        self._label_pct_label = QLabel("40%")
        self._label_pct_label.setStyleSheet(_ov_pct_style)
        self._label_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        nav_row.addWidget(self._label_pct_label)

        self._bbox_check.toggled.connect(self._on_ov_bbox_check)
        self._bbox_slider.valueChanged.connect(self._on_ov_bbox_slider)
        self._label_check.toggled.connect(self._on_ov_label_check)
        self._label_slider.valueChanged.connect(self._on_ov_label_slider)

        nav_row.addStretch()

        self._prev_btn = QPushButton(t("prev_image"))
        self._prev_btn.setEnabled(False)
        self._prev_btn.clicked.connect(self.prev_image_requested)
        nav_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton(t("next_image"))
        self._next_btn.setEnabled(False)
        self._next_btn.clicked.connect(self.next_image_requested)
        nav_row.addWidget(self._next_btn)

        layout.addLayout(nav_row)

        # ── Action buttons ────────────────────────────────────────────────
        # Each button uses Expanding policy so the row fills available width.
        # The stretch factor passed to addWidget() is proportional to the
        # character count of the label: longer text → more allocated width →
        # equal padding density across all buttons regardless of label length.
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        _BTN_STYLE = "QPushButton { padding: 3px 4px; }"

        def _action_btn(text: str) -> QPushButton:
            b = QPushButton(text)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setMinimumWidth(0)
            b.setStyleSheet(_BTN_STYLE)
            return b

        def _add(btn: QPushButton) -> None:
            # stretch ∝ visible character count so label always fits
            btn_row.addWidget(btn, stretch=max(1, len(btn.text())))

        self._open_btn = _action_btn(t("open_file_manager"))
        self._open_btn.setEnabled(False)
        self._open_btn.clicked.connect(self._open_in_filemanager)
        _add(self._open_btn)

        self._zoom_btn = _action_btn(f"🔍 {t('zoom')}")
        self._zoom_btn.setEnabled(False)
        self._zoom_btn.clicked.connect(self._open_zoom)
        _add(self._zoom_btn)

        self._draw_btn = _action_btn(f"✏ {t('selection')}")
        self._draw_btn.setCheckable(True)
        self._draw_btn.setEnabled(False)
        self._draw_btn.setToolTip(t("draw_face_hint"))
        self._draw_btn.toggled.connect(self._on_draw_mode_toggled)
        _add(self._draw_btn)

        self._edit_btn = _action_btn(t("modify_selection"))
        self._edit_btn.setEnabled(False)
        self._edit_btn.clicked.connect(self._start_selected_face_edit)
        _add(self._edit_btn)

        self._assign_btn = _action_btn(t("assign_to_person"))
        self._assign_btn.setEnabled(False)
        self._assign_btn.clicked.connect(self._assign_selected_face)
        _add(self._assign_btn)

        self._delete_btn = _action_btn(t("delete_selection"))
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._delete_selected_face)
        _add(self._delete_btn)

        layout.addLayout(btn_row)

    # ── Overlay control handlers ──────────────────────────────────────────

    def _on_ov_bbox_check(self, checked: bool) -> None:
        self._show_bboxes = checked
        self._bbox_slider.setEnabled(checked)
        if checked:
            self._bbox_opacity = self._prev_bbox_opacity
        self._push_overlay_settings()

    def _on_ov_bbox_slider(self, value: int) -> None:
        pct = value / 100.0
        self._bbox_opacity = pct
        if self._show_bboxes:
            self._prev_bbox_opacity = pct
        self._bbox_pct_label.setText(f"{value}%")
        self._push_overlay_settings()

    def _on_ov_label_check(self, checked: bool) -> None:
        self._show_labels = checked
        self._label_slider.setEnabled(checked)
        if checked:
            self._label_opacity = self._prev_label_opacity
        self._push_overlay_settings()

    def _on_ov_label_slider(self, value: int) -> None:
        pct = value / 100.0
        self._label_opacity = pct
        if self._show_labels:
            self._prev_label_opacity = pct
        self._label_pct_label.setText(f"{value}%")
        self._push_overlay_settings()

    def _push_overlay_settings(self) -> None:
        """Forward current overlay state to the image label (overlay repaint only)."""
        self._image_label.set_overlay_settings(
            self._show_bboxes,
            self._bbox_opacity if self._show_bboxes else 0.0,
            self._show_labels,
            self._label_opacity if self._show_labels else 0.0,
            self._selected_face_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_face(self, face: Face) -> None:
        """Load and display the image for *face*, highlighting all faces."""
        if face.image is None:
            self._image_label.setText(t("no_recognized_face"))
            return

        img_path = face.image.file_path
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
            # If the image was already loaded for this same path, reuse the
            # cached BGR buffer so manual annotations still appear even when
            # the file is temporarily inaccessible (e.g. cloud-sync lock).
            if same_image and self._orig_img_bgr is not None:
                log.warning(
                    "show_face: cannot reload %r — reusing cached frame", img_path
                )
                img_bgr = self._orig_img_bgr
            else:
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
        for f in face.image.faces:
            if not f.is_excluded:
                log.debug(
                    "Preview face entity: FaceId=%s PersonId=%s bbox=(%s,%s,%s,%s)",
                    f.id, f.person_id,
                    f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                )
        self._selected_face_id = face.id
        self._editing_face_id = None
        self._draw_btn.setChecked(False)

        log.debug(
            "show_face: face_id=%d image_id=%d annotations=%d same_image=%s preserve_draw=%s",
            face.id, face.image.id, len(self._face_data), same_image, preserve_draw,
        )
        self._image_label.set_face_data(
            self._face_data,
            img_bgr.shape[1],
            img_bgr.shape[0],
        )
        self._render()

        self._path_label.setText(img_path)
        self._open_btn.setEnabled(True)
        self._zoom_btn.setEnabled(True)
        self._draw_btn.setEnabled(True)
        self._prev_btn.setEnabled(True)
        self._next_btn.setEnabled(True)
        self._update_action_buttons()

        if preserve_draw:
            self._draw_btn.setChecked(True)

    def select_face(self, face_id: int) -> None:
        """Change the highlighted face without reloading the image from disk."""
        if self._selected_face_id == face_id:
            return
        self._selected_face_id = face_id
        # Overlay-only repaint — no image reload needed.
        self._push_overlay_settings()
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
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)
        self._current_image_path = None

    @property
    def current_image_id(self) -> Optional[int]:
        return self._current_image_id

    # ------------------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _render(self) -> None:
        """Convert the raw BGR image to a clean QPixmap and push overlay settings."""
        if self._orig_img_bgr is None:
            return
        log.debug(
            "Render image preview: selected_FaceId=%s faces=%s",
            self._selected_face_id,
            [fd[0] for fd in self._face_data],
        )
        # Convert clean image (no annotations baked in)
        rgb = cv2.cvtColor(self._orig_img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self._full_pixmap = QPixmap.fromImage(qimg)
        self._update_scaled_pixmap()
        self._push_overlay_settings()

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
        self._selected_face_id = face_id
        self._push_overlay_settings()
        self._update_action_buttons()
        self.face_selected.emit(face_id)

    def _on_face_right_clicked(self, face_id: int, gx: int, gy: int) -> None:
        self.show_face_context_menu(face_id, gx, gy)

    def show_face_context_menu(
        self,
        face_id: int,
        gx: int,
        gy: int,
        person_name: Optional[str] = None,
    ) -> None:
        """Show the face context menu at global position (gx, gy)."""
        if self._selected_face_id != face_id:
            self._selected_face_id = face_id
            self._push_overlay_settings()
            self._update_action_buttons()
            self.face_selected.emit(face_id)

        if person_name is None:
            entry = next((f for f in self._face_data if f[0] == face_id), None)
            person_name = entry[5] if entry else None

        menu = QMenu(self)
        title = menu.addAction(f"👤  {person_name}" if person_name else f"👤  {t('unknown_face')}")
        title.setEnabled(False)
        menu.addSeparator()

        assign_action = menu.addAction(f"👤  {t('assign_to_person')}")
        edit_action   = menu.addAction(f"✏  {t('modify_selection')}")
        delete_action = menu.addAction(f"🗑  {t('delete_selection')}")
        menu.addSeparator()
        diag_action = menu.addAction(f"🔍  {t('diag_menu')}")

        chosen = menu.exec(QPoint(gx, gy))
        if chosen == assign_action:
            self.face_assign_requested.emit(face_id)
        elif chosen == edit_action:
            self._start_face_edit(face_id)
        elif chosen == delete_action:
            self.face_delete_requested.emit(face_id)
        elif chosen == diag_action:
            self.face_diagnostics_requested.emit(face_id)

    def _on_draw_mode_toggled(self, active: bool) -> None:
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if not active:
            self._editing_face_id = None
            self._draw_hint.setText(t("draw_face_hint"))

    def _on_rect_drawn(self, label_rect: QRect) -> None:
        log.debug(
            "pointer up: label_rect=(%d,%d,%d,%d) image_id=%s editing=%s",
            label_rect.x(), label_rect.y(), label_rect.width(), label_rect.height(),
            self._current_image_id, self._editing_face_id,
        )
        coords = self._label_rect_to_image(label_rect)
        if coords is None:
            log.debug("pointer up: coords outside image — draw ignored")
            return
        if self._current_image_id is None:
            log.debug("pointer up: _current_image_id is None — draw ignored")
            return
        x, y, w, h = coords
        log.debug("bbox created in image coords: (%d,%d,%d,%d)", x, y, w, h)
        if self._editing_face_id is not None:
            face_id = self._editing_face_id
            self._editing_face_id = None
            self._draw_hint.setText(t("draw_face_hint"))
            self._selected_face_id = face_id
            log.debug("bbox update requested for face_id=%d", face_id)
            self.face_bbox_update_requested.emit(face_id, x, y, w, h)
        else:
            log.debug("face create requested: image_id=%d bbox=(%d,%d,%d,%d)", self._current_image_id, x, y, w, h)
            self.face_create_requested.emit(self._current_image_id, x, y, w, h)

    def _label_rect_to_image(self, rect: QRect) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        if x1 < 0 or y1 < 0:
            return None
        # Use clamped conversion for bottom-right so rects that extend into
        # the margin area are clipped to the image edge instead of becoming 1×1.
        x2, y2 = self._label_to_image_clamped(rect.right(), rect.bottom())
        if x2 < 0 or y2 < 0:
            return None
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

    def _label_to_image_clamped(self, lx: float, ly: float) -> Tuple[int, int]:
        """Like _label_to_image but clamps out-of-bounds to the image edge."""
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
        rx = max(0.0, min(lx - ox, disp_w - 1))
        ry = max(0.0, min(ly - oy, disp_h - 1))
        return int(rx / scale), int(ry / scale)

    def _start_selected_face_edit(self) -> None:
        if self._selected_face_id is not None:
            self._start_face_edit(self._selected_face_id)

    def _start_face_edit(self, face_id: int) -> None:
        self._editing_face_id = face_id
        self._selected_face_id = face_id
        self._push_overlay_settings()
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
        has_face  = self._selected_face_id is not None
        self._draw_btn.setEnabled(has_image)
        self._edit_btn.setEnabled(has_face)
        self._assign_btn.setEnabled(has_face)
        self._delete_btn.setEnabled(has_face)

    def _open_zoom(self) -> None:
        """Open the zoom dialog with a full-resolution annotated image."""
        if self._orig_img_bgr is None:
            return
        annotated = _draw_faces_pil(self._orig_img_bgr, self._face_data, self._selected_face_id)
        zoom_pixmap = _bgr_to_qpixmap(annotated)
        dlg = _ZoomDialog(zoom_pixmap, parent=self)
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
