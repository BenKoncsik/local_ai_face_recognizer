"""Image browser panel — browse all images with clickable face annotations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect, QEvent, Signal
from PySide6.QtGui import (
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Face, Image, Person
from app.services.identity_service import IdentityService
from app.ui.dialogs.person_info_dialog import PersonInfoDialog

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None, person_id_or_None)
_FaceData = Tuple[int, int, int, int, int, Optional[str], Optional[int]]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _get_pil_font(size: int):
    """Return a truetype font that supports UTF-8; falls back to PIL default."""
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
    return ImageFont.load_default()


def _draw_faces(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw

    img = img_bgr.copy()

    # Draw rectangles with OpenCV (fast, no encoding issues)
    for face_id, x, y, w, h, _, _ in faces:
        selected = face_id == selected_id
        color = (50, 220, 50) if selected else (180, 180, 180)
        thickness = 3 if selected else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Use PIL for text so UTF-8 / accented characters render correctly
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    image_h, image_w = img.shape[:2]
    font_size = max(34, min(96, int(min(image_w, image_h) * 0.028)))

    for face_id, x, y, w, h, person_name, _ in faces:
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


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a3a3a;")
    return line


# ──────────────────────────────────────────────────────────────────────────────
# Line edit that emits signals on focus-out and Escape for inline rename
# ──────────────────────────────────────────────────────────────────────────────

class _FocusLineEdit(QLineEdit):
    focus_lost = Signal()
    escape_pressed = Signal()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.focus_lost.emit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.escape_pressed.emit()
        else:
            super().keyPressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Custom image label — supports both click-to-select and drag-to-draw modes
# ──────────────────────────────────────────────────────────────────────────────

class _DrawableImageLabel(QLabel):
    """QLabel supporting click-to-select, draw-mode, wheel zoom, and middle-drag pan."""

    clicked = Signal(int, int)
    rect_drawn = Signal(QRect)
    right_clicked = Signal(int, int)
    wheel_zoomed = Signal(int, int, int)   # angle_delta_y, widget_x, widget_y
    pan_moved = Signal(int, int)           # dx, dy in widget pixels

    _BORDER_NORMAL = "QLabel { background: #1a1a1a; }"
    _BORDER_DRAW   = "QLabel { background: #1a1a1a; border: 2px solid #ffcc00; }"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._mid_start: Optional[QPoint] = None
        self._source_pix: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self.setStyleSheet(self._BORDER_NORMAL)

    # ── Public ────────────────────────────────────────────────────────

    def set_source_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._source_pix = pixmap
        self.update()

    def set_zoom_pan(self, zoom: float, pan_x: float, pan_y: float) -> None:
        self._zoom = zoom
        self._pan_x = pan_x
        self._pan_y = pan_y
        self.update()

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._start = None
        self._end = None
        self.setStyleSheet(self._BORDER_DRAW if enabled else self._BORDER_NORMAL)
        self.update()

    # ── Mouse events ─────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        if self._draw_mode:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()
        else:
            pos = event.position().toPoint()
            self.clicked.emit(pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:
        if self._mid_start is not None:
            cur = event.position().toPoint()
            self.pan_moved.emit(cur.x() - self._mid_start.x(), cur.y() - self._mid_start.y())
            self._mid_start = cur
            return
        if self._draw_mode and self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = None
            self.setCursor(Qt.CrossCursor)
            return
        if not self._draw_mode or self._start is None or event.button() != Qt.LeftButton:
            return
        end = event.position().toPoint()
        rect = QRect(self._start, end).normalized()
        self._start = None
        self._end = None
        self.update()
        if rect.width() >= 8 and rect.height() >= 8:
            self.rect_drawn.emit(rect)

    # ── Context menu (right-click) ────────────────────────────────────

    def contextMenuEvent(self, event) -> None:
        self.right_clicked.emit(event.pos().x(), event.pos().y())

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            pos = event.position().toPoint()
            self.wheel_zoomed.emit(delta, pos.x(), pos.y())
        event.accept()

    # ── Paint ────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        cr = self.contentsRect()
        painter.fillRect(cr, Qt.black)

        if self._source_pix is not None:
            lw, lh = cr.width(), cr.height()
            pw, ph = self._source_pix.width(), self._source_pix.height()
            if pw > 0 and ph > 0:
                base_scale = min(lw / pw, lh / ph)
                eff = base_scale * self._zoom
                disp_w = pw * eff
                disp_h = ph * eff
                ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
                oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
                painter.drawPixmap(
                    QRect(int(ox), int(oy), int(disp_w), int(disp_h)),
                    self._source_pix,
                )
        else:
            painter.setPen(Qt.gray)
            painter.drawText(cr, Qt.AlignCenter | Qt.TextWordWrap, self.text())

        if self._draw_mode and self._start is not None and self._end is not None:
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self._start, self._end).normalized())


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class ImageBrowserPanel(QWidget):
    """Browse all scanned images; click a face to identify or categorise it."""

    person_data_changed = Signal()  # emitted after any rename / assign / create

    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config            # AppConfig | None
        self._image_ids: List[int] = []
        self._current_index: int = 0
        self._current_path: str = ""
        self._current_image_id: Optional[int] = None
        self._face_data: List[_FaceData] = []
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None   # face being bbox-edited
        self._renaming: bool = False                   # inline rename in progress
        self._detection_done: bool = False             # whether current image was processed
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._recent_assignment_person_ids: List[int] = []
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # Fullscreen state
        self._fs_hidden_widgets: List[QWidget] = []
        self._fs_was_maximized: bool = False
        self._build_ui()
        self._setup_shortcuts()

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Navigation bar ────────────────────────────────────────────
        nav = QHBoxLayout()

        self._prev_btn = QPushButton("◀  Előző")
        self._prev_btn.setFixedWidth(110)
        self._prev_btn.clicked.connect(self._on_prev)
        nav.addWidget(self._prev_btn)

        self._counter_label = QLabel("0 / 0")
        self._counter_label.setAlignment(Qt.AlignCenter)
        self._counter_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #ccc;")
        nav.addWidget(self._counter_label, stretch=1)

        self._next_btn = QPushButton("Következő  ▶")
        self._next_btn.setFixedWidth(110)
        self._next_btn.clicked.connect(self._on_next)
        nav.addWidget(self._next_btn)

        nav.addSpacing(12)

        self._draw_mode_btn = QPushButton("✏  Kézi jelölés")
        self._draw_mode_btn.setCheckable(True)
        self._draw_mode_btn.setToolTip(
            "Kattints, majd húzd az egeret a képen egy arc kézi megjelöléséhez"
        )
        self._draw_mode_btn.toggled.connect(self._on_draw_mode_toggled)
        nav.addWidget(self._draw_mode_btn)

        nav.addSpacing(8)

        self._fs_btn = QPushButton("⛶  Teljes képernyő")
        self._fs_btn.setToolTip("Teljes képernyős nézet (F11)")
        self._fs_btn.clicked.connect(self._enter_fullscreen)
        nav.addWidget(self._fs_btn)

        self._exit_fs_btn = QPushButton("✕  Kilépés a teljes képernyőből")
        self._exit_fs_btn.setToolTip("Kilépés a teljes képernyős nézetből (F11)")
        self._exit_fs_btn.setStyleSheet("QPushButton { color: #ff8888; font-weight: bold; }")
        self._exit_fs_btn.clicked.connect(self._exit_fullscreen)
        self._exit_fs_btn.setVisible(False)
        nav.addWidget(self._exit_fs_btn)

        root.addLayout(nav)

        # ── Draw-mode hint (shown only when active) ───────────────────
        self._draw_hint = QLabel(
            "✏  Húzd az egeret az arc körül a jelöléshez — mentés automatikus"
        )
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        root.addWidget(self._draw_hint)

        # ── Main splitter: 3/4 image | 1/4 info ──────────────────────
        splitter = QSplitter(Qt.Horizontal)

        # Left: image area
        self._image_label = _DrawableImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setText(
            "Válassz mappát és futtass beolvasást\n"
            "Select a folder and run a scan"
        )
        self._image_label.clicked.connect(self._on_image_clicked)
        self._image_label.rect_drawn.connect(self._on_rect_drawn)
        self._image_label.right_clicked.connect(self._on_image_right_clicked)
        self._image_label.wheel_zoomed.connect(self._on_wheel_zoom)
        self._image_label.pan_moved.connect(self._on_pan_moved)
        splitter.addWidget(self._image_label)

        # Right: info panel
        info_widget = QWidget()
        info_widget.setMinimumWidth(180)
        info_widget.setMaximumWidth(360)
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(10, 10, 10, 10)
        info_layout.setSpacing(8)

        folder_hdr = QLabel("Mappa:")
        folder_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(folder_hdr)

        self._folder_label = QLabel("")
        self._folder_label.setWordWrap(True)
        self._folder_label.setStyleSheet("color: #88aaff; font-size: 11px;")
        info_layout.addWidget(self._folder_label)

        info_layout.addWidget(_hline())

        date_hdr = QLabel("Kép dátuma / időszaka:")
        date_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(date_hdr)

        date_row = QHBoxLayout()
        date_row.setSpacing(4)
        self._photo_date_edit = QLineEdit()
        self._photo_date_edit.setPlaceholderText(
            "pl. 1954  vagy  1954.03.12  vagy  1930-as évek"
        )
        self._photo_date_edit.setToolTip(
            "A kép készítésének dátuma vagy időszaka (szabad szöveg)"
        )
        self._photo_date_edit.returnPressed.connect(self._save_photo_date)
        self._photo_date_edit.editingFinished.connect(self._save_photo_date)
        date_row.addWidget(self._photo_date_edit, stretch=1)
        info_layout.addLayout(date_row)

        info_layout.addWidget(_hline())

        face_hdr = QLabel("Kiválasztott arc:")
        face_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(face_hdr)

        self._face_status_label = QLabel("Kattints egy arcra a képen")
        self._face_status_label.setWordWrap(True)
        self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._face_status_label)

        name_row = QHBoxLayout()
        name_row.setSpacing(4)

        self._person_name_label = QLabel("")
        self._person_name_label.setWordWrap(True)
        self._person_name_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #88ee88; padding: 4px 0px;"
        )
        self._person_name_label.setVisible(False)
        name_row.addWidget(self._person_name_label, stretch=1)

        self._rename_btn = QPushButton("✏")
        self._rename_btn.setFixedSize(26, 26)
        self._rename_btn.setToolTip("Személy átnevezése")
        self._rename_btn.setVisible(False)
        self._rename_btn.clicked.connect(self._start_rename)
        name_row.addWidget(self._rename_btn)

        self._person_info_btn = QPushButton("📋")
        self._person_info_btn.setFixedSize(26, 26)
        self._person_info_btn.setToolTip("Személyadatok szerkesztése")
        self._person_info_btn.setVisible(False)
        self._person_info_btn.clicked.connect(self._on_person_info)
        name_row.addWidget(self._person_info_btn)

        info_layout.addLayout(name_row)

        self._rename_edit = _FocusLineEdit()
        self._rename_edit.setPlaceholderText("Új név…")
        self._rename_edit.setVisible(False)
        self._rename_edit.returnPressed.connect(self._commit_rename)
        self._rename_edit.focus_lost.connect(self._commit_rename)
        self._rename_edit.escape_pressed.connect(self._cancel_rename)
        info_layout.addWidget(self._rename_edit)

        info_layout.addWidget(_hline())

        assign_hdr = QLabel("Hozzárendelés meglévő személyhez:")
        assign_hdr.setWordWrap(True)
        assign_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(assign_hdr)

        self._person_combo = QComboBox()
        self._person_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout.addWidget(self._person_combo)

        self._assign_btn = QPushButton("Hozzárendelés")
        self._assign_btn.clicked.connect(self._on_assign_existing)
        self._assign_btn.setEnabled(False)
        info_layout.addWidget(self._assign_btn)

        info_layout.addWidget(_hline())

        new_hdr = QLabel("Új személy létrehozása:")
        new_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(new_hdr)

        self._new_name_edit = QLineEdit()
        self._new_name_edit.setPlaceholderText("Személy neve…")
        self._new_name_edit.returnPressed.connect(self._on_create_and_assign)
        info_layout.addWidget(self._new_name_edit)

        self._create_btn = QPushButton("Létrehozás és hozzárendelés")
        self._create_btn.clicked.connect(self._on_create_and_assign)
        self._create_btn.setEnabled(False)
        info_layout.addWidget(self._create_btn)

        info_layout.addStretch()
        splitter.addWidget(info_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Left), self, self._on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self, self._on_next)
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Reload image list from DB and redisplay current (or first) image."""
        with session_scope() as session:
            images = (
                session.query(Image)
                .order_by(Image.file_path)
                .all()
            )
            self._image_ids = [img.id for img in images]

        count = len(self._image_ids)
        if count == 0:
            self._counter_label.setText("0 / 0")
            self._full_pixmap = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText("Nincs kép az adatbázisban")
            self._folder_label.setText("")
            self._clear_face_panel()
            self._update_nav_buttons()
            return

        self._current_index = min(self._current_index, count - 1)
        self._load_current_image()
        self._update_nav_buttons()

    # ──────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────

    def _on_prev(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._selected_face_id = None
            self._reset_zoom()
            self._load_current_image()
            self._update_nav_buttons()

    def _on_next(self) -> None:
        if self._current_index < len(self._image_ids) - 1:
            self._current_index += 1
            self._selected_face_id = None
            self._reset_zoom()
            self._load_current_image()
            self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        count = len(self._image_ids)
        self._prev_btn.setEnabled(self._current_index > 0)
        self._next_btn.setEnabled(self._current_index < count - 1)

    # ──────────────────────────────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────────────────────────────

    def _load_current_image(self) -> None:
        if not self._image_ids:
            return

        img_id = self._image_ids[self._current_index]
        with session_scope() as session:
            img = session.get(Image, img_id)
            if img is None:
                return
            for f in img.faces:
                _ = f.person
            self._current_path = img.file_path
            self._detection_done = img.detection_done
            self._current_image_id = img_id
            photo_date = img.photo_date or ""
            self._face_data = [
                (
                    f.id,
                    f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                    f.person.name if f.person else None,
                    f.person_id,
                )
                for f in img.faces
                if not f.is_excluded
            ]

        self._photo_date_edit.blockSignals(True)
        self._photo_date_edit.setText(photo_date)
        self._photo_date_edit.blockSignals(False)

        self._counter_label.setText(
            f"{self._current_index + 1} / {len(self._image_ids)}"
        )
        self._folder_label.setText(str(Path(self._current_path).parent))

        from app.utils.image_utils import load_image_bgr

        img_bgr = load_image_bgr(self._current_path)
        if img_bgr is None:
            self._orig_img_bgr = None
            self._full_pixmap = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(f"Nem tölthető be:\n{self._current_path}")
            self._clear_face_panel()
            return

        self._orig_img_bgr = img_bgr
        self._redraw_faces()
        self._clear_face_panel()
        self._reload_persons_combo()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._image_label.update()

    # ──────────────────────────────────────────────────────────────────
    # Face click → select
    # ──────────────────────────────────────────────────────────────────

    def _on_image_clicked(self, lx: int, ly: int) -> None:
        # Clear focus from any active text inputs so they commit or deactivate
        self._new_name_edit.clearFocus()
        self._rename_edit.clearFocus()

        if self._full_pixmap is None:
            return

        orig_x, orig_y = self._label_to_image(lx, ly)
        if orig_x < 0 or orig_y < 0:
            return

        clicked_id = self._hit_test(orig_x, orig_y)

        if clicked_id == self._selected_face_id:
            self._selected_face_id = None
            self._redraw_faces()
            self._clear_face_panel()
            return

        self._selected_face_id = clicked_id
        self._redraw_faces()

        if clicked_id is not None:
            self._show_face_info(clicked_id)
        else:
            self._clear_face_panel()

    # ──────────────────────────────────────────────────────────────────
    # Manual face drawing
    # ──────────────────────────────────────────────────────────────────

    _HINT_ADD  = "✏  Húzd az egeret az arc körül a jelöléshez — mentés automatikus"
    _HINT_EDIT = "✏  Rajzold újra az arc körüli téglalapot — a régi helyére kerül"

    def _on_draw_mode_toggled(self, active: bool) -> None:
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if not active:
            # User manually toggled off — cancel any ongoing edit
            self._editing_face_id = None
            self._draw_hint.setText(self._HINT_ADD)

    def _on_rect_drawn(self, label_rect: QRect) -> None:
        """Convert label-space rect to image coords; update existing or create new Face."""
        coords = self._label_rect_to_image(label_rect)
        if coords is None:
            return
        ix, iy, iw, ih = coords

        if not self._image_ids:
            return

        if self._editing_face_id is not None:
            # Edit mode — update existing face bbox
            edited_id = self._editing_face_id
            self._update_face_bbox(edited_id, ix, iy, iw, ih)
            self._editing_face_id = None
            self._draw_hint.setText(self._HINT_ADD)
            self._reload_current_face_data()
            self._selected_face_id = edited_id
            self._redraw_faces()
            self._show_face_info(edited_id)
        else:
            # Add mode — create new manual face
            img_id = self._image_ids[self._current_index]
            new_face_id = self._save_manual_face(img_id, ix, iy, iw, ih)
            self._reload_current_face_data()
            if new_face_id is not None:
                self._selected_face_id = new_face_id
                self._redraw_faces()
                self._show_face_info(new_face_id)

    def _save_manual_face(
        self, img_id: int, x: int, y: int, w: int, h: int
    ) -> Optional[int]:
        """Insert a manual Face record with crop thumbnail. Returns new face_id."""
        from app.detectors.base import Detection
        from app.utils.image_utils import save_face_crop

        detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0)
        crops_dir = None
        thumbnail_size = (128, 128)
        if self._config is not None:
            crops_dir = self._config.crops_dir_resolved
            crops_dir.mkdir(parents=True, exist_ok=True)
            thumbnail_size = self._config.scan.thumbnail_size

        with session_scope() as session:
            existing = session.query(Face).filter(Face.image_id == img_id).count()
            crop_path = None
            if crops_dir is not None and self._orig_img_bgr is not None:
                crop_path = save_face_crop(
                    img_bgr=self._orig_img_bgr,
                    detection=detection,
                    crops_dir=crops_dir,
                    image_id=img_id,
                    thumbnail_size=thumbnail_size,
                    face_index=existing,
                )
            face = Face(
                image_id=img_id,
                bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                confidence=1.0,
                detector_backend="manual",
                crop_path=str(crop_path) if crop_path else None,
            )
            session.add(face)
            session.flush()
            new_id = face.id

        log.info("Manual face saved for image_id=%d: (%d,%d,%d,%d)", img_id, x, y, w, h)
        return new_id

    # ──────────────────────────────────────────────────────────────────
    # Right-click context menu
    # ──────────────────────────────────────────────────────────────────

    def _on_image_right_clicked(self, lx: int, ly: int) -> None:
        orig_x, orig_y = self._label_to_image(lx, ly)
        if orig_x < 0 or orig_y < 0:
            return
        face_id = self._hit_test(orig_x, orig_y)
        if face_id is None:
            return

        entry = next((f for f in self._face_data if f[0] == face_id), None)
        person_name = entry[5] if entry else None

        menu = QMenu(self)

        # Title (disabled)
        title = menu.addAction(f"👤  {person_name}" if person_name else "👤  Ismeretlen arc")
        title.setEnabled(False)
        menu.addSeparator()

        edit_action = menu.addAction("✏  Bbox módosítása")
        menu.addSeparator()
        delete_action = menu.addAction("🗑  Törlés")

        global_pos = self._image_label.mapToGlobal(QPoint(lx, ly))
        chosen = menu.exec(global_pos)

        if chosen == edit_action:
            self._start_face_edit(face_id)
        elif chosen == delete_action:
            self._delete_face(face_id)

    def _start_face_edit(self, face_id: int) -> None:
        """Select the face and enter draw mode so the user can redraw its bbox."""
        self._editing_face_id = face_id
        self._selected_face_id = face_id
        self._redraw_faces()
        self._show_face_info(face_id)
        self._draw_hint.setText(self._HINT_EDIT)
        self._draw_mode_btn.setChecked(True)   # triggers _on_draw_mode_toggled

    def _delete_face(self, face_id: int) -> None:
        """Hard-delete a Face record from the DB."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            session.delete(face)
        log.info("Face %d deleted by user", face_id)
        if self._selected_face_id == face_id:
            self._selected_face_id = None
            self._clear_face_panel()
        if self._editing_face_id == face_id:
            self._editing_face_id = None
            self._draw_mode_btn.setChecked(False)
        self._reload_current_face_data()

    def _update_face_bbox(
        self, face_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        """Update the bounding box of an existing face."""
        with session_scope() as session:
            face = session.get(Face, face_id)
            if face is None:
                return
            face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h = x, y, w, h
        log.info("Face %d bbox updated to (%d,%d,%d,%d)", face_id, x, y, w, h)

    # ──────────────────────────────────────────────────────────────────
    # Coordinate mapping
    # ──────────────────────────────────────────────────────────────────

    def _label_to_image(self, lx: int, ly: int) -> Tuple[int, int]:
        """Convert label (widget) coordinates to original image pixel coordinates."""
        if self._full_pixmap is None:
            return -1, -1
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return -1, -1
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        disp_w = pw * eff
        disp_h = ph * eff
        ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
        oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
        rx = lx - ox
        ry = ly - oy
        if rx < 0 or ry < 0 or rx >= disp_w or ry >= disp_h:
            return -1, -1
        return int(rx / eff), int(ry / eff)

    def _label_rect_to_image(self, rect: QRect) -> Optional[Tuple[int, int, int, int]]:
        """Convert a label-space QRect to (x, y, w, h) in original image coordinates."""
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        x2, y2 = self._label_to_image(rect.right(), rect.bottom())
        if x1 < 0 or y1 < 0:
            return None
        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
            x2 = min(max(x2, 0), iw - 1)
            y2 = min(max(y2, 0), ih - 1)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _hit_test(self, ox: int, oy: int) -> Optional[int]:
        """Return face_id whose bbox contains (ox, oy), or None."""
        for face_id, x, y, w, h, _, _ in self._face_data:
            if x <= ox <= x + w and y <= oy <= y + h:
                return face_id
        return None

    def _redraw_faces(self) -> None:
        if self._orig_img_bgr is None:
            return
        annotated = _draw_faces(self._orig_img_bgr, self._face_data, self._selected_face_id)
        self._full_pixmap = _bgr_to_qpixmap(annotated)
        self._image_label.set_source_pixmap(self._full_pixmap)

    # ──────────────────────────────────────────────────────────────────
    # Zoom / pan
    # ──────────────────────────────────────────────────────────────────

    def _on_wheel_zoom(self, delta: int, lx: int, ly: int) -> None:
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        new_zoom = max(1.0, min(10.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return
        if self._full_pixmap is not None:
            cr = self._image_label.contentsRect()
            lw, lh = cr.width(), cr.height()
            pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
            base_scale = min(lw / pw, lh / ph)
            old_eff = base_scale * self._zoom
            new_eff = base_scale * new_zoom
            old_ox = cr.x() + (lw - pw * old_eff) / 2 + self._pan_x
            old_oy = cr.y() + (lh - ph * old_eff) / 2 + self._pan_y
            ix = (lx - old_ox) / old_eff
            iy = (ly - old_oy) / old_eff
            self._pan_x = lx - (cr.x() + (lw - pw * new_eff) / 2) - ix * new_eff
            self._pan_y = ly - (cr.y() + (lh - ph * new_eff) / 2) - iy * new_eff
        self._zoom = new_zoom
        self._clamp_pan()
        self._image_label.set_zoom_pan(self._zoom, self._pan_x, self._pan_y)

    def _on_pan_moved(self, dx: int, dy: int) -> None:
        self._pan_x += dx
        self._pan_y += dy
        self._clamp_pan()
        self._image_label.set_zoom_pan(self._zoom, self._pan_x, self._pan_y)

    def _clamp_pan(self) -> None:
        if self._full_pixmap is None:
            self._pan_x = self._pan_y = 0.0
            return
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        max_x = max(0.0, (pw * eff - lw) / 2)
        max_y = max(0.0, (ph * eff - lh) / 2)
        self._pan_x = max(-max_x, min(max_x, self._pan_x))
        self._pan_y = max(-max_y, min(max_y, self._pan_y))

    def _reset_zoom(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._image_label.set_zoom_pan(1.0, 0.0, 0.0)

    # ──────────────────────────────────────────────────────────────────
    # Face info panel
    # ──────────────────────────────────────────────────────────────────

    def _show_face_info(self, face_id: int) -> None:
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, _, _, _, _, person_name, _ = entry
        self._assign_btn.setEnabled(True)
        self._create_btn.setEnabled(True)
        if person_name:
            self._face_status_label.setText("Beazonosított személy:")
            self._person_name_label.setText(person_name)
            self._person_name_label.setVisible(True)
            self._rename_btn.setVisible(True)
            self._person_info_btn.setVisible(True)
        else:
            self._face_status_label.setText("Ez az arc nincs kategorizálva")
            self._person_name_label.setVisible(False)
            self._rename_btn.setVisible(False)
            self._person_info_btn.setVisible(False)
        self._rename_edit.setVisible(False)

    def _clear_face_panel(self) -> None:
        self._cancel_rename()
        if self._detection_done and not self._face_data:
            self._face_status_label.setText("Nincs felismert arc ezen a képen")
            self._face_status_label.setStyleSheet(
                "color: #ffaa44; font-size: 11px; font-style: italic;"
            )
        else:
            self._face_status_label.setText("Kattints egy arcra a képen")
            self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._person_name_label.setVisible(False)
        self._rename_btn.setVisible(False)
        self._person_info_btn.setVisible(False)
        self._assign_btn.setEnabled(False)
        self._create_btn.setEnabled(False)
        self._new_name_edit.clear()

    # ──────────────────────────────────────────────────────────────────
    # Photo date save
    # ──────────────────────────────────────────────────────────────────

    def _save_photo_date(self) -> None:
        if self._current_image_id is None:
            return
        value = self._photo_date_edit.text().strip() or None
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            img.photo_date = value
        log.debug("photo_date saved for image %d: %r", self._current_image_id, value)

    # ──────────────────────────────────────────────────────────────────
    # Inline person rename
    # ──────────────────────────────────────────────────────────────────

    def _start_rename(self) -> None:
        if self._selected_face_id is None:
            return
        entry = next((f for f in self._face_data if f[0] == self._selected_face_id), None)
        if entry is None or not entry[5]:
            return
        self._renaming = True
        self._person_name_label.setVisible(False)
        self._rename_btn.setVisible(False)
        self._rename_edit.setText(entry[5])
        self._rename_edit.setVisible(True)
        self._rename_edit.setFocus()
        self._rename_edit.selectAll()

    def _commit_rename(self) -> None:
        if not self._renaming:
            return
        self._renaming = False
        new_name = self._rename_edit.text().strip()
        self._rename_edit.setVisible(False)
        if new_name:
            self._save_person_rename(new_name)
        if self._selected_face_id is not None:
            entry = next((f for f in self._face_data if f[0] == self._selected_face_id), None)
            if entry and entry[5]:
                self._person_name_label.setVisible(True)
                self._rename_btn.setVisible(True)
                self._person_info_btn.setVisible(True)

    def _cancel_rename(self) -> None:
        if not self._renaming:
            return
        self._renaming = False
        self._rename_edit.setVisible(False)
        self._rename_edit.clear()
        if self._selected_face_id is not None:
            entry = next((f for f in self._face_data if f[0] == self._selected_face_id), None)
            if entry and entry[5]:
                self._person_name_label.setVisible(True)
                self._rename_btn.setVisible(True)
                self._person_info_btn.setVisible(True)

    def _save_person_rename(self, new_name: str) -> None:
        if self._selected_face_id is None:
            return
        entry = next((f for f in self._face_data if f[0] == self._selected_face_id), None)
        if entry is None:
            return
        person_id = entry[6]
        if person_id is None:
            return
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            person.name = new_name
            person.is_auto_named = False
        log.info("Person %d renamed to %r", person_id, new_name)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()

    # ──────────────────────────────────────────────────────────────────
    # Person combo
    # ──────────────────────────────────────────────────────────────────

    def _reload_persons_combo(self) -> None:
        self._person_combo.clear()
        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            recent_rank = self._recent_person_rank(session)
            persons.sort(
                key=lambda p: (
                    recent_rank.get(p.id, len(recent_rank) + 10_000),
                    p.name.casefold(),
                )
            )
            for p in persons:
                self._person_combo.addItem(p.name, userData=p.id)

    def _recent_person_rank(self, session) -> dict[int, int]:
        """Rank assignment targets by recent use and nearby previous images."""
        rank: dict[int, int] = {}

        for person_id in self._recent_assignment_person_ids:
            if person_id not in rank:
                rank[person_id] = len(rank)

        if self._current_index <= 0 or not self._image_ids:
            return rank

        window_start = max(0, self._current_index - 50)
        nearby_image_ids = self._image_ids[window_start:self._current_index]
        if not nearby_image_ids:
            return rank

        rows = (
            session.query(Face.image_id, Face.person_id)
            .filter(Face.image_id.in_(nearby_image_ids))
            .filter(Face.person_id.isnot(None))
            .filter(Face.is_excluded == False)  # noqa: E712
            .all()
        )
        persons_by_image: dict[int, set[int]] = {}
        for image_id, person_id in rows:
            if person_id is not None:
                persons_by_image.setdefault(image_id, set()).add(person_id)

        for image_id in reversed(nearby_image_ids):
            for person_id in sorted(persons_by_image.get(image_id, set())):
                if person_id not in rank:
                    rank[person_id] = len(rank)
        return rank

    def _remember_recent_person(self, person_id: int) -> None:
        self._recent_assignment_person_ids = [
            person_id,
            *(pid for pid in self._recent_assignment_person_ids if pid != person_id),
        ][:20]

    # ──────────────────────────────────────────────────────────────────
    # Assign / create actions
    # ──────────────────────────────────────────────────────────────────

    def _on_assign_existing(self) -> None:
        if self._selected_face_id is None:
            return
        person_id = self._person_combo.currentData()
        if person_id is None:
            return
        with session_scope() as session:
            IdentityService(session).reassign_face(self._selected_face_id, person_id)
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()

    def _on_create_and_assign(self) -> None:
        if self._selected_face_id is None:
            return
        name = self._new_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Üres név", "A személynév nem lehet üres.")
            return
        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            IdentityService(session).reassign_face(self._selected_face_id, person_id)
        self._remember_recent_person(person_id)
        self._new_name_edit.clear()
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._open_person_info_dialog(person_id)

    def _open_person_info_dialog(self, person_id: int) -> None:
        """Open the PersonInfoDialog for the given person and save changes."""
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
            if dlg.exec() != PersonInfoDialog.Accepted:
                return
            person.last_name = dlg.last_name() or None
            person.first_name = dlg.first_name() or None
            person.second_name = dlg.second_name() or None
            person.nickname = dlg.nickname() or None
            person.married_name = dlg.married_name() or None
            person.birth_place = dlg.birth_place() or None
            person.birth_date = dlg.birth_date() or None
            person.death_date = dlg.death_date() or None
            person.death_place = dlg.death_place() or None
            person.notes = dlg.notes() or None
        log.info("Személyadatok mentve: person_id=%d", person_id)
        self._reload_persons_combo()
        self.person_data_changed.emit()

    def _on_person_info(self) -> None:
        """Open PersonInfoDialog for the person assigned to the selected face."""
        if self._selected_face_id is None:
            return
        entry = next((f for f in self._face_data if f[0] == self._selected_face_id), None)
        if entry is None:
            return
        person_id = entry[6]
        if person_id is None:
            return
        self._open_person_info_dialog(person_id)

    def _reload_current_face_data(self) -> None:
        """Reload face assignments for the current image without re-reading from disk."""
        if not self._image_ids:
            return
        img_id = self._image_ids[self._current_index]
        with session_scope() as session:
            img = session.get(Image, img_id)
            if img is None:
                return
            for f in img.faces:
                _ = f.person
            self._face_data = [
                (
                    f.id,
                    f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                    f.person.name if f.person else None,
                    f.person_id,
                )
                for f in img.faces
                if not f.is_excluded
            ]
        if self._selected_face_id is not None:
            self._show_face_info(self._selected_face_id)
        self._redraw_faces()

    # ──────────────────────────────────────────────────────────────────
    # Fullscreen
    # ──────────────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self._exit_fs_btn.isVisible():
            self._exit_fullscreen()
        else:
            self._enter_fullscreen()

    def _enter_fullscreen(self) -> None:
        main_win = self.window()
        if main_win is self:
            return  # not embedded in a main window

        self._fs_was_maximized = main_win.isMaximized()
        self._fs_hidden_widgets = []

        for w in main_win.findChildren(QToolBar):
            if w.isVisible():
                self._fs_hidden_widgets.append(w)
                w.hide()
        for w in main_win.findChildren(QDockWidget):
            if w.isVisible():
                self._fs_hidden_widgets.append(w)
                w.hide()
        sb = main_win.statusBar()
        if sb and sb.isVisible():
            self._fs_hidden_widgets.append(sb)
            sb.hide()
        tabs = main_win.centralWidget()
        if isinstance(tabs, QTabWidget):
            tabs.tabBar().hide()

        self._fs_btn.setVisible(False)
        self._exit_fs_btn.setVisible(True)
        main_win.showFullScreen()

    def _exit_fullscreen(self) -> None:
        main_win = self.window()
        if main_win is self:
            return

        for w in self._fs_hidden_widgets:
            w.show()
        self._fs_hidden_widgets = []

        tabs = main_win.centralWidget()
        if isinstance(tabs, QTabWidget):
            tabs.tabBar().show()

        self._fs_btn.setVisible(True)
        self._exit_fs_btn.setVisible(False)

        if self._fs_was_maximized:
            main_win.showMaximized()
        else:
            main_win.showNormal()
