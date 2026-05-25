"""Image Browser Panel — Finder-style tree + preview.

Left   : QTreeWidget — folders expandable, images shown with thumbnail icon + filename
Right  : Full image preview with bbox interaction + face-assign panel

Public API consumed by MainWindow
----------------------------------
    panel = ImageBrowserPanel(config=cfg)
    panel.person_data_changed          # Signal()
    panel.refresh()                    # reload folder list
    panel.retranslate()                # after language change
    panel._reload_current_face_data()  # after external assignment change
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QIcon,
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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Face, Image, Person
from app.services.identity_service import IdentityService
from app.services.image_browser_service import (
    FolderSummary,
    ImageBrowserService,
    ImageSummary,
)
from app.ui.dialogs.person_info_dialog import PersonInfoDialog
from app.ui.i18n import t
from app.workers.thumbnail_worker import ThumbnailRunnable

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None, person_id_or_None)
_FaceData = Tuple[int, int, int, int, int, Optional[str], Optional[int]]

_TREE_THUMB_SIZE = 56   # max side of thumbnail icon in tree

# Data roles stored on QTreeWidgetItem
_ROLE_TYPE    = Qt.UserRole        # "folder" | "image" | "dummy"
_ROLE_PAYLOAD = Qt.UserRole + 1    # folder_path (str) | image_id (int)
_ROLE_CACHE   = Qt.UserRole + 2    # thumbnail_cache_key (str, images only)

_UF_HIDDEN = 0x8000   # macOS BSD hidden flag (chflags hidden)


def _is_visible_dir(folder_path: str) -> bool:
    """Return True only if the folder exists and is not hidden on this platform.

    Checks:
      - directory must exist (is_dir)
      - no path component starts with '.' (Unix/macOS dot-hidden)
      - macOS: neither the folder nor any ancestor up to 6 levels has UF_HIDDEN
      - Windows: FILE_ATTRIBUTE_HIDDEN not set on the folder
    """
    p = Path(folder_path)
    if not p.is_dir():
        return False

    # Dot-prefix convention (Unix / macOS)
    if any(part.startswith(".") for part in p.parts):
        return False

    if sys.platform == "darwin":
        check = p
        for _ in range(6):
            try:
                if os.stat(check).st_flags & _UF_HIDDEN:
                    return False
            except OSError:
                return False
            parent = check.parent
            if parent == check:
                break
            check = parent

    elif sys.platform == "win32":
        try:
            import ctypes
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(p))
            if attrs != -1 and (attrs & 0x2):   # FILE_ATTRIBUTE_HIDDEN
                return False
        except Exception:
            pass

    return True


# ──────────────────────────────────────────────────────────────────────────────
# EXIF helper
# ──────────────────────────────────────────────────────────────────────────────

def _read_exif_date(path: str) -> Optional[str]:
    """Return DateTimeOriginal from EXIF as 'YYYY.MM.DD', or None if unavailable."""
    try:
        from PIL import Image as PilImage
        with PilImage.open(path) as img:
            exif = img._getexif()  # type: ignore[attr-defined]
            if not exif:
                return None
        # Tag 36867 = DateTimeOriginal, 36868 = DateTimeDigitized, 306 = DateTime
        for tag_id in (36867, 36868, 306):
            val = exif.get(tag_id)
            if val:
                # Format is "YYYY:MM:DD HH:MM:SS"
                date_part = val.split(" ")[0]
                return date_part.replace(":", ".")
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers (shared with image display)
# ──────────────────────────────────────────────────────────────────────────────

def _bgr_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


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
    return ImageFont.load_default()


def _draw_faces(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw

    img = img_bgr.copy()
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

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for face_id, x, y, w, h, _, _ in faces:
        selected = face_id == selected_id
        color = (50, 220, 50) if selected else (180, 180, 180)
        thickness = 3 if selected else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


def _hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #3a3a3a;")
    return line


# ──────────────────────────────────────────────────────────────────────────────
# Inline face editor overlay
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


class _InlineFaceEditor(QFrame):
    assign_requested = Signal(int)   # person_id
    create_requested = Signal(str)   # new person name
    closed = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "_InlineFaceEditor { background: #252525; border: 1px solid #666;"
            " border-radius: 4px; }"
        )
        self.setFixedWidth(230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        self._name_lbl = QLabel("?")
        self._name_lbl.setStyleSheet(
            "color: #88ee88; font-weight: bold; font-size: 11px;"
        )
        hdr.addWidget(self._name_lbl, stretch=1)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(16, 16)
        close_btn.setFlat(True)
        close_btn.setStyleSheet(
            "QPushButton { color: #888; font-size: 14px; padding: 0; border: none; }"
            "QPushButton:hover { color: #fff; }"
        )
        close_btn.clicked.connect(self.closed)
        hdr.addWidget(close_btn)
        layout.addLayout(hdr)

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.setMaxVisibleItems(12)
        layout.addWidget(self._combo)

        self._assign_btn = QPushButton()
        self._assign_btn.clicked.connect(self._on_assign)
        layout.addWidget(self._assign_btn)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3a3a;")
        layout.addWidget(sep)

        self._new_edit = QLineEdit()
        self._new_edit.returnPressed.connect(self._on_create)
        layout.addWidget(self._new_edit)

        self._create_btn = QPushButton()
        self._create_btn.clicked.connect(self._on_create)
        layout.addWidget(self._create_btn)

        self.adjustSize()
        self.hide()

    def retranslate(self) -> None:
        self._assign_btn.setText(t("ibp_assign_btn"))
        self._new_edit.setPlaceholderText(t("ibp_new_placeholder"))
        self._create_btn.setText(t("ibp_create_btn"))

    def populate(
        self,
        person_name: Optional[str],
        person_id: Optional[int],
        persons: List[Tuple[int, str]],
    ) -> None:
        self._name_lbl.setText(person_name or "?")
        self._combo.clear()
        current_index = 0
        for i, (pid, name) in enumerate(persons):
            self._combo.addItem(name, userData=pid)
            if pid == person_id:
                current_index = i
        if persons:
            self._combo.setCurrentIndex(current_index)
        self._new_edit.clear()

    def _on_assign(self) -> None:
        pid = self._combo.currentData()
        if pid is not None:
            self.assign_requested.emit(pid)

    def _on_create(self) -> None:
        name = self._new_edit.text().strip()
        if name:
            self.create_requested.emit(name)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.closed.emit()
        else:
            super().keyPressEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Drawable image label (zoom + pan + draw mode)
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Tree widget subclass — intercepts Left/Right on image items for navigation
# ──────────────────────────────────────────────────────────────────────────────

class _BrowserTree(QTreeWidget):
    go_prev = Signal()
    go_next = Signal()

    def keyPressEvent(self, event) -> None:
        current = self.currentItem()
        if current is not None and current.data(0, _ROLE_TYPE) == "image":
            if event.key() == Qt.Key_Left:
                self.go_prev.emit()
                event.accept()
                return
            elif event.key() == Qt.Key_Right:
                self.go_next.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class _DrawableImageLabel(QLabel):
    clicked = Signal(int, int)
    rect_drawn = Signal(QRect)
    right_clicked = Signal(int, int)
    wheel_zoomed = Signal(int, int, int)
    pan_moved = Signal(int, int)
    go_prev = Signal()
    go_next = Signal()

    _BORDER_NORMAL = "QLabel { background: #1a1a1a; }"
    _BORDER_DRAW   = "QLabel { background: #1a1a1a; border: 2px solid #ffcc00; }"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._mid_start: Optional[QPoint] = None
        self._source_pix: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self.setStyleSheet(self._BORDER_NORMAL)

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
            self.pan_moved.emit(
                cur.x() - self._mid_start.x(),
                cur.y() - self._mid_start.y(),
            )
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

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Left:
            self.go_prev.emit()
            event.accept()
        elif event.key() == Qt.Key_Right:
            self.go_next.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        self.right_clicked.emit(event.pos().x(), event.pos().y())

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            pos = event.position().toPoint()
            self.wheel_zoomed.emit(delta, pos.x(), pos.y())
        event.accept()

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
            painter.setPen(QPen(Qt.yellow, 2, Qt.DashLine))
            painter.drawRect(QRect(self._start, self._end).normalized())


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class ImageBrowserPanel(QWidget):
    """Finder-style image browser: folder/image tree | image preview."""

    person_data_changed = Signal()

    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config

        # State
        self._current_folder: Optional[str] = None
        self._current_image_id: Optional[int] = None
        self._current_path: str = ""
        self._detection_done: bool = False
        self._face_data: List[_FaceData] = []
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None
        self._renaming: bool = False
        self._inline_editor_face_id: Optional[int] = None
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._full_pixmap: Optional[QPixmap] = None
        self._recent_assignment_person_ids: List[int] = []

        # Zoom / pan
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Thumbnail / tree state
        self._thumb_cache: Dict[str, QPixmap] = {}
        self._tree_items: Dict[str, QTreeWidgetItem] = {}  # cache_key → image tree item
        self._current_tree_item: Optional[QTreeWidgetItem] = None

        # Fullscreen state
        self._fs_hidden_widgets: List[QWidget] = []
        self._fs_was_maximized: bool = False

        self._build_ui()
        self._setup_shortcuts()
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        main_splitter = QSplitter(Qt.Horizontal)

        # ── Column 1: Finder-style folder/image tree ──────────────────
        tree_panel = QWidget()
        tree_panel.setMinimumWidth(200)
        tree_panel.setMaximumWidth(420)
        tp_layout = QVBoxLayout(tree_panel)
        tp_layout.setContentsMargins(4, 4, 4, 4)
        tp_layout.setSpacing(4)

        self._tree_hdr_lbl = QLabel()
        self._tree_hdr_lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #888;"
        )
        tp_layout.addWidget(self._tree_hdr_lbl)

        self._file_tree = _BrowserTree()
        self._file_tree.setHeaderHidden(True)
        self._file_tree.setIconSize(QSize(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE))
        self._file_tree.setIndentation(16)
        self._file_tree.setAnimated(True)
        self._file_tree.setStyleSheet(
            "QTreeWidget {"
            "  border: none;"
            "  background: #181825;"
            "  color: #cdd6f4;"
            "  outline: 0;"
            "}"
            "QTreeWidget::item {"
            "  padding: 2px 0;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QTreeWidget::item:selected {"
            "  background: #2d2d50;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QTreeWidget::item:hover:!selected {"
            "  background: #22223a;"
            "}"
        )
        self._file_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self._file_tree.itemClicked.connect(self._on_tree_item_clicked)
        self._file_tree.go_prev.connect(self._navigate_prev)
        self._file_tree.go_next.connect(self._navigate_next)
        tp_layout.addWidget(self._file_tree, stretch=1)

        main_splitter.addWidget(tree_panel)

        # ── Column 2: Preview + info sub-splitter ────────────────────
        preview_splitter = QSplitter(Qt.Horizontal)

        # Image area
        image_widget = QWidget()
        image_widget.setMinimumWidth(400)
        im_layout = QVBoxLayout(image_widget)
        im_layout.setContentsMargins(4, 4, 4, 4)
        im_layout.setSpacing(4)

        # Draw mode toggle + nav buttons + fullscreen row
        top_bar = QHBoxLayout()

        self._draw_mode_btn = QPushButton()
        self._draw_mode_btn.setCheckable(True)
        self._draw_mode_btn.toggled.connect(self._on_draw_mode_toggled)
        top_bar.addWidget(self._draw_mode_btn)

        top_bar.addStretch()

        _nav_style = (
            "QPushButton {"
            "  font-size: 16px; padding: 0 10px; min-width: 32px; min-height: 26px;"
            "  background: #313244; border: 1px solid #45475a; border-radius: 4px;"
            "  color: #cdd6f4;"
            "}"
            "QPushButton:hover  { background: #45475a; }"
            "QPushButton:pressed { background: #585b70; }"
            "QPushButton:disabled { color: #45475a; border-color: #313244; }"
        )

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setToolTip("Előző kép  (←)")
        self._prev_btn.setStyleSheet(_nav_style)
        self._prev_btn.clicked.connect(self._navigate_prev)
        top_bar.addWidget(self._prev_btn)

        self._next_btn = QPushButton("▶")
        self._next_btn.setToolTip("Következő kép  (→)")
        self._next_btn.setStyleSheet(_nav_style)
        self._next_btn.clicked.connect(self._navigate_next)
        top_bar.addWidget(self._next_btn)

        top_bar.addStretch()

        self._fs_btn = QPushButton()
        self._fs_btn.clicked.connect(self._enter_fullscreen)
        top_bar.addWidget(self._fs_btn)

        self._exit_fs_btn = QPushButton()
        self._exit_fs_btn.setStyleSheet(
            "QPushButton { color: #ff8888; font-weight: bold; }"
        )
        self._exit_fs_btn.clicked.connect(self._exit_fullscreen)
        self._exit_fs_btn.setVisible(False)
        top_bar.addWidget(self._exit_fs_btn)

        im_layout.addLayout(top_bar)

        self._draw_hint = QLabel()
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        im_layout.addWidget(self._draw_hint)

        self._image_label = _DrawableImageLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumSize(360, 260)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.clicked.connect(self._on_image_clicked)
        self._image_label.rect_drawn.connect(self._on_rect_drawn)
        self._image_label.right_clicked.connect(self._on_image_right_clicked)
        self._image_label.wheel_zoomed.connect(self._on_wheel_zoom)
        self._image_label.pan_moved.connect(self._on_pan_moved)
        self._image_label.go_prev.connect(self._navigate_prev)
        self._image_label.go_next.connect(self._navigate_next)
        im_layout.addWidget(self._image_label, stretch=1)

        # Inline editor (floats over image label)
        self._inline_editor = _InlineFaceEditor(self._image_label)
        self._inline_editor.assign_requested.connect(self._on_inline_assign)
        self._inline_editor.create_requested.connect(self._on_inline_create)
        self._inline_editor.closed.connect(self._hide_inline_editor)

        preview_splitter.addWidget(image_widget)

        # Info panel (right side)
        info_content = QWidget()
        info_layout = QVBoxLayout(info_content)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(6)

        self._folder_hdr = QLabel()
        self._folder_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._folder_hdr)

        self._folder_label = QLabel("")
        self._folder_label.setWordWrap(True)
        self._folder_label.setStyleSheet("color: #88aaff; font-size: 11px;")
        info_layout.addWidget(self._folder_label)

        self._filename_hdr = QLabel()
        self._filename_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._filename_hdr)

        self._filename_label = QLabel("")
        self._filename_label.setWordWrap(True)
        self._filename_label.setStyleSheet("color: #ddaa55; font-size: 11px;")
        info_layout.addWidget(self._filename_label)

        info_layout.addWidget(_hline())

        self._date_hdr = QLabel()
        self._date_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._date_hdr)

        self._photo_date_edit = QLineEdit()
        self._photo_date_edit.returnPressed.connect(self._save_photo_date)
        self._photo_date_edit.editingFinished.connect(self._save_photo_date)
        info_layout.addWidget(self._photo_date_edit)

        # EXIF suggestion row
        exif_row = QWidget()
        exif_row_layout = QHBoxLayout(exif_row)
        exif_row_layout.setContentsMargins(0, 0, 0, 0)
        exif_row_layout.setSpacing(4)
        self._exif_suggestion_hdr = QLabel()
        self._exif_suggestion_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        exif_row_layout.addWidget(self._exif_suggestion_hdr)
        self._exif_suggestion_label = QLabel("")
        self._exif_suggestion_label.setStyleSheet("color: #aaddaa; font-size: 11px;")
        exif_row_layout.addWidget(self._exif_suggestion_label, 1)
        self._exif_accept_btn = QPushButton()
        self._exif_accept_btn.setFixedWidth(64)
        self._exif_accept_btn.clicked.connect(self._accept_exif_suggestion)
        exif_row_layout.addWidget(self._exif_accept_btn)
        self._exif_row_widget = exif_row
        self._exif_row_widget.setVisible(False)
        info_layout.addWidget(self._exif_row_widget)

        info_layout.addWidget(_hline())

        self._face_hdr = QLabel()
        self._face_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._face_hdr)

        self._face_status_label = QLabel()
        self._face_status_label.setWordWrap(True)
        self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._face_status_label)

        self._person_name_label = QLabel("")
        self._person_name_label.setWordWrap(True)
        self._person_name_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #88ee88; padding: 4px 0px;"
        )
        self._person_name_label.setVisible(False)
        info_layout.addWidget(self._person_name_label)

        self._rename_btn = QPushButton()
        self._rename_btn.setVisible(False)
        self._rename_btn.clicked.connect(self._start_rename)
        info_layout.addWidget(self._rename_btn)

        self._person_info_btn = QPushButton()
        self._person_info_btn.setVisible(False)
        self._person_info_btn.clicked.connect(self._on_person_info)
        info_layout.addWidget(self._person_info_btn)

        self._rename_edit = _FocusLineEdit()
        self._rename_edit.setVisible(False)
        self._rename_edit.returnPressed.connect(self._commit_rename)
        self._rename_edit.focus_lost.connect(self._commit_rename)
        self._rename_edit.escape_pressed.connect(self._cancel_rename)
        info_layout.addWidget(self._rename_edit)

        info_layout.addWidget(_hline())

        self._assign_hdr = QLabel()
        self._assign_hdr.setWordWrap(True)
        self._assign_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._assign_hdr)

        self._person_combo = QComboBox()
        self._person_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        info_layout.addWidget(self._person_combo)

        self._assign_btn = QPushButton()
        self._assign_btn.clicked.connect(self._on_assign_existing)
        self._assign_btn.setEnabled(False)
        info_layout.addWidget(self._assign_btn)

        info_layout.addWidget(_hline())

        self._new_hdr = QLabel()
        self._new_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._new_hdr)

        self._new_name_edit = QLineEdit()
        self._new_name_edit.returnPressed.connect(self._on_create_and_assign)
        info_layout.addWidget(self._new_name_edit)

        self._create_btn = QPushButton()
        self._create_btn.clicked.connect(self._on_create_and_assign)
        self._create_btn.setEnabled(False)
        info_layout.addWidget(self._create_btn)

        info_layout.addStretch()

        info_scroll = QScrollArea()
        info_scroll.setWidget(info_content)
        info_scroll.setWidgetResizable(True)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setMinimumWidth(160)
        info_scroll.setMaximumWidth(300)
        preview_splitter.addWidget(info_scroll)

        preview_splitter.setStretchFactor(0, 3)
        preview_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(preview_splitter)

        main_splitter.setSizes([280, 880])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 2)

        root.addWidget(main_splitter, stretch=1)
        self.retranslate()

    def _setup_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_F11), self, self._toggle_fullscreen)

    # ──────────────────────────────────────────────────────────────────
    # Public API  (consumed by MainWindow)
    # ──────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
        self._draw_mode_btn.setText(t("ibp_manual_mark"))
        self._draw_mode_btn.setToolTip(t("ibp_manual_mark_tooltip"))
        self._fs_btn.setText(t("ibp_fullscreen"))
        self._exit_fs_btn.setText(t("ibp_exit_fullscreen"))
        self._draw_hint.setText(t("ibp_draw_hint_add"))
        self._image_label.setText(t("ib3_select_image_hint"))
        self._folder_hdr.setText(t("ibp_folder_hdr"))
        self._filename_hdr.setText(t("ibp_filename_hdr"))
        self._date_hdr.setText(t("ibp_date_hdr"))
        self._photo_date_edit.setPlaceholderText(t("ibp_date_placeholder"))
        self._photo_date_edit.setToolTip(t("ibp_date_tooltip"))
        self._exif_suggestion_hdr.setText(t("ibp_exif_suggestion"))
        self._exif_accept_btn.setText(t("ibp_accept_exif"))
        self._exif_accept_btn.setToolTip(t("ibp_accept_exif_tooltip"))
        self._face_hdr.setText(t("ibp_face_hdr"))
        self._face_status_label.setText(t("ibp_click_face"))
        self._rename_btn.setText(t("ibp_rename_btn"))
        self._rename_btn.setToolTip(t("ibp_rename_tooltip"))
        self._person_info_btn.setText(t("ibp_person_info_btn"))
        self._person_info_btn.setToolTip(t("ibp_person_info_tooltip"))
        self._rename_edit.setPlaceholderText(t("ibp_rename_placeholder"))
        self._assign_hdr.setText(t("ibp_assign_hdr"))
        self._assign_btn.setText(t("ibp_assign_btn"))
        self._new_hdr.setText(t("ibp_new_hdr"))
        self._new_name_edit.setPlaceholderText(t("ibp_new_placeholder"))
        self._create_btn.setText(t("ibp_create_btn"))
        self._inline_editor.retranslate()

    def refresh(self) -> None:
        """Reload folder list from DB; keep the selected folder expanded if still present."""
        log.debug("ImageBrowserPanel.refresh() — reloading folder tree")
        prev_folder = self._current_folder

        self._file_tree.blockSignals(True)
        self._file_tree.clear()
        self._tree_items.clear()
        self._file_tree.blockSignals(False)

        with session_scope() as session:
            folders = ImageBrowserService(session).list_folders()

        folders = [f for f in folders if _is_visible_dir(f.folder_path)]

        if not folders:
            self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
            placeholder = QTreeWidgetItem([t("ib3_no_folders")])
            placeholder.setData(0, _ROLE_TYPE, "placeholder")
            placeholder.setFlags(Qt.NoItemFlags)
            self._file_tree.addTopLevelItem(placeholder)
            return

        self._tree_hdr_lbl.setText(t("ib3_folders_hdr_n", n=len(folders)))

        restore_item: Optional[QTreeWidgetItem] = None
        for summary in folders:
            item = self._make_folder_item(summary)
            self._file_tree.addTopLevelItem(item)
            if summary.folder_path == prev_folder:
                restore_item = item

        if restore_item is not None:
            self._file_tree.blockSignals(True)
            restore_item.setExpanded(True)
            self._file_tree.blockSignals(False)
            # Load children without re-triggering the expanded signal
            self._load_folder_children(restore_item)
            # Re-select the previously open image if still present
            if self._current_image_id is not None:
                cache_key = f"thumb_{self._current_image_id}"
                item = self._tree_items.get(cache_key)
                if item is not None:
                    self._file_tree.setCurrentItem(item)
                    self._file_tree.scrollToItem(item)
                    self._current_tree_item = item
                    self._update_nav_buttons()
        else:
            # No previously open folder — auto-open first image in first folder
            if self._current_image_id is None:
                self._open_first_image()

    def _fetch_face_data(self) -> None:
        """Reload face list from DB without triggering any UI update."""
        if self._current_image_id is None:
            return
        with session_scope() as session:
            svc = ImageBrowserService(session)
            self._face_data = svc.get_image_faces(self._current_image_id)

    def _reload_current_face_data(self) -> None:
        """Reload face assignments for the current image and refresh the UI."""
        if self._current_image_id is None:
            return
        self._fetch_face_data()
        if self._selected_face_id is not None:
            self._show_face_info(self._selected_face_id)
        self._redraw_faces()
        log.debug("Face data reloaded for image_id=%d", self._current_image_id)

    # ──────────────────────────────────────────────────────────────────
    # Tree construction helpers
    # ──────────────────────────────────────────────────────────────────

    def _make_folder_item(self, summary: FolderSummary) -> QTreeWidgetItem:
        """Create a top-level folder QTreeWidgetItem with a lazy-load dummy child."""
        if summary.unknown_faces_count > 0:
            name_color = "#fab387"
        else:
            name_color = "#cdd6f4"

        if summary.total_images == 0 or summary.processed_images == 0:
            dot = "○"
            dot_color = "#45475a"
        elif summary.unprocessed_images == 0:
            dot = "●"
            dot_color = "#a6e3a1"
        else:
            dot = "●"
            dot_color = "#f9e2af"

        label = f"{summary.display_name}  ({summary.total_images})"
        item = QTreeWidgetItem([label])
        item.setData(0, _ROLE_TYPE, "folder")
        item.setData(0, _ROLE_PAYLOAD, summary.folder_path)

        # Colour the folder name
        item.setForeground(0, QColor(name_color))

        # Tooltip with full stats
        tip_lines = [
            f"Összes: {summary.total_images}",
            f"Feldolgozva: {summary.processed_images}",
        ]
        if summary.detected_faces_count > 0:
            tip_lines += [
                f"Arcok: {summary.detected_faces_count}",
                f"Ismeretlen: {summary.unknown_faces_count}",
            ]
        item.setToolTip(0, "\n".join(tip_lines))

        # Dummy child so Qt shows the expand triangle
        dummy = QTreeWidgetItem(["…"])
        dummy.setData(0, _ROLE_TYPE, "dummy")
        dummy.setFlags(Qt.NoItemFlags)
        item.addChild(dummy)

        return item

    def _make_placeholder_icon(self) -> QIcon:
        px = QPixmap(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE)
        px.fill(QColor("#11111b"))
        return QIcon(px)

    # ──────────────────────────────────────────────────────────────────
    # Tree signals
    # ──────────────────────────────────────────────────────────────────

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, _ROLE_TYPE) != "folder":
            return
        # Check if children are already loaded (not just a dummy)
        if item.childCount() == 1 and item.child(0).data(0, _ROLE_TYPE) == "dummy":
            self._load_folder_children(item)

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, _ROLE_TYPE) != "image":
            return
        self._select_tree_item(item)

    def _load_folder_children(self, folder_item: QTreeWidgetItem) -> None:
        """Replace dummy child with real image child items and start thumbnail workers."""
        folder_path: str = folder_item.data(0, _ROLE_PAYLOAD)

        # Remove dummy children
        while folder_item.childCount():
            folder_item.removeChild(folder_item.child(0))

        # Clear stale tree_items entries that belonged to this folder
        # (identified by checking which image items are still parented here)
        self._tree_items = {
            k: v for k, v in self._tree_items.items()
            if v.parent() is not folder_item
        }

        with session_scope() as session:
            summaries = ImageBrowserService(session).list_images(folder_path)

        if not summaries:
            empty = QTreeWidgetItem([t("ib3_no_images_in_folder")])
            empty.setData(0, _ROLE_TYPE, "placeholder")
            empty.setFlags(Qt.NoItemFlags)
            folder_item.addChild(empty)
            return

        pool = QThreadPool.globalInstance()
        placeholder_icon = self._make_placeholder_icon()

        for summary in summaries:
            child = QTreeWidgetItem([summary.filename])
            child.setData(0, _ROLE_TYPE, "image")
            child.setData(0, _ROLE_PAYLOAD, summary.image_id)
            child.setData(0, _ROLE_CACHE, summary.thumbnail_cache_key)
            child.setIcon(0, placeholder_icon)

            # Face-count tooltip
            if summary.face_count > 0:
                tip = f"{summary.face_count} arc"
                if summary.unknown_face_count:
                    tip += f"  ({summary.unknown_face_count} ismeretlen)"
                child.setToolTip(0, tip)

            folder_item.addChild(child)
            self._tree_items[summary.thumbnail_cache_key] = child

            # Use cached thumbnail if available
            if summary.thumbnail_cache_key in self._thumb_cache:
                pixmap = self._thumb_cache[summary.thumbnail_cache_key]
                child.setIcon(0, QIcon(pixmap))
            elif Path(summary.file_path).exists():
                worker = ThumbnailRunnable(
                    file_path=summary.file_path,
                    cache_key=summary.thumbnail_cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(worker)
            else:
                log.warning("Missing file for thumbnail: %s", summary.file_path)

        log.info(
            "Loaded %d image items for folder %s", len(summaries), folder_path
        )

    # ──────────────────────────────────────────────────────────────────
    # Keyboard navigation
    # ──────────────────────────────────────────────────────────────────

    def _select_tree_item(self, item: QTreeWidgetItem) -> None:
        """Select item visually in the tree and load the image."""
        self._current_tree_item = item
        self._file_tree.setCurrentItem(item)
        self._file_tree.scrollToItem(item)
        self._load_image(item.data(0, _ROLE_PAYLOAD))
        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        """Grey out ◀ / ▶ when at the very first / last image across all folders."""
        current = self._current_tree_item
        if current is None:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        parent = current.parent()
        if parent is None:
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            return

        idx = parent.indexOfChild(current)
        folder_idx = self._file_tree.indexOfTopLevelItem(parent)

        has_prev = (idx > 0) or (folder_idx > 0)
        has_next = (idx < parent.childCount() - 1) or (
            folder_idx < self._file_tree.topLevelItemCount() - 1
        )

        self._prev_btn.setEnabled(has_prev)
        self._next_btn.setEnabled(has_next)

    def _open_first_image(self) -> None:
        """Expand the first folder and select its first image."""
        if self._file_tree.topLevelItemCount() == 0:
            return
        first_folder = self._file_tree.topLevelItem(0)
        if not first_folder.isExpanded():
            first_folder.setExpanded(True)
        if first_folder.childCount() == 0:
            return
        first_child = first_folder.child(0)
        if first_child.data(0, _ROLE_TYPE) == "image":
            self._select_tree_item(first_child)

    def _navigate_prev(self) -> None:
        self._navigate(-1)

    def _navigate_next(self) -> None:
        self._navigate(+1)

    def _navigate(self, direction: int) -> None:
        current = self._current_tree_item
        if current is None:
            return

        parent = current.parent()
        if parent is None:
            return

        idx = parent.indexOfChild(current)
        next_idx = idx + direction

        # Stay within the same folder if possible
        if 0 <= next_idx < parent.childCount():
            candidate = parent.child(next_idx)
            if candidate.data(0, _ROLE_TYPE) == "image":
                self._select_tree_item(candidate)
                return

        # Cross to next / previous folder
        folder_idx = self._file_tree.indexOfTopLevelItem(parent)
        next_folder_idx = folder_idx + direction

        if not (0 <= next_folder_idx < self._file_tree.topLevelItemCount()):
            return  # Already at first/last folder

        next_folder = self._file_tree.topLevelItem(next_folder_idx)

        # Collapse current folder, expand the target folder
        parent.setExpanded(False)
        if not next_folder.isExpanded():
            next_folder.setExpanded(True)
            # _on_tree_item_expanded runs synchronously → children populated now

        n = next_folder.childCount()
        if n == 0:
            return

        # Pick first child when going forward, last when going backward
        child_idx = 0 if direction > 0 else n - 1
        candidate = next_folder.child(child_idx)
        if candidate.data(0, _ROLE_TYPE) == "image":
            self._select_tree_item(candidate)

    # ──────────────────────────────────────────────────────────────────
    # Thumbnail callbacks
    # ──────────────────────────────────────────────────────────────────

    @Slot(str, QImage)
    def _on_thumbnail_ready(self, cache_key: str, qimage: QImage) -> None:
        pixmap = QPixmap.fromImage(qimage)
        self._thumb_cache[cache_key] = pixmap
        tree_item = self._tree_items.get(cache_key)
        if tree_item is not None:
            tree_item.setIcon(0, QIcon(pixmap))

    @Slot(str)
    def _on_thumbnail_failed(self, cache_key: str) -> None:
        pass  # placeholder icon stays

    # ──────────────────────────────────────────────────────────────────
    # Image loading
    # ──────────────────────────────────────────────────────────────────

    def _load_image(self, image_id: int) -> None:
        log.info("Image selected: image_id=%d", image_id)

        # CRITICAL: clear face selection state before loading new image
        self._selected_face_id = None
        self._editing_face_id = None
        self._hide_inline_editor()
        self._cancel_rename()

        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                log.warning("Image %d not found in DB", image_id)
                return
            for f in img.faces:
                _ = f.person
            self._current_image_id = image_id
            self._current_path = img.file_path
            self._detection_done = img.detection_done
            photo_date = img.photo_date or ""
            # Track current folder
            self._current_folder = str(Path(img.file_path).parent)
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

        p = Path(self._current_path)
        self._folder_label.setText(str(p.parent))
        self._filename_label.setText(p.name)
        self._load_exif_suggestion(self._current_path)

        from app.utils.image_utils import load_image_bgr
        img_bgr = load_image_bgr(self._current_path)
        if img_bgr is None:
            self._orig_img_bgr = None
            self._full_pixmap = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(t("ibp_load_error", path=self._current_path))
            self._clear_face_panel()
            log.warning("Cannot load image file: %s", self._current_path)
            return

        self._orig_img_bgr = img_bgr
        self._reset_zoom()
        self._redraw_faces()
        self._clear_face_panel()
        self._reload_persons_combo()

    def _clear_current_image(self) -> None:
        """Reset preview when switching folders."""
        self._current_image_id = None
        self._current_path = ""
        self._face_data = []
        self._selected_face_id = None
        self._editing_face_id = None
        self._orig_img_bgr = None
        self._full_pixmap = None
        self._image_label.set_source_pixmap(None)
        self._image_label.setText(t("ib3_select_image_hint"))
        self._folder_label.setText("")
        self._filename_label.setText("")
        self._exif_row_widget.setVisible(False)
        self._clear_face_panel()
        self._hide_inline_editor()
        self._draw_mode_btn.setChecked(False)
        self._reset_zoom()

    # ──────────────────────────────────────────────────────────────────
    # Face click / interaction
    # ──────────────────────────────────────────────────────────────────

    def _on_image_clicked(self, lx: int, ly: int) -> None:
        self._new_name_edit.clearFocus()
        self._rename_edit.clearFocus()

        if self._full_pixmap is None:
            return

        orig_x, orig_y = self._label_to_image(lx, ly)
        if orig_x < 0 or orig_y < 0:
            return

        clicked_id = self._hit_test(orig_x, orig_y)
        if clicked_id is None:
            self._hide_inline_editor()
            self._selected_face_id = None
            self._redraw_faces()
            self._clear_face_panel()
            return

        if self._selected_face_id != clicked_id:
            self._selected_face_id = clicked_id
            self._redraw_faces()
            self._show_face_info(clicked_id)

        self._show_inline_editor(clicked_id)

    def _on_image_right_clicked(self, lx: int, ly: int) -> None:
        orig_x, orig_y = self._label_to_image(lx, ly)
        if orig_x < 0 or orig_y < 0:
            return
        face_id = self._hit_test(orig_x, orig_y)
        if face_id is None:
            return

        if self._selected_face_id != face_id:
            self._selected_face_id = face_id
            self._redraw_faces()
            self._show_face_info(face_id)
        self._hide_inline_editor()

        entry = next((f for f in self._face_data if f[0] == face_id), None)
        person_name = entry[5] if entry else None

        menu = QMenu(self)
        title_text = person_name if person_name else t("ibp_ctx_unknown_face")
        title = menu.addAction(f"[ {title_text} ]")
        title.setEnabled(False)
        menu.addSeparator()
        edit_action = menu.addAction(t("ibp_ctx_edit_bbox"))
        menu.addSeparator()
        delete_action = menu.addAction(t("ibp_ctx_delete"))

        chosen = menu.exec(self._image_label.mapToGlobal(QPoint(lx, ly)))
        if chosen == edit_action:
            self._start_face_edit(face_id)
        elif chosen == delete_action:
            self._delete_face(face_id)

    # ──────────────────────────────────────────────────────────────────
    # Draw mode
    # ──────────────────────────────────────────────────────────────────

    def _on_draw_mode_toggled(self, active: bool) -> None:
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if active:
            self._hide_inline_editor()
        else:
            self._editing_face_id = None
            self._draw_hint.setText(t("ibp_draw_hint_add"))

    def _on_rect_drawn(self, label_rect: QRect) -> None:
        coords = self._label_rect_to_image(label_rect)
        if coords is None:
            return
        ix, iy, iw, ih = coords

        if self._current_image_id is None:
            return

        if self._editing_face_id is not None:
            edited_id = self._editing_face_id
            self._editing_face_id = None
            self._draw_hint.setText(t("ibp_draw_hint_add"))
            self._update_face_bbox(edited_id, ix, iy, iw, ih)
            # Reload data, select the edited face, redraw once
            self._fetch_face_data()
            self._selected_face_id = edited_id
            self._redraw_faces()
            self._show_face_info(edited_id)
        else:
            new_face_id = self._save_manual_face(
                self._current_image_id, ix, iy, iw, ih
            )
            # Reload the full face list (includes the just-saved face),
            # select it and redraw once — draw mode intentionally stays ON
            # so the user can immediately add more faces without re-clicking.
            self._fetch_face_data()
            if new_face_id is not None:
                self._selected_face_id = new_face_id
                self._redraw_faces()
                self._show_face_info(new_face_id)

    def _start_face_edit(self, face_id: int) -> None:
        self._hide_inline_editor()
        self._editing_face_id = face_id
        self._selected_face_id = face_id
        self._redraw_faces()
        self._show_face_info(face_id)
        self._draw_hint.setText(t("ibp_draw_hint_edit"))
        self._draw_mode_btn.setChecked(True)

    def _save_manual_face(
        self, img_id: int, x: int, y: int, w: int, h: int
    ) -> Optional[int]:
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
            face = Face(
                image_id=img_id,
                bbox_x=x, bbox_y=y, bbox_w=w, bbox_h=h,
                confidence=1.0,
                detector_backend="manual",
                crop_path=None,
            )
            session.add(face)
            session.flush()
            new_id = face.id
            if crops_dir is not None and self._orig_img_bgr is not None:
                crop_path = save_face_crop(
                    img_bgr=self._orig_img_bgr,
                    detection=detection,
                    crops_dir=crops_dir,
                    image_id=img_id,
                    thumbnail_size=thumbnail_size,
                    face_index=new_id,
                )
                if crop_path is not None:
                    face.crop_path = str(crop_path)

        log.info(
            "Manual face saved: image_id=%d face_id=%d bbox=(%d,%d,%d,%d)",
            img_id, new_id, x, y, w, h,
        )
        return new_id

    def _delete_face(self, face_id: int) -> None:
        if self._inline_editor_face_id == face_id:
            self._hide_inline_editor()
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

    def _label_rect_to_image(
        self, rect: QRect
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        x2, y2 = self._label_to_image(rect.right(), rect.bottom())
        if x1 < 0 or y1 < 0:
            return None
        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
            x2 = min(max(x2, 0), iw - 1)
            y2 = min(max(y2, 0), ih - 1)
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _image_to_label(self, ix: int, iy: int) -> Tuple[int, int]:
        if self._full_pixmap is None:
            return -1, -1
        cr = self._image_label.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._full_pixmap.width(), self._full_pixmap.height()
        if pw == 0 or ph == 0:
            return -1, -1
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        ox = cr.x() + (lw - pw * eff) / 2 + self._pan_x
        oy = cr.y() + (lh - ph * eff) / 2 + self._pan_y
        return int(ox + ix * eff), int(oy + iy * eff)

    def _hit_test(self, ox: int, oy: int) -> Optional[int]:
        for face_id, x, y, w, h, _, _ in self._face_data:
            if x <= ox <= x + w and y <= oy <= y + h:
                return face_id
        return None

    def _redraw_faces(self) -> None:
        if self._orig_img_bgr is None:
            return
        annotated = _draw_faces(
            self._orig_img_bgr, self._face_data, self._selected_face_id
        )
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
        self._hide_inline_editor()

    def _on_pan_moved(self, dx: int, dy: int) -> None:
        self._pan_x += dx
        self._pan_y += dy
        self._clamp_pan()
        self._image_label.set_zoom_pan(self._zoom, self._pan_x, self._pan_y)
        self._hide_inline_editor()

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
    # Inline face editor
    # ──────────────────────────────────────────────────────────────────

    def _show_inline_editor(self, face_id: int) -> None:
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, bx, by, bw, bh, person_name, person_id = entry

        persons: List[Tuple[int, str]] = [
            (self._person_combo.itemData(i), self._person_combo.itemText(i))
            for i in range(self._person_combo.count())
        ]
        self._inline_editor.populate(person_name, person_id, persons)
        self._inline_editor.adjustSize()

        lx_top, ly_top = self._image_to_label(bx, by)
        _, ly_bot = self._image_to_label(bx, by + bh)
        if lx_top < 0:
            return

        editor_w = self._inline_editor.width()
        editor_h = self._inline_editor.sizeHint().height()
        cr_w = self._image_label.contentsRect().width()
        cr_h = self._image_label.contentsRect().height()

        if ly_top - editor_h - 8 >= 0:
            pos_y = ly_top - editor_h - 8
        elif ly_bot >= 0 and ly_bot + editor_h + 8 <= cr_h:
            pos_y = ly_bot + 8
        else:
            pos_y = max(4, ly_top - editor_h // 2)

        pos_x = max(4, min(lx_top, cr_w - editor_w - 4))
        pos_y = max(4, min(pos_y, cr_h - editor_h - 4))

        self._inline_editor.move(pos_x, pos_y)
        self._inline_editor.show()
        self._inline_editor.raise_()
        self._inline_editor_face_id = face_id

    def _hide_inline_editor(self) -> None:
        self._inline_editor.hide()
        self._inline_editor_face_id = None

    def _on_inline_assign(self, person_id: int) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        log.info(
            "Inline assign: face_id=%d → person_id=%d", face_id, person_id
        )
        self._hide_inline_editor()
        with session_scope() as session:
            IdentityService(session).reassign_face(face_id, person_id)
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()

    def _on_inline_create(self, name: str) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        self._hide_inline_editor()
        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            IdentityService(session).reassign_face(face_id, person_id)
        log.info(
            "Inline create+assign: face_id=%d new person='%s' person_id=%d",
            face_id, name, person_id,
        )
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._open_person_info_dialog(person_id)

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
            self._face_status_label.setText(t("ibp_identified"))
            self._person_name_label.setText(person_name)
            self._person_name_label.setVisible(True)
            self._rename_btn.setVisible(True)
            self._person_info_btn.setVisible(True)
        else:
            self._face_status_label.setText(t("ibp_not_identified"))
            self._person_name_label.setVisible(False)
            self._rename_btn.setVisible(False)
            self._person_info_btn.setVisible(False)
        self._rename_edit.setVisible(False)

    def _clear_face_panel(self) -> None:
        self._cancel_rename()
        if self._detection_done and not self._face_data:
            self._face_status_label.setText(t("ibp_no_face_detected"))
            self._face_status_label.setStyleSheet(
                "color: #ffaa44; font-size: 11px; font-style: italic;"
            )
        else:
            self._face_status_label.setText(t("ibp_click_face"))
            self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self._person_name_label.setVisible(False)
        self._rename_btn.setVisible(False)
        self._person_info_btn.setVisible(False)
        self._assign_btn.setEnabled(False)
        self._create_btn.setEnabled(False)
        self._new_name_edit.clear()

    # ──────────────────────────────────────────────────────────────────
    # Photo date
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
        log.debug(
            "photo_date saved for image %d: %r", self._current_image_id, value
        )

    def _load_exif_suggestion(self, path: str) -> None:
        """Read EXIF DateTimeOriginal and show as a suggestion (not auto-fill)."""
        suggestion = _read_exif_date(path)
        if suggestion:
            self._exif_suggestion_label.setText(suggestion)
            self._exif_row_widget.setVisible(True)
        else:
            self._exif_suggestion_label.setText("")
            self._exif_row_widget.setVisible(False)

    def _accept_exif_suggestion(self) -> None:
        suggested = self._exif_suggestion_label.text()
        if not suggested:
            return
        self._photo_date_edit.blockSignals(True)
        self._photo_date_edit.setText(suggested)
        self._photo_date_edit.blockSignals(False)
        self._save_photo_date()
        self._exif_row_widget.setVisible(False)

    # ──────────────────────────────────────────────────────────────────
    # Inline rename
    # ──────────────────────────────────────────────────────────────────

    def _start_rename(self) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
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
            entry = next(
                (f for f in self._face_data if f[0] == self._selected_face_id), None
            )
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
            entry = next(
                (f for f in self._face_data if f[0] == self._selected_face_id), None
            )
            if entry and entry[5]:
                self._person_name_label.setVisible(True)
                self._rename_btn.setVisible(True)
                self._person_info_btn.setVisible(True)

    def _save_person_rename(self, new_name: str) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
        if entry is None:
            return
        person_id = entry[6]
        if person_id is None:
            return
        with session_scope() as session:
            IdentityService(session).rename_person(person_id, new_name)
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

    def _recent_person_rank(self, session) -> Dict[int, int]:
        rank: Dict[int, int] = {}
        for person_id in self._recent_assignment_person_ids:
            if person_id not in rank:
                rank[person_id] = len(rank)
        return rank

    def _remember_recent_person(self, person_id: int) -> None:
        self._recent_assignment_person_ids = [
            person_id,
            *(pid for pid in self._recent_assignment_person_ids if pid != person_id),
        ][:20]

    # ──────────────────────────────────────────────────────────────────
    # Assign / create actions (info panel buttons)
    # ──────────────────────────────────────────────────────────────────

    def _on_assign_existing(self) -> None:
        if self._selected_face_id is None:
            return
        person_id = self._person_combo.currentData()
        if person_id is None:
            return
        log.info(
            "Assign existing: face_id=%d → person_id=%d",
            self._selected_face_id, person_id,
        )
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
            QMessageBox.warning(
                self, t("ibp_empty_name_title"), t("ibp_empty_name_msg")
            )
            return
        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            IdentityService(session).reassign_face(self._selected_face_id, person_id)
        log.info(
            "Create+assign: face_id=%d new person='%s' person_id=%d",
            self._selected_face_id, name, person_id,
        )
        self._remember_recent_person(person_id)
        self._new_name_edit.clear()
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._open_person_info_dialog(person_id)

    def _open_person_info_dialog(self, person_id: int) -> None:
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
        log.info("Person details saved: person_id=%d", person_id)
        self._reload_persons_combo()
        self.person_data_changed.emit()

    def _on_person_info(self) -> None:
        if self._selected_face_id is None:
            return
        entry = next(
            (f for f in self._face_data if f[0] == self._selected_face_id), None
        )
        if entry is None or entry[6] is None:
            return
        self._open_person_info_dialog(entry[6])

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
            return
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._hide_inline_editor()
        self._image_label.update()
