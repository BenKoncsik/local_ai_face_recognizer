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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, QSize, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.ui.widgets.person_search_select import PersonSearchSelect
from app.ui.widgets.place_search_select import PlaceSearchSelect
from app.ui.widgets.universal_search_bar import UniversalSearchBar

from app.db.database import session_scope
from app.db.models import Face, Image, Person, Place
from app.services.family_service import FamilyImageSearchCriteria, FamilyService
from app.services.identity_service import IdentityService
from app.services.image_browser_service import (
    FolderSummary,
    ImageBrowserService,
    ImageSummary,
)
from app.services.place_service import ANONYMOUS_GPS_PLACE_NAME, PlaceService
from app.ui.dialogs.person_info_dialog import PersonInfoDialog
from app.ui.i18n import t
from app.workers.thumbnail_worker import ThumbnailRunnable

log = logging.getLogger(__name__)

# (face_id, bbox_x, bbox_y, bbox_w, bbox_h, person_name_or_None, person_id_or_None)
_FaceData = Tuple[int, int, int, int, int, Optional[str], Optional[int]]


@dataclass
class _BboxEdit:
    """One undo/redo entry for a face bounding-box modification."""
    face_id: int
    old_bbox: Tuple[int, int, int, int]  # x, y, w, h in original image coords
    new_bbox: Tuple[int, int, int, int]
    op_type: str  # "move" | "resize"


_TREE_THUMB_SIZE = 56   # max side of thumbnail icon in tree

# Data roles stored on QTreeWidgetItem
_ROLE_TYPE      = Qt.UserRole        # "folder" | "image" | "dummy"
_ROLE_PAYLOAD   = Qt.UserRole + 1    # folder_path (str) | image_id (int)
_ROLE_CACHE     = Qt.UserRole + 2    # thumbnail_cache_key (str, images only)
_ROLE_FILE_PATH = Qt.UserRole + 3    # file_path (str, images only)

# How many images ahead to prefetch in Drive mode.
_DRIVE_PREFETCH_AHEAD = 4

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


_RE_FILENAME_DATETIME = re.compile(
    r"^(\d{4})\.(\d{2})\.(\d{2})-(\d{2})\.(\d{2})\.(\d{2})"
)
_RE_FILENAME_DATE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})")


def _parse_date_from_filename(path: str) -> Optional[str]:
    """Return 'YYYY.MM.DD HH:MM:SS' or 'YYYY.MM.DD' if the filename starts with a
    date/datetime pattern, else None."""
    stem = Path(path).stem
    m = _RE_FILENAME_DATETIME.match(stem)
    if m:
        yyyy, mm, dd, hh, mi, ss = m.groups()
        return f"{yyyy}.{mm}.{dd} {hh}:{mi}:{ss}"
    m = _RE_FILENAME_DATE.match(stem)
    if m:
        yyyy, mm, dd = m.groups()
        return f"{yyyy}.{mm}.{dd}"
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
    show_bboxes: bool = True,
    bbox_opacity: float = 1.0,
    show_labels: bool = True,
    label_opacity: float = 1.0,
) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw

    from app.ui.helpers.label_placement import FaceLabel, place_labels

    img = img_bgr.copy()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    image_h, image_w = img.shape[:2]
    font_size = max(34, min(96, int(min(image_w, image_h) * 0.028)))

    # --- Phase 1: measure all labels (needed for layout even if hidden) ---
    face_meta = []  # (name, font, pad_x, pad_y, label_w, label_h, color_rgb)
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
        face_meta.append((name, font, pad_x, pad_y, label_w, label_h, color_rgb))

    # --- Phase 2: compute collision-free positions ---
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

    # --- Phase 3: render labels with opacity via RGBA compositing ---
    if show_labels and label_opacity > 0.0 and faces:
        lbl_alpha = int(label_opacity * 255)
        overlay = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)

        for i, layout in enumerate(layouts):
            name, font, pad_x, pad_y, label_w, label_h, color_rgb = face_meta[i]

            if layout.leader_start and layout.leader_end:
                odraw.line([layout.leader_start, layout.leader_end],
                           fill=(120, 120, 120, lbl_alpha), width=2)

            odraw.rectangle(
                [layout.label_x, layout.label_y,
                 layout.label_x + label_w, layout.label_y + label_h],
                fill=(20, 20, 20, lbl_alpha),
            )
            odraw.text(
                (layout.label_x + pad_x, layout.label_y + pad_y),
                name,
                font=font,
                fill=(*color_rgb, lbl_alpha),
            )

        base_rgba = pil_img.convert("RGBA")
        pil_img = PILImage.alpha_composite(base_rgba, overlay).convert("RGB")

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- Phase 4: render bounding boxes ---
    if show_bboxes and bbox_opacity > 0.0:
        if bbox_opacity >= 0.99:
            for face_id, x, y, w, h, _, _ in faces:
                selected = face_id == selected_id
                color = (50, 220, 50) if selected else (180, 180, 180)
                thickness = 3 if selected else 2
                cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        else:
            overlay_cv = img.copy()
            for face_id, x, y, w, h, _, _ in faces:
                selected = face_id == selected_id
                color = (50, 220, 50) if selected else (180, 180, 180)
                thickness = 3 if selected else 2
                cv2.rectangle(overlay_cv, (x, y), (x + w, y + h), color, thickness)
            cv2.addWeighted(overlay_cv, bbox_opacity, img, 1.0 - bbox_opacity, 0, img)

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

        self._person_search = PersonSearchSelect()
        self._person_search.person_selected.connect(self._on_assign)
        layout.addWidget(self._person_search)

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
        self._person_search.retranslate()

    def populate(
        self,
        person_name: Optional[str],
        person_id: Optional[int],
        persons: List[Person],
        priority_ids: Optional[List[int]] = None,
    ) -> None:
        self._name_lbl.setText(person_name or "?")
        self._person_search.set_persons(persons, priority_ids=priority_ids)
        if person_id is not None:
            self._person_search.set_current_by_id(person_id)
        self._new_edit.clear()

    def reset_search(self) -> None:
        """Clear the person search filter after a successful assignment."""
        self._person_search.reset_filter()

    def focus_search(self) -> None:
        """Set keyboard focus to the person search input."""
        self._person_search.focus_search()

    def _on_assign(self, _person_id: int = 0) -> None:
        pid = self._person_search.current_person_id()
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
            from app.services.shortcut_service import get_shortcut_service, normalize_key
            svc = get_shortcut_service()
            key_str = normalize_key(event)
            if svc.is_enabled():
                prev_sc = svc.get("image.previous")
                next_sc = svc.get("image.next")
                prev_key = (prev_sc.current_key if prev_sc and prev_sc.current_key else "Left")
                next_key = (next_sc.current_key if next_sc and next_sc.current_key else "Right")
            else:
                prev_key, next_key = "Left", "Right"
            if key_str == prev_key:
                self.go_prev.emit()
                event.accept()
                return
            elif key_str == next_key:
                self.go_next.emit()
                event.accept()
                return
        super().keyPressEvent(event)


class _DrawableImageLabel(QLabel):
    clicked              = Signal(int, int)
    rect_drawn           = Signal(QRect)
    right_clicked        = Signal(int, int)
    wheel_zoomed         = Signal(int, int, int)
    pan_moved            = Signal(int, int)
    go_prev              = Signal()
    go_next              = Signal()
    # Interactive bbox-edit mode signals
    bbox_edit_committed  = Signal(int, int, int, int)   # drag-release save, stay in edit mode
    bbox_edit_finalized  = Signal(int, int, int, int)   # click-outside / Enter → save AND exit
    bbox_edit_cancelled  = Signal()
    undo_requested       = Signal()
    redo_requested       = Signal()

    _BORDER_NORMAL = "QLabel { background: #1a1a1a; }"
    _BORDER_DRAW   = "QLabel { background: #1a1a1a; border: 2px solid #ffcc00; }"
    _BORDER_EDIT   = "QLabel { background: #1a1a1a; border: 2px solid #32dc32; }"

    # Handle rendering constants
    _HANDLE_HALF = 6    # half-side of handle squares in display pixels
    _HIT_RADIUS  = 10   # hit-test radius around each handle
    _MIN_DIM     = 12   # minimum bbox dimension in image pixels

    # Handle identifiers (corners, edge-midpoints, centre)
    _HANDLES = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)   # needed for hover cursor changes without button held
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self._mid_start: Optional[QPoint] = None
        self._source_pix: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # Interactive edit mode
        self._imode: bool = False
        self._imode_bbox: Optional[Tuple[int, int, int, int]] = None   # image coords
        self._imode_img_w: int = 0
        self._imode_img_h: int = 0
        self._imode_handle: Optional[str] = None   # which handle is being dragged
        self._imode_drag_start: Optional[QPointF] = None
        self._imode_drag_start_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_op_type: str = "resize"   # set by mouseReleaseEvent
        self.setStyleSheet(self._BORDER_NORMAL)

    # ── Public setters ─────────────────────────────────────────────────

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
        if enabled:
            self._exit_interactive_edit(commit=False, silent=True)
        self.setStyleSheet(self._BORDER_DRAW if enabled else self._BORDER_NORMAL)
        self.update()

    def set_interactive_edit(
        self,
        bbox: Tuple[int, int, int, int],
        img_w: int,
        img_h: int,
    ) -> None:
        """Enter interactive-edit mode showing resize/move handles for *bbox*."""
        self._imode = True
        self._imode_bbox = bbox
        self._imode_img_w = img_w
        self._imode_img_h = img_h
        self._imode_handle = None
        self._imode_drag_start = None
        self._imode_drag_start_bbox = None
        self._draw_mode = False
        self._start = None
        self._end = None
        self.setStyleSheet(self._BORDER_EDIT)
        self.update()

    def cancel_interactive_edit(self) -> None:
        """Exit interactive-edit mode; emit bbox_edit_cancelled."""
        self._exit_interactive_edit(commit=False, silent=False)

    def commit_interactive_edit(self) -> None:
        """Commit the current bbox and exit interactive-edit mode."""
        self._exit_interactive_edit(commit=True, silent=False)

    def dismiss_interactive_edit(self) -> None:
        """Silently exit interactive-edit mode without emitting any signal.

        Use this when the bbox was already persisted by intermediate commits
        and you just want to clear the imode UI state without triggering any
        save or restore logic.
        """
        self._exit_interactive_edit(commit=False, silent=True)

    def _exit_interactive_edit(
        self, *, commit: bool, silent: bool
    ) -> None:
        if not self._imode:
            return
        bbox = self._imode_bbox
        self._imode = False
        self._imode_bbox = None
        self._imode_handle = None
        self._imode_drag_start = None
        self._imode_drag_start_bbox = None
        self.setStyleSheet(self._BORDER_NORMAL)
        self.update()
        if not silent:
            if commit and bbox is not None:
                # finalized = save AND exit (click-outside or Enter)
                self.bbox_edit_finalized.emit(*bbox)
            else:
                self.bbox_edit_cancelled.emit()

    # ── Coordinate helpers ──────────────────────────────────────────────

    def _imode_transform(self) -> Optional[Tuple[float, float, float]]:
        """Return (eff, ox, oy) for image→display mapping, or None."""
        if self._source_pix is None:
            return None
        cr = self.contentsRect()
        lw, lh = cr.width(), cr.height()
        pw, ph = self._source_pix.width(), self._source_pix.height()
        if pw <= 0 or ph <= 0 or lw <= 0 or lh <= 0:
            return None
        base_scale = min(lw / pw, lh / ph)
        eff = base_scale * self._zoom
        disp_w = pw * eff
        disp_h = ph * eff
        ox = cr.x() + (lw - disp_w) / 2 + self._pan_x
        oy = cr.y() + (lh - disp_h) / 2 + self._pan_y
        return eff, ox, oy

    def _img_to_disp(
        self, ix: float, iy: float,
        eff: float, ox: float, oy: float,
    ) -> Tuple[float, float]:
        return ox + ix * eff, oy + iy * eff

    def _disp_to_img(
        self, lx: float, ly: float,
        eff: float, ox: float, oy: float,
    ) -> Tuple[float, float]:
        return (lx - ox) / eff, (ly - oy) / eff

    def _handle_positions(
        self,
        bbox: Tuple[int, int, int, int],
        eff: float,
        ox: float,
        oy: float,
    ) -> Dict[str, Tuple[float, float]]:
        """Return display-coord centre of each handle for *bbox*."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        img_pts = {
            "tl": (x,      y),
            "tc": (cx,     y),
            "tr": (x + w,  y),
            "ml": (x,      cy),
            "mc": (cx,     cy),
            "mr": (x + w,  cy),
            "bl": (x,      y + h),
            "bc": (cx,     y + h),
            "br": (x + w,  y + h),
        }
        return {
            name: self._img_to_disp(ix, iy, eff, ox, oy)
            for name, (ix, iy) in img_pts.items()
        }

    def _hit_test_handle(
        self, lx: float, ly: float
    ) -> Optional[str]:
        """Return the name of the nearest handle if within hit radius."""
        if not self._imode or self._imode_bbox is None:
            return None
        t = self._imode_transform()
        if t is None:
            return None
        eff, ox, oy = t
        positions = self._handle_positions(self._imode_bbox, eff, ox, oy)
        best_name: Optional[str] = None
        best_dist = self._HIT_RADIUS + 1.0
        for name, (hx, hy) in positions.items():
            dist = ((lx - hx) ** 2 + (ly - hy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    # ── Drag update ─────────────────────────────────────────────────────

    def _apply_drag(
        self,
        handle: str,
        start: QPointF,
        current: QPointF,
    ) -> None:
        """Update _imode_bbox according to the current drag position."""
        if self._imode_drag_start_bbox is None:
            return
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, oy = t

        sx, sy = self._disp_to_img(start.x(), start.y(), eff, ox, oy)
        cx, cy = self._disp_to_img(current.x(), current.y(), eff, ox, oy)
        dx, dy = cx - sx, cy - sy

        x0, y0, w0, h0 = self._imode_drag_start_bbox
        iw, ih = self._imode_img_w, self._imode_img_h
        _m = self._MIN_DIM

        if handle == "mc":
            # Move the whole box
            nx = max(0, min(iw - w0, x0 + dx))
            ny = max(0, min(ih - h0, y0 + dy))
            self._imode_bbox = (int(nx), int(ny), w0, h0)
            return

        # Resize: determine which edges move
        nx, ny, nw, nh = x0, y0, w0, h0

        if handle in ("tl", "ml", "bl"):
            new_x = max(0, min(x0 + w0 - _m, x0 + dx))
            nw = w0 - (int(new_x) - x0)
            nx = int(new_x)
        if handle in ("tr", "mr", "br"):
            new_w = max(_m, min(iw - x0, w0 + dx))
            nw = int(new_w)
        if handle in ("tl", "tc", "tr"):
            new_y = max(0, min(y0 + h0 - _m, y0 + dy))
            nh = h0 - (int(new_y) - y0)
            ny = int(new_y)
        if handle in ("bl", "bc", "br"):
            new_h = max(_m, min(ih - y0, h0 + dy))
            nh = int(new_h)

        # Final clamp: ensure box stays inside image
        nx = max(0, min(iw - nw, nx))
        ny = max(0, min(ih - nh, ny))
        nw = max(_m, min(iw - nx, nw))
        nh = max(_m, min(ih - ny, nh))
        self._imode_bbox = (nx, ny, nw, nh)

    # ── Mouse events ────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton:
            return
        pos = event.position()
        lx, ly = pos.x(), pos.y()

        if self._imode:
            handle = self._hit_test_handle(lx, ly)
            if handle is not None:
                self._imode_handle = handle
                self._imode_drag_start = pos
                self._imode_drag_start_bbox = self._imode_bbox
            # Click outside any handle exits edit mode
            elif self._imode_bbox is not None:
                self._exit_interactive_edit(commit=True, silent=False)
            return

        if self._draw_mode:
            self._start = event.position().toPoint()
            self._end = self._start
            self.update()
        else:
            self.clicked.emit(int(lx), int(ly))

    def mouseMoveEvent(self, event) -> None:
        if self._mid_start is not None:
            cur = event.position().toPoint()
            self.pan_moved.emit(
                cur.x() - self._mid_start.x(),
                cur.y() - self._mid_start.y(),
            )
            self._mid_start = cur
            return

        pos = event.position()
        lx, ly = pos.x(), pos.y()

        if self._imode and self._imode_handle is not None:
            self._apply_drag(self._imode_handle, self._imode_drag_start, pos)
            # Update cursor based on which handle is being dragged
            self._update_edit_cursor(self._imode_handle)
            self.update()
            return

        if self._imode:
            # Hover: update cursor to show which handle would be grabbed
            handle = self._hit_test_handle(lx, ly)
            self._update_edit_cursor(handle)
            return

        if self._draw_mode and self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self._mid_start = None
            self.setCursor(Qt.CrossCursor if not self._imode else Qt.ArrowCursor)
            return
        if event.button() != Qt.LeftButton:
            return

        if self._imode and self._imode_handle is not None:
            was_handle = self._imode_handle
            old_bbox = self._imode_drag_start_bbox
            new_bbox = self._imode_bbox
            self._imode_handle = None
            self._imode_drag_start = None
            self._imode_drag_start_bbox = None
            # Emit a committed signal if the bbox actually changed
            if new_bbox is not None and old_bbox != new_bbox:
                # Determine op type and tag BEFORE emitting (slot runs synchronously)
                ow, oh = old_bbox[2], old_bbox[3]
                nw, nh = new_bbox[2], new_bbox[3]
                self._last_op_type = "move" if (ow == nw and oh == nh) else "resize"
                self.bbox_edit_committed.emit(*new_bbox)
            self.update()
            return

        if not self._draw_mode or self._start is None:
            return
        end = event.position().toPoint()
        rect = QRect(self._start, end).normalized()
        self._start = None
        self._end = None
        self.update()
        if rect.width() >= 8 and rect.height() >= 8:
            self.rect_drawn.emit(rect)

    def _update_edit_cursor(self, handle: Optional[str]) -> None:
        cursors = {
            "tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
            "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
            "tc": Qt.SizeVerCursor,   "bc": Qt.SizeVerCursor,
            "ml": Qt.SizeHorCursor,   "mr": Qt.SizeHorCursor,
            "mc": Qt.SizeAllCursor,
        }
        if handle and handle in cursors:
            self.setCursor(cursors[handle])
        else:
            self.setCursor(Qt.CrossCursor)

    # ── Key events ──────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        from app.services.shortcut_service import get_shortcut_service, normalize_key
        svc = get_shortcut_service()
        key_str = normalize_key(event)

        if self._imode:
            if key_str == "Esc":
                self._exit_interactive_edit(commit=False, silent=False)
                event.accept()
                return
            if key_str in ("Return", "Enter"):
                self._exit_interactive_edit(commit=True, silent=False)
                event.accept()
                return

        if svc.is_enabled():
            undo_sc = svc.get("bbox.undo")
            redo_sc = svc.get("bbox.redo")
            undo_key = undo_sc.current_key if undo_sc and undo_sc.current_key else "Ctrl+Z"
            redo_key = redo_sc.current_key if redo_sc and redo_sc.current_key else "Ctrl+Y"
            prev_sc = svc.get("image.previous")
            next_sc = svc.get("image.next")
            prev_key = prev_sc.current_key if prev_sc and prev_sc.current_key else "Left"
            next_key = next_sc.current_key if next_sc and next_sc.current_key else "Right"
        else:
            undo_key, redo_key = "Ctrl+Z", "Ctrl+Y"
            prev_key, next_key = "Left", "Right"

        if key_str == undo_key:
            self.undo_requested.emit()
            event.accept()
        elif key_str == redo_key:
            self.redo_requested.emit()
            event.accept()
        elif key_str == prev_key and not self._imode:
            self.go_prev.emit()
            event.accept()
        elif key_str == next_key and not self._imode:
            self.go_next.emit()
            event.accept()
        else:
            super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        if self._imode:
            # Cancel edit before opening context menu
            self._exit_interactive_edit(commit=True, silent=False)
        self.right_clicked.emit(event.pos().x(), event.pos().y())

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta != 0:
            pos = event.position().toPoint()
            self.wheel_zoomed.emit(delta, pos.x(), pos.y())
        event.accept()

    # ── Painting ────────────────────────────────────────────────────────

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

        if self._imode and self._imode_bbox is not None:
            self._paint_edit_handles(painter)

    def _paint_edit_handles(self, painter: QPainter) -> None:
        t = self._imode_transform()
        if t is None:
            return
        eff, ox, oy = t
        bbox = self._imode_bbox
        x, y, w, h = bbox
        dx = ox + x * eff
        dy = oy + y * eff
        dw = w * eff
        dh = h * eff

        painter.setRenderHint(QPainter.Antialiasing, True)

        # Bbox outline (bright green dashed)
        painter.setPen(QPen(QColor(50, 220, 50), 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(dx, dy, dw, dh))

        # Handles
        positions = self._handle_positions(bbox, eff, ox, oy)
        hs = self._HANDLE_HALF
        for name, (hx, hy) in positions.items():
            if name == "mc":
                # Centre handle: filled circle
                painter.setPen(QPen(QColor(50, 220, 50), 1))
                painter.setBrush(QBrush(QColor(50, 220, 50, 200)))
                painter.drawEllipse(QPointF(hx, hy), float(hs), float(hs))
            else:
                painter.setPen(QPen(QColor(50, 220, 50), 1))
                painter.setBrush(QBrush(QColor(20, 40, 20, 220)))
                painter.drawRect(QRectF(hx - hs, hy - hs, hs * 2, hs * 2))


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class ImageBrowserPanel(QWidget):
    """Finder-style image browser: folder/image tree | image preview."""

    person_data_changed = Signal()
    # Emitted with (image_id, file_path) whenever a new image is displayed.
    image_displayed = Signal(int, str)
    # Emitted with the selected face's person name (str) or None whenever the
    # active face selection / assignment changes — drives the recording log.
    active_person_changed = Signal(object)

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
        self._all_persons: List[Person] = []
        self._all_places: List[Place] = []
        self._place_filter_id: Optional[int] = None

        # Interactive bbox editor state
        self._interactive_edit_face_id: Optional[int] = None
        self._interactive_edit_orig_bbox: Optional[Tuple[int, int, int, int]] = None
        self._undo_stack: List[_BboxEdit] = []
        self._redo_stack: List[_BboxEdit] = []

        # Overlay state
        self._show_bboxes: bool = True
        self._bbox_opacity: float = 0.7
        self._prev_bbox_opacity: float = 0.7
        self._show_labels: bool = True
        self._label_opacity: float = 0.4
        self._prev_label_opacity: float = 0.4

        # Zoom / pan
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0

        # Thumbnail / tree state
        self._thumb_cache: Dict[str, QPixmap] = {}
        self._tree_items: Dict[str, QTreeWidgetItem] = {}  # cache_key → image tree item
        self._search_result_items: Dict[str, QListWidgetItem] = {}  # cache_key → search list item
        self._current_tree_item: Optional[QTreeWidgetItem] = None

        # Drive mode — set by main window when a Drive project is open
        self._drive_client = None          # GDriveClient | None
        self._drive_mirror_dir: Optional[Path] = None
        self._drive_project_name: str = ""
        self._prefetch_set: set = set()    # image_ids whose fetch is in-flight
        # image_id → requested load: when a fetch completes AND this id is
        # the current image, we display it.
        self._drive_pending_image_id: Optional[int] = None

        # Fullscreen state
        self._fs_hidden_widgets: List[QWidget] = []
        self._fs_was_maximized: bool = False

        # Deoldified pairing state (reset on each image load)
        self._deol_pair_orig_id: Optional[int] = None  # original image ID when current is deoldified
        self._deol_pair_color_path: str = ""           # path to colorized variant
        self._deol_viewing_color: bool = False         # True = showing colorized pixels

        self._build_ui()
        self._setup_shortcuts()
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 8, 0)
        root.setSpacing(0)

        main_splitter = QSplitter(Qt.Horizontal)

        # ── Column 1: Finder-style folder/image tree ──────────────────
        self._tree_panel = QWidget()
        self._tree_panel.setMinimumWidth(200)
        self._tree_panel.setMaximumWidth(420)
        tp_layout = QVBoxLayout(self._tree_panel)
        tp_layout.setContentsMargins(4, 4, 4, 4)
        tp_layout.setSpacing(4)

        self._tree_hdr_lbl = QLabel()
        self._tree_hdr_lbl.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #888;"
        )
        tp_layout.addWidget(self._tree_hdr_lbl)

        # Universal search bar (replaces the old place-filter row)
        self._tree_search_bar = UniversalSearchBar(
            suggest_fn=self._tree_suggest,
            parent=self._tree_panel,
        )
        # NB: only wire search_requested — the search bar emits both
        # tokens_changed AND search_requested on every chip add/remove, so
        # connecting both would run the (expensive) search twice per change.
        self._tree_search_bar.search_requested.connect(self._on_tree_search_changed)
        tp_layout.addWidget(self._tree_search_bar)

        # Flat search results list (shown instead of tree when search is active)
        self._search_results_list = QListWidget()
        self._search_results_list.setIconSize(QSize(_TREE_THUMB_SIZE, _TREE_THUMB_SIZE))
        self._search_results_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._search_results_list.setStyleSheet(
            "QListWidget {"
            "  border: none;"
            "  background: #181825;"
            "  color: #cdd6f4;"
            "  outline: 0;"
            "}"
            "QListWidget::item {"
            "  padding: 3px 4px;"
            "  border-bottom: 1px solid #1e1e2e;"
            "}"
            "QListWidget::item:selected { background: #2d2d50; }"
            "QListWidget::item:hover:!selected { background: #22223a; }"
        )
        self._search_results_list.itemDoubleClicked.connect(
            self._on_search_result_double_clicked
        )
        self._search_results_list.hide()
        tp_layout.addWidget(self._search_results_list, stretch=1)

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

        main_splitter.addWidget(self._tree_panel)

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

        _nav_style = (
            "QPushButton {"
            "  font-size: 16px; padding: 0 10px; min-width: 32px; min-height: 26px;"
            "  background: #313244; border: 1px solid #45475a; border-radius: 4px;"
            "  color: #cdd6f4;"
            "}"
            "QPushButton:hover  { background: #45475a; }"
            "QPushButton:pressed { background: #585b70; }"
            "QPushButton:disabled { color: #45475a; border-color: #313244; }"
            "QPushButton:checked { background: #585b70; border-color: #88aaff; }"
        )

        # Toggle for the left (folder) panel
        self._toggle_left_btn = QPushButton("◧")
        self._toggle_left_btn.setCheckable(True)
        self._toggle_left_btn.setChecked(True)
        self._toggle_left_btn.setStyleSheet(_nav_style)
        self._toggle_left_btn.toggled.connect(self._on_toggle_left_panel)
        top_bar.addWidget(self._toggle_left_btn)

        self._draw_mode_btn = QPushButton()
        self._draw_mode_btn.setCheckable(True)
        self._draw_mode_btn.toggled.connect(self._on_draw_mode_toggled)
        top_bar.addWidget(self._draw_mode_btn)

        top_bar.addStretch()

        # ── Overlay controls ─────────────────────────────────────────────
        _ov_lbl_style = "QLabel { color: #aaa; font-size: 11px; }"
        _ov_pct_style = "QLabel { color: #888; font-size: 11px; min-width: 30px; }"
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

        self._bbox_check = QCheckBox(t("overlay_bboxes"))
        self._bbox_check.setChecked(True)
        self._bbox_check.setToolTip(t("overlay_bbox_tip"))
        self._bbox_check.setStyleSheet(_ov_chk_style)
        top_bar.addWidget(self._bbox_check)

        self._bbox_slider = QSlider(Qt.Horizontal)
        self._bbox_slider.setRange(0, 100)
        self._bbox_slider.setValue(70)
        self._bbox_slider.setToolTip(t("overlay_bbox_tip"))
        self._bbox_slider.setMinimumWidth(60)
        self._bbox_slider.setMaximumWidth(110)
        self._bbox_slider.setStyleSheet(_ov_sld_style)
        top_bar.addWidget(self._bbox_slider)

        self._bbox_pct_label = QLabel("70%")
        self._bbox_pct_label.setStyleSheet(_ov_pct_style)
        self._bbox_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_bar.addWidget(self._bbox_pct_label)

        top_bar.addSpacing(12)

        self._label_check = QCheckBox(t("overlay_labels"))
        self._label_check.setChecked(True)
        self._label_check.setToolTip(t("overlay_label_tip"))
        self._label_check.setStyleSheet(_ov_chk_style)
        top_bar.addWidget(self._label_check)

        self._label_slider = QSlider(Qt.Horizontal)
        self._label_slider.setRange(0, 100)
        self._label_slider.setValue(40)
        self._label_slider.setToolTip(t("overlay_label_tip"))
        self._label_slider.setMinimumWidth(60)
        self._label_slider.setMaximumWidth(110)
        self._label_slider.setStyleSheet(_ov_sld_style)
        top_bar.addWidget(self._label_slider)

        self._label_pct_label = QLabel("40%")
        self._label_pct_label.setStyleSheet(_ov_pct_style)
        self._label_pct_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        top_bar.addWidget(self._label_pct_label)

        top_bar.addSpacing(12)

        # Wire overlay signals
        self._bbox_check.toggled.connect(self._on_ov_bbox_check)
        self._bbox_slider.valueChanged.connect(self._on_ov_bbox_slider)
        self._label_check.toggled.connect(self._on_ov_label_check)
        self._label_slider.valueChanged.connect(self._on_ov_label_slider)
        # ─────────────────────────────────────────────────────────────────

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

        # Toggle for the right (info) panel
        self._toggle_right_btn = QPushButton("◨")
        self._toggle_right_btn.setCheckable(True)
        self._toggle_right_btn.setChecked(True)
        self._toggle_right_btn.setStyleSheet(_nav_style)
        self._toggle_right_btn.toggled.connect(self._on_toggle_right_panel)
        top_bar.addWidget(self._toggle_right_btn)

        im_layout.addLayout(top_bar)

        self._draw_hint = QLabel()
        self._draw_hint.setAlignment(Qt.AlignCenter)
        self._draw_hint.setStyleSheet(
            "color: #ffcc00; font-size: 11px; background: #2a2000; padding: 3px;"
        )
        self._draw_hint.setVisible(False)
        im_layout.addWidget(self._draw_hint)

        # ── Deoldified pair toggle bar ────────────────────────────────────
        self._deoldified_bar = QWidget()
        _deol_row = QHBoxLayout(self._deoldified_bar)
        _deol_row.setContentsMargins(0, 2, 0, 2)
        _deol_row.setSpacing(6)
        self._deol_lbl = QLabel()
        self._deol_lbl.setStyleSheet("color: #888; font-size: 11px;")
        _deol_row.addWidget(self._deol_lbl)
        self._btn_view_bw = QPushButton()
        self._btn_view_bw.setCheckable(True)
        self._btn_view_bw.setStyleSheet(_nav_style)
        self._btn_view_bw.clicked.connect(lambda: self._on_deol_view_toggle(False))
        _deol_row.addWidget(self._btn_view_bw)
        self._btn_view_color = QPushButton()
        self._btn_view_color.setCheckable(True)
        self._btn_view_color.setStyleSheet(_nav_style)
        self._btn_view_color.clicked.connect(lambda: self._on_deol_view_toggle(True))
        _deol_row.addWidget(self._btn_view_color)
        _deol_row.addStretch()
        self._deoldified_bar.setVisible(False)
        im_layout.addWidget(self._deoldified_bar)

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
        self._image_label.bbox_edit_committed.connect(self._on_interactive_bbox_committed)
        self._image_label.bbox_edit_finalized.connect(self._on_interactive_bbox_finalized)
        self._image_label.bbox_edit_cancelled.connect(self._on_interactive_bbox_cancelled)
        self._image_label.undo_requested.connect(self._undo_bbox_edit)
        self._image_label.redo_requested.connect(self._redo_bbox_edit)
        im_layout.addWidget(self._image_label, stretch=1)

        # Register undo/redo with the global ShortcutService so Ctrl+Z/Y
        # fires even when the image label doesn't have keyboard focus.
        from app.services.shortcut_service import get_shortcut_service
        _svc = get_shortcut_service()
        _svc.register("bbox.undo", self._undo_bbox_edit)
        _svc.register("bbox.redo", self._redo_bbox_edit)

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
        self._folder_label.setMinimumWidth(0)
        self._folder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._folder_label.setStyleSheet("color: #88aaff; font-size: 11px;")
        info_layout.addWidget(self._folder_label)

        self._filename_hdr = QLabel()
        self._filename_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._filename_hdr)

        self._filename_label = QLabel("")
        self._filename_label.setWordWrap(True)
        self._filename_label.setMinimumWidth(0)
        self._filename_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._filename_label.setStyleSheet("color: #ddaa55; font-size: 11px;")
        info_layout.addWidget(self._filename_label)

        info_layout.addWidget(_hline())

        self._date_hdr = QLabel()
        self._date_hdr.setWordWrap(True)
        self._date_hdr.setMinimumWidth(0)
        self._date_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._date_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._date_hdr)

        self._photo_date_edit = QLineEdit()
        self._photo_date_edit.returnPressed.connect(self._save_photo_date)
        self._photo_date_edit.editingFinished.connect(self._save_photo_date)
        info_layout.addWidget(self._photo_date_edit)

        # EXIF suggestion section — header on its own line, value + button below
        exif_row = QWidget()
        exif_row_layout = QVBoxLayout(exif_row)
        exif_row_layout.setContentsMargins(0, 0, 0, 0)
        exif_row_layout.setSpacing(2)
        self._exif_suggestion_hdr = QLabel()
        self._exif_suggestion_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        exif_row_layout.addWidget(self._exif_suggestion_hdr)
        exif_val_row = QHBoxLayout()
        exif_val_row.setContentsMargins(0, 0, 0, 0)
        exif_val_row.setSpacing(4)
        self._exif_suggestion_label = QLabel("")
        self._exif_suggestion_label.setWordWrap(True)
        self._exif_suggestion_label.setMinimumWidth(0)
        self._exif_suggestion_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._exif_suggestion_label.setStyleSheet("color: #aaddaa; font-size: 11px;")
        exif_val_row.addWidget(self._exif_suggestion_label, 1)
        self._exif_accept_btn = QPushButton()
        self._exif_accept_btn.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._exif_accept_btn.clicked.connect(self._accept_exif_suggestion)
        exif_val_row.addWidget(self._exif_accept_btn, 0)
        exif_row_layout.addLayout(exif_val_row)
        self._exif_row_widget = exif_row
        self._exif_row_widget.setVisible(False)
        info_layout.addWidget(self._exif_row_widget)

        self._update_exif_date_btn = QPushButton()
        self._update_exif_date_btn.clicked.connect(self._update_exif_date)
        info_layout.addWidget(self._update_exif_date_btn)

        info_layout.addWidget(_hline())

        self._place_hdr = QLabel()
        self._place_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._place_hdr)

        self._place_status_label = QLabel("")
        self._place_status_label.setWordWrap(True)
        self._place_status_label.setMinimumWidth(0)
        self._place_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._place_status_label)

        self._place_search = PlaceSearchSelect()
        self._place_search.place_selected.connect(self._on_place_selected)
        self._place_search.place_double_clicked.connect(self._on_place_selected)
        # Enter on a non-matching query creates the place (same as the button).
        self._place_search.create_requested.connect(
            lambda _name: self._create_or_assign_place_by_text()
        )
        info_layout.addWidget(self._place_search)

        place_btns = QHBoxLayout()
        place_btns.setContentsMargins(0, 0, 0, 0)
        place_btns.setSpacing(4)
        self._place_assign_btn = QPushButton()
        self._place_assign_btn.clicked.connect(self._assign_existing_place)
        place_btns.addWidget(self._place_assign_btn)
        self._place_create_btn = QPushButton()
        self._place_create_btn.clicked.connect(self._create_or_assign_place_by_text)
        place_btns.addWidget(self._place_create_btn)
        info_layout.addLayout(place_btns)

        self._place_rename_edit = QLineEdit()
        self._place_rename_edit.returnPressed.connect(self._rename_current_place)
        info_layout.addWidget(self._place_rename_edit)

        self._place_rename_btn = QPushButton()
        self._place_rename_btn.clicked.connect(self._rename_current_place)
        info_layout.addWidget(self._place_rename_btn)

        self._place_coords_label = QLabel()
        self._place_coords_label.setStyleSheet("color: #888; font-size: 10px;")
        self._place_coords_label.setWordWrap(True)
        info_layout.addWidget(self._place_coords_label)

        info_layout.addWidget(_hline())

        # ── Image-level GPS coordinates ───────────────────────────────
        self._gps_hdr = QLabel()
        self._gps_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(self._gps_hdr)

        self._gps_source_label = QLabel()
        self._gps_source_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        info_layout.addWidget(self._gps_source_label)

        self._gps_edit = QLineEdit()
        self._gps_edit.returnPressed.connect(self._save_image_coords)
        info_layout.addWidget(self._gps_edit)

        gps_btns = QHBoxLayout()
        gps_btns.setContentsMargins(0, 0, 0, 0)
        gps_btns.setSpacing(4)
        self._gps_save_btn = QPushButton()
        self._gps_save_btn.clicked.connect(self._save_image_coords)
        gps_btns.addWidget(self._gps_save_btn)
        self._gps_clear_btn = QPushButton()
        self._gps_clear_btn.clicked.connect(self._clear_image_coords)
        gps_btns.addWidget(self._gps_clear_btn)
        info_layout.addLayout(gps_btns)

        self._gps_write_place_btn = QPushButton()
        self._gps_write_place_btn.clicked.connect(self._write_place_coords_to_exif)
        self._gps_write_place_btn.setVisible(False)
        info_layout.addWidget(self._gps_write_place_btn)

        info_layout.addWidget(_hline())

        # ── Image note ────────────────────────────────────────────────
        self._note_hdr = QLabel()
        self._note_hdr.setStyleSheet("font-weight: bold; color: #888; font-size: 11px;")
        info_layout.addWidget(self._note_hdr)

        self._note_edit = QTextEdit()
        self._note_edit.setFixedHeight(72)
        self._note_edit.setAcceptRichText(False)
        info_layout.addWidget(self._note_edit)

        self._note_save_timer = QTimer(self)
        self._note_save_timer.setSingleShot(True)
        self._note_save_timer.setInterval(600)
        self._note_save_timer.timeout.connect(self._save_note)
        self._note_edit.textChanged.connect(self._note_save_timer.start)

        info_layout.addWidget(_hline())

        self._face_hdr = QLabel()
        self._face_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._face_hdr)

        self._face_status_label = QLabel()
        self._face_status_label.setWordWrap(True)
        self._face_status_label.setMinimumWidth(0)
        self._face_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._face_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_layout.addWidget(self._face_status_label)

        self._person_name_label = QLabel("")
        self._person_name_label.setWordWrap(True)
        self._person_name_label.setMinimumWidth(0)
        self._person_name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
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
        self._assign_hdr.setMinimumWidth(0)
        self._assign_hdr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._assign_hdr.setStyleSheet(
            "font-weight: bold; color: #888; font-size: 11px;"
        )
        info_layout.addWidget(self._assign_hdr)

        self._person_search = PersonSearchSelect()
        self._person_search.person_selected.connect(self._on_person_search_selected)
        self._person_search.person_double_clicked.connect(self._on_assign_existing)
        info_layout.addWidget(self._person_search)

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

        self._info_scroll = QScrollArea()
        self._info_scroll.setWidget(info_content)
        self._info_scroll.setWidgetResizable(True)
        self._info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._info_scroll.setFrameShape(QFrame.NoFrame)
        self._info_scroll.setMinimumWidth(160)
        self._info_scroll.setMaximumWidth(340)
        preview_splitter.addWidget(self._info_scroll)

        preview_splitter.setStretchFactor(0, 3)
        preview_splitter.setStretchFactor(1, 1)

        main_splitter.addWidget(preview_splitter)

        main_splitter.setSizes([280, 880])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 2)

        root.addWidget(main_splitter, stretch=1)
        self.retranslate()

    def _setup_shortcuts(self) -> None:
        from app.services.shortcut_service import get_shortcut_service
        get_shortcut_service().register("general.fullscreen", self._toggle_fullscreen)

    # ──────────────────────────────────────────────────────────────────
    # Public API  (consumed by MainWindow)
    # ──────────────────────────────────────────────────────────────────

    def retranslate(self) -> None:
        self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
        self._tree_search_bar.set_placeholder(t("ibp_universal_placeholder"))
        self._tree_search_bar.retranslate()
        self._toggle_left_btn.setToolTip(t("ib3_toggle_left_tip"))
        self._toggle_right_btn.setToolTip(t("ib3_toggle_right_tip"))
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
        self._place_hdr.setText(t("ibp_place_hdr"))
        self._place_assign_btn.setText(t("ibp_place_assign_btn"))
        self._place_create_btn.setText(t("ibp_place_create_btn"))
        self._place_rename_edit.setPlaceholderText(t("ibp_place_rename_placeholder"))
        self._place_rename_btn.setText(t("ibp_place_rename_btn"))
        self._place_search.retranslate()
        self._gps_hdr.setText(t("ibp_gps_hdr"))
        self._gps_edit.setPlaceholderText(t("ibp_gps_placeholder"))
        self._gps_save_btn.setText(t("ibp_gps_save_btn"))
        self._gps_clear_btn.setText(t("ibp_gps_clear_btn"))
        self._gps_write_place_btn.setText(t("ibp_gps_write_exif_place"))
        self._gps_write_place_btn.setToolTip(t("ibp_gps_write_exif_place_tip"))
        self._update_exif_date_btn.setText(t("ibp_update_exif_date_btn"))
        self._update_exif_date_btn.setToolTip(t("ibp_update_exif_date_tip"))
        self._note_hdr.setText(t("ibp_note_hdr"))
        self._note_edit.setPlaceholderText(t("ibp_note_placeholder"))
        self._note_edit.setToolTip(t("ibp_note_tooltip"))
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
        self._deol_lbl.setText(t("ibp_deol_pair_lbl"))
        self._btn_view_bw.setText(t("ibp_view_original_bw"))
        self._btn_view_color.setText(t("ibp_view_colorized"))

    def refresh(self) -> None:
        """Reload folder list from DB; keep the selected folder expanded if still present."""
        log.debug("ImageBrowserPanel.refresh() — reloading folder tree")
        self._reload_persons_combo()
        self._reload_places()
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

    # ──────────────────────────────────────────────────────────────────
    # Drive mode
    # ──────────────────────────────────────────────────────────────────

    def set_drive_mode(self, client, mirror_dir: Path, project_name: str) -> None:
        """Activate Drive-aware lazy loading.

        Called by the main window after a Drive project is opened.  The
        panel uses *client* to download images on demand and *mirror_dir*
        to write them.

        Args:
            client:       Authenticated :class:`~app.gdrive.protocols.GDriveClient`.
            mirror_dir:   Persistent local directory for downloaded images.
            project_name: Human-readable Drive folder name shown in the tree.
        """
        self._drive_client = client
        self._drive_mirror_dir = mirror_dir
        self._drive_project_name = project_name
        self._prefetch_set.clear()
        log.info("ImageBrowser: Drive mode ON — project=%r", project_name)

    def clear_drive_mode(self) -> None:
        """Deactivate Drive mode; revert to local-only file access."""
        self._drive_client = None
        self._drive_mirror_dir = None
        self._drive_project_name = ""
        self._prefetch_set.clear()
        self._drive_pending_image_id = None
        log.info("ImageBrowser: Drive mode OFF")

    def open_image_by_id(self, image_id: int) -> None:
        """Open an image in the existing viewer, expanding its folder when possible."""
        self._current_tree_item = None
        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                return
            folder_path = str(Path(img.file_path).parent)

        for idx in range(self._file_tree.topLevelItemCount()):
            folder = self._file_tree.topLevelItem(idx)
            if folder.data(0, _ROLE_PAYLOAD) != folder_path:
                continue
            if not folder.isExpanded():
                folder.setExpanded(True)
            else:
                self._load_folder_children(folder)
            cache_key = f"thumb_{image_id}"
            item = self._tree_items.get(cache_key)
            if item is not None:
                self._select_tree_item(item)
                return
            break

        self._load_image(image_id)
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(False)

    # ──────────────────────────────────────────────────────────────────
    # Deoldified pairing
    # ──────────────────────────────────────────────────────────────────

    def _setup_deoldified_pair(self, image_id: int, image_path: str) -> None:
        """Detect deoldified pairing and configure the toggle bar."""
        from app.app_settings import app_qsettings
        settings = app_qsettings()
        if not settings.value("deoldified/auto_pair", False, type=bool):
            return

        from app.services.deoldified_pairing_service import (
            is_deoldified_path,
            DeoldifiedPairingService,
        )
        from app.services.image_library_service import resolve_image_path

        if is_deoldified_path(image_path):
            # Current image IS the colorized one → find its original
            with session_scope() as session:
                img = session.get(Image, image_id)
                if img is None:
                    return
                svc = DeoldifiedPairingService(session)
                orig = svc.find_original_for_deoldified(img)
                if orig is None:
                    return
                self._deol_pair_orig_id = orig.id

            self._deol_pair_color_path = image_path
            self._deol_viewing_color = True
            self._btn_view_bw.setChecked(False)
            self._btn_view_color.setChecked(True)
            self._deoldified_bar.setVisible(True)
            log.debug(
                "Deoldified pair: color=%s → orig_id=%d",
                image_path, self._deol_pair_orig_id,
            )
        else:
            # Current image is the original → look for a colorized variant
            with session_scope() as session:
                img = session.get(Image, image_id)
                if img is None:
                    return
                svc = DeoldifiedPairingService(session)
                color_img = svc.find_deoldified_for_original(img)
                if color_img is None:
                    return
                color_resolved = resolve_image_path(color_img)
                self._deol_pair_color_path = (
                    str(color_resolved) if color_resolved else color_img.file_path
                )

            self._deol_viewing_color = False
            self._btn_view_bw.setChecked(True)
            self._btn_view_color.setChecked(False)
            self._deoldified_bar.setVisible(True)
            log.debug(
                "Deoldified pair: orig=%s → color=%s",
                image_path, self._deol_pair_color_path,
            )

    def _on_deol_view_toggle(self, show_colorized: bool) -> None:
        """Switch between original B&W and colorized pixel view."""
        if show_colorized == self._deol_viewing_color:
            return

        from app.utils.image_utils import load_image_bgr
        from app.services.image_library_service import resolve_image_path

        if show_colorized:
            path = self._deol_pair_color_path
        elif self._deol_pair_orig_id is not None:
            # Current image in the tree is the deoldified one; load original pixels
            with session_scope() as session:
                orig = session.get(Image, self._deol_pair_orig_id)
                if orig is None:
                    return
                resolved = resolve_image_path(orig)
                path = str(resolved) if resolved else orig.file_path
        else:
            path = self._current_path  # current IS the original

        img_bgr = load_image_bgr(path)
        if img_bgr is None:
            log.warning("Cannot load image for deoldified toggle: %s", path)
            return

        self._deol_viewing_color = show_colorized
        self._btn_view_bw.setChecked(not show_colorized)
        self._btn_view_color.setChecked(show_colorized)
        self._orig_img_bgr = img_bgr
        self._reset_zoom()
        self._redraw_faces()

    def _fetch_face_data(self) -> None:
        """Reload face list from DB without triggering any UI update."""
        # When pairing is active and current is deoldified, reload from the original
        source_id = self._deol_pair_orig_id or self._current_image_id
        if source_id is None:
            return
        with session_scope() as session:
            if self._config is not None:
                img = session.get(Image, source_id)
                if img is not None:
                    from app.services.face_crop_service import ensure_unique_face_crops
                    ensure_unique_face_crops(
                        session,
                        [f for f in img.faces if not f.is_excluded],
                        self._config.crops_dir_resolved,
                        self._config.scan.thumbnail_size,
                    )
            svc = ImageBrowserService(session)
            self._face_data = svc.get_image_faces(source_id)

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

        # In Drive mode show "☁ <project name>" instead of the ugly mirror path.
        display = summary.display_name
        if (
            self._drive_client is not None
            and self._drive_mirror_dir is not None
            and summary.folder_path == str(self._drive_mirror_dir)
        ):
            display = f"☁ {self._drive_project_name}" if self._drive_project_name else "☁ Drive"

        label = f"{display}  ({summary.total_images})"
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
            summaries = ImageBrowserService(session).list_images(
                folder_path,
                place_id=self._place_filter_id,
            )

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
            child.setData(0, _ROLE_FILE_PATH, summary.file_path)
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
            elif self._drive_client is not None:
                # Mirror file missing — download from Drive and generate thumb.
                from app.workers.drive_image_worker import DriveThumbRunnable
                dw = DriveThumbRunnable(
                    drive_client=self._drive_client,
                    image_id=summary.image_id,
                    local_path=summary.file_path,
                    cache_key=summary.thumbnail_cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                dw.signals.ready.connect(self._on_thumbnail_ready)
                dw.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(dw)
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
        image_id = item.data(0, _ROLE_PAYLOAD)
        self._load_image(image_id)
        self._update_nav_buttons()
        # Smart prefetch: queue downloads for adjacent images in Drive mode.
        if self._drive_client is not None:
            self._prefetch_adjacent(item)

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

    # ── Overlay control handlers ──────────────────────────────────────────

    def _on_ov_bbox_check(self, checked: bool) -> None:
        self._show_bboxes = checked
        self._bbox_slider.setEnabled(checked)
        if checked:
            self._bbox_opacity = self._prev_bbox_opacity
        self._redraw_faces()

    def _on_ov_bbox_slider(self, value: int) -> None:
        pct = value / 100.0
        self._bbox_opacity = pct
        if self._show_bboxes:
            self._prev_bbox_opacity = pct
        self._bbox_pct_label.setText(f"{value}%")
        self._redraw_faces()

    def _on_ov_label_check(self, checked: bool) -> None:
        self._show_labels = checked
        self._label_slider.setEnabled(checked)
        if checked:
            self._label_opacity = self._prev_label_opacity
        self._redraw_faces()

    def _on_ov_label_slider(self, value: int) -> None:
        pct = value / 100.0
        self._label_opacity = pct
        if self._show_labels:
            self._prev_label_opacity = pct
        self._label_pct_label.setText(f"{value}%")
        self._redraw_faces()

    # ── Navigation ───────────────────────────────────────────────────────

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
        icon = QIcon(pixmap)
        tree_item = self._tree_items.get(cache_key)
        if tree_item is not None:
            tree_item.setIcon(0, icon)
        search_item = self._search_result_items.get(cache_key)
        if search_item is not None:
            search_item.setIcon(icon)

    @Slot(str)
    def _on_thumbnail_failed(self, cache_key: str) -> None:
        pass  # placeholder icon stays

    # ──────────────────────────────────────────────────────────────────
    # Drive on-demand download callbacks
    # ──────────────────────────────────────────────────────────────────

    @Slot(int, str)
    def _on_drive_fetch_ready(self, image_id: int, local_path: str) -> None:
        """Called when a DriveFetchRunnable completes.

        If this fetch was for the currently requested image, display it.
        Also update the thumbnail in the tree.
        """
        # Remove from prefetch-in-progress set.
        self._prefetch_set.discard(image_id)

        # Refresh the thumbnail in the tree if we have a cache key.
        cache_key = f"thumb_{image_id}"
        if cache_key not in self._thumb_cache:
            tree_item = self._tree_items.get(cache_key)
            if tree_item is not None:
                from app.workers.drive_image_worker import DriveThumbRunnable
                from app.workers.thumbnail_worker import ThumbnailRunnable
                # File is now in mirror — generate thumbnail normally.
                worker = ThumbnailRunnable(
                    file_path=local_path,
                    cache_key=cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                QThreadPool.globalInstance().start(worker)

        # If this is the image the user is waiting to see, load it now.
        if image_id == self._drive_pending_image_id:
            self._drive_pending_image_id = None
            self._load_image(image_id)

    @Slot(int, str)
    def _on_drive_fetch_failed(self, image_id: int, error: str) -> None:
        self._prefetch_set.discard(image_id)
        if image_id == self._drive_pending_image_id:
            self._drive_pending_image_id = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(
                t("ibp_drive_download_failed", error=error)
            )

    # ──────────────────────────────────────────────────────────────────
    # Drive smart prefetch
    # ──────────────────────────────────────────────────────────────────

    def _prefetch_adjacent(self, current_item: QTreeWidgetItem) -> None:
        """Queue Drive downloads for the next *_DRIVE_PREFETCH_AHEAD* siblings.

        Skips images whose mirror file already exists and ones already
        in-flight.
        """
        if self._drive_client is None:
            return
        parent = current_item.parent()
        if parent is None:
            return

        idx = parent.indexOfChild(current_item)
        pool = QThreadPool.globalInstance()

        for delta in range(1, _DRIVE_PREFETCH_AHEAD + 1):
            si = idx + delta
            if si >= parent.childCount():
                break
            sibling = parent.child(si)
            if sibling.data(0, _ROLE_TYPE) != "image":
                continue
            img_id: int = sibling.data(0, _ROLE_PAYLOAD)
            file_path: str = sibling.data(0, _ROLE_FILE_PATH) or ""
            if not file_path:
                continue
            if Path(file_path).exists():
                continue  # already in mirror
            if img_id in self._prefetch_set:
                continue  # already downloading

            self._prefetch_set.add(img_id)
            from app.workers.drive_image_worker import DriveFetchRunnable
            fetcher = DriveFetchRunnable(
                drive_client=self._drive_client,
                image_id=img_id,
                local_path=file_path,
                silent=True,
            )
            fetcher.signals.ready.connect(self._on_drive_fetch_ready)
            pool.start(fetcher)
            log.debug("Drive prefetch queued: image_id=%d", img_id)

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
        # Exit interactive edit mode without restoring (different image)
        if self._interactive_edit_face_id is not None:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._image_label.cancel_interactive_edit()
            self._draw_hint.setVisible(False)

        # Reset deoldified pairing state
        self._deol_pair_orig_id = None
        self._deol_pair_color_path = ""
        self._deol_viewing_color = False
        self._deoldified_bar.setVisible(False)

        with session_scope() as session:
            img = session.get(Image, image_id)
            if img is None:
                log.warning("Image %d not found in DB", image_id)
                return
            for f in img.faces:
                _ = f.person
            self._current_image_id = image_id
            self._current_path = img.file_path
            self.image_displayed.emit(image_id, img.file_path)
            self._detection_done = img.detection_done
            photo_date = img.photo_date or ""
            note = img.note or ""
            current_place_id = img.place_id
            current_place_name = img.place.name if img.place else ""
            current_place_is_anonymous = bool(img.place and img.place.is_anonymous)
            current_place_source = img.place.source if img.place else None
            current_place_lat = img.place.latitude if img.place else None
            current_place_lon = img.place.longitude if img.place else None
            current_image_lat = img.image_latitude
            current_image_lon = img.image_longitude
            current_exif_lat = img.exif_latitude
            current_exif_lon = img.exif_longitude
            # Auto-fill date from filename when not yet set
            if not photo_date:
                from_name = _parse_date_from_filename(img.file_path)
                if from_name:
                    img.photo_date = from_name
                    photo_date = from_name
                    log.debug(
                        "photo_date auto-set from filename for image %d: %r",
                        image_id, from_name,
                    )
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

        # Detect deoldified pair and override state if pairing is enabled
        self._setup_deoldified_pair(image_id, self._current_path)

        # When current image is deoldified: load face data from the original
        if self._deol_pair_orig_id is not None:
            with session_scope() as session:
                orig = session.get(Image, self._deol_pair_orig_id)
                if orig is not None:
                    for f in orig.faces:
                        _ = f.person
                    self._face_data = [
                        (
                            f.id,
                            f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                            f.person.name if f.person else None,
                            f.person_id,
                        )
                        for f in orig.faces
                        if not f.is_excluded
                    ]
                    if not photo_date:
                        photo_date = orig.photo_date or ""

        # Re-read through the service so duplicate/missing crop paths are
        # repaired before any preview state is rendered.
        self._fetch_face_data()

        self._photo_date_edit.blockSignals(True)
        self._photo_date_edit.setText(photo_date)
        self._photo_date_edit.blockSignals(False)
        self._note_edit.blockSignals(True)
        self._note_edit.setPlainText(note)
        self._note_edit.blockSignals(False)
        self._update_place_panel(
            current_place_id,
            current_place_name,
            current_place_is_anonymous,
            current_place_source,
            current_place_lat,
            current_place_lon,
        )
        self._load_image_coords(
            current_image_lat,
            current_image_lon,
            current_exif_lat,
            current_exif_lon,
            current_place_lat,
            current_place_lon,
            current_place_id,
        )

        p = Path(self._current_path)
        self._folder_label.setText(str(p.parent))
        self._filename_label.setText(p.name)
        self._load_exif_suggestion(self._current_path)

        from app.utils.image_utils import load_image_bgr
        display_path = (
            self._deol_pair_color_path
            if (self._deol_viewing_color and self._deol_pair_color_path)
            else self._current_path
        )

        # Drive mode: if mirror file is missing, download it first.
        if not Path(display_path).exists() and self._drive_client is not None:
            self._drive_pending_image_id = image_id
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(t("ibp_drive_downloading"))
            from app.workers.drive_image_worker import DriveFetchRunnable
            fetcher = DriveFetchRunnable(
                drive_client=self._drive_client,
                image_id=image_id,
                local_path=display_path,
                silent=False,
            )
            fetcher.signals.ready.connect(self._on_drive_fetch_ready)
            fetcher.signals.failed.connect(self._on_drive_fetch_failed)
            QThreadPool.globalInstance().start(fetcher)
            return

        img_bgr = load_image_bgr(display_path)
        if img_bgr is None:
            self._orig_img_bgr = None
            self._full_pixmap = None
            self._image_label.set_source_pixmap(None)
            self._image_label.setText(t("ibp_load_error", path=display_path))
            self._clear_face_panel()
            log.warning("Cannot load image file: %s", display_path)
            return

        self._orig_img_bgr = img_bgr
        self._reset_zoom()
        self._redraw_faces()
        self._clear_face_panel()
        self._reload_persons_combo()
        self._reload_places()

    def _clear_current_image(self) -> None:
        """Reset preview when switching folders."""
        self._current_image_id = None
        self._current_path = ""
        self._face_data = []
        self._selected_face_id = None
        self._editing_face_id = None
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._orig_img_bgr = None
        self._full_pixmap = None
        self._deol_pair_orig_id = None
        self._deol_pair_color_path = ""
        self._deol_viewing_color = False
        self._deoldified_bar.setVisible(False)
        self._image_label.set_source_pixmap(None)
        self._image_label.setText(t("ib3_select_image_hint"))
        self._folder_label.setText("")
        self._filename_label.setText("")
        self._note_edit.blockSignals(True)
        self._note_edit.clear()
        self._note_edit.blockSignals(False)
        self._exif_row_widget.setVisible(False)
        self._update_place_panel(None, "", False, None)
        self._load_image_coords(None, None, None, None, None, None, None)
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
            log.debug(
                "Face selection: click at image=(%d,%d) — no face hit, clearing selection",
                orig_x, orig_y,
            )
            self._hide_inline_editor()
            self._selected_face_id = None
            self._redraw_faces()
            self._clear_face_panel()
            return

        log.debug(
            "Face selection: click at image=(%d,%d) → face_id=%d "
            "(prev_selected=%s image_id=%s)",
            orig_x, orig_y, clicked_id, self._selected_face_id, self._current_image_id,
        )
        if self._selected_face_id != clicked_id:
            self._selected_face_id = clicked_id
            self._redraw_faces()
            self._show_face_info(clicked_id)

        entry = next((f for f in self._face_data if f[0] == clicked_id), None)
        person_name = entry[5] if entry else None
        if not person_name:
            self._show_inline_editor(clicked_id)

    def _on_image_right_clicked(self, lx: int, ly: int) -> None:
        orig_x, orig_y = self._label_to_image(lx, ly)
        face_id: Optional[int] = None
        if orig_x >= 0 and orig_y >= 0:
            face_id = self._hit_test(orig_x, orig_y)

        log.debug(
            "Right-click: label=(%d,%d) image=(%d,%d) image_id=%s faces_loaded=%d hit_face=%s",
            lx, ly, orig_x, orig_y,
            self._current_image_id,
            len(self._face_data),
            face_id,
        )

        menu = QMenu(self)
        assign_action: Optional[object] = None
        edit_action: Optional[object] = None
        delete_action: Optional[object] = None

        if face_id is not None:
            if self._selected_face_id != face_id:
                self._selected_face_id = face_id
                self._redraw_faces()
                self._show_face_info(face_id)
            self._hide_inline_editor()

            entry = next((f for f in self._face_data if f[0] == face_id), None)
            person_name = entry[5] if entry else None

            title_text = person_name if person_name else t("ibp_ctx_unknown_face")
            title = menu.addAction(f"[ {title_text} ]")
            title.setEnabled(False)
            menu.addSeparator()
            assign_action = menu.addAction(t("ibp_ctx_assign_person"))
            menu.addSeparator()
            edit_action = menu.addAction(t("ibp_ctx_edit_frame"))
            menu.addSeparator()
            delete_action = menu.addAction(t("ibp_ctx_delete"))
            menu.addSeparator()

        manual_mark_action = menu.addAction(t("ibp_ctx_manual_mark"))

        global_pos = self._image_label.mapToGlobal(QPoint(lx, ly))
        chosen = menu.exec(global_pos)
        log.debug(
            "Context menu result: chosen=%s face_id=%s",
            chosen.text() if chosen else None,
            face_id,
        )

        if chosen is assign_action and face_id is not None:
            self._show_inline_editor(face_id)
        elif chosen is edit_action and face_id is not None:
            self._start_interactive_face_edit(face_id)
        elif chosen is delete_action and face_id is not None:
            self._delete_face(face_id)
        elif chosen is manual_mark_action:
            self._draw_mode_btn.setChecked(True)

    # ──────────────────────────────────────────────────────────────────
    # Draw mode
    # ──────────────────────────────────────────────────────────────────

    def _on_draw_mode_toggled(self, active: bool) -> None:
        if active and self._interactive_edit_face_id is not None:
            # Switching to draw mode cancels interactive edit without restoring
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
        self._image_label.set_draw_mode(active)
        self._draw_hint.setVisible(active)
        if active:
            self._hide_inline_editor()
            self._draw_hint.setText(t("ibp_draw_hint_add"))
        else:
            self._editing_face_id = None
            self._draw_hint.setText(t("ibp_draw_hint_add"))

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
        ix, iy, iw, ih = coords
        log.debug("bbox created in image coords: (%d,%d,%d,%d)", ix, iy, iw, ih)

        if self._current_image_id is None:
            log.debug("pointer up: _current_image_id is None — draw ignored")
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
            # Always save manual faces to the canonical (non-deoldified) image
            face_img_id = self._deol_pair_orig_id or self._current_image_id
            log.debug("face create requested: image_id=%d bbox=(%d,%d,%d,%d)", face_img_id, ix, iy, iw, ih)
            new_face_id = self._save_manual_face(
                face_img_id, ix, iy, iw, ih
            )
            log.debug("bbox added to state: face_id=%s", new_face_id)
            # Reload the full face list (includes the just-saved face),
            # select it and redraw once — draw mode intentionally stays ON
            # so the user can immediately add more faces without re-clicking.
            self._fetch_face_data()
            log.debug("render annotations count: %d", len(self._face_data))
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

    # ──────────────────────────────────────────────────────────────────
    # Interactive bbox editor (handle-based move / resize)
    # ──────────────────────────────────────────────────────────────────

    def _start_interactive_face_edit(self, face_id: int) -> None:
        """Enter interactive handle-drag edit mode for *face_id*."""
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, bx, by, bw, bh, *_ = entry
        self._hide_inline_editor()
        self._draw_mode_btn.setChecked(False)
        self._interactive_edit_face_id = face_id
        self._interactive_edit_orig_bbox = (bx, by, bw, bh)
        self._selected_face_id = face_id

        # Redraw without baking in the editing face's bbox so only handles show it
        self._redraw_faces(exclude_edit_id=face_id)
        self._show_face_info(face_id)

        if self._orig_img_bgr is not None:
            ih, iw = self._orig_img_bgr.shape[:2]
        else:
            iw = self._full_pixmap.width() if self._full_pixmap else 1
            ih = self._full_pixmap.height() if self._full_pixmap else 1

        self._image_label.set_interactive_edit((bx, by, bw, bh), iw, ih)
        self._image_label.setFocus()   # ensure Esc / Enter key events reach the label
        self._draw_hint.setText(t("ibp_bbox_edit_hint"))
        self._draw_hint.setVisible(True)

    def _on_interactive_bbox_committed(
        self, ix: int, iy: int, iw: int, ih: int
    ) -> None:
        """Called each time the user releases a drag handle with a changed bbox."""
        face_id = self._interactive_edit_face_id
        if face_id is None:
            return
        log.debug(
            "BBox intermediate commit: face_id=%d new_bbox=(%d,%d,%d,%d)",
            face_id, ix, iy, iw, ih,
        )

        # Find old bbox for undo record
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        old_bbox: Tuple[int, int, int, int]
        if entry is not None:
            old_bbox = (entry[1], entry[2], entry[3], entry[4])
        else:
            old_bbox = (ix, iy, iw, ih)

        new_bbox = (ix, iy, iw, ih)
        op_type = getattr(self._image_label, "_last_op_type", "resize")

        # Save to DB
        self._update_face_bbox(face_id, ix, iy, iw, ih)

        # Push to undo stack, clear redo
        if old_bbox != new_bbox:
            self._undo_stack.append(
                _BboxEdit(face_id=face_id, old_bbox=old_bbox, new_bbox=new_bbox, op_type=op_type)
            )
            self._redo_stack.clear()

        # Reload face data and update the handle overlay with the latest bbox
        self._fetch_face_data()
        self._selected_face_id = face_id

        if self._orig_img_bgr is not None:
            img_h, img_w = self._orig_img_bgr.shape[:2]
        else:
            img_w = self._full_pixmap.width() if self._full_pixmap else 1
            img_h = self._full_pixmap.height() if self._full_pixmap else 1

        self._redraw_faces(exclude_edit_id=face_id)
        self._image_label.set_interactive_edit((ix, iy, iw, ih), img_w, img_h)
        self._show_face_info(face_id)

    def _on_interactive_bbox_finalized(
        self, ix: int, iy: int, iw: int, ih: int
    ) -> None:
        """Called when the user clicks outside the box or presses Enter — save and exit."""
        face_id = self._interactive_edit_face_id
        if face_id is None:
            return
        log.info(
            "BBox finalized: face_id=%d final_bbox=(%d,%d,%d,%d)",
            face_id, ix, iy, iw, ih,
        )

        # Grab old bbox before clearing state
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        old_bbox: Tuple[int, int, int, int]
        if entry is not None:
            old_bbox = (entry[1], entry[2], entry[3], entry[4])
        else:
            old_bbox = (ix, iy, iw, ih)

        new_bbox = (ix, iy, iw, ih)
        op_type = getattr(self._image_label, "_last_op_type", "resize")

        # Exit edit state first
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._draw_hint.setVisible(False)

        # Save to DB
        self._update_face_bbox(face_id, ix, iy, iw, ih)

        # Push to undo stack only if bbox actually changed
        if old_bbox != new_bbox:
            self._undo_stack.append(
                _BboxEdit(face_id=face_id, old_bbox=old_bbox, new_bbox=new_bbox, op_type=op_type)
            )
            self._redo_stack.clear()

        # Normal redraw (all faces baked in, no handle overlay)
        self._fetch_face_data()
        self._selected_face_id = face_id
        self._redraw_faces()
        self._show_face_info(face_id)

    def _on_interactive_bbox_cancelled(self) -> None:
        """Called when the user presses Esc or otherwise cancels the edit."""
        face_id = self._interactive_edit_face_id
        orig_bbox = self._interactive_edit_orig_bbox
        self._interactive_edit_face_id = None
        self._interactive_edit_orig_bbox = None
        self._draw_hint.setVisible(False)

        if face_id is not None and orig_bbox is not None:
            # Restore original bbox
            self._update_face_bbox(face_id, *orig_bbox)
            # Remove any undo entries added during this edit session for this face
            # (since the cancel should restore everything to the start-of-session state)
            self._undo_stack = [e for e in self._undo_stack if e.face_id != face_id]
            self._redo_stack = [e for e in self._redo_stack if e.face_id != face_id]
            self._fetch_face_data()
            self._selected_face_id = face_id
            self._redraw_faces()
            self._show_face_info(face_id)

    def _undo_bbox_edit(self) -> None:
        """Undo the last bbox modification (Ctrl+Z)."""
        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        self._redo_stack.append(entry)
        self._apply_bbox_edit(entry.face_id, entry.old_bbox)

    def _redo_bbox_edit(self) -> None:
        """Redo the last undone bbox modification (Ctrl+Y)."""
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        self._undo_stack.append(entry)
        self._apply_bbox_edit(entry.face_id, entry.new_bbox)

    def _apply_bbox_edit(
        self, face_id: int, bbox: Tuple[int, int, int, int]
    ) -> None:
        """Apply *bbox* to *face_id* in DB and refresh the view."""
        self._update_face_bbox(face_id, *bbox)
        self._fetch_face_data()
        self._selected_face_id = face_id

        # If we're in interactive edit mode for this face, update the handle overlay
        if self._interactive_edit_face_id == face_id and self._orig_img_bgr is not None:
            img_h, img_w = self._orig_img_bgr.shape[:2]
            self._redraw_faces(exclude_edit_id=face_id)
            self._image_label.set_interactive_edit(bbox, img_w, img_h)
        else:
            self._redraw_faces()
        self._show_face_info(face_id)

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
        if self._interactive_edit_face_id == face_id:
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._image_label.cancel_interactive_edit()
            self._draw_hint.setVisible(False)
        # Remove any undo/redo entries for the deleted face
        self._undo_stack = [e for e in self._undo_stack if e.face_id != face_id]
        self._redo_stack = [e for e in self._redo_stack if e.face_id != face_id]
        self._reload_current_face_data()

    def _update_face_bbox(
        self, face_id: int, x: int, y: int, w: int, h: int
    ) -> None:
        log.info(
            "BBox save: face_id=%d bbox=(%d,%d,%d,%d) "
            "caller_edit_face=%s selected=%s",
            face_id, x, y, w, h,
            self._interactive_edit_face_id, self._selected_face_id,
        )
        old_crop_path: Optional[str] = None
        new_crop_path: Optional[str] = None
        with session_scope() as session:
            from app.detectors.base import Detection
            from app.services.face_crop_service import (
                crop_path_is_shared,
                face_debug_state,
                save_crop_for_face,
            )
            from app.utils.image_utils import load_image_bgr

            face = session.get(Face, face_id)
            if face is None or face.image is None:
                return
            img_bgr = load_image_bgr(face.image.file_path)
            if img_bgr is None:
                return
            detection = Detection(x=x, y=y, w=w, h=h, confidence=1.0).clamp(
                img_bgr.shape[1], img_bgr.shape[0]
            )
            face.bbox_x = detection.x
            face.bbox_y = detection.y
            face.bbox_w = detection.w
            face.bbox_h = detection.h
            if face.person:
                _ = face.person.name
            old_crop_path = face.crop_path
            if crop_path_is_shared(session, face):
                log.warning(
                    "Shared crop path before image-browser bbox update: %s",
                    face_debug_state(face, old_crop_path),
                )
            if self._config is not None:
                crops_dir = self._config.crops_dir_resolved
                crops_dir.mkdir(parents=True, exist_ok=True)
                crop_path = save_crop_for_face(
                    face,
                    crops_dir=crops_dir,
                    thumbnail_size=self._config.scan.thumbnail_size,
                    img_bgr=img_bgr,
                )
                if crop_path is not None:
                    new_crop_path = str(crop_path)

        if old_crop_path or new_crop_path:
            from PySide6.QtGui import QPixmapCache
            if old_crop_path:
                QPixmapCache.remove(old_crop_path)
            if new_crop_path:
                QPixmapCache.remove(new_crop_path)
        log.info(
            "Face %d bbox updated to (%d,%d,%d,%d) crop=%s",
            face_id, x, y, w, h, new_crop_path,
        )

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

    def _label_to_image_clamped(self, lx: int, ly: int) -> Tuple[int, int]:
        """Like _label_to_image but clamps out-of-bounds to the image edge."""
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
        rx = max(0.0, min(lx - ox, disp_w - 1))
        ry = max(0.0, min(ly - oy, disp_h - 1))
        return int(rx / eff), int(ry / eff)

    def _label_rect_to_image(
        self, rect: QRect
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1 = self._label_to_image(rect.left(), rect.top())
        if x1 < 0 or y1 < 0:
            return None
        # Use clamped conversion for bottom-right so rects that extend into
        # the margin area are clipped to the image edge instead of becoming 1×1.
        x2, y2 = self._label_to_image_clamped(rect.right(), rect.bottom())
        if x2 < 0 or y2 < 0:
            return None
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

    def _redraw_faces(
        self, *, exclude_edit_id: Optional[int] = None
    ) -> None:
        """Render the annotated image and push it to the image label.

        When *exclude_edit_id* is given, that face's bounding box is omitted
        from the baked-in PIL rendering so the handle overlay is the only
        visual representation of the editing face.
        """
        if self._orig_img_bgr is None:
            return
        log.debug(
            "Overlay re-render: image_id=%s selected_face_id=%s "
            "faces=%s exclude_edit_id=%s imode=%s imode_face=%s",
            self._current_image_id,
            self._selected_face_id,
            [(fd[0], fd[5]) for fd in self._face_data],
            exclude_edit_id,
            self._image_label._imode,
            self._interactive_edit_face_id,
        )
        render_faces = (
            [f for f in self._face_data if f[0] != exclude_edit_id]
            if exclude_edit_id is not None
            else self._face_data
        )
        annotated = _draw_faces(
            self._orig_img_bgr, render_faces, self._selected_face_id,
            show_bboxes=self._show_bboxes,
            bbox_opacity=self._bbox_opacity if self._show_bboxes else 0.0,
            show_labels=self._show_labels,
            label_opacity=self._label_opacity if self._show_labels else 0.0,
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
            log.debug("_show_inline_editor: face_id=%d not found in face_data", face_id)
            return
        _, bx, by, bw, bh, person_name, person_id = entry

        self._inline_editor.populate(
            person_name,
            person_id,
            self._all_persons,
            priority_ids=list(self._recent_assignment_person_ids),
        )
        self._inline_editor.adjustSize()

        lx_top, ly_top = self._image_to_label(bx, by)
        _, ly_bot = self._image_to_label(bx, by + bh)

        # _image_to_label returns (-1, -1) only when _full_pixmap is None.
        # For off-screen faces (pan/zoom) the values can be negative — clamp
        # rather than bailing out, so the editor still appears after a right-click.
        if self._full_pixmap is None:
            log.debug("_show_inline_editor: no pixmap, aborting")
            return
        lx_top = max(0, lx_top)
        ly_top = max(0, ly_top)
        if ly_bot < 0:
            ly_bot = ly_top + int(bh * min(
                self._image_label.contentsRect().width() / self._full_pixmap.width(),
                self._image_label.contentsRect().height() / self._full_pixmap.height(),
            ) * self._zoom)

        log.debug(
            "_show_inline_editor: face_id=%d person=%r label_top=(%d,%d) label_bot_y=%d",
            face_id, person_name, lx_top, ly_top, ly_bot,
        )

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
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._inline_editor.focus_search)

    def _hide_inline_editor(self) -> None:
        self._inline_editor.hide()
        self._inline_editor_face_id = None

    def _on_inline_assign(self, person_id: int) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        log.info(
            "Face assignment: face_id=%d → person_id=%d (inline editor)",
            face_id, person_id,
        )
        self._inline_editor.reset_search()
        self._hide_inline_editor()
        with session_scope() as session:
            IdentityService(session).reassign_face(face_id, person_id)
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()

    def _on_inline_create(self, name: str) -> None:
        face_id = self._inline_editor_face_id
        if face_id is None:
            return
        self._hide_inline_editor()

        # Safety: exit any stale interactive edit state.  Normally the inline
        # editor can only be opened when _imode is False, but defensive cleanup
        # prevents _imode from leaking into the post-creation state.
        if self._interactive_edit_face_id is not None:
            log.warning(
                "New person (inline): stale interactive-edit detected for "
                "face_id=%d — dismissing silently",
                self._interactive_edit_face_id,
            )
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._draw_hint.setVisible(False)
            self._image_label.dismiss_interactive_edit()

        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            log.info(
                "New person creation: face_id=%d new_person='%s' person_id=%d",
                face_id, name, person_id,
            )
            IdentityService(session).reassign_face(face_id, person_id)
            log.info(
                "Face assignment: face_id=%d → person_id=%d (new person via inline)",
                face_id, person_id,
            )
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()
        self._open_person_info_dialog(person_id)

    # ──────────────────────────────────────────────────────────────────
    # Face info panel
    # ──────────────────────────────────────────────────────────────────

    def _show_face_info(self, face_id: int) -> None:
        entry = next((f for f in self._face_data if f[0] == face_id), None)
        if entry is None:
            return
        _, _, _, _, _, person_name, _ = entry
        self.active_person_changed.emit(person_name)
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
        self.active_person_changed.emit(None)

    # ──────────────────────────────────────────────────────────────────
    # Photo date
    # ──────────────────────────────────────────────────────────────────

    def _save_photo_date(self) -> None:
        # Save to the original image when viewing a deoldified pair
        target_id = self._deol_pair_orig_id or self._current_image_id
        if target_id is None:
            return
        value = self._photo_date_edit.text().strip() or None
        with session_scope() as session:
            img = session.get(Image, target_id)
            if img is None:
                return
            img.photo_date = value
        log.debug("photo_date saved for image %d: %r", target_id, value)

    def _save_note(self) -> None:
        target_id = self._deol_pair_orig_id or self._current_image_id
        if target_id is None:
            return
        value = self._note_edit.toPlainText().strip() or None
        with session_scope() as session:
            img = session.get(Image, target_id)
            if img is None:
                return
            img.note = value
        log.debug("note saved for image %d: %r", target_id, value)

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
    # Place assignment
    # ──────────────────────────────────────────────────────────────────

    def _reload_places(self) -> None:
        with session_scope() as session:
            self._all_places = session.query(Place).order_by(Place.name).all()
        self._place_search.set_places(self._all_places)

    def _update_place_panel(
        self,
        place_id: Optional[int],
        place_name: str,
        is_anonymous: bool,
        source: Optional[str],
        place_lat: Optional[float] = None,
        place_lon: Optional[float] = None,
    ) -> None:
        self._place_search.set_current_by_id(place_id)
        self._place_rename_edit.setEnabled(place_id is not None)
        self._place_rename_btn.setEnabled(place_id is not None)
        if place_id is None:
            self._place_status_label.setText(t("ibp_place_none"))
            self._place_rename_edit.clear()
            self._place_coords_label.setText(t("ibp_place_coords_none"))
            return
        label = place_name
        if is_anonymous and source == "exif":
            label = f"{ANONYMOUS_GPS_PLACE_NAME} #{place_id}"
        self._place_status_label.setText(label)
        self._place_rename_edit.setText("" if is_anonymous else place_name)
        if place_lat is not None and place_lon is not None:
            self._place_coords_label.setText(
                t("ibp_place_coords_label", lat=place_lat, lon=place_lon)
            )
        else:
            self._place_coords_label.setText(t("ibp_place_coords_none"))

    def _on_place_selected(self, _place_id: int) -> None:
        pass

    def _assign_existing_place(self) -> None:
        if self._current_image_id is None:
            return
        place_id = self._place_search.current_place_id()
        if place_id is None:
            return
        with session_scope() as session:
            PlaceService(session).assign_place_to_image(self._current_image_id, place_id)
            place = session.get(Place, place_id)
            name = place.name if place else ""
            anon = bool(place and place.is_anonymous)
            source = place.source if place else None
            place_lat = place.latitude if place else None
            place_lon = place.longitude if place else None
            img = session.get(Image, self._current_image_id)
            img_exif_lat = img.exif_latitude if img else None
            img_image_lat = img.image_latitude if img else None
        log.info("Place assigned to image %d: %s", self._current_image_id, place_id)
        self._reload_places()
        self._update_place_panel(place_id, name, anon, source, place_lat, place_lon)
        # Requirement #4: write place GPS to EXIF if no EXIF GPS and no image-level coords
        if (
            img_exif_lat is None
            and img_image_lat is None
            and place_lat is not None
            and place_lon is not None
            and self._current_path
        ):
            from app.utils.exif import write_exif_gps
            write_exif_gps(self._current_path, place_lat, place_lon)
        self._reload_gps_panel()

    def _create_or_assign_place_by_text(self) -> None:
        # Read the name from the search box, falling back to the rename field so
        # a name typed in either input works.  Empty input used to fail silently
        # (the button "did nothing"); now we tell the user instead.
        name = self._place_search.current_query()
        if not name:
            name = self._place_rename_edit.text().strip()
        if not name:
            QMessageBox.information(
                self, t("ibp_place_create_btn"), t("ibp_place_need_name")
            )
            return
        with session_scope() as session:
            svc = PlaceService(session)
            if self._current_image_id is not None:
                # Image open → create (if new) and assign to it.
                place = svc.assign_place_to_image_by_name(self._current_image_id, name)
            else:
                # No image selected → still create the place so it exists and
                # shows up in the list, rather than silently doing nothing.
                place = svc.get_or_create_by_name(name)
            place_id = place.id
            place_name = place.name
            anon = place.is_anonymous
            source = place.source
            place_lat = place.latitude
            place_lon = place.longitude
            img = (
                session.get(Image, self._current_image_id)
                if self._current_image_id is not None
                else None
            )
            img_exif_lat = img.exif_latitude if img else None
            img_image_lat = img.image_latitude if img else None
        log.info(
            "Place created/assigned: image=%s name=%r", self._current_image_id, name
        )
        self._reload_places()
        self._update_place_panel(place_id, place_name, anon, source, place_lat, place_lon)
        # Requirement #4: write place GPS to EXIF if no EXIF GPS and no image-level coords
        if (
            img is not None
            and img_exif_lat is None
            and img_image_lat is None
            and place_lat is not None
            and place_lon is not None
            and self._current_path
        ):
            from app.utils.exif import write_exif_gps
            write_exif_gps(self._current_path, place_lat, place_lon)
        self._reload_gps_panel()

    def _rename_current_place(self) -> None:
        if self._current_image_id is None:
            return
        new_name = self._place_rename_edit.text().strip()
        if not new_name:
            return
        try:
            with session_scope() as session:
                img = session.get(Image, self._current_image_id)
                if img is None or img.place_id is None:
                    return
                place = PlaceService(session).name_place(img.place_id, new_name)
                place_id = place.id
                place_name = place.name
                anon = place.is_anonymous
                source = place.source
                place_lat = place.latitude
                place_lon = place.longitude
        except Exception as exc:  # noqa: BLE001
            from PySide6.QtWidgets import QMessageBox
            from app.ui.i18n import t
            QMessageBox.critical(
                self,
                t("ibp_place_rename_btn"),
                t("places_merge_error", error=str(exc)),
            )
            return
        log.info("Place renamed: %d -> %r", place_id, new_name)
        self._reload_places()
        self._update_place_panel(place_id, place_name, anon, source, place_lat, place_lon)

    # ── Image GPS coordinates ─────────────────────────────────────────

    def _load_image_coords(
        self,
        image_lat: Optional[float],
        image_lon: Optional[float],
        exif_lat: Optional[float],
        exif_lon: Optional[float],
        place_lat: Optional[float],
        place_lon: Optional[float],
        place_id: Optional[int],
    ) -> None:
        """Populate the GPS coordinate widgets according to priority order."""
        self._gps_edit.blockSignals(True)
        if image_lat is not None and image_lon is not None:
            self._gps_edit.setText(f"{image_lat:.6f}, {image_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_manual"))
            self._gps_source_label.setStyleSheet("color: #88ee88; font-size: 10px; font-style: italic;")
        elif exif_lat is not None and exif_lon is not None:
            self._gps_edit.setText(f"{exif_lat:.6f}, {exif_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_exif"))
            self._gps_source_label.setStyleSheet("color: #aaddaa; font-size: 10px; font-style: italic;")
        elif place_lat is not None and place_lon is not None:
            self._gps_edit.setText(f"{place_lat:.6f}, {place_lon:.6f}")
            self._gps_source_label.setText(t("ibp_gps_source_place"))
            self._gps_source_label.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        else:
            self._gps_edit.clear()
            self._gps_source_label.setText(t("ibp_gps_source_none"))
            self._gps_source_label.setStyleSheet("color: #555; font-size: 10px; font-style: italic;")
        self._gps_edit.blockSignals(False)

        # Show the "write place coords to EXIF" button only when applicable:
        # no EXIF GPS, no image-level coords, but place has coords
        show_place_btn = (
            exif_lat is None
            and image_lat is None
            and place_lat is not None
            and place_id is not None
        )
        self._gps_write_place_btn.setVisible(bool(show_place_btn))

    def _reload_gps_panel(self) -> None:
        """Reload GPS panel from DB for the current image."""
        if self._current_image_id is None:
            return
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            self._load_image_coords(
                img.image_latitude,
                img.image_longitude,
                img.exif_latitude,
                img.exif_longitude,
                img.place.latitude if img.place else None,
                img.place.longitude if img.place else None,
                img.place_id,
            )

    def _save_image_coords(self) -> None:
        """Validate and save the image-level GPS coordinate to DB and optionally EXIF."""
        if self._current_image_id is None:
            return
        from app.utils.exif import validate_coords, write_exif_gps

        text = self._gps_edit.text()
        parsed = validate_coords(text)
        if parsed is None:
            if text.strip():
                log.warning(
                    "Image %d: invalid GPS coordinate input %r — not saved",
                    self._current_image_id, text,
                )
                self._gps_source_label.setText(t("ibp_gps_invalid"))
                self._gps_source_label.setStyleSheet(
                    "color: #ee8888; font-size: 10px; font-style: italic;"
                )
            else:
                # Empty field → clear stored coords
                self._clear_image_coords()
            return

        lat, lon = parsed
        needs_exif_write = False
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            img.image_latitude = lat
            img.image_longitude = lon
            needs_exif_write = img.exif_latitude is None

        log.info("Image %d: image_latitude=%.6f image_longitude=%.6f saved", self._current_image_id, lat, lon)
        self._gps_source_label.setText(t("ibp_gps_source_manual"))
        self._gps_source_label.setStyleSheet("color: #88ee88; font-size: 10px; font-style: italic;")
        self._gps_write_place_btn.setVisible(False)

        # Requirement #3: write to EXIF if no EXIF GPS exists
        if needs_exif_write and self._current_path:
            write_exif_gps(self._current_path, lat, lon)

    def _clear_image_coords(self) -> None:
        """Remove the image-level GPS coordinate."""
        if self._current_image_id is None:
            return
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None:
                return
            img.image_latitude = None
            img.image_longitude = None
        log.info("Image %d: image_latitude/longitude cleared", self._current_image_id)
        self._reload_gps_panel()

    def _write_place_coords_to_exif(self) -> None:
        """Write the linked place's coordinates into this image's EXIF GPS fields."""
        if self._current_image_id is None or not self._current_path:
            return
        from app.utils.exif import write_exif_gps
        with session_scope() as session:
            img = session.get(Image, self._current_image_id)
            if img is None or img.place is None:
                return
            place_lat = img.place.latitude
            place_lon = img.place.longitude
        if place_lat is None or place_lon is None:
            return
        write_exif_gps(self._current_path, place_lat, place_lon)
        self._gps_write_place_btn.setVisible(False)

    def _update_exif_date(self) -> None:
        """Write the photo date field into the image EXIF DateTimeOriginal."""
        if not self._current_path:
            return
        from app.utils.exif import parse_flexible_date, write_exif_date

        date_str = self._photo_date_edit.text().strip()
        if not date_str:
            # Also try the DB value in case the field is stale
            if self._current_image_id is not None:
                with session_scope() as session:
                    img = session.get(Image, self._current_image_id)
                    date_str = (img.photo_date or "") if img else ""

        dt = parse_flexible_date(date_str)
        if dt is None:
            log.info(
                "EXIF date update skipped — unparseable date for image %r: %r",
                self._current_path, date_str,
            )
            return

        write_exif_date(self._current_path, dt)

    # ── Universal tree search ─────────────────────────────────────────

    def _tree_suggest(self, query: str) -> list[tuple[str, str, str]]:
        """Provide autocomplete suggestions for the image-browser search bar."""
        try:
            with session_scope() as session:
                return FamilyService(session).get_search_suggestions(query)
        except Exception:
            log.exception("ImageBrowserPanel: tree search autocomplete error")
            return []

    def _on_tree_search_changed(self) -> None:
        tokens = self._tree_search_bar.get_tokens()
        if not tokens:
            self._search_results_list.hide()
            self._file_tree.show()
            self._tree_hdr_lbl.setText(t("ib3_folders_hdr"))
            return
        self._run_tree_search(tokens)

    def _run_tree_search(self, tokens) -> None:
        from app.ui.widgets.universal_search_bar import (
            TOKEN_ANY, TOKEN_DATE, TOKEN_FAMILY_CODE, TOKEN_IMAGE,
            TOKEN_NICKNAME, TOKEN_PERSON, TOKEN_PLACE,
        )
        name_terms: list[str] = []
        any_terms: list[str] = []
        place_text = ""
        photo_date = ""
        image_text = ""

        for tok in tokens:
            if tok.token_type in (TOKEN_PERSON, TOKEN_NICKNAME, TOKEN_FAMILY_CODE):
                name_terms.append(tok.value)
            elif tok.token_type == TOKEN_PLACE:
                if not place_text:
                    place_text = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_DATE:
                if not photo_date:
                    photo_date = tok.value
                else:
                    any_terms.append(tok.value)
            elif tok.token_type == TOKEN_IMAGE:
                if not image_text:
                    image_text = tok.value
                else:
                    any_terms.append(tok.value)
            else:
                any_terms.append(tok.value)

        criteria = FamilyImageSearchCriteria(
            name_terms=tuple(name_terms),
            allow_other_people=True,
            image_text=image_text,
            photo_date=photo_date,
            place_text=place_text,
            any_terms=tuple(any_terms),
        )
        log.debug("ImageBrowserPanel tree search criteria: %r", criteria)

        try:
            with session_scope() as session:
                results, total = FamilyService(session).search_images_by_criteria(
                    criteria, limit=200
                )
        except Exception:
            log.exception("ImageBrowserPanel: tree search error")
            results, total = [], 0

        self._search_results_list.clear()
        self._search_result_items.clear()
        pool = QThreadPool.globalInstance()
        placeholder_icon = self._make_placeholder_icon()
        for result in results:
            item = QListWidgetItem(result.filename)
            item.setData(Qt.UserRole, result.image_id)
            item.setToolTip(result.file_path)
            item.setIcon(placeholder_icon)
            self._search_results_list.addItem(item)

            # Async thumbnail — never decode the full-resolution image on the
            # GUI thread (that froze the UI with many large photos).  Reuse the
            # shared thumbnail cache and the same worker pattern as the tree.
            cache_key = f"thumb_{result.image_id}"
            self._search_result_items[cache_key] = item
            cached = self._thumb_cache.get(cache_key)
            if cached is not None:
                item.setIcon(QIcon(cached))
            elif Path(result.file_path).exists():
                worker = ThumbnailRunnable(
                    file_path=result.file_path,
                    cache_key=cache_key,
                    size=_TREE_THUMB_SIZE,
                )
                worker.signals.ready.connect(self._on_thumbnail_ready)
                worker.signals.failed.connect(self._on_thumbnail_failed)
                pool.start(worker)

        if not results:
            placeholder = QListWidgetItem(t("ibp_search_no_results"))
            placeholder.setFlags(Qt.NoItemFlags)
            self._search_results_list.addItem(placeholder)

        self._tree_hdr_lbl.setText(t("ibp_search_results_hdr", n=total))
        self._file_tree.hide()
        self._search_results_list.show()

    def _on_search_result_double_clicked(self, item: QListWidgetItem) -> None:
        image_id = item.data(Qt.UserRole)
        if image_id is not None:
            self._load_image(int(image_id))

    # ── (legacy placeholder — no longer exposed in UI) ────────────────

    def _apply_place_filter_by_name(self) -> None:
        pass

    def _clear_place_filter(self) -> None:
        self._place_filter_id = None
        self._current_tree_item = None
        self.refresh()

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
        with session_scope() as session:
            self._all_persons = session.query(Person).order_by(Person.name).all()
        self._person_search.set_persons(
            self._all_persons,
            priority_ids=list(self._recent_assignment_person_ids),
        )

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

    def _on_person_search_selected(self, person_id: int) -> None:
        """Called when user picks a person via the search widget (single click or Enter)."""
        # Store pending selection; the Assign button still commits it.
        # This makes the widget's selection immediately visible but doesn't
        # save until the user presses Assign.
        pass  # selection is already reflected in current_person_id()

    def _on_assign_existing(self) -> None:
        if self._selected_face_id is None:
            return
        person_id = self._person_search.current_person_id()
        if person_id is None:
            return

        # Same defensive cleanup as _on_create_and_assign: the panel Assign
        # button can be pressed while the image label is in interactive-edit
        # mode.  Dismiss silently so the next face click works normally.
        if self._interactive_edit_face_id is not None:
            log.warning(
                "Assign existing: stale interactive-edit active for face_id=%d "
                "— dismissing before assignment",
                self._interactive_edit_face_id,
            )
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._draw_hint.setVisible(False)
            self._image_label.dismiss_interactive_edit()

        log.info(
            "Face assignment: face_id=%d → person_id=%d (panel assign button)",
            self._selected_face_id, person_id,
        )
        with session_scope() as session:
            IdentityService(session).reassign_face(self._selected_face_id, person_id)
        self._remember_recent_person(person_id)
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()

    def _on_create_and_assign(self) -> None:
        if self._selected_face_id is None:
            return
        name = self._new_name_edit.text().strip()
        if not name:
            QMessageBox.warning(
                self, t("ibp_empty_name_title"), t("ibp_empty_name_msg")
            )
            return

        # CRITICAL: the panel "Create" button can be pressed while the image
        # label is still in interactive-edit mode (_imode=True), because the
        # button is a separate widget outside the label.  If we don't clean up
        # here, _image_label._imode stays True after person creation, and the
        # next click on ANY face will be silently consumed by the stale edit-
        # exit handler instead of selecting that face.  The intermediate bbox
        # commits already persisted the latest position, so we just dismiss.
        if self._interactive_edit_face_id is not None:
            log.warning(
                "New person (panel): stale interactive-edit active for face_id=%d "
                "while creating person — dismissing interactive edit",
                self._interactive_edit_face_id,
            )
            self._interactive_edit_face_id = None
            self._interactive_edit_orig_bbox = None
            self._draw_hint.setVisible(False)
            self._image_label.dismiss_interactive_edit()

        face_id = self._selected_face_id
        with session_scope() as session:
            person = Person(name=name, is_auto_named=False)
            session.add(person)
            session.flush()
            person_id = person.id
            log.info(
                "New person creation: face_id=%d new_person='%s' person_id=%d",
                face_id, name, person_id,
            )
            IdentityService(session).reassign_face(face_id, person_id)
            log.info(
                "Face assignment: face_id=%d → person_id=%d (new person via panel)",
                face_id, person_id,
            )
        self._remember_recent_person(person_id)
        self._new_name_edit.clear()
        self._reload_current_face_data()
        self._reload_persons_combo()
        self.person_data_changed.emit()
        self._image_label.setFocus()
        self._open_person_info_dialog(person_id)

    def _open_person_info_dialog(self, person_id: int) -> None:
        with session_scope() as session:
            person = session.get(Person, person_id)
            if person is None:
                return
            dlg = PersonInfoDialog(person, parent=self)
            if dlg.exec() != PersonInfoDialog.Accepted:
                return
            person.gender = dlg.gender()
            person.family_code = dlg.family_code() or None
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

    # ──────────────────────────────────────────────────────────────────
    # Side panel visibility toggles
    # ──────────────────────────────────────────────────────────────────

    def _on_toggle_left_panel(self, checked: bool) -> None:
        """Show / hide the folder tree column. Image area grows to fill the space."""
        self._tree_panel.setVisible(checked)
        self._hide_inline_editor()
        self._image_label.update()

    def _on_toggle_right_panel(self, checked: bool) -> None:
        """Show / hide the info column. Image area grows to fill the space."""
        self._info_scroll.setVisible(checked)
        self._hide_inline_editor()
        self._image_label.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._hide_inline_editor()
        self._image_label.update()
