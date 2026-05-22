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


def _draw_faces(
    img_bgr: np.ndarray,
    faces: List[_FaceData],
    selected_id: Optional[int],
) -> np.ndarray:
    img = img_bgr.copy()
    for face_id, x, y, w, h, person_name, _ in faces:
        selected = face_id == selected_id
        color = (50, 220, 50) if selected else (180, 180, 180)
        thickness = 3 if selected else 2
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

        name = person_name or "?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.4, min(1.1, w / 80))
        (tw, th), bl = cv2.getTextSize(name, font, scale, 2)
        ty = max(y - 6, th + 6)
        cv2.rectangle(img, (x, ty - th - bl - 4), (x + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(img, name, (x + 3, ty - bl), font, scale, color, 2)
    return img


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
    """QLabel that can either emit a click position or a drawn rectangle.

    In click mode (default): left-click emits ``clicked(x, y)`` in label coords.
    In draw mode: mouse drag emits ``rect_drawn(QRect)`` in label coords;
                  a yellow dashed rectangle is rendered live while dragging.
    """

    clicked = Signal(int, int)
    rect_drawn = Signal(QRect)
    right_clicked = Signal(int, int)

    _BORDER_NORMAL = "QLabel { background: #1a1a1a; }"
    _BORDER_DRAW = "QLabel { background: #1a1a1a; border: 2px solid #ffcc00; }"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCursor(Qt.CrossCursor)
        self._draw_mode = False
        self._start: Optional[QPoint] = None
        self._end: Optional[QPoint] = None
        self.setStyleSheet(self._BORDER_NORMAL)

    # ── Public ────────────────────────────────────────────────────────

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self._start = None
        self._end = None
        self.setStyleSheet(self._BORDER_DRAW if enabled else self._BORDER_NORMAL)
        self.update()

    # ── Mouse events ─────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
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
        if self._draw_mode and self._start is not None:
            self._end = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
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

    # ── Paint rubber-band rect ────────────────────────────────────────

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._draw_mode and self._start is not None and self._end is not None:
            painter = QPainter(self)
            pen = QPen(Qt.yellow, 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self._start, self._end).normalized())


# ──────────────────────────────────────────────────────────────────────────────
# Main panel
# ──────────────────────────────────────────────────────────────────────────────

class ImageBrowserPanel(QWidget):
    """Browse all scanned images; click a face to identify or categorise it."""

    def __init__(self, config=None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._config = config            # AppConfig | None
        self._image_ids: List[int] = []
        self._current_index: int = 0
        self._current_path: str = ""
        self._face_data: List[_FaceData] = []
        self._selected_face_id: Optional[int] = None
        self._editing_face_id: Optional[int] = None   # face being bbox-edited
        self._renaming: bool = False                   # inline rename in progress
        self._detection_done: bool = False             # whether current image was processed
        self._orig_img_bgr: Optional[np.ndarray] = None
        self._full_pixmap: Optional[QPixmap] = None
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
            self._image_label.setText("Nincs kép az adatbázisban")
            self._folder_label.setText("")
            self._clear_face_panel()
            self._update_nav_buttons()
            return

        self._current_index = min(self._current_index, count - 1)
        self._load_current_image()
        self._update_nav_buttons()
        self._reload_persons_combo()

    # ──────────────────────────────────────────────────────────────────
    # Navigation
    # ──────────────────────────────────────────────────────────────────

    def _on_prev(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._selected_face_id = None
            self._load_current_image()
            self._update_nav_buttons()

    def _on_next(self) -> None:
        if self._current_index < len(self._image_ids) - 1:
            self._current_index += 1
            self._selected_face_id = None
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

        self._counter_label.setText(
            f"{self._current_index + 1} / {len(self._image_ids)}"
        )
        self._folder_label.setText(str(Path(self._current_path).parent))

        from app.utils.image_utils import load_image_bgr

        img_bgr = load_image_bgr(self._current_path)
        if img_bgr is None:
            self._orig_img_bgr = None
            self._full_pixmap = None
            self._image_label.setText(f"Nem tölthető be:\n{self._current_path}")
            self._clear_face_panel()
            return

        self._orig_img_bgr = img_bgr
        self._redraw_faces()
        self._clear_face_panel()

    def _update_scaled_pixmap(self) -> None:
        if self._full_pixmap is None:
            return
        lw = self._image_label.width()
        lh = self._image_label.height()
        if lw <= 0 or lh <= 0:
            return
        scaled = self._full_pixmap.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

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
            self._draw_mode_btn.setChecked(False)
            self._reload_current_face_data()
            self._selected_face_id = edited_id
            self._redraw_faces()
            self._show_face_info(edited_id)
        else:
            # Add mode — create new manual face
            img_id = self._image_ids[self._current_index]
            new_face_id = self._save_manual_face(img_id, ix, iy, iw, ih)
            self._draw_mode_btn.setChecked(False)
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
        """Convert label (display) coordinates to original image pixel coordinates."""
        if self._full_pixmap is None:
            return -1, -1
        pw = self._full_pixmap.width()
        ph = self._full_pixmap.height()
        lw = self._image_label.width()
        lh = self._image_label.height()
        if pw == 0 or ph == 0:
            return -1, -1
        scale = min(lw / pw, lh / ph)
        disp_w = pw * scale
        disp_h = ph * scale
        offset_x = (lw - disp_w) / 2.0
        offset_y = (lh - disp_h) / 2.0
        rx = lx - offset_x
        ry = ly - offset_y
        if rx < 0 or ry < 0 or rx >= disp_w or ry >= disp_h:
            return -1, -1
        return int(rx / scale), int(ry / scale)

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
        self._update_scaled_pixmap()

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
        else:
            self._face_status_label.setText("Ez az arc nincs kategorizálva")
            self._person_name_label.setVisible(False)
            self._rename_btn.setVisible(False)
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
        self._assign_btn.setEnabled(False)
        self._create_btn.setEnabled(False)
        self._new_name_edit.clear()

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

    # ──────────────────────────────────────────────────────────────────
    # Person combo
    # ──────────────────────────────────────────────────────────────────

    def _reload_persons_combo(self) -> None:
        self._person_combo.clear()
        with session_scope() as session:
            persons = session.query(Person).order_by(Person.name).all()
            for p in persons:
                self._person_combo.addItem(p.name, userData=p.id)

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
        self._reload_current_face_data()
        self._reload_persons_combo()

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
        self._new_name_edit.clear()
        self._reload_current_face_data()
        self._reload_persons_combo()

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
