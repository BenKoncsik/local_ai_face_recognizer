"""Face viewer dialogs for the suggestion review workflow.

FullImageDialog   — original photo with highlighted face bbox and zoom/pan
FaceGalleryDialog — scrollable grid of all face crops for one person
CompareDialog     — side-by-side gallery comparison of two persons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
from PySide6.QtCore import QEvent, QPoint, QTimer, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QImage,
    QImageReader,
    QPainter,
    QPixmap,
    QPixmapCache,
)
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.db.database import session_scope
from app.db.models import Face
from app.services.identity_service import IdentityService
from app.ui.i18n import t

log = logging.getLogger(__name__)

_GALLERY_THUMB_SIZE = 100
_GALLERY_COLS = 4
_COMPARE_COLS = 3


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FaceGalleryEntry:
    face_id: int
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float = 0.0
    is_merge_excluded: bool = False


def _load_thumb_pixmap(path: Optional[str], size: int) -> Optional[QPixmap]:
    if not path or not Path(path).exists():
        return None
    QPixmapCache.remove(path)
    reader = QImageReader(path)
    image = reader.read()
    if image.isNull():
        return None
    return QPixmap.fromImage(image).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def _load_faces_for_person(
    person_id: int, exclude_pending: bool = False
) -> List[FaceGalleryEntry]:
    """Gallery entries for *person_id*.

    When *exclude_pending* is set, faces still awaiting auto-merge review are
    left out, so the gallery shows only the person's confirmed reference faces
    (used by the merge-review comparison panel).
    """
    from sqlalchemy import or_  # local import to avoid touching module header

    with session_scope() as session:
        query = (
            session.query(Face)
            .filter(
                Face.person_id == person_id,
                Face.is_excluded == False,  # noqa: E712
            )
        )
        if exclude_pending:
            query = query.filter(
                or_(
                    Face.auto_merge_review_status.is_(None),
                    Face.auto_merge_review_status != "pending",
                )
            )
        faces = query.all()
        return [
            FaceGalleryEntry(
                face_id=f.id,
                crop_path=f.crop_path,
                image_path=f.image.file_path if f.image else None,
                bbox=(f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h),
                confidence=f.confidence,
                is_merge_excluded=bool(f.is_merge_excluded),
            )
            for f in faces
        ]


def get_other_bboxes(
    image_path: str, exclude_face_id: int
) -> List[Tuple[int, int, int, int]]:
    """Return (x,y,w,h) of every non-excluded face on *image_path* except *exclude_face_id*."""
    with session_scope() as session:
        from app.db.models import Image as _Image  # avoid circular at module level

        img = session.query(_Image).filter(_Image.file_path == image_path).first()
        if not img:
            return []
        return [
            (f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h)
            for f in img.faces
            if f.id != exclude_face_id and not f.is_excluded
        ]


def _build_annotated_pixmap(
    image_path: str,
    highlight_bbox: Tuple[int, int, int, int],
    other_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[QPixmap]:
    """Load *image_path*, draw bboxes, return QPixmap or None on failure."""
    from app.utils.image_utils import load_image_bgr_normalized

    img = load_image_bgr_normalized(image_path)
    if img is None:
        log.warning("FullImageDialog: cannot load %r", image_path)
        return None
    img = img.copy()
    if other_bboxes:
        for x, y, w, h in other_bboxes:
            cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
    x, y, w, h = highlight_bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 220, 50), 3)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ih, iw, ch = rgb.shape
    qimg = QImage(rgb.data.tobytes(), iw, ih, ch * iw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Zoom scroll area (self-contained copy so we don't depend on preview_panel internals)
# ---------------------------------------------------------------------------

class _WheelZoomScrollArea(QScrollArea):
    """Scroll area that zooms the contained image with the mouse wheel and supports drag-to-pan."""

    def __init__(
        self,
        pixmap: QPixmap,
        focus_bbox: Optional[Tuple[int, int, int, int]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._source_pixmap = pixmap
        self._zoom = 1.0
        self._focus_bbox = focus_bbox
        self._drag_start: Optional[QPoint] = None
        self._drag_hval = 0
        self._drag_vval = 0

        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(pixmap)
        lbl.resize(pixmap.size())
        self._image_label = lbl
        self.setWidget(lbl)

        self.viewport().setCursor(Qt.OpenHandCursor)
        self.viewport().installEventFilter(self)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._focus_bbox is not None:
            QTimer.singleShot(0, self._scroll_to_focus)

    def _scroll_to_focus(self) -> None:
        if self._focus_bbox is None:
            return
        x, y, w, h = self._focus_bbox
        cx = (x + w / 2) * self._zoom
        cy = (y + h / 2) * self._zoom
        vp_w = self.viewport().width()
        vp_h = self.viewport().height()
        self.horizontalScrollBar().setValue(int(cx - vp_w / 2))
        self.verticalScrollBar().setValue(int(cy - vp_h / 2))

    def eventFilter(self, obj, event) -> bool:
        if obj is self.viewport():
            etype = event.type()
            if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
                self._drag_start = event.position().toPoint()
                self._drag_hval = self.horizontalScrollBar().value()
                self._drag_vval = self.verticalScrollBar().value()
                self.viewport().setCursor(Qt.ClosedHandCursor)
                return True
            elif etype == QEvent.Type.MouseMove and self._drag_start is not None:
                delta = event.position().toPoint() - self._drag_start
                self.horizontalScrollBar().setValue(self._drag_hval - delta.x())
                self.verticalScrollBar().setValue(self._drag_vval - delta.y())
                return True
            elif etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton:
                self._drag_start = None
                self.viewport().setCursor(Qt.OpenHandCursor)
                return True
        return super().eventFilter(obj, event)

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
        scaled = self._source_pixmap.scaled(
            self._source_pixmap.size() * self._zoom,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)
        self._image_label.resize(scaled.size())
        self.horizontalScrollBar().setValue(int(old_x * scale - cursor_pos.x()))
        self.verticalScrollBar().setValue(int(old_y * scale - cursor_pos.y()))
        event.accept()


# ---------------------------------------------------------------------------
# FullImageDialog
# ---------------------------------------------------------------------------

class FullImageDialog(QDialog):
    """Original photo with the selected face highlighted in green.

    Other faces on the same image are shown with a dim gray border.
    Supports mouse-wheel zoom and scroll-to-pan.
    """

    def __init__(
        self,
        image_path: str,
        highlight_bbox: Tuple[int, int, int, int],
        other_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("suggestions_full_img_title"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        pixmap = _build_annotated_pixmap(image_path, highlight_bbox, other_bboxes)
        if pixmap is not None and not pixmap.isNull():
            scroll = _WheelZoomScrollArea(pixmap, focus_bbox=highlight_bbox)
            scroll.setWidgetResizable(False)
            layout.addWidget(scroll)

            path_lbl = QLabel(image_path)
            path_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
            path_lbl.setWordWrap(True)
            layout.addWidget(path_lbl)

            screen = self.screen().availableGeometry() if self.screen() else None
            if screen:
                self.resize(
                    min(pixmap.width() + 40, screen.width() - 60),
                    min(pixmap.height() + 80, screen.height() - 80),
                )
            else:
                self.resize(900, 700)
        else:
            layout.addWidget(QLabel(t("cannot_load", path=image_path)))
            self.resize(500, 150)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)


# ---------------------------------------------------------------------------
# Gallery thumbnail
# ---------------------------------------------------------------------------

def _mark_excluded_pixmap(pixmap: QPixmap) -> QPixmap:
    """Return a dimmed copy of *pixmap* with a "?" badge in the corner."""
    result = QPixmap(pixmap.size())
    result.fill(Qt.transparent)
    painter = QPainter(result)
    painter.setRenderHint(QPainter.Antialiasing)
    # Dim the underlying crop so excluded faces visibly recede.
    painter.setOpacity(0.45)
    painter.drawPixmap(0, 0, pixmap)
    painter.setOpacity(1.0)
    # "?" badge, top-right.
    w = pixmap.width()
    r = max(16, w // 4)
    margin = 3
    cx, cy = w - r - margin, margin
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#e0883a"))
    painter.drawEllipse(cx, cy, r, r)
    painter.setPen(QColor("#1a1a1a"))
    font = QFont()
    font.setBold(True)
    font.setPixelSize(int(r * 0.75))
    painter.setFont(font)
    painter.drawText(cx, cy, r, r, Qt.AlignCenter, "?")
    painter.end()
    return result


class _GalleryThumb(QLabel):
    """Clickable thumbnail that emits the associated FaceGalleryEntry on click.

    When *allow_merge_exclusion* is set, a right-click context menu lets the
    user exclude the face from (or restore it to) the merge, and excluded faces
    render dimmed with a "?" badge and a distinct border.
    """

    clicked = Signal(object)  # FaceGalleryEntry
    exclusion_toggled = Signal(int, bool)  # (face_id, new_is_merge_excluded)

    def __init__(
        self,
        entry: FaceGalleryEntry,
        thumb_size: int = _GALLERY_THUMB_SIZE,
        allow_merge_exclusion: bool = False,
    ) -> None:
        super().__init__()
        self._entry = entry
        self._allow_merge_exclusion = allow_merge_exclusion
        self.setFixedSize(thumb_size, thumb_size)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        if allow_merge_exclusion:
            self.setContextMenuPolicy(Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(self._show_menu)

        excluded = entry.is_merge_excluded
        pixmap = _load_thumb_pixmap(entry.crop_path, thumb_size)
        if pixmap is not None:
            self.setPixmap(_mark_excluded_pixmap(pixmap) if excluded else pixmap)
            if excluded:
                self.setStyleSheet(
                    "QLabel { border: 2px dashed #e0883a; border-radius: 4px; }"
                    "QLabel:hover { border: 2px solid #e0883a; }"
                )
            else:
                self.setStyleSheet(
                    "QLabel { border: 1px solid #555; border-radius: 4px; }"
                    "QLabel:hover { border: 2px solid #88aaff; }"
                )
        else:
            self.setText("?")
            border = "#e0883a" if excluded else "#555"
            self.setStyleSheet(
                f"QLabel {{ background: #333; color: #888; border: 2px "
                f"{'dashed' if excluded else 'solid'} {border}; "
                "font-size: 20px; border-radius: 4px; }"
            )
        fname = Path(entry.image_path).name if entry.image_path else "?"
        tip = f"Face {entry.face_id}\n{fname}\n{entry.confidence:.0%}"
        if excluded:
            tip += f"\n{t('gallery_excluded_tooltip')}"
        self.setToolTip(tip)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._entry)

    def _show_menu(self, pos) -> None:
        menu = QMenu(self)
        if self._entry.is_merge_excluded:
            act = QAction(t("gallery_include_in_merge"), menu)
            act.triggered.connect(
                lambda: self.exclusion_toggled.emit(self._entry.face_id, False)
            )
        else:
            act = QAction(t("gallery_exclude_from_merge"), menu)
            act.triggered.connect(
                lambda: self.exclusion_toggled.emit(self._entry.face_id, True)
            )
        menu.addAction(act)
        menu.exec(self.mapToGlobal(pos))


def _open_full_image(entry: FaceGalleryEntry, parent: QWidget) -> None:
    """Open FullImageDialog for *entry*, loading sibling face bboxes from the DB."""
    if not entry.image_path:
        return
    other = get_other_bboxes(entry.image_path, entry.face_id)
    dlg = FullImageDialog(entry.image_path, entry.bbox, other, parent=parent)
    dlg.exec()


def _make_gallery_scroll(
    person_id: int,
    cols: int,
    parent: QWidget,
    allow_merge_exclusion: bool = False,
    on_exclusion_toggled=None,
    exclude_pending: bool = False,
) -> QScrollArea:
    """Return a QScrollArea containing a grid of face thumbnails for *person_id*.

    When *allow_merge_exclusion* is set, each thumbnail offers a right-click
    "exclude from merge" action; toggles are forwarded to
    *on_exclusion_toggled(face_id, new_value)*.  When *exclude_pending* is set,
    faces still awaiting auto-merge review are omitted.
    """
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)

    grid_widget = QWidget()
    grid = QGridLayout(grid_widget)
    grid.setSpacing(6)
    scroll.setWidget(grid_widget)

    entries = _load_faces_for_person(person_id, exclude_pending=exclude_pending)
    for i, entry in enumerate(entries):
        row_i, col_i = divmod(i, cols)
        thumb = _GalleryThumb(entry, allow_merge_exclusion=allow_merge_exclusion)
        thumb.clicked.connect(lambda e, p=parent: _open_full_image(e, p))
        if allow_merge_exclusion and on_exclusion_toggled is not None:
            thumb.exclusion_toggled.connect(on_exclusion_toggled)
        grid.addWidget(thumb, row_i, col_i)

    if not entries:
        lbl = QLabel(t("suggestions_no_faces"))
        lbl.setStyleSheet("color: #888; font-style: italic;")
        grid.addWidget(lbl, 0, 0)

    return scroll


# ---------------------------------------------------------------------------
# FaceGalleryDialog
# ---------------------------------------------------------------------------

class FaceGalleryDialog(QDialog):
    """Scrollable grid of all face crops for one person.

    Clicking any thumbnail opens FullImageDialog for the source photo.

    When *allow_merge_exclusion* is set (used for the Unknown candidate in the
    name-suggestion workflow), each face can be excluded from / restored to the
    merge via a right-click context menu.  Excluded faces stay visible but
    render dimmed with a "?" badge; the change is persisted immediately.
    """

    #: True if the user toggled any exclusion while the dialog was open — lets
    #: the caller know it should refresh derived state (counts, centroids).
    changed: bool

    def __init__(
        self,
        person_id: int,
        person_name: str,
        parent: Optional[QWidget] = None,
        allow_merge_exclusion: bool = False,
    ) -> None:
        super().__init__(parent)
        self._person_id = person_id
        self._allow_merge_exclusion = allow_merge_exclusion
        self.changed = False
        self.setWindowTitle(t("suggestions_gallery_title", name=person_name))
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(660, 540)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel(t("suggestions_gallery_title", name=person_name))
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #eee;")
        layout.addWidget(title)

        if allow_merge_exclusion:
            hint = QLabel(t("gallery_merge_exclusion_hint"))
            hint.setStyleSheet("color: #e0a060; font-size: 11px;")
            layout.addWidget(hint)
            self._status = QLabel("")
            self._status.setStyleSheet("color: #e0883a; font-size: 11px;")
            layout.addWidget(self._status)

        # Container that holds the (rebuildable) scroll area.
        self._scroll_holder = QVBoxLayout()
        self._scroll_holder.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._scroll_holder, stretch=1)
        self._scroll: Optional[QScrollArea] = None
        self._populate()

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

    def _populate(self) -> None:
        if self._scroll is not None:
            self._scroll_holder.removeWidget(self._scroll)
            self._scroll.deleteLater()
        self._scroll = _make_gallery_scroll(
            self._person_id,
            _GALLERY_COLS,
            self,
            allow_merge_exclusion=self._allow_merge_exclusion,
            on_exclusion_toggled=self._on_exclusion_toggled,
        )
        self._scroll_holder.addWidget(self._scroll)
        if self._allow_merge_exclusion:
            self._update_status()

    def _update_status(self) -> None:
        entries = _load_faces_for_person(self._person_id)
        n_excluded = sum(1 for e in entries if e.is_merge_excluded)
        self._status.setText(
            t("gallery_excluded_count", n=n_excluded) if n_excluded else ""
        )

    def _on_exclusion_toggled(self, face_id: int, excluded: bool) -> None:
        try:
            with session_scope() as session:
                IdentityService(session).set_face_merge_excluded(face_id, excluded)
        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to toggle merge exclusion for face %d", face_id)
            QMessageBox.warning(self, t("error"), str(exc))
            return
        self.changed = True
        self._populate()


# ---------------------------------------------------------------------------
# CompareDialog
# ---------------------------------------------------------------------------

class CompareDialog(QDialog):
    """Side-by-side gallery of two persons for visual comparison."""

    def __init__(
        self,
        candidate_id: int,
        candidate_name: str,
        target_id: int,
        target_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            t("suggestions_compare_title", cand=candidate_name, target=target_name)
        )
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self.resize(1060, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        for person_id, person_name, role in [
            (candidate_id, candidate_name, t("suggestions_candidate")),
            (target_id, target_name, t("suggestions_target")),
        ]:
            side = QWidget()
            side_layout = QVBoxLayout(side)
            side_layout.setContentsMargins(0, 0, 0, 0)
            side_layout.setSpacing(4)

            header = QLabel(f"{role}:  {person_name}")
            header.setStyleSheet(
                "font-size: 13px; font-weight: bold; color: #eee;"
            )
            side_layout.addWidget(header)

            gallery = _make_gallery_scroll(person_id, _COMPARE_COLS, side)
            side_layout.addWidget(gallery, stretch=1)

            splitter.addWidget(side)

        splitter.setSizes([530, 530])
        layout.addWidget(splitter, stretch=1)

        close_btn = QPushButton(t("close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)
