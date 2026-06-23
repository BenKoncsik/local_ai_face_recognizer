"""Sidebar panel — person / cluster list with search."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap, QPixmapCache
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.ui.i18n import t
from app.ui.widgets.person_search_select import PersonSearchSelect
from app.utils.person_search import PersonEntry

log = logging.getLogger(__name__)

_FACE_THUMB = 64
_FACE_COLS = 3
_POPUP_MAX = 380  # max width/height of hover popup


@dataclass
class _FaceData:
    """Plain-value snapshot of a face — safe to use after session closes."""
    face_id: Optional[int]
    person_id: Optional[int]
    crop_path: Optional[str]
    image_path: Optional[str]
    bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h


# Public alias — main_window builds these when refreshing the sidebar.
FaceData = _FaceData


@dataclass
class SidebarPerson:
    """Lightweight sidebar row — plain values only, no live ORM objects.

    Loading full Person/Face ORM graphs (including embedding blobs) for the
    sidebar froze the UI for many seconds at scale; the refresh now runs a few
    aggregate queries and builds these instead.
    """

    id: int
    name: str
    is_protected: bool
    face_count: int
    face: _FaceData


def _crop_mtime(crop_path: Optional[str]) -> Optional[float]:
    """Return a crop file's mtime so an in-place regenerated crop (same path,
    new pixels — e.g. after a bbox edit) still invalidates a reused thumbnail."""
    if not crop_path:
        return None
    try:
        return Path(crop_path).stat().st_mtime
    except OSError:
        return None


def _load_crop_pixmap(
    crop_path: Optional[str],
    size: int,
    face_id: Optional[int],
    person_id: Optional[int],
) -> Optional[QPixmap]:
    if not crop_path or not Path(crop_path).exists():
        return None
    QPixmapCache.remove(crop_path)
    reader = QImageReader(crop_path)
    reader.setAutoTransform(True)
    image = reader.read()
    if image.isNull():
        log.warning(
            "Sidebar crop load failed: FaceId=%s PersonId=%s preview_source=%s error=%s",
            face_id,
            person_id,
            crop_path,
            reader.errorString(),
        )
        return None
    pixmap = QPixmap.fromImage(image).scaled(
        size,
        size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )
    log.debug(
        "Render sidebar preview: FaceId=%s PersonId=%s preview_source=%s",
        face_id,
        person_id,
        crop_path,
    )
    return pixmap


def _render_original_with_box(
    image_path: str,
    bbox: Tuple[int, int, int, int],
    max_size: int,
) -> Optional[QPixmap]:
    """Load original image, draw bbox, return scaled QPixmap."""
    from app.utils.image_utils import load_image_bgr_normalized
    img = load_image_bgr_normalized(image_path)
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

    def __init__(self, person: SidebarPerson, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._person_id = person.id
        self._person_name = person.name
        self._face_data = person.face
        self._pixmap_loaded = False
        self.setObjectName(f"person-thumb-{person.id}")
        self.setProperty("person_id", person.id)
        self.setProperty("face_id", self._face_data.face_id or 0)
        self.setFixedSize(_FACE_THUMB, _FACE_THUMB)
        self.setAlignment(Qt.AlignCenter)
        # Placeholder look until the crop is actually loaded — the pixmap is
        # only read from disk when the thumb scrolls into the viewport, so
        # populating thousands of persons stays instant.
        self.setStyleSheet(
            "QLabel { background: #2a2a2a; border: 1px solid #555; border-radius: 4px; }"
            "QLabel:hover { border: 2px solid #88aaff; }"
        )
        self.setMouseTracking(True)

    def ensure_pixmap(self) -> None:
        """Load the crop image once, the first time the thumb is visible."""
        if self._pixmap_loaded:
            return
        self._pixmap_loaded = True
        self.setStyleSheet(
            "QLabel { border: 1px solid #555; border-radius: 4px; }"
            "QLabel:hover { border: 2px solid #88aaff; }"
        )
        self._load_pixmap()

    def _load_pixmap(self) -> None:
        crop = self._face_data.crop_path
        pixmap = _load_crop_pixmap(
            crop,
            _FACE_THUMB,
            self._face_data.face_id,
            self._face_data.person_id,
        )
        if pixmap is not None:
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


class SidebarPanel(QWidget):
    """Left sidebar showing persons as face thumbnails + searchable list.

    Signals:
        person_selected: ``(person_id: int)``
    """

    person_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._all_persons: list[SidebarPerson] = []
        # Incremental thumb-grid state: person_id → live widget, and the
        # signature that widget was built from (so unchanged persons keep
        # their existing widget instead of being re-created on every refresh).
        self._thumbs: dict[int, _PersonThumb] = {}
        self._thumb_sigs: dict[int, tuple] = {}
        self._build_ui()
        # Debounced viewport-visibility pass: crop pixmaps load only for
        # thumbs that are (nearly) visible, batched across scroll events.
        from PySide6.QtCore import QTimer
        self._visible_load_timer = QTimer(self)
        self._visible_load_timer.setSingleShot(True)
        self._visible_load_timer.setInterval(40)
        self._visible_load_timer.timeout.connect(self._load_visible_thumbs)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Face thumbnail grid at top ---
        face_box = QGroupBox(t("all_faces"))
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
        self._thumb_scroll = scroll
        scroll.verticalScrollBar().valueChanged.connect(
            lambda _=0: self._schedule_visible_load()
        )

        # --- Searchable person list below ---
        search_box = QGroupBox(t("people_label"))
        search_layout = QVBoxLayout(search_box)

        self._person_select = PersonSearchSelect()
        # The sidebar is a navigation surface: every person, including unknown
        # clusters, must stay clickable so they can be reviewed and named.
        self._person_select.set_hide_unknown_in_base(False)
        # Up/Down keyboard navigation should immediately select the person.
        self._person_select.set_navigate_selects(True)
        self._person_select.person_selected.connect(self.person_selected.emit)
        self._person_select.person_double_clicked.connect(self.person_selected.emit)
        search_layout.addWidget(self._person_select)

        self._count_label = QLabel(t("n_persons", n=0))
        self._count_label.setAlignment(Qt.AlignCenter)
        search_layout.addWidget(self._count_label)

        layout.addWidget(search_box, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, persons: list[SidebarPerson]) -> None:
        """Rebuild the list and thumbnail grid with the given persons."""
        self._all_persons = persons
        self._rebuild_thumb_grid(persons)
        entries = [
            PersonEntry(
                person_id=p.id,
                name=p.name,
                display_text=f"{'🔒 ' if p.is_protected else ''}{p.name}  ({p.face_count})",
            )
            for p in persons
        ]
        self._person_select.set_entries(entries)
        self._count_label.setText(t("n_persons", n=len(persons)))

    def current_person_id(self) -> Optional[int]:
        return self._person_select.current_person_id()

    # ------------------------------------------------------------------

    def _rebuild_thumb_grid(self, persons: list[SidebarPerson]) -> None:
        """Refresh the thumbnail grid, reusing widgets for unchanged persons.

        Re-creating every ``_PersonThumb`` reads a crop file from disk per
        person, so a full teardown froze the UI with many persons on every
        refresh (e.g. after a single face assignment).  Instead we diff against
        the live widgets: persons whose representative face/crop/name is
        unchanged keep their existing widget; only new or changed persons build
        a fresh one, and removed persons are deleted.
        """
        visible = [p for p in persons if not p.is_protected]

        # Compute the desired order and a change-signature per person.
        desired_ids: list[int] = []
        desired_sig: dict[int, tuple] = {}
        person_by_id: dict[int, SidebarPerson] = {}
        for p in visible:
            fd = p.face
            desired_ids.append(p.id)
            desired_sig[p.id] = (
                fd.face_id, fd.crop_path, fd.image_path, fd.bbox,
                p.name, p.is_protected, _crop_mtime(fd.crop_path),
            )
            person_by_id[p.id] = p

        # Detach every widget from the layout (cheap — no pixmap reload); kept
        # widgets are re-added below in the new order.
        while self._thumb_grid.count():
            self._thumb_grid.takeAt(0)

        # Drop widgets for persons that vanished or whose signature changed.
        desired_set = set(desired_ids)
        for pid in list(self._thumbs.keys()):
            if pid not in desired_set or self._thumb_sigs.get(pid) != desired_sig[pid]:
                self._thumbs.pop(pid).deleteLater()
                self._thumb_sigs.pop(pid, None)

        # Place every desired person, creating widgets only where missing.
        for i, pid in enumerate(desired_ids):
            thumb = self._thumbs.get(pid)
            if thumb is None:
                thumb = _PersonThumb(person_by_id[pid])
                thumb.clicked.connect(self.person_selected.emit)
                self._thumbs[pid] = thumb
                self._thumb_sigs[pid] = desired_sig[pid]
            row, col = divmod(i, _FACE_COLS)
            self._thumb_grid.addWidget(thumb, row, col)

        # Load pixmaps for the initially visible window once layout settles.
        self._schedule_visible_load()

    # ------------------------------------------------------------------
    # Viewport-lazy pixmap loading
    # ------------------------------------------------------------------

    #: Extra margin (px) above/below the viewport that is preloaded, so fast
    #: scrolling shows images instead of placeholders.
    _VISIBLE_BUFFER_PX = 160

    def _schedule_visible_load(self) -> None:
        self._visible_load_timer.start()

    def _load_visible_thumbs(self) -> None:
        """Load crop pixmaps for thumbs inside (or near) the viewport."""
        from PySide6.QtCore import QPoint, QRect

        viewport = self._thumb_scroll.viewport()
        region = viewport.rect().adjusted(
            0, -self._VISIBLE_BUFFER_PX, 0, self._VISIBLE_BUFFER_PX
        )
        for thumb in self._thumbs.values():
            if thumb._pixmap_loaded:
                continue
            top_left = thumb.mapTo(viewport, QPoint(0, 0))
            if region.intersects(QRect(top_left, thumb.size())):
                thumb.ensure_pixmap()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        self._schedule_visible_load()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._schedule_visible_load()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
