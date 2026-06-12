"""Cluster detail panel — face thumbnail grid for a selected person."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.db.models import Face
from app.services.face_date_service import FaceDateResolver
from app.ui.helpers.thumbnail_cache import get_thumbnail
from app.ui.i18n import t
from app.ui.widgets.face_timeline_view import FaceTimelineView
from app.utils.fuzzy_date import FuzzyDate

# View modes for the central area.
VIEW_GRID = "grid"
VIEW_TIMELINE = "timeline"

# Sort modes for the face grid (combo userData values).
SORT_ORIGINAL = "original"
SORT_DATE_ASC = "date_asc"
SORT_DATE_DESC = "date_desc"
SORT_QUALITY = "quality"

# Warning badge drawn over thumbnails of low-quality faces
_BADGE_COLOR = "#f57c00"   # orange

log = logging.getLogger(__name__)

_THUMB_SIZE = 96
_THUMB_SPACING = 4
_ZOOM_SIZE = 280


def _load_crop_pixmap(
    crop_path: Optional[str],
    size: int,
    face_id: Optional[int] = None,
) -> Optional[QPixmap]:
    """Load a crop thumbnail via the shared, content-aware cache.

    The cache keys on the file's mtime, so a regenerated crop at the same path
    is reloaded automatically (the reason this used to bypass Qt's path cache).
    """
    return get_thumbnail(crop_path, size, face_id)


class _ZoomPopup(QLabel):
    """Floating popup that shows a larger version of a face crop."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { background: #1a1a1a; border: 2px solid #88aaff; "
            "border-radius: 6px; padding: 4px; }"
        )
        self.setFixedSize(_ZOOM_SIZE + 8, _ZOOM_SIZE + 8)

    def show_for(self, crop_path: Optional[str], global_pos) -> None:
        pixmap = _load_crop_pixmap(crop_path, _ZOOM_SIZE)
        if pixmap is not None:
            self.setPixmap(pixmap)
        else:
            self.setText("?")

        screen = QApplication.primaryScreen().geometry()
        x = global_pos.x() + 16
        y = global_pos.y() - self.height() // 2
        if x + self.width() > screen.right():
            x = global_pos.x() - self.width() - 16
        if y < screen.top():
            y = screen.top() + 4
        if y + self.height() > screen.bottom():
            y = screen.bottom() - self.height() - 4
        self.move(x, y)
        self.show()


_zoom_popup = None


def _get_zoom_popup() -> _ZoomPopup:
    global _zoom_popup
    if _zoom_popup is None:
        _zoom_popup = _ZoomPopup()
    return _zoom_popup


class FaceThumbnail(QLabel):
    """Clickable face thumbnail widget.

    Signals:
        clicked: ``(face_id: int)``
        right_clicked: ``(face_id: int, global_x: int, global_y: int)``
    """

    clicked = Signal(int)
    right_clicked = Signal(int, int, int)

    def __init__(self, face: Face, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.face_id = face.id
        self._crop_path = face.crop_path
        self._is_low_quality: bool = bool(face.is_low_quality)
        self._is_pending: bool = face.auto_merge_review_status == "pending"
        self._is_uncertain: bool = bool(face.is_uncertain_identification)
        self._selected: bool = False
        self._person_name: Optional[str] = face.person.name if face.person else None
        self.setObjectName(f"face-thumb-{face.id}")
        self.setProperty("face_id", face.id)
        self.setProperty("crop_path", face.crop_path or "")
        # Lazy: the crop is decoded only when the tile scrolls into view
        # (see ClusterPanel._load_visible_thumbs), so large grids stay snappy.
        self._loaded = False
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        person_name = face.person.name if face.person else "—"
        quality_suffix = (
            f"\n⚠ {t('fq_low_quality_tip')}" if self._is_low_quality else ""
        )
        uncertain_suffix = (
            f"\n? {t('face_uncertain_tooltip')}" if self._is_uncertain else ""
        )
        if self._is_uncertain and face.identification_note:
            uncertain_suffix += f"\n{face.identification_note}"
        self.setToolTip(
            t(
                "face_tooltip",
                person=person_name,
                id=face.id,
                confidence=face.confidence,
                backend=face.detector_backend,
                file=Path(face.image.file_path).name if face.image else "?",
            ) + quality_suffix + uncertain_suffix
        )
        if self._is_pending:
            self.setToolTip(
                (self.toolTip() + "\n" if self.toolTip() else "")
                + t("amerge_pending_badge")
            )
        # Priority: pending (amber) > uncertain (orange) > low-quality (orange) > neutral.
        self._base_border = (
            "#ffb020" if self._is_pending
            else "#FF8C00" if self._is_uncertain
            else "#f57c00" if self._is_low_quality
            else "#555"
        )
        self._apply_border()
        self.setMouseTracking(True)

    def _apply_border(self) -> None:
        """Style the border according to the selection state."""
        if self._selected:
            self.setStyleSheet(
                "QLabel { border: 3px solid #89B4FA; border-radius: 4px; "
                "background: #1E2030; }"
            )
        else:
            self.setStyleSheet(
                f"QLabel {{ border: 1px solid {self._base_border}; border-radius: 4px; }}"
                "QLabel:hover { border: 2px solid #88aaff; }"
            )

    def set_selected(self, selected: bool) -> None:
        """Toggle the multi-selection highlight on this thumbnail."""
        if self._selected == selected:
            return
        self._selected = selected
        self._apply_border()

    def ensure_loaded(self) -> None:
        """Decode the crop on first display; cheap and idempotent afterwards."""
        if self._loaded:
            return
        self._loaded = True
        self._load_pixmap(self._crop_path)

    def _draw_corner_badge(self, pixmap, *, color, glyph, top_right, bottom=False):
        """Return a copy of *pixmap* with a small circular badge in a corner.

        *top_right* selects left (False) or right (True) on the horizontal axis.
        *bottom* selects the bottom row instead of the top row.
        """
        from PySide6.QtGui import QBrush, QColor, QFont
        badged = QPixmap(pixmap.size())
        badged.fill(Qt.transparent)
        painter = QPainter(badged)
        painter.drawPixmap(0, 0, pixmap)
        painter.setPen(Qt.NoPen)
        badge_size = 18
        margin = 2
        bx = (badged.width() - badge_size - margin) if top_right else margin
        by = (badged.height() - badge_size - margin) if bottom else margin
        painter.setBrush(QBrush(QColor(color)))
        painter.drawEllipse(bx, by, badge_size, badge_size)
        font = QFont()
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(bx, by, badge_size, badge_size, Qt.AlignCenter, glyph)
        painter.end()
        return badged

    def _load_pixmap(self, crop_path: Optional[str]) -> None:
        pixmap = _load_crop_pixmap(crop_path, _THUMB_SIZE, self.face_id)
        if pixmap is not None:
            if self._is_pending:
                pixmap = self._draw_corner_badge(
                    pixmap, color="#ffb020", glyph="?", top_right=False
                )
            if self._is_low_quality:
                pixmap = self._draw_corner_badge(
                    pixmap, color=_BADGE_COLOR, glyph="!", top_right=True
                )
            if self._is_uncertain:
                # Orange "?" badge in the bottom-right corner — distinct from the
                # pending (top-left amber) and low-quality (top-right orange "!") badges.
                pixmap = self._draw_corner_badge(
                    pixmap, color="#FF8C00", glyph="?", top_right=True,
                    bottom=True,
                )
            self.setPixmap(pixmap)
        else:
            self.setText("?")
            self.setStyleSheet(
                "QLabel { background: #333; color: #888; "
                "border: 1px solid #555; font-size: 20px; "
                "border-radius: 4px; }"
            )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        popup = _get_zoom_popup()
        popup.show_for(self._crop_path, self.mapToGlobal(self.rect().center()))

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        _get_zoom_popup().hide()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.face_id)

    def contextMenuEvent(self, event) -> None:
        gp = self.mapToGlobal(event.pos())
        self.right_clicked.emit(self.face_id, gp.x(), gp.y())


# Stacked-tile + group-preview tunables.
_GROUP_PREVIEW_THUMB = 72
_GROUP_PREVIEW_MAX = 10
_STACK_OFFSET = 5          # px each backing "card" is shifted by


def _compose_stacked_pixmap(rep_pixmap: QPixmap, count: int) -> QPixmap:
    """Render *rep_pixmap* as the top of a stack of offset cards + count badge.

    The result is always ``_THUMB_SIZE`` square so it tiles like a normal
    thumbnail.  Up to two backing cards peek out at the bottom-right to suggest
    "several photos", and a circular badge in the top-right shows *count*.
    """
    from PySide6.QtGui import QBrush, QColor, QFont, QPen

    canvas = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)

    # Backing cards (at most two), drawn back-to-front, offset down-right.
    layers = min(2, max(0, count - 1))
    inner = _THUMB_SIZE - layers * _STACK_OFFSET
    for i in range(layers, 0, -1):
        off = i * _STACK_OFFSET
        shade = 90 + i * 25
        painter.setPen(QPen(QColor(60, 60, 70)))
        painter.setBrush(QBrush(QColor(shade, shade, shade + 10)))
        painter.drawRoundedRect(off, off, inner, inner, 4, 4)

    # Representative crop on top (top-left aligned within the inner square).
    rep = rep_pixmap
    if rep.width() != inner or rep.height() != inner:
        rep = rep.scaled(inner, inner, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    px = (inner - rep.width()) // 2
    py = (inner - rep.height()) // 2
    painter.drawPixmap(px, py, rep)

    # Count badge — top-right of the whole tile.
    label = "99+" if count > 99 else str(count)
    badge_d = 22
    margin = 1
    bx = _THUMB_SIZE - badge_d - margin
    by = margin
    painter.setPen(QPen(QColor("#1e1e2e")))
    painter.setBrush(QBrush(QColor("#89b4fa")))
    painter.drawEllipse(bx, by, badge_d, badge_d)
    font = QFont()
    font.setPixelSize(11 if len(label) >= 3 else 12)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor("#11111b"))
    painter.drawText(bx, by, badge_d, badge_d, Qt.AlignCenter, label)
    painter.end()
    return canvas


class _GroupPreviewPopup(QWidget):
    """Floating strip showing every member crop of a merged group side by side."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "background: #1a1a1a; border: 2px solid #89b4fa; border-radius: 6px;"
        )
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(6, 6, 6, 6)
        self._row.setSpacing(4)
        self._labels: list[QLabel] = []

    def _clear(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._labels = []

    def show_for(self, crop_paths: list[str], global_pos) -> None:
        self._clear()
        shown = crop_paths[:_GROUP_PREVIEW_MAX]
        for cp in shown:
            lbl = QLabel()
            lbl.setFixedSize(_GROUP_PREVIEW_THUMB, _GROUP_PREVIEW_THUMB)
            lbl.setAlignment(Qt.AlignCenter)
            pm = _load_crop_pixmap(cp, _GROUP_PREVIEW_THUMB)
            if pm is not None:
                lbl.setPixmap(pm)
            else:
                lbl.setText("?")
            self._row.addWidget(lbl)
            self._labels.append(lbl)
        extra = len(crop_paths) - len(shown)
        if extra > 0:
            more = QLabel(f"+{extra}")
            more.setFixedSize(_GROUP_PREVIEW_THUMB // 2, _GROUP_PREVIEW_THUMB)
            more.setAlignment(Qt.AlignCenter)
            more.setStyleSheet("color: #cdd6f4; font-weight: bold;")
            self._row.addWidget(more)

        self.adjustSize()
        screen = QApplication.primaryScreen().geometry()
        x = global_pos.x() - self.width() // 2
        y = global_pos.y() + 16
        x = max(screen.left() + 4, min(x, screen.right() - self.width() - 4))
        if y + self.height() > screen.bottom():
            y = global_pos.y() - self.height() - 16
        self.move(x, y)
        self.show()


_group_popup = None


def _get_group_popup() -> _GroupPreviewPopup:
    global _group_popup
    if _group_popup is None:
        _group_popup = _GroupPreviewPopup()
    return _group_popup


class FaceGroupTile(QLabel):
    """A stacked tile representing a :class:`FaceGroup` of near-identical faces.

    A left click previews the representative face *and* toggles the group open
    (its members are spliced into the grid in place); clicking again collapses
    it.  Hovering pops up a strip of every member crop.

    Signals:
        clicked: ``(rep_face_id: int)`` — preview the representative.
        toggle_requested: ``(rep_face_id: int)`` — expand/collapse in place.
        right_clicked: ``(rep_face_id: int, global_x: int, global_y: int)``
    """

    clicked = Signal(int)
    toggle_requested = Signal(int)
    right_clicked = Signal(int, int, int)

    def __init__(self, group, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.rep_face_id = group.representative.id
        self._count = group.count
        self._rep_crop = group.representative.crop_path
        self._member_crops = [
            f.crop_path for f in group.members if f.crop_path
        ]
        self.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.setToolTip(t("cluster_group_tooltip", n=group.count))
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Compose the stacked pixmap on first display (idempotent)."""
        if self._loaded:
            return
        self._loaded = True
        rep_pm = _load_crop_pixmap(self._rep_crop, _THUMB_SIZE)
        if rep_pm is not None:
            self.setPixmap(_compose_stacked_pixmap(rep_pm, self._count))
        else:
            self.setText(f"×{self._count}")
            self.setStyleSheet(
                "QLabel { background: #333; color: #cdd6f4; border: 1px solid #555; "
                "border-radius: 4px; font-size: 18px; }"
            )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        _get_group_popup().show_for(
            self._member_crops, self.mapToGlobal(self.rect().center())
        )

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        _get_group_popup().hide()

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            _get_group_popup().hide()
            self.clicked.emit(self.rep_face_id)
            self.toggle_requested.emit(self.rep_face_id)

    def contextMenuEvent(self, event) -> None:
        gp = self.mapToGlobal(event.pos())
        self.right_clicked.emit(self.rep_face_id, gp.x(), gp.y())


class ClusterPanel(QWidget):
    """Scrollable grid of face thumbnails for the selected person.

    Signals:
        face_selected: ``(face_id: int)``
        face_right_clicked: ``(face_id: int, global_x: int, global_y: int)``
    """

    face_selected = Signal(int)
    face_right_clicked = Signal(int, int, int)
    selection_changed = Signal(int)  # number of multi-selected faces

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._face_names: dict[int, Optional[str]] = {}
        self._thumbs: dict[int, FaceThumbnail] = {}
        self._selected_ids: set[int] = set()
        # Ordered thumbnails as currently shown, used to reflow on resize.
        self._ordered_thumbs: list[FaceThumbnail] = []
        self._cur_cols: int = 0
        # Faces in their original (incoming) order + the active sort mode.
        self._faces: list[Face] = []
        self._faces_by_id: dict[int, Face] = {}
        self._sort_mode: str = SORT_ORIGINAL
        self._date_resolver = FaceDateResolver()
        # Merged ("stacked") view state.
        self._merged_mode: bool = False
        self._groups: list = []                       # list[FaceGroup]
        self._group_tiles: dict[int, FaceGroupTile] = {}   # rep_face_id → tile
        self._expanded_reps: set[int] = set()         # groups shown expanded
        # Timeline view state.
        self._view_mode: str = VIEW_GRID
        self._birth_date: Optional[str] = None
        self._death_date: Optional[str] = None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(2)

        self._header = QLabel(t("select_person_sidebar"))
        self._header.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._header)

        self._hint = QLabel(t("cluster_multiselect_hint"))
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #888; font-size: 11px;")
        outer.addWidget(self._hint)

        # --- Controls row: sort selector ---------------------------------
        controls = QHBoxLayout()
        controls.setContentsMargins(2, 0, 2, 0)
        controls.setSpacing(4)
        self._sort_label = QLabel(t("cluster_sort_label"))
        controls.addWidget(self._sort_label)
        self._sort_combo = QComboBox()
        self._populate_sort_combo()
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        controls.addWidget(self._sort_combo)
        controls.addStretch(1)
        self._view_combo = QComboBox()
        self._populate_view_combo()
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        controls.addWidget(self._view_combo)
        self._merge_check = QCheckBox(t("cluster_merge_toggle"))
        self._merge_check.setToolTip(t("cluster_merge_toggle_tip"))
        self._merge_check.toggled.connect(self._on_merge_toggled)
        controls.addWidget(self._merge_check)
        outer.addLayout(controls)

        # Central area: grid (page 0) and timeline (page 1) swapped via stack.
        self._stack = QStackedWidget()

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._grid_widget = QWidget()
        self._grid = QGridLayout(self._grid_widget)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(_THUMB_SPACING)
        # Pack thumbnails to the top-left; leftover space goes to the edges
        # rather than spreading the fixed-size cells apart.
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._grid_widget)
        # React directly to the viewport width changing — this is the value the
        # column count is derived from, so reflowing here makes the grid track
        # the splitter live as the preview panel is dragged wider/narrower.
        self._scroll.viewport().installEventFilter(self)
        # Decode crops lazily as they scroll into view.
        self._scroll.verticalScrollBar().valueChanged.connect(
            self._load_visible_thumbs
        )
        self._stack.addWidget(self._scroll)

        self._timeline = FaceTimelineView()
        self._timeline.face_selected.connect(self.face_selected.emit)
        self._stack.addWidget(self._timeline)

        outer.addWidget(self._stack)

    def _compute_cols(self) -> int:
        """How many thumbnail columns fit in the current viewport width."""
        avail = self._scroll.viewport().width()
        per_item = _THUMB_SIZE + _THUMB_SPACING
        return max(1, (avail + _THUMB_SPACING) // per_item)

    def _maybe_reflow(self) -> None:
        cols = self._compute_cols()
        if cols != self._cur_cols:
            self._reflow(cols)

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if obj is self._scroll.viewport() and event.type() == QEvent.Resize:
            self._maybe_reflow()
            # Viewport height may have just become real (e.g. first show) —
            # decode whatever is now on screen even if the column count is same.
            self._load_visible_thumbs()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._maybe_reflow()
        self._load_visible_thumbs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_person(
        self,
        person_name: str,
        faces: list[Face],
        *,
        birth_date: Optional[str] = None,
        death_date: Optional[str] = None,
    ) -> None:
        """Populate the grid with *faces* belonging to *person_name*.

        ``birth_date`` / ``death_date`` are the person's flexible date strings,
        used to bound the timeline view.
        """
        self._clear_grid()
        self._face_names.clear()
        self._thumbs.clear()
        self._ordered_thumbs.clear()
        self._selected_ids.clear()
        self._faces = list(faces)
        self._faces_by_id = {f.id: f for f in faces}
        self._date_resolver = FaceDateResolver()
        self._birth_date = birth_date
        self._death_date = death_date
        # Reset merged-view state for the new person.
        self._groups = []
        self._group_tiles = {}
        self._expanded_reps = set()
        self.selection_changed.emit(0)
        self._header.setText(t("face_count_header", name=person_name, n=len(faces)))

        for face in faces:
            log.debug(
                "Cluster grid item: FaceId=%s PersonId=%s crop_path=%r "
                "image_path=%r bbox=(%s,%s,%s,%s) preview_source=%r",
                face.id,
                face.person_id,
                face.crop_path,
                face.image.file_path if face.image else None,
                face.bbox_x,
                face.bbox_y,
                face.bbox_w,
                face.bbox_h,
                face.crop_path,
            )
            thumb = FaceThumbnail(face)
            thumb.clicked.connect(self._on_thumb_clicked)
            thumb.right_clicked.connect(self.face_right_clicked.emit)
            self._face_names[face.id] = thumb._person_name
            self._thumbs[face.id] = thumb

        # Order according to the active sort/merge mode, then lay out.
        self._rebuild_layout()
        if self._view_mode == VIEW_TIMELINE:
            self._refresh_timeline()

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _populate_sort_combo(self) -> None:
        self._sort_combo.blockSignals(True)
        self._sort_combo.clear()
        for key in (SORT_ORIGINAL, SORT_DATE_ASC, SORT_DATE_DESC, SORT_QUALITY):
            self._sort_combo.addItem(t(f"cluster_sort_{key}"), key)
        idx = self._sort_combo.findData(self._sort_mode)
        self._sort_combo.setCurrentIndex(max(0, idx))
        self._sort_combo.blockSignals(False)

    def _on_sort_changed(self, _index: int) -> None:
        mode = self._sort_combo.currentData()
        if mode and mode != self._sort_mode:
            self._sort_mode = mode
            self._rebuild_layout()
            if self._view_mode == VIEW_TIMELINE:
                self._refresh_timeline()

    def _populate_view_combo(self) -> None:
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItem(t("cluster_view_grid"), VIEW_GRID)
        self._view_combo.addItem(t("cluster_view_timeline"), VIEW_TIMELINE)
        idx = self._view_combo.findData(self._view_mode)
        self._view_combo.setCurrentIndex(max(0, idx))
        self._view_combo.blockSignals(False)

    def _on_view_changed(self, _index: int) -> None:
        mode = self._view_combo.currentData()
        if not mode or mode == self._view_mode:
            return
        self._view_mode = mode
        if mode == VIEW_TIMELINE:
            self._stack.setCurrentWidget(self._timeline)
            self._refresh_timeline()
        else:
            self._stack.setCurrentWidget(self._scroll)

    def _refresh_timeline(self) -> None:
        """(Re)build the timeline scene from the current groups + biography."""
        self._ensure_groups()
        self._timeline.set_data(
            self._groups,
            birth=FuzzyDate.parse(self._birth_date),
            death=FuzzyDate.parse(self._death_date),
            resolver=self._date_resolver,
        )

    def _sorted_faces(self) -> list[Face]:
        """Return ``self._faces`` ordered by the active sort mode."""
        mode = self._sort_mode
        faces = self._faces
        if mode == SORT_DATE_ASC or mode == SORT_DATE_DESC:
            known, unknown = [], []
            for f in faces:
                (known if self._date_resolver.for_face(f).is_known else unknown).append(f)
            known.sort(
                key=lambda f: self._date_resolver.for_face(f).sort_key,
                reverse=(mode == SORT_DATE_DESC),
            )
            return known + unknown
        if mode == SORT_QUALITY:
            # Highest quality first; unscored faces count as usable (1.0).
            return sorted(
                faces,
                key=lambda f: (f.quality_score if f.quality_score is not None else 1.0),
                reverse=True,
            )
        return list(faces)  # SORT_ORIGINAL — incoming order

    def _sort_key_for(self, face: Face):
        """Sort key for a single face under the active mode (matches ``_sorted_faces``)."""
        mode = self._sort_mode
        if mode in (SORT_DATE_ASC, SORT_DATE_DESC):
            return self._date_resolver.for_face(face).sort_key
        if mode == SORT_QUALITY:
            q = face.quality_score if face.quality_score is not None else 1.0
            return -q  # ascending key → highest quality first
        return 0

    # ------------------------------------------------------------------
    # Merged ("stacked") view
    # ------------------------------------------------------------------

    def _on_merge_toggled(self, checked: bool) -> None:
        self._merged_mode = bool(checked)
        self._expanded_reps.clear()
        self._rebuild_layout()

    def set_merged_mode(self, enabled: bool) -> None:
        """Programmatically toggle the merged view (keeps the checkbox in sync)."""
        if self._merge_check.isChecked() != enabled:
            self._merge_check.setChecked(enabled)  # triggers _on_merge_toggled

    def _ensure_groups(self) -> None:
        if self._groups:
            return
        from app.services.face_grouping_service import group_faces
        # Group in incoming order; ordering for display is applied separately.
        self._groups = group_faces(self._faces)

    def _ordered_groups(self) -> list:
        """Return the merged groups ordered to match the active sort mode."""
        self._ensure_groups()
        groups = self._groups
        mode = self._sort_mode
        if mode in (SORT_DATE_ASC, SORT_DATE_DESC):
            known, unknown = [], []
            for g in groups:
                span = g.date_span(self._date_resolver)
                (known if span.is_known else unknown).append(g)
            known.sort(
                key=lambda g: g.date_span(self._date_resolver).sort_key,
                reverse=(mode == SORT_DATE_DESC),
            )
            return known + unknown
        if mode == SORT_QUALITY:
            return sorted(
                groups,
                key=lambda g: (
                    g.representative.quality_score
                    if g.representative.quality_score is not None else 1.0
                ),
                reverse=True,
            )
        return list(groups)  # original first-seen order

    def _group_tile_for(self, group) -> "FaceGroupTile":
        rep_id = group.representative.id
        tile = self._group_tiles.get(rep_id)
        if tile is None:
            tile = FaceGroupTile(group)
            tile.clicked.connect(self._on_group_clicked)
            tile.toggle_requested.connect(self._on_group_toggle)
            tile.right_clicked.connect(self.face_right_clicked.emit)
            self._group_tiles[rep_id] = tile
        return tile

    def _on_group_clicked(self, rep_face_id: int) -> None:
        # Preview the representative without altering the multi-selection.
        self.face_selected.emit(rep_face_id)

    def _on_group_toggle(self, rep_face_id: int) -> None:
        if rep_face_id in self._expanded_reps:
            self._expanded_reps.discard(rep_face_id)
        else:
            self._expanded_reps.add(rep_face_id)
        self._rebuild_layout()

    # ------------------------------------------------------------------

    def _ordered_widgets(self) -> list[QWidget]:
        """Build the ordered list of widgets to lay out for the current mode."""
        if not self._merged_mode:
            return [
                self._thumbs[f.id]
                for f in self._sorted_faces()
                if f.id in self._thumbs
            ]
        widgets: list[QWidget] = []
        for group in self._ordered_groups():
            rep_id = group.representative.id
            expanded = rep_id in self._expanded_reps
            if group.is_singleton or expanded:
                members = sorted(group.members, key=self._sort_key_for)
                widgets.extend(
                    self._thumbs[m.id] for m in members if m.id in self._thumbs
                )
            else:
                widgets.append(self._group_tile_for(group))
        return widgets

    def _rebuild_layout(self) -> None:
        """Re-order/compose the grid per the active sort + merge state."""
        # Hide widgets that are not part of the new layout so stale tiles or
        # collapsed members don't linger in the grid.
        new_widgets = self._ordered_widgets()
        new_set = set(map(id, new_widgets))
        for thumb in self._thumbs.values():
            if id(thumb) not in new_set:
                thumb.setParent(None)
        for tile in self._group_tiles.values():
            if id(tile) not in new_set:
                tile.setParent(None)
        self._ordered_thumbs = new_widgets
        self._cur_cols = 0
        self._reflow(self._compute_cols())

    def _reflow(self, cols: int) -> None:
        """Arrange the existing thumbnails into *cols* columns."""
        self._cur_cols = cols
        # Detach without deleting — the thumbnails are reused.
        for thumb in self._ordered_thumbs:
            self._grid.removeWidget(thumb)
        for i, thumb in enumerate(self._ordered_thumbs):
            row, col = divmod(i, cols)
            self._grid.addWidget(thumb, row, col)
            # setParent(None) (used to hide stale tiles) also hides the widget;
            # ensure anything we re-add to the grid is visible again.
            thumb.show()
        # Decode the crops that are (or are about to be) on screen.
        self._load_visible_thumbs()

    def _load_visible_thumbs(self) -> None:
        """Lazily decode crops for tiles within (or near) the visible viewport.

        Tiles keep an unpainted placeholder until they scroll close to view,
        so a person with thousands of faces never decodes them all up front.
        A one-viewport buffer above and below keeps scrolling smooth.

        Row positions are derived arithmetically from the index and column count
        rather than read back from Qt geometry — the layout settles
        asynchronously, so ``widget.y()`` is unreliable right after a reflow.
        """
        if not self._ordered_thumbs:
            return
        cols = max(1, self._cur_cols)
        row_h = _THUMB_SIZE + _THUMB_SPACING
        vp_h = self._scroll.viewport().height()
        top = self._scroll.verticalScrollBar().value()
        buffer = vp_h  # one screen of look-ahead in each direction
        lo_row = max(0, (top - buffer) // row_h)
        hi_row = (top + vp_h + buffer) // row_h
        for i, w in enumerate(self._ordered_thumbs):
            loader = getattr(w, "ensure_loaded", None)
            if loader is None:
                continue
            if lo_row <= (i // cols) <= hi_row:
                loader()

    def get_face_person_name(self, face_id: int) -> Optional[str]:
        """Return the person name for *face_id* as loaded in the current grid."""
        return self._face_names.get(face_id)

    def selected_face_ids(self) -> list[int]:
        """Return the face ids currently multi-selected (in grid order)."""
        return [fid for fid in self._face_names if fid in self._selected_ids]

    def clear_selection(self) -> None:
        """Deselect every thumbnail without rebuilding the grid."""
        for fid in list(self._selected_ids):
            thumb = self._thumbs.get(fid)
            if thumb is not None:
                thumb.set_selected(False)
        self._selected_ids.clear()
        self.selection_changed.emit(0)

    def clear(self) -> None:
        self._clear_grid()
        self._face_names.clear()
        self._thumbs.clear()
        self._group_tiles.clear()
        self._ordered_thumbs.clear()
        self._faces = []
        self._faces_by_id = {}
        self._groups = []
        self._expanded_reps = set()
        self._cur_cols = 0
        self._selected_ids.clear()
        self.selection_changed.emit(0)
        self._header.setText(t("select_person_sidebar"))

    def retranslate(self) -> None:
        """Refresh translatable strings after a language change."""
        self._hint.setText(t("cluster_multiselect_hint"))
        self._sort_label.setText(t("cluster_sort_label"))
        self._merge_check.setText(t("cluster_merge_toggle"))
        self._merge_check.setToolTip(t("cluster_merge_toggle_tip"))
        self._populate_sort_combo()
        self._populate_view_combo()
        if self._view_mode == VIEW_TIMELINE and self._faces:
            self._refresh_timeline()

    # ------------------------------------------------------------------

    def _on_thumb_clicked(self, face_id: int) -> None:
        """Toggle multi-selection for *face_id* and forward the preview select.

        A plain click both toggles whether the face is part of the batch
        selection and previews it (preserving the single-face workflow, which
        keys off the most recently clicked face).
        """
        if face_id in self._selected_ids:
            self._selected_ids.discard(face_id)
            sel = False
        else:
            self._selected_ids.add(face_id)
            sel = True
        thumb = self._thumbs.get(face_id)
        if thumb is not None:
            thumb.set_selected(sel)
        self.face_selected.emit(face_id)
        self.selection_changed.emit(len(self._selected_ids))

    # ------------------------------------------------------------------

    def _clear_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
