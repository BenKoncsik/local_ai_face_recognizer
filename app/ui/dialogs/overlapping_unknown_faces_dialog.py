"""Dialog listing unknown face boxes that overlap known faces."""

from __future__ import annotations

import logging
import os
from typing import Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.services.duplicate_unknown_face_finder import OverlappingUnknownFaceMatch
from app.ui.i18n import t

log = logging.getLogger(__name__)

# ── Column indices ────────────────────────────────────────────────────────────
_COL_IMAGE = 0
_COL_PREVIEW = 1
_COL_PERSON = 2
_COL_UNKNOWN_ID = 3
_COL_KNOWN_ID = 4
_COL_OVERLAP = 5
_COL_DELETE = 6
_NUM_COLS = 7

# ── Preview sizes ─────────────────────────────────────────────────────────────
_THUMB_PX = 80        # thumbnail cell size (square)
_PREVIEW_PX = 440     # max dimension of the hover popup image
_MARGIN_FACTOR = 0.45 # padding around the union bounding box (fraction of union size)

# ── Box colours ───────────────────────────────────────────────────────────────
_COLOUR_KNOWN = QColor(40, 220, 40)    # green  — named/known face
_COLOUR_UNKNOWN = QColor(255, 100, 0)  # orange — ? / unknown face


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _union_crop_rect(
    unknown_bbox: tuple[int, int, int, int],
    known_bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Return (x, y, w, h) crop rectangle clamped to image bounds."""
    ux, uy, uw, uh = unknown_bbox
    kx, ky, kw, kh = known_bbox
    x1 = min(ux, kx)
    y1 = min(uy, ky)
    x2 = max(ux + uw, kx + kw)
    y2 = max(uy + uh, ky + kh)
    mw = int((x2 - x1) * _MARGIN_FACTOR)
    mh = int((y2 - y1) * _MARGIN_FACTOR)
    cx = max(0, x1 - mw)
    cy = max(0, y1 - mh)
    cx2 = min(img_w, x2 + mw)
    cy2 = min(img_h, y2 + mh)
    return cx, cy, cx2 - cx, cy2 - cy


def _draw_boxes(
    pix: QPixmap,
    unknown_bbox: tuple[int, int, int, int],
    known_bbox: tuple[int, int, int, int],
    crop_x: int,
    crop_y: int,
    sx: float,
    sy: float,
) -> QPixmap:
    """Return a copy of *pix* with both bounding boxes painted on it."""
    out = pix.copy()
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, False)
    for bbox, colour, width in [
        (known_bbox, _COLOUR_KNOWN, 2),
        (unknown_bbox, _COLOUR_UNKNOWN, 2),
    ]:
        bx, by, bw, bh = bbox
        rx = int((bx - crop_x) * sx)
        ry = int((by - crop_y) * sy)
        rw = max(1, int(bw * sx))
        rh = max(1, int(bh * sy))
        pen = QPen(colour, width)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rx, ry, rw, rh)
    painter.end()
    return out


# ── Thumbnail / preview cache ─────────────────────────────────────────────────

class _ThumbnailCache:
    """Main-thread pixmap cache keyed by match coordinates."""

    def __init__(self) -> None:
        self._thumbs: dict[str, QPixmap | None] = {}
        self._previews: dict[str, QPixmap | None] = {}

    @staticmethod
    def _key(match: OverlappingUnknownFaceMatch) -> str:
        return (
            f"{match.image_path}"
            f"|u{match.unknown_face_id}k{match.known_face_id}"
            f"|{match.unknown_bbox}|{match.known_bbox}"
        )

    def _render(self, match: OverlappingUnknownFaceMatch, max_px: int) -> QPixmap | None:
        try:
            if not os.path.isfile(match.image_path):
                log.warning("Overlap preview: missing %s", match.image_path)
                return None
            full = QPixmap(match.image_path)
            if full.isNull():
                log.warning("Overlap preview: cannot load %s", match.image_path)
                return None
            cx, cy, cw, ch = _union_crop_rect(
                match.unknown_bbox, match.known_bbox,
                full.width(), full.height(),
            )
            if cw <= 0 or ch <= 0:
                log.warning("Overlap preview: degenerate crop for %s", match.image_path)
                return None
            cropped = full.copy(cx, cy, cw, ch)
            scaled = cropped.scaled(
                max_px, max_px,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            return _draw_boxes(
                scaled,
                match.unknown_bbox, match.known_bbox,
                cx, cy,
                scaled.width() / cw,
                scaled.height() / ch,
            )
        except Exception:
            log.exception("Overlap preview: unexpected error for %s", match.image_path)
            return None

    def thumbnail(self, match: OverlappingUnknownFaceMatch) -> QPixmap | None:
        k = self._key(match)
        if k not in self._thumbs:
            self._thumbs[k] = self._render(match, _THUMB_PX)
        return self._thumbs[k]

    def preview(self, match: OverlappingUnknownFaceMatch) -> QPixmap | None:
        k = self._key(match)
        if k not in self._previews:
            self._previews[k] = self._render(match, _PREVIEW_PX)
        return self._previews[k]


# ── Hover popup ───────────────────────────────────────────────────────────────

class _HoverPopup(QLabel):
    """Frameless floating window that shows an enlarged preview."""

    def __init__(self, cache: _ThumbnailCache, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self._cache = cache
        self._match: OverlappingUnknownFaceMatch | None = None
        self._source: QWidget | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_now)
        self.setStyleSheet(
            "QLabel {"
            "  background: #1a1a1a;"
            "  border: 1px solid #666;"
            "  padding: 4px;"
            "}"
        )
        self.setAlignment(Qt.AlignCenter)

    def schedule(self, match: OverlappingUnknownFaceMatch, source: QWidget) -> None:
        self._match = match
        self._source = source
        self._timer.start(160)

    def cancel(self) -> None:
        self._timer.stop()
        self.hide()

    def _show_now(self) -> None:
        if self._match is None:
            return
        pix = self._cache.preview(self._match)
        if pix and not pix.isNull():
            self.setPixmap(pix)
            self.setText("")
        else:
            self.setPixmap(QPixmap())
            self.setText(t("overlap_preview_unavailable"))
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()

    def _reposition(self) -> None:
        if self._source is None:
            return
        # Prefer right of source; fall back to left; clamp vertically.
        tr = self._source.mapToGlobal(self._source.rect().topRight())
        x, y = tr.x() + 12, tr.y()
        pw, ph = self.width(), self.height()
        screen = QApplication.screenAt(tr)
        if screen:
            sg = screen.availableGeometry()
            if x + pw > sg.right():
                x = tr.x() - pw - 12
            if x < sg.left():
                x = sg.left() + 4
            if y + ph > sg.bottom():
                y = sg.bottom() - ph - 4
            if y < sg.top():
                y = sg.top() + 4
        self.move(x, y)


# ── Thumbnail cell widget ─────────────────────────────────────────────────────

class _ThumbnailCell(QWidget):
    """Compact cell widget that loads its pixmap lazily and shows a hover popup."""

    def __init__(
        self,
        match: OverlappingUnknownFaceMatch,
        cache: _ThumbnailCache,
        popup: _HoverPopup,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._match = match
        self._cache = cache
        self._popup = popup
        self._loaded = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setAlignment(Qt.AlignCenter)

        self._label = QLabel()
        self._label.setFixedSize(_THUMB_PX, _THUMB_PX)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "QLabel {"
            "  background: #252525;"
            "  color: #666;"
            "  font-size: 9px;"
            "  border: 1px solid #3a3a3a;"
            "}"
        )
        self._label.setText("…")
        lay.addWidget(self._label)
        self.setMouseTracking(True)
        self._label.setMouseTracking(True)

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Generate and display the thumbnail (idempotent)."""
        if self._loaded:
            return
        self._loaded = True
        pix = self._cache.thumbnail(self._match)
        if pix and not pix.isNull():
            self._label.setPixmap(pix)
            self._label.setText("")
        else:
            self._label.setText(t("overlap_preview_unavailable"))

    # ── Hover ─────────────────────────────────────────────────────────────────

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._popup.schedule(self._match, self._label)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._popup.cancel()
        super().leaveEvent(event)


# ── Main dialog ───────────────────────────────────────────────────────────────

class OverlappingUnknownFacesDialog(QDialog):
    """Review and select overlapping question-mark faces for deletion."""

    open_face_requested = Signal(int)

    def __init__(
        self,
        matches: Sequence[OverlappingUnknownFaceMatch],
        images_examined: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._matches = list(matches)
        self._delete_requested = False
        self._checkboxes: dict[int, QCheckBox] = {}
        self._thumbnail_cells: list[_ThumbnailCell] = []
        self._cache = _ThumbnailCache()
        self._popup = _HoverPopup(self._cache)

        self.setWindowTitle(t("overlap_dialog_title"))
        self.resize(1100, 540)
        self._build_ui(images_examined)

    # ── Public API ────────────────────────────────────────────────────────────

    def delete_requested(self) -> bool:
        return self._delete_requested

    def selected_unknown_face_ids(self) -> list[int]:
        return [
            match.unknown_face_id
            for match in self._matches
            if self._checkboxes[match.unknown_face_id].isChecked()
        ]

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self, images_examined: int) -> None:
        layout = QVBoxLayout(self)

        summary = QLabel(
            t("overlap_summary", images=images_examined, matches=len(self._matches))
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self._table = QTableWidget(len(self._matches), _NUM_COLS)
        self._table.setHorizontalHeaderLabels(
            [
                t("overlap_col_image"),
                t("overlap_col_preview"),
                t("overlap_col_person"),
                t("overlap_col_unknown_id"),
                t("overlap_col_known_id"),
                t("overlap_col_overlap"),
                t("overlap_col_delete"),
            ]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(_THUMB_PX + 8)
        self._table.itemDoubleClicked.connect(self._open_selected_row)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_IMAGE, QHeaderView.Stretch)
        header.setSectionResizeMode(_COL_PREVIEW, QHeaderView.Fixed)
        self._table.setColumnWidth(_COL_PREVIEW, _THUMB_PX + 12)
        for col in (_COL_PERSON, _COL_UNKNOWN_ID, _COL_KNOWN_ID, _COL_OVERLAP, _COL_DELETE):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

        for row, match in enumerate(self._matches):
            self._add_row(row, match)
        if self._matches:
            self._table.selectRow(0)

        # Lazy-load thumbnails as the user scrolls.
        self._table.verticalScrollBar().valueChanged.connect(
            self._load_visible_thumbnails
        )
        layout.addWidget(self._table)

        quick_row = QHBoxLayout()
        self._select_all_btn = QPushButton(t("overlap_select_all"))
        self._select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        quick_row.addWidget(self._select_all_btn)

        self._select_none_btn = QPushButton(t("overlap_select_none"))
        self._select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        quick_row.addWidget(self._select_none_btn)

        self._open_btn = QPushButton(t("overlap_open_image"))
        self._open_btn.clicked.connect(self._open_selected_row)
        quick_row.addWidget(self._open_btn)

        quick_row.addStretch()
        layout.addLayout(quick_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self._delete_btn = QPushButton(t("overlap_delete_selected"))
        self._delete_btn.setEnabled(bool(self._matches))
        self._delete_btn.clicked.connect(self._request_delete)
        buttons.addButton(self._delete_btn, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Load thumbnails for the initially visible rows once the event loop runs.
        QTimer.singleShot(0, self._load_visible_thumbnails)

    def _add_row(self, row: int, match: OverlappingUnknownFaceMatch) -> None:
        path_item = QTableWidgetItem(match.display_path)
        path_item.setToolTip(match.image_path)
        self._table.setItem(row, _COL_IMAGE, path_item)

        cell = _ThumbnailCell(match, self._cache, self._popup)
        self._thumbnail_cells.append(cell)
        self._table.setCellWidget(row, _COL_PREVIEW, cell)

        self._table.setItem(row, _COL_PERSON, QTableWidgetItem(match.known_person_name))
        self._table.setItem(row, _COL_UNKNOWN_ID, QTableWidgetItem(str(match.unknown_face_id)))
        self._table.setItem(row, _COL_KNOWN_ID, QTableWidgetItem(str(match.known_face_id)))
        self._table.setItem(row, _COL_OVERLAP, QTableWidgetItem(f"{match.overlap:.2f}"))

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.setToolTip(t("overlap_delete_checkbox_tip"))
        wrapper = QWidget()
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(checkbox, alignment=Qt.AlignCenter)
        self._table.setCellWidget(row, _COL_DELETE, wrapper)
        self._checkboxes[match.unknown_face_id] = checkbox

    # ── Lazy thumbnail loading ────────────────────────────────────────────────

    def _load_visible_thumbnails(self) -> None:
        viewport = self._table.viewport()
        viewport_rect = viewport.rect()
        for row, cell in enumerate(self._thumbnail_cells):
            idx = self._table.model().index(row, _COL_PREVIEW)
            if viewport_rect.intersects(self._table.visualRect(idx)):
                cell.load()

    # ── Interaction ───────────────────────────────────────────────────────────

    def _selected_match(self) -> OverlappingUnknownFaceMatch | None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._matches):
            return None
        return self._matches[row]

    def _open_selected_row(self, *_args) -> None:
        match = self._selected_match()
        if match is not None:
            self.open_face_requested.emit(match.known_face_id)

    def _set_all_checked(self, checked: bool) -> None:
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(checked)

    def _request_delete(self) -> None:
        if not self.selected_unknown_face_ids():
            QMessageBox.information(
                self,
                t("overlap_no_selection_title"),
                t("overlap_no_selection_msg"),
            )
            return
        self._delete_requested = True
        self.accept()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._popup.cancel()
        super().closeEvent(event)
