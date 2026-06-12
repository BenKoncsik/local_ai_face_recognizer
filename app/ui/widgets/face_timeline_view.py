"""Horizontal timeline of a person's faces, grouped and dated.

Each :class:`~app.services.face_grouping_service.FaceGroup` becomes a marker
positioned by its effective date.  The axis runs from the person's birth (if
known) to their death (if known), otherwise from the earliest to the latest
photo.  Marker styling encodes date precision per the plan:

* exact / year .......... normal thumbnail
* estimated ("kb.") ..... faded thumbnail + "kb." label
* interval / decade ..... translucent bar spanning the range, thumb at midpoint
* merged group .......... stacked badge with the member count

The view supports wheel-zoom (anchored under the cursor) and left-drag panning;
a plain click on a marker selects its representative face.  Markers that fall on
overlapping dates are stacked into collision-free rows.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
)

from app.services.face_date_service import FaceDateResolver
from app.ui.helpers.thumbnail_cache import get_thumbnail
from app.ui.i18n import t
from app.utils.fuzzy_date import FuzzyDate, Precision

log = logging.getLogger(__name__)

_THUMB = 48
_PX_PER_YEAR = 36.0
_LEFT_PAD = 40.0
_AXIS_Y = 26.0
_LANE_TOP = 48.0
_ROW_H = _THUMB + 22
_MIN_SCALE = 0.2
_MAX_SCALE = 5.0
_DRAG_THRESHOLD = 5


class _MarkerItem(QGraphicsPixmapItem):
    """A timeline thumbnail that remembers which face it represents."""

    def __init__(self, pixmap: QPixmap, rep_face_id: int) -> None:
        super().__init__(pixmap)
        self.rep_face_id = rep_face_id
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)


class FaceTimelineView(QGraphicsView):
    """Scrollable, zoomable timeline of dated face groups.

    Signals:
        face_selected: ``(rep_face_id: int)`` — a marker was clicked.
    """

    face_selected = Signal(int)

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor("#1e1e2e"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._resolver = FaceDateResolver()
        self._scale = 1.0
        self._panning = False
        self._press_pos = None
        self._press_item: Optional[_MarkerItem] = None
        self._start_year = 0.0
        self._end_year = 0.0

        # Fixed viewport overlay for the "N undated photos" note — kept out of
        # the scene so it never scrolls into / collides with the year axis.
        self._undated_label = QLabel(self.viewport())
        self._undated_label.setStyleSheet(
            "QLabel { color: #9399b2; background: rgba(30,30,46,200); "
            "border-radius: 4px; padding: 2px 6px; }"
        )
        self._undated_label.hide()

    # ------------------------------------------------------------------
    def set_data(
        self,
        groups: List,
        *,
        birth: Optional[FuzzyDate] = None,
        death: Optional[FuzzyDate] = None,
        resolver: Optional[FaceDateResolver] = None,
    ) -> None:
        """Render *groups* between the *birth* and *death* milestones."""
        self._resolver = resolver or FaceDateResolver()
        self._scene.clear()
        self.resetTransform()
        self._scale = 1.0

        dated = []
        undated = 0
        for g in groups:
            fd = g.date_span(self._resolver)
            if fd.span_frac is None:
                undated += 1
            else:
                dated.append((g, fd))

        bounds = self._compute_bounds(dated, birth, death)
        if bounds is None:
            self._show_empty()
            self._show_undated(undated)
            return
        self._start_year, self._end_year = bounds

        # Markers first so the axis/milestones can extend to the real content
        # height instead of an arbitrary fixed length.
        rows = self._draw_markers(dated)
        height = _LANE_TOP + max(1, rows) * _ROW_H + 6
        self._draw_axis(height)
        self._draw_milestones(birth, death, height)
        self._show_undated(undated)

        # Give the scene a little vertical breathing room below the lanes.
        rect = self._scene.itemsBoundingRect()
        rect.adjust(-_LEFT_PAD, -10, _LEFT_PAD, 20)
        self._scene.setSceneRect(rect)

    # ------------------------------------------------------------------
    def _compute_bounds(self, dated, birth, death):
        lo = hi = None
        if birth is not None and birth.span_frac is not None:
            lo = birth.span_frac[0]
        if death is not None and death.span_frac is not None:
            hi = death.span_frac[1]
        for _g, fd in dated:
            s = fd.span_frac
            lo = s[0] if lo is None else min(lo, s[0])
            hi = s[1] if hi is None else max(hi, s[1])
        if lo is None or hi is None:
            return None
        if hi - lo < 1.0:           # at least a one-year window
            hi = lo + 1.0
        return lo - 0.5, hi + 0.5

    def _x(self, frac_year: float) -> float:
        return _LEFT_PAD + (frac_year - self._start_year) * _PX_PER_YEAR

    def _tick_step(self) -> int:
        span = self._end_year - self._start_year
        if span <= 15:
            return 1
        if span <= 45:
            return 5
        if span <= 120:
            return 10
        return 25

    def _draw_axis(self, height: float) -> None:
        pen = QPen(QColor("#45475a"))
        x0, x1 = self._x(self._start_year), self._x(self._end_year)
        baseline = self._scene.addLine(x0, _AXIS_Y, x1, _AXIS_Y, QPen(QColor("#6c7086")))
        baseline.setZValue(-2)
        step = self._tick_step()
        first = int(self._start_year) - (int(self._start_year) % step)
        font = QFont()
        font.setPixelSize(11)
        year = first
        while year <= self._end_year + step:
            x = self._x(year)
            line = self._scene.addLine(x, _AXIS_Y - 4, x, height, pen)
            line.setZValue(-2)
            label = self._scene.addText(str(year), font)
            label.setDefaultTextColor(QColor("#bac2de"))
            label.setPos(x - 14, 2)
            label.setZValue(-1)
            year += step

    def _draw_milestones(self, birth, death, height: float) -> None:
        for fd, color, key in (
            (birth, "#a6e3a1", "timeline_born"),
            (death, "#f38ba8", "timeline_died"),
        ):
            if fd is None or fd.span_frac is None:
                continue
            x = self._x(fd.sort_key)
            line = self._scene.addLine(
                x, _AXIS_Y, x, height, QPen(QColor(color), 2, Qt.DashLine)
            )
            line.setZValue(-1)
            font = QFont()
            font.setPixelSize(11)
            font.setBold(True)
            label = self._scene.addText(t(key, date=fd.display), font)
            label.setDefaultTextColor(QColor(color))
            label.setPos(x + 3, _AXIS_Y - 2)

    def _draw_markers(self, dated) -> int:
        """Place markers in collision-free rows; return the number of rows used."""
        # Earliest first, then greedy collision-free row assignment.
        dated = sorted(dated, key=lambda gf: gf[1].sort_key)
        row_right: list[float] = []          # rightmost x used per row
        for group, fd in dated:
            span = fd.span_frac
            left = self._x(span[0])
            right = self._x(span[1])
            mid = self._x(fd.sort_key)
            is_bar = fd.precision in (Precision.RANGE, Precision.DECADE)
            marker_left = min(left, mid - _THUMB / 2)
            marker_right = max(right, mid + _THUMB / 2)

            row = 0
            while row < len(row_right) and row_right[row] > marker_left - 6:
                row += 1
            if row == len(row_right):
                row_right.append(marker_right)
            else:
                row_right[row] = marker_right
            y = _LANE_TOP + row * _ROW_H

            if is_bar:
                bar = self._scene.addRect(
                    QRectF(left, y + _THUMB / 2 - 5, max(2.0, right - left), 10),
                    QPen(Qt.NoPen),
                    QBrush(QColor(137, 180, 250, 90)),
                )
                bar.setZValue(0)

            pm = self._make_marker_pixmap(group, fd.is_estimated)
            item = _MarkerItem(pm, group.representative.id)
            item.setZValue(1)
            item.setPos(mid - pm.width() / 2, y)
            if fd.is_estimated:
                item.setOpacity(0.6)
            item.setToolTip(self._marker_tooltip(group, fd))
            self._scene.addItem(item)

            if fd.is_estimated:
                font = QFont()
                font.setPixelSize(9)
                est = self._scene.addText("kb.", font)
                est.setDefaultTextColor(QColor("#f9e2af"))
                est.setPos(mid - _THUMB / 2, y + _THUMB)

        return len(row_right)

    def _marker_tooltip(self, group, fd: FuzzyDate) -> str:
        if group.count > 1:
            return t("timeline_marker_group_tip", date=fd.display, n=group.count)
        return fd.display

    def _make_marker_pixmap(self, group, estimated: bool) -> QPixmap:
        rep = get_thumbnail(group.representative.crop_path, _THUMB)
        if rep is None:
            rep = QPixmap(_THUMB, _THUMB)
            rep.fill(QColor("#313244"))
        if group.count <= 1:
            return rep
        # Stacked look + count badge for merged groups.
        canvas = QPixmap(_THUMB + 6, _THUMB + 6)
        canvas.fill(Qt.transparent)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#45475a")))
        p.setBrush(QBrush(QColor(120, 120, 130)))
        p.drawRoundedRect(4, 4, _THUMB, _THUMB, 4, 4)
        p.drawPixmap(0, 0, rep)
        label = "99+" if group.count > 99 else str(group.count)
        d = 17
        p.setBrush(QBrush(QColor("#89b4fa")))
        p.setPen(Qt.NoPen)
        p.drawEllipse(canvas.width() - d - 1, 0, d, d)
        f = QFont()
        f.setPixelSize(10)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#11111b"))
        p.drawText(canvas.width() - d - 1, 0, d, d, Qt.AlignCenter, label)
        p.end()
        return canvas

    def _show_undated(self, count: int) -> None:
        """Update the fixed corner overlay noting how many photos are undated."""
        if count <= 0:
            self._undated_label.hide()
            return
        self._undated_label.setText(t("timeline_undated", n=count))
        self._undated_label.adjustSize()
        self._undated_label.show()
        self._reposition_undated()

    def _reposition_undated(self) -> None:
        m = 8
        vp = self.viewport()
        y = max(m, vp.height() - self._undated_label.height() - m)
        self._undated_label.move(m, y)

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._reposition_undated()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        self._reposition_undated()

    def _show_empty(self) -> None:
        font = QFont()
        font.setPixelSize(13)
        msg = self._scene.addText(t("timeline_empty"), font)
        msg.setDefaultTextColor(QColor("#9399b2"))
        msg.setPos(10, 10)
        self._scene.setSceneRect(0, 0, 320, 80)

    # ------------------------------------------------------------------
    # Interaction: wheel-zoom, drag-pan, click-select
    # ------------------------------------------------------------------
    def wheelEvent(self, event) -> None:  # noqa: ANN001
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        new_scale = self._scale * factor
        if not (_MIN_SCALE <= new_scale <= _MAX_SCALE):
            return
        self._scale = new_scale
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            self._press_pos = event.position().toPoint()
            item = self.itemAt(self._press_pos)
            self._press_item = item if isinstance(item, _MarkerItem) else None
            self._panning = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: ANN001
        if self._press_pos is not None and event.buttons() & Qt.LeftButton:
            pos = event.position().toPoint()
            if not self._panning:
                moved = (pos - self._press_pos).manhattanLength()
                if moved > _DRAG_THRESHOLD:
                    self._panning = True
            if self._panning:
                delta = pos - self._press_pos
                self._press_pos = pos
                self.horizontalScrollBar().setValue(
                    self.horizontalScrollBar().value() - delta.x()
                )
                self.verticalScrollBar().setValue(
                    self.verticalScrollBar().value() - delta.y()
                )
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.LeftButton:
            if not self._panning and self._press_item is not None:
                self.face_selected.emit(self._press_item.rep_face_id)
            self._press_pos = None
            self._press_item = None
            self._panning = False
        super().mouseReleaseEvent(event)
