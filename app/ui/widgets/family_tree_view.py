"""Graphical family-tree view.

Renders a :class:`FamilyTreeGraph` as a classic genealogy chart: persons are
boxes laid out in horizontal generation bands, coloured by sex (blue = male,
orange = female), connected by spouse lines and parent→child elbows. The
focused / selected person is outlined in red. This replaces the old collapsible
tree-list with the picture-style layout.

Layout is a barycentre row layout: each generation is a row, and within a row
nodes are ordered by the average column of their parents, so children sit under
their parents without overlaps. It is intentionally simple and overlap-free
rather than a full Sugiyama layout.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)

from app.services.family_tree_service import FamilyTreeGraph

# Colours (kept close to the reference chart).
_MALE_FILL = QColor("#bcdcf2")
_MALE_BORDER = QColor("#5b9bd5")
_FEMALE_FILL = QColor("#f6d08a")
_FEMALE_BORDER = QColor("#d99a2b")
_UNKNOWN_FILL = QColor("#dcdcdc")
_UNKNOWN_BORDER = QColor("#9a9a9a")
_SELECTED_BORDER = QColor("#e03030")    # the person clicked right now
_FOCUS_BORDER = QColor("#8e44ad")       # the focus/root person (purple)
_ANCESTOR_BORDER = QColor("#1f6fbf")    # selected person's ancestors
_DESCENDANT_BORDER = QColor("#2e9e4f")  # selected person's descendants
_SPOUSE_BORDER = QColor("#c0392b")      # selected person's spouses
_LINE = QColor("#7a7a7a")
_BAND_A = QColor("#ffffff")
_BAND_B = QColor("#faf0dd")
_TEXT = QColor("#222222")
_GEN_NUMBER = QColor("#c8a050")

_ROLE_PID = 0


class FamilyTreeView(QGraphicsView):
    """Scrollable, zoomable graphical family tree."""

    person_selected = Signal(int)
    person_activated = Signal(int)

    BOX_W = 168
    BOX_H = 86
    PHOTO = 56
    H_GAP = 26
    V_GAP = 64
    GUTTER = 44  # left column showing the generation number

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setBackgroundBrush(QBrush(_BAND_A))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Large-tree friendliness: index for fast viewport queries, partial
        # repaints, cached background bands.
        self._scene.setItemIndexMethod(QGraphicsScene.BspTreeIndex)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheBackground)

        self._node_items: dict[int, QGraphicsItemGroup] = {}
        self._rect_items: dict[int, QGraphicsRectItem] = {}
        self._node_borders: dict[int, QColor] = {}
        self._graph: Optional[FamilyTreeGraph] = None
        self._focus_id: Optional[int] = None
        self._selected_id: Optional[int] = None
        self._ancestors: set[int] = set()
        self._descendants: set[int] = set()
        self._spouses: set[int] = set()
        self._connector_items: list = []
        self._connector_groups: list = []

        # Photos are loaded lazily for boxes that scroll into view, so a tree
        # with thousands of people opens instantly. pid -> (group, pos, path).
        self._pending_photos: dict[int, tuple] = {}
        self._photo_timer = QTimer(self)
        self._photo_timer.setSingleShot(True)
        self._photo_timer.setInterval(80)
        self._photo_timer.timeout.connect(self._load_visible_photos)
        self.horizontalScrollBar().valueChanged.connect(self._schedule_photo_load)
        self.verticalScrollBar().valueChanged.connect(self._schedule_photo_load)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_graph(
        self,
        graph: FamilyTreeGraph,
        *,
        focus_id: Optional[int] = None,
        selected_id: Optional[int] = None,
    ) -> None:
        self._scene.clear()
        self._node_items = {}
        self._rect_items = {}
        self._node_borders = {}
        self._pending_photos = {}
        self._connector_items = []
        self._connector_groups = []
        self._graph = graph
        self._focus_id = focus_id
        self._selected_id = selected_id
        if not graph.nodes:
            self._scene.setSceneRect(0, 0, 1, 1)
            return

        # Highlight the selected person's direct line in distinct colours.
        self._ancestors, self._descendants, self._spouses = self._lineage_sets(graph, selected_id)

        h_gap = self._min_spouse_gap(graph)
        positions, total_w, gens = self._layout(graph, h_gap=h_gap)

        col_h = self.BOX_H + self.V_GAP
        # Generation bands (alternating background stripes) with a left gutter
        # carrying an increasing generation number (1 at the top).
        ordered_gens = sorted(gens)
        band_left = -h_gap - self.GUTTER
        band_width = total_w + 2 * h_gap + self.GUTTER
        for band_index, gen in enumerate(ordered_gens):
            y = band_index * col_h - self.V_GAP / 2
            band = self._scene.addRect(
                band_left, y, band_width, col_h,
                QPen(Qt.NoPen),
                QBrush(_BAND_A if band_index % 2 == 0 else _BAND_B),
            )
            band.setZValue(-10)

            number = QGraphicsSimpleTextItem(str(band_index + 1))
            font = number.font()
            font.setPointSize(16)
            font.setBold(True)
            number.setFont(font)
            number.setBrush(QBrush(_GEN_NUMBER))
            num_rect = number.boundingRect()
            number.setPos(
                band_left + (self.GUTTER - num_rect.width()) / 2,
                y + (col_h - num_rect.height()) / 2,
            )
            number.setZValue(-9)
            self._scene.addItem(number)

        # Connector lines first (under the boxes).
        self._draw_connectors(graph, positions, focus_id)
        self._update_line_colors()

        # Boxes.
        for pid, (x, y) in positions.items():
            node = graph.node(pid)
            if node is not None:
                self._add_box(node, x, y)

        margin = 40
        self._scene.setSceneRect(
            band_left - margin,
            -self.V_GAP,
            band_width + 2 * margin,
            len(ordered_gens) * col_h + margin,
        )

        # Bring the focus/selected person into view, else start at the top.
        anchor = selected_id if selected_id in graph.nodes else focus_id
        if anchor in self._node_items:
            self.centerOn(self._node_items[anchor])
        else:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())
            self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())
        self._schedule_photo_load()

    # ------------------------------------------------------------------
    # Lazy photo loading + zoom
    # ------------------------------------------------------------------

    def _schedule_photo_load(self) -> None:
        if self._pending_photos:
            self._photo_timer.start()

    # Below this many photos, load them all at once (small/medium trees);
    # above it, load only what is in/near the viewport so huge trees stay fast.
    EAGER_PHOTO_LIMIT = 250

    def _load_visible_photos(self, *, budget: int = 120) -> None:
        """Load pending photos — all of them when small, else viewport-near."""
        if not self._pending_photos:
            return

        if len(self._pending_photos) <= self.EAGER_PHOTO_LIMIT:
            targets = list(self._pending_photos.keys())
            budget = len(targets)  # no partial pass for small trees
        else:
            # Pad the viewport so photos are ready slightly before they scroll in.
            view_rect = self.viewport().rect().adjusted(-200, -200, 200, 200)
            scene_rect = self.mapToScene(view_rect).boundingRect()
            targets = [
                int(pid)
                for item in self._scene.items(scene_rect)
                if (pid := item.data(_ROLE_PID)) is not None
                and int(pid) in self._pending_photos
            ]

        loaded = 0
        for pid in targets:
            pending = self._pending_photos.pop(pid, None)
            if pending is None:
                continue
            group, pos, path = pending
            pixmap = self._load_photo(path)
            if pixmap is not None:
                pix = QGraphicsPixmapItem(pixmap)
                # addToGroup first so setPos is in group-local coords, not scene coords.
                group.addToGroup(pix)
                pix.setPos(pos[0], pos[1] + (self.PHOTO - pixmap.height()) / 2)
            loaded += 1
            if loaded >= budget:
                if self._pending_photos:
                    self._photo_timer.start()  # finish remaining next tick
                break

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_photo_load()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._schedule_photo_load()

    def zoom_in(self) -> None:
        self._zoom(1.2)

    def zoom_out(self) -> None:
        self._zoom(1 / 1.2)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._schedule_photo_load()

    def _zoom(self, factor: float) -> None:
        scale = self.transform().m11() * factor
        if scale < 0.05 or scale > 4.0:
            return
        self.scale(factor, factor)
        self._schedule_photo_load()

    def set_selected(self, selected_id: Optional[int]) -> None:
        """Re-colour borders and lines to mark the selection's relatives.

        Cheap update of existing boxes — no full re-layout.
        """
        if self._graph is None:
            return
        self._selected_id = selected_id
        self._ancestors, self._descendants, self._spouses = self._lineage_sets(
            self._graph, selected_id
        )
        for pid, rect in self._rect_items.items():
            rect.setPen(self._pen_for(pid, self._node_borders.get(pid, _LINE)))
        self._update_line_colors()

    def _pen_for(self, pid: int, border: QColor) -> QPen:
        if pid == self._selected_id:
            pen = QPen(_SELECTED_BORDER)
            pen.setWidth(3)
        elif pid == self._focus_id:
            pen = QPen(_FOCUS_BORDER)
            pen.setWidth(3)
        elif pid in self._ancestors:
            pen = QPen(_ANCESTOR_BORDER)
            pen.setWidth(3)
        elif pid in self._descendants:
            pen = QPen(_DESCENDANT_BORDER)
            pen.setWidth(3)
        elif pid in self._spouses:
            pen = QPen(_SPOUSE_BORDER)
            pen.setWidth(3)
        else:
            pen = QPen(border)
            pen.setWidth(1)
        return pen

    @staticmethod
    def _lineage_sets(graph, selected_id) -> tuple[set[int], set[int], set[int]]:
        """Transitive ancestors, descendants, and direct spouses of selected person."""
        if selected_id is None or selected_id not in graph.nodes:
            return set(), set(), set()

        def _walk(start: int, attr: str) -> set[int]:
            seen: set[int] = set()
            stack = [start]
            while stack:
                node = graph.node(stack.pop())
                if node is None:
                    continue
                for nxt in getattr(node, attr):
                    if nxt not in seen and nxt != selected_id:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        node = graph.node(selected_id)
        spouses = set(node.spouse_ids) if node else set()
        return _walk(selected_id, "parent_ids"), _walk(selected_id, "child_ids"), spouses

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _min_spouse_gap(self, graph: FamilyTreeGraph) -> int:
        """Minimum horizontal gap so the widest marriage label fits between boxes."""
        from PySide6.QtGui import QFont, QFontMetrics
        f = QFont()
        f.setPointSize(8)
        fm = QFontMetrics(f)
        seen: set[tuple[int, int]] = set()
        max_label_w = 0
        for pid, node in graph.nodes.items():
            for spouse in node.spouse_ids:
                key = tuple(sorted((pid, spouse)))
                if key in seen:
                    continue
                seen.add(key)
                label = _marriage_label(*graph.marriage(pid, spouse))
                if label:
                    max_label_w = max(max_label_w, fm.horizontalAdvance(label) + 20)
        return max(self.H_GAP, max_label_w)

    def _layout(self, graph: FamilyTreeGraph, h_gap: Optional[int] = None):
        """Return (positions{pid:(x,y)}, total_width, gens{gen:[nodes]})."""
        if h_gap is None:
            h_gap = self.H_GAP
        gens: dict[int, list] = defaultdict(list)
        for node in graph.nodes.values():
            gens[node.generation].append(node)

        col_of: dict[int, int] = {}
        sorted_gens = sorted(gens)
        for band_index, gen in enumerate(sorted_gens):
            nodes = gens[gen]
            # Base ordering key per node: the top band has no parents to anchor
            # to, so it sorts by name; lower bands sort by the barycentre of
            # their parents' columns so children sit under their parents.
            if band_index == 0:
                base_key = {n.person_id: (n.name.lower(), n.person_id) for n in nodes}
            else:
                def _bary(node):
                    cols = [col_of[p] for p in node.parent_ids if p in col_of]
                    return sum(cols) / len(cols) if cols else float("inf")

                base_key = {
                    n.person_id: (_bary(n), n.name.lower(), n.person_id) for n in nodes
                }

            nodes_sorted = self._order_generation(nodes, base_key)
            gens[gen] = nodes_sorted
            for column, node in enumerate(nodes_sorted):
                col_of[node.person_id] = column

        step_x = self.BOX_W + h_gap
        step_y = self.BOX_H + self.V_GAP
        positions: dict[int, tuple[float, float]] = {}
        for band_index, gen in enumerate(sorted_gens):
            for node in gens[gen]:
                positions[node.person_id] = (
                    col_of[node.person_id] * step_x,
                    band_index * step_y,
                )

        max_cols = max((len(gens[g]) for g in sorted_gens), default=1)
        total_w = max(max_cols, 1) * step_x - h_gap
        return positions, total_w, gens

    @staticmethod
    def _order_generation(nodes: list, base_key: dict) -> list:
        """Order one generation so married couples sit side by side.

        Barycentre sorting alone scatters spouses: a married-in partner with no
        parents in the tree gets an ``inf`` barycentre and lands at the far edge,
        so their spouse line stretches across everyone else's boxes and the
        marriage label ends up over unrelated people. Here we first group nodes
        into spouse-connected clusters (using only spouse links *within* this
        generation), lay each cluster out as a chain with partners adjacent, then
        order the clusters by their strongest (smallest) base key. Non-married
        people stay single-node clusters and keep their barycentre position.
        """
        ids_here = {n.person_id for n in nodes}
        node_by_id = {n.person_id: n for n in nodes}

        # Spouse adjacency restricted to this generation.
        adj: dict[int, list[int]] = {pid: [] for pid in ids_here}
        for n in nodes:
            for s in n.spouse_ids:
                if s in ids_here:
                    adj[n.person_id].append(s)

        # Connected components over the intra-generation spouse edges.
        seen: set[int] = set()
        clusters: list[list[int]] = []
        for pid in sorted(ids_here, key=lambda p: base_key[p]):
            if pid in seen:
                continue
            comp: list[int] = []
            stack = [pid]
            seen.add(pid)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            clusters.append(comp)

        ordered_clusters = []
        for comp in clusters:
            internal = FamilyTreeView._order_cluster(comp, adj, base_key)
            ordered_clusters.append((min(base_key[p] for p in comp), internal))
        ordered_clusters.sort(key=lambda t: t[0])

        return [
            node_by_id[pid]
            for _, internal in ordered_clusters
            for pid in internal
        ]

    @staticmethod
    def _order_cluster(comp: list, adj: dict, base_key: dict) -> list:
        """Order one spouse cluster as a chain with partners adjacent.

        Walks the spouse subgraph starting from a chain endpoint (a person with
        a single partner) so simple couples and remarriage chains keep every
        pair side by side; a star (one person with several partners) keeps as
        many pairs adjacent as a single row allows.
        """
        if len(comp) <= 1:
            return list(comp)
        comp_set = set(comp)
        endpoints = [p for p in comp if len([q for q in adj[p] if q in comp_set]) == 1]
        start = min(endpoints or comp, key=lambda p: base_key[p])

        order: list[int] = []
        visited: set[int] = set()
        cur: Optional[int] = start
        while cur is not None:
            order.append(cur)
            visited.add(cur)
            nxts = [q for q in adj[cur] if q in comp_set and q not in visited]
            cur = min(nxts, key=lambda q: base_key[q]) if nxts else None
        # Anything unreachable in a single walk (star/cycle leftovers) trails by key.
        leftovers = [p for p in comp if p not in visited]
        order.extend(sorted(leftovers, key=lambda p: base_key[p]))
        return order

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _center(self, x: float, y: float) -> tuple[float, float]:
        return x + self.BOX_W / 2, y + self.BOX_H / 2

    def _draw_connectors(self, graph, positions, focus_id) -> None:
        # _connector_items: list of dicts describing each drawn line + enough
        # metadata to recolour it for any selection (see _update_line_colors).
        # _connector_groups: per parent→children group context (parents, the
        # junction x where the drop meets the bus, and each child's column x).
        self._connector_items = []
        self._connector_groups = []

        # --- Spouse links (horizontal) ---
        drawn_spouse: set[tuple[int, int]] = set()
        for pid, node in graph.nodes.items():
            if pid not in positions:
                continue
            x, y = positions[pid]
            for spouse in node.spouse_ids:
                if spouse not in positions:
                    continue
                key = tuple(sorted((pid, spouse)))
                if key in drawn_spouse:
                    continue
                drawn_spouse.add(key)
                sx, sy = positions[spouse]
                if x <= sx:
                    lx, ly, rx, ry = x, y, sx, sy
                else:
                    lx, ly, rx, ry = sx, sy, x, y
                item = self._add_line_path([
                    (lx + self.BOX_W, ly + self.BOX_H / 2),
                    (rx, ry + self.BOX_H / 2),
                ])
                self._connector_items.append(
                    {"item": item, "kind": "spouse", "pair": frozenset([pid, spouse])}
                )

                # Marriage date label centred ON the horizontal spouse line.
                label = _marriage_label(*graph.marriage(pid, spouse))
                if label:
                    mid_x = (lx + self.BOX_W + rx) / 2
                    line_y = (ly + ry) / 2 + self.BOX_H / 2
                    self._add_edge_label(mid_x, line_y, label)

        # --- Parent → children: group siblings by their canonical couple so they
        #     all share one horizontal bus instead of each getting its own line.
        #
        # Problem: some children may only have ONE parent listed in the DB (e.g.
        # only linked to Anya, not to Apa), giving a different frozenset key than
        # siblings who have both parents. Fix: expand each child's parent set to
        # include their parents' spouses who are also parents in this graph.
        all_parents: set[int] = set()
        for _node in graph.nodes.values():
            for p in _node.parent_ids:
                if p in positions:
                    all_parents.add(p)

        children_by_parents: dict[frozenset, list[int]] = defaultdict(list)
        for pid, node in graph.nodes.items():
            if pid not in positions:
                continue
            parents_in_graph = frozenset(p for p in node.parent_ids if p in positions)
            if not parents_in_graph:
                continue
            # Expand: add each listed parent's spouse if that spouse is also a
            # parent somewhere in the graph, so partial-link siblings share a bus.
            couple_key: set[int] = set(parents_in_graph)
            for p in parents_in_graph:
                pnode = graph.node(p)
                if pnode:
                    for spouse in pnode.spouse_ids:
                        if spouse in all_parents:
                            couple_key.add(spouse)
            children_by_parents[frozenset(couple_key)].append(pid)

        for parent_ids_fs, children in children_by_parents.items():
            parents_in_pos = [p for p in parent_ids_fs if p in positions]
            if not parents_in_pos:
                continue

            parent_centers = [self._center(*positions[p]) for p in parents_in_pos]
            # Horizontal midpoint between the parents (or single parent).
            drop_x = sum(c[0] for c in parent_centers) / len(parent_centers)
            # Bottom edge of the parents' row.
            py = max(c[1] for c in parent_centers) + self.BOX_H / 2
            bus_y = py + self.V_GAP / 2

            children_sorted = sorted(children, key=lambda c: positions[c][0])
            child_x = {
                c: positions[c][0] + self.BOX_W / 2 for c in children_sorted
            }

            gidx = len(self._connector_groups)
            self._connector_groups.append({
                "parents": frozenset(parents_in_pos),
                "children": frozenset(children_sorted),
                "drop_x": drop_x,
                "child_x": child_x,
            })

            # A married couple already has its spouse line connecting the boxes,
            # so the children hang straight from that line's midpoint — no extra
            # bridge. A non-couple pair (rare: two parents not married to each
            # other) keeps a bottom-edge bridge so the drop has something to meet.
            is_couple = False
            if len(parents_in_pos) == 2:
                first, second = parents_in_pos
                first_node = graph.node(first)
                is_couple = first_node is not None and second in first_node.spouse_ids

            if is_couple:
                # Spouse line sits at the boxes' mid-height; drop from there.
                junction_top = min(c[1] for c in parent_centers)
            else:
                if len(parent_centers) == 2:
                    left_cx = min(c[0] for c in parent_centers)
                    right_cx = max(c[0] for c in parent_centers)
                    item = self._add_line_path([(left_cx, py), (right_cx, py)])
                    self._connector_items.append(
                        {"item": item, "kind": "group", "group": gidx, "role": "bridge"}
                    )
                junction_top = py

            # Vertical drop from the parents down to the bus level.
            item = self._add_line_path([(drop_x, junction_top), (drop_x, bus_y)])
            self._connector_items.append(
                {"item": item, "kind": "group", "group": gidx, "role": "junction"}
            )

            # Horizontal bus split into per-gap segments between the connection
            # points (each child's column + the junction). Segmenting lets a
            # selection highlight only the stretch from one child to the junction.
            points = sorted(set(child_x.values()) | {drop_x})
            for a, b in zip(points, points[1:]):
                item = self._add_line_path([(a, bus_y), (b, bus_y)])
                self._connector_items.append(
                    {"item": item, "kind": "group", "group": gidx,
                     "role": "bus", "a": a, "b": b}
                )

            # Individual vertical drops from the bus to each child box.
            for child_id in children_sorted:
                cx = child_x[child_id]
                cy = positions[child_id][1]
                item = self._add_line_path([(cx, bus_y), (cx, cy)])
                self._connector_items.append(
                    {"item": item, "kind": "group", "group": gidx,
                     "role": "childdrop", "child": child_id}
                )

    def _add_edge_label(self, cx: float, cy: float, text: str) -> None:
        """Draw a small white-backed text tag centred on (cx, cy).

        Used for the marriage-date label placed on the spouse connecting line.
        The white background keeps the label legible over any line that crosses it.
        """
        label = QGraphicsSimpleTextItem(text)
        font = label.font()
        font.setPointSize(8)
        label.setFont(font)
        label.setBrush(QBrush(_TEXT))
        rect = label.boundingRect()
        pad = 2.5
        top = cy - rect.height() / 2 - pad
        bg = QGraphicsRectItem(
            cx - rect.width() / 2 - pad, top,
            rect.width() + 2 * pad, rect.height() + 2 * pad,
        )
        bg.setBrush(QBrush(QColor("#ffffff")))
        bg.setPen(QPen(_LINE, 0))
        bg.setZValue(2)
        self._scene.addItem(bg)
        label.setPos(cx - rect.width() / 2, cy - rect.height() / 2)
        label.setZValue(3)
        self._scene.addItem(label)

    def _add_line_path(self, points: list[tuple[float, float]]) -> QGraphicsPathItem:
        path = QPainterPath()
        path.moveTo(*points[0])
        for pt in points[1:]:
            path.lineTo(*pt)
        item = QGraphicsPathItem(path)
        pen = QPen(_LINE)
        pen.setWidth(2)
        item.setPen(pen)
        item.setZValue(-5)
        self._scene.addItem(item)
        return item

    def _update_line_colors(self) -> None:
        """Recolour connectors to trace the selected person's direct lineage.

        For the selected person S we light up exactly the path that leads from S
        up to its ancestors (blue) and down to its descendants (green); spouse
        links of S go red. Sibling branches that merely share a bus with S stay
        grey, so the highlight reaches only where it should.
        """
        sel = self._selected_id
        anc = self._ancestors
        desc = self._descendants
        up_or_self = (anc | {sel}) if sel is not None else set()
        down_or_self = (desc | {sel}) if sel is not None else set()

        # Per-group relevance: is this group on S's ancestor line (and through
        # which child), or on S's descendant line (whole group lights up)?
        group_state = []
        for g in self._connector_groups:
            parents = g["parents"]
            up_child = None
            if sel is not None and parents and parents <= anc:
                for c in g["children"]:
                    if c in up_or_self:
                        up_child = c
                        break
            down = sel is not None and bool(parents & down_or_self)
            group_state.append((up_child, down))

        for it in self._connector_items:
            color = _LINE
            if it["kind"] == "spouse":
                if sel is not None and sel in it["pair"]:
                    color = _SPOUSE_BORDER
            else:
                g = it["group"]
                up_child, down = group_state[g]
                if down:
                    color = _DESCENDANT_BORDER
                elif up_child is not None:
                    color = self._up_color_for(it, g, up_child)
            pen = QPen(color)
            pen.setWidth(3 if color is not _LINE else 2)
            it["item"].setPen(pen)

    def _up_color_for(self, it: dict, g: int, up_child: int) -> QColor:
        """Colour for a group connector when the group is on S's ancestor line.

        Only the single stretch from ``up_child`` to the parents' junction is
        lit; the other siblings' drops and bus segments stay grey.
        """
        role = it["role"]
        if role in ("bridge", "junction"):
            return _ANCESTOR_BORDER
        if role == "childdrop":
            return _ANCESTOR_BORDER if it["child"] == up_child else _LINE
        if role == "bus":
            grp = self._connector_groups[g]
            cx = grp["child_x"].get(up_child)
            if cx is None:
                return _LINE
            lo, hi = min(cx, grp["drop_x"]), max(cx, grp["drop_x"])
            # Segment [a, b] is on the up_child→junction stretch.
            if it["a"] >= lo - 0.5 and it["b"] <= hi + 0.5:
                return _ANCESTOR_BORDER
        return _LINE

    def _add_box(self, node, x, y) -> None:
        if node.gender == "male":
            fill, border = _MALE_FILL, _MALE_BORDER
        elif node.gender == "female":
            fill, border = _FEMALE_FILL, _FEMALE_BORDER
        else:
            fill, border = _UNKNOWN_FILL, _UNKNOWN_BORDER

        group = QGraphicsItemGroup()
        group.setData(_ROLE_PID, node.person_id)

        pid = node.person_id
        rect = QGraphicsRectItem(0, 0, self.BOX_W, self.BOX_H)
        rect.setBrush(QBrush(fill))
        rect.setPen(self._pen_for(pid, border))
        if not node.is_member:
            rect.setOpacity(0.45)
        group.addToGroup(rect)
        self._rect_items[pid] = rect
        self._node_borders[pid] = border

        text_x = 6.0
        if node.thumbnail_path:
            # Reserve the photo slot now; the image itself is loaded lazily when
            # the box scrolls into view (keeps large trees fast to open).
            slot = QGraphicsRectItem(6, (self.BOX_H - self.PHOTO) / 2, self.PHOTO, self.PHOTO)
            slot.setBrush(QBrush(QColor("#cfcfcf")))
            slot.setPen(QPen(Qt.NoPen))
            group.addToGroup(slot)
            text_x = 6 + self.PHOTO + 6
            self._pending_photos[pid] = (group, (6.0, (self.BOX_H - self.PHOTO) / 2), node.thumbnail_path)

        lines = [node.name]
        years = _years(node.birth_date, node.death_date)
        if years:
            lines.append(years)
        if node.family_code:
            lines.append(node.family_code)

        ty = 8.0
        for i, line in enumerate(lines):
            text = QGraphicsSimpleTextItem(_elide(line, 22))
            font = text.font()
            font.setPointSize(9 if i == 0 else 8)
            font.setBold(i == 0)
            text.setFont(font)
            text.setBrush(QBrush(_TEXT))
            text.setPos(text_x, ty)
            group.addToGroup(text)
            ty += 18 if i == 0 else 15

        group.setPos(x, y)
        group.setZValue(1)
        self._scene.addItem(group)
        self._node_items[node.person_id] = group

    def _load_photo(self, path: Optional[str]) -> Optional[QPixmap]:
        if not path or not Path(path).exists():
            return None
        from app.utils.image_utils import load_pixmap_exif

        pixmap = load_pixmap_exif(path)
        if pixmap.isNull():
            return None
        return pixmap.scaled(
            self.PHOTO, self.PHOTO, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _pid_at(self, view_pos) -> Optional[int]:
        # Explicit bounding-box check is more reliable than itemAt() with
        # BspTreeIndex + QGraphicsItemGroup, which can return a neighbour's item.
        scene_pos = self.mapToScene(view_pos)
        sx, sy = scene_pos.x(), scene_pos.y()
        for pid, group in self._node_items.items():
            gx, gy = group.x(), group.y()
            if gx <= sx <= gx + self.BOX_W and gy <= sy <= gy + self.BOX_H:
                return pid
        return None

    def mousePressEvent(self, event) -> None:
        pid = self._pid_at(event.position().toPoint())
        if pid is not None:
            self.person_selected.emit(pid)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        pid = self._pid_at(event.position().toPoint())
        if pid is not None:
            self.person_activated.emit(pid)
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:
        # Wheel zooms toward the cursor; drag (or scrollbars) pans. Shift+wheel
        # scrolls horizontally for users who prefer that.
        if event.modifiers() & Qt.ShiftModifier:
            super().wheelEvent(event)
            return
        self._zoom(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.zoom_out()
        elif key == Qt.Key_0:
            self.reset_zoom()
        else:
            super().keyPressEvent(event)


def _marriage_label(
    start: Optional[str], end: Optional[str], place: Optional[str] = None
) -> str:
    """Compact marriage label for the spouse line: '1984–1990 · Budapest'."""
    start = (start or "").strip()
    end = (end or "").strip()
    place = (place or "").strip()
    if start and end:
        date_part = f"{start}–{end}"
    elif start:
        date_part = start
    elif end:
        date_part = f"–{end}"
    else:
        date_part = ""
    if date_part and place:
        return f"{date_part} · {place}"
    return date_part or place


def _years(birth: Optional[str], death: Optional[str]) -> str:
    if not birth and not death:
        return ""
    return f"{birth or ''}–{death or ''}"


def _elide(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
