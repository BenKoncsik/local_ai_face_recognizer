"""Family-tree graph construction and lineage queries.

Builds an in-memory family graph from the normalized ``relationships`` table
and exposes tree-rooted views, generation levels, lowest-common-ancestor and
lineage-path queries, plus a JSON renderer for export.

Relationship *semantics* — what counts as a spouse / parent / sibling and the
cycle prevention applied on write — live in :class:`FamilyService`. This layer
adds the graph/tree *structure* on top of the already-materialised relationship
rows, and never mutates relationships itself (edits go through FamilyService).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Literal, Optional

from sqlalchemy.orm import Session

from app.db.models import FamilyTreeSettings, Person, Relationship
from app.services.family_service import (
    REL_PARENT_CHILD,
    REL_SPOUSE,
    FamilyService,
)

# How the second person relates to the first in a LineagePath.
LineageKind = Literal[
    "self",
    "spouse",
    "ancestor",     # to_id is an ancestor of from_id
    "descendant",   # to_id is a descendant of from_id
    "sibling",
    "cousin",       # shares an ancestor, neither direct line nor sibling
    "unrelated",
]


@dataclass
class FamilyTreeNode:
    """A single person in a built family tree.

    ``generation`` is relative to the tree root: 0 is the root's generation,
    ancestors are negative, descendants positive. Edge lists are restricted to
    persons that are also present in the same :class:`FamilyTreeGraph`.
    """

    person_id: int
    name: str
    generation: int
    parent_ids: tuple[int, ...] = ()
    child_ids: tuple[int, ...] = ()
    spouse_ids: tuple[int, ...] = ()
    is_member: bool = True
    gender: Optional[str] = None
    family_code: Optional[str] = None
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    thumbnail_path: Optional[str] = None


@dataclass
class FamilyTreeGraph:
    """A rooted family tree: a connected set of nodes keyed by person id."""

    root_id: Optional[int]
    nodes: dict[int, FamilyTreeNode] = field(default_factory=dict)
    # Marriage (start, end, place) per spouse pair, keyed by the sorted id pair.
    # Drives the label rendered on each couple's connecting line.
    marriages: dict[tuple[int, int], tuple[Optional[str], Optional[str], Optional[str]]] = field(
        default_factory=dict
    )

    def marriage(
        self, person_id_a: int, person_id_b: int
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (start, end, place) for a couple, or (None, None, None)."""
        return self.marriages.get(tuple(sorted((person_id_a, person_id_b))), (None, None, None))

    def __contains__(self, person_id: int) -> bool:
        return person_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def node(self, person_id: int) -> Optional[FamilyTreeNode]:
        return self.nodes.get(person_id)

    def ordered(self) -> list[FamilyTreeNode]:
        """Nodes sorted by generation then id (stable for rendering/export)."""
        return sorted(self.nodes.values(), key=lambda n: (n.generation, n.person_id))


def infer_coparent_links(
    parents: dict[int, set[int]],
    spouses: dict[int, set[int]],
    included: set[int],
) -> list[tuple[int, int]]:
    """Infer ``(coparent_id, child_id)`` edges so couples render as joint parents.

    A child linked to only **one** parent of an unambiguous couple is shown as a
    child of **both** spouses. This is what makes a couple render correctly: all
    their children share a single sibling bus connecting up to the pair, and
    selecting any child highlights *both* parents (and their lines) as ancestors.

    The inference fires only when the child has exactly one recorded parent and
    that parent has exactly one spouse within ``included``. Blended families and
    remarriages are disambiguated by explicit dual-parent links (two recorded
    parents) or braced family codes, both of which already materialise both real
    parents — those cases are left untouched, so no step-parent is mis-attributed.

    Pure function (no DB / session): the caller adds each returned edge to both
    the parent→child and child→parent maps. Shared by the in-app tree and the
    web export so both lay couples out identically.
    """
    extra: list[tuple[int, int]] = []
    for child_id in included:
        recorded = [p for p in parents.get(child_id, ()) if p in included]
        if len(recorded) != 1:
            continue
        parent = recorded[0]
        parent_spouses = [s for s in spouses.get(parent, ()) if s in included]
        if len(parent_spouses) != 1:
            continue
        coparent = parent_spouses[0]
        if coparent == child_id or coparent in recorded:
            continue
        extra.append((coparent, child_id))
    return extra


@dataclass
class LineagePath:
    """The connection between two people through a common ancestor.

    ``path`` lists person ids from ``from_id`` up to the common ancestor and
    back down to ``to_id`` (empty when unrelated). ``up`` is the number of
    generations from ``from_id`` to the common ancestor, ``down`` from that
    ancestor to ``to_id``.
    """

    from_id: int
    to_id: int
    kind: LineageKind
    path: tuple[int, ...] = ()
    up: int = 0
    down: int = 0


class FamilyTreeService:
    """Read-side graph queries over the family relationship table."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._family = FamilyService(session)

    # ------------------------------------------------------------------
    # Tree / graph construction
    # ------------------------------------------------------------------

    def build_tree(
        self,
        root_id: int,
        *,
        include_non_members: bool = True,
        max_generations_up: Optional[int] = None,
        max_generations_down: Optional[int] = None,
    ) -> FamilyTreeGraph:
        """Build the family tree reachable from ``root_id``.

        Traverses the connected component around the root following parent
        (up), child (down) and spouse (same level) edges, assigning each person
        a generation relative to the root. The root is always included even if
        it is not a family-tree member.

        Args:
            include_non_members: when False, persons flagged
                ``is_family_tree_member == False`` are skipped and not traversed
                through (the root is still kept).
            max_generations_up / max_generations_down: optional pruning of how
                far above/below the root to walk (``None`` = unlimited).
        """
        self._require_person(root_id)
        parents, children, spouses = self._build_adjacency()
        members = self._member_flags()

        gen: dict[int, int] = {root_id: 0}
        queue: deque[int] = deque([root_id])

        def _allowed(person_id: int) -> bool:
            if person_id == root_id:
                return True
            if not include_non_members and not members.get(person_id, True):
                return False
            return True

        def _within_limits(generation: int) -> bool:
            if max_generations_up is not None and generation < -max_generations_up:
                return False
            if max_generations_down is not None and generation > max_generations_down:
                return False
            return True

        while queue:
            current = queue.popleft()
            g = gen[current]
            neighbours = (
                [(p, g - 1) for p in parents.get(current, ())]
                + [(c, g + 1) for c in children.get(current, ())]
                + [(s, g) for s in spouses.get(current, ())]
            )
            for neighbour, ngen in neighbours:
                if neighbour in gen:
                    continue
                if not _allowed(neighbour) or not _within_limits(ngen):
                    continue
                gen[neighbour] = ngen
                queue.append(neighbour)

        return self._materialise_graph(root_id, gen, parents, children, spouses)

    def build_lineage(
        self,
        root_id: int,
        *,
        include_non_members: bool = True,
    ) -> FamilyTreeGraph:
        """Build only the direct line of ``root_id``.

        Includes the root, every ancestor (parents upward), every descendant
        (children downward), the root's spouse(s) and each descendant's
        spouse(s) so couples render together. Collateral relatives — siblings,
        cousins, aunts/uncles — are intentionally excluded.

        Generations are relative to the root: ancestors negative, descendants
        positive, spouses share their partner's band.
        """
        self._require_person(root_id)
        parents, children, spouses = self._build_adjacency()
        members = self._member_flags()

        def _allowed(person_id: int) -> bool:
            if person_id == root_id:
                return True
            return include_non_members or members.get(person_id, True)

        gen: dict[int, int] = {root_id: 0}

        # Ancestors (upward).
        queue: deque[int] = deque([root_id])
        while queue:
            current = queue.popleft()
            for parent in parents.get(current, ()):
                if parent not in gen and _allowed(parent):
                    gen[parent] = gen[current] - 1
                    queue.append(parent)

        # Descendants (downward). Track them so we add their spouses below.
        descendants: set[int] = set()
        queue = deque([root_id])
        while queue:
            current = queue.popleft()
            for child in children.get(current, ()):
                if child not in gen and _allowed(child):
                    gen[child] = gen[current] + 1
                    descendants.add(child)
                    queue.append(child)

        # Spouses of the root and of every descendant (to show couples).
        for pid in [root_id, *descendants]:
            for spouse in spouses.get(pid, ()):
                if spouse not in gen and _allowed(spouse):
                    gen[spouse] = gen[pid]

        return self._materialise_graph(root_id, gen, parents, children, spouses)

    def build_graph(self, *, include_non_members: bool = True) -> FamilyTreeGraph:
        """Build the full forest of every person that has a relationship.

        Generations are assigned across the whole forest by longest path
        (a child sits one band below its lowest parent, spouses share a band),
        so the result can be laid out in generation bands without a single root.
        """
        parents, children, spouses = self._build_adjacency()
        members = self._member_flags()
        ids = set(parents) | set(children) | set(spouses)
        if not include_non_members:
            ids = {pid for pid in ids if members.get(pid, True)}
        gen = self._assign_generations(ids, parents, spouses)
        return self._materialise_graph(None, gen, parents, children, spouses)

    @staticmethod
    def _assign_generations(
        ids: set[int],
        parents: dict[int, set[int]],
        spouses: dict[int, set[int]],
    ) -> dict[int, int]:
        """Generation per node: child = max(parent)+1, spouses aligned.

        Uses bounded relaxation (the parent graph is acyclic, so it converges).
        """
        gen = {pid: 0 for pid in ids}
        for _ in range(len(ids) + 1):
            changed = False
            for pid in ids:
                for parent in parents.get(pid, ()):
                    if parent in gen and gen[pid] < gen[parent] + 1:
                        gen[pid] = gen[parent] + 1
                        changed = True
            for pid in ids:
                for spouse in spouses.get(pid, ()):
                    if spouse in gen:
                        top = max(gen[pid], gen[spouse])
                        if gen[pid] != top or gen[spouse] != top:
                            gen[pid] = gen[spouse] = top
                            changed = True
            if not changed:
                break
        return gen

    # ------------------------------------------------------------------
    # Lineage queries
    # ------------------------------------------------------------------

    def lowest_common_ancestor(self, person_id_a: int, person_id_b: int) -> Optional[int]:
        """Return the closest shared ancestor of two people, or ``None``.

        A person is considered an ancestor of itself, so if one person is an
        ancestor of the other the result is that ancestor.
        """
        parents, _, _ = self._build_adjacency()
        depth_a = self._ancestor_depths(person_id_a, parents)
        depth_b = self._ancestor_depths(person_id_b, parents)
        common = set(depth_a) & set(depth_b)
        if not common:
            return None
        return min(common, key=lambda pid: depth_a[pid] + depth_b[pid])

    def relationship_path(self, person_id_a: int, person_id_b: int) -> LineagePath:
        """Classify how ``person_id_b`` relates to ``person_id_a``."""
        if person_id_a == person_id_b:
            return LineagePath(person_id_a, person_id_b, "self", (person_id_a,), 0, 0)

        parents, _, spouses = self._build_adjacency()
        if person_id_b in spouses.get(person_id_a, set()):
            return LineagePath(
                person_id_a, person_id_b, "spouse", (person_id_a, person_id_b), 0, 0
            )

        depth_a = self._ancestor_depths(person_id_a, parents)
        depth_b = self._ancestor_depths(person_id_b, parents)
        common = set(depth_a) & set(depth_b)
        if not common:
            return LineagePath(person_id_a, person_id_b, "unrelated")

        lca = min(common, key=lambda pid: depth_a[pid] + depth_b[pid])
        up, down = depth_a[lca], depth_b[lca]

        if lca == person_id_b:
            kind: LineageKind = "ancestor"
        elif lca == person_id_a:
            kind = "descendant"
        elif up == 1 and down == 1:
            kind = "sibling"
        else:
            kind = "cousin"

        up_path = self._path_to_ancestor(person_id_a, lca, parents) or []
        down_path = self._path_to_ancestor(person_id_b, lca, parents) or []
        # a .. lca  +  (lca .. b without the duplicated lca)
        full = tuple(up_path) + tuple(reversed(down_path[:-1]))
        return LineagePath(person_id_a, person_id_b, kind, full, up, down)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def render_tree_json(
        self,
        root_id: int,
        *,
        include_non_members: bool = True,
    ) -> dict:
        """Serialise the tree rooted at ``root_id`` to a plain-dict structure.

        Suitable for JSON export and for feeding a D3.js / Graphviz renderer.
        """
        graph = self.build_tree(root_id, include_non_members=include_non_members)
        return {
            "root_id": graph.root_id,
            "persons": [
                {
                    "id": node.person_id,
                    "name": node.name,
                    "generation": node.generation,
                    "gender": node.gender,
                    "family_code": node.family_code,
                    "birth_date": node.birth_date,
                    "death_date": node.death_date,
                    "thumbnail_path": node.thumbnail_path,
                    "is_member": node.is_member,
                    "parents": list(node.parent_ids),
                    "children": list(node.child_ids),
                    "spouses": list(node.spouse_ids),
                }
                for node in graph.ordered()
            ],
        }

    # ------------------------------------------------------------------
    # Settings / membership
    # ------------------------------------------------------------------

    def get_settings(self) -> FamilyTreeSettings:
        """Return the singleton settings row, creating it on first access."""
        settings = self._session.get(FamilyTreeSettings, 1)
        if settings is None:
            settings = FamilyTreeSettings(id=1)
            self._session.add(settings)
            self._session.flush()
        return settings

    def update_settings(self, **fields) -> FamilyTreeSettings:
        """Update settings fields (only known columns are applied)."""
        settings = self.get_settings()
        allowed = {
            "default_root_person_id",
            "layout",
            "include_non_members",
            "color_scheme",
        }
        for key, value in fields.items():
            if key in allowed:
                setattr(settings, key, value)
        self._session.flush()
        return settings

    def set_family_tree_member(self, person_id: int, is_member: bool) -> Person:
        """Toggle whether a person participates in the family tree."""
        person = self._require_person(person_id)
        person.is_family_tree_member = bool(is_member)
        self._session.flush()
        return person

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_adjacency(
        self,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]], dict[int, set[int]]]:
        """Load all relationships once into parent/child/spouse adjacency maps."""
        parents: dict[int, set[int]] = defaultdict(set)
        children: dict[int, set[int]] = defaultdict(set)
        spouses: dict[int, set[int]] = defaultdict(set)
        rows = self._session.query(
            Relationship.relationship_type,
            Relationship.person_a_id,
            Relationship.person_b_id,
        ).all()
        for rtype, a_id, b_id in rows:
            if rtype == REL_PARENT_CHILD:
                children[a_id].add(b_id)
                parents[b_id].add(a_id)
            elif rtype == REL_SPOUSE:
                spouses[a_id].add(b_id)
                spouses[b_id].add(a_id)
        return parents, children, spouses

    def _marriage_dates(
        self, included: set[int]
    ) -> dict[tuple[int, int], tuple[Optional[str], Optional[str], Optional[str]]]:
        """Map each spouse pair (within ``included``) to its (start, end, place)."""
        if not included:
            return {}
        rows = self._session.query(
            Relationship.person_a_id,
            Relationship.person_b_id,
            Relationship.start_date,
            Relationship.end_date,
            Relationship.marriage_place,
        ).filter(Relationship.relationship_type == REL_SPOUSE).all()
        out: dict[tuple[int, int], tuple[Optional[str], Optional[str], Optional[str]]] = {}
        for a_id, b_id, start, end, place in rows:
            if a_id in included and b_id in included:
                out[tuple(sorted((a_id, b_id)))] = (start, end, place)
        return out

    def _member_flags(self) -> dict[int, bool]:
        rows = self._session.query(
            Person.id, Person.is_family_tree_member
        ).all()
        return {pid: bool(flag) for pid, flag in rows}

    def _materialise_graph(
        self,
        root_id: Optional[int],
        generations: dict[int, int],
        parents: dict[int, set[int]],
        children: dict[int, set[int]],
        spouses: dict[int, set[int]],
    ) -> FamilyTreeGraph:
        included = set(generations)
        persons = {
            p.id: p
            for p in self._session.query(Person)
            .filter(Person.id.in_(included))
            .all()
        }
        crops = self._representative_crops(included)

        # Show couples as joint parents: a child linked to only one member of a
        # couple is treated as a child of both spouses (see infer_coparent_links).
        # Mutates fresh per-build adjacency only — the LCA / relationship_path
        # queries read their own _build_adjacency() and stay unaffected.
        for coparent, child in infer_coparent_links(parents, spouses, included):
            parents[child].add(coparent)
            children[coparent].add(child)

        def _present(ids: set[int]) -> tuple[int, ...]:
            return tuple(sorted(ids & included))

        graph = FamilyTreeGraph(
            root_id=root_id, marriages=self._marriage_dates(included)
        )
        for pid, generation in generations.items():
            person = persons.get(pid)
            if person is None:
                continue
            graph.nodes[pid] = FamilyTreeNode(
                person_id=pid,
                name=person.name,
                generation=generation,
                parent_ids=_present(parents.get(pid, set())),
                child_ids=_present(children.get(pid, set())),
                spouse_ids=_present(spouses.get(pid, set())),
                is_member=bool(person.is_family_tree_member),
                gender=person.gender,
                family_code=person.family_code,
                birth_date=person.birth_date,
                death_date=person.death_date,
                # Explicit thumbnail wins; otherwise fall back to the person's
                # best face crop so every box shows a photo when one exists.
                thumbnail_path=person.thumbnail_path or crops.get(pid),
            )
        return graph

    def _representative_crops(self, person_ids: set[int]) -> dict[int, str]:
        """Best (highest-confidence) face crop per person, for thumbnails."""
        if not person_ids:
            return {}
        from app.db.models import Face

        rows = (
            self._session.query(Face.person_id, Face.crop_path, Face.confidence)
            .filter(
                Face.person_id.in_(person_ids),
                Face.crop_path.isnot(None),
                Face.is_excluded == False,  # noqa: E712
            )
            .all()
        )
        best: dict[int, tuple[float, str]] = {}
        for pid, crop_path, confidence in rows:
            score = confidence or 0.0
            current = best.get(pid)
            if current is None or score > current[0]:
                best[pid] = (score, crop_path)
        return {pid: path for pid, (_, path) in best.items()}

    @staticmethod
    def _ancestor_depths(
        person_id: int, parents: dict[int, set[int]]
    ) -> dict[int, int]:
        """Map every ancestor (and the person itself, depth 0) to its depth."""
        depth = {person_id: 0}
        queue: deque[int] = deque([person_id])
        while queue:
            current = queue.popleft()
            for parent in parents.get(current, ()):
                if parent not in depth:
                    depth[parent] = depth[current] + 1
                    queue.append(parent)
        return depth

    @staticmethod
    def _path_to_ancestor(
        start_id: int, ancestor_id: int, parents: dict[int, set[int]]
    ) -> Optional[list[int]]:
        """Shortest id path from ``start_id`` up to ``ancestor_id`` (inclusive)."""
        if start_id == ancestor_id:
            return [start_id]
        prev: dict[int, Optional[int]] = {start_id: None}
        queue: deque[int] = deque([start_id])
        while queue:
            current = queue.popleft()
            if current == ancestor_id:
                break
            for parent in parents.get(current, ()):
                if parent not in prev:
                    prev[parent] = current
                    queue.append(parent)
        if ancestor_id not in prev:
            return None
        path: list[int] = []
        node: Optional[int] = ancestor_id
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def _require_person(self, person_id: int) -> Person:
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person {person_id} not found.")
        return person
