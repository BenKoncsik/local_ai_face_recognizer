"""Tests for FamilyTreeService graph build, lineage queries and settings."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import FamilyTreeSettings, Person
from app.services.family_service import FamilyService
from app.services.family_tree_service import FamilyTreeService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "tree.db")


def _person(session, name: str, gender: str | None = None) -> Person:
    person = Person(name=name, is_auto_named=False, gender=gender)
    session.add(person)
    session.flush()
    return person


def _build_family(session) -> dict[str, int]:
    """A three-generation family.

        GpA == GpB                (grandparents, spouses)
          |                       both parents of P1 and P2
        +-+--------+
        P1 == Sp   P2             P1 married to Sp
        |          |
      +-+-+        |
      C1  C2     Cousin           C1/C2 siblings; Cousin is their cousin
    """
    fam = FamilyService(session)
    p = {name: _person(session, name).id for name in
         ("GpA", "GpB", "P1", "P2", "Sp", "C1", "C2", "Cousin")}

    fam.add_spouse(p["GpA"], p["GpB"])
    for child in ("P1", "P2"):
        fam.add_parent_child(p["GpA"], p[child])
        fam.add_parent_child(p["GpB"], p[child])
    fam.add_spouse(p["P1"], p["Sp"])
    for child in ("C1", "C2"):
        fam.add_parent_child(p["P1"], p[child])
        fam.add_parent_child(p["Sp"], p[child])
    fam.add_parent_child(p["P2"], p["Cousin"])
    session.flush()
    return p


# ── build_tree ────────────────────────────────────────────────────────────────

def test_build_tree_assigns_generations(db):
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_tree(p["C1"])

        # Whole connected component is reachable from C1.
        assert set(graph.nodes) == set(p.values())

        assert graph.root_id == p["C1"]
        assert graph.node(p["C1"]).generation == 0
        assert graph.node(p["C2"]).generation == 0
        assert graph.node(p["P1"]).generation == -1
        assert graph.node(p["Sp"]).generation == -1
        assert graph.node(p["P2"]).generation == -1   # sibling of P1
        assert graph.node(p["GpA"]).generation == -2
        assert graph.node(p["GpB"]).generation == -2
        assert graph.node(p["Cousin"]).generation == 0  # P2's child


def test_build_tree_edges_reference_included_nodes(db):
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_tree(p["C1"])

        c1 = graph.node(p["C1"])
        assert set(c1.parent_ids) == {p["P1"], p["Sp"]}
        assert set(graph.node(p["P1"]).child_ids) == {p["C1"], p["C2"]}
        assert set(graph.node(p["P1"]).spouse_ids) == {p["Sp"]}


def test_build_tree_generation_limits(db):
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_tree(
            p["C1"], max_generations_up=1, max_generations_down=0
        )
        # Grandparents (gen -2) pruned; parents (gen -1) kept.
        assert p["GpA"] not in graph
        assert p["GpB"] not in graph
        assert p["P1"] in graph


def test_build_graph_assigns_forest_generations(db):
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_graph()
        # Grandparents at band 0, parents at 1, youngest at 2.
        assert graph.node(p["GpA"]).generation == 0
        assert graph.node(p["GpB"]).generation == 0
        assert graph.node(p["P1"]).generation == 1
        assert graph.node(p["P2"]).generation == 1
        assert graph.node(p["Sp"]).generation == 1   # spouse aligned to P1
        assert graph.node(p["C1"]).generation == 2
        assert graph.node(p["Cousin"]).generation == 2


def test_build_lineage_excludes_collateral_relatives(db):
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_lineage(p["P1"])
        present = set(graph.nodes)
        # Direct line + spouse only.
        assert p["GpA"] in present and p["GpB"] in present   # ancestors
        assert p["C1"] in present and p["C2"] in present      # descendants
        assert p["Sp"] in present                             # root's spouse
        # Collateral relatives excluded.
        assert p["P2"] not in present                         # sibling
        assert p["Cousin"] not in present                     # sibling's child


def test_node_thumbnail_falls_back_to_face_crop(db):
    from app.db.models import Face, Image

    with session_scope() as session:
        p = _build_family(session)
        image = Image(file_path="/tmp/p.jpg", file_hash="h", file_mtime=1.0)
        session.add(image)
        session.flush()
        session.add(
            Face(
                image_id=image.id,
                person_id=p["C1"],
                bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1,
                confidence=0.9, detector_backend="manual",
                crop_path="/tmp/c1_crop.jpg",
            )
        )
        session.flush()
        graph = FamilyTreeService(session).build_graph()
        assert graph.node(p["C1"]).thumbnail_path == "/tmp/c1_crop.jpg"


def test_coparent_inferred_for_single_parent_linked_children(db):
    """A couple's children linked to only one parent still render joint-parented.

    Mirrors the bug report: codes derive each child under one parent (e.g. C86
    under C8/Anya), so only Anya is a recorded parent. The other spouse (Apa)
    must be inferred as a co-parent so both parents show as ancestors and all
    siblings share one bus.
    """
    with session_scope() as session:
        fam = FamilyService(session)
        ids = {
            "mom": _person(session, "Anya", "female").id,
            "dad": _person(session, "Apa", "male").id,
            "k1": _person(session, "Benedek", "male").id,
            "k2": _person(session, "Matyi", "male").id,
        }
        fam.add_spouse(ids["mom"], ids["dad"])
        # Each child linked to ONE parent only (the bug-triggering case).
        fam.add_parent_child(ids["mom"], ids["k1"])
        fam.add_parent_child(ids["mom"], ids["k2"])
        session.flush()

        graph = FamilyTreeService(session).build_graph()

        # Both parents are now recorded for every child …
        for kid in ("k1", "k2"):
            assert set(graph.node(ids[kid]).parent_ids) == {ids["mom"], ids["dad"]}
        # … and the inferred co-parent gained both children.
        assert set(graph.node(ids["dad"]).child_ids) == {ids["k1"], ids["k2"]}


def test_coparent_not_inferred_when_two_real_parents(db):
    """Children already linked to both parents are left exactly as-is."""
    with session_scope() as session:
        p = _build_family(session)
        graph = FamilyTreeService(session).build_graph()
        # C1 has both P1 and Sp recorded — no third parent invented.
        assert set(graph.node(p["C1"]).parent_ids) == {p["P1"], p["Sp"]}


def test_coparent_not_inferred_with_ambiguous_remarriage(db):
    """A parent with two spouses is ambiguous → no co-parent is guessed."""
    with session_scope() as session:
        fam = FamilyService(session)
        ids = {
            "mom": _person(session, "Mom", "female").id,
            "dad1": _person(session, "Dad1", "male").id,
            "dad2": _person(session, "Dad2", "male").id,
            "kid": _person(session, "Kid", "male").id,
        }
        fam.add_spouse(ids["mom"], ids["dad1"])
        fam.add_spouse(ids["mom"], ids["dad2"])
        fam.add_parent_child(ids["mom"], ids["kid"])
        session.flush()

        graph = FamilyTreeService(session).build_graph()
        # Two possible co-parents → can't disambiguate → only the real one stays.
        assert set(graph.node(ids["kid"]).parent_ids) == {ids["mom"]}


def test_graph_carries_marriage_dates(db):
    """Spouse start/end dates flow into the graph for the date-on-line label."""
    with session_scope() as session:
        fam = FamilyService(session)
        a = _person(session, "A", "male").id
        b = _person(session, "B", "female").id
        fam.add_spouse(a, b, start_date="1984", end_date="1990")
        session.flush()

        graph = FamilyTreeService(session).build_graph()
        assert graph.marriage(a, b) == ("1984", "1990", None)
        assert graph.marriage(b, a) == ("1984", "1990", None)  # order-independent


def test_lone_person_tree(db):
    with session_scope() as session:
        loner = _person(session, "Loner")
        graph = FamilyTreeService(session).build_tree(loner.id)
        assert set(graph.nodes) == {loner.id}
        assert graph.node(loner.id).generation == 0


# ── non-member filtering ────────────────────────────────────────────────────────

def test_non_members_excluded_when_disabled(db):
    with session_scope() as session:
        p = _build_family(session)
        svc = FamilyTreeService(session)
        svc.set_family_tree_member(p["Sp"], False)

        graph = svc.build_tree(p["C1"], include_non_members=False)
        assert p["Sp"] not in graph
        # Spouse edge no longer references the excluded node.
        assert p["Sp"] not in graph.node(p["P1"]).spouse_ids
        # Rest of the family remains reachable.
        assert p["GpA"] in graph

        with_members = svc.build_tree(p["C1"], include_non_members=True)
        assert p["Sp"] in with_members
        assert with_members.node(p["Sp"]).is_member is False


def test_non_member_root_still_included(db):
    with session_scope() as session:
        p = _build_family(session)
        svc = FamilyTreeService(session)
        svc.set_family_tree_member(p["C1"], False)
        graph = svc.build_tree(p["C1"], include_non_members=False)
        assert p["C1"] in graph


# ── lineage queries ──────────────────────────────────────────────────────────

def test_lowest_common_ancestor(db):
    with session_scope() as session:
        p = _build_family(session)
        svc = FamilyTreeService(session)
        # C1 and C2 share both parents at equal depth.
        assert svc.lowest_common_ancestor(p["C1"], p["C2"]) in {p["P1"], p["Sp"]}
        # C1 and Cousin share grandparents.
        assert svc.lowest_common_ancestor(p["C1"], p["Cousin"]) in {p["GpA"], p["GpB"]}
        # Ancestor of self.
        assert svc.lowest_common_ancestor(p["C1"], p["GpA"]) == p["GpA"]


def test_relationship_path_kinds(db):
    with session_scope() as session:
        p = _build_family(session)
        svc = FamilyTreeService(session)

        assert svc.relationship_path(p["C1"], p["C1"]).kind == "self"
        assert svc.relationship_path(p["P1"], p["Sp"]).kind == "spouse"

        sib = svc.relationship_path(p["C1"], p["C2"])
        assert sib.kind == "sibling"
        assert (sib.up, sib.down) == (1, 1)

        anc = svc.relationship_path(p["C1"], p["GpA"])
        assert anc.kind == "ancestor"
        assert (anc.up, anc.down) == (2, 0)
        assert anc.path[0] == p["C1"] and anc.path[-1] == p["GpA"]

        desc = svc.relationship_path(p["GpA"], p["C1"])
        assert desc.kind == "descendant"
        assert (desc.up, desc.down) == (0, 2)

        cousin = svc.relationship_path(p["C1"], p["Cousin"])
        assert cousin.kind == "cousin"
        assert (cousin.up, cousin.down) == (2, 2)


def test_relationship_path_unrelated(db):
    with session_scope() as session:
        p = _build_family(session)
        stranger = _person(session, "Stranger")
        path = FamilyTreeService(session).relationship_path(p["C1"], stranger.id)
        assert path.kind == "unrelated"
        assert path.path == ()


# ── export ─────────────────────────────────────────────────────────────────────

def test_render_tree_json_shape(db):
    with session_scope() as session:
        p = _build_family(session)
        data = FamilyTreeService(session).render_tree_json(p["C1"])

        assert data["root_id"] == p["C1"]
        persons = data["persons"]
        assert len(persons) == len(p)
        # Sorted by (generation, id).
        gens = [row["generation"] for row in persons]
        assert gens == sorted(gens)

        ids = {row["id"] for row in persons}
        for row in persons:
            for edge in (*row["parents"], *row["children"], *row["spouses"]):
                assert edge in ids  # edges only reference included persons


# ── settings & membership column ─────────────────────────────────────────────────

def test_settings_singleton_and_update(db):
    with session_scope() as session:
        p = _build_family(session)
        svc = FamilyTreeService(session)
        settings = svc.get_settings()
        assert settings.id == 1
        assert settings.layout == "tree"
        assert settings.include_non_members is True

        svc.update_settings(default_root_person_id=p["C1"], layout="radial")
        again = svc.get_settings()
        assert again.id == 1  # still singleton
        assert again.default_root_person_id == p["C1"]
        assert again.layout == "radial"

        assert session.query(FamilyTreeSettings).count() == 1


def test_is_family_tree_member_defaults_true(db):
    with session_scope() as session:
        person = _person(session, "Fresh")
        assert person.is_family_tree_member is True
