"""Tests for the graphical family-tree view widget."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_service import FamilyService
from app.services.family_tree_service import FamilyTreeService
from app.ui.widgets.family_tree_view import FamilyTreeView


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "view.db")


def _seed(session) -> dict[str, int]:
    def mk(name, code, gender):
        p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
        session.add(p)
        session.flush()
        return p.id

    ids = {
        "gpa": mk("Grandpa", "C", "male"),
        "gpb": mk("Grandma", "C0", "female"),
        "p1": mk("Father", "C1", "male"),
        "sp": mk("Mother", "C10", "female"),
        "c1": mk("Kid", "C11", "male"),
    }
    fam = FamilyService(session)
    fam.add_spouse(ids["gpa"], ids["gpb"])
    fam.add_parent_child(ids["gpa"], ids["p1"])
    fam.add_spouse(ids["p1"], ids["sp"])
    fam.add_parent_child(ids["p1"], ids["c1"])
    fam.add_parent_child(ids["sp"], ids["c1"])
    return ids


def test_set_graph_draws_one_box_per_person(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    assert set(view._node_items) == set(ids.values())


def test_set_graph_empty_is_safe(db, qtbot):
    with session_scope() as session:
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    assert view._node_items == {}


def test_boxes_positioned_by_generation(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    # Grandparent sits above parent, parent above child (increasing y).
    y_gpa = view._node_items[ids["gpa"]].pos().y()
    y_p1 = view._node_items[ids["p1"]].pos().y()
    y_c1 = view._node_items[ids["c1"]].pos().y()
    assert y_gpa < y_p1 < y_c1


def test_selected_and_focus_use_distinct_colors(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    # Root (focus) is the grandparent; the clicked person is the child.
    view.set_graph(graph, focus_id=ids["gpa"], selected_id=ids["c1"])

    focus_color = view._rect_items[ids["gpa"]].pen().color().name()
    selected_color = view._rect_items[ids["c1"]].pen().color().name()
    assert focus_color != selected_color

    # Clicking another person moves the red selection off the old one.
    view.set_selected(ids["p1"])
    assert view._rect_items[ids["p1"]].pen().color().name() == selected_color
    assert view._rect_items[ids["c1"]].pen().color().name() != selected_color


def test_zoom_in_out_reset(db, qtbot):
    with session_scope() as session:
        _seed(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    base = view.transform().m11()
    view.zoom_in()
    assert view.transform().m11() > base
    view.reset_zoom()
    assert abs(view.transform().m11() - 1.0) < 1e-6
    view.zoom_out()
    assert view.transform().m11() < 1.0


def test_photos_are_loaded_lazily(db, qtbot, tmp_path):
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QGraphicsPixmapItem

    from app.db.models import Face, Image

    crop = tmp_path / "crop.png"
    pm = QPixmap(60, 60)
    pm.fill()
    pm.save(str(crop))

    with session_scope() as session:
        ids = _seed(session)
        image = Image(file_path=str(tmp_path / "i.jpg"), file_hash="h", file_mtime=1.0)
        session.add(image)
        session.flush()
        session.add(
            Face(
                image_id=image.id, person_id=ids["gpa"],
                bbox_x=0, bbox_y=0, bbox_w=1, bbox_h=1,
                confidence=0.9, detector_backend="manual", crop_path=str(crop),
            )
        )
        session.flush()
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    view.resize(800, 600)
    qtbot.addWidget(view)
    view.set_graph(graph)

    # Photo registered as pending (not yet loaded) right after layout.
    assert ids["gpa"] in view._pending_photos
    view._load_visible_photos()
    assert ids["gpa"] not in view._pending_photos
    assert any(
        isinstance(it, QGraphicsPixmapItem) for it in view._scene.items()
    )


def test_generation_numbers_in_left_gutter(db, qtbot):
    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    with session_scope() as session:
        _seed(session)  # 3 generations
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    numbers = sorted(
        item.text()
        for item in view._scene.items()
        if isinstance(item, QGraphicsSimpleTextItem) and item.text().isdigit()
    )
    assert numbers == ["1", "2", "3"]
    # The gutter sits left of the content (negative scene x).
    assert view._scene.sceneRect().left() < 0


def test_selection_marks_ancestors_and_descendants(db, qtbot):
    with session_scope() as session:
        ids = _seed(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    view.set_selected(ids["p1"])

    assert ids["gpa"] in view._ancestors      # parent's parent
    assert ids["c1"] in view._descendants     # parent's child
    assert ids["sp"] not in view._ancestors   # spouse is neither
    assert ids["sp"] not in view._descendants


def _seed_siblings(session) -> dict[str, int]:
    """A couple with three children — mirrors the multi-child bug report."""
    def mk(name, code, gender):
        p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
        session.add(p)
        session.flush()
        return p.id

    ids = {
        "mom": mk("Anya", "C8", "female"),
        "dad": mk("Apa", "C80", "male"),
        "k1": mk("Benedek", "C86", "male"),
        "k2": mk("Matyi", "C81", "male"),
        "k3": mk("Teszt", "C8B", "female"),
    }
    fam = FamilyService(session)
    fam.add_spouse(ids["mom"], ids["dad"])
    for kid in ("k1", "k2", "k3"):
        fam.add_parent_child(ids["mom"], ids[kid])
        fam.add_parent_child(ids["dad"], ids[kid])
    return ids


def _childdrop_color(view, child_id):
    for it in view._connector_items:
        if it.get("role") == "childdrop" and it.get("child") == child_id:
            return it["item"].pen().color().name()
    return None


def test_siblings_share_one_parent_bus(db, qtbot):
    """All siblings group under a single couple, not one bus per child."""
    with session_scope() as session:
        _seed_siblings(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    # Exactly one parent→children group, holding all three children.
    assert len(view._connector_groups) == 1
    assert len(view._connector_groups[0]["children"]) == 3


def test_selecting_child_highlights_only_its_path_up(db, qtbot):
    from app.ui.widgets.family_tree_view import _ANCESTOR_BORDER, _LINE

    with session_scope() as session:
        ids = _seed_siblings(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    view.set_selected(ids["k3"])

    # The selected child's drop to the bus is lit; siblings' drops stay grey.
    assert _childdrop_color(view, ids["k3"]) == _ANCESTOR_BORDER.name()
    assert _childdrop_color(view, ids["k1"]) == _LINE.name()
    assert _childdrop_color(view, ids["k2"]) == _LINE.name()


def test_couple_has_no_extra_bridge_line(db, qtbot):
    """A married couple's children hang from the spouse line — no bridge line."""
    with session_scope() as session:
        _seed_siblings(session)  # both parents linked AND married
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    assert not any(it.get("role") == "bridge" for it in view._connector_items)
    # The single junction drop still exists (children connect to the couple).
    assert any(it.get("role") == "junction" for it in view._connector_items)


def test_marriage_date_label_is_drawn(db, qtbot):
    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    with session_scope() as session:
        def mk(name, code, gender):
            p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
            session.add(p)
            session.flush()
            return p.id

        mom = mk("Anya", "C8", "female")
        dad = mk("Apa", "C80", "male")
        FamilyService(session).add_spouse(mom, dad, start_date="1984", end_date="1990")
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    texts = {
        it.text()
        for it in view._scene.items()
        if isinstance(it, QGraphicsSimpleTextItem)
    }
    assert "1984–1990" in texts


def _seed_single_parent_siblings(session) -> dict[str, int]:
    """A couple whose children are each linked to ONLY ONE parent.

    Reproduces the bug: codes derive every child under one parent (Anya), so the
    other spouse (Apa) is never a recorded parent until co-parent inference adds
    them. Both bugs (only one ancestor lights up; siblings don't share a bus)
    stem from this single-parent linkage.
    """
    def mk(name, code, gender):
        p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
        session.add(p)
        session.flush()
        return p.id

    ids = {
        "mom": mk("Anya", "C8", "female"),
        "dad": mk("Apa", "C80", "male"),
        "k1": mk("Benedek", "C86", "male"),
        "k2": mk("Matyi", "C81", "male"),
        "k3": mk("Teszt", "C82", "female"),
    }
    fam = FamilyService(session)
    fam.add_spouse(ids["mom"], ids["dad"])
    for kid in ("k1", "k2", "k3"):
        fam.add_parent_child(ids["mom"], ids[kid])  # one parent only
    return ids


def test_single_parent_siblings_share_one_bus(db, qtbot):
    """Children linked to one parent still group under the whole couple."""
    with session_scope() as session:
        _seed_single_parent_siblings(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    assert len(view._connector_groups) == 1
    group = view._connector_groups[0]
    assert len(group["children"]) == 3
    assert len(group["parents"]) == 2  # both spouses bridged in, not just one


def test_selecting_single_parent_child_highlights_both_parents(db, qtbot):
    """Selecting such a child marks BOTH parents as ancestors (bug #1)."""
    with session_scope() as session:
        ids = _seed_single_parent_siblings(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    view.set_selected(ids["k1"])

    assert ids["mom"] in view._ancestors
    assert ids["dad"] in view._ancestors  # was missing before the fix


def _column_of(view, pid):
    """Layout column index of a person (boxes are on a fixed x grid)."""
    step_x = view.BOX_W + view._min_spouse_gap(view._graph)
    return round(view._node_items[pid].pos().x() / step_x)


def test_spouses_are_placed_side_by_side(db, qtbot):
    """Each couple occupies adjacent columns so the spouse line is short.

    Reproduces the 'who is married to whom blurs together' bug: a married-in
    spouse with no parents in the tree used to sort to the far edge, stretching
    the spouse line across everyone else.
    """
    with session_scope() as session:
        def mk(name, code, gender):
            p = Person(name=name, is_auto_named=False, family_code=code, gender=gender)
            session.add(p)
            session.flush()
            return p.id

        # A top-band couple with three children, each child married to a
        # married-in spouse who has no parents recorded in the tree.
        gm = mk("Nagymama", "C", "female")
        gp = mk("Nagypapa", "C0", "male")
        fam = FamilyService(session)
        fam.add_spouse(gm, gp)

        couples = []
        for i, code in enumerate(("C1", "C2", "C3")):
            kid = mk(f"Gyerek{i}", code, "male")
            partner = mk(f"Par{i}", f"{code}0", "female")
            fam.add_parent_child(gm, kid)
            fam.add_parent_child(gp, kid)
            fam.add_spouse(kid, partner)
            couples.append((kid, partner))

        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)

    # Top-band couple is adjacent.
    assert abs(_column_of(view, gm) - _column_of(view, gp)) == 1
    # Every married pair in the child band is adjacent, not scattered.
    for kid, partner in couples:
        assert abs(_column_of(view, kid) - _column_of(view, partner)) == 1


def test_selecting_parent_highlights_all_children_down(db, qtbot):
    from app.ui.widgets.family_tree_view import _DESCENDANT_BORDER

    with session_scope() as session:
        ids = _seed_siblings(session)
        graph = FamilyTreeService(session).build_graph()

    view = FamilyTreeView()
    qtbot.addWidget(view)
    view.set_graph(graph)
    view.set_selected(ids["mom"])

    # Every child is a descendant of the selected parent → all drops green.
    for kid in ("k1", "k2", "k3"):
        assert _childdrop_color(view, ids[kid]) == _DESCENDANT_BORDER.name()
