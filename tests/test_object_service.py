"""Unit tests for :class:`app.services.object_service.ObjectService`."""

from __future__ import annotations

import os
import tempfile

import pytest

from app.db.database import init_db, session_scope
from app.db.models import (
    OBJECT_ROLE_DRIVER,
    OBJECT_ROLE_OWNER,
    Image,
    ObjectOccurrence,
    Person,
    TaggedObject,
)
from app.services.object_service import ObjectFilters, ObjectService


@pytest.fixture()
def db():
    d = tempfile.mkdtemp()
    init_db(os.path.join(d, "objects.db"))
    yield


def _make_image(session, path: str) -> Image:
    img = Image(file_path=path, file_hash="h" + path, file_mtime=0.0)
    session.add(img)
    session.flush()
    return img


def _make_person(session, name: str, *, protected: bool = False) -> Person:
    p = Person(name=name, is_auto_named=False, is_protected=protected)
    session.add(p)
    session.flush()
    return p


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_create_and_find(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW E91", "2004 330D", "jegyzet")
        assert obj.id is not None
        assert svc.find_by_name("bmw e91").id == obj.id
        assert svc.find_by_name("nope") is None


def test_create_empty_name_rejected(db):
    with session_scope() as s:
        with pytest.raises(ValueError):
            ObjectService(s).create_object("   ")


def test_get_or_create_is_idempotent(db):
    with session_scope() as s:
        svc = ObjectService(s)
        a = svc.get_or_create("Klotild")
        b = svc.get_or_create("Klotild")
        assert a.id == b.id


def test_update_object(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("Hajó")
        svc.update_object(obj.id, name="Klotild", description="vitorlás", notes="x")
        assert obj.name == "Klotild"
        assert obj.description == "vitorlás"
        with pytest.raises(ValueError):
            svc.update_object(obj.id, name="  ")


def test_delete_cascades_occurrences(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        svc.add_occurrence(obj.id, img.id, 10, 20)
        svc.delete_object(obj.id)
        assert s.query(ObjectOccurrence).count() == 0


# ---------------------------------------------------------------------------
# Occurrences
# ---------------------------------------------------------------------------

def test_add_occurrence_idempotent_same_point(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        o1 = svc.add_occurrence(obj.id, img.id, 10, 20)
        o2 = svc.add_occurrence(obj.id, img.id, 10, 20, note="frissen mosva")
        assert o1.id == o2.id
        assert o2.note == "frissen mosva"
        assert s.query(ObjectOccurrence).count() == 1


def test_same_object_multiple_images(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW")
        for i in range(3):
            img = _make_image(s, f"/img{i}.jpg")
            svc.add_occurrence(obj.id, img.id, i, i, note=f"note {i}")
        occs = svc.get_occurrences(obj.id)
        assert len(occs) == 3
        assert {o.note for o in occs} == {"note 0", "note 1", "note 2"}


def test_add_occurrence_bbox(db):
    from app.db.models import OBJECT_GEOMETRY_BBOX

    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        occ = svc.add_occurrence_bbox(obj.id, img.id, 10, 20, 100, 60, note="kék")
        assert occ.geometry_type == OBJECT_GEOMETRY_BBOX
        assert (occ.bbox_x, occ.bbox_y, occ.bbox_w, occ.bbox_h) == (10, 20, 100, 60)
        # top-left stored as point for the unique constraint
        assert (occ.point_x, occ.point_y) == (10, 20)

        info = svc.get_occurrences_for_image(img.id)[0]
        assert (info.bbox_x, info.bbox_y, info.bbox_w, info.bbox_h) == (10, 20, 100, 60)

        # Idempotent on the top-left point; updates the box.
        occ2 = svc.add_occurrence_bbox(obj.id, img.id, 10, 20, 120, 80)
        assert occ2.id == occ.id
        assert (occ2.bbox_w, occ2.bbox_h) == (120, 80)


def test_thumbnail_specs_and_manual(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        svc.add_occurrence(obj.id, img.id, 5, 6)  # point only
        svc.add_occurrence_bbox(obj.id, img.id, 10, 20, 100, 60)  # bbox preferred
        specs = svc.get_thumbnail_specs()
        assert specs[obj.id] == ("/a.jpg", (10, 20, 100, 60))

        # Manual thumbnail wins and resolves to (path, None).
        svc.set_thumbnail_path(obj.id, "/crops/obj.jpg", manual=True)
        specs = svc.get_thumbnail_specs()
        assert specs[obj.id] == ("/crops/obj.jpg", None)

        svc.set_thumbnail_path(obj.id, None)
        specs = svc.get_thumbnail_specs()
        assert specs[obj.id] == ("/a.jpg", (10, 20, 100, 60))


def test_update_occurrence_bbox(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        occ = svc.add_occurrence_bbox(obj.id, img.id, 1, 2, 30, 40)
        svc.update_occurrence_bbox(occ.id, 5, 6, 70, 80)
        info = svc.get_occurrences_for_image(img.id)[0]
        assert (info.bbox_x, info.bbox_y, info.bbox_w, info.bbox_h) == (5, 6, 70, 80)
        assert (info.point_x, info.point_y) == (5, 6)


def test_update_and_remove_occurrence(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        obj = svc.create_object("Autó")
        occ = svc.add_occurrence(obj.id, img.id, 1, 2)
        svc.update_occurrence_note(occ.id, "új jegyzet")
        assert svc.get_occurrences(obj.id)[0].note == "új jegyzet"
        svc.remove_occurrence(occ.id)
        assert svc.get_occurrences(obj.id) == []


def test_occurrences_for_image(db):
    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/a.jpg")
        a = svc.create_object("Autó")
        b = svc.create_object("Ház")
        svc.add_occurrence(a.id, img.id, 1, 1)
        svc.add_occurrence(b.id, img.id, 2, 2)
        rows = svc.get_occurrences_for_image(img.id)
        assert {r.object_id for r in rows} == {a.id, b.id}


# ---------------------------------------------------------------------------
# Aggregation / list
# ---------------------------------------------------------------------------

def test_list_objects_counts(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW", "leiras")
        img1 = _make_image(s, "/1.jpg")
        img2 = _make_image(s, "/2.jpg")
        svc.add_occurrence(obj.id, img1.id, 1, 1, note="a")
        svc.add_occurrence(obj.id, img2.id, 2, 2)  # no note
        person = _make_person(s, "Benedek")
        svc.add_person_link(obj.id, person.id, OBJECT_ROLE_OWNER)

        summaries = svc.list_objects(ObjectFilters())
        assert len(summaries) == 1
        sm = summaries[0]
        assert sm.image_count == 2
        assert sm.occurrence_count == 2
        assert sm.note_count == 1
        assert sm.person_count == 1
        assert sm.thumbnail_path == "/1.jpg"  # fallback to first image


def test_list_objects_name_filter(db):
    with session_scope() as s:
        svc = ObjectService(s)
        svc.create_object("BMW E91")
        svc.create_object("Klotild")
        res = svc.list_objects(ObjectFilters(name="bmw"))
        assert [r.name for r in res] == ["BMW E91"]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_by_name_description_note_accent_insensitive(db):
    with session_scope() as s:
        svc = ObjectService(s)
        bmw = svc.create_object("BMW E91", "Balaton parton")
        klo = svc.create_object("Klotild")
        img = _make_image(s, "/a.jpg")
        svc.add_occurrence(klo.id, img.id, 1, 1, note="A Balatonnál nyaraltunk")

        # description match
        assert {r.object_id for r in svc.search_objects("balaton")} == {
            bmw.object_id if hasattr(bmw, "object_id") else bmw.id,
            klo.id,
        }
        # name match
        assert [r.name for r in svc.search_objects("klot")] == ["Klotild"]
        # empty -> all
        assert len(svc.search_objects("")) == 2


# ---------------------------------------------------------------------------
# Person links
# ---------------------------------------------------------------------------

def test_person_links_roles(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW")
        ben = _make_person(s, "Benedek")
        apa = _make_person(s, "Apa")
        svc.add_person_link(obj.id, ben.id, OBJECT_ROLE_OWNER)
        svc.add_person_link(obj.id, ben.id, OBJECT_ROLE_DRIVER)  # same person, 2 roles
        svc.add_person_link(obj.id, apa.id, OBJECT_ROLE_OWNER)

        persons = svc.get_object_persons(obj.id)
        assert len(persons) == 3
        ben_objs = svc.get_objects_for_person(ben.id)
        assert {o.role for o in ben_objs} == {OBJECT_ROLE_OWNER, OBJECT_ROLE_DRIVER}


def test_person_link_protected_rejected(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW")
        unknown = _make_person(s, "Ismeretlen", protected=True)
        with pytest.raises(ValueError):
            svc.add_person_link(obj.id, unknown.id, OBJECT_ROLE_OWNER)


def test_person_link_bad_role_rejected(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW")
        ben = _make_person(s, "Benedek")
        with pytest.raises(ValueError):
            svc.add_person_link(obj.id, ben.id, "spaceship-pilot")


def test_remove_person_link(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("BMW")
        ben = _make_person(s, "Benedek")
        svc.add_person_link(obj.id, ben.id, OBJECT_ROLE_OWNER)
        svc.remove_person_link(obj.id, ben.id, OBJECT_ROLE_OWNER)
        assert svc.get_object_persons(obj.id) == []


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def test_merge_preserves_everything(db):
    with session_scope() as s:
        svc = ObjectService(s)
        target = svc.create_object("BMW")
        source = svc.create_object("BMW E91", "330D")
        img1 = _make_image(s, "/1.jpg")
        img2 = _make_image(s, "/2.jpg")
        svc.add_occurrence(target.id, img1.id, 1, 1, note="t")
        svc.add_occurrence(source.id, img2.id, 2, 2, note="s")
        ben = _make_person(s, "Benedek")
        svc.add_person_link(source.id, ben.id, OBJECT_ROLE_OWNER)

        merged = svc.merge_objects([source.id], target.id, name="BMW E91")
        assert merged.id == target.id
        assert merged.name == "BMW E91"
        assert s.query(TaggedObject).count() == 1
        occs = svc.get_occurrences(target.id)
        assert len(occs) == 2
        assert {o.note for o in occs} == {"t", "s"}
        assert len(svc.get_object_persons(target.id)) == 1


def test_merge_dedups_duplicate_occurrence(db):
    with session_scope() as s:
        svc = ObjectService(s)
        target = svc.create_object("A")
        source = svc.create_object("B")
        img = _make_image(s, "/1.jpg")
        svc.add_occurrence(target.id, img.id, 5, 5)
        svc.add_occurrence(source.id, img.id, 5, 5)  # same point -> dup on merge
        svc.merge_objects([source.id], target.id)
        assert s.query(ObjectOccurrence).count() == 1


def test_merge_self_is_noop(db):
    with session_scope() as s:
        svc = ObjectService(s)
        obj = svc.create_object("A")
        res = svc.merge_objects([obj.id], obj.id)
        assert res.id == obj.id


# ---------------------------------------------------------------------------
# Universal search integration (objects surface images via any_terms)
# ---------------------------------------------------------------------------

def test_universal_search_finds_object_images(db):
    from app.services.family_service import (
        FamilyImageSearchCriteria,
        FamilyService,
    )

    with session_scope() as s:
        svc = ObjectService(s)
        img = _make_image(s, "/holiday.jpg")
        obj = svc.create_object("BMW E91", "330D")
        # The term only appears in a per-image occurrence note.
        svc.add_occurrence(obj.id, img.id, 3, 4, note="A Balatonnál nyaraltunk")

    with session_scope() as s:
        fam = FamilyService(s)
        results, total = fam.search_images_by_criteria(
            FamilyImageSearchCriteria(any_terms=("Balaton",))
        )
        assert total == 1
        assert results[0].file_path == "/holiday.jpg"

        # Object name also surfaces its image.
        results, total = fam.search_images_by_criteria(
            FamilyImageSearchCriteria(any_terms=("BMW",))
        )
        assert total == 1


def test_search_suggestions_include_objects(db):
    from app.services.family_service import FamilyService

    with session_scope() as s:
        ObjectService(s).create_object("Klotild")
    with session_scope() as s:
        suggestions = FamilyService(s).get_search_suggestions("klot")
        assert any(t == "object" and v == "Klotild" for v, t, _ in suggestions)
