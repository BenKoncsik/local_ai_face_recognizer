from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Face, Image, Person, Relationship
from app.services.family_service import (
    FamilyImageSearchCriteria,
    FamilyService,
    REL_PARENT_CHILD,
    REL_SPOUSE,
    parse_family_code,
)


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "family.db")


def _person(session, name: str, gender: str | None = None) -> Person:
    person = Person(name=name, is_auto_named=False, gender=gender)
    session.add(person)
    session.flush()
    return person


def _image(session, path: str) -> Image:
    image = Image(file_path=path, file_hash=path, file_mtime=1.0)
    session.add(image)
    session.flush()
    return image


def _face(session, image: Image, person: Person) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=0,
        bbox_y=0,
        bbox_w=10,
        bbox_h=10,
        confidence=1.0,
        detector_backend="manual",
    )
    session.add(face)
    session.flush()
    return face


def test_spouse_is_canonical_and_not_duplicated(db):
    with session_scope() as session:
        anna = _person(session, "Anna")
        bela = _person(session, "Bela")
        svc = FamilyService(session)
        svc.add_spouse(anna.id, bela.id)
        svc.add_spouse(bela.id, anna.id)

        rows = session.query(Relationship).filter(
            Relationship.relationship_type == REL_SPOUSE
        ).all()
        assert len(rows) == 1
        assert rows[0].person_a_id == min(anna.id, bela.id)
        assert rows[0].person_b_id == max(anna.id, bela.id)


def test_parent_child_validation_and_sibling_derivation(db):
    with session_scope() as session:
        parent = _person(session, "Parent")
        child_a = _person(session, "Child A")
        child_b = _person(session, "Child B")
        svc = FamilyService(session)

        svc.add_parent_child(parent.id, child_a.id)
        svc.add_parent_child(parent.id, child_b.id)

        assert svc.is_parent_child(parent.id, child_a.id)
        assert svc.are_siblings(child_a.id, child_b.id)
        assert not svc.are_siblings(child_a.id, child_a.id)
        with pytest.raises(ValueError):
            svc.add_parent_child(child_a.id, parent.id)

        assert session.query(Relationship).filter(
            Relationship.relationship_type == REL_PARENT_CHILD
        ).count() == 2


def test_search_images_for_two_people_modes(db):
    with session_scope() as session:
        anna = _person(session, "Anna")
        bela = _person(session, "Bela")
        cecil = _person(session, "Cecil")
        together = _image(session, "/tmp/together.jpg")
        group = _image(session, "/tmp/group.jpg")
        only_anna = _image(session, "/tmp/anna.jpg")

        _face(session, together, anna)
        _face(session, together, bela)
        _face(session, group, anna)
        _face(session, group, bela)
        _face(session, group, cecil)
        _face(session, only_anna, anna)

        svc = FamilyService(session)
        both, both_total = svc.search_images_for_people(anna.id, bela.id, mode="both")
        exact, exact_total = svc.search_images_for_people(anna.id, bela.id, mode="exact")

        assert both_total == 2
        assert {r.filename for r in both} == {"together.jpg", "group.jpg"}
        assert exact_total == 1
        assert [r.filename for r in exact] == ["together.jpg"]


def test_relationship_filter_blocks_unmatched_pairs(db):
    with session_scope() as session:
        anna = _person(session, "Anna")
        bela = _person(session, "Bela")
        image = _image(session, "/tmp/together.jpg")
        _face(session, image, anna)
        _face(session, image, bela)

        svc = FamilyService(session)
        rows, total = svc.search_images_for_people(
            anna.id,
            bela.id,
            relationship_filter="spouse",
        )
        assert rows == []
        assert total == 0

        svc.add_spouse(anna.id, bela.id)
        rows, total = svc.search_images_for_people(
            anna.id,
            bela.id,
            relationship_filter="spouse",
        )
        assert total == 1
        assert rows[0].filename == "together.jpg"


def test_search_images_by_comma_name_terms_and_allow_others(db):
    with session_scope() as session:
        benedek = _person(session, "Koncsik Benedek")
        matyi = _person(session, "Matyi")
        domi = _person(session, "Domi")
        together = _image(session, "/tmp/benedek-matyi.jpg")
        group = _image(session, "/tmp/benedek-matyi-domi.jpg")

        _face(session, together, benedek)
        _face(session, together, matyi)
        _face(session, group, benedek)
        _face(session, group, matyi)
        _face(session, group, domi)

        svc = FamilyService(session)
        rows, total = svc.search_images_by_criteria(
            FamilyImageSearchCriteria(
                name_terms=("Benedek", "Matyi"),
                allow_other_people=True,
            )
        )
        assert total == 2
        assert {row.filename for row in rows} == {
            "benedek-matyi.jpg",
            "benedek-matyi-domi.jpg",
        }

        rows, total = svc.search_images_by_criteria(
            FamilyImageSearchCriteria(
                name_terms=("Benedek", "Matyi"),
                allow_other_people=False,
            )
        )
        assert total == 1
        assert [row.filename for row in rows] == ["benedek-matyi.jpg"]


# ── {n} / H derivation and linking ────────────────────────────────────────────

def test_parse_braced_child_derives_both_parents():
    info = parse_family_code("G1{2}3")
    assert info is not None
    assert info.parent_code == "G1"
    assert info.second_parent_code == "G1H2"
    assert not info.is_spouse


def test_parse_numbered_spouse_is_spouse_of_base():
    info = parse_family_code("G1H2")
    assert info is not None
    assert info.is_spouse
    assert info.spouse_of_code == "G1"
    assert info.parent_code is None


def test_short_child_form_has_no_second_parent():
    info = parse_family_code("G13")
    assert info is not None
    assert info.parent_code == "G1"
    assert info.second_parent_code is None


def test_link_derived_parents_braced_child(db):
    with session_scope() as session:
        gabor = _person(session, "Gábor", gender="male")
        gabor.family_code = "G1"
        wife2 = _person(session, "Második feleség", gender="female")
        wife2.family_code = "G1H2"
        child = _person(session, "Gyerek")
        child.family_code = "G1{2}3"
        session.flush()

        svc = FamilyService(session)
        svc.link_derived_parents(child.id)

        # Both parents linked to the child …
        assert svc.is_parent_child(gabor.id, child.id)
        assert svc.is_parent_child(wife2.id, child.id)
        # … and the two parents linked as spouses.
        assert svc.are_spouses(gabor.id, wife2.id)

        assert session.query(Relationship).filter(
            Relationship.relationship_type == REL_PARENT_CHILD
        ).count() == 2
        assert session.query(Relationship).filter(
            Relationship.relationship_type == REL_SPOUSE
        ).count() == 1


def test_link_derived_parents_numbered_spouse(db):
    with session_scope() as session:
        gabor = _person(session, "Gábor")
        gabor.family_code = "G1"
        spouse = _person(session, "Feleség")
        spouse.family_code = "G1H2"
        session.flush()

        svc = FamilyService(session)
        svc.link_derived_parents(spouse.id)

        assert svc.are_spouses(gabor.id, spouse.id)
        assert session.query(Relationship).filter(
            Relationship.relationship_type == REL_SPOUSE
        ).count() == 1


def test_link_skips_missing_second_parent(db):
    with session_scope() as session:
        gabor = _person(session, "Gábor")
        gabor.family_code = "G1"
        child = _person(session, "Gyerek")
        child.family_code = "G1{2}3"  # G1H2 person does not exist
        session.flush()

        svc = FamilyService(session)
        created = svc.link_derived_parents(child.id)

        assert svc.is_parent_child(gabor.id, child.id)
        # Only the existing first parent got a stored row.
        assert session.query(Relationship).filter(
            Relationship.relationship_type == REL_PARENT_CHILD
        ).count() == 1
        assert len(created) == 1


def test_is_parent_child_derives_second_parent_without_rows(db):
    with session_scope() as session:
        wife2 = _person(session, "Második feleség")
        wife2.family_code = "G1H2"
        child = _person(session, "Gyerek")
        child.family_code = "G1{2}3"
        session.flush()

        svc = FamilyService(session)
        # No explicit relationship rows created — derivation alone must work.
        assert svc.is_parent_child(wife2.id, child.id)


def test_update_person_auto_links_parents(db):
    from app.services.person_service import PersonService

    with session_scope() as session:
        gabor = _person(session, "Gábor")
        gabor.family_code = "G1"
        wife2 = _person(session, "Feleség")
        wife2.family_code = "G1H2"
        child = _person(session, "Gyerek")
        child_id = child.id
        session.flush()

        PersonService(session).update_person(child_id, family_code="G1{2}3")

        svc = FamilyService(session)
        assert svc.is_parent_child(gabor.id, child_id)
        assert svc.is_parent_child(wife2.id, child_id)


def test_search_images_by_criteria_detail_filters(db):
    with session_scope() as session:
        anna = _person(session, "Anna", gender="female")
        anna.nickname = "Panni"
        anna.family_code = "C85"
        image = _image(session, "/tmp/family-pic.jpg")
        image.photo_date = "1954"
        _face(session, image, anna)

        rows, total = FamilyService(session).search_images_by_criteria(
            FamilyImageSearchCriteria(
                name_terms=("Anna",),
                person_text="Panni",
                gender="female",
                family_code="C8",
                image_text="family",
                photo_date="1954",
            )
        )

        assert total == 1
        assert rows[0].filename == "family-pic.jpg"
