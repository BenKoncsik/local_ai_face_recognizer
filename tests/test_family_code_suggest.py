"""Family-code suggestion tests (spouse / child / parent auto-fill)."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_service import FamilyService


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "suggest.db")


def _person(session, name: str, code: str | None = None) -> int:
    p = Person(name=name, is_auto_named=False, family_code=code)
    session.add(p)
    session.flush()
    return p.id


def test_first_spouse_uses_zero_suffix(db):
    with session_scope() as session:
        base = _person(session, "Base", "C8")
        assert FamilyService(session).suggest_spouse_code(base) == "C80"


def test_further_marriages_are_numbered(db):
    with session_scope() as session:
        base = _person(session, "Base", "C8")
        _person(session, "First spouse", "C80")
        fam = FamilyService(session)
        assert fam.suggest_spouse_code(base) == "C8H2"
        _person(session, "Second spouse", "C8H2")
        assert fam.suggest_spouse_code(base) == "C8H3"


def test_spouse_of_a_spouse_record_is_none(db):
    with session_scope() as session:
        _person(session, "Base", "C8")
        spouse = _person(session, "Spouse", "C80")
        assert FamilyService(session).suggest_spouse_code(spouse) is None


def test_child_uses_next_free_digit(db):
    with session_scope() as session:
        base = _person(session, "Base", "C8")
        fam = FamilyService(session)
        assert fam.suggest_child_code(base) == "C81"
        _person(session, "Child 1", "C81")
        _person(session, "Child 2", "C82")
        assert fam.suggest_child_code(base) == "C83"


def test_child_of_spouse_resolves_to_base(db):
    with session_scope() as session:
        _person(session, "Base", "C8")
        spouse = _person(session, "Spouse", "C80")
        # A child added on the spouse record is still coded under the base C8.
        assert FamilyService(session).suggest_child_code(spouse) == "C81"


def test_child_of_numbered_spouse_is_braced(db):
    with session_scope() as session:
        _person(session, "Base", "C8")
        _person(session, "Child 1", "C81")
        second = _person(session, "Second spouse", "C8H2")
        # Children from the 2nd marriage carry the {2} brace and have their own
        # digit namespace, so the first slot is free.
        assert FamilyService(session).suggest_child_code(second) == "C8{2}1"


def test_parent_code_is_derived(db):
    with session_scope() as session:
        child = _person(session, "Child", "C81")
        assert FamilyService(session).suggest_parent_code(child) == "C8"


def test_no_code_yields_no_suggestion(db):
    with session_scope() as session:
        loner = _person(session, "Loner", None)
        fam = FamilyService(session)
        assert fam.suggest_spouse_code(loner) is None
        assert fam.suggest_child_code(loner) is None
        assert fam.suggest_parent_code(loner) is None


def test_suggesters_follow_active_scheme(db):
    """A scheme that renames the spouse marker is respected by the suggester."""
    from app.services import family_code_schemes as fcs

    base = fcs.builtin_example_scheme().to_dict()
    base["markers"] = {"ancestor": "F", "sibling": "T", "spouse": "S", "friend": "B"}
    fcs.set_active_scheme(fcs.FamilyCodeScheme.from_dict(base))
    try:
        with session_scope() as session:
            person = _person(session, "Base", "C8")
            _person(session, "First spouse", "C80")
            fam = FamilyService(session)
            assert fam.suggest_spouse_code(person) == "C8S2"
            second = _person(session, "Second spouse", "C8S2")
            assert fam.suggest_child_code(second) == "C8{2}1"
    finally:
        fcs.set_active_scheme(None)


def test_add_spouse_with_marriage_period(db):
    with session_scope() as session:
        a = _person(session, "Husband", "C8")
        b = _person(session, "Wife", "C80")
        fam = FamilyService(session)
        fam.add_spouse(a, b, start_date="1950", end_date="1990")
        assert fam.marriage_period(a, b) == ("1950", "1990", None)
        # Symmetric lookup (order-independent).
        assert fam.marriage_period(b, a) == ("1950", "1990", None)


def test_set_marriage_period_updates_existing(db):
    with session_scope() as session:
        a = _person(session, "Husband", "C8")
        b = _person(session, "Wife", "C80")
        fam = FamilyService(session)
        fam.add_spouse(a, b)
        assert fam.marriage_period(a, b) == (None, None, None)
        fam.set_marriage_period(a, b, "1960", "")
        assert fam.marriage_period(a, b) == ("1960", None, None)


def test_regenerate_links_from_codes(db):
    with session_scope() as session:
        _person(session, "Base", "C8")
        _person(session, "Child", "C81")
        _person(session, "Spouse", "C80")
        fam = FamilyService(session)
        touched = fam.regenerate_all_links()
        assert touched >= 2

        from app.db.models import Relationship
        from app.services.family_service import REL_PARENT_CHILD, REL_SPOUSE

        assert (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_PARENT_CHILD)
            .count()
            == 1
        )
        assert (
            session.query(Relationship)
            .filter(Relationship.relationship_type == REL_SPOUSE)
            .count()
            == 1
        )
