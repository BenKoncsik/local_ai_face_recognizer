"""Tests for FamilyCode validation and derived family relationships."""

from __future__ import annotations

import sqlite3

import pytest
from sqlalchemy.exc import IntegrityError

from app.db.database import init_db, session_scope
from app.db.models import Person
from app.services.family_service import (
    FamilyService,
    parse_family_code,
    validate_family_code,
)


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "family_code.db"
    init_db(db_path)
    return db_path


def test_empty_family_code_is_accepted():
    assert validate_family_code("") is None
    assert validate_family_code(None) is None


def test_family_code_examples_are_parsed():
    c8 = parse_family_code("c8")
    c0 = parse_family_code("C0")
    c80 = parse_family_code("C80")
    c85 = parse_family_code("C85")
    c851 = parse_family_code("C851")

    assert c8 is not None
    assert c8.code == "C8"
    assert c8.root == "C"
    assert c8.generation == 1
    assert c0 is not None
    assert c0.spouse_of_code == "C"
    assert c80 is not None
    assert c80.spouse_of_code == "C8"
    assert c80.parent_code is None
    assert c85 is not None
    assert c85.parent_code == "C8"
    assert c851 is not None
    assert c851.parent_code == "C85"


def test_invalid_family_code_is_rejected():
    # C08 = child hung off the bare root marker; C800 = spouse of a spouse.
    for code in ["8C", "C08", "C800", "CC8", "C-8"]:
        with pytest.raises(ValueError):
            validate_family_code(code)


def test_spec_defined_simple_codes_are_accepted():
    # Relationships defined in the spec that the old strict regex rejected.
    assert validate_family_code("C00") == "C00"    # spouse of the root person
    assert validate_family_code("C2201") == "C2201"  # spouse's child (stepchild)
    assert validate_family_code("C805") == "C805"    # C8's stepchild via spouse C80
    assert validate_family_code("C810") == "C810"    # spouse of a grandchild


def test_extended_family_codes_are_accepted():
    # Ancestors (F), siblings (T), friends (B), multi-codes and ranges must all
    # be accepted and canonicalised, not rejected.
    assert validate_family_code("C81B") == "C81B"
    assert validate_family_code("c81b") == "C81B"
    assert validate_family_code("C0F1") == "C0F1"
    assert validate_family_code("C0F22") == "C0F22"
    assert validate_family_code("C00T1") == "C00T1"
    assert validate_family_code("C810T4") == "C810T4"
    assert validate_family_code("C[1-9]B") == "C[1-9]B"
    assert "C81B" in validate_family_code("C81B,C82B")
    assert "C82B" in validate_family_code("C81B,C82B")


def test_invalid_extended_family_codes_are_rejected():
    for code in ["C0F3", "C81B1", "CF1", "C00T0"]:
        with pytest.raises(ValueError):
            validate_family_code(code)


def test_extended_codes_have_no_simple_tree_derivation():
    # parse_family_code returns None for extended codes (no parent/spouse tree).
    assert parse_family_code("C81B") is None
    assert parse_family_code("C0F1") is None
    assert parse_family_code("C00T1") is None
    # Simple codes still parse as before.
    assert parse_family_code("C85") is not None


def test_duplicate_family_code_is_rejected(db):
    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(Person(name="Anna", is_auto_named=False, family_code="C8"))
            session.add(Person(name="Bela", is_auto_named=False, family_code="C8"))


def test_service_reports_duplicate_family_code(db):
    with session_scope() as session:
        person = Person(name="Anna", is_auto_named=False, family_code="C8")
        session.add(person)

    with session_scope() as session:
        with pytest.raises(ValueError):
            FamilyService(session).ensure_unique_family_code("c8")


def test_friend_codes_are_not_unique(db):
    # The same B (friend) code may be assigned to several different people.
    with session_scope() as session:
        session.add(Person(name="Friend A", is_auto_named=False, family_code="C81B"))

    with session_scope() as session:
        svc = FamilyService(session)
        # Does not raise, even though C81B already exists.
        assert svc.ensure_unique_family_code("C81B") == "C81B"


def test_identity_codes_remain_unique(db):
    with session_scope() as session:
        session.add(Person(name="Matyi", is_auto_named=False, family_code="C81"))

    with session_scope() as session:
        with pytest.raises(ValueError):
            FamilyService(session).ensure_unique_family_code("C81")


def test_family_code_relationships_are_derived(db):
    with session_scope() as session:
        c8 = Person(name="C8", is_auto_named=False, family_code="C8")
        c80 = Person(name="C80", is_auto_named=False, family_code="C80")
        c85 = Person(name="C85", is_auto_named=False, family_code="C85")
        c86 = Person(name="C86", is_auto_named=False, family_code="C86")
        c851 = Person(name="C851", is_auto_named=False, family_code="C851")
        no_code = Person(name="No code", is_auto_named=False)
        session.add_all([c8, c80, c85, c86, c851, no_code])
        session.flush()
        ids = {p.name: p.id for p in [c8, c80, c85, c86, c851, no_code]}

    with session_scope() as session:
        svc = FamilyService(session)
        assert svc.are_spouses(ids["C8"], ids["C80"])
        assert svc.is_parent_child(ids["C8"], ids["C85"])
        assert svc.is_parent_child(ids["C85"], ids["C851"])
        assert svc.are_siblings(ids["C85"], ids["C86"])
        assert not svc.are_siblings(ids["C8"], ids["No code"])
        assert [p.family_code for p in svc.list_branch_by_family_code("C8")] == [
            "C8",
            "C80",
            "C85",
            "C851",
            "C86",
        ]


def test_legacy_persons_without_family_code_survive_migration(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(255) NOT NULL,
                is_auto_named BOOLEAN NOT NULL DEFAULT 1,
                thumbnail_path TEXT,
                created_at DATETIME,
                updated_at DATETIME
            )
            """
        )
        conn.execute("INSERT INTO persons (name, is_auto_named) VALUES ('Legacy', 0)")

    init_db(db_path)

    with session_scope() as session:
        legacy = session.query(Person).filter(Person.name == "Legacy").one()
        assert legacy.family_code is None
