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
    for code in ["8C", "C08", "C805", "C800", "CC8", "C-8"]:
        with pytest.raises(ValueError):
            validate_family_code(code)


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
