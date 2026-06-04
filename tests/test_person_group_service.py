"""Unit tests for PersonGroupService."""

from __future__ import annotations

import pytest

from app.db.database import init_db, session_scope
from app.db.models import Person, PersonGroup, PersonGroupMembership
from app.services.person_group_service import PersonGroupService


@pytest.fixture()
def db(tmp_path):
    db_path = tmp_path / "test_groups.db"
    init_db(db_path)
    return db_path


@pytest.fixture()
def alice(db):
    with session_scope() as s:
        p = Person(name="Alice", is_auto_named=False)
        s.add(p)
        s.flush()
        return p.id


@pytest.fixture()
def bob(db):
    with session_scope() as s:
        p = Person(name="Bob", is_auto_named=False)
        s.add(p)
        s.flush()
        return p.id


# ---------------------------------------------------------------------------
# Group CRUD
# ---------------------------------------------------------------------------

class TestCreateGroup:
    def test_create_returns_group(self, db):
        with session_scope() as s:
            svc = PersonGroupService(s)
            group = svc.create_group("Kórus")
        assert group.id is not None
        assert group.name == "Kórus"

    def test_create_trims_whitespace(self, db):
        with session_scope() as s:
            svc = PersonGroupService(s)
            group = svc.create_group("  Munkahely  ")
        assert group.name == "Munkahely"

    def test_create_empty_name_raises(self, db):
        with pytest.raises(ValueError, match="empty"):
            with session_scope() as s:
                PersonGroupService(s).create_group("  ")

    def test_duplicate_name_raises(self, db):
        with session_scope() as s:
            PersonGroupService(s).create_group("Kórus")
        with pytest.raises(ValueError, match="already exists"):
            with session_scope() as s:
                PersonGroupService(s).create_group("Kórus")

    def test_duplicate_name_case_insensitive(self, db):
        with session_scope() as s:
            PersonGroupService(s).create_group("kórus")
        with pytest.raises(ValueError, match="already exists"):
            with session_scope() as s:
                PersonGroupService(s).create_group("KÓRUS")

    def test_list_groups_alphabetical(self, db):
        with session_scope() as s:
            svc = PersonGroupService(s)
            svc.create_group("Zenekar")
            svc.create_group("Kórus")
            svc.create_group("Barátok")
        with session_scope() as s:
            infos = PersonGroupService(s).list_groups()
        names = [g.name for g in infos]
        assert names == sorted(names, key=str.lower)

    def test_rename_group(self, db):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("OldName").id
        with session_scope() as s:
            svc = PersonGroupService(s)
            renamed = svc.rename_group(gid, "NewName")
        assert renamed.name == "NewName"

    def test_rename_to_existing_raises(self, db):
        with session_scope() as s:
            svc = PersonGroupService(s)
            gid = svc.create_group("First").id
            svc.create_group("Second")
        with pytest.raises(ValueError, match="already exists"):
            with session_scope() as s:
                PersonGroupService(s).rename_group(gid, "Second")

    def test_delete_group(self, db):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("ToDelete").id
        with session_scope() as s:
            PersonGroupService(s).delete_group(gid)
        with session_scope() as s:
            assert s.get(PersonGroup, gid) is None

    def test_get_or_create_returns_existing(self, db):
        with session_scope() as s:
            first = PersonGroupService(s).create_group("Unique")
        with session_scope() as s:
            second = PersonGroupService(s).get_or_create("Unique")
        assert first.id == second.id


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------

class TestMembership:
    def test_set_person_groups_single(self, db, alice):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid])
        with session_scope() as s:
            groups = PersonGroupService(s).get_person_groups(alice)
        assert len(groups) == 1
        assert groups[0].name == "Kórus"

    def test_set_person_multiple_groups(self, db, alice):
        with session_scope() as s:
            svc = PersonGroupService(s)
            gid1 = svc.create_group("Kórus").id
            gid2 = svc.create_group("Munkahely").id
            gid3 = svc.create_group("Barátok").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid1, gid2, gid3])
        with session_scope() as s:
            groups = PersonGroupService(s).get_person_groups(alice)
        assert len(groups) == 3

    def test_remove_person_from_group(self, db, alice):
        with session_scope() as s:
            svc = PersonGroupService(s)
            gid1 = svc.create_group("Kórus").id
            gid2 = svc.create_group("Munkahely").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid1, gid2])
        # Remove gid2 by setting only gid1
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid1])
        with session_scope() as s:
            groups = PersonGroupService(s).get_person_groups(alice)
        assert len(groups) == 1
        assert groups[0].id == gid1

    def test_clear_all_groups(self, db, alice):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid])
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [])
        with session_scope() as s:
            groups = PersonGroupService(s).get_person_groups(alice)
        assert groups == []

    def test_multiple_persons_same_group(self, db, alice, bob):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            svc = PersonGroupService(s)
            svc.set_person_groups(alice, [gid])
            svc.set_person_groups(bob, [gid])
        with session_scope() as s:
            members = PersonGroupService(s).get_persons_in_group(gid)
        names = {p.name for p in members}
        assert names == {"Alice", "Bob"}

    def test_protected_person_cannot_have_groups(self, db):
        with session_scope() as s:
            protected = Person(name="Ismeretlen", is_auto_named=False, is_protected=True)
            s.add(protected)
            s.flush()
            pid = protected.id
            gid = PersonGroupService(s).create_group("Kórus").id
        with pytest.raises(ValueError, match="protected"):
            with session_scope() as s:
                PersonGroupService(s).set_person_groups(pid, [gid])

    def test_member_count_in_list(self, db, alice, bob):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            svc = PersonGroupService(s)
            svc.set_person_groups(alice, [gid])
            svc.set_person_groups(bob, [gid])
        with session_scope() as s:
            infos = PersonGroupService(s).list_groups()
        korus = next(g for g in infos if g.name == "Kórus")
        assert korus.member_count == 2

    def test_cascade_delete_group_removes_memberships(self, db, alice):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid])
        with session_scope() as s:
            PersonGroupService(s).delete_group(gid)
        with session_scope() as s:
            memberships = (
                s.query(PersonGroupMembership)
                .filter_by(group_id=gid)
                .all()
            )
        assert memberships == []

    def test_duplicate_group_ids_deduplicated(self, db, alice):
        with session_scope() as s:
            gid = PersonGroupService(s).create_group("Kórus").id
        with session_scope() as s:
            PersonGroupService(s).set_person_groups(alice, [gid, gid, gid])
        with session_scope() as s:
            memberships = (
                s.query(PersonGroupMembership)
                .filter_by(person_id=alice)
                .all()
            )
        assert len(memberships) == 1
