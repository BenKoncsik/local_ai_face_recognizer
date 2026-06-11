"""Service for managing person groups (communities/categories).

Groups are user-defined labels such as "Kórus", "Munkahely", "Iskolai osztály".
A person can belong to any number of groups (many-to-many).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import Person, PersonGroup, PersonGroupMembership

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonGroupInfo:
    """Summary of a group returned by :meth:`PersonGroupService.list_groups`."""

    id: int
    name: str
    description: Optional[str]
    color: Optional[str]
    member_count: int


class PersonGroupService:
    """CRUD + membership operations for person groups.

    All mutating methods call ``session.flush()`` but do **not** commit;
    the caller is responsible for committing (typically via ``session_scope``).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Group CRUD
    # ------------------------------------------------------------------

    def list_groups(self) -> List[PersonGroupInfo]:
        """Return all groups ordered alphabetically with member counts."""
        rows = (
            self._session.query(
                PersonGroup,
                func.count(PersonGroupMembership.person_id).label("cnt"),
            )
            .outerjoin(
                PersonGroupMembership,
                PersonGroupMembership.group_id == PersonGroup.id,
            )
            .group_by(PersonGroup.id)
            .order_by(PersonGroup.name)
            .all()
        )
        return [
            PersonGroupInfo(
                id=g.id,
                name=g.name,
                description=g.description,
                color=g.color,
                member_count=cnt,
            )
            for g, cnt in rows
        ]

    def find_by_name(self, name: str) -> Optional[PersonGroup]:
        """Case-insensitive lookup by exact name."""
        name = name.strip()
        return (
            self._session.query(PersonGroup)
            .filter(func.lower(PersonGroup.name) == name.lower())
            .first()
        )

    def create_group(
        self,
        name: str,
        description: str = "",
        color: str = "",
    ) -> PersonGroup:
        """Create a new group.

        Raises:
            ValueError: if the name is empty or a group with that name already exists.
        """
        name = name.strip()
        if not name:
            raise ValueError("Group name cannot be empty.")
        if self.find_by_name(name) is not None:
            raise ValueError(f"Group '{name}' already exists.")
        group = PersonGroup(
            name=name,
            description=description.strip() or None,
            color=color.strip() or None,
        )
        self._session.add(group)
        self._session.flush()
        log.info("Created person group: %r (id=%d)", name, group.id)
        return group

    def get_or_create(self, name: str) -> PersonGroup:
        """Return the group with this name, creating it if necessary."""
        existing = self.find_by_name(name)
        if existing is not None:
            return existing
        return self.create_group(name)

    def rename_group(self, group_id: int, new_name: str) -> PersonGroup:
        """Rename an existing group.

        Raises:
            ValueError: if *new_name* is empty, the group does not exist,
                        or another group already has that name.
        """
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Group name cannot be empty.")
        group = self._session.get(PersonGroup, group_id)
        if group is None:
            raise ValueError(f"Group id={group_id} not found.")
        conflict = self.find_by_name(new_name)
        if conflict is not None and conflict.id != group_id:
            raise ValueError(f"Group '{new_name}' already exists.")
        group.name = new_name
        self._session.flush()
        return group

    def update_group(
        self,
        group_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
    ) -> PersonGroup:
        """Update a group's name, description and/or color.

        Only the arguments that are not ``None`` are applied, so callers can
        update just the note without touching the name. Pass an empty string
        for *description*/*color* to clear them.

        Raises:
            ValueError: if the group does not exist, *name* is given but empty,
                        or another group already uses *name*.
        """
        group = self._session.get(PersonGroup, group_id)
        if group is None:
            raise ValueError(f"Group id={group_id} not found.")
        if name is not None:
            name = name.strip()
            if not name:
                raise ValueError("Group name cannot be empty.")
            conflict = self.find_by_name(name)
            if conflict is not None and conflict.id != group_id:
                raise ValueError(f"Group '{name}' already exists.")
            group.name = name
        if description is not None:
            group.description = description.strip() or None
        if color is not None:
            group.color = color.strip() or None
        self._session.flush()
        log.info("Updated person group id=%d", group_id)
        return group

    def delete_group(self, group_id: int) -> None:
        """Delete a group and all its memberships (cascade).

        Raises:
            ValueError: if the group does not exist.
        """
        group = self._session.get(PersonGroup, group_id)
        if group is None:
            raise ValueError(f"Group id={group_id} not found.")
        self._session.delete(group)
        self._session.flush()
        log.info("Deleted person group id=%d", group_id)

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def get_person_groups(self, person_id: int) -> List[PersonGroup]:
        """Return the groups that *person_id* belongs to, alphabetically."""
        return (
            self._session.query(PersonGroup)
            .join(
                PersonGroupMembership,
                PersonGroupMembership.group_id == PersonGroup.id,
            )
            .filter(PersonGroupMembership.person_id == person_id)
            .order_by(PersonGroup.name)
            .all()
        )

    def set_person_groups(self, person_id: int, group_ids: List[int]) -> None:
        """Replace all group memberships for *person_id* with *group_ids*.

        The operation is transactional within the current session:
        existing memberships are deleted, then the new ones are inserted.

        Raises:
            ValueError: if *person_id* is for a protected person.
        """
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found.")
        if person.is_protected:
            raise ValueError(
                f"Cannot assign groups to protected person '{person.name}'."
            )

        self._session.query(PersonGroupMembership).filter(
            PersonGroupMembership.person_id == person_id
        ).delete(synchronize_session="fetch")

        seen: set[int] = set()
        for gid in group_ids:
            if gid in seen:
                continue
            seen.add(gid)
            self._session.add(
                PersonGroupMembership(person_id=person_id, group_id=gid)
            )
        self._session.flush()
        log.debug(
            "set_person_groups: person=%d groups=%s", person_id, list(seen)
        )

    def get_persons_in_group(self, group_id: int) -> List[Person]:
        """Return all persons in *group_id*, ordered by name."""
        return (
            self._session.query(Person)
            .join(
                PersonGroupMembership,
                PersonGroupMembership.person_id == Person.id,
            )
            .filter(PersonGroupMembership.group_id == group_id)
            .order_by(Person.name)
            .all()
        )
