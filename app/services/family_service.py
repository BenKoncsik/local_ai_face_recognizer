"""Family relationship and person/image search service."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from sqlalchemy import or_, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person, Place, Relationship
from app.services.family_code_interpreter import (
    parse_extended_code,
    validate_extended_code,
    validate_external_family_code,
)

REL_PARENT_CHILD = "ParentChild"
REL_SPOUSE = "Spouse"

RelationshipFilter = Literal["any", "spouse", "parent", "child", "sibling"]
ImageSearchMode = Literal["both", "exact"]

# Simple descendant/spouse codes (C, C0, C8, C85, C80): one root letter, child
# indexes, optional trailing spouse 0. Used by parse_family_code to decide
# whether a code has a simple descendant-tree derivation (extended F/T/B codes
# do not). Validation itself lives in the family_code_interpreter.
_FAMILY_CODE_RE = re.compile(r"^[A-Z](?:0|[1-9]+0?)?$")


@dataclass(frozen=True)
class FamilyCodeInfo:
    code: str
    root: str
    path: str
    generation: int
    parent_code: Optional[str]
    spouse_of_code: Optional[str]
    # The other parent, derived from the {n} brace notation (X{n}m → XHn).
    # None for codes without a parent-disambiguation brace.
    second_parent_code: Optional[str] = None

    @property
    def is_spouse(self) -> bool:
        return self.spouse_of_code is not None


@dataclass(frozen=True)
class FamilyImageResult:
    image_id: int
    file_path: str
    filename: str
    width: Optional[int]
    height: Optional[int]
    person_count: int
    face_count: int


@dataclass(frozen=True)
class FamilyImageSearchCriteria:
    """Detailed filters for the family image search panel."""

    name_terms: tuple[str, ...] = ()
    allow_other_people: bool = True
    person_text: str = ""
    gender: Optional[str] = None
    family_code: str = ""
    image_text: str = ""
    photo_date: str = ""
    place_text: str = ""
    relationship_filter: RelationshipFilter = "any"
    # Free-text terms from the universal search bar that match any field
    # (person name, image path, photo date, place name).
    any_terms: tuple[str, ...] = ()


def normalize_family_code(value: Optional[str]) -> Optional[str]:
    """Return the canonical FamilyCode form, or None for an empty value."""
    if value is None:
        return None
    code = value.strip().upper()
    return code or None


def validate_family_code(value: Optional[str]) -> Optional[str]:
    """Validate and canonicalise a FamilyCode.

    Supports the full family relationship grammar:
      - simple descendant/spouse codes: C, C0, C8, C85, C80 (trailing 0 = spouse);
      - ancestors (F): C0F1, C0F11, C00F2, ...;
      - siblings / collateral relatives (T): C00T1, C00T11, C810T4, ...;
      - friends / acquaintances (B): C81B;
      - shared-friend multi-codes: ``C81B,C82B`` / ``C81B C82B``;
      - range notation: ``C[1-9]B``.

    Extended codes are delegated to validate_extended_code, which owns the
    full grammar. Returns None for an empty value; raises ValueError otherwise.
    """
    code = normalize_family_code(value)
    if code is None:
        return None
    # The interpreter owns the full grammar (descendants, spouse-0, stepchild,
    # ancestors F, siblings T, friends B, multi-codes and ranges) including the
    # structural 0-rules, so all validation is delegated to it.
    return validate_extended_code(code)


def parse_family_code(value: Optional[str]) -> Optional[FamilyCodeInfo]:
    """Parse a FamilyCode into derived relationship hints."""
    code = validate_family_code(value)
    if code is None:
        return None
    # Simple descendant/spouse codes (C, C0, C8, C85, C80) keep the original
    # derivation. Numbered-spouse (H) and parent-disambiguated ({n}) codes are
    # also descendant-tree codes and get their own derivation below; the
    # remaining extended codes (F/T/B suffixes, multi-codes, ranges, external
    # #...#) describe ancestors, collateral relatives or friends and have no
    # simple parent/spouse derivation.
    if not _FAMILY_CODE_RE.fullmatch(code):
        return _parse_extended_tree_code(code)

    root = code[0]
    raw_path = code[1:]
    spouse_of_code: Optional[str] = None
    child_path = raw_path
    is_spouse = False
    if raw_path.endswith("0"):
        is_spouse = True
        spouse_of_code = root + raw_path[:-1]
        child_path = raw_path[:-1]

    parent_code: Optional[str] = None
    if child_path and not is_spouse:
        parent_code = root if len(child_path) == 1 else root + child_path[:-1]

    return FamilyCodeInfo(
        code=code,
        root=root,
        path=raw_path,
        generation=len(child_path),
        parent_code=parent_code,
        spouse_of_code=spouse_of_code,
    )


def _parse_extended_tree_code(code: str) -> Optional[FamilyCodeInfo]:
    """Derive relationship hints for numbered-spouse (H) and braced ({n}) codes.

    ``G1H2``  → spouse of the base person ``G1``.
    ``G1{2}3`` → child of ``G1`` (first parent) and ``G1H2`` (second parent,
    derived from the brace). Returns None for any other extended code.
    """
    try:
        info = parse_extended_code(code)
    except ValueError:
        return None
    if info.is_external:
        return None
    root = info.root
    digits = info.path_digits
    path = code[len(root):]

    # Numbered spouse, e.g. G1H2 = the 2nd spouse of G1.  Brought-children of a
    # numbered spouse (G1H21) are not simple tree nodes, so only the bare XHn
    # form is derived here.
    if info.suffix_type == "spouse":
        if len(info.spouse_path) != 1:
            return None
        base_code = root + "".join(str(d) for d in digits)
        return FamilyCodeInfo(
            code=code,
            root=root,
            path=path,
            generation=len(digits),
            parent_code=None,
            spouse_of_code=base_code,
        )

    # Parent-disambiguated child, e.g. G1{2}3.  The terminal child carries the
    # spouse-number annotation; the first parent is the code without that child,
    # the second parent is firstParent + Hn.
    if info.suffix_type is None and any(info.path_spouse_nums):
        if not digits or digits[-1] == 0:
            return None
        first_parent = root + "".join(str(d) for d in digits[:-1])
        spouse_num = info.path_spouse_nums[-1]
        second_parent = f"{first_parent}H{spouse_num}" if spouse_num else None
        return FamilyCodeInfo(
            code=code,
            root=root,
            path=path,
            generation=len(digits),
            parent_code=first_parent or None,
            spouse_of_code=None,
            second_parent_code=second_parent,
        )

    return None


def family_codes_are_siblings(code_a: Optional[str], code_b: Optional[str]) -> bool:
    """Return True when two non-spouse FamilyCodes have the same derived parent."""
    info_a = parse_family_code(code_a)
    info_b = parse_family_code(code_b)
    if info_a is None or info_b is None or info_a.code == info_b.code:
        return False
    if info_a.is_spouse or info_b.is_spouse:
        return False
    return info_a.parent_code is not None and info_a.parent_code == info_b.parent_code


class FamilyService:
    """Validated family relationships plus optimized image/person queries."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # FamilyCode helpers
    # ------------------------------------------------------------------

    def parse_person_code(self, person_id: int) -> Optional[FamilyCodeInfo]:
        person = self._require_person(person_id)
        return parse_family_code(person.family_code)

    def find_by_family_code(self, family_code: Optional[str]) -> Optional[Person]:
        code = validate_family_code(family_code)
        if code is None:
            return None
        return (
            self._session.query(Person)
            .filter(Person.family_code == code)
            .first()
        )

    def ensure_unique_family_code(
        self,
        family_code: Optional[str],
        *,
        current_person_id: Optional[int] = None,
    ) -> Optional[str]:
        code = validate_family_code(family_code)
        if code is None:
            return None
        # Friend/acquaintance codes (B) are relational markers, not unique
        # identities — several different external people may share e.g. "C81B".
        # Only the actual family identity codes are enforced as unique.
        if "B" in code:
            return code
        q = self._session.query(Person).filter(Person.family_code == code)
        if current_person_id is not None:
            q = q.filter(Person.id != current_person_id)
        existing = q.first()
        if existing is not None:
            raise ValueError(
                f"Family code {code} is already assigned to {existing.name}."
            )
        return code

    def ensure_unique_external_family_code(
        self,
        external_code: Optional[str],
        *,
        current_person_id: Optional[int] = None,
    ) -> Optional[str]:
        """Validate an external family identifier and enforce uniqueness.

        Each ``#root#path`` is a unique external identity, so (unlike friend
        codes) every external code must be unique across persons. Returns the
        canonical form, or None for an empty value; raises ValueError otherwise.
        """
        code = validate_external_family_code(external_code)
        if code is None:
            return None
        q = self._session.query(Person).filter(Person.external_family_code == code)
        if current_person_id is not None:
            q = q.filter(Person.id != current_person_id)
        existing = q.first()
        if existing is not None:
            raise ValueError(
                f"External family code {code} is already assigned to {existing.name}."
            )
        return code

    def family_root(self, family_code: Optional[str]) -> Optional[str]:
        info = parse_family_code(family_code)
        return info.root if info is not None else None

    def generation_level(self, family_code: Optional[str]) -> Optional[int]:
        info = parse_family_code(family_code)
        return info.generation if info is not None else None

    def parent_family_code(self, family_code: Optional[str]) -> Optional[str]:
        info = parse_family_code(family_code)
        return info.parent_code if info is not None else None

    def spouse_family_code(self, family_code: Optional[str]) -> Optional[str]:
        info = parse_family_code(family_code)
        return info.spouse_of_code if info is not None else None

    def list_children_by_family_code(self, family_code: Optional[str]) -> list[Person]:
        code = validate_family_code(family_code)
        if code is None:
            return []
        child_pattern = f"{code}_"
        spouse_pattern = f"{code}0"
        return (
            self._session.query(Person)
            .filter(Person.family_code.like(child_pattern))
            .filter(Person.family_code != spouse_pattern)
            .all()
        )

    def list_siblings_by_family_code(self, family_code: Optional[str]) -> list[Person]:
        info = parse_family_code(family_code)
        if info is None or info.parent_code is None or info.is_spouse:
            return []
        sibling_pattern = f"{info.parent_code}_"
        spouse_pattern = f"{info.parent_code}%0"
        return (
            self._session.query(Person)
            .filter(Person.family_code.like(sibling_pattern))
            .filter(Person.family_code != info.code)
            .filter(~Person.family_code.like(spouse_pattern))
            .all()
        )

    def list_branch_by_family_code(self, family_code: Optional[str]) -> list[Person]:
        code = validate_family_code(family_code)
        if code is None:
            return []
        return (
            self._session.query(Person)
            .filter(Person.family_code.like(f"{code}%"))
            .order_by(Person.family_code)
            .all()
        )

    def codes_are_siblings(
        self,
        family_code_a: Optional[str],
        family_code_b: Optional[str],
    ) -> bool:
        return family_codes_are_siblings(family_code_a, family_code_b)

    # ------------------------------------------------------------------
    # Relationship commands
    # ------------------------------------------------------------------

    def add_spouse(self, person_id_a: int, person_id_b: int) -> Relationship:
        if person_id_a == person_id_b:
            raise ValueError("A person cannot be their own spouse.")
        self._require_person(person_id_a)
        self._require_person(person_id_b)
        a_id, b_id = sorted((person_id_a, person_id_b))
        return self._get_or_create_relationship(REL_SPOUSE, a_id, b_id)

    def add_parent_child(self, parent_id: int, child_id: int) -> Relationship:
        if parent_id == child_id:
            raise ValueError("A person cannot be their own parent.")
        self._require_person(parent_id)
        self._require_person(child_id)
        if self._has_ancestor(parent_id, child_id):
            raise ValueError("This parent/child relationship would create a cycle.")
        return self._get_or_create_relationship(REL_PARENT_CHILD, parent_id, child_id)

    def link_derived_parents(self, person_id: int) -> list[Relationship]:
        """Materialise the relationships a person's family code implies.

        For a parent-disambiguated child code (``G1{2}3``) this links both the
        first parent (``G1``) and the brace-derived second parent (``G1H2``) as
        ParentChild, and links the two parents to each other as spouses.  For a
        numbered-spouse code (``G1H2``) it links the spouse relationship with the
        base person (``G1``).

        Linking is best-effort: only relationships whose other person already
        exists (matched by family code) are created.  Returns the list of
        relationships created or already present.  Self-references and missing
        persons are skipped silently; genuine errors (e.g. a relationship cycle)
        propagate so the caller can surface them.
        """
        person = self._require_person(person_id)
        info = parse_family_code(person.family_code)
        if info is None:
            return []

        created: list[Relationship] = []

        # Numbered-spouse code (G1H2): link as spouse of the base person.
        if info.spouse_of_code and info.parent_code is None:
            base = self.find_by_family_code(info.spouse_of_code)
            if base is not None and base.id != person.id:
                created.append(self.add_spouse(base.id, person.id))
            return created

        # Braced child (G1{2}3): link both parents, and the parents as spouses.
        parent_codes = [c for c in (info.parent_code, info.second_parent_code) if c]
        parent_persons: list[Person] = []
        for parent_code in parent_codes:
            parent = self.find_by_family_code(parent_code)
            if parent is None or parent.id == person.id:
                continue
            parent_persons.append(parent)
            created.append(self.add_parent_child(parent.id, person.id))

        if len(parent_persons) == 2:
            created.append(
                self.add_spouse(parent_persons[0].id, parent_persons[1].id)
            )

        return created

    def are_spouses(self, person_id_a: int, person_id_b: int) -> bool:
        if person_id_a == person_id_b:
            return False
        a_id, b_id = sorted((person_id_a, person_id_b))
        stored = (
            self._session.query(Relationship.id)
            .filter(
                Relationship.relationship_type == REL_SPOUSE,
                Relationship.person_a_id == a_id,
                Relationship.person_b_id == b_id,
            )
            .first()
            is not None
        )
        if stored:
            return True
        person_a = self._session.get(Person, person_id_a)
        person_b = self._session.get(Person, person_id_b)
        info_a = self._safe_parse_person_code(person_a)
        info_b = self._safe_parse_person_code(person_b)
        if info_a is None or info_b is None:
            return False
        return info_a.spouse_of_code == info_b.code or info_b.spouse_of_code == info_a.code

    def is_parent_child(self, parent_id: int, child_id: int) -> bool:
        if parent_id == child_id:
            return False
        stored = (
            self._session.query(Relationship.id)
            .filter(
                Relationship.relationship_type == REL_PARENT_CHILD,
                Relationship.person_a_id == parent_id,
                Relationship.person_b_id == child_id,
            )
            .first()
            is not None
        )
        if stored:
            return True
        parent = self._session.get(Person, parent_id)
        child = self._session.get(Person, child_id)
        parent_info = self._safe_parse_person_code(parent)
        child_info = self._safe_parse_person_code(child)
        if parent_info is None or child_info is None or child_info.is_spouse:
            return False
        return parent_info.code in (
            child_info.parent_code,
            child_info.second_parent_code,
        )

    def are_siblings(self, person_id_a: int, person_id_b: int) -> bool:
        if person_id_a == person_id_b:
            return False
        person_a = self._session.get(Person, person_id_a)
        person_b = self._session.get(Person, person_id_b)
        info_a = self._safe_parse_person_code(person_a)
        info_b = self._safe_parse_person_code(person_b)
        if info_a is not None and info_b is not None:
            if family_codes_are_siblings(info_a.code, info_b.code):
                return True
        common_parent = (
            self._session.query(Relationship.person_a_id)
            .filter(
                Relationship.relationship_type == REL_PARENT_CHILD,
                Relationship.person_b_id == person_id_a,
                Relationship.person_a_id.in_(
                    self._session.query(Relationship.person_a_id).filter(
                        Relationship.relationship_type == REL_PARENT_CHILD,
                        Relationship.person_b_id == person_id_b,
                    )
                ),
            )
            .first()
        )
        return common_parent is not None

    def relationship_matches(
        self,
        person_id_a: int,
        person_id_b: int,
        relationship_filter: RelationshipFilter,
    ) -> bool:
        if relationship_filter == "any":
            return True
        if relationship_filter == "spouse":
            return self.are_spouses(person_id_a, person_id_b)
        if relationship_filter == "parent":
            return self.is_parent_child(person_id_a, person_id_b)
        if relationship_filter == "child":
            return self.is_parent_child(person_id_b, person_id_a)
        if relationship_filter == "sibling":
            return self.are_siblings(person_id_a, person_id_b)
        return False

    def describe_relationship(self, person_id_a: int, person_id_b: int) -> str:
        """Return a compact Hungarian relationship summary for display."""
        a = self._session.get(Person, person_id_a)
        b = self._session.get(Person, person_id_b)
        if a is None or b is None or person_id_a == person_id_b:
            return ""
        parts: list[str] = []
        if self.are_spouses(person_id_a, person_id_b):
            parts.append("házastársak")
        if self.is_parent_child(person_id_a, person_id_b):
            parts.append(f"{a.name}: {self._parent_label(a)}")
        if self.is_parent_child(person_id_b, person_id_a):
            parts.append(f"{a.name}: {self._child_label(a)}")
        if self.are_siblings(person_id_a, person_id_b):
            parts.append("testvérek")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Image queries
    # ------------------------------------------------------------------

    def search_images_for_people(
        self,
        person_id_a: int,
        person_id_b: int,
        *,
        mode: ImageSearchMode = "both",
        relationship_filter: RelationshipFilter = "any",
        limit: int = 50,
        offset: int = 0,
        folder_path: Optional[str] = None,
        place_id: Optional[int] = None,
    ) -> tuple[list[FamilyImageResult], int]:
        """Find images containing both people.

        The filtering is performed in SQL via grouped ``faces`` queries.  In
        ``exact`` mode only images with exactly these two assigned people are
        returned; unassigned faces do not count as a third person.
        """
        if person_id_a == person_id_b:
            return [], 0
        if not self.relationship_matches(person_id_a, person_id_b, relationship_filter):
            return [], 0

        matched = (
            self._session.query(Face.image_id.label("image_id"))
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.in_([person_id_a, person_id_b]))
            .group_by(Face.image_id)
            .having(func.count(func.distinct(Face.person_id)) == 2)
            .subquery()
        )

        totals = (
            self._session.query(
                Face.image_id.label("image_id"),
                func.count(Face.id).label("face_count"),
                func.count(func.distinct(Face.person_id)).label("person_count"),
            )
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.isnot(None))
            .group_by(Face.image_id)
            .subquery()
        )

        q = (
            self._session.query(
                Image.id,
                Image.file_path,
                Image.width,
                Image.height,
                totals.c.person_count,
                totals.c.face_count,
            )
            .join(matched, matched.c.image_id == Image.id)
            .join(totals, totals.c.image_id == Image.id)
        )
        if mode == "exact":
            q = q.filter(totals.c.person_count == 2)
        if place_id is not None:
            q = q.filter(Image.place_id == place_id)
        if folder_path:
            safe_prefix = folder_path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            q = q.filter(Image.file_path.like(f"{safe_prefix}%", escape="\\"))

        total = q.count()
        rows = q.order_by(Image.file_path).limit(limit).offset(offset).all()

        results = [
            FamilyImageResult(
                image_id=row.id,
                file_path=row.file_path,
                filename=Path(row.file_path).name,
                width=row.width,
                height=row.height,
                person_count=int(row.person_count or 0),
                face_count=int(row.face_count or 0),
            )
            for row in rows
            if not folder_path or str(Path(row.file_path).parent) == folder_path
        ]
        return results, total

    def search_images_by_criteria(
        self,
        criteria: FamilyImageSearchCriteria,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[FamilyImageResult], int]:
        """Find images matching comma-separated person names plus detailed filters."""
        terms = tuple(term.strip() for term in criteria.name_terms if term.strip())
        term_person_ids: list[list[int]] = []
        for term in terms:
            ids = self._person_ids_matching_name(term)
            if not ids:
                return [], 0
            term_person_ids.append(ids)

        totals = self._image_person_totals_subquery()
        q = (
            self._session.query(
                Image.id,
                Image.file_path,
                Image.width,
                Image.height,
                totals.c.person_count,
                totals.c.face_count,
            )
            .outerjoin(totals, totals.c.image_id == Image.id)
        )

        for idx, person_ids in enumerate(term_person_ids):
            matched = (
                self._session.query(Face.image_id.label(f"image_id_{idx}"))
                .filter(Face.is_excluded == False)  # noqa: E712
                .filter(Face.person_id.in_(person_ids))
                .group_by(Face.image_id)
                .subquery()
            )
            q = q.join(matched, matched.c[f"image_id_{idx}"] == Image.id)

        allowed_person_ids = {pid for ids in term_person_ids for pid in ids}
        if terms and not criteria.allow_other_people:
            extra_people = (
                self._session.query(Face.image_id)
                .filter(Face.is_excluded == False)  # noqa: E712
                .filter(Face.person_id.isnot(None))
                .filter(~Face.person_id.in_(allowed_person_ids))
                .subquery()
            )
            q = q.outerjoin(extra_people, extra_people.c.image_id == Image.id)
            q = q.filter(extra_people.c.image_id.is_(None))

        if criteria.relationship_filter != "any" and len(term_person_ids) == 2:
            pair_image_ids = self._relationship_filtered_image_ids(
                term_person_ids[0],
                term_person_ids[1],
                criteria.relationship_filter,
            )
            if not pair_image_ids:
                return [], 0
            q = q.filter(Image.id.in_(pair_image_ids))

        if criteria.person_text.strip():
            q = q.filter(Image.id.in_(self._image_ids_matching_person_text(criteria.person_text)))
        if criteria.gender:
            q = q.filter(Image.id.in_(self._image_ids_matching_person_fields(Person.gender == criteria.gender)))
        if criteria.family_code.strip():
            pattern = self._like_pattern(criteria.family_code)
            q = q.filter(Image.id.in_(self._image_ids_matching_person_fields(Person.family_code.ilike(pattern))))
        if criteria.image_text.strip():
            pattern = self._like_pattern(criteria.image_text)
            q = q.filter(
                or_(
                    Image.file_path.ilike(pattern),
                    Image.relative_path.ilike(pattern),
                    Image.photo_date.ilike(pattern),
                    Image.place.has(Place.name.ilike(pattern)),
                )
            )
        if criteria.photo_date.strip():
            q = q.filter(Image.photo_date.ilike(self._like_pattern(criteria.photo_date)))
        if criteria.place_text.strip():
            q = q.filter(Image.place.has(Place.name.ilike(self._like_pattern(criteria.place_text))))

        for term in criteria.any_terms:
            pattern = self._like_pattern(term)
            person_ids = self._person_ids_matching_name(term)
            conditions = [
                Image.file_path.ilike(pattern),
                Image.photo_date.ilike(pattern),
                Image.place.has(Place.name.ilike(pattern)),
            ]
            if person_ids:
                face_subq = (
                    self._session.query(Face.image_id)
                    .filter(Face.is_excluded == False)  # noqa: E712
                    .filter(Face.person_id.in_(person_ids))
                    .subquery()
                )
                conditions.append(Image.id.in_(face_subq))
            q = q.filter(or_(*conditions))

        total = q.count()
        rows = q.order_by(Image.file_path).limit(limit).offset(offset).all()
        return [self._row_to_result(row) for row in rows], total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _person_ids_matching_name(self, term: str) -> list[int]:
        pattern = self._like_pattern(term)
        return [
            row[0]
            for row in self._session.query(Person.id)
            .filter(
                or_(
                    Person.name.ilike(pattern),
                    Person.last_name.ilike(pattern),
                    Person.first_name.ilike(pattern),
                    Person.second_name.ilike(pattern),
                    Person.nickname.ilike(pattern),
                    Person.married_name.ilike(pattern),
                    Person.family_code.ilike(pattern),
                )
            )
            .all()
        ]

    def _image_person_totals_subquery(self):
        return (
            self._session.query(
                Face.image_id.label("image_id"),
                func.count(Face.id).label("face_count"),
                func.count(func.distinct(Face.person_id)).label("person_count"),
            )
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(Face.person_id.isnot(None))
            .group_by(Face.image_id)
            .subquery()
        )

    def _image_ids_matching_person_text(self, text: str):
        pattern = self._like_pattern(text)
        return self._image_ids_matching_person_fields(
            or_(
                Person.name.ilike(pattern),
                Person.last_name.ilike(pattern),
                Person.first_name.ilike(pattern),
                Person.second_name.ilike(pattern),
                Person.nickname.ilike(pattern),
                Person.married_name.ilike(pattern),
                Person.birth_place.ilike(pattern),
                Person.birth_date.ilike(pattern),
                Person.death_place.ilike(pattern),
                Person.death_date.ilike(pattern),
                Person.notes.ilike(pattern),
                Person.family_code.ilike(pattern),
            )
        )

    def _image_ids_matching_person_fields(self, *conditions):
        return (
            self._session.query(Face.image_id)
            .join(Person, Face.person_id == Person.id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .filter(*conditions)
            .distinct()
        )

    def _relationship_filtered_image_ids(
        self,
        left_ids: list[int],
        right_ids: list[int],
        relationship_filter: RelationshipFilter,
    ) -> set[int]:
        image_ids: set[int] = set()
        for left_id in left_ids:
            for right_id in right_ids:
                if left_id == right_id:
                    continue
                if not self.relationship_matches(left_id, right_id, relationship_filter):
                    continue
                rows = (
                    self._session.query(Face.image_id)
                    .filter(Face.is_excluded == False)  # noqa: E712
                    .filter(Face.person_id.in_([left_id, right_id]))
                    .group_by(Face.image_id)
                    .having(func.count(func.distinct(Face.person_id)) == 2)
                    .all()
                )
                image_ids.update(row[0] for row in rows)
        return image_ids

    @staticmethod
    def _like_pattern(value: str) -> str:
        escaped = value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    @staticmethod
    def _row_to_result(row) -> FamilyImageResult:
        return FamilyImageResult(
            image_id=row.id,
            file_path=row.file_path,
            filename=Path(row.file_path).name,
            width=row.width,
            height=row.height,
            person_count=int(row.person_count or 0),
            face_count=int(row.face_count or 0),
        )

    # ------------------------------------------------------------------
    # Autocomplete suggestions
    # ------------------------------------------------------------------

    def get_search_suggestions(
        self, query: str, max_per_type: int = 5
    ) -> list[tuple[str, str, str]]:
        """Return autocomplete suggestions for the universal search bar.

        Returns a list of (value, token_type, display_text) tuples.
        token_type is one of: "person", "nickname", "family_code",
        "place", "date", "image".
        """
        if not query.strip():
            return []
        pattern = f"%{query.strip()}%"
        results: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()

        def _add(value: str, token_type: str) -> None:
            if value and (value, token_type) not in seen:
                seen.add((value, token_type))
                results.append((value, token_type, value))

        for person in (
            self._session.query(Person)
            .filter(
                or_(
                    Person.name.ilike(pattern),
                    Person.first_name.ilike(pattern),
                    Person.last_name.ilike(pattern),
                    Person.second_name.ilike(pattern),
                )
            )
            .limit(max_per_type)
            .all()
        ):
            _add(person.name, "person")

        for (nick,) in (
            self._session.query(Person.nickname)
            .filter(Person.nickname.ilike(pattern), Person.nickname.isnot(None))
            .distinct()
            .limit(max_per_type)
            .all()
        ):
            _add(nick, "nickname")

        for (code,) in (
            self._session.query(Person.family_code)
            .filter(Person.family_code.ilike(pattern), Person.family_code.isnot(None))
            .distinct()
            .limit(max_per_type)
            .all()
        ):
            _add(code, "family_code")

        for (pname,) in (
            self._session.query(Place.name)
            .filter(Place.name.ilike(pattern))
            .limit(max_per_type)
            .all()
        ):
            _add(pname, "place")

        for (date,) in (
            self._session.query(Image.photo_date)
            .filter(Image.photo_date.ilike(pattern), Image.photo_date.isnot(None))
            .distinct()
            .limit(max_per_type)
            .all()
        ):
            _add(date, "date")

        q_lower = query.strip().lower()
        for (fpath,) in (
            self._session.query(Image.file_path)
            .filter(Image.file_path.ilike(pattern))
            .limit(max_per_type * 3)
            .all()
        ):
            fname = Path(fpath).name
            if q_lower in fname.lower():
                _add(fname, "image")
            if len([r for r in results if r[1] == "image"]) >= max_per_type:
                break

        return results

    def _get_or_create_relationship(
        self,
        relationship_type: str,
        person_a_id: int,
        person_b_id: int,
    ) -> Relationship:
        rel = (
            self._session.query(Relationship)
            .filter(
                Relationship.relationship_type == relationship_type,
                Relationship.person_a_id == person_a_id,
                Relationship.person_b_id == person_b_id,
            )
            .first()
        )
        if rel is not None:
            return rel
        rel = Relationship(
            relationship_type=relationship_type,
            person_a_id=person_a_id,
            person_b_id=person_b_id,
        )
        self._session.add(rel)
        try:
            self._session.flush()
        except IntegrityError:
            self._session.rollback()
            rel = (
                self._session.query(Relationship)
                .filter(
                    Relationship.relationship_type == relationship_type,
                    Relationship.person_a_id == person_a_id,
                    Relationship.person_b_id == person_b_id,
                )
                .one()
            )
        return rel

    def _has_ancestor(self, person_id: int, ancestor_id: int) -> bool:
        pending = [person_id]
        seen: set[int] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            parent_ids = [
                row[0]
                for row in self._session.query(Relationship.person_a_id)
                .filter(
                    Relationship.relationship_type == REL_PARENT_CHILD,
                    Relationship.person_b_id == current,
                )
                .all()
            ]
            if ancestor_id in parent_ids:
                return True
            pending.extend(parent_ids)
        return False

    def _require_person(self, person_id: int) -> Person:
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person not found: {person_id}")
        return person

    @staticmethod
    def _safe_parse_person_code(person: Optional[Person]) -> Optional[FamilyCodeInfo]:
        if person is None or not person.family_code:
            return None
        try:
            return parse_family_code(person.family_code)
        except ValueError:
            return None

    @staticmethod
    def _parent_label(person: Person) -> str:
        if person.gender == "male":
            return "apa"
        if person.gender == "female":
            return "anya"
        return "szülő"

    @staticmethod
    def _child_label(person: Person) -> str:
        if person.gender == "male":
            return "fiúgyermek"
        if person.gender == "female":
            return "lánygyermek"
        return "gyerek"
