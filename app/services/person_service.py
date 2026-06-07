"""Service layer for the standalone Persons maintenance page.

Mirrors the structure of :class:`app.services.place_service.PlaceService`:
the service never commits implicitly except where noted; callers control the
session transaction through :func:`session_scope`.  Business logic for the
Persons tab lives here, not in the widget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sqlalchemy import distinct, func, or_
from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonFilters:
    """Filter criteria for the Persons table."""

    name: str = ""
    family_code: str = ""


@dataclass(frozen=True)
class PersonSummary:
    """Flattened person record plus aggregate counts for the table view."""

    person_id: int
    name: str
    thumbnail_path: Optional[str]
    family_code: Optional[str]
    external_family_code: Optional[str]
    last_name: Optional[str]
    first_name: Optional[str]
    second_name: Optional[str]
    nickname: Optional[str]
    married_name: Optional[str]
    gender: Optional[str]
    birth_place: Optional[str]
    birth_date: Optional[str]
    death_place: Optional[str]
    death_date: Optional[str]
    notes: Optional[str]
    is_auto_named: bool
    is_protected: bool
    face_count: int
    image_count: int


@dataclass(frozen=True)
class PersonFaceCrop:
    """A single face crop belonging to a person."""

    face_id: int
    crop_path: Optional[str]
    image_path: Optional[str]
    confidence: float


class PersonService:
    """List, edit, and maintain Person records for the Persons page."""

    # Fields that the maintenance page may write back to a Person row.
    EDITABLE_FIELDS = (
        "gender",
        "family_code",
        "external_family_code",
        "last_name",
        "first_name",
        "second_name",
        "nickname",
        "married_name",
        "birth_place",
        "birth_date",
        "death_place",
        "death_date",
        "notes",
    )

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_persons(
        self, filters: Optional[PersonFilters] = None
    ) -> List[PersonSummary]:
        """Return every person, optionally filtered by name / family code."""
        filters = filters or PersonFilters()
        face_count = func.count(distinct(Face.id)).label("face_count")
        image_count = func.count(distinct(Face.image_id)).label("image_count")

        q = (
            self._session.query(Person, face_count, image_count)
            .outerjoin(Face, Face.person_id == Person.id)
            .group_by(Person.id)
        )

        # Representative face crop per person — used as a thumbnail fallback when
        # the person has no explicit ``thumbnail_path`` yet (the common case for
        # freshly clustered "Unknown N" people).  One query, best confidence wins.
        fallback_crop: dict[int, str] = {}
        crop_rows = (
            self._session.query(Face.person_id, Face.crop_path)
            .filter(
                Face.person_id.isnot(None),
                Face.crop_path.isnot(None),
                Face.is_excluded == False,  # noqa: E712
            )
            .order_by(Face.person_id, Face.confidence.desc())
            .all()
        )
        for pid, crop in crop_rows:
            if pid not in fallback_crop:
                fallback_crop[pid] = crop

        name = filters.name.strip()
        if name:
            term = f"%{name}%"
            q = q.filter(
                or_(
                    Person.name.ilike(term),
                    Person.last_name.ilike(term),
                    Person.first_name.ilike(term),
                    Person.second_name.ilike(term),
                    Person.nickname.ilike(term),
                    Person.married_name.ilike(term),
                )
            )
        code = filters.family_code.strip()
        if code:
            q = q.filter(Person.family_code.ilike(f"%{code}%"))

        rows = q.order_by(Person.name).all()
        return [
            PersonSummary(
                person_id=person.id,
                name=person.name,
                thumbnail_path=person.thumbnail_path or fallback_crop.get(person.id),
                family_code=person.family_code,
                external_family_code=person.external_family_code,
                last_name=person.last_name,
                first_name=person.first_name,
                second_name=person.second_name,
                nickname=person.nickname,
                married_name=person.married_name,
                gender=person.gender,
                birth_place=person.birth_place,
                birth_date=person.birth_date,
                death_place=person.death_place,
                death_date=person.death_date,
                notes=person.notes,
                is_auto_named=bool(person.is_auto_named),
                is_protected=bool(person.is_protected),
                face_count=int(f_count or 0),
                image_count=int(i_count or 0),
            )
            for person, f_count, i_count in rows
        ]

    def list_face_crops(self, person_id: int) -> List[PersonFaceCrop]:
        """Return the face crops assigned to *person_id* (best confidence first)."""
        faces = (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .order_by(Face.confidence.desc())
            .all()
        )
        result: List[PersonFaceCrop] = []
        for face in faces:
            image_path = face.image.file_path if face.image is not None else None
            result.append(
                PersonFaceCrop(
                    face_id=face.id,
                    crop_path=face.crop_path,
                    image_path=image_path,
                    confidence=float(face.confidence or 0.0),
                )
            )
        return result

    def list_images_for_person(self, person_id: int) -> List[str]:
        """Return the distinct source image paths a person appears in."""
        rows = (
            self._session.query(Image)
            .join(Face, Face.image_id == Image.id)
            .filter(Face.person_id == person_id)
            .filter(Face.is_excluded == False)  # noqa: E712
            .group_by(Image.id)
            .order_by(Image.photo_date, Image.file_path)
            .all()
        )
        return [img.file_path for img in rows]

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def rename_person(self, person_id: int, new_name: str) -> Person:
        """Rename a person; protected persons and empty names are rejected."""
        person = self._require_person(person_id)
        clean = (new_name or "").strip()
        if not clean:
            raise ValueError("A név nem lehet üres.")
        if person.is_protected:
            raise ValueError("Védett személy nem nevezhető át.")
        person.name = clean
        person.is_auto_named = False
        self._session.commit()
        log.info("Renamed person %d → %r", person_id, clean)
        return person

    def update_person(self, person_id: int, **fields) -> Person:
        """Update the structured fields of a person.

        Unknown keys are ignored.  ``name`` is handled separately via
        :meth:`rename_person` so this method never touches the display name.
        Empty strings are normalised to ``None``.  A blank ``family_code`` is
        cleared so the unique index does not collide on empty strings.
        """
        person = self._require_person(person_id)
        for key, value in fields.items():
            if key not in self.EDITABLE_FIELDS:
                continue
            if isinstance(value, str):
                value = value.strip() or None
            setattr(person, key, value)
        self._session.flush()
        if "family_code" in fields:
            self._link_derived_parents(person_id)
        self._session.commit()
        log.info("Updated structured data for person %d", person_id)
        return person

    def _link_derived_parents(self, person_id: int) -> None:
        """Materialise relationships implied by a {n}/H family code (best-effort).

        Delegates to :meth:`FamilyService.link_derived_parents`; never raises, so
        a structural issue (e.g. a relationship cycle) cannot block the save.
        """
        from app.services.family_service import FamilyService

        try:
            FamilyService(self._session).link_derived_parents(person_id)
        except Exception:
            log.exception("Could not link derived parents for person %d", person_id)

    # ------------------------------------------------------------------
    # Thumbnail
    # ------------------------------------------------------------------

    def set_thumbnail_from_face(self, person_id: int, face_id: int) -> Person:
        """Use a specific face crop as the person's manual thumbnail.

        Raises ``ValueError`` when the face does not belong to the person or
        its crop file is missing, so the UI can show a Hungarian message
        instead of crashing.
        """
        person = self._require_person(person_id)
        face = self._session.get(Face, face_id)
        if face is None:
            raise ValueError("A kiválasztott arc nem található.")
        if face.person_id != person_id:
            raise ValueError("A kiválasztott arc nem ehhez a személyhez tartozik.")
        if not face.crop_path or not Path(face.crop_path).exists():
            raise ValueError("A kiválasztott arc kivágott képe hiányzik.")
        person.thumbnail_path = face.crop_path
        person.thumbnail_is_manual = True
        self._session.commit()
        log.info("Set person %d thumbnail → face %d (%s)", person_id, face_id, face.crop_path)
        return person

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_person(self, person_id: int) -> Person:
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found")
        return person
