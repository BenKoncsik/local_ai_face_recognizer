"""Service layer for the Object Tagging domain.

Objects are arbitrary user-tagged things in photos (a car, a boat, a house, a
gravestone, …).  They are a domain entirely separate from face recognition:
no embeddings, no biometric data, no participation in any detection/clustering
step.

Mirrors the structure of :class:`app.services.place_service.PlaceService` and
:class:`app.services.person_group_service.PersonGroupService`: the service takes
a :class:`Session`, mutating methods call ``flush()`` but do **not** commit
(the caller controls the transaction via ``session_scope``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

from sqlalchemy import distinct, func
from sqlalchemy.orm import Session

from app.db.models import (
    OBJECT_GEOMETRY_BBOX,
    OBJECT_ROLES,
    Image,
    ObjectAlias,
    ObjectOccurrence,
    ObjectPersonLink,
    Person,
    TaggedObject,
)
from app.utils.person_search import normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectFilters:
    """Filter criteria for the objects table."""

    name: str = ""


@dataclass(frozen=True)
class ObjectSummary:
    """Flattened object record plus aggregate counts for the table view."""

    object_id: int
    name: str
    description: Optional[str]
    notes: Optional[str]
    thumbnail_path: Optional[str]
    image_count: int
    occurrence_count: int
    note_count: int
    person_count: int
    created_at: Optional[str]
    updated_at: Optional[str]


@dataclass(frozen=True)
class ObjectOccurrenceInfo:
    """A single occurrence of an object in an image."""

    occurrence_id: int
    object_id: int
    image_id: int
    image_path: Optional[str]
    photo_date: Optional[str]
    point_x: Optional[int]
    point_y: Optional[int]
    note: Optional[str]
    bbox_x: Optional[int] = None
    bbox_y: Optional[int] = None
    bbox_w: Optional[int] = None
    bbox_h: Optional[int] = None


@dataclass(frozen=True)
class ObjectPersonInfo:
    """A person linked to an object, with the role of the link."""

    person_id: int
    name: str
    role: str
    note: Optional[str]


@dataclass(frozen=True)
class PersonObjectInfo:
    """An object linked to a person (for the person detail view)."""

    object_id: int
    name: str
    role: str
    note: Optional[str]


class ObjectService:
    """CRUD + occurrence + person-link + merge operations for tagged objects."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Object CRUD
    # ------------------------------------------------------------------

    def find_by_name(self, name: str) -> Optional[TaggedObject]:
        """Case-insensitive lookup by exact name."""
        name = (name or "").strip()
        if not name:
            return None
        return (
            self._session.query(TaggedObject)
            .filter(func.lower(TaggedObject.name) == name.lower())
            .first()
        )

    def create_object(
        self,
        name: str,
        description: str = "",
        notes: str = "",
    ) -> TaggedObject:
        """Create a new object.

        Raises:
            ValueError: if *name* is empty.
        """
        name = (name or "").strip()
        if not name:
            raise ValueError("Az objektum neve nem lehet üres.")
        obj = TaggedObject(
            name=name,
            description=(description or "").strip() or None,
            notes=(notes or "").strip() or None,
        )
        self._session.add(obj)
        self._session.flush()
        log.info("Created tagged object %r (id=%d)", name, obj.id)
        return obj

    def get_or_create(self, name: str) -> TaggedObject:
        """Return the object with this name, creating it if necessary."""
        existing = self.find_by_name(name)
        if existing is not None:
            return existing
        return self.create_object(name)

    def update_object(
        self,
        object_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> TaggedObject:
        """Update the editable fields of an object.

        ``None`` means "leave unchanged"; an empty string clears the field
        (except *name*, which may never be blanked).
        """
        obj = self._require_object(object_id)
        if name is not None:
            clean = name.strip()
            if not clean:
                raise ValueError("Az objektum neve nem lehet üres.")
            obj.name = clean
        if description is not None:
            obj.description = description.strip() or None
        if notes is not None:
            obj.notes = notes.strip() or None
        self._session.flush()
        log.info("Updated tagged object %d", object_id)
        return obj

    def set_thumbnail_path(
        self, object_id: int, path: Optional[str], *, manual: bool = False
    ) -> TaggedObject:
        """Set (or clear) the object's thumbnail image path."""
        obj = self._require_object(object_id)
        obj.thumbnail_path = (path or None)
        obj.thumbnail_is_manual = bool(manual and path)
        self._session.flush()
        return obj

    def delete_object(self, object_id: int) -> None:
        """Delete an object and all its occurrences/links (cascade)."""
        obj = self._require_object(object_id)
        self._session.delete(obj)
        self._session.flush()
        log.info("Deleted tagged object %d", object_id)

    # ------------------------------------------------------------------
    # Querying / aggregation
    # ------------------------------------------------------------------

    def list_objects(
        self, filters: Optional[ObjectFilters] = None
    ) -> List[ObjectSummary]:
        """Return every object with aggregate counts, ordered by name."""
        filters = filters or ObjectFilters()

        occ_count = func.count(distinct(ObjectOccurrence.id)).label("occ_count")
        image_count = func.count(distinct(ObjectOccurrence.image_id)).label("image_count")

        # SQLite-friendly note counting: count occurrence ids whose note is set.
        note_count = func.count(
            distinct(
                func.iif(  # type: ignore[attr-defined]
                    func.coalesce(func.trim(ObjectOccurrence.note), "") != "",
                    ObjectOccurrence.id,
                    None,
                )
            )
        ).label("note_count")

        q = (
            self._session.query(
                TaggedObject,
                occ_count,
                image_count,
                note_count,
            )
            .outerjoin(
                ObjectOccurrence, ObjectOccurrence.object_id == TaggedObject.id
            )
            .group_by(TaggedObject.id)
        )

        name = filters.name.strip()
        if name:
            q = q.filter(TaggedObject.name.ilike(f"%{name}%"))

        rows = q.order_by(TaggedObject.name).all()

        # Person counts in one extra query (avoids a second outer join that would
        # inflate the occurrence/image counts).
        person_counts = dict(
            self._session.query(
                ObjectPersonLink.object_id,
                func.count(distinct(ObjectPersonLink.person_id)),
            )
            .group_by(ObjectPersonLink.object_id)
            .all()
        )

        # Thumbnail fallback: first occurrence's image path when none is set.
        fallback_thumb: dict[int, str] = {}
        thumb_rows = (
            self._session.query(ObjectOccurrence.object_id, Image.file_path)
            .join(Image, Image.id == ObjectOccurrence.image_id)
            .order_by(ObjectOccurrence.object_id, ObjectOccurrence.id)
            .all()
        )
        for oid, path in thumb_rows:
            if oid not in fallback_thumb and path:
                fallback_thumb[oid] = path

        return [
            ObjectSummary(
                object_id=obj.id,
                name=obj.name,
                description=obj.description,
                notes=obj.notes,
                thumbnail_path=obj.thumbnail_path or fallback_thumb.get(obj.id),
                image_count=int(i_count or 0),
                occurrence_count=int(o_count or 0),
                note_count=int(n_count or 0),
                person_count=int(person_counts.get(obj.id, 0)),
                created_at=obj.created_at.isoformat() if obj.created_at else None,
                updated_at=obj.updated_at.isoformat() if obj.updated_at else None,
            )
            for obj, o_count, i_count, n_count in rows
        ]

    def get_thumbnail_specs(self) -> dict:
        """Return ``{object_id: (path, bbox_or_None)}`` for table thumbnails.

        Manual thumbnails resolve to ``(thumbnail_path, None)`` (a ready crop
        file); otherwise the first occurrence is used — preferring one with a
        bounding box so the crop tightly frames the object.
        """
        specs: dict = {}
        manual: dict = dict(
            self._session.query(TaggedObject.id, TaggedObject.thumbnail_path)
            .filter(TaggedObject.thumbnail_path.isnot(None))
            .all()
        )
        # First bbox occurrence and first any-occurrence per object.
        first_bbox: dict = {}
        first_any: dict = {}
        rows = (
            self._session.query(
                ObjectOccurrence.object_id,
                ObjectOccurrence.image_id,
                ObjectOccurrence.bbox_x,
                ObjectOccurrence.bbox_y,
                ObjectOccurrence.bbox_w,
                ObjectOccurrence.bbox_h,
                Image.file_path,
            )
            .join(Image, Image.id == ObjectOccurrence.image_id)
            .order_by(ObjectOccurrence.object_id, ObjectOccurrence.id)
            .all()
        )
        for oid, _img, bx, by, bw, bh, path in rows:
            if oid not in first_any and path:
                first_any[oid] = (path, None)
            if (
                oid not in first_bbox
                and path
                and None not in (bx, by, bw, bh)
            ):
                first_bbox[oid] = (path, (bx, by, bw, bh))

        all_ids = [oid for (oid,) in self._session.query(TaggedObject.id).all()]
        for oid in all_ids:
            if oid in manual and manual[oid]:
                specs[oid] = (manual[oid], None)
            elif oid in first_bbox:
                specs[oid] = first_bbox[oid]
            elif oid in first_any:
                specs[oid] = first_any[oid]
        return specs

    def get_summary(self, object_id: int) -> ObjectSummary:
        """Return the :class:`ObjectSummary` for a single object."""
        results = self.list_objects(ObjectFilters())
        for s in results:
            if s.object_id == object_id:
                return s
        raise ValueError(f"Object id={object_id} not found")

    def search_objects(self, query: str, max_results: int = 100) -> List[ObjectSummary]:
        """Accent-insensitive search over name, description and per-image notes.

        An empty query returns all objects (capped at *max_results*).
        """
        all_objects = self.list_objects(ObjectFilters())
        q = normalize((query or "").strip())
        if not q:
            return all_objects[:max_results]

        # Object ids whose occurrence notes match (DB-side LIKE for speed, then
        # accent-normalised in Python for correctness).
        note_rows = (
            self._session.query(
                ObjectOccurrence.object_id, ObjectOccurrence.note
            )
            .filter(ObjectOccurrence.note.isnot(None))
            .all()
        )
        note_match_ids = {
            oid for oid, note in note_rows if note and q in normalize(note)
        }

        matched = [
            s
            for s in all_objects
            if q in normalize(s.name)
            or (s.description and q in normalize(s.description))
            or s.object_id in note_match_ids
        ]
        return matched[:max_results]

    # ------------------------------------------------------------------
    # Occurrences
    # ------------------------------------------------------------------

    def add_occurrence(
        self,
        object_id: int,
        image_id: int,
        point_x: int,
        point_y: int,
        note: str = "",
    ) -> ObjectOccurrence:
        """Record that *object_id* appears at ``(point_x, point_y)`` in an image.

        Idempotent on ``(object_id, image_id, point_x, point_y)``: an existing
        occurrence at the same point is returned (its note updated when given).
        """
        self._require_object(object_id)
        existing = (
            self._session.query(ObjectOccurrence)
            .filter(
                ObjectOccurrence.object_id == object_id,
                ObjectOccurrence.image_id == image_id,
                ObjectOccurrence.point_x == point_x,
                ObjectOccurrence.point_y == point_y,
            )
            .first()
        )
        if existing is not None:
            if note.strip():
                existing.note = note.strip()
            self._session.flush()
            return existing
        occ = ObjectOccurrence(
            object_id=object_id,
            image_id=image_id,
            point_x=point_x,
            point_y=point_y,
            note=(note or "").strip() or None,
        )
        self._session.add(occ)
        self._session.flush()
        log.info(
            "Added occurrence of object %d in image %d at (%d,%d)",
            object_id, image_id, point_x, point_y,
        )
        return occ

    def add_occurrence_bbox(
        self,
        object_id: int,
        image_id: int,
        x: int,
        y: int,
        w: int,
        h: int,
        note: str = "",
    ) -> ObjectOccurrence:
        """Record a *bounding box* appearance of *object_id* in an image.

        The top-left corner is also stored in ``point_x/point_y`` so the unique
        constraint and any point-based code keep working; ``geometry_type`` is
        set to ``"bbox"``.  Idempotent on the top-left point.
        """
        self._require_object(object_id)
        existing = (
            self._session.query(ObjectOccurrence)
            .filter(
                ObjectOccurrence.object_id == object_id,
                ObjectOccurrence.image_id == image_id,
                ObjectOccurrence.point_x == x,
                ObjectOccurrence.point_y == y,
            )
            .first()
        )
        if existing is not None:
            existing.bbox_x, existing.bbox_y = x, y
            existing.bbox_w, existing.bbox_h = w, h
            existing.geometry_type = OBJECT_GEOMETRY_BBOX
            if note.strip():
                existing.note = note.strip()
            self._session.flush()
            return existing
        occ = ObjectOccurrence(
            object_id=object_id,
            image_id=image_id,
            point_x=x,
            point_y=y,
            bbox_x=x,
            bbox_y=y,
            bbox_w=w,
            bbox_h=h,
            geometry_type=OBJECT_GEOMETRY_BBOX,
            note=(note or "").strip() or None,
        )
        self._session.add(occ)
        self._session.flush()
        log.info(
            "Added bbox occurrence of object %d in image %d at (%d,%d,%d,%d)",
            object_id, image_id, x, y, w, h,
        )
        return occ

    def update_occurrence_note(self, occurrence_id: int, note: str) -> ObjectOccurrence:
        """Set the per-image comment for an occurrence."""
        occ = self._session.get(ObjectOccurrence, occurrence_id)
        if occ is None:
            raise ValueError(f"Occurrence id={occurrence_id} not found")
        occ.note = (note or "").strip() or None
        self._session.flush()
        return occ

    def update_occurrence_bbox(
        self, occurrence_id: int, x: int, y: int, w: int, h: int
    ) -> ObjectOccurrence:
        """Resize/move an occurrence's bounding box (image coords)."""
        occ = self._session.get(ObjectOccurrence, occurrence_id)
        if occ is None:
            raise ValueError(f"Occurrence id={occurrence_id} not found")
        occ.bbox_x, occ.bbox_y = int(x), int(y)
        occ.bbox_w, occ.bbox_h = max(1, int(w)), max(1, int(h))
        occ.point_x, occ.point_y = int(x), int(y)
        occ.geometry_type = OBJECT_GEOMETRY_BBOX
        self._session.flush()
        return occ

    def remove_occurrence(self, occurrence_id: int) -> None:
        """Delete a single occurrence."""
        occ = self._session.get(ObjectOccurrence, occurrence_id)
        if occ is None:
            raise ValueError(f"Occurrence id={occurrence_id} not found")
        self._session.delete(occ)
        self._session.flush()

    def get_occurrences(self, object_id: int) -> List[ObjectOccurrenceInfo]:
        """Return an object's occurrences ordered by photo date then path."""
        rows = (
            self._session.query(ObjectOccurrence, Image)
            .join(Image, Image.id == ObjectOccurrence.image_id)
            .filter(ObjectOccurrence.object_id == object_id)
            .order_by(Image.photo_date, Image.file_path, ObjectOccurrence.id)
            .all()
        )
        return [
            ObjectOccurrenceInfo(
                occurrence_id=occ.id,
                object_id=occ.object_id,
                image_id=occ.image_id,
                image_path=img.file_path,
                photo_date=img.photo_date,
                point_x=occ.point_x,
                point_y=occ.point_y,
                note=occ.note,
                bbox_x=occ.bbox_x,
                bbox_y=occ.bbox_y,
                bbox_w=occ.bbox_w,
                bbox_h=occ.bbox_h,
            )
            for occ, img in rows
        ]

    def get_occurrences_for_image(self, image_id: int) -> List[ObjectOccurrenceInfo]:
        """Return all object occurrences shown on a given image (for the overlay)."""
        rows = (
            self._session.query(ObjectOccurrence, TaggedObject)
            .join(TaggedObject, TaggedObject.id == ObjectOccurrence.object_id)
            .filter(ObjectOccurrence.image_id == image_id)
            .order_by(ObjectOccurrence.id)
            .all()
        )
        result: List[ObjectOccurrenceInfo] = []
        for occ, obj in rows:
            result.append(
                ObjectOccurrenceInfo(
                    occurrence_id=occ.id,
                    object_id=occ.object_id,
                    image_id=occ.image_id,
                    image_path=None,
                    photo_date=None,
                    point_x=occ.point_x,
                    point_y=occ.point_y,
                    note=occ.note,
                    bbox_x=occ.bbox_x,
                    bbox_y=occ.bbox_y,
                    bbox_w=occ.bbox_w,
                    bbox_h=occ.bbox_h,
                )
            )
        return result

    # ------------------------------------------------------------------
    # Person links
    # ------------------------------------------------------------------

    def add_person_link(
        self,
        object_id: int,
        person_id: int,
        role: str,
        note: str = "",
    ) -> ObjectPersonLink:
        """Link a person to an object under a role.

        Raises:
            ValueError: if *role* is unknown, the person/object is missing, or
                        the person is protected (e.g. the "Ismeretlen" person).
        """
        if role not in OBJECT_ROLES:
            raise ValueError(f"Ismeretlen szerep: {role!r}")
        self._require_object(object_id)
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found")
        if person.is_protected:
            raise ValueError(
                f"Védett személy ('{person.name}') nem köthető objektumhoz."
            )
        existing = self._session.get(
            ObjectPersonLink, {"object_id": object_id, "person_id": person_id, "role": role}
        )
        if existing is not None:
            if note.strip():
                existing.note = note.strip()
            self._session.flush()
            return existing
        link = ObjectPersonLink(
            object_id=object_id,
            person_id=person_id,
            role=role,
            note=(note or "").strip() or None,
        )
        self._session.add(link)
        self._session.flush()
        log.info("Linked person %d to object %d as %r", person_id, object_id, role)
        return link

    def remove_person_link(self, object_id: int, person_id: int, role: str) -> None:
        """Remove a single (object, person, role) link."""
        link = self._session.get(
            ObjectPersonLink, {"object_id": object_id, "person_id": person_id, "role": role}
        )
        if link is not None:
            self._session.delete(link)
            self._session.flush()

    def get_object_persons(self, object_id: int) -> List[ObjectPersonInfo]:
        """Return the persons linked to an object with their roles."""
        rows = (
            self._session.query(ObjectPersonLink, Person)
            .join(Person, Person.id == ObjectPersonLink.person_id)
            .filter(ObjectPersonLink.object_id == object_id)
            .order_by(Person.name, ObjectPersonLink.role)
            .all()
        )
        return [
            ObjectPersonInfo(
                person_id=person.id,
                name=person.name,
                role=link.role,
                note=link.note,
            )
            for link, person in rows
        ]

    def get_objects_for_person(self, person_id: int) -> List[PersonObjectInfo]:
        """Return the objects linked to a person (for the person detail view)."""
        rows = (
            self._session.query(ObjectPersonLink, TaggedObject)
            .join(TaggedObject, TaggedObject.id == ObjectPersonLink.object_id)
            .filter(ObjectPersonLink.person_id == person_id)
            .order_by(TaggedObject.name, ObjectPersonLink.role)
            .all()
        )
        return [
            PersonObjectInfo(
                object_id=obj.id,
                name=obj.name,
                role=link.role,
                note=link.note,
            )
            for link, obj in rows
        ]

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_objects(
        self,
        source_ids: Iterable[int],
        target_id: int,
        *,
        name: Optional[str] = None,
    ) -> TaggedObject:
        """Merge *source_ids* into *target_id*, preserving all relations.

        Every occurrence, per-image note and person link of the sources is
        re-pointed to the target (duplicates dropped to satisfy the unique
        constraints), source name/description are kept as :class:`ObjectAlias`
        rows for traceability, then the sources are deleted.
        """
        target = self._require_object(target_id)
        ids = [sid for sid in dict.fromkeys(source_ids) if sid != target_id]
        if not ids:
            return target

        sources = (
            self._session.query(TaggedObject)
            .filter(TaggedObject.id.in_(ids))
            .order_by(TaggedObject.id)
            .all()
        )
        if len(sources) != len(ids):
            raise ValueError("Egy vagy több forrásobjektum nem létezik.")

        # Snapshot a few source attributes before mutation/deletion.
        source_descriptions = [s.description for s in sources if s.description]
        source_thumbs = [s.thumbnail_path for s in sources if s.thumbnail_path]

        # Existing (image, point) keys on the target to avoid unique collisions.
        target_occ_keys = {
            (o.image_id, o.point_x, o.point_y) for o in target.occurrences
        }
        # Existing (person, role) keys on the target.
        target_link_keys = {
            (pl.person_id, pl.role) for pl in target.person_links
        }

        # Provenance aliases (capture before deleting sources).
        for source in sources:
            self._session.add(
                ObjectAlias(
                    object_id=target.id,
                    source_object_id=source.id,
                    name=source.name,
                    description=source.description,
                )
            )
            for alias in source.aliases:
                self._session.add(
                    ObjectAlias(
                        object_id=target.id,
                        source_object_id=alias.source_object_id,
                        name=alias.name,
                        description=alias.description,
                    )
                )

        # Re-point occurrences via bulk update, deleting exact-point duplicates
        # first so the unique index never collides.
        dup_occ_ids: list[int] = []
        keep_occ_ids: list[int] = []
        for occ in (
            self._session.query(ObjectOccurrence)
            .filter(ObjectOccurrence.object_id.in_(ids))
            .all()
        ):
            key = (occ.image_id, occ.point_x, occ.point_y)
            if key in target_occ_keys:
                dup_occ_ids.append(occ.id)
            else:
                keep_occ_ids.append(occ.id)
                target_occ_keys.add(key)
        if dup_occ_ids:
            self._session.query(ObjectOccurrence).filter(
                ObjectOccurrence.id.in_(dup_occ_ids)
            ).delete(synchronize_session="fetch")
        if keep_occ_ids:
            self._session.query(ObjectOccurrence).filter(
                ObjectOccurrence.id.in_(keep_occ_ids)
            ).update({ObjectOccurrence.object_id: target.id}, synchronize_session="fetch")

        # Re-point person links, merging duplicate (person, role) pairs.
        for link in (
            self._session.query(ObjectPersonLink)
            .filter(ObjectPersonLink.object_id.in_(ids))
            .all()
        ):
            key = (link.person_id, link.role)
            if key in target_link_keys:
                self._session.delete(link)
            else:
                # Composite-PK row: cannot UPDATE the PK in place cleanly, so
                # recreate under the target and drop the source row.
                self._session.add(
                    ObjectPersonLink(
                        object_id=target.id,
                        person_id=link.person_id,
                        role=link.role,
                        note=link.note,
                    )
                )
                self._session.delete(link)
                target_link_keys.add(key)

        if name is not None and name.strip():
            target.name = name.strip()
        if target.description is None and source_descriptions:
            target.description = source_descriptions[0]
        if target.thumbnail_path is None and source_thumbs:
            target.thumbnail_path = source_thumbs[0]

        self._session.flush()
        # Expire sources so their (now empty) occurrence/link collections are
        # reloaded from the DB and the delete-orphan cascade does not re-delete
        # the rows we just re-pointed to the target.
        for source in sources:
            self._session.expire(source)
            self._session.delete(source)
        self._session.flush()
        log.info("Merged objects %s into %d", ids, target_id)
        return target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_object(self, object_id: int) -> TaggedObject:
        obj = self._session.get(TaggedObject, object_id)
        if obj is None:
            raise ValueError(f"Object id={object_id} not found")
        return obj
