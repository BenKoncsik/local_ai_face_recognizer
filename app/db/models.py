"""SQLAlchemy ORM models for face-local.

Schema overview
---------------
images          – one row per image file (path + hash + mtime)
faces           – one row per detected face (bbox + crop + embedding)
persons         – one named cluster / person identity
face_corrections – manual same/not-same judgements for future re-clustering
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

class Image(Base):
    """Represents a discovered image file on disk."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Absolute path as recorded on the indexing machine; kept for backward
    # compatibility.  On a different machine this path may not exist — use
    # relative_path + ImageLibraryService.resolve_path() for portability.
    file_path: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)

    # POSIX-style path relative to the configured ImageLibraryRoot.
    # NULL for records created before the portable-library feature was added.
    # Always stored with forward slashes for cross-platform compatibility.
    relative_path: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True, index=True
    )

    # SHA-256 hex digest of file content — used to skip unchanged files
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    # os.path.getmtime() value at index time
    file_mtime: Mapped[float] = mapped_column(Float, nullable=False)

    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Flexible date string for when the photo was taken:
    # e.g. "1930-as évek", "1954", "1954.03.12", "kb. 1930"
    photo_date: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Optional *estimated* date, used only as a fallback when ``photo_date`` is
    # empty (e.g. inferred from context/family data).  Same flexible-string
    # format; surfaced with an "estimated" flag in the UI.  See
    # app.services.face_date_service.FaceDateResolver.
    estimated_date: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # Free-text note attached to the image by the user.
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    place_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("places.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Raw EXIF GPS coordinates observed during import/re-analysis.  The linked
    # Place is the user-facing location, but these retain the original evidence.
    exif_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exif_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Image-level precise GPS coordinates (separate from the linked Place's
    # coordinates).  NULL means not set.  Priority over place coordinates when
    # resolving the effective GPS of a photo.
    image_latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    image_longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # True once detection + embedding have been attempted for this file
    detection_done: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_done: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    faces: Mapped[List["Face"]] = relationship(
        "Face", back_populates="image", cascade="all, delete-orphan"
    )
    place: Mapped[Optional["Place"]] = relationship("Place", back_populates="images")

    # One-to-one optional: set when this image was sourced from a remote provider.
    remote_image: Mapped[Optional["RemoteImage"]] = relationship(
        "RemoteImage",
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Image id={self.id} path={self.file_path!r}>"


# ---------------------------------------------------------------------------
# Place / Location
# ---------------------------------------------------------------------------

class Place(Base):
    """A reusable place/location that can be linked to many images."""

    __tablename__ = "places"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # True when the user manually chose this thumbnail (auto-refresh skips it)
    thumbnail_is_manual: Mapped[bool] = mapped_column(Boolean, default=False)

    # Granularity of the place: "exact" (a few metres — a house, church, beach),
    # "area" (town/settlement sized) or "region" (large geographic unit).
    # Existing rows migrate to "area" (see _migrate_add_columns default).
    place_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="area", index=True
    )
    # Radius within which a GPS coordinate is considered to belong to this
    # place. Defaults derive from place_type when not set explicitly.
    accuracy_radius_meters: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Self-referential hierarchy: an "exact" place can sit under an "area",
    # an "area" under a "region". NULL means a top-level place.
    parent_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("places.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Structured address. settlement_name is required for new addresses (UI
    # enforced); street_name / house_number are optional. Legacy rows leave
    # these NULL and keep their free-text `name`.
    settlement_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    street_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    house_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Auto-built human label ("Balatonszemes, Bajcsy-Zsilinszky utca 12."). For
    # legacy rows this equals `name` (see _migrate_place_display_name).
    display_name: Mapped[Optional[str]] = mapped_column(String(512), nullable=True, index=True)
    # Where the coordinate came from: geocoded_settlement / geocoded_street /
    # geocoded_address / manual_map_pin / exif / imported.
    coordinate_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # True when the coordinate marks a concrete spot (a precise address or a
    # hand-placed pin); False for a settlement centre or a broad area.
    is_exact_coordinate: Mapped[bool] = mapped_column(Boolean, default=False)

    # EXIF-derived records start as anonymous, then become normal named places
    # once the user names them or merges them into another place.
    is_anonymous: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    images: Mapped[List["Image"]] = relationship("Image", back_populates="place")
    aliases: Mapped[List["PlaceAlias"]] = relationship(
        "PlaceAlias", back_populates="place", cascade="all, delete-orphan"
    )
    parent: Mapped[Optional["Place"]] = relationship(
        "Place", remote_side=[id], back_populates="children"
    )
    children: Mapped[List["Place"]] = relationship(
        "Place", back_populates="parent", overlaps="parent"
    )

    def __repr__(self) -> str:
        return f"<Place id={self.id} name={self.name!r} type={self.place_type!r}>"


class PlaceAlias(Base):
    """Preserved source data from places merged into another place."""

    __tablename__ = "place_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    place_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("places.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_place_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    place: Mapped["Place"] = relationship("Place", back_populates="aliases")


class GeocodingCache(Base):
    """Cached geocoding provider responses, keyed by query type + text.

    Stores both autocomplete suggestion lists and single geocode results as
    ``result_json``; the service decodes it per ``query_type``. Acts as the
    offline layer so a repeated lookup never re-hits the network.
    """

    __tablename__ = "geocoding_cache"
    __table_args__ = (
        UniqueConstraint("query_type", "query_text", name="ux_geocoding_query"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # "settlement" | "street" | "address" | "reverse"
    query_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Normalized lookup key (accent/case-folded query).
    query_text: Mapped[str] = mapped_column(String(512), nullable=False)
    settlement_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    street_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    house_number: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PlaceAddressSuggestion(Base):
    """Local autocomplete source built from addresses the user has used.

    Survives offline: settlement/street pairs entered or geocoded before are
    suggested again without any network call. A NULL street_name means a
    "settlement only" entry.
    """

    __tablename__ = "place_address_suggestions"
    __table_args__ = (
        UniqueConstraint(
            "settlement_name", "street_name", name="ux_address_suggestion"
        ),
        Index("ix_address_suggestion_settlement", "settlement_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    settlement_name: Mapped[str] = mapped_column(String(255), nullable=False)
    street_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    # "user" (typed) | "geocoded" (came from a provider result)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="user")
    last_used_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


# ---------------------------------------------------------------------------
# Person
# ---------------------------------------------------------------------------

class Person(Base):
    """A named identity (cluster of faces).

    Unnamed clusters are given sequential placeholder names: "Unknown 1",
    "Unknown 2", etc.  The ``is_auto_named`` flag tracks whether the user
    has supplied a real name.
    """

    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_auto_named: Mapped[bool] = mapped_column(Boolean, default=True)

    # Binary gender used for family-tree labels and relationship display.
    # Values: "male" | "female" | NULL when not yet set.
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)

    # Representative thumbnail path (one crop selected to stand for the person)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # True when the user manually chose this thumbnail (auto-refresh skips it)
    thumbnail_is_manual: Mapped[bool] = mapped_column(Boolean, default=False)

    # Structured personal data.
    # Format and uniqueness are NOT enforced at the DB level: validation
    # follows the user-editable family code scheme
    # (app/services/family_code_schemes.py) and is soft — the edit dialog
    # warns about format problems and duplicate identity codes
    # (FamilyService.ensure_unique_family_code), but the user may save anyway,
    # so any string can end up here.
    family_code: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # External family identifier (#root#path), describing the person's own
    # family when they belong to a separate, external family tree (e.g. a
    # friend who is the root of their own family). Kept distinct from
    # family_code, which only records how the person links to the main family.
    external_family_code: Mapped[Optional[str]] = mapped_column(
        String(128), nullable=True, index=True
    )
    last_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    second_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    nickname: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    married_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    birth_place: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Flexible date string: "1930-as évek", "1954", "1954.03.12", etc.
    birth_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    death_place: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    death_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Notes / comments entered by the user
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # System-managed persons (e.g. "Ismeretlen") that must not be renamed or deleted
    is_protected: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    faces: Mapped[List["Face"]] = relationship("Face", back_populates="person")
    relationships_a: Mapped[List["Relationship"]] = relationship(
        "Relationship",
        foreign_keys="Relationship.person_a_id",
        back_populates="person_a",
        cascade="all, delete-orphan",
    )
    relationships_b: Mapped[List["Relationship"]] = relationship(
        "Relationship",
        foreign_keys="Relationship.person_b_id",
        back_populates="person_b",
        cascade="all, delete-orphan",
    )
    group_memberships: Mapped[List["PersonGroupMembership"]] = relationship(
        "PersonGroupMembership",
        back_populates="person",
        cascade="all, delete-orphan",
    )
    object_links: Mapped[List["ObjectPersonLink"]] = relationship(
        "ObjectPersonLink",
        back_populates="person",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# PersonGroup / PersonGroupMembership
# ---------------------------------------------------------------------------

class PersonGroup(Base):
    """A user-defined community/category that persons can belong to.

    Examples: Kórus, Munkahely, Iskolai osztály, Szomszéd, Egyesület.
    A person can belong to multiple groups (many-to-many via PersonGroupMembership).
    """

    __tablename__ = "person_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    memberships: Mapped[List["PersonGroupMembership"]] = relationship(
        "PersonGroupMembership",
        back_populates="group",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<PersonGroup id={self.id} name={self.name!r}>"


class PersonGroupMembership(Base):
    """Join table linking persons to person_groups (many-to-many)."""

    __tablename__ = "person_group_memberships"
    __table_args__ = (
        UniqueConstraint("person_id", "group_id", name="uq_pgm_person_group"),
        Index("ix_pgm_group_id", "group_id"),
    )

    person_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("persons.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    group_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("person_groups.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )

    person: Mapped["Person"] = relationship("Person", back_populates="group_memberships")
    group: Mapped["PersonGroup"] = relationship("PersonGroup", back_populates="memberships")

    def __repr__(self) -> str:
        return f"<PersonGroupMembership person={self.person_id} group={self.group_id}>"


# ---------------------------------------------------------------------------
# Relationship
# ---------------------------------------------------------------------------

class Relationship(Base):
    """A normalized family relationship between two people.

    Stored relationship types:
    * ParentChild: person_a is parent, person_b is child.
    * Spouse: person_a/person_b are stored in ascending id order to avoid
      directional duplicates.

    Sibling is intentionally not stored; it is derived from shared parents.
    """

    __tablename__ = "relationships"
    __table_args__ = (
        UniqueConstraint("relationship_type", "person_a_id", "person_b_id"),
        CheckConstraint("person_a_id != person_b_id", name="ck_relationship_not_self"),
        Index("ix_relationship_type_a", "relationship_type", "person_a_id"),
        Index("ix_relationship_type_b", "relationship_type", "person_b_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    person_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    person_a: Mapped["Person"] = relationship(
        "Person", foreign_keys=[person_a_id], back_populates="relationships_a"
    )
    person_b: Mapped["Person"] = relationship(
        "Person", foreign_keys=[person_b_id], back_populates="relationships_b"
    )

    def __repr__(self) -> str:
        return (
            f"<Relationship id={self.id} type={self.relationship_type!r} "
            f"a={self.person_a_id} b={self.person_b_id}>"
        )


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

class Face(Base):
    """A single detected face within an image."""

    __tablename__ = "faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Bounding box in original image pixels (top-left origin)
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)

    # Detection confidence [0.0 – 1.0]
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Detector that produced this result: "coral" | "cpu"
    detector_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")

    # Path to the stored crop thumbnail (relative to crops_dir)
    crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Embedding + landmark blobs live in the face_blobs side table (see
    # FaceBlob): inline ~2 KB blobs made every full faces scan read the whole
    # table.  Use the unchanged Face.get_embedding()/set_embedding() and
    # get_landmarks()/set_landmarks() helpers; for SQL filters use
    # Face.embedding_exists().  lazy="selectin" batch-loads blob rows whenever
    # Face entities load, so per-face access never degrades to N+1; large
    # Face loads that don't need blobs opt out with lazyload(Face.blob).

    # Whether this face was manually excluded from clustering
    is_excluded: Mapped[bool] = mapped_column(Boolean, default=False)

    # Whether this face is excluded from name-matching merges.  Unlike
    # ``is_excluded`` (which removes the face from clustering/recognition
    # entirely), a merge-excluded face stays visible and assigned to its
    # current (usually Unknown) person, but is held back when that person is
    # merged into a named identity and never counts as a positive example for
    # the merge.  This is the persistent backing for the "exclude from merge"
    # action in the name-suggestion gallery.
    is_merge_excluded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # How the current person_id was assigned: NULL for legacy/manual data,
    # "manual", "manual_merge", "recognition", "clustering", etc.
    assignment_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    assignment_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # --- Auto-merge from Unknown (reviewable) ---
    # When the user manually assigns ONE face of an "Unknown N" cluster to a
    # named person, the cluster's *other* faces are moved along automatically but
    # flagged here as needing review (the user only positively identified the one
    # face).  The manually picked face is NOT flagged.  These markers let the
    # auto-moved faces be listed, confirmed, re-moved or deleted later, and they
    # are cleared whenever a face is confirmed or manually re-assigned.
    auto_merged_from_unknown: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # NULL normally; "pending" while the auto-move awaits user review.
    auto_merge_review_status: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True, index=True
    )
    # Id of the original "Unknown N" person the face was moved away from (may no
    # longer exist once the emptied cluster is cleaned up).
    auto_merge_source_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    auto_merge_confirmed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    # True only when a human explicitly confirmed the move; False for an
    # auto-confirm (high-similarity) clearance.
    auto_merge_confirmed_by_user: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Serialized decision graph captured when this face was auto-moved from an
    # Unknown cluster: the ranked target scores, the threshold/margin gates and
    # the outcome.  Lets the review dialog show *why* the merge was suggested
    # (and render the same flow graph as the AI debug view) long after the fact.
    # NULL for faces merged before this column existed.
    auto_merge_decision_json: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    # Face quality evaluation — populated by FaceQualityService after detection.
    # NULL means "not yet evaluated" and is treated as usable (backward compat).
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Comma-separated reason codes, e.g. "low_confidence,too_small,blurry"
    quality_reasons: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # True = poor quality; pipeline stages skip this face when filtering is on.
    # Manually assigned faces can still be used for training regardless.
    is_low_quality: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # User-flagged uncertain identification.  When True the person assignment is
    # considered a hypothesis (e.g. "probably Poszi") rather than a confirmed
    # identity.  Old records default to False (certain).
    is_uncertain_identification: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Free-text note attached to the identification of this face.  Used to record
    # research evidence, doubts or conflicting family memories.  Optional.
    identification_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    image: Mapped["Image"] = relationship("Image", back_populates="faces")
    person: Mapped[Optional["Person"]] = relationship("Person", back_populates="faces")

    corrections_a: Mapped[List["FaceCorrection"]] = relationship(
        "FaceCorrection",
        foreign_keys="FaceCorrection.face_id_a",
        back_populates="face_a",
        cascade="all, delete-orphan",
    )
    corrections_b: Mapped[List["FaceCorrection"]] = relationship(
        "FaceCorrection",
        foreign_keys="FaceCorrection.face_id_b",
        back_populates="face_b",
        cascade="all, delete-orphan",
    )

    blob: Mapped[Optional["FaceBlob"]] = relationship(
        "FaceBlob",
        back_populates="face",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @classmethod
    def embedding_exists(cls):
        """SQL filter: the face has a stored embedding (EXISTS on face_blobs)."""
        return cls.blob.has(FaceBlob.embedding.isnot(None))

    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialise the stored embedding bytes to a float32 numpy array."""
        if self.blob is None or self.blob.embedding is None:
            return None
        return np.frombuffer(self.blob.embedding, dtype=np.float32).copy()

    def set_embedding(self, vector: np.ndarray) -> None:
        """Serialise a float32 numpy array and store it."""
        if self.blob is None:
            self.blob = FaceBlob()
        self.blob.embedding = vector.astype(np.float32).tobytes()

    def get_landmarks(self) -> Optional[np.ndarray]:
        """Deserialise the stored landmarks to a float32 ``(5, 2)`` array."""
        if self.blob is None or self.blob.landmarks is None:
            return None
        return np.frombuffer(
            self.blob.landmarks, dtype=np.float32
        ).reshape(5, 2).copy()

    def set_landmarks(self, points: Optional[object]) -> None:
        """Serialise 5×2 facial landmarks and store them.

        Accepts any array-like of shape ``(5, 2)``; ``None`` clears the field.
        """
        if points is None:
            if self.blob is not None:
                self.blob.landmarks = None
            return
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (5, 2):
            raise ValueError(f"landmarks must be 5x2, got {arr.shape}")
        if self.blob is None:
            self.blob = FaceBlob()
        self.blob.landmarks = arr.tobytes()

    def __repr__(self) -> str:
        return (
            f"<Face id={self.id} image_id={self.image_id} "
            f"bbox=({self.bbox_x},{self.bbox_y},{self.bbox_w},{self.bbox_h}) "
            f"conf={self.confidence:.2f}>"
        )


class FaceBlob(Base):
    """Embedding + landmark bytes for one face, split out of ``faces``.

    The blobs dominated the faces row size (~2 KB each), which made every
    full-table scan — counts, crop lookups, exports — read the entire table.
    Keeping them in a 1:1 side table keeps the hot ``faces`` table small.
    """

    __tablename__ = "face_blobs"

    face_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("faces.id", ondelete="CASCADE"),
        primary_key=True,
    )

    # numpy float32 array → tobytes(); use the Face helpers to (de)serialise.
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # numpy float32 (5, 2) array → tobytes(); NULL when the detector did not
    # produce landmarks (Coral, Caffe SSD, Haar).
    landmarks: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    face: Mapped["Face"] = relationship("Face", back_populates="blob")


# ---------------------------------------------------------------------------
# IgnoredFace
# ---------------------------------------------------------------------------

class IgnoredFace(Base):
    """A face embedding the user permanently excluded from recognition.

    When the user "ignores forever" an Unknown person, every face embedding of
    that person is copied here.  The pipeline compares freshly embedded,
    still-unassigned faces against these vectors and suppresses matches, so the
    same physical person never resurfaces as a new "Unknown N" after a re-run.

    The exclusion is embedding-based on purpose: "Unknown 327"-style names are
    regenerated on every run and cannot serve as a stable identity.
    """

    __tablename__ = "ignored_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Embedding stored as raw bytes (numpy float32 array → tobytes()),
    # same format as Face.embedding.
    _embedding: Mapped[bytes] = mapped_column(
        "embedding", LargeBinary, nullable=False
    )

    # Crop thumbnail shown in the ignored-faces manager (may go stale; the UI
    # falls back to a placeholder when the file no longer exists).
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Face row the embedding was taken from, when it still exists.  Used to
    # re-enable the face on un-ignore; a force rescan may delete it (SET NULL).
    source_face_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("faces.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Display snapshot of the person the face belonged to (e.g. "Unknown 327").
    source_person_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialise the stored embedding bytes to a float32 numpy array."""
        if self._embedding is None:
            return None
        return np.frombuffer(self._embedding, dtype=np.float32).copy()

    def set_embedding(self, vector: np.ndarray) -> None:
        """Serialise a float32 numpy array and store it."""
        self._embedding = vector.astype(np.float32).tobytes()

    def __repr__(self) -> str:
        return (
            f"<IgnoredFace id={self.id} source_face_id={self.source_face_id} "
            f"from={self.source_person_name!r}>"
        )


# ---------------------------------------------------------------------------
# FaceCorrection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Collage
# ---------------------------------------------------------------------------

class Collage(Base):
    """A Picasa-style collage imported from a .cxf / .cfx file."""

    __tablename__ = "collages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Stable identifier from the XML albumUID attribute (may be empty)
    collage_uid: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # Absolute path of the imported .cxf/.cfx file
    source_file: Mapped[str] = mapped_column(Text, unique=True, nullable=False)

    album_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    album_date: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    # format="2858:1000" → stored separately
    format_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    format_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    orientation: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    bg_color: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    spacing: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    nodes: Mapped[List["CollageNode"]] = relationship(
        "CollageNode", back_populates="collage", cascade="all, delete-orphan",
        order_by="CollageNode.id",
    )

    def __repr__(self) -> str:
        return f"<Collage id={self.id} title={self.album_title!r}>"


class CollageNode(Base):
    """A single image cell inside a Picasa collage."""

    __tablename__ = "collage_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    collage_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("collages.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # UID from the XML <uid> element
    node_uid: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    # Relative position/size in [0, 1] coordinates (as parsed from XML)
    rel_x: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rel_y: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rel_w: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rel_h: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Rotation in radians; 0 = no rotation
    theta: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Picasa zoom scale (100 = fit, 133 = 33% zoom-in)
    scale: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)

    # Picasa theme string (e.g. "noborder")
    theme: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Raw src path as it appears in the XML (Windows-style, may have [D]\ prefix)
    src_raw: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Resolved absolute path on the current system (None if not resolved)
    src_resolved: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # True if the source file could not be located on disk
    src_missing: Mapped[bool] = mapped_column(Boolean, default=False)

    # FK to Image record if this source file has been scanned
    image_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("images.id", ondelete="SET NULL"), nullable=True, index=True
    )

    # Optional user-entered metadata
    year: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    event_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    collage: Mapped["Collage"] = relationship("Collage", back_populates="nodes")
    image: Mapped[Optional["Image"]] = relationship("Image")

    def pixel_bbox(self, collage_w: int, collage_h: int) -> tuple[int, int, int, int]:
        """Return (px, py, pw, ph) in pixels for a given collage canvas size."""
        return (
            round(self.rel_x * collage_w),
            round(self.rel_y * collage_h),
            round(self.rel_w * collage_w),
            round(self.rel_h * collage_h),
        )

    def __repr__(self) -> str:
        return f"<CollageNode id={self.id} uid={self.node_uid!r} src={self.src_raw!r}>"


# ---------------------------------------------------------------------------
# FaceCorrection
# ---------------------------------------------------------------------------

class FaceCorrection(Base):
    """Manual same / not-same judgements from the user.

    These are used to constrain or guide future re-clustering runs.
    They are NOT automatically applied — the clustering service reads them
    as soft constraints.
    """

    __tablename__ = "face_corrections"
    __table_args__ = (UniqueConstraint("face_id_a", "face_id_b"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    face_id_a: Mapped[int] = mapped_column(
        Integer, ForeignKey("faces.id", ondelete="CASCADE"), nullable=False
    )
    face_id_b: Mapped[int] = mapped_column(
        Integer, ForeignKey("faces.id", ondelete="CASCADE"), nullable=False
    )

    # True → user confirmed same person; False → user confirmed different
    same_person: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    face_a: Mapped["Face"] = relationship(
        "Face", foreign_keys=[face_id_a], back_populates="corrections_a"
    )
    face_b: Mapped["Face"] = relationship(
        "Face", foreign_keys=[face_id_b], back_populates="corrections_b"
    )

    def __repr__(self) -> str:
        return (
            f"<FaceCorrection a={self.face_id_a} b={self.face_id_b} "
            f"same={self.same_person}>"
        )


# ---------------------------------------------------------------------------
# MergeSuggestion
# ---------------------------------------------------------------------------

# Lifecycle states for a merge suggestion.
MERGE_STATUS_PENDING = "pending"      # awaiting a user decision
MERGE_STATUS_ACCEPTED = "accepted"    # user approved → persons were merged
MERGE_STATUS_REJECTED = "rejected"    # user rejected this particular suggestion
MERGE_STATUS_DEFERRED = "deferred"    # "decide later" — hidden until re-surfaced
MERGE_STATUS_DISMISSED = "dismissed"  # "never suggest this pair again"

MERGE_OPEN_STATUSES = (MERGE_STATUS_PENDING, MERGE_STATUS_DEFERRED)
MERGE_SUPPRESSED_STATUSES = (MERGE_STATUS_REJECTED, MERGE_STATUS_DISMISSED)


class MergeSuggestion(Base):
    """A background-computed proposal that two persons may be the same identity.

    Suggestions are *never* applied automatically — they are persisted so the
    UI can display them incrementally while the background job is still running
    and across application restarts.  Only an explicit user "accept" performs
    the actual :class:`Person` merge.

    Pairs are stored with ``source_person_id < target_person_id`` (normalised
    order) and a unique constraint so concurrent worker chunks cannot create
    duplicate rows for the same pair.
    """

    __tablename__ = "merge_suggestions"
    __table_args__ = (
        UniqueConstraint("source_person_id", "target_person_id", name="ux_merge_pair"),
        Index("ix_merge_status", "status"),
        Index("ix_merge_source", "source_person_id"),
        Index("ix_merge_target", "target_person_id"),
        CheckConstraint(
            "source_person_id != target_person_id", name="ck_merge_not_self"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Normalised pair (source_person_id < target_person_id).  ON DELETE CASCADE
    # so deleting/merging a person removes its stale suggestions automatically.
    source_person_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False
    )
    target_person_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False
    )

    # Combined confidence [0.0 – 1.0] used for ranking/display.
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Component scores (NULL when that signal was unavailable).
    face_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    name_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=MERGE_STATUS_PENDING, index=True
    )

    # Identifier of the background job that produced/last-refreshed this row.
    # Lets the UI tell stale (old-job) suggestions from current ones.
    job_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    # When the user last acted on the suggestion (accept/reject/defer/dismiss).
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    source_person: Mapped["Person"] = relationship(
        "Person", foreign_keys=[source_person_id]
    )
    target_person: Mapped["Person"] = relationship(
        "Person", foreign_keys=[target_person_id]
    )

    def __repr__(self) -> str:
        return (
            f"<MergeSuggestion src={self.source_person_id} "
            f"tgt={self.target_person_id} conf={self.confidence:.2f} "
            f"status={self.status!r}>"
        )


# Decision kinds recorded in the merge-decision history.
MERGE_DECISION_ACCEPTED = "accepted"
MERGE_DECISION_REJECTED = "rejected"
MERGE_DECISION_DISMISSED = "dismissed"

# Who made the decision.
MERGE_DECISION_SOURCE_MANUAL = "manual"
MERGE_DECISION_SOURCE_AUTO = "auto"


class MergeDecision(Base):
    """History of decided merge suggestions ("already handled" list).

    A *snapshot* on purpose: accepting a suggestion merges the candidate
    person away, which CASCADE-deletes the :class:`MergeSuggestion` row, so
    the decided list cannot be derived from live suggestions.  Names and crop
    paths are copied at decision time and stay displayable even after the
    persons were merged or renamed.
    """

    __tablename__ = "merge_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # The suggestion this decision came from (plain int — the row may be gone).
    suggestion_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Person ids at decision time (not FKs — they may no longer exist).
    candidate_person_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    target_person_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Display snapshots.
    candidate_name: Mapped[str] = mapped_column(String(255), nullable=False)
    target_name: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    target_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # "accepted" | "rejected" | "dismissed"
    decision: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    # "manual" | "auto" (auto-merge)
    source: Mapped[str] = mapped_column(
        String(16), nullable=False, default=MERGE_DECISION_SOURCE_MANUAL
    )

    decided_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )

    def __repr__(self) -> str:
        return (
            f"<MergeDecision id={self.id} {self.candidate_name!r}→"
            f"{self.target_name!r} {self.decision!r}>"
        )


# ---------------------------------------------------------------------------
# Deep recognition — training runs and the automatic-assignment review log
# ---------------------------------------------------------------------------

# Lifecycle states of an automatic deep-recognition assignment.
AUTO_ASSIGN_STATUS_AUTO = "auto"            # made by the engine, awaiting review
AUTO_ASSIGN_STATUS_CONFIRMED = "confirmed"  # user approved → trusted training data
AUTO_ASSIGN_STATUS_CORRECTED = "corrected"  # user moved the face to someone else
AUTO_ASSIGN_STATUS_REVERTED = "reverted"    # user sent the face back to unknown


class TrainingRun(Base):
    """One training + recognition run of the deep-learning engine.

    Stores diagnostics about the model that was trained (how many people /
    examples, validation accuracy) so runs are comparable over time, and acts
    as the parent of the :class:`AutoAssignment` rows produced by the run.
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # "rescan" | "rebuild" — which pipeline mode produced this run.
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="rescan")

    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Training-set statistics.
    n_persons: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_examples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_augmented: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Cross-validated top-1 accuracy on held-out labeled faces (NULL when the
    # dataset was too small to cross-validate).
    validation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Hash of the labeled training data — identical hash ⇒ identical model.
    data_fingerprint: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # Where the trained model file was persisted.
    model_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    assignments: Mapped[List["AutoAssignment"]] = relationship(
        "AutoAssignment", back_populates="training_run", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<TrainingRun id={self.id} mode={self.mode!r} "
            f"persons={self.n_persons} examples={self.n_examples}>"
        )


class AutoAssignment(Base):
    """One automatic face → person assignment made by the deep engine.

    These rows power the "automatic groupings" review tab: the user sees what
    the last run grouped, and can confirm, correct (move to the right person)
    or revert each decision.  Corrections are recorded as
    :class:`FaceCorrection` rows so the engine learns from them.
    """

    __tablename__ = "auto_assignments"
    __table_args__ = (
        Index("ix_auto_assign_run_status", "training_run_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    training_run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    face_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("faces.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # The person the engine assigned the face to.
    person_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Where the face lived before the run (usually an "Unknown N" or nothing).
    previous_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )
    # Display snapshot of the previous person's name: the "Unknown N" group is
    # often deleted right after it loses its last face, which SET-NULLs the FK.
    previous_person_name: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    # The person the user moved the face to when status == "corrected".
    corrected_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )

    # Blended engine confidence [0, 1] for the assignment.
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=AUTO_ASSIGN_STATUS_AUTO, index=True
    )

    # Serialized deep-engine decision graph (gates + winner probabilities)
    # captured at assignment time, so the review tab can show *why* the AI made
    # this grouping — the same flow graph as the live AI debug view.  NULL for
    # rows created before this column existed.
    decision_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    training_run: Mapped["TrainingRun"] = relationship(
        "TrainingRun", back_populates="assignments"
    )
    face: Mapped["Face"] = relationship("Face")
    person: Mapped["Person"] = relationship("Person", foreign_keys=[person_id])
    previous_person: Mapped[Optional["Person"]] = relationship(
        "Person", foreign_keys=[previous_person_id]
    )
    corrected_person: Mapped[Optional["Person"]] = relationship(
        "Person", foreign_keys=[corrected_person_id]
    )

    def __repr__(self) -> str:
        return (
            f"<AutoAssignment id={self.id} face={self.face_id} "
            f"person={self.person_id} score={self.score:.2f} status={self.status!r}>"
        )


# ---------------------------------------------------------------------------
# Object tagging  (a domain entirely separate from face recognition)
# ---------------------------------------------------------------------------
#
# Objects are arbitrary user-tagged things in photos (a car, a house, a
# gravestone, a painting, …).  They are NOT persons, NOT faces, carry no
# embedding/biometric data and never take part in any face-recognition step.
#
# Geometry: today only a single point per occurrence is stored.  The schema is
# intentionally future-proofed for bounding boxes, polygons and AI detection
# via the nullable bbox_*/polygon_json/geometry_type/detection_source/confidence
# columns, so a later YOLO/CLIP integration needs no migration of existing data.

# Roles a person can have relative to an object.
OBJECT_ROLE_OWNER = "owner"
OBJECT_ROLE_FORMER_OWNER = "former_owner"
OBJECT_ROLE_DRIVER = "driver"
OBJECT_ROLE_CREATOR = "creator"
OBJECT_ROLE_USER = "user"
OBJECT_ROLE_FAMILY = "family"
OBJECT_ROLE_OTHER = "other"

OBJECT_ROLES = (
    OBJECT_ROLE_OWNER,
    OBJECT_ROLE_FORMER_OWNER,
    OBJECT_ROLE_DRIVER,
    OBJECT_ROLE_CREATOR,
    OBJECT_ROLE_USER,
    OBJECT_ROLE_FAMILY,
    OBJECT_ROLE_OTHER,
)

# Occurrence geometry kinds.
OBJECT_GEOMETRY_POINT = "point"
OBJECT_GEOMETRY_BBOX = "bbox"
OBJECT_GEOMETRY_POLYGON = "polygon"


class TaggedObject(Base):
    """A user-tagged object identity that can appear in many images.

    Examples: "BMW E91", "Klotild" (a boat), a gravestone, a painting.  The
    same object is referenced from every image it appears in via
    :class:`ObjectOccurrence`.
    """

    __tablename__ = "tagged_objects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Required display name (e.g. "BMW E91").
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # Optional global description (e.g. "2004-es BMW E91 330D Touring").
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Optional free-text notes attached to the object globally.  Per-image
    # comments live on ObjectOccurrence.note, NOT here.
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Representative thumbnail (path to a source image or crop).
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_is_manual: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    occurrences: Mapped[List["ObjectOccurrence"]] = relationship(
        "ObjectOccurrence",
        back_populates="tagged_object",
        cascade="all, delete-orphan",
    )
    person_links: Mapped[List["ObjectPersonLink"]] = relationship(
        "ObjectPersonLink",
        back_populates="tagged_object",
        cascade="all, delete-orphan",
    )
    aliases: Mapped[List["ObjectAlias"]] = relationship(
        "ObjectAlias",
        back_populates="tagged_object",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<TaggedObject id={self.id} name={self.name!r}>"


class ObjectOccurrence(Base):
    """A single appearance of a :class:`TaggedObject` in one image.

    Currently a single point ``(point_x, point_y)`` in original-image pixels is
    enough; the bbox/polygon/AI columns are reserved for future selection modes
    and automatic detection.  A per-image comment lives in ``note`` (kept
    separate from the object's global description/notes).
    """

    __tablename__ = "object_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "object_id", "image_id", "point_x", "point_y", name="ux_object_occurrence"
        ),
        Index("ix_object_occ_image", "image_id"),
        Index("ix_object_occ_object", "object_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    object_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tagged_objects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Primary position today: a single point in original image pixels.
    point_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    point_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # ── Future-proofing (all nullable; unused by the manual point workflow) ──
    bbox_x: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bbox_y: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bbox_w: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bbox_h: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # JSON-encoded list of [x, y] vertices for a future polygon selection.
    polygon_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # "point" | "bbox" | "polygon"
    geometry_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default=OBJECT_GEOMETRY_POINT
    )
    # "manual" | "ai" — reserved for later automatic object detection.
    detection_source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="manual"
    )
    # Detector confidence [0.0 – 1.0]; NULL for manual occurrences.
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Per-image comment for this occurrence ("Frissen mosva.").
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    tagged_object: Mapped["TaggedObject"] = relationship(
        "TaggedObject", back_populates="occurrences"
    )
    image: Mapped["Image"] = relationship("Image")

    def __repr__(self) -> str:
        return (
            f"<ObjectOccurrence id={self.id} object_id={self.object_id} "
            f"image_id={self.image_id} point=({self.point_x},{self.point_y})>"
        )


class ObjectPersonLink(Base):
    """A many-to-many link between a :class:`TaggedObject` and a :class:`Person`.

    A person may be linked under several roles (e.g. owner *and* driver), so the
    role is part of the primary key.
    """

    __tablename__ = "object_person_links"
    __table_args__ = (
        Index("ix_opl_person", "person_id"),
        Index("ix_opl_object", "object_id"),
    )

    object_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tagged_objects.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    person_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("persons.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    # One of OBJECT_ROLES.
    role: Mapped[str] = mapped_column(
        String(64), primary_key=True, nullable=False, default=OBJECT_ROLE_OTHER
    )
    # Optional free-text qualifier for the role (e.g. "2010–2015").
    note: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tagged_object: Mapped["TaggedObject"] = relationship(
        "TaggedObject", back_populates="person_links"
    )
    person: Mapped["Person"] = relationship("Person", back_populates="object_links")

    def __repr__(self) -> str:
        return (
            f"<ObjectPersonLink object={self.object_id} person={self.person_id} "
            f"role={self.role!r}>"
        )


class ObjectAlias(Base):
    """Preserved source data from objects merged into another object.

    Mirrors :class:`PlaceAlias`: when two objects are merged, the source's
    name/description is kept here for traceability.
    """

    __tablename__ = "object_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    object_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tagged_objects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_object_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tagged_object: Mapped["TaggedObject"] = relationship(
        "TaggedObject", back_populates="aliases"
    )

    def __repr__(self) -> str:
        return f"<ObjectAlias id={self.id} object_id={self.object_id} name={self.name!r}>"


# ---------------------------------------------------------------------------
# RemoteImage
# ---------------------------------------------------------------------------

class RemoteImage(Base):
    """Records that a local :class:`Image` row was sourced from a remote provider.

    One-to-one with ``Image`` (via a unique FK).  When present it stores
    enough metadata to detect whether the remote file has changed since we
    last downloaded it, and to upload crops/annotations back.

    Currently only ``provider = "google_drive"`` is used, but the schema
    is intentionally provider-agnostic for future extensibility.
    """

    __tablename__ = "remote_images"
    __table_args__ = (
        Index("ix_remote_images_provider_file_id", "provider", "drive_file_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # FK to the local Image record — nullable=False, unique to enforce one-to-one.
    image_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Storage provider identifier — currently always "google_drive".
    provider: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # Provider-specific remote file ID (Drive file_id for google_drive).
    drive_file_id: Mapped[str] = mapped_column(String(256), nullable=False)

    # Parent folder on the remote (Drive folder_id for google_drive).
    drive_folder_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    # Filename as it appears on the remote.
    remote_name: Mapped[str] = mapped_column(String(512), nullable=False)

    # Drive metadata captured at time of last sync (RFC 3339 string).
    modified_time: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # MD5 / SHA-256 checksum as reported by the provider.
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # When we last confirmed the file still exists on the remote.
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # True once the provider no longer returns this file (deleted / moved out).
    deleted_remote: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    image: Mapped["Image"] = relationship("Image", back_populates="remote_image")

    def __repr__(self) -> str:
        return (
            f"<RemoteImage id={self.id} image_id={self.image_id} "
            f"provider={self.provider!r} file_id={self.drive_file_id!r}>"
        )


# ---------------------------------------------------------------------------
# RecognitionMergeLog
# ---------------------------------------------------------------------------


class RecognitionMergeLog(Base):
    """Audit + undo record for one automatic re-recognition merge.

    Each row captures a single face that the "re-recognize faces" workflow moved
    from an ``Unknown N`` cluster (or no person) onto a named identity, together
    with the *previous* assignment state so the merge can be undone later.  Rows
    sharing a ``batch_id`` belong to one re-recognition run; undoing a batch
    restores every face in it (and re-creates any Unknown cluster that the merge
    emptied and the cleanup pass removed).

    This table is the single schema addition for the feature.  It is created
    automatically by ``Base.metadata.create_all`` — no separate migration.
    """

    __tablename__ = "recognition_merge_log"
    __table_args__ = (
        Index("ix_recmerge_batch", "batch_id"),
        Index("ix_recmerge_face", "face_id"),
        Index("ix_recmerge_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Groups all rows produced by one re-recognition run.
    batch_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # The moved face and the image it lives in (image_id kept even if the face
    # row is later deleted, so the audit log stays readable).
    face_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("faces.id", ondelete="SET NULL"), nullable=True
    )
    image_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- Previous assignment state (for undo) ---
    prev_person_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prev_person_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    prev_assignment_source: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True
    )
    prev_assignment_confidence: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    prev_assigned_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    # True when the source person was an auto-named Unknown cluster that may have
    # been deleted by the empty-cluster cleanup; lets undo recreate it.
    prev_person_was_auto: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Match result ---
    matched_person_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("persons.id", ondelete="SET NULL"), nullable=True
    )
    matched_person_name: Mapped[Optional[str]] = mapped_column(
        String(256), nullable=True
    )
    score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # How the merge was decided: "auto" (threshold) or "manual" (review dialog).
    decision: Mapped[str] = mapped_column(String(16), nullable=False, default="auto")

    # Set when the batch this row belongs to has been undone.
    undone_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<RecognitionMergeLog batch={self.batch_id!r} face={self.face_id} "
            f"{self.prev_person_name!r}→{self.matched_person_name!r} "
            f"score={self.score:.3f}>"
        )


# ---------------------------------------------------------------------------
# AI face detection (analysis-only)
# ---------------------------------------------------------------------------

class AiFaceDetection(Base):
    """One face found by the AI (deep learning) face-detection analysis pass.

    Pure analysis output — *is* there a face, *where*, and with what
    confidence.  Rows are written by
    :class:`~app.services.ai_face_detection_service.AiFaceDetectionService`
    and are completely independent of the ``faces`` table: no identity is
    assigned and the classic detection/recognition results are never touched.
    Each new run replaces the previous rows of the processed images, so the
    table always holds the latest AI detection result per image.

    Created automatically by ``Base.metadata.create_all`` — no separate
    migration.
    """

    __tablename__ = "ai_face_detections"
    __table_args__ = (
        Index("ix_aifd_image", "image_id"),
        Index("ix_aifd_run", "run_id"),
        Index("ix_aifd_detected_at", "detected_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    image_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    # Path snapshot at detection time, so results stay readable even if the
    # image row is later re-pathed or removed.
    image_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # --- Bounding box (image pixels) ---
    bbox_x: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_y: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_w: Mapped[int] = mapped_column(Integer, nullable=False)
    bbox_h: Mapped[int] = mapped_column(Integer, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Which deep-learning detector produced this row (e.g. "yunet").
    detector_name: Mapped[str] = mapped_column(String(64), nullable=False)

    # Groups all rows produced by one detection run.
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # What triggered the run: "manual" (AI tab), "pipeline" (deep pipeline
    # stage) or "rerecognition" (image-browser re-recognize action).
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")

    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<AiFaceDetection image={self.image_id} "
            f"bbox=({self.bbox_x},{self.bbox_y},{self.bbox_w},{self.bbox_h}) "
            f"conf={self.confidence:.3f} detector={self.detector_name!r}>"
        )
