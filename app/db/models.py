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

    def __repr__(self) -> str:
        return f"<Place id={self.id} name={self.name!r}>"


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

    # Structured personal data
    family_code: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, nullable=True, index=True
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

    # Embedding stored as raw bytes (numpy float32 array → tobytes())
    # Use Face.get_embedding() / Face.set_embedding() helpers.
    _embedding: Mapped[Optional[bytes]] = mapped_column(
        "embedding", LargeBinary, nullable=True
    )

    # 5 facial landmarks (right-eye, left-eye, nose, right-mouth, left-mouth)
    # stored as raw bytes (numpy float32 (5, 2) array → tobytes()).  NULL when
    # the detector did not produce landmarks (Coral, Caffe SSD, Haar).  Used by
    # the "aligned" crop mode.  Use get_landmarks() / set_landmarks() helpers.
    _landmarks: Mapped[Optional[bytes]] = mapped_column(
        "landmarks", LargeBinary, nullable=True
    )

    # Whether this face was manually excluded from clustering
    is_excluded: Mapped[bool] = mapped_column(Boolean, default=False)

    # How the current person_id was assigned: NULL for legacy/manual data,
    # "manual", "manual_merge", "recognition", "clustering", etc.
    assignment_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    assignment_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    assigned_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Face quality evaluation — populated by FaceQualityService after detection.
    # NULL means "not yet evaluated" and is treated as usable (backward compat).
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Comma-separated reason codes, e.g. "low_confidence,too_small,blurry"
    quality_reasons: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    # True = poor quality; pipeline stages skip this face when filtering is on.
    # Manually assigned faces can still be used for training regardless.
    is_low_quality: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

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

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def get_embedding(self) -> Optional[np.ndarray]:
        """Deserialise the stored embedding bytes to a float32 numpy array."""
        if self._embedding is None:
            return None
        return np.frombuffer(self._embedding, dtype=np.float32).copy()

    def set_embedding(self, vector: np.ndarray) -> None:
        """Serialise a float32 numpy array and store it."""
        self._embedding = vector.astype(np.float32).tobytes()

    def get_landmarks(self) -> Optional[np.ndarray]:
        """Deserialise the stored landmarks to a float32 ``(5, 2)`` array."""
        if self._landmarks is None:
            return None
        return np.frombuffer(self._landmarks, dtype=np.float32).reshape(5, 2).copy()

    def set_landmarks(self, points: Optional[object]) -> None:
        """Serialise 5×2 facial landmarks and store them.

        Accepts any array-like of shape ``(5, 2)``; ``None`` clears the field.
        """
        if points is None:
            self._landmarks = None
            return
        arr = np.asarray(points, dtype=np.float32)
        if arr.shape != (5, 2):
            raise ValueError(f"landmarks must be 5x2, got {arr.shape}")
        self._landmarks = arr.tobytes()

    def __repr__(self) -> str:
        return (
            f"<Face id={self.id} image_id={self.image_id} "
            f"bbox=({self.bbox_x},{self.bbox_y},{self.bbox_w},{self.bbox_h}) "
            f"conf={self.confidence:.2f}>"
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
