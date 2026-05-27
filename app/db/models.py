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
    DateTime,
    Float,
    ForeignKey,
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

    def __repr__(self) -> str:
        return f"<Image id={self.id} path={self.file_path!r}>"


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

    # Representative thumbnail path (one crop selected to stand for the person)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Structured personal data
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

    def __repr__(self) -> str:
        return f"<Person id={self.id} name={self.name!r}>"


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
