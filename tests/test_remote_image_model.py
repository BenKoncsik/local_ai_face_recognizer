"""Tests for the RemoteImage ORM model and its relationship with Image."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, Image, RemoteImage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session(tmp_path: Path) -> Session:
    """In-memory SQLite session with the full schema."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_image(session: Session, path: str = "/photos/test.jpg") -> Image:
    img = Image(
        file_path=path,
        file_hash="abc123" * 10 + "ab",
        file_mtime=1_700_000_000.0,
    )
    session.add(img)
    session.flush()
    return img


# ---------------------------------------------------------------------------
# Creation / retrieval
# ---------------------------------------------------------------------------

class TestRemoteImageCreation:
    def test_can_create_remote_image(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fileXYZ",
            drive_folder_id="folderABC",
            remote_name="test.jpg",
        )
        db_session.add(ri)
        db_session.flush()
        assert ri.id is not None

    def test_remote_image_repr(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fileXYZ",
            remote_name="test.jpg",
        )
        db_session.add(ri)
        db_session.flush()
        r = repr(ri)
        assert "google_drive" in r
        assert "fileXYZ" in r

    def test_defaults(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fileXYZ",
            remote_name="test.jpg",
        )
        db_session.add(ri)
        db_session.flush()
        assert ri.deleted_remote is False
        assert ri.drive_folder_id is None
        assert ri.modified_time is None
        assert ri.checksum is None
        assert ri.last_seen_at is None

    def test_all_fields(self, db_session: Session) -> None:
        img = _make_image(db_session)
        now = datetime.utcnow()
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid001",
            drive_folder_id="par001",
            remote_name="photo.png",
            modified_time="2024-01-15T12:00:00Z",
            checksum="abcdef0123456789",
            last_seen_at=now,
            deleted_remote=False,
        )
        db_session.add(ri)
        db_session.commit()

        loaded = db_session.get(RemoteImage, ri.id)
        assert loaded.drive_file_id == "fid001"
        assert loaded.drive_folder_id == "par001"
        assert loaded.remote_name == "photo.png"
        assert loaded.checksum == "abcdef0123456789"
        assert loaded.modified_time == "2024-01-15T12:00:00Z"


# ---------------------------------------------------------------------------
# Relationship with Image
# ---------------------------------------------------------------------------

class TestRemoteImageRelationship:
    def test_image_remote_image_backref(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid",
            remote_name="f.jpg",
        )
        db_session.add(ri)
        db_session.flush()
        db_session.refresh(img)

        assert img.remote_image is ri
        assert img.remote_image.provider == "google_drive"

    def test_remote_image_to_image_backref(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid",
            remote_name="f.jpg",
        )
        db_session.add(ri)
        db_session.flush()

        assert ri.image is img
        assert ri.image.file_path == "/photos/test.jpg"

    def test_image_without_remote_has_none(self, db_session: Session) -> None:
        img = _make_image(db_session)
        db_session.flush()
        db_session.refresh(img)
        assert img.remote_image is None

    def test_cascade_delete_removes_remote_image(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid",
            remote_name="f.jpg",
        )
        db_session.add(ri)
        db_session.flush()
        ri_id = ri.id

        db_session.delete(img)
        db_session.commit()

        assert db_session.get(RemoteImage, ri_id) is None

    def test_one_to_one_uniqueness(self, db_session: Session) -> None:
        """Two RemoteImages cannot share the same image_id."""
        import sqlalchemy.exc

        img = _make_image(db_session)
        ri1 = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid1",
            remote_name="f.jpg",
        )
        ri2 = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid2",
            remote_name="g.jpg",
        )
        db_session.add(ri1)
        db_session.flush()
        db_session.add(ri2)
        with pytest.raises(sqlalchemy.exc.IntegrityError):
            db_session.flush()

    def test_deleted_remote_flag_update(self, db_session: Session) -> None:
        img = _make_image(db_session)
        ri = RemoteImage(
            image_id=img.id,
            provider="google_drive",
            drive_file_id="fid",
            remote_name="f.jpg",
        )
        db_session.add(ri)
        db_session.commit()

        ri.deleted_remote = True
        db_session.commit()

        loaded = db_session.get(RemoteImage, ri.id)
        assert loaded.deleted_remote is True
