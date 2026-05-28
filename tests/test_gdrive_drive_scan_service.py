"""Tests for DriveScanService — Drive-aware image scan."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.models import Base, Image, RemoteImage
from app.gdrive.drive_scan_service import DriveScanService
from app.config import ScanConfig
from tests._gdrive_fakes import FakeDriveClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session(tmp_path: Path) -> Session:
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def client() -> FakeDriveClient:
    return FakeDriveClient()


@pytest.fixture()
def mirror_dir(tmp_path: Path) -> Path:
    d = tmp_path / "mirror"
    d.mkdir()
    return d


def make_svc(db_session, client, mirror_dir, root_id=None) -> DriveScanService:
    cfg = ScanConfig(image_extensions=[".jpg", ".jpeg", ".png"])
    return DriveScanService(
        session=db_session,
        client=client,
        root_folder_id=root_id or client.root_id,
        local_mirror_dir=mirror_dir,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Basic scan
# ---------------------------------------------------------------------------

class TestDriveScanBasic:
    def test_empty_folder_returns_empty(self, db_session, client, mirror_dir):
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert ids == []

    def test_single_image_is_indexed(self, db_session, client, mirror_dir):
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF_DATA")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert len(ids) == 1
        img = db_session.get(Image, ids[0])
        assert img is not None

    def test_image_downloaded_to_mirror(self, db_session, client, mirror_dir):
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF_DATA")
        svc = make_svc(db_session, client, mirror_dir)
        svc.scan()
        # Mirror should contain a file named <file_id>.jpg
        mirrors = list(mirror_dir.glob("*.jpg"))
        assert len(mirrors) == 1
        assert mirrors[0].read_bytes() == b"JFIF_DATA"

    def test_non_image_not_indexed(self, db_session, client, mirror_dir):
        client.add_file_bytes(client.root_id, "readme.txt", b"TEXT")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert ids == []

    def test_remote_image_record_created(self, db_session, client, mirror_dir):
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF_DATA")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        ri = (
            db_session.query(RemoteImage)
            .filter(RemoteImage.drive_file_id == fid)
            .first()
        )
        assert ri is not None
        assert ri.provider == "google_drive"
        assert ri.remote_name == "photo.jpg"

    def test_image_file_path_is_in_mirror(self, db_session, client, mirror_dir):
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF_DATA")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        img = db_session.get(Image, ids[0])
        assert Path(img.file_path).parent == mirror_dir

    def test_multiple_images(self, db_session, client, mirror_dir):
        client.add_file_bytes(client.root_id, "a.jpg", b"A")
        client.add_file_bytes(client.root_id, "b.png", b"B")
        client.add_file_bytes(client.root_id, "c.jpeg", b"C")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert len(ids) == 3


# ---------------------------------------------------------------------------
# Reserved folders skipped
# ---------------------------------------------------------------------------

class TestSkipReservedFolders:
    def test_database_folder_skipped(self, db_session, client, mirror_dir):
        db_folder = client.add_subfolder(client.root_id, "database")
        client.add_file_bytes(db_folder, "faces.db", b"SQLITE_DATA")
        # Also add a real image so we know the scan ran.
        client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert len(ids) == 1  # only the photo, not the db file

    def test_metadata_folder_skipped(self, db_session, client, mirror_dir):
        meta_folder = client.add_subfolder(client.root_id, "metadata")
        client.add_file_bytes(meta_folder, "project.json", b"{}")
        client.add_file_bytes(client.root_id, "photo.jpg", b"JFIF")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert len(ids) == 1

    def test_images_in_regular_subfolder_included(self, db_session, client, mirror_dir):
        album = client.add_subfolder(client.root_id, "album")
        client.add_file_bytes(album, "pic.jpg", b"JFIF")
        svc = make_svc(db_session, client, mirror_dir)
        ids = svc.scan()
        assert len(ids) == 1


# ---------------------------------------------------------------------------
# Incremental scan (unchanged / changed)
# ---------------------------------------------------------------------------

class TestIncrementalScan:
    def test_unchanged_file_not_redownloaded(self, db_session, client, mirror_dir):
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"DATA")
        # Store a known checksum in the fake
        from app.gdrive.protocols import DriveFile as DF
        # Manually set md5 via monkeypatch-style: rebuild meta with checksum
        old = client._meta[fid]
        client._meta[fid] = DF(
            file_id=old.file_id, name=old.name, mime_type=old.mime_type,
            size=old.size, parents=old.parents, md5_checksum="abc123",
        )

        svc = make_svc(db_session, client, mirror_dir)
        svc.scan()
        first_downloads = len(client.download_calls)

        # Second scan — checksum same → should NOT download again
        svc2 = make_svc(db_session, client, mirror_dir)
        ids2 = svc2.scan()
        assert ids2 == []
        assert len(client.download_calls) == first_downloads  # no new download

    def test_changed_file_redownloaded(self, db_session, client, mirror_dir):
        from app.gdrive.protocols import DriveFile as DF
        fid = client.add_file_bytes(client.root_id, "photo.jpg", b"V1")
        old = client._meta[fid]
        client._meta[fid] = DF(
            file_id=old.file_id, name=old.name, mime_type=old.mime_type,
            size=old.size, parents=old.parents, md5_checksum="checksum_v1",
        )

        svc = make_svc(db_session, client, mirror_dir)
        ids1 = svc.scan()
        assert len(ids1) == 1

        # Simulate file change on Drive: different checksum.
        client._bytes[fid] = b"V2"
        client._meta[fid] = DF(
            file_id=old.file_id, name=old.name, mime_type=old.mime_type,
            size=2, parents=old.parents, md5_checksum="checksum_v2",
        )

        svc2 = make_svc(db_session, client, mirror_dir)
        ids2 = svc2.scan()
        # Changed file → detection must be re-run
        assert len(ids2) == 1
        assert ids2[0] == ids1[0]  # same image record

    def test_second_scan_same_image_no_duplicate_records(self, db_session, client, mirror_dir):
        client.add_file_bytes(client.root_id, "photo.jpg", b"DATA")
        make_svc(db_session, client, mirror_dir).scan()
        make_svc(db_session, client, mirror_dir).scan()
        count = db_session.query(Image).count()
        ri_count = db_session.query(RemoteImage).count()
        assert count == 1
        assert ri_count == 1
