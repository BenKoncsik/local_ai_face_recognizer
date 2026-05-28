"""Tests for LocalStorageProvider and GoogleDriveStorageProvider."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.gdrive.storage_provider import (
    GoogleDriveStorageProvider,
    ImageRef,
    LocalStorageProvider,
    StorageProvider,
)
from tests._gdrive_fakes import FakeDriveClient


# ---------------------------------------------------------------------------
# ImageRef
# ---------------------------------------------------------------------------

class TestImageRef:
    def test_local_ref_not_remote(self) -> None:
        ref = ImageRef(name="photo.jpg", local_path=Path("/photos/photo.jpg"))
        assert not ref.is_remote

    def test_remote_ref_is_remote(self) -> None:
        ref = ImageRef(name="photo.jpg", remote_id="fid001")
        assert ref.is_remote

    def test_suffix(self) -> None:
        assert ImageRef(name="photo.JPG").suffix == ".jpg"
        assert ImageRef(name="archive.tar.gz").suffix == ".gz"
        assert ImageRef(name="noext").suffix == ""

    def test_frozen(self) -> None:
        ref = ImageRef(name="a.jpg")
        with pytest.raises((AttributeError, TypeError)):
            ref.name = "b.jpg"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_local_is_storage_provider(tmp_path: Path) -> None:
    provider = LocalStorageProvider(tmp_path)
    assert isinstance(provider, StorageProvider)


def test_gdrive_is_storage_provider(tmp_path: Path) -> None:
    client = FakeDriveClient()
    provider = GoogleDriveStorageProvider(client, client.root_id, "My Project")
    assert isinstance(provider, StorageProvider)


# ---------------------------------------------------------------------------
# LocalStorageProvider
# ---------------------------------------------------------------------------

@pytest.fixture()
def photo_dir(tmp_path: Path) -> Path:
    """A temp directory with a small image tree."""
    (tmp_path / "album1").mkdir()
    (tmp_path / "album2").mkdir()
    (tmp_path / "album1" / "a.jpg").write_bytes(b"JFIF_a")
    (tmp_path / "album1" / "b.png").write_bytes(b"PNG_b")
    (tmp_path / "album2" / "c.jpeg").write_bytes(b"JFIF_c")
    (tmp_path / "album2" / "ignored.txt").write_bytes(b"TEXT")
    return tmp_path


class TestLocalStorageProvider:
    def test_description_is_root_path(self, tmp_path: Path) -> None:
        provider = LocalStorageProvider(tmp_path)
        assert provider.description == str(tmp_path)

    def test_repr(self, tmp_path: Path) -> None:
        provider = LocalStorageProvider(tmp_path)
        assert "LocalStorageProvider" in repr(provider)

    def test_iter_images_finds_jpg_png(self, photo_dir: Path) -> None:
        provider = LocalStorageProvider(photo_dir)
        refs = list(provider.iter_images([".jpg", ".jpeg", ".png"]))
        names = {r.name for r in refs}
        assert "a.jpg" in names
        assert "b.png" in names
        assert "c.jpeg" in names

    def test_iter_images_excludes_non_image(self, photo_dir: Path) -> None:
        provider = LocalStorageProvider(photo_dir)
        refs = list(provider.iter_images([".jpg", ".jpeg", ".png"]))
        names = {r.name for r in refs}
        assert "ignored.txt" not in names

    def test_iter_images_empty_extensions(self, photo_dir: Path) -> None:
        provider = LocalStorageProvider(photo_dir)
        refs = list(provider.iter_images([]))
        assert refs == []

    def test_iter_images_local_path_set(self, photo_dir: Path) -> None:
        provider = LocalStorageProvider(photo_dir)
        refs = list(provider.iter_images([".jpg"]))
        for ref in refs:
            assert ref.local_path is not None
            assert ref.local_path.exists()
            assert not ref.is_remote

    def test_open_image_yields_path(self, photo_dir: Path) -> None:
        provider = LocalStorageProvider(photo_dir)
        refs = list(provider.iter_images([".jpg"]))
        assert refs
        ref = refs[0]
        with provider.open_image(ref) as local_path:
            assert local_path == ref.local_path
            assert local_path.exists()

    def test_open_image_raises_if_no_local_path(self) -> None:
        provider = LocalStorageProvider(Path("/some/dir"))
        ref = ImageRef(name="x.jpg", remote_id="fid")
        with pytest.raises(ValueError, match="local_path"):
            with provider.open_image(ref):
                pass

    def test_iter_images_case_insensitive_extension(self, tmp_path: Path) -> None:
        (tmp_path / "caps.JPG").write_bytes(b"JFIF")
        provider = LocalStorageProvider(tmp_path)
        refs = list(provider.iter_images([".jpg"]))
        names = {r.name for r in refs}
        assert "caps.JPG" in names


# ---------------------------------------------------------------------------
# GoogleDriveStorageProvider
# ---------------------------------------------------------------------------

@pytest.fixture()
def drive_client() -> FakeDriveClient:
    return FakeDriveClient()


@pytest.fixture()
def drive_provider(drive_client: FakeDriveClient, tmp_path: Path) -> GoogleDriveStorageProvider:
    from app.gdrive.cache import GDriveCacheManager
    cache = GDriveCacheManager(tmp_path / "cache")
    return GoogleDriveStorageProvider(
        drive_client,
        drive_client.root_id,
        "Test Project",
        cache=cache,
    )


class TestGoogleDriveStorageProvider:
    def test_description_is_folder_name(self, drive_provider: GoogleDriveStorageProvider) -> None:
        assert drive_provider.description == "Test Project"

    def test_description_fallback_to_id(self, drive_client: FakeDriveClient, tmp_path: Path) -> None:
        from app.gdrive.cache import GDriveCacheManager
        cache = GDriveCacheManager(tmp_path / "cache2")
        provider = GoogleDriveStorageProvider(drive_client, "some-folder-id", "", cache=cache)
        assert provider.description == "some-folder-id"

    def test_repr(self, drive_provider: GoogleDriveStorageProvider) -> None:
        assert "GoogleDriveStorageProvider" in repr(drive_provider)

    def test_iter_images_finds_files(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        drive_client.add_file_bytes(drive_client.root_id, "photo.jpg", b"JFIF", "image/jpeg")
        drive_client.add_file_bytes(drive_client.root_id, "other.txt", b"TXT", "text/plain")

        refs = list(drive_provider.iter_images([".jpg"]))
        names = {r.name for r in refs}
        assert "photo.jpg" in names
        assert "other.txt" not in names

    def test_iter_images_skips_folders(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        drive_client.add_subfolder(drive_client.root_id, "subfolder")
        drive_client.add_file_bytes(drive_client.root_id, "img.png", b"PNG")

        refs = list(drive_provider.iter_images([".png"]))
        assert all(not r.is_remote or r.remote_id is not None for r in refs)
        assert all(r.name != "subfolder" for r in refs)

    def test_iter_images_ref_has_remote_id(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        fid = drive_client.add_file_bytes(drive_client.root_id, "x.jpg", b"JFIF")

        refs = list(drive_provider.iter_images([".jpg"]))
        assert len(refs) == 1
        assert refs[0].remote_id == fid
        assert refs[0].is_remote

    def test_iter_images_ref_has_size(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        drive_client.add_file_bytes(drive_client.root_id, "x.jpg", b"ABCDE")

        refs = list(drive_provider.iter_images([".jpg"]))
        assert refs[0].size == 5

    def test_open_image_downloads_file(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        fid = drive_client.add_file_bytes(drive_client.root_id, "x.jpg", b"JFIF_DATA")

        ref = ImageRef(name="x.jpg", remote_id=fid)
        with drive_provider.open_image(ref) as local_path:
            assert local_path.exists()
            assert local_path.read_bytes() == b"JFIF_DATA"

    def test_open_image_cleans_up_after_context(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        fid = drive_client.add_file_bytes(drive_client.root_id, "x.jpg", b"DATA")

        ref = ImageRef(name="x.jpg", remote_id=fid)
        local_path_outside: Path = Path()
        with drive_provider.open_image(ref) as local_path:
            local_path_outside = local_path
            assert local_path.exists()
        assert not local_path_outside.exists()

    def test_open_image_records_download_call(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        fid = drive_client.add_file_bytes(drive_client.root_id, "x.jpg", b"DATA")

        ref = ImageRef(name="x.jpg", remote_id=fid)
        with drive_provider.open_image(ref):
            pass
        assert fid in drive_client.download_calls

    def test_open_image_raises_without_remote_id(
        self, drive_provider: GoogleDriveStorageProvider
    ) -> None:
        ref = ImageRef(name="x.jpg", local_path=Path("/some/local.jpg"))
        with pytest.raises(ValueError, match="remote_id"):
            with drive_provider.open_image(ref):
                pass

    def test_open_image_cleans_up_on_exception(
        self, drive_provider: GoogleDriveStorageProvider, drive_client: FakeDriveClient
    ) -> None:
        fid = drive_client.add_file_bytes(drive_client.root_id, "boom.jpg", b"DATA")
        ref = ImageRef(name="boom.jpg", remote_id=fid)
        local_captured: list[Path] = []
        with pytest.raises(RuntimeError):
            with drive_provider.open_image(ref) as lp:
                local_captured.append(lp)
                raise RuntimeError("simulated error")
        assert local_captured and not local_captured[0].exists()
