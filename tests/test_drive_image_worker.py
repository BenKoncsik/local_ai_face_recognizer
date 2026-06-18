"""Tests for Drive image lazy-loading workers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtGui import QImage

from app.db.database import init_db, session_scope
from app.db.models import Image, RemoteImage
from app.workers.drive_image_worker import (
    DriveFetchRunnable,
    DriveThumbRunnable,
    _download,
    _resolve_drive_file_id,
)


@pytest.fixture()
def tmp_db(tmp_path):
    init_db(tmp_path / "drive.db")
    return tmp_path


def _seed_remote_image(file_id: str) -> int:
    with session_scope() as session:
        image = Image(
            file_path=f"/img/{file_id}.jpg",
            file_hash=file_id,
            file_mtime=0.0,
        )
        session.add(image)
        session.flush()
        session.add(
            RemoteImage(
                image_id=image.id,
                provider="google_drive",
                drive_file_id=file_id,
                remote_name=f"photo_{file_id}.jpg",
            )
        )
        return image.id


class TestResolveDriveFileId:
    def test_returns_file_id(self, tmp_db):
        image_id = _seed_remote_image("drive-abc")
        assert _resolve_drive_file_id(image_id) == "drive-abc"

    def test_missing_record_returns_none(self, tmp_db):
        assert _resolve_drive_file_id(99) is None


class TestDownload:
    def test_creates_parent_and_downloads(self, tmp_path):
        client = MagicMock()
        local = tmp_path / "nested" / "img.jpg"
        _download(client, "fid", local)
        client.download_file.assert_called_once_with("fid", str(local))
        assert local.parent.exists()


class TestDriveThumbRunnable:
    def test_uses_existing_local_file(self, qtbot, tmp_path, monkeypatch):
        local = tmp_path / "mirror.jpg"
        bgr = np.zeros((80, 120, 3), dtype=np.uint8)
        local.write_bytes(b"not-a-real-jpeg")

        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            return_value=bgr,
        ):
            worker = DriveThumbRunnable(
                drive_client=MagicMock(),
                image_id=1,
                local_path=str(local),
                cache_key="tree-1",
                size=56,
            )
            with qtbot.waitSignal(worker.signals.ready, timeout=3000) as blocker:
                worker.run()

        assert blocker.args[0] == "tree-1"
        assert isinstance(blocker.args[1], QImage)

    def test_downloads_when_missing(self, qtbot, tmp_path, tmp_db, monkeypatch):
        local = tmp_path / "mirror.jpg"
        image_id = _seed_remote_image("file-7")
        bgr = np.zeros((60, 60, 3), dtype=np.uint8)

        def _fake_download(_client, _fid, path):
            Path(path).write_bytes(b"x")

        monkeypatch.setattr(
            "app.workers.drive_image_worker._download", _fake_download
        )
        with patch(
            "app.utils.image_utils.load_image_bgr_normalized",
            return_value=bgr,
        ):
            worker = DriveThumbRunnable(MagicMock(), image_id, str(local), "k7")
            with qtbot.waitSignal(worker.signals.ready, timeout=3000):
                worker.run()
        assert local.exists()

    def test_failed_without_remote_record(self, qtbot, tmp_path):
        worker = DriveThumbRunnable(
            MagicMock(), 42, str(tmp_path / "x.jpg"), "bad"
        )
        with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
            worker.run()
        assert list(blocker.args) == ["bad"]


class TestDriveFetchRunnable:
    def test_ready_when_already_local(self, qtbot, tmp_path):
        local = tmp_path / "cached.jpg"
        local.write_bytes(b"x")
        worker = DriveFetchRunnable(MagicMock(), 1, str(local))
        with qtbot.waitSignal(worker.signals.ready, timeout=3000) as blocker:
            worker.run()
        assert list(blocker.args) == [1, str(local)]

    def test_downloads_and_emits_ready(self, qtbot, tmp_path, tmp_db, monkeypatch):
        local = tmp_path / "fetch.jpg"
        image_id = _seed_remote_image("fid-3")

        def _fake_download(_client, _fid, path):
            Path(path).write_bytes(b"data")

        monkeypatch.setattr(
            "app.workers.drive_image_worker._download", _fake_download
        )
        worker = DriveFetchRunnable(MagicMock(), image_id, str(local))
        with qtbot.waitSignal(worker.signals.ready, timeout=3000) as blocker:
            worker.run()
        assert list(blocker.args) == [image_id, str(local)]

    def test_silent_failure_suppresses_signal(self, qtbot, tmp_db):
        worker = DriveFetchRunnable(MagicMock(), 8, "/no/such/path.jpg", silent=True)
        failed = []
        worker.signals.failed.connect(lambda *a: failed.append(a))
        worker.run()
        assert failed == []

    def test_non_silent_failure_emits(self, qtbot, tmp_db):
        worker = DriveFetchRunnable(MagicMock(), 8, "/no/such/path.jpg", silent=False)
        with qtbot.waitSignal(worker.signals.failed, timeout=3000) as blocker:
            worker.run()
        assert blocker.args[0] == 8
