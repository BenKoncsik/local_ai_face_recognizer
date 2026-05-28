"""In-memory fakes that satisfy the :class:`GDriveClient` protocol.

Used by all tests that exercise Drive workflows without hitting Google.
"""

from __future__ import annotations

import io
import shutil
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from app.gdrive.protocols import DriveAccountInfo, DriveFile, GDriveClient

_FOLDER_MIME = "application/vnd.google-apps.folder"


class FakeDriveClient(GDriveClient):
    """A minimal in-memory Drive simulation.

    * Files are stored as bytes in an internal dict keyed by file_id.
    * Folders are virtual entries with the Drive folder mime type.
    * Metadata is reconstructed from a parallel dict of :class:`DriveFile`.
    """

    def __init__(self, *, account_email: str = "test@example.com") -> None:
        self._account = DriveAccountInfo(email=account_email)
        self._bytes: Dict[str, bytes] = {}
        self._meta: Dict[str, DriveFile] = {}
        # Root sentinel — every project sits under this id.
        self.root_id = self._new_id()
        self._meta[self.root_id] = DriveFile(
            file_id=self.root_id,
            name="root",
            mime_type=_FOLDER_MIME,
        )
        # Telemetry — useful for asserting "downloaded only once" etc.
        self.download_calls: List[str] = []
        self.upload_calls: List[str] = []
        self.delete_calls: List[str] = []

    # ------------------------------------------------------------------
    # Helpers exposed to tests
    # ------------------------------------------------------------------

    def add_file_bytes(self, parent_id: str, name: str, data: bytes, mime: str = "application/octet-stream") -> str:
        file_id = self._new_id()
        self._bytes[file_id] = data
        self._meta[file_id] = DriveFile(
            file_id=file_id,
            name=name,
            mime_type=mime,
            size=len(data),
            parents=(parent_id,),
        )
        return file_id

    def add_subfolder(self, parent_id: str, name: str) -> str:
        file_id = self._new_id()
        self._meta[file_id] = DriveFile(
            file_id=file_id,
            name=name,
            mime_type=_FOLDER_MIME,
            parents=(parent_id,),
        )
        return file_id

    # ------------------------------------------------------------------
    # GDriveClient protocol
    # ------------------------------------------------------------------

    @property
    def account(self) -> DriveAccountInfo:
        return self._account

    def download_file(self, file_id: str, local_path: str) -> None:
        self.download_calls.append(file_id)
        if file_id not in self._bytes:
            raise FileNotFoundError(f"FakeDrive: no such file_id {file_id}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        Path(local_path).write_bytes(self._bytes[file_id])

    def upload_file(
        self,
        local_path: str,
        *,
        parent_folder_id: str,
        remote_name: Optional[str] = None,
        existing_file_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> str:
        data = Path(local_path).read_bytes()
        name = remote_name or Path(local_path).name
        if existing_file_id:
            file_id = existing_file_id
            old = self._meta.get(file_id)
            self._meta[file_id] = DriveFile(
                file_id=file_id,
                name=name,
                mime_type=mime_type or (old.mime_type if old else "application/octet-stream"),
                size=len(data),
                parents=(parent_folder_id,) if old is None else old.parents,
            )
        else:
            file_id = self._new_id()
            self._meta[file_id] = DriveFile(
                file_id=file_id,
                name=name,
                mime_type=mime_type or "application/octet-stream",
                size=len(data),
                parents=(parent_folder_id,),
            )
        self._bytes[file_id] = data
        self.upload_calls.append(file_id)
        return file_id

    def list_folder(
        self,
        folder_id: str,
        *,
        mime_types: Optional[Iterable[str]] = None,
        include_subfolders: bool = False,
    ) -> List[DriveFile]:
        out: List[DriveFile] = []
        for f in self._meta.values():
            if folder_id not in f.parents:
                continue
            if not include_subfolders and f.is_folder:
                continue
            if mime_types is not None and f.mime_type not in set(mime_types):
                continue
            out.append(f)
        return out

    def find_child(self, folder_id: str, name: str) -> Optional[DriveFile]:
        for f in self._meta.values():
            if folder_id in f.parents and f.name == name:
                return f
        return None

    def get_metadata(self, file_id: str) -> DriveFile:
        if file_id not in self._meta:
            raise FileNotFoundError(f"FakeDrive: no such file_id {file_id}")
        return self._meta[file_id]

    def create_folder(self, parent_folder_id: str, name: str) -> DriveFile:
        new_id = self.add_subfolder(parent_folder_id, name)
        return self._meta[new_id]

    def delete_file(self, file_id: str) -> None:
        self.delete_calls.append(file_id)
        self._meta.pop(file_id, None)
        self._bytes.pop(file_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return f"drv-{uuid.uuid4().hex[:12]}"


# Suppress unused-import warnings for items that may appear in extensions.
_ = io
_ = shutil
