"""Google Drive integration — protocols and value objects.

This module defines the contracts the rest of the application uses to
interact with Drive.  The concrete implementation (OAuth flow,
google-api-python-client) lives in :mod:`app.gdrive.drive_client`; this
module only declares the interface so that :mod:`app.gdrive.cache`,
:mod:`app.gdrive.db_sync` and the UI stay testable without a live Drive
connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriveFile:
    """Metadata for a single file or folder on Google Drive."""

    file_id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    modified_time: Optional[str] = None  # RFC 3339 string from Drive
    md5_checksum: Optional[str] = None
    parents: tuple[str, ...] = ()

    @property
    def is_folder(self) -> bool:
        return self.mime_type == "application/vnd.google-apps.folder"


@dataclass(frozen=True)
class DriveAccountInfo:
    """User identity returned by the OAuth flow."""

    email: str
    display_name: Optional[str] = None
    picture_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class GDriveClient(Protocol):
    """Minimal Drive client interface used across the app.

    The concrete implementation handles OAuth, retries, rate-limit backoff
    and chunked transfers internally.  Callers receive simple value
    objects and raise :class:`~app.gdrive.connectivity.GDriveOfflineError`
    on connectivity failure.
    """

    # -- File transfer ------------------------------------------------------

    def download_file(self, file_id: str, local_path: str) -> None:
        """Download Drive file *file_id* to *local_path*."""
        ...

    def upload_file(
        self,
        local_path: str,
        *,
        parent_folder_id: str,
        remote_name: Optional[str] = None,
        existing_file_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> str:
        """Upload *local_path* to Drive.

        Replaces *existing_file_id* if given, otherwise creates a new file
        under *parent_folder_id*.  Returns the file_id of the uploaded item.
        """
        ...

    # -- Folder / metadata --------------------------------------------------

    def list_folder(
        self,
        folder_id: str,
        *,
        mime_types: Optional[Iterable[str]] = None,
        include_subfolders: bool = False,
    ) -> Iterable[DriveFile]:
        """Yield direct children of *folder_id* (optionally filtered)."""
        ...

    def find_child(
        self, folder_id: str, name: str
    ) -> Optional[DriveFile]:
        """Return the direct child of *folder_id* named *name*, or ``None``."""
        ...

    def get_metadata(self, file_id: str) -> DriveFile:
        """Return metadata for *file_id*."""
        ...

    def create_folder(self, parent_folder_id: str, name: str) -> DriveFile:
        """Create *name* under *parent_folder_id* and return its metadata."""
        ...

    def delete_file(self, file_id: str) -> None:
        """Remove *file_id* from Drive (idempotent on missing files)."""
        ...

    # -- Identity -----------------------------------------------------------

    @property
    def account(self) -> DriveAccountInfo:
        """Identity of the user currently authenticated with the client."""
        ...
