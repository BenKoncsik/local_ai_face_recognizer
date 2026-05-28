"""StorageProvider — uniform interface to local-disk and Drive image sources.

The scan pipeline and image loader talk only to this interface; the
concrete implementation is chosen at startup from the app configuration.

Usage example::

    # Local mode
    provider = LocalStorageProvider(Path("/photos"))

    # Drive mode
    provider = GoogleDriveStorageProvider(drive_client, folder_id="abc123")

    for ref in provider.iter_images([".jpg", ".png"]):
        with provider.open_image(ref) as local_path:
            process(local_path)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Iterable, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from app.gdrive.cache import GDriveCacheManager
    from app.gdrive.protocols import GDriveClient


# ---------------------------------------------------------------------------
# Value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRef:
    """A reference to an image file in a storage location.

    Exactly one of ``local_path`` / ``remote_id`` is set depending on
    whether the image lives on disk or on a remote provider.
    """

    # Human-readable filename (no directory — e.g. "photo.jpg")
    name: str

    # Set for local-disk images
    local_path: Optional[Path] = None

    # Set for Drive images — the provider's opaque file identifier
    remote_id: Optional[str] = None

    # Parent collection ID on the remote (Drive folder_id for google_drive)
    remote_parent_id: Optional[str] = None

    # Optional remote metadata
    size: Optional[int] = None
    modified_time: Optional[str] = None   # RFC 3339 for Drive
    checksum: Optional[str] = None        # MD5 hex from Drive

    # ---------------------------------------------------------------------------

    @property
    def is_remote(self) -> bool:
        """True when this reference points to a remote (not local) file."""
        return self.remote_id is not None

    @property
    def suffix(self) -> str:
        """Lower-case file extension including the dot, e.g. ``'.jpg'``."""
        return Path(self.name).suffix.lower()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StorageProvider(Protocol):
    """Uniform interface over local-disk and remote image sources.

    Implementations must be safe to call from a background thread.
    """

    def iter_images(self, extensions: Iterable[str]) -> Iterable[ImageRef]:
        """Yield references to every image file that matches *extensions*.

        Args:
            extensions: Lower-case extensions to include (e.g. ``[".jpg"]``).
                        Implementations must match case-insensitively.

        Yields:
            :class:`ImageRef` for each matching file, in unspecified order.
        """
        ...

    @contextmanager
    def open_image(self, ref: ImageRef) -> Generator[Path, None, None]:
        """Context manager that ensures the image is available at a local path.

        * For :class:`LocalStorageProvider` this is a no-op — it simply
          yields ``ref.local_path``.
        * For :class:`GoogleDriveStorageProvider` the file is downloaded to
          the OS cache directory; the local copy is deleted on ``__exit__``.

        Args:
            ref: An :class:`ImageRef` previously returned by
                 :meth:`iter_images`.

        Yields:
            Absolute :class:`~pathlib.Path` to a readable local copy.
        """
        ...

    @property
    def description(self) -> str:
        """Human-readable description of this source (e.g. path or Drive folder name)."""
        ...


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------

class LocalStorageProvider:
    """Storage provider backed by a directory on the local file system."""

    def __init__(self, root: Path) -> None:
        self._root = root

    # -- StorageProvider interface ------------------------------------------

    def iter_images(self, extensions: Iterable[str]) -> Iterable[ImageRef]:
        ext_set = {e.lower() for e in extensions}
        for dirpath, _, filenames in os.walk(self._root):
            for filename in sorted(filenames):
                if Path(filename).suffix.lower() in ext_set:
                    path = Path(dirpath) / filename
                    yield ImageRef(name=filename, local_path=path)

    @contextmanager
    def open_image(self, ref: ImageRef) -> Generator[Path, None, None]:
        if ref.local_path is None:
            raise ValueError(
                f"LocalStorageProvider.open_image: ref has no local_path: {ref!r}"
            )
        yield ref.local_path

    @property
    def description(self) -> str:
        return str(self._root)

    def __repr__(self) -> str:
        return f"LocalStorageProvider({self._root!r})"


# ---------------------------------------------------------------------------
# Google Drive implementation
# ---------------------------------------------------------------------------

class GoogleDriveStorageProvider:
    """Storage provider backed by a folder on Google Drive.

    Images are listed via the Drive API.  :meth:`open_image` downloads the
    file to the OS cache directory and cleans up after the context exits.
    """

    def __init__(
        self,
        client: "GDriveClient",
        folder_id: str,
        folder_name: str = "",
        *,
        cache: Optional["GDriveCacheManager"] = None,
    ) -> None:
        self._client = client
        self._folder_id = folder_id
        self._folder_name = folder_name
        if cache is None:
            from app.gdrive.cache import get_cache_manager
            cache = get_cache_manager()
        self._cache = cache

    # -- StorageProvider interface ------------------------------------------

    def iter_images(self, extensions: Iterable[str]) -> Iterable[ImageRef]:
        ext_set = {e.lower() for e in extensions}
        for drive_file in self._client.list_folder(
            self._folder_id, include_subfolders=True
        ):
            if drive_file.is_folder:
                continue
            if Path(drive_file.name).suffix.lower() in ext_set:
                yield ImageRef(
                    name=drive_file.name,
                    remote_id=drive_file.file_id,
                    remote_parent_id=(
                        drive_file.parents[0] if drive_file.parents else None
                    ),
                    size=drive_file.size,
                    modified_time=drive_file.modified_time,
                    checksum=drive_file.md5_checksum,
                )

    @contextmanager
    def open_image(self, ref: ImageRef) -> Generator[Path, None, None]:
        if ref.remote_id is None:
            raise ValueError(
                f"GoogleDriveStorageProvider.open_image: ref has no remote_id: {ref!r}"
            )
        local = self._cache.allocate_path(ref.name)
        self._cache.mark_active(local)
        try:
            self._client.download_file(ref.remote_id, str(local))
            yield local
        finally:
            self._cache.release(local, delete=True)

    @property
    def description(self) -> str:
        return self._folder_name or self._folder_id

    def __repr__(self) -> str:
        return (
            f"GoogleDriveStorageProvider("
            f"folder_id={self._folder_id!r}, name={self._folder_name!r})"
        )
