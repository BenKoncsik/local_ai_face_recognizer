"""Concrete Google Drive client built on google-api-python-client.

Implements the :class:`~app.gdrive.protocols.GDriveClient` protocol on top
of the official Google client library.  The Drive API specifics
(pagination, MediaIoBaseDownload chunks, exponential backoff for 429/5xx)
are isolated here so the rest of the app stays Drive-agnostic.

Design notes
------------

* The client is constructed from a :class:`StoredCredential`; the OAuth
  dance itself lives in :mod:`app.gdrive.oauth_flow`.
* When a token refresh happens, the updated access-token and expiry are
  pushed back to the :class:`CredentialService` so the next process can
  reuse them.  Refresh tokens never change after the first issue.
* Every public method calls :func:`require_online` first so callers get
  a localised "no internet" message rather than a confusing
  ``httplib2.ServerNotFoundError``.
* Transient failures (HTTP 429 / 500 / 503) are retried with capped
  exponential backoff up to :data:`_MAX_RETRIES` times.
"""

from __future__ import annotations

import io
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

from app.gdrive import oauth_config
from app.gdrive.connectivity import GDriveOfflineError, require_online
from app.gdrive.credential_store import (
    CredentialService,
    StoredCredential,
    get_credential_store,
)
from app.gdrive.oauth_flow import restore_credentials_object
from app.gdrive.protocols import DriveAccountInfo, DriveFile, GDriveClient

log = logging.getLogger(__name__)

# How many transient-failure retries before giving up.
_MAX_RETRIES = 5
# Base back-off in seconds (multiplied by 2^attempt + jitter).
_BASE_BACKOFF = 0.5

# MIME type for Drive folders
_FOLDER_MIME = "application/vnd.google-apps.folder"

# Fields requested for file metadata — kept small to reduce quota usage.
_FILE_FIELDS = (
    "id,name,mimeType,size,modifiedTime,md5Checksum,parents"
)
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"


# ---------------------------------------------------------------------------
# Driver factory
# ---------------------------------------------------------------------------

def build_drive_client(
    account_email: str,
    *,
    credential_store: Optional[CredentialService] = None,
) -> "GoogleDriveClient":
    """Create a :class:`GoogleDriveClient` for *account_email*.

    Reads the credential from the credential store, refreshes the token if
    needed, and returns a ready-to-use client.

    Raises:
        RuntimeError: if no credentials are stored for *account_email*.
        GDriveOfflineError: if the network is unreachable.
    """
    store = credential_store or get_credential_store()
    stored = store.load(account_email)
    if stored is None:
        raise RuntimeError(
            f"No stored Google credentials for {account_email!r}. "
            "Sign in first via Settings → Google Drive."
        )
    return GoogleDriveClient(stored, credential_store=store)


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

class GoogleDriveClient(GDriveClient):
    """:class:`GDriveClient` implementation backed by the official SDK."""

    def __init__(
        self,
        stored_credential: StoredCredential,
        *,
        credential_store: Optional[CredentialService] = None,
    ) -> None:
        self._stored = stored_credential
        self._credential_store = credential_store or get_credential_store()
        self._account = DriveAccountInfo(email=stored_credential.account_email)
        self._creds = None  # Lazily built — see _ensure_creds()
        self._service = None  # Lazily built — see _ensure_service()

    # ------------------------------------------------------------------
    # Protocol — identity
    # ------------------------------------------------------------------

    @property
    def account(self) -> DriveAccountInfo:
        return self._account

    # ------------------------------------------------------------------
    # Protocol — file transfer
    # ------------------------------------------------------------------

    def download_file(self, file_id: str, local_path: str) -> None:
        require_online()
        from googleapiclient.http import MediaIoBaseDownload

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        def _do() -> None:
            service = self._ensure_service()
            request = service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )
            with open(local_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
                done = False
                while not done:
                    _status, done = downloader.next_chunk()

        self._with_retry(_do, op_label=f"download {file_id}")
        log.info("Drive download complete: file_id=%s → %s", file_id, local_path)

    def upload_file(
        self,
        local_path: str,
        *,
        parent_folder_id: str,
        remote_name: Optional[str] = None,
        existing_file_id: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> str:
        require_online()
        from googleapiclient.http import MediaFileUpload

        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"Upload source not found: {local_path}")

        name = remote_name or path.name
        media = MediaFileUpload(
            str(path),
            mimetype=mime_type or "application/octet-stream",
            resumable=True,
            chunksize=8 * 1024 * 1024,
        )

        def _do() -> str:
            service = self._ensure_service()
            if existing_file_id:
                # Update keeps the file_id stable — important for our
                # database "the lock & DB live at fixed file_ids" model.
                result = service.files().update(
                    fileId=existing_file_id,
                    media_body=media,
                    body={"name": name},
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
            else:
                result = service.files().create(
                    media_body=media,
                    body={"name": name, "parents": [parent_folder_id]},
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
            return result["id"]

        file_id = self._with_retry(_do, op_label=f"upload {name}")
        log.info(
            "Drive upload complete: %s → file_id=%s (parent=%s)",
            name, file_id, parent_folder_id,
        )
        return file_id

    # ------------------------------------------------------------------
    # Protocol — folder / metadata
    # ------------------------------------------------------------------

    def list_folder(
        self,
        folder_id: str,
        *,
        mime_types: Optional[Iterable[str]] = None,
        include_subfolders: bool = False,
    ) -> List[DriveFile]:
        require_online()

        query_parts = [f"'{folder_id}' in parents", "trashed=false"]
        if mime_types is not None:
            # Whitelist of allowed mime types.
            mime_filter = " or ".join(
                f"mimeType='{m}'" for m in mime_types
            )
            query_parts.append(f"({mime_filter})")
        if not include_subfolders:
            query_parts.append(f"mimeType != '{_FOLDER_MIME}'")
        query = " and ".join(query_parts)

        results: List[DriveFile] = []
        page_token: Optional[str] = None

        def _do() -> dict:
            service = self._ensure_service()
            return service.files().list(
                q=query,
                pageSize=200,
                pageToken=page_token,
                fields=_LIST_FIELDS,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

        while True:
            resp = self._with_retry(_do, op_label=f"list {folder_id}")
            for raw in resp.get("files", []):
                results.append(_from_raw(raw))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return results

    def find_child(self, folder_id: str, name: str) -> Optional[DriveFile]:
        require_online()
        # Drive's name field supports an = operator with escaped quotes.
        escaped = name.replace("\\", "\\\\").replace("'", "\\'")
        query = (
            f"'{folder_id}' in parents and trashed=false and name='{escaped}'"
        )

        def _do() -> dict:
            service = self._ensure_service()
            return service.files().list(
                q=query,
                pageSize=1,
                fields=_LIST_FIELDS,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()

        resp = self._with_retry(_do, op_label=f"find {name}")
        files = resp.get("files", [])
        return _from_raw(files[0]) if files else None

    def get_metadata(self, file_id: str) -> DriveFile:
        require_online()

        def _do() -> dict:
            service = self._ensure_service()
            return service.files().get(
                fileId=file_id,
                fields=_FILE_FIELDS,
                supportsAllDrives=True,
            ).execute()

        return _from_raw(self._with_retry(_do, op_label=f"metadata {file_id}"))

    def create_folder(self, parent_folder_id: str, name: str) -> DriveFile:
        require_online()

        def _do() -> dict:
            service = self._ensure_service()
            return service.files().create(
                body={
                    "name": name,
                    "mimeType": _FOLDER_MIME,
                    "parents": [parent_folder_id],
                },
                fields=_FILE_FIELDS,
                supportsAllDrives=True,
            ).execute()

        raw = self._with_retry(_do, op_label=f"mkdir {name}")
        log.info("Drive folder created: %s (parent=%s)", name, parent_folder_id)
        return _from_raw(raw)

    def delete_file(self, file_id: str) -> None:
        require_online()
        from googleapiclient.errors import HttpError

        def _do() -> None:
            service = self._ensure_service()
            try:
                service.files().delete(
                    fileId=file_id, supportsAllDrives=True
                ).execute()
            except HttpError as exc:
                if exc.resp.status == 404:
                    return  # idempotent on missing
                raise

        self._with_retry(_do, op_label=f"delete {file_id}")
        log.info("Drive file deleted: %s", file_id)

    # ------------------------------------------------------------------
    # Internals — service + token refresh
    # ------------------------------------------------------------------

    def _ensure_creds(self):  # noqa: ANN001 — google credentials type
        if self._creds is None:
            self._creds = restore_credentials_object(self._stored)
        return self._creds

    def _ensure_service(self):  # noqa: ANN001 — googleapiclient resource
        if self._service is None:
            try:
                from googleapiclient.discovery import build
            except ImportError as exc:
                raise RuntimeError(
                    "google-api-python-client is not installed. "
                    "Run: pip install '.[gdrive]'"
                ) from exc
            creds = self._ensure_creds()
            self._service = build(
                "drive", "v3", credentials=creds, cache_discovery=False,
            )
        return self._service

    def _persist_refreshed_token(self) -> None:
        """If google-auth refreshed the access token, update the store."""
        creds = self._creds
        if creds is None or not creds.token or creds.token == self._stored.access_token:
            return
        updated = StoredCredential(
            account_email=self._stored.account_email,
            refresh_token=self._stored.refresh_token,
            access_token=creds.token,
            token_expiry=creds.expiry.isoformat() if creds.expiry else None,
            client_id=creds.client_id or oauth_config.CLIENT_ID,
            client_secret=creds.client_secret or oauth_config.CLIENT_SECRET,
            token_uri=creds.token_uri,
            scopes=list(creds.scopes or []),
            saved_at=datetime.utcnow().isoformat(),
        )
        try:
            self._credential_store.save(updated)
            self._stored = updated
            log.debug("Persisted refreshed access token for %s", updated.account_email)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not persist refreshed token: %s", exc)

    # ------------------------------------------------------------------
    # Retry helper
    # ------------------------------------------------------------------

    def _with_retry(self, op, *, op_label: str):  # noqa: ANN001, ANN201
        from googleapiclient.errors import HttpError

        attempt = 0
        while True:
            try:
                result = op()
                # Successful call → persist any refreshed token & return.
                self._persist_refreshed_token()
                return result
            except GDriveOfflineError:
                raise  # No retry — connectivity issue.
            except HttpError as exc:
                status = exc.resp.status if exc.resp is not None else 0
                if status in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
                    delay = _BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
                    log.warning(
                        "Drive API %s for %s — retrying in %.1fs (attempt %d/%d)",
                        status, op_label, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                if status == 401:
                    # Token may have expired; the next attempt will refresh.
                    if attempt == 0:
                        attempt += 1
                        continue
                log.error("Drive API error during %s: %s", op_label, exc)
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("Unexpected error during %s: %s", op_label, exc)
                raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _from_raw(raw: dict) -> DriveFile:
    """Build a :class:`DriveFile` from a Drive API ``files`` response."""
    size_raw = raw.get("size")
    size: Optional[int] = None
    if size_raw is not None:
        try:
            size = int(size_raw)
        except (TypeError, ValueError):
            size = None
    return DriveFile(
        file_id=raw["id"],
        name=raw["name"],
        mime_type=raw.get("mimeType", "application/octet-stream"),
        size=size,
        modified_time=raw.get("modifiedTime"),
        md5_checksum=raw.get("md5Checksum"),
        parents=tuple(raw.get("parents", []) or []),
    )
