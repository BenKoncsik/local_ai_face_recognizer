"""Google OAuth 2.0 installed-application flow.

Encapsulates the browser-based sign-in dance so the UI layer only sees a
single ``run_login_flow()`` call returning a :class:`StoredCredential`.

The flow:

1. ``InstalledAppFlow`` from ``google-auth-oauthlib`` opens the default
   browser pointing at Google's consent screen.
2. A short-lived local HTTP server on a random loopback port receives the
   redirect with the authorisation code.
3. The code is exchanged for an access + refresh token over HTTPS.
4. We hit ``oauth2.googleapis.com/userinfo`` once to learn the account
   email so we can key the stored credential by user.
5. The token is wrapped in a :class:`StoredCredential` and handed to the
   caller, who is expected to persist it via :class:`CredentialService`.

Two exceptions are raised that callers should handle:

* :class:`GDriveOfflineError` — no network connectivity.
* :class:`OAuthConfigError` — the bundled placeholder OAuth client_id has
  not been replaced with real credentials yet.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from app.gdrive import oauth_config
from app.gdrive.connectivity import GDriveOfflineError, require_online
from app.gdrive.credential_store import StoredCredential

log = logging.getLogger(__name__)


class OAuthConfigError(RuntimeError):
    """Raised when the OAuth client credentials are missing or placeholder."""


class OAuthCancelled(RuntimeError):
    """Raised when the user cancels the OAuth flow (closes browser, etc.)."""


def run_login_flow(timeout_seconds: int = 300) -> StoredCredential:
    """Open the browser and complete the Google sign-in flow.

    Blocks until the user finishes (or cancels) the consent screen.  Designed
    to be called from a background thread so the GUI does not freeze.

    Args:
        timeout_seconds: Max seconds to wait for the redirect.  Default 5 min.

    Returns:
        A populated :class:`StoredCredential`.

    Raises:
        OAuthConfigError:    Bundled credentials are still placeholders.
        GDriveOfflineError:  No network connectivity to Google.
        OAuthCancelled:      User aborted the flow.
        RuntimeError:        Any other Google API / OAuth failure.
    """
    if not oauth_config.is_configured():
        raise OAuthConfigError(
            "Google OAuth client_id is not configured. "
            "Replace the placeholder in app/gdrive/oauth_config.py or set "
            "FACE_LOCAL_GOOGLE_CLIENT_ID / FACE_LOCAL_GOOGLE_CLIENT_SECRET."
        )

    require_online()  # Fast-fail with a clear error if we can't reach Google

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise OAuthConfigError(
            "google-auth-oauthlib is not installed. "
            "Run: pip install '.[gdrive]'"
        ) from exc

    log.info("Starting Google OAuth installed-app flow …")
    flow = InstalledAppFlow.from_client_config(
        oauth_config.client_config(),
        scopes=oauth_config.SCOPES,
    )

    try:
        creds = flow.run_local_server(
            host=oauth_config.REDIRECT_HOST,
            port=0,  # let the OS pick a free port
            authorization_prompt_message=(
                "Megnyitottuk a böngészőt a Google bejelentkezéshez. "
                "Térj vissza ide, amikor végeztél."
            ),
            success_message=(
                "Sikeres bejelentkezés. Visszatérhetsz a Face-Local alkalmazáshoz."
            ),
            open_browser=True,
            timeout_seconds=timeout_seconds,
        )
    except GDriveOfflineError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Distinguish user cancel from generic errors where we can.
        msg = str(exc).lower()
        if "cancel" in msg or "access_denied" in msg or "timeout" in msg:
            raise OAuthCancelled(str(exc)) from exc
        raise RuntimeError(f"OAuth flow failed: {exc}") from exc

    # creds is a google.oauth2.credentials.Credentials.  Extract identity.
    account_email = _fetch_account_email(creds)

    return StoredCredential(
        account_email=account_email,
        refresh_token=creds.refresh_token,
        access_token=creds.token,
        token_expiry=creds.expiry.isoformat() if creds.expiry else None,
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        token_uri=creds.token_uri,
        scopes=list(creds.scopes or []),
        saved_at=datetime.utcnow().isoformat(),
    )


def _fetch_account_email(creds) -> str:  # noqa: ANN001 — google credentials type
    """Call Google's userinfo endpoint to learn the signed-in account email."""
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise OAuthConfigError(
            "google-api-python-client is not installed. "
            "Run: pip install '.[gdrive]'"
        ) from exc

    try:
        service = build("oauth2", "v2", credentials=creds, cache_discovery=False)
        info = service.userinfo().get().execute()
        email = info.get("email")
        if not email:
            raise RuntimeError("Userinfo response did not include an email.")
        return email
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not fetch userinfo: %s", exc)
        # Fall back to a synthetic identifier so the credential is still
        # storable; the user can re-add the account later if needed.
        return "unknown@gdrive"


def restore_credentials_object(stored: StoredCredential):  # noqa: ANN201
    """Rebuild a ``google.oauth2.credentials.Credentials`` from storage.

    Returned object can be passed to ``googleapiclient.discovery.build``.
    Performs a token refresh on first use if the access token is expired.
    """
    try:
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise OAuthConfigError(
            "google-auth is not installed. Run: pip install '.[gdrive]'"
        ) from exc

    expiry: Optional[datetime] = None
    if stored.token_expiry:
        try:
            expiry = datetime.fromisoformat(stored.token_expiry)
        except ValueError:
            expiry = None

    creds = Credentials(
        token=stored.access_token,
        refresh_token=stored.refresh_token,
        token_uri=stored.token_uri,
        client_id=stored.client_id or oauth_config.CLIENT_ID,
        client_secret=stored.client_secret or oauth_config.CLIENT_SECRET,
        scopes=list(stored.scopes),
    )
    if expiry is not None:
        creds.expiry = expiry
    return creds
