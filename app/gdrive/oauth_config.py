"""OAuth client configuration for Google Drive sign-in.

The application uses a *public desktop OAuth client*: a single ``client_id``
is baked into the build and shared by every Face-Local user.  Desktop clients
do not have a confidential ``client_secret`` in the security sense — Google
generates a value for compatibility with the OAuth spec, but it is treated as
non-secret and is shipped with the app.

Setup instructions for the maintainer
-------------------------------------

1. Open https://console.cloud.google.com/apis/credentials
2. Create OAuth 2.0 Client ID  →  Application type: *Desktop app*.
3. Enable the **Google Drive API** in *APIs & Services → Library*.
4. Add the Drive scopes listed below to the consent screen.
5. Replace :data:`CLIENT_ID` and :data:`CLIENT_SECRET` below with the values
   from the Cloud Console.  Until they are real, the OAuth flow will fail
   with a clear error message.

Override at runtime
-------------------

Power users can override the bundled values without rebuilding the app by
exporting these environment variables before launching:

* ``FACE_LOCAL_GOOGLE_CLIENT_ID``
* ``FACE_LOCAL_GOOGLE_CLIENT_SECRET``

This is useful for developers running against their own Cloud project and
during automated tests.
"""

from __future__ import annotations

import os
from typing import List

# ---------------------------------------------------------------------------
# Bundled OAuth client (Google Cloud Console → Desktop app).
#
# Replace CLIENT_SECRET below with the value paired with the CLIENT_ID in your
# Google Cloud Console (Credentials → click the desktop OAuth client → the
# "Client secret" string).  The secret is technically required by the OAuth
# spec but for desktop clients Google treats it as non-confidential.
# ---------------------------------------------------------------------------

# Sentinel string used to detect that the values are still placeholders.
_PLACEHOLDER = "REPLACE_ME"

_BUNDLED_CLIENT_ID = (
    "695041722567-6troo6v3cmq28j7ccj1r5kds67mvvetd.apps.googleusercontent.com"
)
_BUNDLED_CLIENT_SECRET = _PLACEHOLDER  # paste from Google Cloud Console here

CLIENT_ID: str = os.environ.get("FACE_LOCAL_GOOGLE_CLIENT_ID", _BUNDLED_CLIENT_ID)
CLIENT_SECRET: str = os.environ.get(
    "FACE_LOCAL_GOOGLE_CLIENT_SECRET", _BUNDLED_CLIENT_SECRET
)

# OAuth scopes required by Face-Local.
#
# * ``drive`` gives full access to all Drive files the signed-in user can
#   see — required for accessing folders shared with the user and for
#   reading/writing image files in arbitrary Drive locations.
#   (``drive.file`` would restrict access to files the app created itself,
#   which is too narrow for shared-folder workflows.)
# * ``userinfo.email`` and ``openid`` let us show the signed-in account
#   name in the Settings dialog.
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/drive",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# Loopback redirect URI used by the installed-app flow.  Google's
# google-auth-oauthlib library handles the port allocation automatically.
REDIRECT_HOST: str = "localhost"


def is_configured() -> bool:
    """Return True if a real client_id / client_secret are available.

    The bundled placeholder values still allow the rest of the app to load,
    but any actual sign-in attempt should be blocked with a clear message
    until the maintainer fills in the real credentials.
    """
    return _PLACEHOLDER not in CLIENT_ID and _PLACEHOLDER not in CLIENT_SECRET


def client_config() -> dict:
    """Return the ``client_config`` dict expected by google-auth-oauthlib."""
    return {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [f"http://{REDIRECT_HOST}"],
        }
    }
