"""Helpers for accepting Drive folder URLs / IDs from the user."""

from __future__ import annotations

import re
from typing import Optional

# A Drive folder URL looks like:
#   https://drive.google.com/drive/folders/<ID>?usp=sharing
#   https://drive.google.com/drive/u/0/folders/<ID>
#   https://drive.google.com/open?id=<ID>
#
# We also accept a bare ID (no slashes, no spaces, 10+ chars of [A-Za-z0-9_-]).
_URL_PATTERN = re.compile(
    r"folders/([A-Za-z0-9_-]{10,})"
)
_OPEN_PATTERN = re.compile(
    r"[?&]id=([A-Za-z0-9_-]{10,})"
)
_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{10,}$")


def parse_folder_input(raw: str) -> Optional[str]:
    """Extract a Drive folder ID from *raw* user input.

    Returns ``None`` when no plausible ID can be found.  The function does
    NOT verify that the ID actually exists on Drive — that has to be done
    by calling ``GDriveClient.get_metadata`` afterwards.
    """
    if not raw:
        return None
    text = raw.strip()
    m = _URL_PATTERN.search(text)
    if m:
        return m.group(1)
    m = _OPEN_PATTERN.search(text)
    if m:
        return m.group(1)
    if _BARE_ID.match(text):
        return text
    return None
