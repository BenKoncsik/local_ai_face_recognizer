"""Tiny `.env` file loader — no external dependency.

Reads ``<project root>/.env`` (or a path of your choice) and copies the
recognised variables into :data:`os.environ`, **without overwriting any
already-set value**.  This lets the user keep their Google OAuth
``CLIENT_SECRET`` out of the public repository while still letting the
desktop OAuth flow find it.

File format (subset of full `.env` syntax)::

    # comments and blank lines are ignored
    FACE_LOCAL_GOOGLE_CLIENT_ID=695041722567-…apps.googleusercontent.com
    FACE_LOCAL_GOOGLE_CLIENT_SECRET=GOCSPX-…
    KEY_WITH_QUOTES="value with spaces"

We intentionally keep the parser minimal so it has no third-party
dependency.  Anything more complex (multi-line values, variable expansion,
etc.) can be done by adding `python-dotenv` later.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

# Variables we are willing to copy from a .env file.  This whitelist keeps
# the loader from accidentally pulling unrelated, possibly sensitive
# values into the process environment.
_ALLOWED_VARS: frozenset[str] = frozenset({
    "FACE_LOCAL_GOOGLE_CLIENT_ID",
    "FACE_LOCAL_GOOGLE_CLIENT_SECRET",
    "FACE_LOCAL_CONFIG",
})


def load_dotenv(path: Optional[Path] = None, *, allowed: Iterable[str] = _ALLOWED_VARS) -> int:
    """Read *path* (default: ``./.env``) and populate :data:`os.environ`.

    Args:
        path: Explicit .env file path; ``None`` means "look at the current
              working directory's `.env`".
        allowed: Whitelist of variable names that may be imported.  Lines
                 with other names are silently ignored.

    Returns:
        Number of variables actually copied into the environment.  Zero
        when the file does not exist (this is not an error).
    """
    env_file = path or Path(".env")
    if not env_file.is_file():
        return 0

    allow = set(allowed)
    n = 0
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip an optional leading "export "
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if key not in allow:
                continue

            # Strip matching surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            # Do NOT overwrite an already-set process environment variable —
            # callers may have exported it explicitly.
            if key in os.environ:
                continue
            os.environ[key] = value
            n += 1
        if n:
            log.info("Loaded %d variable(s) from %s", n, env_file)
    except OSError as exc:
        log.warning("Could not read %s: %s", env_file, exc)
    return n
