"""Small SQL query helpers shared across services.

SQLite caps the number of host parameters (``?``) per statement.  Newer builds
allow 32766, but the build shipped with Python on Windows still uses the older
999 limit — so an ``IN (?, ?, …)`` over a large id list raises
``OperationalError: too many SQL variables`` on Windows even when it works on
macOS/Linux.  :func:`in_chunks` splits an id list into safe-sized batches.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, TypeVar

# Stay comfortably under SQLite's lowest common limit (999), leaving room for
# the handful of other bound parameters a query may carry.
SQLITE_MAX_VARS = 900

T = TypeVar("T")


def in_chunks(items: Sequence[T], size: int = SQLITE_MAX_VARS) -> Iterator[List[T]]:
    """Yield *items* in lists of at most *size* elements.

    Use to fan an ``IN (...)`` filter out over several statements so the bound
    parameter count never exceeds SQLite's per-statement limit::

        for chunk in in_chunks(image_ids):
            rows += session.query(...).filter(Model.id.in_(chunk)).all()

    An empty input yields nothing.
    """
    seq = list(items)
    for start in range(0, len(seq), size):
        yield seq[start:start + size]
