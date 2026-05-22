"""Database engine and session management.

All application code should obtain sessions via :func:`get_session` or
the :func:`session_scope` context manager.  The engine is created once
at startup via :func:`init_db`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def init_db(db_path: Path | str) -> Engine:
    """Create (or open) the SQLite database and run schema migrations.

    Call once at application startup.

    Args:
        db_path: Absolute or relative path to the SQLite file.
                 The parent directory is created if it does not exist.

    Returns:
        The SQLAlchemy :class:`Engine` instance.
    """
    global _engine, _SessionFactory

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    log.info("Opening database: %s", url)

    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,  # Set True to debug SQL
    )

    # Enable WAL mode for better concurrency (readers don't block writers)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    # Create all tables that don't exist yet (idempotent)
    Base.metadata.create_all(engine)

    # Add columns introduced in later versions to existing databases
    _migrate_add_columns(engine)

    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    log.info("Database ready: %s tables", len(Base.metadata.tables))
    return engine


def _migrate_add_columns(engine: Engine) -> None:
    """Add columns that didn't exist in older schema versions (idempotent)."""
    new_columns = {
        "persons": [
            ("last_name",    "VARCHAR(255)"),
            ("first_name",   "VARCHAR(255)"),
            ("second_name",  "VARCHAR(255)"),
            ("nickname",     "VARCHAR(255)"),
            ("married_name",  "VARCHAR(255)"),
            ("birth_place",  "VARCHAR(512)"),
            ("birth_date",   "VARCHAR(64)"),
            ("death_place",  "VARCHAR(512)"),
            ("death_date",   "VARCHAR(64)"),
            ("is_protected", "BOOLEAN NOT NULL DEFAULT 0"),
        ],
        "images": [
            ("photo_date", "VARCHAR(128)"),
        ],
    }
    with engine.connect() as conn:
        for table, cols in new_columns.items():
            existing = {
                row[1]
                for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            }
            for col_name, col_type in cols:
                if col_name not in existing:
                    conn.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    )
                    log.info("Migration: added column %s.%s", table, col_name)
        conn.commit()


UNKNOWN_PERSON_NAME = "Ismeretlen"


def ensure_unknown_person() -> None:
    """Create the protected 'Ismeretlen' person if it does not yet exist."""
    from app.db.models import Person

    with session_scope() as session:
        existing = (
            session.query(Person)
            .filter(Person.is_protected == True)  # noqa: E712
            .first()
        )
        if existing is None:
            session.add(
                Person(
                    name=UNKNOWN_PERSON_NAME,
                    is_auto_named=False,
                    is_protected=True,
                )
            )
            log.info("Created protected person: '%s'", UNKNOWN_PERSON_NAME)


def get_engine() -> Engine:
    """Return the application-level engine (must call :func:`init_db` first)."""
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _engine


def get_session() -> Session:
    """Create and return a new :class:`Session`.

    The caller is responsible for closing it.  Prefer :func:`session_scope`
    for automatic commit/rollback handling.
    """
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _SessionFactory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional session scope.

    Commits on clean exit, rolls back on any exception, always closes.

    Usage::

        with session_scope() as session:
            session.add(some_object)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
