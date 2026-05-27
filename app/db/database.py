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

    # Initialise the image library service for this database path
    from app.services.image_library_service import init_image_library
    init_image_library(db_path)

    log.info("Database ready: %s tables", len(Base.metadata.tables))
    return engine


def _migrate_add_columns(engine: Engine) -> None:
    """Add columns that didn't exist in older schema versions (idempotent)."""
    new_columns = {
        "persons": [
            ("gender",       "VARCHAR(16)"),
            ("family_code",  "VARCHAR(64)"),
            ("last_name",    "VARCHAR(255)"),
            ("first_name",   "VARCHAR(255)"),
            ("second_name",  "VARCHAR(255)"),
            ("nickname",     "VARCHAR(255)"),
            ("married_name",  "VARCHAR(255)"),
            ("birth_place",  "VARCHAR(512)"),
            ("birth_date",   "VARCHAR(64)"),
            ("death_place",  "VARCHAR(512)"),
            ("death_date",   "VARCHAR(64)"),
            ("notes",        "TEXT"),
            ("is_protected", "BOOLEAN NOT NULL DEFAULT 0"),
        ],
        "images": [
            ("photo_date", "VARCHAR(128)"),
            ("relative_path", "VARCHAR(1024)"),
            ("place_id", "INTEGER REFERENCES places(id) ON DELETE SET NULL"),
            ("exif_latitude", "FLOAT"),
            ("exif_longitude", "FLOAT"),
        ],
        "faces": [
            ("assignment_source", "VARCHAR(32)"),
            ("assignment_confidence", "FLOAT"),
            ("assigned_at", "DATETIME"),
            ("quality_score", "FLOAT"),
            ("quality_reasons", "VARCHAR(256)"),
            ("is_low_quality", "BOOLEAN"),
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
    _migrate_add_indexes(engine)
    _migrate_collage_locations_to_places(engine)


def _migrate_add_indexes(engine: Engine) -> None:
    """Create indexes used by person/image and family queries."""
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_persons_gender ON persons(gender)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_persons_family_code "
            "ON persons(family_code) WHERE family_code IS NOT NULL"
        ),
        "CREATE INDEX IF NOT EXISTS ix_faces_person_image ON faces(person_id, image_id)",
        "CREATE INDEX IF NOT EXISTS ix_faces_image_person ON faces(image_id, person_id)",
        (
            "CREATE INDEX IF NOT EXISTS ix_relationship_type_a "
            "ON relationships(relationship_type, person_a_id)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS ix_relationship_type_b "
            "ON relationships(relationship_type, person_b_id)"
        ),
    ]
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        for stmt in statements:
            if "relationships" in stmt and "relationships" not in tables:
                continue
            conn.execute(text(stmt))


def _migrate_collage_locations_to_places(engine: Engine) -> None:
    """Promote legacy collage node location text to reusable Place records."""
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if not {"places", "images", "collage_nodes"}.issubset(tables):
            return

        rows = conn.execute(
            text(
                """
                SELECT DISTINCT TRIM(location) AS name
                FROM collage_nodes
                WHERE image_id IS NOT NULL
                  AND location IS NOT NULL
                  AND TRIM(location) != ''
                """
            )
        ).fetchall()

        created = 0
        for row in rows:
            name = row[0]
            existing = conn.execute(
                text("SELECT id FROM places WHERE lower(name) = lower(:name) LIMIT 1"),
                {"name": name},
            ).fetchone()
            if existing is None:
                conn.execute(
                    text(
                        """
                        INSERT INTO places
                            (name, is_anonymous, source, created_at, updated_at)
                        VALUES
                            (:name, 0, 'legacy_collage', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """
                    ),
                    {"name": name},
                )
                created += 1

        updated = conn.execute(
            text(
                """
                UPDATE images
                SET place_id = (
                    SELECT p.id
                    FROM collage_nodes cn
                    JOIN places p ON lower(p.name) = lower(TRIM(cn.location))
                    WHERE cn.image_id = images.id
                      AND cn.location IS NOT NULL
                      AND TRIM(cn.location) != ''
                    ORDER BY cn.id
                    LIMIT 1
                )
                WHERE place_id IS NULL
                  AND EXISTS (
                    SELECT 1
                    FROM collage_nodes cn
                    WHERE cn.image_id = images.id
                      AND cn.location IS NOT NULL
                      AND TRIM(cn.location) != ''
                  )
                """
            )
        ).rowcount

    if created or updated:
        log.info(
            "Migration: promoted collage locations to places (created=%d, linked=%d)",
            created,
            updated,
        )


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
