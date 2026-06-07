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
        # Wait up to 5s for a competing writer instead of failing immediately
        # with "database is locked" — background match jobs write concurrently
        # with foreground manual edits under WAL (single-writer).
        cursor.execute("PRAGMA busy_timeout=5000")
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
            ("gender",               "VARCHAR(16)"),
            ("family_code",          "VARCHAR(64)"),
            ("external_family_code", "VARCHAR(128)"),
            ("last_name",            "VARCHAR(255)"),
            ("first_name",           "VARCHAR(255)"),
            ("second_name",          "VARCHAR(255)"),
            ("nickname",             "VARCHAR(255)"),
            ("married_name",         "VARCHAR(255)"),
            ("birth_place",          "VARCHAR(512)"),
            ("birth_date",           "VARCHAR(64)"),
            ("death_place",          "VARCHAR(512)"),
            ("death_date",           "VARCHAR(64)"),
            ("notes",                "TEXT"),
            ("is_protected",         "BOOLEAN NOT NULL DEFAULT 0"),
            ("thumbnail_is_manual",  "BOOLEAN NOT NULL DEFAULT 0"),
        ],
        "places": [
            ("thumbnail_is_manual",     "BOOLEAN NOT NULL DEFAULT 0"),
            ("place_type",              "VARCHAR(16) NOT NULL DEFAULT 'area'"),
            ("accuracy_radius_meters",  "FLOAT"),
            ("parent_id",               "INTEGER REFERENCES places(id) ON DELETE SET NULL"),
            ("settlement_name",         "VARCHAR(255)"),
            ("street_name",             "VARCHAR(255)"),
            ("house_number",            "VARCHAR(32)"),
            ("display_name",            "VARCHAR(512)"),
            ("coordinate_source",       "VARCHAR(32)"),
            ("is_exact_coordinate",     "BOOLEAN NOT NULL DEFAULT 0"),
        ],
        "images": [
            ("photo_date", "VARCHAR(128)"),
            ("relative_path", "VARCHAR(1024)"),
            ("place_id", "INTEGER REFERENCES places(id) ON DELETE SET NULL"),
            ("exif_latitude", "FLOAT"),
            ("exif_longitude", "FLOAT"),
            ("image_latitude", "FLOAT"),
            ("image_longitude", "FLOAT"),
            ("note", "TEXT"),
        ],
        "faces": [
            ("assignment_source", "VARCHAR(32)"),
            ("assignment_confidence", "FLOAT"),
            ("assigned_at", "DATETIME"),
            ("quality_score", "FLOAT"),
            ("quality_reasons", "VARCHAR(256)"),
            ("is_low_quality", "BOOLEAN"),
            ("landmarks", "BLOB"),
            ("is_merge_excluded", "BOOLEAN NOT NULL DEFAULT 0"),
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
    _migrate_person_groups(engine)
    _migrate_collage_locations_to_places(engine)
    _migrate_place_types(engine)
    _migrate_geocoding_tables(engine)
    _migrate_place_display_name(engine)
    _migrate_object_tagging(engine)


def _migrate_object_tagging(engine: Engine) -> None:
    """Create the object-tagging tables if missing (idempotent).

    Objects are a domain entirely separate from face recognition: tagged_objects
    (the object identity), object_occurrences (one point per appearance in an
    image, future-proofed for bbox/polygon/AI), object_person_links (people
    related to the object) and object_aliases (merge provenance).
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS tagged_objects (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    name                VARCHAR(255) NOT NULL,
                    description         TEXT,
                    notes               TEXT,
                    thumbnail_path      TEXT,
                    thumbnail_is_manual BOOLEAN NOT NULL DEFAULT 0,
                    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at          DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_tagged_objects_name "
                "ON tagged_objects(name)"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS object_occurrences (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id        INTEGER NOT NULL
                        REFERENCES tagged_objects(id) ON DELETE CASCADE,
                    image_id         INTEGER NOT NULL
                        REFERENCES images(id) ON DELETE CASCADE,
                    point_x          INTEGER,
                    point_y          INTEGER,
                    bbox_x           INTEGER,
                    bbox_y           INTEGER,
                    bbox_w           INTEGER,
                    bbox_h           INTEGER,
                    polygon_json     TEXT,
                    geometry_type    VARCHAR(16) NOT NULL DEFAULT 'point',
                    detection_source VARCHAR(32) NOT NULL DEFAULT 'manual',
                    confidence       FLOAT,
                    note             TEXT,
                    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_object_occurrence "
                "ON object_occurrences(object_id, image_id, point_x, point_y)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_object_occ_image "
                "ON object_occurrences(image_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_object_occ_object "
                "ON object_occurrences(object_id)"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS object_person_links (
                    object_id  INTEGER NOT NULL
                        REFERENCES tagged_objects(id) ON DELETE CASCADE,
                    person_id  INTEGER NOT NULL
                        REFERENCES persons(id) ON DELETE CASCADE,
                    role       VARCHAR(64) NOT NULL DEFAULT 'other',
                    note       VARCHAR(255),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (object_id, person_id, role)
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_opl_person "
                "ON object_person_links(person_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_opl_object "
                "ON object_person_links(object_id)"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS object_aliases (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_id        INTEGER NOT NULL
                        REFERENCES tagged_objects(id) ON DELETE CASCADE,
                    source_object_id INTEGER,
                    name             VARCHAR(255),
                    description      TEXT,
                    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_object_aliases_object "
                "ON object_aliases(object_id)"
            )
        )
    log.debug("Migration: object tagging tables ensured")


def _migrate_geocoding_tables(engine: Engine) -> None:
    """Create geocoding_cache and place_address_suggestions if missing."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS geocoding_cache (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_type      VARCHAR(32) NOT NULL,
                    query_text      VARCHAR(512) NOT NULL,
                    settlement_name VARCHAR(255),
                    street_name     VARCHAR(255),
                    house_number    VARCHAR(32),
                    result_json     TEXT NOT NULL,
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_geocoding_query "
                "ON geocoding_cache(query_type, query_text)"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS place_address_suggestions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    settlement_name VARCHAR(255) NOT NULL,
                    street_name     VARCHAR(255),
                    source          VARCHAR(32) NOT NULL DEFAULT 'user',
                    last_used_at    DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_address_suggestion "
                "ON place_address_suggestions(settlement_name, street_name)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_address_suggestion_settlement "
                "ON place_address_suggestions(settlement_name)"
            )
        )
    log.debug("Migration: geocoding tables ensured")


def _migrate_place_display_name(engine: Engine) -> None:
    """Backfill display_name/coordinate_source for legacy place rows (idempotent).

    Preserves existing free-text names: display_name defaults to name, and the
    coordinate source is tagged 'imported' so legacy coordinates are not mistaken
    for precise geocoded addresses.
    """
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "places" not in tables:
            return
        columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(places)")).fetchall()
        }
        if "display_name" in columns:
            conn.execute(
                text(
                    "UPDATE places SET display_name = name "
                    "WHERE display_name IS NULL OR display_name = ''"
                )
            )
        if "coordinate_source" in columns:
            # EXIF-sourced rows keep an 'exif' tag; everything else legacy → 'imported'.
            if "source" in columns:
                conn.execute(
                    text(
                        "UPDATE places SET coordinate_source = 'exif' "
                        "WHERE coordinate_source IS NULL AND source = 'exif'"
                    )
                )
            conn.execute(
                text(
                    "UPDATE places SET coordinate_source = 'imported' "
                    "WHERE coordinate_source IS NULL"
                )
            )
    log.debug("Migration: place display_name/coordinate_source backfilled")


def _migrate_place_types(engine: Engine) -> None:
    """Backfill place_type/accuracy_radius for legacy rows (idempotent).

    Existing places become "area" (the column default already handles new
    columns, but this also covers any NULL left by older partial migrations)
    and receive the area-sized accuracy radius when none was set.
    """
    default_radius = {"exact": 50.0, "area": 5000.0, "region": 50000.0}
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "places" not in tables:
            return
        columns = {
            row[1]
            for row in conn.execute(text("PRAGMA table_info(places)")).fetchall()
        }
        if "place_type" not in columns:
            return
        conn.execute(
            text("UPDATE places SET place_type = 'area' WHERE place_type IS NULL")
        )
        if "accuracy_radius_meters" in columns:
            for place_type, radius in default_radius.items():
                conn.execute(
                    text(
                        "UPDATE places SET accuracy_radius_meters = :r "
                        "WHERE accuracy_radius_meters IS NULL AND place_type = :t"
                    ),
                    {"r": radius, "t": place_type},
                )
    log.debug("Migration: place_type/accuracy_radius backfilled")


def _migrate_add_indexes(engine: Engine) -> None:
    """Create indexes used by person/image and family queries."""
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_persons_gender ON persons(gender)",
        # Friend/acquaintance codes (containing "B") are relational markers, not
        # unique identities, so they are excluded from the uniqueness rule.
        # Older databases may have an unconditional unique index on family_code
        # (from the column-level unique=True) or a NOT NULL-only partial index;
        # drop both and recreate them with the friend-aware definitions.
        "DROP INDEX IF EXISTS ix_persons_family_code",
        "CREATE INDEX IF NOT EXISTS ix_persons_family_code ON persons(family_code)",
        "DROP INDEX IF EXISTS ux_persons_family_code",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_persons_family_code "
            "ON persons(family_code) "
            "WHERE family_code IS NOT NULL AND family_code NOT LIKE '%B%'"
        ),
        # External family identifiers are unique identities (each #root#path is
        # one external person), so enforce uniqueness where present.
        "CREATE INDEX IF NOT EXISTS ix_persons_external_family_code ON persons(external_family_code)",
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_persons_external_family_code "
            "ON persons(external_family_code) "
            "WHERE external_family_code IS NOT NULL"
        ),
        "CREATE INDEX IF NOT EXISTS ix_faces_person_image ON faces(person_id, image_id)",
        "CREATE INDEX IF NOT EXISTS ix_faces_image_person ON faces(image_id, person_id)",
        "CREATE INDEX IF NOT EXISTS ix_places_type ON places(place_type)",
        "CREATE INDEX IF NOT EXISTS ix_places_parent ON places(parent_id)",
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


def _migrate_person_groups(engine: Engine) -> None:
    """Create person_groups and person_group_memberships tables if missing (idempotent)."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS person_groups (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      VARCHAR(255) NOT NULL,
                    description TEXT,
                    color     VARCHAR(32),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_person_groups_name "
                "ON person_groups(name)"
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS person_group_memberships (
                    person_id INTEGER NOT NULL
                        REFERENCES persons(id) ON DELETE CASCADE,
                    group_id  INTEGER NOT NULL
                        REFERENCES person_groups(id) ON DELETE CASCADE,
                    PRIMARY KEY (person_id, group_id)
                )
                """
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_pgm_group_id "
                "ON person_group_memberships(group_id)"
            )
        )
    log.debug("Migration: person_groups tables ensured")


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
