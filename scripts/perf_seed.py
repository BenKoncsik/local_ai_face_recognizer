"""Synthetic database generator for performance/load testing.

Creates a fully migrated Face-Local database populated with a configurable
number of persons, images and faces (with realistic 512-dim float32
embeddings), so UI and service hot paths can be measured at scale without
real photos.

Usage::

    python -m scripts.perf_seed --db /tmp/perf_1000p_10k.db \
        --persons 1000 --images 10000 --faces-per-image 2.0
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np

EMBED_DIM = 512

FIRST_NAMES = [
    "Anna", "Béla", "Csaba", "Dóra", "Erzsébet", "Ferenc", "Gábor", "Hanna",
    "István", "János", "Katalin", "László", "Mária", "Nóra", "Olivér", "Péter",
    "Réka", "Sándor", "Tamás", "Zsófia",
]
LAST_NAMES = [
    "Kovács", "Szabó", "Tóth", "Nagy", "Horváth", "Varga", "Kiss", "Molnár",
    "Németh", "Farkas", "Balogh", "Papp", "Takács", "Juhász", "Lakatos",
]


def seed_database(
    db_path: Path,
    n_persons: int,
    n_images: int,
    faces_per_image: float,
    rng_seed: int = 42,
) -> dict:
    """Create and populate *db_path*; returns counts + elapsed seconds."""
    from app.db.database import get_engine, init_db

    t0 = time.perf_counter()
    if db_path.exists():
        db_path.unlink()
    init_db(db_path)
    engine = get_engine()
    rng = random.Random(rng_seed)
    np_rng = np.random.default_rng(rng_seed)

    with engine.begin() as conn:
        # Persons — ~10% unknown clusters, the rest named.
        person_rows = []
        for i in range(n_persons):
            if i % 10 == 0:
                name = f"Ismeretlen {i // 10 + 1}"
                auto = True
            else:
                last = rng.choice(LAST_NAMES)
                first = rng.choice(FIRST_NAMES)
                name = f"{last} {first} {i}"
                auto = False
            person_rows.append(
                {
                    "name": name,
                    "is_auto_named": auto,
                    "is_protected": False,
                    "gender": rng.choice([None, "male", "female"]),
                    "family_code": f"C{i}" if not auto else None,
                    "last_name": None if auto else name.split()[0],
                    "first_name": None if auto else name.split()[1],
                    "birth_place": rng.choice(
                        [None, "Budapest", "Szeged", "Debrecen", "Pécs"]
                    ),
                }
            )
        _bulk_insert(conn, "persons", person_rows)

        image_rows = []
        for i in range(n_images):
            image_rows.append(
                {
                    "file_path": f"/library/album_{i % 50:02d}/img_{i:06d}.jpg",
                    "relative_path": f"album_{i % 50:02d}/img_{i:06d}.jpg",
                    "file_hash": f"{i:064x}",
                    "file_mtime": 1.7e9 + i,
                    "width": 3000,
                    "height": 2000,
                    "photo_date": f"{1920 + (i % 100)}",
                    "embedding_done": True,
                }
            )
        _bulk_insert(conn, "images", image_rows)

        # Faces — embeddings are unit-normalised random vectors stored in the
        # face_blobs side table (matching the production schema).
        n_faces = int(n_images * faces_per_image)
        face_rows = []
        blob_rows = []
        for i in range(n_faces):
            emb = np_rng.standard_normal(EMBED_DIM).astype(np.float32)
            emb /= np.linalg.norm(emb)
            image_id = (i % n_images) + 1
            person_id = rng.randint(1, n_persons) if rng.random() > 0.05 else None
            face_rows.append(
                {
                    "image_id": image_id,
                    "person_id": person_id,
                    "bbox_x": rng.randint(0, 2000),
                    "bbox_y": rng.randint(0, 1500),
                    "bbox_w": rng.randint(80, 400),
                    "bbox_h": rng.randint(80, 400),
                    "confidence": rng.uniform(0.5, 1.0),
                    "detector_backend": "cpu",
                    "crop_path": f"data/crops/{image_id}_{i + 1}.jpg",
                    "is_excluded": False,
                    "is_merge_excluded": False,
                    "auto_merged_from_unknown": False,
                    "auto_merge_confirmed_by_user": False,
                    "is_uncertain_identification": False,
                    "quality_score": rng.uniform(0.2, 1.0),
                }
            )
            blob_rows.append({"face_id": i + 1, "embedding": emb.tobytes()})
            if len(face_rows) >= 5000:
                _bulk_insert(conn, "faces", face_rows)
                _bulk_insert(conn, "face_blobs", blob_rows)
                face_rows = []
                blob_rows = []
        if face_rows:
            _bulk_insert(conn, "faces", face_rows)
            _bulk_insert(conn, "face_blobs", blob_rows)

    elapsed = time.perf_counter() - t0
    return {
        "persons": n_persons,
        "images": n_images,
        "faces": n_faces,
        "seconds": elapsed,
        "db_mb": db_path.stat().st_size / (1024 * 1024),
    }


def _bulk_insert(conn, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    from app.db.models import Base

    # Core insert on the Table applies Python-side column defaults
    # (created_at, boolean flags), unlike a raw text() statement.
    conn.execute(Base.metadata.tables[table].insert(), rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--persons", type=int, default=1000)
    parser.add_argument("--images", type=int, default=10000)
    parser.add_argument("--faces-per-image", type=float, default=2.0)
    args = parser.parse_args()

    stats = seed_database(args.db, args.persons, args.images, args.faces_per_image)
    print(
        f"Seeded {stats['persons']} persons, {stats['images']} images, "
        f"{stats['faces']} faces in {stats['seconds']:.1f}s "
        f"→ {args.db} ({stats['db_mb']:.1f} MB)"
    )


if __name__ == "__main__":
    main()
