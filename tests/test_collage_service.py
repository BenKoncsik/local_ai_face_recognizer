"""Tests for collage import, linking and rendering."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.db.database import init_db, session_scope
from app.db.models import Collage, CollageNode, Face, Image, Person
from app.services.collage_service import CollageService
from tests.test_collage_parser import SAMPLE_XML


@pytest.fixture()
def db(tmp_path):
    init_db(tmp_path / "collage.db")
    return tmp_path


@pytest.fixture()
def collage_file(tmp_path):
    f = tmp_path / "album.cxf"
    f.write_text(SAMPLE_XML, encoding="utf-8")
    (tmp_path / "foto1.JPG").write_bytes(b"FAKE1")
    (tmp_path / "foto2.jpg").write_bytes(b"FAKE2")
    return f


def _add_image(session, path: str, *, width=400, height=300) -> Image:
    image = Image(
        file_path=path,
        file_hash=f"hash_{path}",
        file_mtime=0.0,
        width=width,
        height=height,
    )
    session.add(image)
    session.flush()
    return image


def _add_person(session, name: str) -> Person:
    person = Person(name=name)
    session.add(person)
    session.flush()
    return person


def _add_face(session, image: Image, person: Person) -> Face:
    face = Face(
        image_id=image.id,
        person_id=person.id,
        bbox_x=50,
        bbox_y=40,
        bbox_w=80,
        bbox_h=90,
        confidence=0.9,
        detector_backend="cpu",
    )
    session.add(face)
    session.flush()
    return face


def test_import_collage_persists_nodes(db, collage_file, tmp_path):
    with session_scope() as session:
        collage = CollageService(session).import_collage(collage_file)

    assert collage.id is not None
    assert collage.album_title == "Teszt album"
    assert collage.collage_uid == "516de887be88912fed726867c2bbee6e"
    assert len(collage.nodes) == 2

    with session_scope() as session:
        stored = session.get(Collage, collage.id)
        assert stored is not None
        assert len(stored.nodes) == 2
        assert stored.nodes[0].node_uid == "aabbcc1122334455"


def test_import_collage_returns_existing_without_overwrite(db, collage_file):
    with session_scope() as session:
        first = CollageService(session).import_collage(collage_file)
        first_id = first.id

    with session_scope() as session:
        second = CollageService(session).import_collage(collage_file, overwrite=False)

    assert second.id == first_id


def test_import_collage_overwrite_reimports(db, collage_file):
    with session_scope() as session:
        first = CollageService(session).import_collage(collage_file)
        first_created = first.created_at

    with session_scope() as session:
        second = CollageService(session).import_collage(collage_file, overwrite=True)

    assert len(second.nodes) == 2
    assert second.created_at >= first_created
    with session_scope() as session:
        assert session.query(Collage).count() == 1


def test_import_collage_links_existing_images(db, collage_file, tmp_path):
    foto1 = str(tmp_path / "foto1.JPG")
    foto2 = str(tmp_path / "foto2.jpg")
    with session_scope() as session:
        _add_image(session, foto1)
        _add_image(session, foto2)

    with session_scope() as session:
        collage = CollageService(session).import_collage(collage_file)

    linked = [n for n in collage.nodes if n.image_id is not None]
    assert len(linked) == 2
    assert all(not n.src_missing for n in collage.nodes)


def test_list_and_get_collages(db, collage_file):
    with session_scope() as session:
        imported = CollageService(session).import_collage(collage_file)
        imported_id = imported.id

    with session_scope() as session:
        svc = CollageService(session)
        collages = svc.list_collages()
        assert len(collages) == 1
        assert collages[0].album_title == "Teszt album"
        assert svc.get_collage(imported_id) is not None
        assert svc.get_collage(999_999) is None


def test_get_nodes_and_update_metadata(db, collage_file):
    with session_scope() as session:
        collage = CollageService(session).import_collage(collage_file)
        collage_id = collage.id

    with session_scope() as session:
        svc = CollageService(session)
        nodes = svc.get_nodes(collage_id)
        node_id = nodes[0].id
        updated = svc.update_node_metadata(
            node_id,
            year="1969",
            location="Balaton",
            event_name="Nyaralás",
            notes="teszt",
        )

    assert updated.year == "1969"
    assert updated.location == "Balaton"
    assert updated.event_name == "Nyaralás"
    assert updated.notes == "teszt"


def test_update_node_metadata_missing_raises(db):
    with session_scope() as session:
        with pytest.raises(ValueError, match="CollageNode id=424242 not found"):
            CollageService(session).update_node_metadata(424242, year="1969")


def test_relink_images_after_scan(db, collage_file, tmp_path):
    with session_scope() as session:
        collage = CollageService(session).import_collage(collage_file)
        collage_id = collage.id

    foto1 = str(tmp_path / "foto1.JPG")
    with session_scope() as session:
        _add_image(session, foto1)
        linked = CollageService(session).relink_images(collage_id)

    assert linked >= 1


def test_relink_images_missing_collage_raises(db):
    with session_scope() as session:
        with pytest.raises(ValueError, match="Collage id=424242 not found"):
            CollageService(session).relink_images(424242)


def test_get_faces_for_node_and_projected_faces(db, collage_file, tmp_path):
    foto1 = str(tmp_path / "foto1.JPG")
    with session_scope() as session:
        image = _add_image(session, foto1)
        person = _add_person(session, "Anna")
        _add_face(session, image, person)
        collage = CollageService(session).import_collage(collage_file)
        collage_id = collage.id

    with session_scope() as session:
        svc = CollageService(session)
        collage = svc.get_collage(collage_id)
        node = next(n for n in collage.nodes if n.src_resolved == foto1)
        faces = svc.get_faces_for_node(node)
        assert len(faces) == 1

        projected = svc.projected_faces(collage, render_w=1000, render_h=500)
        assert len(projected) == 1
        assert projected[0]["person_name"] == "Anna"
        assert projected[0]["bbox_collage"] is not None


def test_render_collage_image_returns_canvas(db, collage_file):
    with session_scope() as session:
        collage = CollageService(session).import_collage(collage_file)

    with session_scope() as session:
        collage = session.get(Collage, collage.id)
        canvas = CollageService(session).render_collage_image(collage, render_h=200)

    assert canvas is not None
    assert canvas.shape == (200, int(2858 * 200 / 1000), 3)


def test_render_collage_image_with_source_file(db, collage_file, tmp_path):
    foto1 = str(tmp_path / "foto1.JPG")
    img = np.full((120, 160, 3), (20, 80, 140), dtype=np.uint8)
    cv2.imwrite(foto1, img)
    with session_scope() as session:
        _add_image(session, foto1)
        collage = CollageService(session).import_collage(collage_file)

    with session_scope() as session:
        collage = session.get(Collage, collage.id)
        canvas = CollageService(session).render_collage_image(
            collage, render_h=200, draw_borders=True, draw_faces=False
        )

    assert canvas is not None
    assert canvas.mean() > 40


def test_export_annotated_collage_writes_files(db, collage_file, tmp_path):
    foto1 = str(tmp_path / "foto1.JPG")
    out_dir = tmp_path / "export"
    with session_scope() as session:
        image = _add_image(session, foto1)
        person = _add_person(session, "Béla")
        _add_face(session, image, person)
        collage = CollageService(session).import_collage(collage_file)

    with session_scope() as session:
        collage = session.get(Collage, collage.id)
        jpg_path = CollageService(session).export_annotated_collage(
            collage, out_dir, render_h=200
        )

    assert jpg_path.exists()
    assert jpg_path.suffix == ".jpg"
    cxf_files = list(out_dir.glob("*_annotated.cxf"))
    assert len(cxf_files) == 1
