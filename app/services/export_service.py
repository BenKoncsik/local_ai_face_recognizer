"""Export service.

Exports face images and metadata for a selected person (or all persons).
Output formats:
  * Image folder — copies all face crops (or original images) to a target dir.
  * CSV report — face-level metadata table.
  * JSON report — structured person/face records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person, Relationship
from app.utils.image_utils import load_image_bgr, save_image_bgr

# Image variant long-edge sizes (px) and JPEG quality for the Astro export.
ASTRO_THUMB_EDGE = 320
ASTRO_MEDIUM_EDGE = 1280
ASTRO_THUMB_QUALITY = 70
ASTRO_MEDIUM_QUALITY = 82
ASTRO_PAGE_SIZE = 60

log = logging.getLogger(__name__)


class ExportService:
    """Exports faces and metadata for one or all persons.

    Args:
        session: SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_person_images(
        self,
        person_id: int,
        target_dir: str,
        copy_originals: bool = False,
    ) -> int:
        """Copy face crops (or original images) for *person_id* to *target_dir*.

        Args:
            person_id:       Person to export.
            target_dir:      Destination directory (created if absent).
            copy_originals:  If ``True``, copy the full original image instead
                             of just the face crop thumbnail.

        Returns:
            Number of files copied.
        """
        person = self._session.get(Person, person_id)
        if person is None:
            raise ValueError(f"Person id={person_id} not found")

        dest = Path(target_dir)
        dest.mkdir(parents=True, exist_ok=True)

        faces = self._get_faces(person_id)
        copied = 0

        for face in faces:
            src = self._resolve_source(face, copy_originals)
            if src is None or not src.exists():
                log.debug("Source missing for face %d — skipping", face.id)
                continue

            dst_name = f"face_{face.id}_{src.name}"
            dst = dest / dst_name
            shutil.copy2(src, dst)
            copied += 1

        log.info(
            "Exported %d image(s) for person %r to %s", copied, person.name, dest
        )
        return copied

    def export_csv(
        self,
        target_path: str,
        person_id: Optional[int] = None,
    ) -> Path:
        """Write a CSV report to *target_path*.

        Columns: person_id, person_name, face_id, image_path, bbox_x, bbox_y,
                 bbox_w, bbox_h, confidence, detector_backend, crop_path.

        Args:
            target_path: Destination ``.csv`` file path.
            person_id:   Export only this person.  ``None`` → all persons.

        Returns:
            Path to the written CSV file.
        """
        rows = self._build_rows(person_id)
        out = Path(target_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "person_id", "person_name", "groups", "face_id",
            "image_path", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "confidence", "detector_backend", "crop_path",
        ]

        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        log.info("CSV export: %d row(s) → %s", len(rows), out)
        return out

    def export_json(
        self,
        target_path: str,
        person_id: Optional[int] = None,
    ) -> Path:
        """Write a JSON report to *target_path*.

        Structure::

            [
              {
                "person_id": 1,
                "person_name": "Alice",
                "faces": [
                  {
                    "face_id": 42,
                    "image_path": "/path/to/photo.jpg",
                    "bbox": [x, y, w, h],
                    "confidence": 0.97,
                    "detector_backend": "coral",
                    "crop_path": "/path/to/crop.jpg"
                  },
                  ...
                ]
              },
              ...
            ]
        """
        from app.services.person_group_service import PersonGroupService
        group_svc = PersonGroupService(self._session)
        persons = self._get_persons(person_id)
        records = []

        for person in persons:
            faces = self._get_faces(person.id)
            face_records = []
            for f in faces:
                face_records.append(
                    {
                        "face_id": f.id,
                        "image_path": f.image.file_path if f.image else None,
                        "bbox": [f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h],
                        "confidence": round(f.confidence, 4),
                        "detector_backend": f.detector_backend,
                        "crop_path": f.crop_path,
                    }
                )
            groups = [g.name for g in group_svc.get_person_groups(person.id)]
            records.append(
                {
                    "person_id": person.id,
                    "person_name": person.name,
                    "groups": groups,
                    "faces": face_records,
                }
            )

        out = Path(target_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)

        log.info("JSON export: %d person(s) → %s", len(records), out)
        return out

    # ------------------------------------------------------------------
    # Astro static-site export
    # ------------------------------------------------------------------

    def export_astro(
        self,
        target_dir: str,
        person_id: Optional[int] = None,
        *,
        run_build: bool = True,
        astro_project_dir: Optional[str] = None,
        progress_callback: Optional["Callable[[int, str], None]"] = None,
    ) -> Path:
        """Export a paginated, lazy-loading static site via the Astro project.

        Writes a chunked JSON + multi-size image *bundle* into the Astro
        project, then optionally runs ``npm run build`` and copies the generated
        ``dist/`` into *target_dir*.

        The browser only ever loads one paginated page of pre-baked HTML plus a
        minimal ``search-index.json`` — so the site stays fast with thousands of
        persons/photos. No backend is required after the build.

        Args:
            target_dir: Destination directory for the finished static site.
            person_id:  Restrict to a single person, or ``None`` for everyone.
            run_build:  When ``True``, invoke ``npm run build`` and copy ``dist``
                        to *target_dir*. When ``False``, only the data+image
                        bundle is written (useful for tests / manual builds).
            astro_project_dir: Override the Astro project location (defaults to
                        the repo's ``web/astro``).

        Returns:
            Path to *target_dir* (the finished site) when ``run_build`` is true,
            otherwise the Astro project's ``src/data`` bundle directory.
        """
        project = Path(astro_project_dir) if astro_project_dir else _default_astro_project_dir()
        data_dir = project / "src" / "data"
        images_root = project / "public" / "assets" / "images"
        _reset_dir(data_dir)
        _reset_dir(images_root)
        (data_dir.parent.parent / "data").mkdir(parents=True, exist_ok=True)  # ensure src/data

        def _report(percent: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(percent, message)

        _report(2, "Adatok előkészítése…")
        persons = self._get_persons(person_id)

        # image_path -> [(person, face)]; plus the DB Image for metadata.
        image_faces: Dict[str, List[Tuple[Person, Face]]] = {}
        image_by_path: Dict[str, Image] = {}
        faceless_image_paths: set[str] = set()
        person_face_thumbs: Dict[int, List[str]] = {}

        for person in persons:
            faces = self._get_faces(person.id)
            person_face_thumbs.setdefault(person.id, [])
            for face in faces:
                if face.image:
                    ip = face.image.file_path
                    image_faces.setdefault(ip, []).append((person, face))
                    image_by_path[ip] = face.image
                if face.crop_path and Path(face.crop_path).exists():
                    name = _copy_asset(face.crop_path, images_root / "faces", f"face_{face.id}.jpg")
                    if name:
                        person_face_thumbs[person.id].append(f"/assets/images/faces/{name}")

        if person_id is None:
            faceless_images = (
                self._session.query(Image)
                .filter(~Image.faces.any())
                .order_by(Image.file_path)
                .all()
            )
            for image in faceless_images:
                image_faces.setdefault(image.file_path, [])
                image_by_path[image.file_path] = image
                faceless_image_paths.add(image.file_path)

        # Object occurrences keyed by image id, so the per-image loop (which has
        # the image's pixel dimensions) can convert their coordinates to the same
        # image-relative percentages the face overlay uses. Objects are a domain
        # separate from faces — missing/empty is normal and must not break export.
        objects_by_image = self._build_object_occurrence_map()

        # --- per-image variants + photo records ---
        photos: List[dict] = []
        person_image_ids: Dict[int, List[str]] = {pid: [] for pid in person_face_thumbs}
        map_records: List[dict] = []
        tour_records: List[dict] = []
        # DB image id → exported photo id, so object records (joined on image id)
        # can reference the photos actually written by this export.
        image_id_to_photo_id: Dict[int, str] = {}

        # Resizing every image is the bulk of the work → map it to 5–72%.
        total_images = len(image_faces) or 1
        for idx, (img_path, face_list) in enumerate(image_faces.items()):
            _report(5 + int(67 * idx / total_images),
                    f"Képek feldolgozása ({idx + 1}/{total_images})…")
            img = load_image_bgr(img_path)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]
            photo_id = _astro_image_id(img_path)
            variants = _export_image_variants(img, photo_id, images_root)
            if variants is None:
                continue

            face_records: List[dict] = []
            person_ids: List[int] = []
            persons_in_image: List[str] = []
            has_identified = False
            for person, face in face_list:
                bbox = _face_bbox_for_export(face, img_w, img_h)
                if bbox is None:
                    continue
                face_records.append(
                    {
                        "face_id": face.id,
                        "person_id": person.id,
                        "name": person.name,
                        "bbox": bbox,
                    }
                )
                if person.id not in person_ids:
                    person_ids.append(person.id)
                    person_image_ids.setdefault(person.id, []).append(photo_id)
                _append_unique(persons_in_image, person.name)
                if not person.is_auto_named:
                    has_identified = True

            source = Path(img_path)
            db_image = image_by_path.get(img_path)
            compare = self._build_astro_compare_group(db_image, img_path, images_root)
            pair = self._pair_from_compare(compare)
            object_records: List[dict] = []
            if db_image is not None:
                image_id_to_photo_id[db_image.id] = photo_id
                for occ, obj_name in objects_by_image.get(db_image.id, []):
                    record = _object_record_for_export(occ, obj_name, img_w, img_h)
                    if record is not None:
                        object_records.append(record)
            photos.append(
                {
                    "id": photo_id,
                    "thumb": variants["thumb"],
                    "thumbW": variants["thumbW"],
                    "thumbH": variants["thumbH"],
                    "medium": variants["medium"],
                    "mediumW": variants["mediumW"],
                    "mediumH": variants["mediumH"],
                    "original": variants["original"],
                    "width": img_w,
                    "height": img_h,
                    "fileName": source.name,
                    "folder": source.parent.name,
                    "date": (db_image.photo_date if db_image else "") or "",
                    "faces": face_records,
                    "objects": object_records,
                    "personIds": person_ids,
                    "persons": persons_in_image,
                    "pair": pair,
                    "compare": compare,
                    "hasGps": db_image is not None and _effective_image_gps(db_image) is not None,
                }
            )

            # Reuse the existing map/slideshow record builders for parity views.
            # The legacy map.html/slideshow.html live at the dist root, so their
            # image paths must be root-relative *without* a leading slash (the
            # Astro relative-link integration does not touch public/ passthrough
            # files). _no_slash() converts the Astro "/assets/..." paths.
            map_record = _build_map_export_record(db_image, photo_id, source, face_records)
            if map_record is not None:
                map_record["thumbnailPath"] = _no_slash(variants["thumb"])
                map_record["relativePath"] = _no_slash(variants["medium"])
                map_record["detailPage"] = f"photos/{photo_id}/index.html"
                map_records.append(map_record)
            tour = _build_tour_export_record(
                db_image, photo_id, source, face_records,
                persons_in_image, has_identified, img_w, img_h, pair,
                objects=object_records,
            )
            tour["imagePath"] = _no_slash(variants["medium"])
            tour["thumbnailPath"] = _no_slash(variants["thumb"])
            if pair:
                tour["pair"] = {"bw": _no_slash(pair["bw"]), "color": _no_slash(pair["color"])}
            if compare:
                # Whole comparison group for the slideshow picker (root-relative,
                # no leading slash like the other slideshow.html image paths).
                tour["compare"] = [
                    {"label": m["label"], "src": _no_slash(m["src"]), "isBw": m["isBw"]}
                    for m in compare
                ]
            tour_records.append(tour)

        _report(74, "Személyek feldolgozása…")
        # --- person records ---
        relationships = _build_person_relationship_map(self._session, persons)
        person_records: List[dict] = []
        for person in persons:
            thumb_path, tw, th = _export_person_thumb(person, images_root)
            birth_year = _extract_year(person.birth_date)
            death_year = _extract_year(person.death_date)
            from app.services.person_group_service import PersonGroupService
            groups = [g.name for g in PersonGroupService(self._session).get_person_groups(person.id)]
            person_records.append(
                {
                    "id": person.id,
                    "name": person.name,
                    "isAutoNamed": bool(person.is_auto_named),
                    "thumb": thumb_path,
                    "thumbW": tw,
                    "thumbH": th,
                    "imageCount": len(person_image_ids.get(person.id, [])),
                    "imageIds": person_image_ids.get(person.id, []),
                    "faceThumbs": person_face_thumbs.get(person.id, [])[:12],
                    "gender": person.gender or "",
                    "nickname": person.nickname or "",
                    "marriedName": person.married_name or "",
                    "familyCode": person.family_code or "",
                    "externalFamilyCode": person.external_family_code or "",
                    "birthDate": person.birth_date or "",
                    "birthPlace": person.birth_place or "",
                    "deathDate": person.death_date or "",
                    "deathPlace": person.death_place or "",
                    "birthYear": birth_year,
                    "deathYear": death_year,
                    "notes": person.notes or "",
                    "groups": groups,
                    "relationships": relationships.get(person.id, []),
                }
            )

        _report(78, "Objektumok feldolgozása…")
        # --- object records (collection page, mirrors person records) ---
        object_records = self._build_object_export_records(
            images_root, image_id_to_photo_id
        )

        # --- search index (minimal) ---
        search_index: List[dict] = []
        for p in person_records:
            search_index.append({
                "id": p["id"], "name": p["name"], "type": "person",
                "birthYear": p["birthYear"], "deathYear": p["deathYear"],
                "thumb": p["thumb"], "url": f"persons/{p['id']}/",
            })
        for ph in photos:
            label = ph["fileName"]
            search_index.append({
                "id": ph["id"], "name": label, "type": "photo",
                "thumb": ph["thumb"], "url": f"photos/{ph['id']}/",
            })
        for o in object_records:
            search_index.append({
                "id": o["id"], "name": o["name"], "type": "object",
                "thumb": o["thumb"], "url": f"objects/{o['id']}/",
            })

        # --- write bundle ---
        # Slideshow person panel data, keyed by stringified id (built from the
        # records we already produced — avoids a second thumbnail copy).
        person_details = {
            str(p["id"]): {
                "id": p["id"], "name": p["name"], "isAutoNamed": p["isAutoNamed"],
                "gender": p["gender"], "familyCode": p["familyCode"],
                "externalFamilyCode": p["externalFamilyCode"], "nickname": p["nickname"],
                "marriedName": p["marriedName"], "birthDate": p["birthDate"],
                "birthPlace": p["birthPlace"], "deathDate": p["deathDate"],
                "deathPlace": p["deathPlace"], "notes": p["notes"],
                "groups": p["groups"], "thumb": _no_slash(p["thumb"]),
            }
            for p in person_records
        }
        # Slideshow object panel data, keyed by stringified id (mirrors persons).
        object_details = {
            str(o["id"]): {
                "id": o["id"], "name": o["name"], "description": o["description"],
                "notes": o["notes"], "persons": o["persons"],
                "imageCount": o["imageCount"], "thumb": _no_slash(o["thumb"]),
            }
            for o in object_records
        }

        _report(80, "Galéria adatok kiírása…")
        # Parity views (map / slideshow) reuse the existing, tested standalone
        # generators, written into public/ so Astro copies them to the dist root.
        public_dir = project / "public"
        has_collages = False
        if map_records:
            _write_map_export_files(public_dir, map_records)
        if tour_records:
            _write_tour_export_files(
                public_dir, tour_records, person_details, object_details
            )
        try:
            if self._has_collages():
                self.export_collage_html(str(public_dir))
                has_collages = True
        except Exception as exc:  # noqa: BLE001 — collages are optional parity
            log.warning("Collage export skipped: %s", exc)

        manifest = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "pageSize": ASTRO_PAGE_SIZE,
            "personCount": len(person_records),
            "photoCount": len(photos),
            "objectCount": len(object_records),
            "hasMap": bool(map_records),
            "hasSlideshow": bool(tour_records),
            "hasObjects": bool(object_records),
            "hasCollages": has_collages,
            "title": "Face Gallery",
        }
        _write_json(data_dir / "manifest.json", manifest)
        _write_json(data_dir / "persons.json", person_records)
        _write_json(data_dir / "objects.json", object_records)
        _write_json(data_dir / "photos.json", photos)
        _write_json(data_dir / "map-data.json", map_records)
        # The search index is consumed by the browser at runtime (not at build
        # time), so it ships under public/ → dist/assets/data/.
        _write_json(public_dir / "assets" / "data" / "search-index.json", search_index)
        _write_json(data_dir / "slideshow-data.json",
                    {"records": tour_records, "persons": person_details,
                     "objects": object_details})

        log.info("Astro bundle: %d person(s), %d object(s), %d photo(s) → %s",
                 len(person_records), len(object_records), len(photos), data_dir)

        if not run_build:
            _report(100, "Kész.")
            return data_dir

        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)
        # npm build is one opaque step → report it as indeterminate (-1).
        _report(-1, "Weboldal építése (npm)…")
        _run_astro_build(project)
        dist = project / "dist"
        if not dist.is_dir():
            raise RuntimeError(f"Astro build produced no dist/ at {dist}")
        _report(95, "Fájlok másolása a célmappába…")
        _copy_tree_into(dist, out)
        log.info("Astro export → %s", out)
        _report(100, "Kész.")
        return out

    def _has_collages(self) -> bool:
        """True when at least one collage exists (parity collage export)."""
        from app.services.collage_service import CollageService
        return bool(CollageService(self._session).list_collages())

    def _build_object_occurrence_map(
        self,
    ) -> Dict[int, List[Tuple["ObjectOccurrence", str]]]:
        """Return ``image_id → [(occurrence, object_name)]`` for every image.

        Tagged objects are an optional, manually-curated domain that is wholly
        separate from face recognition.  Old databases predate the object tables,
        so any failure here (missing tables, no rows) degrades to an empty map —
        the Astro export simply emits no object overlays.
        """
        from app.db.models import ObjectOccurrence, TaggedObject

        result: Dict[int, List[Tuple["ObjectOccurrence", str]]] = {}
        try:
            rows = (
                self._session.query(ObjectOccurrence, TaggedObject.name)
                .join(TaggedObject, TaggedObject.id == ObjectOccurrence.object_id)
                .order_by(ObjectOccurrence.id)
                .all()
            )
        except Exception as exc:  # noqa: BLE001 — objects are optional metadata
            log.warning("Object occurrences unavailable for export: %s", exc)
            return result
        for occ, name in rows:
            result.setdefault(occ.image_id, []).append((occ, name))
        return result

    def _build_object_export_records(
        self,
        images_root: Path,
        image_id_to_photo_id: Dict[int, str],
    ) -> List[dict]:
        """Build the per-object collection records (mirrors ``person_records``).

        Each record carries the object's identity (name/description/notes), an
        exported thumbnail, aggregate counts, the linked persons (with role
        labels) and the exported photo ids it appears in.  Tagged objects are an
        optional domain — any failure degrades to an empty list so the export
        still succeeds on databases without the object tables.
        """
        try:
            from app.services.object_service import ObjectService
        except Exception as exc:  # noqa: BLE001 — objects are optional
            log.warning("Object export unavailable: %s", exc)
            return []

        svc = ObjectService(self._session)
        try:
            summaries = svc.list_objects()
            thumb_specs = svc.get_thumbnail_specs()
        except Exception as exc:  # noqa: BLE001 — old DB without object tables
            log.warning("Object export skipped: %s", exc)
            return []

        records: List[dict] = []
        for summ in summaries:
            oid = summ.object_id
            thumb_path, tw, th = _export_object_thumb(
                thumb_specs.get(oid), oid, images_root
            )
            persons = []
            for link in svc.get_object_persons(oid):
                persons.append({
                    "id": link.person_id,
                    "name": link.name,
                    "role": link.role,
                    "roleLabel": _OBJECT_ROLE_LABELS_HU.get(link.role, link.role),
                    "note": link.note or "",
                })
            # Photo ids this object appears in (only those actually exported).
            image_ids: List[str] = []
            for occ in svc.get_occurrences(oid):
                pid = image_id_to_photo_id.get(occ.image_id)
                if pid is not None and pid not in image_ids:
                    image_ids.append(pid)
            records.append({
                "id": oid,
                "name": summ.name or "",
                "description": summ.description or "",
                "notes": summ.notes or "",
                "thumb": thumb_path,
                "thumbW": tw,
                "thumbH": th,
                "imageCount": len(image_ids),
                "imageIds": image_ids,
                "occurrenceCount": summ.occurrence_count,
                "noteCount": summ.note_count,
                "persons": persons,
            })
        return records

    def _build_astro_compare_group(
        self,
        db_image: Optional[Image],
        img_path: str,
        images_root: Path,
    ) -> Optional[List[dict]]:
        """Export medium variants of the whole comparison group, if present.

        Reuses :class:`DeoldifiedPairingService.get_comparison_group`
        (filename-based, folder independent): the B&W original plus every
        colorized variant.  Each member is exported as a medium-size variant and
        returned as ``{"label", "src", "isBw"}``.  Returns None when fewer than
        two members exist, so the photo simply has no compare UI.
        """
        from app.services.deoldified_pairing_service import DeoldifiedPairingService

        if db_image is None:
            return None
        group = DeoldifiedPairingService(self._session).get_comparison_group(db_image)
        if len(group) < 2:
            return None

        members: List[dict] = []
        for m in group:
            med = _export_one_variant(
                load_image_bgr(m.file_path), _astro_image_id(m.file_path),
                images_root, "medium", ASTRO_MEDIUM_EDGE, ASTRO_MEDIUM_QUALITY)
            if med is None:
                continue
            members.append({"label": m.label, "src": med[0], "isBw": m.is_bw})
        # Need both a B&W side and at least one colorized side to compare.
        if len(members) < 2 or not any(x["isBw"] for x in members):
            return None
        return members

    @staticmethod
    def _pair_from_compare(compare: Optional[List[dict]]) -> Optional[dict]:
        """Derive the legacy ``{bw, color}`` pair from a comparison group.

        Used by the slideshow/tour views, which keep the simple two-image
        before/after slider.  ``bw`` is the original, ``color`` the first
        colorized variant.
        """
        if not compare:
            return None
        bw = next((m["src"] for m in compare if m["isBw"]), None)
        color = next((m["src"] for m in compare if not m["isBw"]), None)
        if bw is None or color is None:
            return None
        return {"bw": bw, "color": color}

    # ------------------------------------------------------------------
    # Collage HTML export
    # ------------------------------------------------------------------

    def export_collage_html(
        self,
        target_dir: str,
        collage_id: Optional[int] = None,
    ) -> Path:
        """Generate a static HTML page for one or all collages.

        The page renders each collage as a full-width image with:
        * SVG overlay showing node boundaries,
        * hover tooltip (desktop) / tap panel (mobile) per node,
        * person names shown on face bounding boxes,
        * full-text search by person name.

        Args:
            target_dir:  Output directory (created if absent).
            collage_id:  Export only this collage.  ``None`` → all collages.

        Returns:
            Path to the generated ``collage_index.html``.
        """
        from app.services.collage_service import CollageService

        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)
        img_dir = out / "collage_images"
        img_dir.mkdir(exist_ok=True)

        svc = CollageService(self._session)

        if collage_id is not None:
            c = svc.get_collage(collage_id)
            collages = [c] if c else []
        else:
            collages = svc.list_collages()

        collage_records = []
        render_h = 800

        for collage in collages:
            cw = collage.format_width or 2858
            ch = collage.format_height or 1000
            scale = render_h / ch
            render_w = int(cw * scale)

            # Render the collage image
            canvas = svc.render_collage_image(
                collage, render_h=render_h, draw_borders=False, draw_faces=False
            )
            safe = _safe_filename(collage.album_title or f"collage_{collage.id}")
            img_name = f"{safe}_{collage.id}.jpg"
            if canvas is not None:
                save_image_bgr(img_dir / img_name, canvas)
            else:
                img_name = ""

            # Build node data with face projections
            from app.services.collage_parser import (
                CollageNodeData,
                project_face_to_collage,
            )
            nodes_json = []
            for node in collage.nodes:
                nd = CollageNodeData(
                    rel_x=node.rel_x, rel_y=node.rel_y,
                    rel_w=node.rel_w, rel_h=node.rel_h,
                    theta=node.theta, scale=node.scale,
                )
                px = int(node.rel_x * render_w)
                py = int(node.rel_y * render_h)
                pw = max(int(node.rel_w * render_w), 1)
                ph = max(int(node.rel_h * render_h), 1)

                from pathlib import Path as _P
                src_name = (
                    _P(node.src_raw.replace("\\", "/")).name
                    if node.src_raw else ""
                )

                face_rects = []
                if node.image_id:
                    image = self._session.get(Image, node.image_id)
                    if image and image.width and image.height:
                        for face in (
                            self._session.query(Face)
                            .filter(Face.image_id == node.image_id)
                            .all()
                        ):
                            bbox = project_face_to_collage(
                                (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                                image.width, image.height,
                                nd, render_w, render_h,
                            )
                            if bbox:
                                person = self._session.get(Person, face.person_id) if face.person_id else None
                                face_rects.append({
                                    "x": bbox[0], "y": bbox[1],
                                    "w": bbox[2], "h": bbox[3],
                                    "name": person.name if person else "",
                                    "notes": person.notes if person else "",
                                })

                year = node.year or ""
                location = node.location or ""
                event_name = node.event_name or ""
                notes = node.notes or ""

                nodes_json.append({
                    "x": px, "y": py, "w": pw, "h": ph,
                    "src": src_name,
                    "uid": node.node_uid or "",
                    "missing": node.src_missing,
                    "year": year,
                    "location": location,
                    "event": event_name,
                    "notes": notes,
                    "faces": face_rects,
                })

            collage_records.append({
                "id": collage.id,
                "title": collage.album_title or f"Kollázs #{collage.id}",
                "date": collage.album_date or "",
                "img": f"collage_images/{img_name}" if img_name else "",
                "width": render_w,
                "height": render_h,
                "nodes": nodes_json,
            })

        js_data = _json_for_script(collage_records, indent=1)
        html_out = _COLLAGE_HTML_TEMPLATE.replace("__COLLAGES_JSON__", js_data)
        html_path = out / "collage_index.html"
        html_path.write_text(html_out, encoding="utf-8")

        log.info("Collage HTML export: %d collage(s) → %s", len(collages), html_path)
        return html_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_persons(self, person_id: Optional[int]) -> List[Person]:
        if person_id is not None:
            p = self._session.get(Person, person_id)
            return [p] if p else []
        return self._session.query(Person).order_by(Person.name).all()

    def _get_faces(self, person_id: int) -> List[Face]:
        return (
            self._session.query(Face)
            .filter(Face.person_id == person_id)
            .all()
        )

    @staticmethod
    def _resolve_source(face: Face, copy_originals: bool) -> Optional[Path]:
        if copy_originals and face.image:
            return Path(face.image.file_path)
        if face.crop_path:
            return Path(face.crop_path)
        return None

    def _build_rows(self, person_id: Optional[int]) -> List[dict]:
        from app.services.person_group_service import PersonGroupService
        svc = PersonGroupService(self._session)
        persons = self._get_persons(person_id)
        rows = []
        for person in persons:
            groups_csv = ", ".join(
                g.name for g in svc.get_person_groups(person.id)
            )
            for face in self._get_faces(person.id):
                rows.append(
                    {
                        "person_id": person.id,
                        "person_name": person.name,
                        "groups": groups_csv,
                        "face_id": face.id,
                        "image_path": face.image.file_path if face.image else "",
                        "bbox_x": face.bbox_x,
                        "bbox_y": face.bbox_y,
                        "bbox_w": face.bbox_w,
                        "bbox_h": face.bbox_h,
                        "confidence": round(face.confidence, 4),
                        "detector_backend": face.detector_backend,
                        "crop_path": face.crop_path or "",
                    }
                )
        return rows


def _image_export_filename(image_path: str) -> str:
    """Return a stable browser-friendly filename for an exported source image."""
    digest = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:16]
    return f"img_{digest}.jpg"


def _face_bbox_for_export(
    face: Face,
    image_w: int,
    image_h: int,
) -> Optional[Dict[str, float]]:
    """Normalize, clamp, and export a face bbox as image-relative percentages.

    The database schema stores face boxes in original image pixels. The
    normalized-coordinate branch is defensive for old/manual imports that may
    have stored 0..1 values. The generated HTML uses these percentages
    directly as CSS positions, keeping overlays responsive without rewriting
    the image.
    """
    if image_w <= 0 or image_h <= 0:
        return None

    x = float(face.bbox_x)
    y = float(face.bbox_y)
    w = float(face.bbox_w)
    h = float(face.bbox_h)

    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
        x *= image_w
        w *= image_w
        y *= image_h
        h *= image_h

    x1 = max(0.0, min(x, float(image_w)))
    y1 = max(0.0, min(y, float(image_h)))
    x2 = max(x1, min(x + w, float(image_w)))
    y2 = max(y1, min(y + h, float(image_h)))
    out_w = x2 - x1
    out_h = y2 - y1
    if out_w <= 0 or out_h <= 0:
        return None

    return {
        "left": _percent(x1, image_w),
        "top": _percent(y1, image_h),
        "width": _percent(out_w, image_w),
        "height": _percent(out_h, image_h),
    }


def _percent(value: float, total: int) -> float:
    return round((value / float(total)) * 100.0, 6)


def _object_record_for_export(
    occ: "ObjectOccurrence",
    name: str,
    image_w: int,
    image_h: int,
) -> Optional[dict]:
    """Build one export record for a tagged-object occurrence.

    Coordinates are emitted in the *same* image-relative percentage space the
    face overlay uses (see :func:`_face_bbox_for_export`), so the Astro frontend
    can position face and object overlays with identical CSS maths.

    The record always carries the object's id/name/note so it can be listed even
    when geometry is absent.  ``bbox`` is populated when the occurrence has a
    bounding box, otherwise ``point`` is populated when it has a click point;
    both are ``None`` when no geometry was recorded.
    """
    record: dict = {
        "object_id": occ.object_id,
        "name": name or "",
        "confidence": (
            round(occ.confidence, 4) if occ.confidence is not None else None
        ),
        "note": occ.note or "",
        "bbox": None,
        "point": None,
    }

    if image_w <= 0 or image_h <= 0:
        return record

    has_bbox = (
        occ.bbox_x is not None
        and occ.bbox_y is not None
        and occ.bbox_w is not None
        and occ.bbox_h is not None
        and occ.bbox_w > 0
        and occ.bbox_h > 0
    )
    if has_bbox:
        x1 = max(0.0, min(float(occ.bbox_x), float(image_w)))
        y1 = max(0.0, min(float(occ.bbox_y), float(image_h)))
        x2 = max(x1, min(float(occ.bbox_x) + float(occ.bbox_w), float(image_w)))
        y2 = max(y1, min(float(occ.bbox_y) + float(occ.bbox_h), float(image_h)))
        if x2 > x1 and y2 > y1:
            record["bbox"] = {
                "left": _percent(x1, image_w),
                "top": _percent(y1, image_h),
                "width": _percent(x2 - x1, image_w),
                "height": _percent(y2 - y1, image_h),
            }
    elif occ.point_x is not None and occ.point_y is not None:
        px = max(0.0, min(float(occ.point_x), float(image_w)))
        py = max(0.0, min(float(occ.point_y), float(image_h)))
        record["point"] = {
            "left": _percent(px, image_w),
            "top": _percent(py, image_h),
        }

    return record


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _build_map_export_record(
    image: Optional[Image],
    exported_name: str,
    source_path: Path,
    face_records: List[dict],
) -> Optional[dict]:
    if image is None:
        return None
    coords = _effective_image_gps(image)
    if coords is None:
        return None
    latitude, longitude = coords
    people = sorted({str(face.get("name", "")) for face in face_records if face.get("name")})
    place_name = image.place.name if image.place else ""
    folder = source_path.parent.name
    return {
        "imageId": image.id,
        "fileName": source_path.name,
        "relativePath": f"images/{exported_name}",
        "thumbnailPath": f"images/{exported_name}",
        "detailPage": f"index.html#image={exported_name}",
        "latitude": round(latitude, 8),
        "longitude": round(longitude, 8),
        "dateTaken": image.photo_date or "",
        "people": people,
        "placeName": place_name or "",
        "folder": folder,
    }


def _effective_image_gps(image: Image) -> Optional[Tuple[float, float]]:
    candidates = (
        (image.image_latitude, image.image_longitude),
        (image.exif_latitude, image.exif_longitude),
        (
            image.place.latitude if image.place else None,
            image.place.longitude if image.place else None,
        ),
    )
    for latitude, longitude in candidates:
        if _valid_gps_pair(latitude, longitude):
            return float(latitude), float(longitude)
    return None


def _valid_gps_pair(latitude: Optional[float], longitude: Optional[float]) -> bool:
    if latitude is None or longitude is None:
        return False
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(lat) or not math.isfinite(lon):
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def _write_map_export_files(out: Path, records: List[dict]) -> None:
    data_json = json.dumps(records, ensure_ascii=False, indent=2)
    (out / "map-data.json").write_text(data_json + "\n", encoding="utf-8")
    (out / "map-data.js").write_text(
        "window.MAP_EXPORT_DATA = " + _json_for_script(records, indent=2) + ";\n",
        encoding="utf-8",
    )
    (out / "map.html").write_text(_MAP_HTML_TEMPLATE, encoding="utf-8")
    (out / "map.css").write_text(_MAP_CSS, encoding="utf-8")
    (out / "map.js").write_text(_MAP_JS, encoding="utf-8")


_YEAR_RE = re.compile(r"(\d{4})")


def _extract_year(date_str: Optional[str]) -> str:
    """Pull a 4-digit year out of a free-text date string, or '' if none."""
    if not date_str:
        return ""
    match = _YEAR_RE.search(str(date_str))
    return match.group(1) if match else ""


def _decade_label(year: str) -> str:
    """Return a Hungarian decade label like '1980-as évek' for a year, or ''."""
    if not year or len(year) < 4:
        return ""
    return f"{year[:3]}0-as évek"


def _build_tour_export_record(
    image: Optional[Image],
    exported_name: str,
    source: Path,
    face_records: List[dict],
    persons: List[str],
    has_identified: bool,
    width: int,
    height: int,
    pair: Optional[dict] = None,
    objects: Optional[List[dict]] = None,
) -> dict:
    """Build one rich per-image record for the slideshow / image-tour page.

    Every field has a defined fallback so the front-end never has to guard
    against a missing key — absent data is represented as ``""``/``None``/
    ``[]``/``False`` rather than being omitted.
    """
    coords = _effective_image_gps(image) if image is not None else None
    latitude, longitude = coords if coords is not None else (None, None)

    date = (image.photo_date if image is not None else "") or ""
    year = _extract_year(date)
    place_name = image.place.name if (image is not None and image.place) else ""
    note = (image.note if image is not None else "") or ""

    return {
        "id": exported_name,
        "imagePath": f"images/{exported_name}",
        "thumbnailPath": f"images/{exported_name}",
        "fileName": source.name,
        "folder": source.parent.name,
        "title": source.stem,
        "caption": note,
        "description": note,
        "date": date,
        "year": year,
        "decade": _decade_label(year),
        "locationName": place_name,
        "city": place_name,
        "gpsLatitude": round(latitude, 8) if latitude is not None else None,
        "gpsLongitude": round(longitude, 8) if longitude is not None else None,
        "persons": persons,
        "faces": face_records,
        "objects": objects or [],
        "width": width,
        "height": height,
        "isFavorite": False,
        "hasCaption": bool(note),
        "hasIdentifiedPersons": has_identified,
        "hasLocation": latitude is not None and longitude is not None,
        "pair": pair,
    }


def _write_tour_export_files(
    out: Path,
    records: List[dict],
    persons: Optional[Dict[str, dict]] = None,
    objects: Optional[Dict[str, dict]] = None,
) -> None:
    persons = persons or {}
    objects = objects or {}
    data_json = json.dumps(records, ensure_ascii=False, indent=2)
    (out / "slideshow-data.json").write_text(data_json + "\n", encoding="utf-8")
    (out / "slideshow-data.js").write_text(
        "window.TOUR_DATA = " + _json_for_script(records, indent=2) + ";\n"
        + "window.TOUR_PERSONS = " + _json_for_script(persons, indent=2) + ";\n"
        + "window.TOUR_OBJECTS = " + _json_for_script(objects, indent=2) + ";\n",
        encoding="utf-8",
    )
    (out / "slideshow.html").write_text(_TOUR_HTML_TEMPLATE, encoding="utf-8")
    (out / "slideshow.css").write_text(_TOUR_CSS, encoding="utf-8")
    (out / "slideshow.js").write_text(_TOUR_JS, encoding="utf-8")


def _json_for_script(value, *, indent: Optional[int] = None) -> str:
    """Serialize JSON safely for an inline script tag."""
    return (
        json.dumps(value, ensure_ascii=False, indent=indent)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


# ---------------------------------------------------------------------------
# Astro export helpers
# ---------------------------------------------------------------------------


def _default_astro_project_dir() -> Path:
    """Return the repo's bundled Astro project (``web/astro``)."""
    return Path(__file__).resolve().parents[2] / "web" / "astro"


def _reset_dir(path: Path) -> None:
    """Remove *path* if present and recreate it empty."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _astro_image_id(image_path: str) -> str:
    """Stable, browser-safe id for an exported image (no extension)."""
    return _image_export_filename(image_path).rsplit(".", 1)[0]


def _no_slash(path: Optional[str]) -> Optional[str]:
    """Strip a leading '/' so a root-relative path works from a dist-root file.

    The Astro pages use "/assets/..." (rewritten to page-relative by the
    relative-links integration), but the standalone map/slideshow pages dropped
    into public/ are not processed by that integration — they need plain
    root-relative paths like "assets/...".
    """
    if isinstance(path, str) and path.startswith("/"):
        return path[1:]
    return path


def _resize_long_edge(img, max_edge: int):
    """Return *img* scaled so its longest edge is at most *max_edge* px."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_edge or longest == 0:
        return img
    scale = max_edge / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _export_one_variant(
    img,
    image_id: str,
    images_root: Path,
    subdir: str,
    max_edge: Optional[int],
    quality: int,
) -> Optional[Tuple[str, int, int]]:
    """Resize+save one image variant; return (web_path, width, height) or None.

    ``max_edge`` of ``None`` saves the image at full resolution (the original).
    The returned web path is root-absolute (``/assets/images/...``) so the Astro
    relative-link integration can rewrite it for file:// use.
    """
    if img is None:
        return None
    variant = img if max_edge is None else _resize_long_edge(img, max_edge)
    h, w = variant.shape[:2]
    out_dir = images_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    if not save_image_bgr(out_dir / f"{image_id}.jpg", variant, params):
        return None
    return f"/assets/images/{subdir}/{image_id}.jpg", w, h


def _export_image_variants(img, image_id: str, images_root: Path) -> Optional[dict]:
    """Export thumb + medium + original variants of a loaded BGR image."""
    thumb = _export_one_variant(img, image_id, images_root, "thumbs",
                                ASTRO_THUMB_EDGE, ASTRO_THUMB_QUALITY)
    medium = _export_one_variant(img, image_id, images_root, "medium",
                                 ASTRO_MEDIUM_EDGE, ASTRO_MEDIUM_QUALITY)
    original = _export_one_variant(img, image_id, images_root, "original", None, 92)
    if thumb is None or medium is None or original is None:
        return None
    return {
        "thumb": thumb[0], "thumbW": thumb[1], "thumbH": thumb[2],
        "medium": medium[0], "mediumW": medium[1], "mediumH": medium[2],
        "original": original[0],
    }


def _export_person_thumb(person: Person, images_root: Path) -> Tuple[Optional[str], int, int]:
    """Export a person's representative thumbnail; (web_path, w, h) or (None,0,0)."""
    path = person.thumbnail_path
    if not path or not Path(path).exists():
        return None, 0, 0
    img = load_image_bgr(path)
    if img is None:
        return None, 0, 0
    result = _export_one_variant(
        img, f"person_{person.id}", images_root, "persons",
        ASTRO_THUMB_EDGE, ASTRO_THUMB_QUALITY)
    if result is None:
        return None, 0, 0
    return result


# Hungarian labels for object↔person roles (mirrors app.ui.i18n object_role_*),
# defined locally so the service layer stays independent of the Qt UI module.
_OBJECT_ROLE_LABELS_HU = {
    "owner": "Tulajdonos",
    "former_owner": "Korábbi tulajdonos",
    "driver": "Sofőr",
    "creator": "Készítő",
    "user": "Használó",
    "family": "Családtag",
    "other": "Egyéb",
}


def _export_object_thumb(
    spec: Optional[Tuple],
    object_id: int,
    images_root: Path,
) -> Tuple[Optional[str], int, int]:
    """Export a tagged object's thumbnail; (web_path, w, h) or (None, 0, 0).

    *spec* is ``(image_path, bbox_or_None)`` as returned by
    :meth:`ObjectService.get_thumbnail_specs` — a manual crop file (bbox None)
    or a source image to crop to the occurrence's bounding box.
    """
    if not spec:
        return None, 0, 0
    path, bbox = spec
    if not path or not Path(path).exists():
        return None, 0, 0
    img = load_image_bgr(path)
    if img is None:
        return None, 0, 0
    if bbox is not None:
        h, w = img.shape[:2]
        bx, by, bw, bh = bbox
        x1 = max(0, min(int(bx), w))
        y1 = max(0, min(int(by), h))
        x2 = max(x1, min(int(bx) + int(bw), w))
        y2 = max(y1, min(int(by) + int(bh), h))
        if x2 > x1 and y2 > y1:
            img = img[y1:y2, x1:x2]
    result = _export_one_variant(
        img, f"object_{object_id}", images_root, "objects",
        ASTRO_THUMB_EDGE, ASTRO_THUMB_QUALITY)
    if result is None:
        return None, 0, 0
    return result


def _copy_asset(src: str, dst_dir: Path, name: str) -> Optional[str]:
    """Copy a small asset (e.g. a face crop) verbatim; return its filename."""
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / name)
        return name
    except OSError:
        return None


def _build_person_relationship_map(
    session: Session,
    persons: List[Person],
) -> Dict[int, List[dict]]:
    """Map each person id \u2192 [{relatedPersonId, relatedName, label}].

    Uses the stored ``Relationship`` rows (ParentChild / Spouse). Sibling links
    are intentionally not stored in the schema and are omitted here.
    """
    ids = [p.id for p in persons]
    if not ids:
        return {}
    rows = (
        session.query(Relationship)
        .filter(
            (Relationship.person_a_id.in_(ids)) | (Relationship.person_b_id.in_(ids))
        )
        .all()
    )
    name_cache: Dict[int, str] = {p.id: p.name for p in persons}

    def name_of(pid: int) -> str:
        if pid not in name_cache:
            other = session.get(Person, pid)
            name_cache[pid] = other.name if other else f"#{pid}"
        return name_cache[pid]

    result: Dict[int, List[dict]] = {pid: [] for pid in ids}
    id_set = set(ids)
    for r in rows:
        rtype = (r.relationship_type or "").lower()
        if rtype == "parentchild":
            if r.person_a_id in id_set:  # a is the parent
                result[r.person_a_id].append(
                    {"relatedPersonId": r.person_b_id, "relatedName": name_of(r.person_b_id), "label": "Gyermek"})
            if r.person_b_id in id_set:  # b is the child
                result[r.person_b_id].append(
                    {"relatedPersonId": r.person_a_id, "relatedName": name_of(r.person_a_id), "label": "Sz\u00fcl\u0151"})
        elif rtype == "spouse":
            if r.person_a_id in id_set:
                result[r.person_a_id].append(
                    {"relatedPersonId": r.person_b_id, "relatedName": name_of(r.person_b_id), "label": "H\u00e1zast\u00e1rs"})
            if r.person_b_id in id_set:
                result[r.person_b_id].append(
                    {"relatedPersonId": r.person_a_id, "relatedName": name_of(r.person_a_id), "label": "H\u00e1zast\u00e1rs"})
    return result


def _write_json(path: Path, data) -> None:
    """Write compact UTF-8 JSON (no ASCII escaping, minimal separators)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _npm_command() -> str:
    """Return the npm executable name for the current platform."""
    return "npm.cmd" if sys.platform.startswith("win") else "npm"


def _run_astro_build(project: Path) -> None:
    """Install deps if needed and run ``npm run build`` inside *project*."""
    npm = _npm_command()
    if shutil.which(npm) is None:
        raise RuntimeError(
            "Node.js/npm not found on PATH \u2014 the Astro export requires Node "
            "to build. Install Node.js (https://nodejs.org) and try again."
        )
    if not (project / "node_modules").is_dir():
        log.info("Installing Astro dependencies (npm install) \u2026")
        subprocess.run([npm, "install"], cwd=str(project), check=True)
    log.info("Building Astro site (npm run build) \u2026")
    subprocess.run([npm, "run", "build"], cwd=str(project), check=True)


def _copy_tree_into(src: Path, dst: Path) -> None:
    """Copy the contents of *src* into *dst* (merging into an existing dir)."""
    shutil.copytree(src, dst, dirs_exist_ok=True)


_MAP_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Gallery - Térkép</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">
<link rel="stylesheet" href="map.css">
</head>
<body>
<header>
  <h1>Face Gallery</h1>
  <nav>
    <a href="index.html">Galéria</a>
    <a href="slideshow.html">Diavetítés</a>
    <a aria-current="page" href="map.html">Térkép</a>
  </nav>
  <span id="map-count"></span>
</header>
<main>
  <aside id="filters" aria-label="Térkép szűrők">
    <label>Személy
      <select id="person-filter"><option value="">Minden személy</option></select>
    </label>
    <label>Év
      <select id="year-filter"><option value="">Minden év</option></select>
    </label>
    <label>Mappa
      <select id="folder-filter"><option value="">Minden mappa</option></select>
    </label>
    <div id="status"></div>
    <div id="visible-list"></div>
  </aside>
  <section id="map-wrap">
    <div id="map" role="region" aria-label="GPS térkép"></div>
    <div id="fallback-list"></div>
  </section>
</main>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
<script src="map-data.js"></script>
<script src="map.js"></script>
</body>
</html>
"""


_MAP_CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:#111;color:#ddd;font-family:system-ui,sans-serif}
button,input,select{font:inherit}
header{background:#1a1a1a;padding:14px 20px;border-bottom:1px solid #333;
       display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:1.2rem;color:#88aaff;white-space:nowrap}
nav{display:flex;gap:8px;flex-wrap:wrap}
nav a{color:#dbe6ff;text-decoration:none;border:1px solid #3e5f9c;border-radius:6px;
      padding:7px 10px;background:#1d2b45;white-space:nowrap}
nav a:hover,nav a:focus,nav a[aria-current="page"]{border-color:#88aaff;background:#26385a;outline:none}
#map-count{margin-left:auto;color:#aaa;font-size:.9rem}
main{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 62px)}
#filters{border-right:1px solid #333;background:#171717;padding:14px;display:flex;
         flex-direction:column;gap:12px;min-width:0}
label{display:flex;flex-direction:column;gap:5px;color:#aaa;font-size:.78rem;font-weight:700}
select{width:100%;padding:8px 10px;background:#222;border:1px solid #444;border-radius:6px;
       color:#fff;min-width:0}
select:focus{outline:none;border-color:#88aaff}
#status{font-size:.85rem;color:#aaa;line-height:1.4}
#visible-list{display:flex;flex-direction:column;gap:8px;overflow:auto;padding-right:2px}
.list-item{display:grid;grid-template-columns:54px 1fr;gap:8px;color:#ddd;text-decoration:none;
           border:1px solid #333;border-radius:6px;padding:6px;background:#1e1e1e;min-width:0}
.list-item:hover,.list-item:focus{border-color:#88aaff;outline:none}
.list-item img{width:54px;height:42px;object-fit:cover;border-radius:4px;background:#050505}
.list-title{font-size:.82rem;font-weight:700;overflow-wrap:anywhere;line-height:1.25}
.list-meta{font-size:.74rem;color:#999;overflow-wrap:anywhere;line-height:1.25;margin-top:2px}
#map-wrap{position:relative;min-width:0;background:#0d0d0d}
#map{height:calc(100vh - 62px);min-height:520px;width:100%}
#fallback-list{display:none;padding:16px;max-width:980px}
.fallback-card{display:grid;grid-template-columns:92px 1fr;gap:12px;margin-bottom:10px;
               border:1px solid #333;border-radius:6px;background:#1b1b1b;padding:10px;color:#ddd;
               text-decoration:none}
.fallback-card:hover,.fallback-card:focus{border-color:#88aaff;outline:none}
.fallback-card img{width:92px;height:68px;object-fit:cover;border-radius:4px;background:#050505}
.popup{width:min(310px,76vw)}
.popup-grid{display:grid;grid-template-columns:72px 1fr;gap:8px;margin:6px 0;color:#222}
.popup-grid img{width:72px;height:58px;object-fit:cover;border-radius:4px;background:#eee}
.popup-title{font-weight:700;line-height:1.25;overflow-wrap:anywhere}
.popup-meta{font-size:.82rem;color:#444;line-height:1.3;overflow-wrap:anywhere;margin-top:3px}
.popup a{color:#1d5fd1}
.leaflet-popup-content{margin:10px 12px}
.leaflet-container{background:#151515}
.map-unavailable #map{display:none}
.map-unavailable #fallback-list{display:block}
@media (max-width:760px){
  header{padding:12px}
  #map-count{margin-left:0;width:100%}
  main{display:flex;flex-direction:column}
  #filters{border-right:0;border-bottom:1px solid #333;display:grid;grid-template-columns:1fr;
           max-height:none}
  #visible-list{max-height:160px}
  #map{height:62vh;min-height:360px}
}
"""


_MAP_JS = """const records = Array.isArray(window.MAP_EXPORT_DATA) ? window.MAP_EXPORT_DATA : [];
let map = null;
let markerLayer = null;
let activeRecords = [];

const els = {
  person: document.getElementById('person-filter'),
  year: document.getElementById('year-filter'),
  folder: document.getElementById('folder-filter'),
  count: document.getElementById('map-count'),
  status: document.getElementById('status'),
  visibleList: document.getElementById('visible-list'),
  fallback: document.getElementById('fallback-list'),
  wrap: document.getElementById('map-wrap')
};

function text(value){
  return value == null ? '' : String(value);
}

function option(select, value, label){
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label;
  select.appendChild(opt);
}

function yearOf(record){
  const match = text(record.dateTaken).match(/\\b(\\d{4})\\b/);
  return match ? match[1] : '';
}

function fillFilters(){
  const people = new Set();
  const years = new Set();
  const folders = new Set();
  records.forEach(record => {
    (record.people || []).forEach(name => { if(name) people.add(name); });
    const year = yearOf(record);
    if(year) years.add(year);
    if(record.folder) folders.add(record.folder);
  });
  [...people].sort((a,b)=>a.localeCompare(b,'hu-HU')).forEach(name => option(els.person, name, name));
  [...years].sort().forEach(year => option(els.year, year, year));
  [...folders].sort((a,b)=>a.localeCompare(b,'hu-HU')).forEach(folder => option(els.folder, folder, folder));
}

function filteredRecords(){
  const person = els.person.value;
  const year = els.year.value;
  const folder = els.folder.value;
  return records.filter(record => {
    if(person && !(record.people || []).includes(person)) return false;
    if(year && yearOf(record) !== year) return false;
    if(folder && record.folder !== folder) return false;
    return true;
  });
}

function groupByCoordinate(items){
  const groups = new Map();
  items.forEach(record => {
    const lat = Number(record.latitude);
    const lon = Number(record.longitude);
    if(!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    const key = lat.toFixed(6) + ',' + lon.toFixed(6);
    if(!groups.has(key)) groups.set(key, {lat, lon, items: []});
    groups.get(key).items.push(record);
  });
  return [...groups.values()];
}

function popupHtml(items){
  const rows = items.slice(0, 12).map(record => {
    const people = (record.people || []).join(', ');
    const meta = [record.dateTaken, people, record.placeName, record.folder].filter(Boolean).join(' · ');
    const href = record.detailPage || record.relativePath || '#';
    const image = record.thumbnailPath || record.relativePath || '';
    return '<div class="popup-grid">' +
      (image ? '<img src="' + escapeAttr(image) + '" alt="">' : '<span></span>') +
      '<div><div class="popup-title">' + escapeHtml(record.fileName || 'Kép') + '</div>' +
      (meta ? '<div class="popup-meta">' + escapeHtml(meta) + '</div>' : '') +
      '<div class="popup-meta">' + Number(record.latitude).toFixed(6) + ', ' + Number(record.longitude).toFixed(6) + '</div>' +
      '<a href="' + escapeAttr(href) + '">Kép megnyitása</a></div></div>';
  }).join('');
  const more = items.length > 12 ? '<div class="popup-meta">+' + (items.length - 12) + ' további kép ezen a ponton</div>' : '';
  return '<div class="popup">' + rows + more + '</div>';
}

function escapeHtml(value){
  return text(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',\"'\":'&#39;'}[ch]));
}

function escapeAttr(value){
  return escapeHtml(value);
}

function initMap(){
  if(!window.L){
    els.wrap.classList.add('map-unavailable');
    els.status.textContent = 'A térképkönyvtár nem töltődött be. A GPS-es képek listában láthatók.';
    return;
  }
  map = L.map('map', {scrollWheelZoom: true});
  const tiles = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  });
  tiles.on('tileerror', () => {
    els.status.textContent = 'A térképcsempék nem tölthetők be, de a markerek és a lista elérhetők.';
  });
  tiles.addTo(map);
  markerLayer = L.markerClusterGroup ? L.markerClusterGroup({chunkedLoading: true}) : L.layerGroup();
  markerLayer.addTo(map);
}

function renderMarkers(items){
  activeRecords = items;
  if(markerLayer) markerLayer.clearLayers();
  const groups = groupByCoordinate(items);
  if(map && markerLayer){
    const bounds = [];
    groups.forEach(group => {
      const marker = L.marker([group.lat, group.lon], {title: group.items[0].fileName || ''});
      marker.on('click', () => marker.bindPopup(popupHtml(group.items), {maxWidth: 340}).openPopup());
      marker.on('dblclick', () => {
        const first = group.items[0];
        if(first && first.detailPage) window.location.href = first.detailPage;
      });
      markerLayer.addLayer(marker);
      bounds.push([group.lat, group.lon]);
    });
    if(bounds.length) map.fitBounds(bounds, {padding: [28, 28], maxZoom: 13});
    else map.setView([47.1625, 19.5033], 6);
  }
  renderLists(items);
  els.count.textContent = items.length + ' / ' + records.length + ' GPS-es kép';
  if(!records.length) els.status.textContent = 'Nincs GPS-koordinátával rendelkező kép az exportban.';
  else if(!items.length) els.status.textContent = 'A szűrés nem adott találatot.';
  else els.status.textContent = groups.length + ' térképpont, ' + items.length + ' kép.';
}

function renderLists(items){
  els.visibleList.replaceChildren();
  els.fallback.replaceChildren();
  items.slice(0, 250).forEach(record => {
    els.visibleList.appendChild(listItem(record, 'list-item'));
    els.fallback.appendChild(listItem(record, 'fallback-card'));
  });
  if(items.length > 250){
    const note = document.createElement('div');
    note.className = 'list-meta';
    note.textContent = 'A lista az első 250 képet mutatja. Szűkíts a szűrőkkel.';
    els.visibleList.appendChild(note);
  }
}

function listItem(record, className){
  const link = document.createElement('a');
  link.className = className;
  link.href = record.detailPage || record.relativePath || '#';
  const img = document.createElement('img');
  img.src = record.thumbnailPath || record.relativePath || '';
  img.alt = '';
  img.onerror = () => { img.style.visibility = 'hidden'; };
  const body = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'list-title';
  title.textContent = record.fileName || 'Kép';
  const meta = document.createElement('div');
  meta.className = 'list-meta';
  meta.textContent = [record.dateTaken, (record.people || []).join(', '), record.placeName, record.folder]
    .filter(Boolean).join(' · ');
  const gps = document.createElement('div');
  gps.className = 'list-meta';
  gps.textContent = Number(record.latitude).toFixed(6) + ', ' + Number(record.longitude).toFixed(6);
  body.append(title, meta, gps);
  link.append(img, body);
  return link;
}

function applyFilters(){
  renderMarkers(filteredRecords());
}

fillFilters();
initMap();
['change', 'input'].forEach(evt => {
  els.person.addEventListener(evt, applyFilters);
  els.year.addEventListener(evt, applyFilters);
  els.folder.addEventListener(evt, applyFilters);
});
applyFilters();
"""


_TOUR_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Gallery - Képtúra</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<link rel="stylesheet" href="slideshow.css">
</head>
<body>
<header>
  <h1>Képtúra</h1>
  <nav>
    <a href="index.html">Galéria</a>
    <a aria-current="page" href="slideshow.html">Diavetítés</a>
    <a href="map.html">Térkép</a>
  </nav>
  <button id="filters-toggle" type="button" aria-expanded="false">Szűrők</button>
  <span id="tour-count"></span>
</header>

<section id="filters" hidden aria-label="Szűrők">
  <div class="filter-grid">
    <label>Év
      <select id="f-year"><option value="">Minden év</option></select>
    </label>
    <label>Évtized
      <select id="f-decade"><option value="">Minden évtized</option></select>
    </label>
    <label>Évtől
      <input id="f-year-from" type="number" inputmode="numeric" placeholder="pl. 1950">
    </label>
    <label>Évig
      <input id="f-year-to" type="number" inputmode="numeric" placeholder="pl. 1990">
    </label>
    <label>Település
      <select id="f-city"><option value="">Minden település</option></select>
    </label>
    <label>Helyszín
      <select id="f-location"><option value="">Minden helyszín</option></select>
    </label>
    <label>Mappa
      <select id="f-folder"><option value="">Minden mappa</option></select>
    </label>
  </div>

  <div class="filter-persons">
    <div class="filter-persons-head">
      <span>Személyek</span>
      <span class="person-mode">
        <label><input type="radio" name="person-mode" value="any" checked> bármelyik</label>
        <label><input type="radio" name="person-mode" value="all"> mindegyik</label>
      </span>
    </div>
    <div id="f-persons" class="person-list"></div>
  </div>

  <div class="filter-flags">
    <label><input id="f-identified" type="checkbox"> csak azonosított személyekkel</label>
    <label><input id="f-favorite" type="checkbox"> csak kedvencek</label>
    <label><input id="f-caption" type="checkbox"> csak feliratozott</label>
    <button id="f-reset" type="button">Szűrők törlése</button>
  </div>
</section>

<main>
  <section id="stage-col">
    <div id="stage" class="labels-full">
      <div id="stage-media" class="stage-media">
        <img id="tour-img" src="" alt="">
        <img id="tour-img-top" alt="" aria-hidden="true">
        <div id="tour-overlay" class="bbox-overlay"></div>
        <div id="tour-slider" aria-hidden="true">
          <div class="tour-slider-line"></div>
          <div class="tour-slider-handle">⇔</div>
        </div>
        <div id="tour-pick" aria-hidden="true">
          <label>Bal <select id="tour-left"></select></label>
          <label>Jobb <select id="tour-right"></select></label>
        </div>
        <button id="nav-prev" class="nav-zone nav-prev" type="button" aria-label="Előző kép">‹</button>
        <button id="nav-next" class="nav-zone nav-next" type="button" aria-label="Következő kép">›</button>
      </div>
      <div id="empty-msg" hidden>A szűrés nem adott találatot.</div>
    </div>

    <div id="controls">
      <button id="btn-prev" type="button" title="Előző (←)">⏮</button>
      <button id="btn-play" type="button" title="Lejátszás / Szünet (szóköz)">▶</button>
      <button id="btn-next" type="button" title="Következő (→)">⏭</button>
      <span id="position">0 / 0</span>
      <label class="speed">Sebesség
        <select id="speed">
          <option value="2000">2 mp</option>
          <option value="4000" selected>4 mp</option>
          <option value="6000">6 mp</option>
          <option value="10000">10 mp</option>
        </select>
      </label>
      <span class="label-modes">Névfeliratok:
        <button data-mode="full" class="lm active" type="button">Teljes</button>
        <button data-mode="dim" class="lm" type="button">Halvány</button>
        <button data-mode="off" class="lm" type="button">Kikapcsolva</button>
      </span>
      <button id="btn-fullscreen" type="button" title="Teljes képernyő">⛶</button>
    </div>

    <div id="info"></div>
  </section>

  <aside id="side">
    <div id="map-panel">
      <div class="panel-head">
        <span>Térkép</span>
        <button id="map-toggle" type="button" aria-expanded="true">Összecsukás</button>
      </div>
      <div id="map-body">
        <div id="map" role="region" aria-label="GPS térkép"></div>
        <div id="map-status"></div>
      </div>
    </div>
  </aside>

  <aside id="person-panel" hidden aria-label="Személy adatai">
    <div class="pp-head">
      <span id="pp-title">Személy</span>
      <button id="pp-close" type="button" aria-label="Bezárás">✕</button>
    </div>
    <div id="pp-body"></div>
  </aside>
</main>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="slideshow-data.js"></script>
<script src="slideshow.js"></script>
</body>
</html>
"""


_TOUR_CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{min-height:100vh;background:#111;color:#ddd;font-family:system-ui,sans-serif}
button,input,select{font:inherit}
header{background:#1a1a1a;padding:12px 18px;border-bottom:1px solid #333;
       display:flex;align-items:center;gap:14px;flex-wrap:wrap}
header h1{font-size:1.2rem;color:#88aaff;white-space:nowrap}
nav{display:flex;gap:8px;flex-wrap:wrap}
nav a{color:#dbe6ff;text-decoration:none;border:1px solid #3e5f9c;border-radius:6px;
      padding:7px 10px;background:#1d2b45;white-space:nowrap}
nav a:hover,nav a:focus,nav a[aria-current="page"]{border-color:#88aaff;background:#26385a;outline:none}
#filters-toggle{padding:7px 12px;border:1px solid #3e5f9c;border-radius:6px;color:#eaf0ff;
                background:#1d2b45;cursor:pointer}
#filters-toggle:hover{border-color:#88aaff;background:#26385a}
#tour-count{margin-left:auto;color:#aaa;font-size:.9rem;white-space:nowrap}

#filters{background:#171717;border-bottom:1px solid #333;padding:14px 18px;
         display:flex;flex-direction:column;gap:14px}
#filters[hidden]{display:none}
.filter-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
label{display:flex;flex-direction:column;gap:5px;color:#aaa;font-size:.76rem;font-weight:700}
select,input[type=number],input[type=text]{width:100%;padding:7px 9px;background:#222;
       border:1px solid #444;border-radius:6px;color:#fff;min-width:0}
select:focus,input:focus{outline:none;border-color:#88aaff}
.filter-persons-head{display:flex;justify-content:space-between;align-items:center;gap:10px;
                     color:#aaa;font-size:.76rem;font-weight:700;margin-bottom:6px;flex-wrap:wrap}
.person-mode{display:flex;gap:12px;font-weight:400}
.person-mode label{flex-direction:row;align-items:center;gap:5px;color:#ccc}
.person-list{display:flex;flex-wrap:wrap;gap:6px;max-height:140px;overflow:auto;
             border:1px solid #333;border-radius:6px;padding:8px;background:#141414}
.person-list label{flex-direction:row;align-items:center;gap:6px;color:#ddd;font-weight:400;
                   font-size:.82rem;background:#1e1e1e;border:1px solid #333;border-radius:14px;
                   padding:4px 10px;cursor:pointer}
.person-list label:hover{border-color:#88aaff}
.filter-flags{display:flex;gap:18px;flex-wrap:wrap;align-items:center}
.filter-flags label{flex-direction:row;align-items:center;gap:6px;color:#ddd;font-weight:400;font-size:.85rem}
#f-reset{margin-left:auto;padding:7px 12px;border:1px solid #555;border-radius:6px;
         background:#222;color:#ddd;cursor:pointer}
#f-reset:hover{border-color:#88aaff}

main{display:grid;grid-template-columns:1fr 360px;gap:14px;padding:14px;align-items:start;position:relative}
main.no-map,main.map-collapsed{grid-template-columns:1fr}
main.map-collapsed #map-panel{box-shadow:0 4px 14px rgba(0,0,0,.5)}
main:fullscreen{background:#111;width:100vw;height:100vh;overflow:auto}
main:fullscreen #side{position:sticky;top:0}
main:fullscreen #stage .stage-media>img{max-height:92vh}
/* Fullscreen: hide the control bar; keep only the map show/hide button.
   Esc exits, arrows/space still navigate. */
main:fullscreen #controls{display:none}
/* Collapsed (incl. fullscreen): float the map panel's show/hide button to the
   top-right instead of dropping it below the full-width image. Declared after
   the fullscreen rules so it wins on equal specificity in fullscreen too. */
main.map-collapsed #side{position:absolute;top:14px;right:14px;width:auto;z-index:5}
#stage-col{min-width:0;display:flex;flex-direction:column;gap:12px}
#stage{position:relative;display:flex;align-items:center;justify-content:center;
       background:#050505;border:1px solid #2a2a2a;border-radius:8px;min-height:50vh;
       padding:10px;overflow:hidden}
.stage-media{position:relative;display:inline-block;line-height:0;max-width:100%}
.stage-media>img{display:block;max-width:100%;max-height:72vh;width:auto;height:auto;border-radius:4px}
#empty-msg{color:#aaa;font-size:1rem;padding:40px}
.bbox-overlay{position:absolute;inset:0;pointer-events:none;z-index:45}
/* deoldified compare slider (slideshow: slider only, no toggle buttons) */
#tour-img-top{display:none;position:absolute;inset:0;width:100%;height:100%;
              object-fit:contain;max-width:none;max-height:none;border-radius:4px;
              pointer-events:none;z-index:1}
.stage-media.compare{touch-action:none}
.stage-media.compare #tour-img-top{display:block}
#tour-slider{display:none;position:absolute;top:0;bottom:0;left:50%;width:40px;
             transform:translateX(-50%);cursor:ew-resize;z-index:46;
             align-items:center;justify-content:center;touch-action:none}
.stage-media.compare #tour-slider{display:flex}
/* Left/right pickers — shown only when the image has 3+ related variants.
   Anchored inside #stage-media so they stay visible in fullscreen too. */
#tour-pick{display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);
           z-index:47;gap:10px;padding:5px 9px;border-radius:8px;
           background:rgba(17,17,17,.82);box-shadow:0 1px 6px rgba(0,0,0,.6)}
.stage-media.has-pick #tour-pick{display:flex}
#tour-pick label{display:inline-flex;align-items:center;gap:5px;color:#dbe6ff;
           font-size:12px;white-space:nowrap}
#tour-pick select{padding:3px 6px;border:1px solid #3e5f9c;border-radius:5px;
           color:#eaf0ff;background:#1d2b45;cursor:pointer;font-size:12px}
#tour-slider .tour-slider-line{position:absolute;top:0;bottom:0;left:50%;width:3px;
             transform:translateX(-50%);background:#fff;box-shadow:0 0 4px rgba(0,0,0,.85)}
#tour-slider .tour-slider-handle{width:36px;height:36px;border-radius:50%;background:#fff;
             color:#111;display:flex;align-items:center;justify-content:center;
             font-size:18px;font-weight:700;box-shadow:0 0 7px rgba(0,0,0,.75);
             position:relative;user-select:none;line-height:1}
/* Edge navigation bands: pinned to the SCREEN edges (fixed → also works in
   fullscreen and sits in front of the map). Transparent until hovered; the
   face boxes (z-index 45) stay above them so face taps still win. */
.nav-zone{position:fixed;top:50%;transform:translateY(-50%);height:60vh;
          width:12%;min-width:46px;max-width:120px;
          display:flex;align-items:center;justify-content:center;border:0;cursor:pointer;
          color:#fff;font-size:2.6rem;line-height:1;background:transparent;opacity:0;
          z-index:40;-webkit-tap-highlight-color:transparent;
          transition:opacity .18s,background .18s}
.nav-prev{left:0;justify-content:flex-start;padding-left:8px;
          background:linear-gradient(to right,rgba(0,0,0,.5),transparent)}
.nav-next{right:0;justify-content:flex-end;padding-right:8px;
          background:linear-gradient(to left,rgba(0,0,0,.5),transparent)}
.stage-media:hover .nav-zone{opacity:.55}
.nav-zone:hover,.nav-zone:focus-visible{opacity:1;outline:none}
/* Touch devices navigate by swiping; hide the click bands there. */
@media (hover:none){.nav-zone{display:none}}
.tb{position:absolute;outline:2px solid rgba(72,220,122,.5);background:rgba(72,220,122,.05);
    pointer-events:auto;cursor:pointer;transition:outline-color .15s,background .15s}
.tb:hover,.tb:focus,.tb.show{outline-color:rgba(125,255,166,.98);background:rgba(72,220,122,.13);
    z-index:3;outline-width:3px}
/* Labels off → hide the frame as well; reveal it only on hover/focus/tap. */
.labels-off .tb{outline-color:transparent;background:transparent}
.labels-off .tb:hover,.labels-off .tb:focus,.labels-off .tb.show{
    outline-color:rgba(125,255,166,.98);background:rgba(72,220,122,.13)}
.tb-label{position:absolute;left:0;top:0;transform:translateY(calc(-100% - 4px));
          max-width:min(280px,70vw);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
          background:rgba(12,18,12,.92);color:#75ff75;border:1px solid rgba(50,220,50,.85);
          border-radius:4px;padding:2px 6px;font-size:12px;font-weight:700;line-height:1.25;
          transition:opacity .12s}
.tb-label.inside{transform:none;top:2px;left:2px}
.labels-full .tb-label{opacity:1}
.labels-dim .tb-label{opacity:.28}
.labels-off .tb-label{opacity:0}
.tb:hover .tb-label,.tb:focus .tb-label,.tb.show .tb-label{opacity:1}

/* Tagged objects: amber + dashed (boxes) or amber dots (points), clearly
   distinct from the green solid face frames. Labels follow the label-mode. */
.tob{position:absolute;outline:2px dashed rgba(255,176,32,.85);background:rgba(255,176,32,.05);
     pointer-events:auto;cursor:pointer;transition:outline-color .15s,background .15s}
.tob:hover,.tob:focus,.tob.show{outline-color:rgba(255,204,102,.98);
     background:rgba(255,176,32,.14);z-index:3;outline-width:3px}
.labels-off .tob{outline-color:transparent;background:transparent}
.labels-off .tob:hover,.labels-off .tob:focus,.labels-off .tob.show{
     outline-color:rgba(255,204,102,.98);background:rgba(255,176,32,.14)}
.tobp{position:absolute;transform:translate(-50%,-50%);pointer-events:auto;cursor:pointer;
     display:flex;align-items:center;z-index:3}
.tobp-dot{width:14px;height:14px;border-radius:50%;background:rgba(255,176,32,.9);
     border:2px solid rgba(20,14,0,.85);box-shadow:0 0 8px rgba(255,176,32,.6)}
.tobp:hover .tobp-dot,.tobp:focus .tobp-dot{background:rgba(255,204,102,1)}
.tob-label{position:absolute;left:0;top:0;transform:translateY(calc(-100% - 4px));
          max-width:min(280px,70vw);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
          background:rgba(28,18,0,.92);color:#ffce6e;border:1px solid rgba(255,176,32,.85);
          border-radius:4px;padding:2px 6px;font-size:12px;font-weight:700;line-height:1.25;
          transition:opacity .12s}
.tob-label.inside{transform:none;top:2px;left:2px}
.tob-label.point{left:12px;top:50%;transform:translateY(-50%)}
.labels-full .tob-label{opacity:1}
.labels-dim .tob-label{opacity:.28}
.labels-off .tob-label{opacity:0}
.tob:hover .tob-label,.tob:focus .tob-label,.tob.show .tob-label,
.tobp:hover .tob-label,.tobp:focus .tob-label{opacity:1}

#controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#1a1a1a;
          border:1px solid #2a2a2a;border-radius:8px;padding:10px 12px}
#controls button{min-width:42px;padding:8px 12px;border:1px solid #3e5f9c;border-radius:6px;
                 color:#eaf0ff;background:#1d2b45;cursor:pointer;font-size:1rem}
#controls button:hover:not(:disabled){border-color:#88aaff;background:#26385a}
#controls button:disabled{opacity:.35;cursor:default}
#position{color:#aaa;font-size:.9rem;min-width:64px;text-align:center}
.speed{flex-direction:row;align-items:center;gap:6px;color:#aaa;font-size:.76rem;font-weight:700}
.speed select{width:auto}
.label-modes{display:flex;align-items:center;gap:4px;color:#aaa;font-size:.76rem;font-weight:700;flex-wrap:wrap}
.label-modes .lm{min-width:0;padding:6px 9px;font-size:.8rem}
.label-modes .lm.active{background:#2d4a78;border-color:#88aaff}
#btn-fullscreen{margin-left:auto}

#info{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:12px 14px;
      display:flex;flex-direction:column;gap:6px;line-height:1.5}
#info .info-title{font-size:1.05rem;font-weight:700;color:#cfe0ff;overflow-wrap:anywhere}
#info .info-row{font-size:.9rem;color:#cfcfcf}
#info .info-row b{color:#9fb6e6;font-weight:700}
#info .info-desc{font-size:.9rem;color:#bdbdbd;font-style:italic;overflow-wrap:anywhere}
.badge{display:inline-block;font-size:.72rem;background:#243a5e;color:#bcd2ff;border:1px solid #3e5f9c;
       border-radius:10px;padding:1px 8px;margin-right:4px}

#side{position:sticky;top:14px;min-width:0}
#map-panel{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;align-items:center;gap:10px;
            padding:10px 12px;border-bottom:1px solid #2a2a2a;color:#cfe0ff;font-weight:700}
#map-toggle{padding:5px 10px;border:1px solid #3e5f9c;border-radius:6px;color:#eaf0ff;
            background:#1d2b45;cursor:pointer;font-size:.8rem}
#map-toggle:hover{border-color:#88aaff;background:#26385a}
#map-body{display:block}
#map-panel.collapsed #map-body{display:none}
#map{height:340px;width:100%;background:#151515}
.leaflet-container{background:#151515}
#map-status{padding:10px 12px;color:#aaa;font-size:.85rem;line-height:1.4}
.leaflet-popup-content{margin:8px 10px;color:#222}

/* clickable person detail panel (works in fullscreen too) */
#person-panel{position:fixed;top:0;right:0;width:min(360px,90vw);height:100vh;
              background:#161616;border-left:1px solid #333;z-index:120;
              display:flex;flex-direction:column;box-shadow:-6px 0 24px rgba(0,0,0,.55)}
#person-panel[hidden]{display:none}
.pp-head{display:flex;justify-content:space-between;align-items:center;gap:10px;
         padding:14px 16px;border-bottom:1px solid #2a2a2a;background:#1a1a1a}
#pp-title{font-size:1.05rem;font-weight:700;color:#cfe0ff;overflow-wrap:anywhere}
#pp-close{min-width:36px;padding:6px 10px;border:1px solid #3e5f9c;border-radius:6px;
          color:#eaf0ff;background:#1d2b45;cursor:pointer;font-size:1.1rem;line-height:1}
#pp-close:hover{border-color:#88aaff;background:#26385a}
#pp-body{padding:16px;overflow:auto;display:flex;flex-direction:column;gap:10px}
#pp-body .pp-thumb{width:120px;height:120px;object-fit:cover;border-radius:8px;
                   border:1px solid #333;align-self:center;background:#050505}
#pp-body .pp-row{font-size:.92rem;color:#d4d4d4;line-height:1.45}
#pp-body .pp-row b{color:#9fb6e6;font-weight:700}
#pp-body .pp-groups{display:flex;flex-wrap:wrap;gap:6px}
#pp-body .pp-group{font-size:.78rem;background:#243a5e;color:#bcd2ff;
                   border:1px solid #3e5f9c;border-radius:10px;padding:2px 8px}
#pp-body .pp-notes{font-size:.9rem;color:#cfcfcf;white-space:pre-wrap;
                   overflow-wrap:anywhere;border-top:1px solid #2a2a2a;padding-top:10px}
#pp-body .pp-empty{color:#888;font-size:.9rem}

@media (max-width:900px){
  main{display:flex;flex-direction:column}
  #side{position:static}
  #stage{min-height:40vh}
  .stage-media>img{max-height:60vh}
  #btn-fullscreen{margin-left:0}
}
"""


_TOUR_JS = """const TOUR = Array.isArray(window.TOUR_DATA) ? window.TOUR_DATA : [];
const PERSONS = (window.TOUR_PERSONS && typeof window.TOUR_PERSONS === 'object') ? window.TOUR_PERSONS : {};
const OBJECTS = (window.TOUR_OBJECTS && typeof window.TOUR_OBJECTS === 'object') ? window.TOUR_OBJECTS : {};
let filtered = TOUR.slice();
let index = 0;
let playing = false;
let timer = null;
let speedMs = 4000;
let labelMode = 'full';
let map = null, marker = null, mapInit = false;

const $ = id => document.getElementById(id);
let tourSplit = 50;
// Members of the current photo's comparison group (B&W + colorized variants)
// when the left/right picker is active; empty for a simple two-image pair.
let compareMembers = [];
const els = {
  count: $('tour-count'), stage: $('stage'), media: $('stage-media'),
  img: $('tour-img'), imgTop: $('tour-img-top'), slider: $('tour-slider'),
  leftSel: $('tour-left'), rightSel: $('tour-right'),
  overlay: $('tour-overlay'), empty: $('empty-msg'),
  info: $('info'), position: $('position'),
  prev: $('btn-prev'), next: $('btn-next'), play: $('btn-play'),
  navPrev: $('nav-prev'), navNext: $('nav-next'),
  speed: $('speed'), full: $('btn-fullscreen'),
  fYear: $('f-year'), fDecade: $('f-decade'), fFrom: $('f-year-from'), fTo: $('f-year-to'),
  fCity: $('f-city'), fLocation: $('f-location'), fFolder: $('f-folder'),
  fPersons: $('f-persons'), fIdentified: $('f-identified'),
  fFavorite: $('f-favorite'), fCaption: $('f-caption'), fReset: $('f-reset'),
  filters: $('filters'), filtersToggle: $('filters-toggle'),
  main: document.querySelector('main'), mapPanel: $('map-panel'),
  mapToggle: $('map-toggle'), mapStatus: $('map-status'),
  personPanel: $('person-panel'), ppTitle: $('pp-title'),
  ppBody: $('pp-body'), ppClose: $('pp-close')
};

function text(v){ return v == null ? '' : String(v); }
function esc(v){
  return text(v).replace(/[&<>\"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch]));
}
function setPct(el, prop, value){
  const n = Number(value);
  el.style[prop] = Number.isFinite(n) ? n.toFixed(6) + '%' : '0%';
}
function option(select, value, label){
  const opt = document.createElement('option');
  opt.value = value; opt.textContent = label;
  select.appendChild(opt);
}
function uniqueSorted(values){
  return [...new Set(values.filter(Boolean))].sort((a,b)=>a.localeCompare(b,'hu-HU'));
}

function buildFilters(){
  uniqueSorted(TOUR.map(r => r.year)).forEach(y => option(els.fYear, y, y));
  uniqueSorted(TOUR.map(r => r.decade)).forEach(d => option(els.fDecade, d, d));
  uniqueSorted(TOUR.map(r => r.city)).forEach(c => option(els.fCity, c, c));
  uniqueSorted(TOUR.map(r => r.locationName)).forEach(l => option(els.fLocation, l, l));
  uniqueSorted(TOUR.map(r => r.folder)).forEach(f => option(els.fFolder, f, f));
  const people = uniqueSorted(TOUR.flatMap(r => r.persons || []));
  people.forEach(name => {
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = name; cb.className = 'person-cb';
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(' ' + name));
    els.fPersons.appendChild(lab);
  });
  if(!people.length){
    els.fPersons.innerHTML = '<span style=\"color:#888;font-size:.8rem\">Nincs megnevezett személy.</span>';
  }
}

function selectedPersons(){
  return [...document.querySelectorAll('.person-cb:checked')].map(cb => cb.value);
}
function personMode(){
  const r = document.querySelector('input[name=person-mode]:checked');
  return r ? r.value : 'any';
}
function yearNum(record){
  const n = parseInt(record.year, 10);
  return Number.isFinite(n) ? n : null;
}

function applyFilters(){
  const year = els.fYear.value, decade = els.fDecade.value;
  const from = parseInt(els.fFrom.value, 10), to = parseInt(els.fTo.value, 10);
  const city = els.fCity.value, location = els.fLocation.value, folder = els.fFolder.value;
  const persons = selectedPersons(), mode = personMode();
  const onlyId = els.fIdentified.checked, onlyFav = els.fFavorite.checked, onlyCap = els.fCaption.checked;

  filtered = TOUR.filter(r => {
    if(year && r.year !== year) return false;
    if(decade && r.decade !== decade) return false;
    const yn = yearNum(r);
    if(Number.isFinite(from) && (yn === null || yn < from)) return false;
    if(Number.isFinite(to) && (yn === null || yn > to)) return false;
    if(city && r.city !== city) return false;
    if(location && r.locationName !== location) return false;
    if(folder && r.folder !== folder) return false;
    if(persons.length){
      const have = r.persons || [];
      if(mode === 'all'){ if(!persons.every(p => have.includes(p))) return false; }
      else { if(!persons.some(p => have.includes(p))) return false; }
    }
    if(onlyId && !r.hasIdentifiedPersons) return false;
    if(onlyFav && !r.isFavorite) return false;
    if(onlyCap && !r.hasCaption) return false;
    return true;
  });

  index = 0;
  render();
}

function setTourSplit(pct){
  tourSplit = Math.max(0, Math.min(100, pct));
  els.imgTop.style.clipPath = 'inset(0 ' + (100 - tourSplit).toFixed(3) + '% 0 0)';
  els.slider.style.left = tourSplit.toFixed(3) + '%';
}

function memberLabel(m){
  return m.isBw ? 'Fekete-fehér' : (m.label ? 'Színes ' + m.label : 'Színes');
}

function fillCompareSelect(select, members, selected){
  select.innerHTML = '';
  members.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = memberLabel(m);
    if(i === selected) opt.selected = true;
    select.appendChild(opt);
  });
}

// Drive the two image layers from the picked left/right members. `imgTop` is
// the LEFT side (clipped from the right by the slider); `img` is the RIGHT.
function applyCompareSides(){
  if(!compareMembers.length) return;
  const last = compareMembers.length - 1;
  const li = Math.min(last, Math.max(0, parseInt(els.leftSel.value, 10) || 0));
  const ri = Math.min(last, Math.max(0, parseInt(els.rightSel.value, 10) || 0));
  els.imgTop.src = compareMembers[li].src;
  els.img.src = compareMembers[ri].src;
}

function setupPairMedia(record){
  const members = (record && record.compare) || null;
  const pair = record && record.pair;
  if(members && members.length >= 3){
    // Three+ members (B&W + several colorized variants): show the left/right
    // picker so any two can be compared, exactly like the gallery detail page.
    compareMembers = members;
    const bwIndex = members.findIndex(m => m.isBw);
    const firstColor = members.findIndex(m => !m.isBw);
    fillCompareSelect(els.leftSel, members, bwIndex >= 0 ? bwIndex : 0);
    fillCompareSelect(els.rightSel, members,
      firstColor >= 0 ? firstColor : Math.min(1, members.length - 1));
    els.media.classList.add('compare', 'has-pick');
    applyCompareSides();
    setTourSplit(50);
  } else if(pair){
    // Simple two-image pair → slider only, no picker.
    compareMembers = [];
    els.media.classList.remove('has-pick');
    els.media.classList.add('compare');
    els.imgTop.src = pair.bw;      // B&W layer clipped on the left, color base on the right
    setTourSplit(50);
  } else {
    compareMembers = [];
    els.media.classList.remove('compare', 'has-pick');
    els.imgTop.src = '';
  }
}

function genderLabel(g){
  if(g === 'male') return 'Férfi';
  if(g === 'female') return 'Nő';
  return '';
}

function openPersonPanel(personId, fallbackName){
  const info = (personId != null) ? PERSONS[String(personId)] : null;
  const name = (info && info.name) || fallbackName || 'Ismeretlen';
  els.ppTitle.textContent = name;
  const parts = [];
  if(info && info.thumb){
    parts.push('<img class=\"pp-thumb\" src=\"' + esc(info.thumb) + '\" alt=\"' + esc(name) + '\">');
  }
  const rows = [];
  if(info){
    if(info.nickname) rows.push(['Becenév', info.nickname]);
    if(info.marriedName) rows.push(['Asszonynév', info.marriedName]);
    const gl = genderLabel(info.gender);
    if(gl) rows.push(['Nem', gl]);
    const birth = [info.birthDate, info.birthPlace].filter(Boolean).join(', ');
    if(birth) rows.push(['Születés', birth]);
    const death = [info.deathDate, info.deathPlace].filter(Boolean).join(', ');
    if(death) rows.push(['Halálozás', death]);
    if(info.familyCode) rows.push(['Családi kód', info.familyCode]);
    if(info.externalFamilyCode) rows.push(['Külső családi kód', info.externalFamilyCode]);
    if(info.groups && info.groups.length){
      rows.push(['Csoportok',
        '<span class=\"pp-groups\">' +
        info.groups.map(g => '<span class=\"pp-group\">' + esc(g) + '</span>').join('') +
        '</span>']);
    }
  }
  rows.forEach(([k,v]) => {
    const val = (k === 'Csoportok') ? v : esc(v);
    parts.push('<div class=\"pp-row\"><b>' + esc(k) + ':</b> ' + val + '</div>');
  });
  if(info && info.notes){
    parts.push('<div class=\"pp-notes\">' + esc(info.notes) + '</div>');
  }
  if(parts.length === (info && info.thumb ? 1 : 0)){
    parts.push('<div class=\"pp-empty\">Nincs további adat ehhez a személyhez.</div>');
  }
  els.ppBody.innerHTML = parts.join('');
  els.personPanel.hidden = false;
}

function closePersonPanel(){
  els.personPanel.hidden = true;
}

// The clickable detail panel is shared by persons and objects. Objects show
// their description and notes (mirrors the person panel's bio + notes).
function openObjectPanel(objectId, fallbackName){
  const info = (objectId != null) ? OBJECTS[String(objectId)] : null;
  const name = (info && info.name) || fallbackName || 'Objektum';
  els.ppTitle.textContent = name;
  const parts = [];
  if(info && info.thumb){
    parts.push('<img class=\"pp-thumb\" src=\"' + esc(info.thumb) + '\" alt=\"' + esc(name) + '\">');
  }
  if(info && info.description){
    parts.push('<div class=\"pp-row\">' + esc(info.description) + '</div>');
  }
  if(info && info.persons && info.persons.length){
    const people = info.persons.map(p => {
      const lab = p.roleLabel || p.role;
      return '<span class=\"pp-group\">' + esc(p.name) + (lab ? ' — ' + esc(lab) : '') + '</span>';
    }).join('');
    parts.push('<div class=\"pp-row\"><b>Kapcsolódó személyek:</b></div>'
      + '<div class=\"pp-groups\">' + people + '</div>');
  }
  if(info && info.imageCount){
    parts.push('<div class=\"pp-row\"><b>Képek száma:</b> ' + esc(info.imageCount) + '</div>');
  }
  if(info && info.notes){
    parts.push('<div class=\"pp-notes\">' + esc(info.notes) + '</div>');
  }
  if(parts.length === (info && info.thumb ? 1 : 0)){
    parts.push('<div class=\"pp-empty\">Nincs további adat ehhez az objektumhoz.</div>');
  }
  els.ppBody.innerHTML = parts.join('');
  els.personPanel.hidden = false;
}

function clearOverlay(){
  while(els.overlay.firstChild) els.overlay.removeChild(els.overlay.firstChild);
}
function renderOverlay(record){
  clearOverlay();
  (record.faces || []).forEach(face => {
    const bbox = face.bbox || {};
    const box = document.createElement('div');
    box.className = 'tb';
    box.tabIndex = 0;
    const name = face.name || 'Ismeretlen arc';
    box.title = name;
    box.setAttribute('role', 'button');
    box.setAttribute('aria-label', 'Arc: ' + name);
    setPct(box, 'left', bbox.left);
    setPct(box, 'top', bbox.top);
    setPct(box, 'width', bbox.width);
    setPct(box, 'height', bbox.height);
    const label = document.createElement('div');
    label.className = 'tb-label';
    if(Number(bbox.top) < 8) label.classList.add('inside');
    label.textContent = name;
    box.appendChild(label);
    // Click/tap opens the closable person detail panel (works in fullscreen).
    box.addEventListener('click', e => {
      e.stopPropagation();
      openPersonPanel(face.person_id, name);
    });
    els.overlay.appendChild(box);
  });
  // Tagged objects: amber boxes (bbox) or dots (point), distinct from faces.
  // Clicking opens the same panel, filled with the object's description.
  (record.objects || []).forEach(obj => {
    const name = obj.name || 'Objektum';
    const title = obj.note ? (name + ' — ' + obj.note) : name;
    let el;
    if(obj.bbox){
      el = document.createElement('div');
      el.className = 'tob';
      setPct(el, 'left', obj.bbox.left);
      setPct(el, 'top', obj.bbox.top);
      setPct(el, 'width', obj.bbox.width);
      setPct(el, 'height', obj.bbox.height);
      const label = document.createElement('div');
      label.className = 'tob-label';
      if(Number(obj.bbox.top) < 8) label.classList.add('inside');
      label.textContent = name;
      el.appendChild(label);
    } else if(obj.point){
      el = document.createElement('div');
      el.className = 'tobp';
      setPct(el, 'left', obj.point.left);
      setPct(el, 'top', obj.point.top);
      const dot = document.createElement('span');
      dot.className = 'tobp-dot';
      el.appendChild(dot);
      const label = document.createElement('div');
      label.className = 'tob-label point';
      label.textContent = name;
      el.appendChild(label);
    } else {
      return;  // no geometry → nothing to anchor on the image
    }
    el.tabIndex = 0;
    el.title = title;
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', 'Objektum: ' + name);
    el.addEventListener('click', e => {
      e.stopPropagation();
      openObjectPanel(obj.object_id, name);
    });
    els.overlay.appendChild(el);
  });
}

function renderInfo(record){
  const rows = [];
  const titleText = record.title || record.fileName || 'Kép';
  let html = '<div class=\"info-title\">' + esc(titleText) + '</div>';
  const badges = [];
  if(record.isFavorite) badges.push('★ Kedvenc');
  if(record.hasIdentifiedPersons) badges.push('Azonosított');
  if(record.hasCaption) badges.push('Felirat');
  if(badges.length) html += '<div>' + badges.map(b => '<span class=\"badge\">' + esc(b) + '</span>').join('') + '</div>';
  if(record.date || record.year) rows.push(['Dátum', record.date || record.year]);
  else if(record.decade) rows.push(['Évtized', record.decade]);
  const place = [record.locationName, record.city].filter((v,i,a)=>v && a.indexOf(v)===i).join(', ');
  if(place) rows.push(['Helyszín', place]);
  if(record.persons && record.persons.length) rows.push(['Személyek', record.persons.join(', ')]);
  if(record.objects && record.objects.length){
    const objNames = [...new Set(record.objects.map(o => o.name).filter(Boolean))];
    if(objNames.length) rows.push(['Objektumok', objNames.join(', ')]);
  }
  if(record.folder) rows.push(['Mappa', record.folder]);
  if(record.fileName) rows.push(['Fájl', record.fileName]);
  rows.forEach(([k,v]) => { html += '<div class=\"info-row\"><b>' + esc(k) + ':</b> ' + esc(v) + '</div>'; });
  if(record.description) html += '<div class=\"info-desc\">' + esc(record.description) + '</div>';
  els.info.innerHTML = html;
}

function ensureMap(){
  if(mapInit || !window.L) return;
  map = L.map('map', { scrollWheelZoom: true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  map.setView([47.1625, 19.5033], 6);
  mapInit = true;
}

function updateMap(record){
  if(!window.L){
    els.mapStatus.textContent = 'A térképkönyvtár nem töltődött be (nincs internet?).';
    return;
  }
  ensureMap();
  const lat = Number(record && record.gpsLatitude);
  const lon = Number(record && record.gpsLongitude);
  if(record && record.hasLocation && Number.isFinite(lat) && Number.isFinite(lon)){
    if(marker) marker.remove();
    marker = L.marker([lat, lon]).addTo(map);
    const place = record.locationName || record.city || '';
    if(place) marker.bindPopup(esc(place)).openPopup();
    map.setView([lat, lon], 13);
    els.mapStatus.textContent = place
      ? place + ' (' + lat.toFixed(5) + ', ' + lon.toFixed(5) + ')'
      : lat.toFixed(5) + ', ' + lon.toFixed(5);
    setTimeout(() => map && map.invalidateSize(), 60);
  } else {
    if(marker){ marker.remove(); marker = null; }
    els.mapStatus.textContent = 'Ehhez a képhez nincs elérhető helyadat.';
  }
}

function render(){
  const total = filtered.length;
  els.count.textContent = total + ' / ' + TOUR.length + ' kép';
  if(!total){
    stopPlay();
    els.media.hidden = true;
    els.empty.hidden = false;
    els.position.textContent = '0 / 0';
    els.info.innerHTML = '';
    els.prev.disabled = els.next.disabled = els.play.disabled = true;
    if(marker){ marker.remove(); marker = null; }
    els.mapStatus.textContent = 'Nincs megjeleníthető kép.';
    return;
  }
  els.media.hidden = false;
  els.empty.hidden = true;
  els.play.disabled = false;
  if(index < 0) index = total - 1;
  if(index >= total) index = 0;
  const record = filtered[index];
  els.img.onload = () => renderOverlay(record);
  els.img.alt = record.title || record.fileName || '';
  // With a deoldified pair the base layer is the color side (shown on the
  // right); the B&W layer is clipped on top from the left via the slider.
  els.img.src = record.pair ? record.pair.color : record.imagePath;
  setupPairMedia(record);
  renderOverlay(record);
  renderInfo(record);
  updateMap(record);
  els.position.textContent = (index + 1) + ' / ' + total;
  els.prev.disabled = total < 2;
  els.next.disabled = total < 2;
}

function go(delta){
  if(!filtered.length) return;
  index = (index + delta + filtered.length) % filtered.length;
  render();
}

function startPlay(){
  if(filtered.length < 2) return;
  playing = true;
  els.play.textContent = '⏸';
  clearInterval(timer);
  timer = setInterval(() => go(1), speedMs);
}
function stopPlay(){
  playing = false;
  els.play.textContent = '▶';
  clearInterval(timer);
  timer = null;
}
function togglePlay(){ playing ? stopPlay() : startPlay(); }

function setLabelMode(mode){
  labelMode = mode;
  els.stage.classList.remove('labels-full', 'labels-dim', 'labels-off');
  els.stage.classList.add('labels-' + mode);
  document.querySelectorAll('.label-modes .lm').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
}

function toggleFullscreen(){
  // Fullscreen the whole two-panel area so the map stays visible.
  const target = els.main;
  if(!document.fullscreenElement){
    if(target.requestFullscreen) target.requestFullscreen();
  } else if(document.exitFullscreen){
    document.exitFullscreen();
  }
}

// --- wiring ---
[els.fYear, els.fDecade, els.fCity, els.fLocation, els.fFolder].forEach(s =>
  s.addEventListener('change', applyFilters));
[els.fFrom, els.fTo].forEach(i => i.addEventListener('input', applyFilters));
[els.fIdentified, els.fFavorite, els.fCaption].forEach(c =>
  c.addEventListener('change', applyFilters));
els.fPersons.addEventListener('change', applyFilters);
document.querySelectorAll('input[name=person-mode]').forEach(r =>
  r.addEventListener('change', applyFilters));

els.fReset.addEventListener('click', () => {
  [els.fYear, els.fDecade, els.fCity, els.fLocation, els.fFolder].forEach(s => s.value = '');
  els.fFrom.value = ''; els.fTo.value = '';
  els.fIdentified.checked = els.fFavorite.checked = els.fCaption.checked = false;
  document.querySelectorAll('.person-cb').forEach(cb => cb.checked = false);
  const anyMode = document.querySelector('input[name=person-mode][value=any]');
  if(anyMode) anyMode.checked = true;
  applyFilters();
});

els.prev.addEventListener('click', () => { stopPlay(); go(-1); });
els.next.addEventListener('click', () => { stopPlay(); go(1); });
els.navPrev.addEventListener('click', () => { stopPlay(); go(-1); });
els.navNext.addEventListener('click', () => { stopPlay(); go(1); });
els.play.addEventListener('click', togglePlay);

// Touch swipe (works in fullscreen too): swipe left → next, right → previous.
let touchX = null, touchY = null;
els.stage.addEventListener('touchstart', e => {
  const t = e.changedTouches[0];
  touchX = t.clientX; touchY = t.clientY;
}, { passive: true });
els.stage.addEventListener('touchend', e => {
  if(touchX === null) return;
  const t = e.changedTouches[0];
  const dx = t.clientX - touchX, dy = t.clientY - touchY;
  touchX = touchY = null;
  if(Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5){
    stopPlay();
    go(dx < 0 ? 1 : -1);
  }
}, { passive: true });
// Deoldified compare slider: drag anywhere on the image to reveal the
// colorized half over the B&W base. Active only when the image has a pair.
(function initTourCompareDrag(){
  let dragging = false;
  function pctFromEvent(e){
    const rect = els.media.getBoundingClientRect();
    if(!rect.width) return tourSplit;
    return ((e.clientX - rect.left) / rect.width) * 100;
  }
  els.media.addEventListener('pointerdown', e => {
    if(!els.media.classList.contains('compare')) return;
    // Clicking a face/object panel or the compare picker — don't move the slider.
    if(e.target.closest && e.target.closest('.tb, .tob, .tobp, #tour-pick')) return;
    dragging = true;
    setTourSplit(pctFromEvent(e));
    e.preventDefault();
  });
  window.addEventListener('pointermove', e => {
    if(!dragging || !els.media.classList.contains('compare')) return;
    setTourSplit(pctFromEvent(e));
  });
  window.addEventListener('pointerup', () => { dragging = false; });
  // Stop touch-drags on a paired image from triggering swipe navigation.
  els.media.addEventListener('touchstart', e => {
    if(els.media.classList.contains('compare')) e.stopPropagation();
  }, { passive: true });
})();

els.speed.addEventListener('change', () => {
  speedMs = parseInt(els.speed.value, 10) || 4000;
  if(playing) startPlay();
});
els.leftSel.addEventListener('change', applyCompareSides);
els.rightSel.addEventListener('change', applyCompareSides);
els.full.addEventListener('click', toggleFullscreen);
els.ppClose.addEventListener('click', closePersonPanel);
document.querySelectorAll('.label-modes .lm').forEach(b =>
  b.addEventListener('click', () => setLabelMode(b.dataset.mode)));

els.filtersToggle.addEventListener('click', () => {
  const open = els.filters.hidden;
  els.filters.hidden = !open;
  els.filtersToggle.setAttribute('aria-expanded', String(open));
});
els.mapToggle.addEventListener('click', () => {
  const collapsed = els.mapPanel.classList.toggle('collapsed');
  // Collapse the side column too so the image grows into the freed space.
  els.main.classList.toggle('map-collapsed', collapsed);
  els.mapToggle.textContent = collapsed ? 'Kinyitás' : 'Összecsukás';
  els.mapToggle.setAttribute('aria-expanded', String(!collapsed));
  if(!collapsed && map) setTimeout(() => map.invalidateSize(), 60);
});

// Keep Leaflet sized correctly when entering/leaving fullscreen.
document.addEventListener('fullscreenchange', () => {
  if(map) setTimeout(() => map.invalidateSize(), 80);
});

// Clicking anywhere outside a face/object hides any tapped-open frame/label.
document.addEventListener('click', e => {
  if(e.target.closest && e.target.closest('.tb, .tob, .tobp')) return;
  document.querySelectorAll('.tb.show, .tob.show, .tobp.show')
    .forEach(b => b.classList.remove('show'));
});

document.addEventListener('keydown', e => {
  if(/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  if(e.key === 'Escape' && !els.personPanel.hidden){ closePersonPanel(); return; }
  if(e.key === 'ArrowLeft'){ e.preventDefault(); stopPlay(); go(-1); }
  else if(e.key === 'ArrowRight'){ e.preventDefault(); stopPlay(); go(1); }
  else if(e.key === ' '){ e.preventDefault(); togglePlay(); }
});

buildFilters();
setLabelMode('full');
render();
"""


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:120] or "collage"


# ---------------------------------------------------------------------------
# Collage static HTML template
# ---------------------------------------------------------------------------

_COLLAGE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kollázs Galéria</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111;color:#ddd;font-family:system-ui,sans-serif}
  header{background:#1a1a1a;padding:14px 20px;border-bottom:1px solid #333;
         display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:1.2rem;color:#88aaff;white-space:nowrap}
  #search{flex:1;min-width:180px;padding:8px 12px;background:#222;
          border:1px solid #444;border-radius:6px;color:#fff;font-size:1rem}
  #search:focus{outline:none;border-color:#88aaff}
  #count{font-size:.85rem;color:#888;white-space:nowrap}
  .collage-block{margin:24px 16px;border:1px solid #333;border-radius:8px;overflow:hidden}
  .collage-header{background:#1c1c1c;padding:12px 16px;border-bottom:1px solid #333}
  .collage-title{font-size:1.05rem;font-weight:bold;color:#aaccff}
  .collage-date{font-size:.85rem;color:#777;margin-top:2px}
  .collage-canvas{position:relative;overflow:hidden;background:#222;display:block}
  .collage-canvas img.base-img{display:block;width:100%;height:auto}
  .collage-canvas svg.overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:all}
  .node-rect{fill:transparent;stroke:rgba(80,160,255,.5);stroke-width:1.5;cursor:pointer;transition:stroke .15s}
  .node-rect:hover{stroke:rgba(255,200,60,.9);stroke-width:2.5}
  .node-rect.missing{stroke:rgba(200,60,60,.7)}
  .face-rect{fill:transparent;stroke:rgba(50,220,50,.8);stroke-width:1.5;pointer-events:none}
  #tip{display:none;position:fixed;z-index:200;background:#1a1a2e;border:1px solid #88aaff;
       border-radius:8px;padding:12px 16px;max-width:320px;font-size:.88rem;
       color:#ddd;box-shadow:0 4px 24px rgba(0,0,0,.7);line-height:1.6}
  #tip .tip-title{font-weight:bold;color:#aaccff;margin-bottom:6px}
  #tip .tip-warn{color:#e57373}
  #mob-panel{display:none;position:fixed;bottom:0;left:0;right:0;z-index:300;
             background:#1a1a2e;border-top:2px solid #88aaff;padding:16px;
             max-height:50vh;overflow-y:auto;font-size:.92rem;line-height:1.7}
  #mob-panel-close{float:right;font-size:1.4rem;cursor:pointer;color:#aaa;margin-top:-4px}
  #mob-panel .tip-title{font-weight:bold;color:#aaccff;font-size:1rem;margin-bottom:8px;display:block}
</style>
</head>
<body>
<header>
  <h1>\U0001f5bc Kollázs Gal\u00e9ria</h1>
  <input id="search" type="text" placeholder="Szem\u00e9ly neve\u2026" oninput="filterByPerson()">
  <span id="count"></span>
</header>
<div id="collages"></div>
<div id="tip"></div>
<div id="mob-panel">
  <span id="mob-panel-close" onclick="closeMob()">\u2715</span>
  <div id="mob-content"></div>
</div>
<script>
const COLLAGES = __COLLAGES_JSON__;
const tip = document.getElementById('tip');
let tipTimer;
function esc(value){
  return String(value??'').replace(/[&<>"']/g,ch=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[ch]));
}
function showTip(evt, html){clearTimeout(tipTimer);tip.innerHTML=html;tip.style.display='block';moveTip(evt);}
function moveTip(evt){
  const mx=evt.clientX,my=evt.clientY,tw=tip.offsetWidth,th=tip.offsetHeight,ww=window.innerWidth,wh=window.innerHeight;
  tip.style.left=(mx+tw+20>ww?mx-tw-12:mx+14)+'px';
  tip.style.top=(my+th+20>wh?my-th-12:my+14)+'px';
}
function hideTip(){tipTimer=setTimeout(()=>{tip.style.display='none';},120);}
function showMob(html){document.getElementById('mob-content').innerHTML=html;document.getElementById('mob-panel').style.display='block';}
function closeMob(){document.getElementById('mob-panel').style.display='none';}
function nodeInfoHtml(node){
  let h='<div class="tip-title">'+esc(node.src||'\u2014')+'</div>';
  if(node.missing)h+='<div class="tip-warn">\u26a0 Forr\u00e1sf\u00e1jl hi\u00e1nyzik</div>';
  if(node.year)h+='<div><b>\u00c9v:</b> '+esc(node.year)+'</div>';
  if(node.location)h+='<div><b>Helysz\u00edn:</b> '+esc(node.location)+'</div>';
  if(node.event)h+='<div><b>Esem\u00e9ny:</b> '+esc(node.event)+'</div>';
  if(node.notes)h+='<div><b>Megjegyz\u00e9s:</b> '+esc(node.notes)+'</div>';
  if(node.faces&&node.faces.length){
    const names=[...new Set(node.faces.map(f=>f.name).filter(Boolean))];
    if(names.length)h+='<div><b>Szem\u00e9lyek:</b> '+names.map(esc).join(', ')+'</div>';
  }
  return h;
}
function filterByPerson(){
  const q=document.getElementById('search').value.toLowerCase().trim();
  document.querySelectorAll('[data-collage-id]').forEach(block=>{
    if(!q){block.style.display='';return;}
    const cid=parseInt(block.dataset.collageId);
    const col=COLLAGES.find(c=>c.id===cid);
    if(!col){block.style.display='none';return;}
    const match=col.nodes.some(n=>n.faces&&n.faces.some(f=>f.name&&f.name.toLowerCase().includes(q)));
    block.style.display=match?'':'none';
    block.querySelectorAll('.node-rect').forEach(r=>{
      const nidx=parseInt(r.dataset.nidx);
      const node=col.nodes[nidx];
      const has=node&&node.faces&&node.faces.some(f=>f.name&&f.name.toLowerCase().includes(q));
      r.style.stroke=has?'rgba(255,200,60,.95)':'';
      r.style.strokeWidth=has?'3':'';
    });
  });
  updateCount();
}
function updateCount(){
  const total=document.querySelectorAll('[data-collage-id]').length;
  const vis=document.querySelectorAll('[data-collage-id]:not([style*="none"])').length;
  document.getElementById('count').textContent=vis+' / '+total+' koll\u00e1zs';
}
function buildSvg(col,svgEl){
  svgEl.setAttribute('viewBox','0 0 '+col.width+' '+col.height);
  svgEl.setAttribute('preserveAspectRatio','xMidYMid meet');
  col.nodes.forEach((node,nidx)=>{
    const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
    rect.setAttribute('x',node.x);rect.setAttribute('y',node.y);
    rect.setAttribute('width',node.w);rect.setAttribute('height',node.h);
    rect.classList.add('node-rect');
    if(node.missing)rect.classList.add('missing');
    rect.dataset.nidx=nidx;
    const infoHtml=nodeInfoHtml(node);
    const isMob=()=>window.matchMedia('(hover:none)').matches;
    rect.addEventListener('mouseenter',e=>{if(!isMob())showTip(e,infoHtml);});
    rect.addEventListener('mousemove',e=>{if(!isMob())moveTip(e);});
    rect.addEventListener('mouseleave',()=>hideTip());
    rect.addEventListener('click',e=>{e.stopPropagation();if(isMob())showMob(infoHtml);else showTip(e,infoHtml);});
    svgEl.appendChild(rect);
    if(node.faces){
      node.faces.forEach(f=>{
        const fr=document.createElementNS('http://www.w3.org/2000/svg','rect');
        fr.setAttribute('x',f.x);fr.setAttribute('y',f.y);
        fr.setAttribute('width',f.w);fr.setAttribute('height',f.h);
        fr.classList.add('face-rect');
        svgEl.appendChild(fr);
        if(f.name){
          const txt=document.createElementNS('http://www.w3.org/2000/svg','text');
          txt.setAttribute('x',f.x+2);
          txt.setAttribute('y',Math.max(f.y-3,12));
          txt.setAttribute('font-size',Math.max(9,Math.min(14,f.w/5)));
          txt.setAttribute('fill','rgba(50,220,50,.95)');
          txt.setAttribute('pointer-events','none');
          txt.textContent=f.name;
          svgEl.appendChild(txt);
        }
      });
    }
  });
}
function buildAll(){
  const wrap=document.getElementById('collages');
  COLLAGES.forEach(col=>{
    const block=document.createElement('div');
    block.className='collage-block';
    block.dataset.collageId=col.id;
    const hdr=document.createElement('div');
    hdr.className='collage-header';
    const title=document.createElement('div');
    title.className='collage-title';
    title.textContent=col.title;
    hdr.appendChild(title);
    if(col.date){
      const date=document.createElement('div');
      date.className='collage-date';
      date.textContent=col.date;
      hdr.appendChild(date);
    }
    block.appendChild(hdr);
    const canvas=document.createElement('div');
    canvas.className='collage-canvas';
    if(col.img){
      const img=document.createElement('img');
      img.className='base-img';img.src=col.img;img.alt=col.title;
      canvas.appendChild(img);
    }
    const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');
    svg.classList.add('overlay');
    buildSvg(col,svg);
    canvas.appendChild(svg);
    block.appendChild(canvas);
    wrap.appendChild(block);
  });
  updateCount();
}
document.addEventListener('click',e=>{
  const p=document.getElementById('mob-panel');
  if(p.style.display!=='none'&&!p.contains(e.target))closeMob();
});
buildAll();
</script>
</body>
</html>
"""
