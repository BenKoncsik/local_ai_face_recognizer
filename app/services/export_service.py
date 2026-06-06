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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import Face, Image, Person
from app.utils.image_utils import load_image_bgr, save_image_bgr

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

    def export_html(
        self,
        target_dir: str,
        person_id: Optional[int] = None,
    ) -> Path:
        """Generate a static HTML gallery to *target_dir*.

        Creates:
          index.html   – searchable gallery with per-person filtering
          images/      – normalized original images used by the HTML overlay
          thumbs/      – face-crop thumbnails
        """
        out = Path(target_dir)
        img_dir = out / "images"
        thumb_dir = out / "thumbs"
        img_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)

        persons = self._get_persons(person_id)

        # --- build data structures ---
        # image_path -> list of (person, face) records. Face bounding boxes are
        # exported as percentages so CSS overlays stay aligned after resizing.
        image_faces: Dict[str, List[Tuple[Person, Face]]] = {}
        image_by_path: Dict[str, Image] = {}
        faceless_image_paths: set[str] = set()
        # person_name → list of thumb filenames
        person_thumbs: Dict[str, List[str]] = {}
        # person_name → list of image record ids
        person_images: Dict[str, List[str]] = {}

        for person in persons:
            faces = self._get_faces(person.id)
            person_thumbs.setdefault(person.name, [])
            person_images.setdefault(person.name, [])
            for face in faces:
                if face.image:
                    ip = face.image.file_path
                    image_faces.setdefault(ip, []).append((person, face))
                    image_by_path[ip] = face.image
                if face.crop_path and Path(face.crop_path).exists():
                    dst_thumb = thumb_dir / f"face_{face.id}.jpg"
                    shutil.copy2(face.crop_path, dst_thumb)
                    person_thumbs[person.name].append(dst_thumb.name)

        if person_id is None:
            faceless_images = (
                self._session.query(Image)
                .filter(~Image.faces.any())
                .order_by(Image.file_path)
                .all()
            )
            if faceless_images:
                no_face_group = "Arc nélküli képek"
                person_thumbs.setdefault(no_face_group, [])
                person_images.setdefault(no_face_group, [])
                for image in faceless_images:
                    image_faces.setdefault(image.file_path, [])
                    image_by_path[image.file_path] = image
                    faceless_image_paths.add(image.file_path)

        # --- export normalized originals and HTML overlay metadata ---
        saved_names: set[str] = set()

        image_records: Dict[str, dict] = {}
        map_records: List[dict] = []
        tour_records: List[dict] = []
        image_order: List[str] = []
        for img_path, face_list in image_faces.items():
            img = load_image_bgr(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            dst_name = _image_export_filename(img_path)
            if not save_image_bgr(img_dir / dst_name, img):
                continue
            saved_names.add(dst_name)

            face_records = []
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
                _append_unique(person_images.setdefault(person.name, []), dst_name)
                _append_unique(persons_in_image, person.name)
                if not person.is_auto_named:
                    has_identified = True

            if img_path in faceless_image_paths:
                _append_unique(person_images.setdefault("Arc nélküli képek", []), dst_name)

            source = Path(img_path)
            db_image = image_by_path.get(img_path)
            image_records[dst_name] = {
                "id": dst_name,
                "src": f"images/{dst_name}",
                "file_name": source.name,
                "folder_name": source.parent.name,
                "width": img_w,
                "height": img_h,
                "faces": face_records,
            }
            pair = self._build_deoldified_pair_export(
                db_image, img_path, dst_name, img_dir, saved_names
            )
            if pair is not None:
                image_records[dst_name]["pair"] = pair
            map_record = _build_map_export_record(db_image, dst_name, source, face_records)
            if map_record is not None:
                map_records.append(map_record)
            tour_records.append(
                _build_tour_export_record(
                    db_image, dst_name, source, face_records,
                    persons_in_image, has_identified, img_w, img_h, pair,
                )
            )
            image_order.append(dst_name)

        # --- build JS data ---
        persons_data = (
            [
                {
                    "name": pname,
                    "thumbs": person_thumbs.get(pname, []),
                    "images": person_images.get(pname, []),
                }
                for pname in sorted(person_thumbs.keys())
            ]
        )

        # --- render HTML ---
        html = (
            _HTML_TEMPLATE
            .replace("__PERSONS_JSON__", _json_for_script(persons_data))
            .replace("__IMAGES_JSON__", _json_for_script(image_records))
            .replace("__IMAGE_ORDER_JSON__", _json_for_script(image_order))
        )
        (out / "index.html").write_text(html, encoding="utf-8")
        _write_map_export_files(out, map_records)

        # Rich person details for the slideshow's clickable person panel.
        person_objs: Dict[int, Person] = {}
        for face_list in image_faces.values():
            for person, _face in face_list:
                person_objs[person.id] = person
        person_details = self._build_person_details_export(
            list(person_objs.values()), thumb_dir
        )
        _write_tour_export_files(out, tour_records, person_details)

        log.info("HTML export: %d person(s) → %s", len(persons), out)
        return out

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

    def _build_deoldified_pair_export(
        self,
        db_image: Optional[Image],
        img_path: str,
        dst_name: str,
        img_dir: Path,
        saved_names: set,
    ) -> Optional[dict]:
        """Find the deoldified counterpart of an image and export both sides.

        Returns ``{"bw": "images/...", "color": "images/..."}`` so the HTML
        viewer can toggle between, and slide-compare, the black-and-white
        original and the colorized version.  Returns None when there is no
        paired image on disk.
        """
        from app.services.deoldified_pairing_service import (
            DeoldifiedPairingService,
            is_deoldified_path,
        )
        from app.services.image_library_service import resolve_image_path

        if db_image is None:
            return None

        svc = DeoldifiedPairingService(self._session)
        current_is_color = is_deoldified_path(img_path)
        partner = (
            svc.find_original_for_deoldified(db_image)
            if current_is_color
            else svc.find_deoldified_for_original(db_image)
        )
        if partner is None:
            return None

        resolved = resolve_image_path(partner)
        partner_disk = str(resolved) if resolved else partner.file_path
        partner_name = _image_export_filename(partner.file_path)
        if partner_name not in saved_names:
            partner_img = load_image_bgr(partner_disk)
            if partner_img is None:
                return None
            if not save_image_bgr(img_dir / partner_name, partner_img):
                return None
            saved_names.add(partner_name)

        partner_src = f"images/{partner_name}"
        current_src = f"images/{dst_name}"
        if current_is_color:
            return {"color": current_src, "bw": partner_src}
        return {"color": partner_src, "bw": current_src}

    def _build_person_details_export(
        self,
        persons: List[Person],
        thumb_dir: Path,
    ) -> Dict[str, dict]:
        """Build a person_id → details map for the slideshow person panel.

        Keyed by stringified person id (JSON object keys are strings).  Copies
        each person's representative thumbnail into ``thumbs/`` when available.
        """
        from app.services.person_group_service import PersonGroupService

        group_svc = PersonGroupService(self._session)
        details: Dict[str, dict] = {}
        for person in persons:
            groups = [g.name for g in group_svc.get_person_groups(person.id)]
            thumb = None
            if person.thumbnail_path and Path(person.thumbnail_path).exists():
                dst = thumb_dir / f"person_{person.id}.jpg"
                try:
                    shutil.copy2(person.thumbnail_path, dst)
                    thumb = f"thumbs/{dst.name}"
                except OSError:
                    thumb = None
            details[str(person.id)] = {
                "id": person.id,
                "name": person.name,
                "isAutoNamed": bool(person.is_auto_named),
                "gender": person.gender or "",
                "familyCode": person.family_code or "",
                "nickname": person.nickname or "",
                "marriedName": person.married_name or "",
                "birthDate": person.birth_date or "",
                "birthPlace": person.birth_place or "",
                "deathDate": person.death_date or "",
                "deathPlace": person.death_place or "",
                "notes": person.notes or "",
                "groups": groups,
                "thumb": thumb,
            }
        return details


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
) -> None:
    persons = persons or {}
    data_json = json.dumps(records, ensure_ascii=False, indent=2)
    (out / "slideshow-data.json").write_text(data_json + "\n", encoding="utf-8")
    (out / "slideshow-data.js").write_text(
        "window.TOUR_DATA = " + _json_for_script(records, indent=2) + ";\n"
        + "window.TOUR_PERSONS = " + _json_for_script(persons, indent=2) + ";\n",
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
# Static HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Gallery</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111;color:#ddd;font-family:system-ui,sans-serif}
  button,input{font:inherit}
  header{background:#1a1a1a;padding:16px 24px;border-bottom:1px solid #333;
         display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:1.2rem;color:#88aaff;white-space:nowrap}
  .nav-link{color:#dbe6ff;text-decoration:none;border:1px solid #3e5f9c;
            border-radius:6px;padding:7px 10px;background:#1d2b45;white-space:nowrap}
  .nav-link:hover,.nav-link:focus{border-color:#88aaff;background:#26385a;outline:none}
  #search{flex:1;min-width:180px;padding:8px 12px;background:#222;
          border:1px solid #444;border-radius:6px;color:#fff;font-size:1rem}
  #search:focus{outline:none;border-color:#88aaff}
  #count{font-size:.85rem;color:#888;white-space:nowrap}

  #persons{display:flex;flex-wrap:wrap;gap:20px;padding:20px}
  .person-card{background:#1c1c1c;border:1px solid #333;border-radius:8px;
               padding:14px;width:260px;transition:border-color .2s}
  .person-card.hidden{display:none}
  .person-card:hover{border-color:#88aaff}
  .person-name{font-weight:bold;font-size:1rem;margin-bottom:10px;color:#eee}
  .thumbs{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
  .thumbs img{width:56px;height:56px;object-fit:cover;border-radius:4px;
              border:1px solid #444;cursor:pointer;transition:border-color .2s}
  .thumbs img:hover{border-color:#88aaff}
  .images-label{font-size:.75rem;color:#888;margin-bottom:6px}
  .img-strip{display:flex;flex-wrap:wrap;gap:4px}
  .image-tile{position:relative;display:inline-block;line-height:0;background:#050505;
              border:1px solid #333;border-radius:4px;overflow:hidden;cursor:pointer;
              transition:border-color .2s,opacity .2s}
  .image-tile:hover,.image-tile:focus{border-color:#88aaff;opacity:.92;outline:none}
  .image-tile>img{display:block;height:86px;max-width:220px;width:auto;object-fit:contain}
  .media-wrap{position:relative;display:inline-block;line-height:0}
  .media-wrap>img{display:block}
  .bbox-overlay{position:absolute;inset:0;pointer-events:none}
  .bbox{position:absolute;outline:2px solid rgba(72,220,122,.42);
        background:rgba(72,220,122,.06);box-shadow:0 0 0 1px rgba(0,0,0,.35);
        pointer-events:auto;cursor:pointer;transition:outline-color .15s,
        outline-width .15s,background .15s,box-shadow .15s}
  .bbox:hover,.bbox:focus{outline-color:rgba(125,255,166,.98);outline-width:3px;
        background:rgba(72,220,122,.13);box-shadow:0 0 0 1px rgba(0,0,0,.55),
        0 0 18px rgba(72,220,122,.35);z-index:3}
  .bbox-label{position:absolute;left:0;top:0;transform:translateY(calc(-100% - 4px));
              max-width:min(280px,70vw);overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap;background:rgba(12,18,12,.9);color:#75ff75;
              border:1px solid rgba(50,220,50,.85);border-radius:4px;
              padding:2px 6px;font-size:12px;font-weight:700;line-height:1.25;
              opacity:0;pointer-events:none;transition:opacity .12s}
  .bbox:hover .bbox-label,.bbox:focus .bbox-label{opacity:1}
  .bbox-label.inside{transform:none;top:2px;left:2px}

  /* lightbox */
  #lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
      z-index:100;align-items:center;justify-content:center;flex-direction:column;
      gap:12px;padding:18px}
  #lb.open{display:flex}
  #lb-toolbar{width:min(92vw,1200px);display:flex;align-items:center;justify-content:center;
              gap:10px;flex-wrap:wrap;color:#ddd}
  #lb-caption{min-width:180px;text-align:center;color:#aaa;font-size:.9rem;
              overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .lb-nav{padding:7px 12px;border:1px solid #3e5f9c;border-radius:6px;
          color:#eaf0ff;background:#1d2b45;cursor:pointer}
  .lb-nav:hover:not(:disabled){border-color:#88aaff;background:#26385a}
  .lb-nav:disabled{opacity:.35;cursor:default}
  #lb-media{max-width:92vw;max-height:82vh;border-radius:6px;
            box-shadow:0 0 0 2px #88aaff;background:#050505}
  #lb-media img{max-width:92vw;max-height:82vh;width:auto;height:auto;border-radius:6px}
  #lb-close{position:fixed;top:16px;right:20px;font-size:2rem;cursor:pointer;
             color:#aaa;line-height:1;background:none;border:none}
  #lb-close:hover{color:#fff}

  /* deoldified pair controls + compare slider */
  #lb-pair-controls{display:none;width:min(92vw,1200px);align-items:center;
                    justify-content:center;gap:8px;flex-wrap:wrap}
  #lb-pair-controls.show{display:flex}
  .lb-nav.active{background:#2f6df0;border-color:#88aaff;color:#fff}
  #lb-img-top{display:none;position:absolute;inset:0;width:100%;height:100%;
              object-fit:contain;border-radius:6px;pointer-events:none}
  #lb-media.compare #lb-img-top{display:block}
  #lb-slider{display:none;position:absolute;top:0;bottom:0;left:50%;width:40px;
             transform:translateX(-50%);cursor:ew-resize;z-index:6;
             align-items:center;justify-content:center;touch-action:none}
  #lb-media.compare #lb-slider{display:flex}
  #lb-slider .lb-slider-line{position:absolute;top:0;bottom:0;left:50%;width:3px;
             transform:translateX(-50%);background:#fff;box-shadow:0 0 4px rgba(0,0,0,.85)}
  #lb-slider .lb-slider-handle{width:36px;height:36px;border-radius:50%;background:#fff;
             color:#111;display:flex;align-items:center;justify-content:center;
             font-size:18px;font-weight:700;box-shadow:0 0 7px rgba(0,0,0,.75);
             position:relative;user-select:none}
  #lb-media.compare>#lb-overlay{z-index:5}
</style>
</head>
<body>
<header>
  <h1>Face Gallery</h1>
  <a class="nav-link" href="slideshow.html">Diavetítés</a>
  <a class="nav-link" href="map.html">Térkép</a>
  <input id="search" type="text" placeholder="Keresés / Search…" oninput="filter()">
  <span id="count"></span>
</header>
<div id="persons"></div>

<!-- lightbox -->
<div id="lb">
  <button id="lb-close" onclick="closeLb()">✕</button>
  <div id="lb-toolbar">
    <button id="lb-prev" class="lb-nav" onclick="showRelativeImage(-1)">Előző kép</button>
    <div id="lb-caption"></div>
    <button id="lb-next" class="lb-nav" onclick="showRelativeImage(1)">Következő kép</button>
  </div>
  <div id="lb-pair-controls">
    <button id="lb-color" class="lb-nav" onclick="setPairView('color')">Színes</button>
    <button id="lb-bw" class="lb-nav" onclick="setPairView('bw')">Fekete-fehér</button>
    <button id="lb-compare" class="lb-nav" onclick="toggleCompare()">Összehasonlítás</button>
  </div>
  <div id="lb-media" class="media-wrap">
    <img id="lb-img" src="" alt="">
    <img id="lb-img-top" alt="" aria-hidden="true">
    <div id="lb-overlay" class="bbox-overlay"></div>
    <div id="lb-slider" aria-hidden="true">
      <div class="lb-slider-line"></div>
      <div class="lb-slider-handle">⇔</div>
    </div>
  </div>
</div>

<script>
const PERSONS = __PERSONS_JSON__;
const IMAGES = __IMAGES_JSON__;
const IMAGE_ORDER = __IMAGE_ORDER_JSON__;
let currentImageId = null;
let currentRecord = null;
let currentPair = null;       // {bw, color} when the image has a deoldified pair
let compareOn = false;        // slider compare mode active
let pairMode = 'color';       // single-view mode: 'color' | 'bw'
let currentSplit = 50;        // compare slider position (percent)

function clearOverlay(overlay){
  while(overlay.firstChild)overlay.removeChild(overlay.firstChild);
}

function setPercentStyle(el, prop, value){
  const n=Number(value);
  el.style[prop]=Number.isFinite(n)?n.toFixed(6)+'%':'0%';
}

function createFaceBox(face){
  const bbox=face.bbox||{};
  const box=document.createElement('div');
  box.className='bbox';
  box.tabIndex=0;
  box.setAttribute('role','button');
  const name=face.name||'Ismeretlen arc';
  box.title=name;
  box.setAttribute('aria-label','Arc: '+name);
  setPercentStyle(box,'left',bbox.left);
  setPercentStyle(box,'top',bbox.top);
  setPercentStyle(box,'width',bbox.width);
  setPercentStyle(box,'height',bbox.height);
  if(face.name){
    const label=document.createElement('div');
    label.className='bbox-label';
    if(Number(bbox.top)<6)label.classList.add('inside');
    label.textContent=face.name;
    box.appendChild(label);
  }
  return box;
}

function renderOverlay(record, overlay){
  clearOverlay(overlay);
  if(!record||!Array.isArray(record.faces))return;
  record.faces.forEach(face=>{
    overlay.appendChild(createFaceBox(face));
  });
}

function imageTitle(record){
  if(!record)return'';
  const parts=[];
  if(record.folder_name)parts.push(record.folder_name);
  if(record.file_name)parts.push(record.file_name);
  return parts.join(' / ');
}

function updatePairButtons(){
  const bw=document.getElementById('lb-bw');
  const color=document.getElementById('lb-color');
  const cmp=document.getElementById('lb-compare');
  if(bw)bw.classList.toggle('active',!compareOn&&pairMode==='bw');
  if(color)color.classList.toggle('active',!compareOn&&pairMode==='color');
  if(cmp)cmp.classList.toggle('active',compareOn);
}

function setCompareSplit(pct){
  currentSplit=Math.max(0,Math.min(100,pct));
  const top=document.getElementById('lb-img-top');
  top.style.clipPath='inset(0 '+(100-currentSplit).toFixed(3)+'% 0 0)';
  document.getElementById('lb-slider').style.left=currentSplit.toFixed(3)+'%';
}

function refreshPairMedia(){
  const media=document.getElementById('lb-media');
  const img=document.getElementById('lb-img');
  const top=document.getElementById('lb-img-top');
  if(!currentPair){
    media.classList.remove('compare');
    top.src='';
    if(currentRecord)img.src=currentRecord.src;
    return;
  }
  if(compareOn){
    media.classList.add('compare');
    top.src=currentPair.bw;        // B&W layer clipped on the left, color base on the right
    img.src=currentPair.color;
    setCompareSplit(currentSplit);
  }else{
    media.classList.remove('compare');
    top.src='';
    img.src=(pairMode==='bw')?currentPair.bw:currentPair.color;
  }
  updatePairButtons();
}

function setPairView(mode){
  if(!currentPair)return;
  compareOn=false;
  pairMode=mode;
  refreshPairMedia();
}

function toggleCompare(){
  if(!currentPair)return;
  compareOn=!compareOn;
  if(compareOn)currentSplit=50;
  refreshPairMedia();
}

function openLb(imageId){
  const record=IMAGES[imageId];
  if(!record)return;
  currentImageId=imageId;
  currentRecord=record;
  currentPair=record.pair||null;
  compareOn=false;
  pairMode='color';
  currentSplit=50;
  document.getElementById('lb-pair-controls').classList.toggle('show',!!currentPair);
  const img=document.getElementById('lb-img');
  const overlay=document.getElementById('lb-overlay');
  img.onload=()=>renderOverlay(record,overlay);
  img.alt=imageTitle(record);
  refreshPairMedia();
  if(!currentPair)img.src=record.src;
  document.getElementById('lb').classList.add('open');
  renderOverlay(record,overlay);
  updateLbNav();
}

function openThumb(src,title){
  currentImageId=null;
  currentRecord=null;
  currentPair=null;
  compareOn=false;
  document.getElementById('lb-pair-controls').classList.remove('show');
  document.getElementById('lb-media').classList.remove('compare');
  document.getElementById('lb-img-top').src='';
  const img=document.getElementById('lb-img');
  img.onload=null;
  img.src=src;
  img.alt=title||'';
  clearOverlay(document.getElementById('lb-overlay'));
  document.getElementById('lb').classList.add('open');
  setThumbNav(title||'Arc kivágás');
}

function closeLb(){
  document.getElementById('lb').classList.remove('open');
  document.getElementById('lb-media').classList.remove('compare');
  currentImageId=null;
  currentRecord=null;
  currentPair=null;
  compareOn=false;
}

(function initCompareDrag(){
  const media=document.getElementById('lb-media');
  let dragging=false;
  function pctFromEvent(e){
    const rect=media.getBoundingClientRect();
    if(!rect.width)return currentSplit;
    return ((e.clientX-rect.left)/rect.width)*100;
  }
  media.addEventListener('pointerdown',function(e){
    if(!compareOn)return;
    dragging=true;
    setCompareSplit(pctFromEvent(e));
    e.preventDefault();
  });
  window.addEventListener('pointermove',function(e){
    if(!compareOn||!dragging)return;
    setCompareSplit(pctFromEvent(e));
  });
  window.addEventListener('pointerup',function(){dragging=false;});
})();

function setThumbNav(title){
  document.getElementById('lb-prev').disabled=true;
  document.getElementById('lb-next').disabled=true;
  document.getElementById('lb-caption').textContent=title;
}

function updateLbNav(){
  const idx=IMAGE_ORDER.indexOf(currentImageId);
  const prev=document.getElementById('lb-prev');
  const next=document.getElementById('lb-next');
  if(idx<0){
    prev.disabled=true;
    next.disabled=true;
    return;
  }
  prev.disabled=idx===0;
  next.disabled=idx===IMAGE_ORDER.length-1;
  const record=IMAGES[currentImageId];
  const names=[...new Set((record.faces||[]).map(f=>f.name).filter(Boolean))];
  let caption=(idx+1)+' / '+IMAGE_ORDER.length;
  const title=imageTitle(record);
  if(title)caption+=' - '+title;
  if(names.length)caption+=' - '+names.join(', ');
  document.getElementById('lb-caption').textContent=caption;
}

function showRelativeImage(delta){
  const idx=IMAGE_ORDER.indexOf(currentImageId);
  const nextIdx=idx+delta;
  if(idx<0||nextIdx<0||nextIdx>=IMAGE_ORDER.length)return;
  openLb(IMAGE_ORDER[nextIdx]);
}

function openImageFromHash(){
  const hash=window.location.hash.startsWith('#')?window.location.hash.slice(1):'';
  const params=new URLSearchParams(hash);
  const imageId=params.get('image');
  if(imageId&&IMAGES[imageId])openLb(imageId);
}

document.getElementById('lb').addEventListener('click',function(e){
  if(e.target===this)closeLb();
});
document.addEventListener('keydown',function(e){
  const open=document.getElementById('lb').classList.contains('open');
  if(!open)return;
  if(e.key==='Escape')closeLb();
  if(e.key==='ArrowLeft'){
    e.preventDefault();
    showRelativeImage(-1);
  }
  if(e.key==='ArrowRight'){
    e.preventDefault();
    showRelativeImage(1);
  }
  if((e.key==='c'||e.key==='C')&&currentPair){
    e.preventDefault();
    toggleCompare();
  }
});

function createImageTile(imageId,personName){
  const rec=IMAGES[imageId];
  if(!rec)return null;
  const tile=document.createElement('div');
  tile.className='image-tile';
  tile.tabIndex=0;
  tile.title=imageTitle(rec)||personName||'';
  tile.onclick=()=>openLb(imageId);
  tile.addEventListener('keydown',e=>{
    if(e.key==='Enter'||e.key===' '){
      e.preventDefault();
      openLb(imageId);
    }
  });
  const img=document.createElement('img');
  img.src=rec.src;
  img.alt=imageTitle(rec)||personName||'';
  tile.appendChild(img);
  const overlay=document.createElement('div');
  overlay.className='bbox-overlay';
  renderOverlay(rec,overlay);
  tile.appendChild(overlay);
  return tile;
}

function buildCards(){
  const wrap=document.getElementById('persons');
  while(wrap.firstChild)wrap.removeChild(wrap.firstChild);
  PERSONS.forEach(p=>{
    const card=document.createElement('div');
    card.className='person-card';
    card.dataset.name=String(p.name||'').toLocaleLowerCase('hu-HU');

    const nameEl=document.createElement('div');
    nameEl.className='person-name';
    nameEl.textContent=p.name+' ('+p.images.length+' kép)';
    card.appendChild(nameEl);

    if(p.thumbs.length){
      const thumbs=document.createElement('div');
      thumbs.className='thumbs';
      p.thumbs.forEach(t=>{
        const img=document.createElement('img');
        img.src='thumbs/'+t;
        img.title=p.name;
        img.alt=p.name;
        img.onclick=()=>openThumb('thumbs/'+t,p.name);
        thumbs.appendChild(img);
      });
      card.appendChild(thumbs);
    }

    if(p.images.length){
      const lbl=document.createElement('div');
      lbl.className='images-label';
      lbl.textContent='Eredeti képek / Original photos:';
      card.appendChild(lbl);
      const strip=document.createElement('div');
      strip.className='img-strip';
      p.images.forEach(imageId=>{
        const tile=createImageTile(imageId,p.name);
        if(tile)strip.appendChild(tile);
      });
      card.appendChild(strip);
    }

    wrap.appendChild(card);
  });
  updateCount();
}

function filter(){
  const q=document.getElementById('search').value.toLocaleLowerCase('hu-HU').trim();
  document.querySelectorAll('.person-card').forEach(c=>{
    c.classList.toggle('hidden', q && !c.dataset.name.includes(q));
  });
  updateCount();
}

function updateCount(){
  const total=document.querySelectorAll('.person-card').length;
  const vis=document.querySelectorAll('.person-card:not(.hidden)').length;
  document.getElementById('count').textContent=vis+' / '+total+' személy';
}

buildCards();
openImageFromHash();
window.addEventListener('hashchange',openImageFromHash);
</script>
</body>
</html>
"""


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
let filtered = TOUR.slice();
let index = 0;
let playing = false;
let timer = null;
let speedMs = 4000;
let labelMode = 'full';
let map = null, marker = null, mapInit = false;

const $ = id => document.getElementById(id);
let tourSplit = 50;
const els = {
  count: $('tour-count'), stage: $('stage'), media: $('stage-media'),
  img: $('tour-img'), imgTop: $('tour-img-top'), slider: $('tour-slider'),
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

function setupPairMedia(record){
  const pair = record && record.pair;
  if(pair){
    els.media.classList.add('compare');
    els.imgTop.src = pair.bw;      // B&W layer clipped on the left, color base on the right
    setTourSplit(50);
  } else {
    els.media.classList.remove('compare');
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
    // Clicking a face opens the person panel — don't move the slider there.
    if(e.target.closest && e.target.closest('.tb')) return;
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

// Clicking anywhere outside a face hides any tapped-open frame/label.
document.addEventListener('click', e => {
  if(e.target.closest && e.target.closest('.tb')) return;
  document.querySelectorAll('.tb.show').forEach(b => b.classList.remove('show'));
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
