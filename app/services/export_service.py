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
            "person_id", "person_name", "face_id",
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
            records.append(
                {
                    "person_id": person.id,
                    "person_name": person.name,
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
        image_records: Dict[str, dict] = {}
        map_records: List[dict] = []
        image_order: List[str] = []
        for img_path, face_list in image_faces.items():
            img = load_image_bgr(img_path)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            dst_name = _image_export_filename(img_path)
            if not save_image_bgr(img_dir / dst_name, img):
                continue

            face_records = []
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
            map_record = _build_map_export_record(db_image, dst_name, source, face_records)
            if map_record is not None:
                map_records.append(map_record)
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
        persons = self._get_persons(person_id)
        rows = []
        for person in persons:
            for face in self._get_faces(person.id):
                rows.append(
                    {
                        "person_id": person.id,
                        "person_name": person.name,
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
</style>
</head>
<body>
<header>
  <h1>Face Gallery</h1>
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
  <div id="lb-media" class="media-wrap">
    <img id="lb-img" src="" alt="">
    <div id="lb-overlay" class="bbox-overlay"></div>
  </div>
</div>

<script>
const PERSONS = __PERSONS_JSON__;
const IMAGES = __IMAGES_JSON__;
const IMAGE_ORDER = __IMAGE_ORDER_JSON__;
let currentImageId = null;

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

function openLb(imageId){
  const record=IMAGES[imageId];
  if(!record)return;
  currentImageId=imageId;
  const img=document.getElementById('lb-img');
  const overlay=document.getElementById('lb-overlay');
  img.onload=()=>renderOverlay(record,overlay);
  img.alt=imageTitle(record);
  img.src=record.src;
  document.getElementById('lb').classList.add('open');
  renderOverlay(record,overlay);
  updateLbNav();
}

function openThumb(src,title){
  currentImageId=null;
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
  currentImageId=null;
}

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


def _safe_filename(name: str) -> str:
    import re
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
