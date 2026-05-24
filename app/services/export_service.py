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
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import Collage, CollageNode, Face, Image, Person
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
        # image_path → list of (person, face) records. Face bounding boxes are
        # stored in original image pixels and transformed in the generated HTML.
        image_faces: Dict[str, List[Tuple[Person, Face]]] = {}
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
                if face.crop_path and Path(face.crop_path).exists():
                    dst_thumb = thumb_dir / f"face_{face.id}.jpg"
                    shutil.copy2(face.crop_path, dst_thumb)
                    person_thumbs[person.name].append(dst_thumb.name)

        # --- export normalized originals and HTML overlay metadata ---
        image_records: Dict[str, dict] = {}
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

            image_records[dst_name] = {
                "id": dst_name,
                "src": f"images/{dst_name}",
                "width": img_w,
                "height": img_h,
                "faces": face_records,
            }

        # --- build JS data ---
        js_persons = json.dumps(
            [
                {
                    "name": pname,
                    "thumbs": person_thumbs.get(pname, []),
                    "images": person_images.get(pname, []),
                }
                for pname in sorted(person_thumbs.keys())
            ],
            ensure_ascii=False,
        )
        js_images = json.dumps(image_records, ensure_ascii=False)

        # --- render HTML ---
        html = (
            _HTML_TEMPLATE
            .replace("__PERSONS_JSON__", _json_for_script(js_persons))
            .replace("__IMAGES_JSON__", _json_for_script(js_images))
        )
        (out / "index.html").write_text(html, encoding="utf-8")

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
                CollageNodeData, project_face_to_collage,
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

        js_data = json.dumps(collage_records, ensure_ascii=False, indent=1)
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
) -> Optional[Dict[str, int]]:
    """Normalize and clamp a face bbox to original image pixel coordinates.

    The database schema stores face boxes in original image pixels. The
    normalized-coordinate branch is defensive for old/manual imports that may
    have stored 0..1 values.
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
        "x": int(round(x1)),
        "y": int(round(y1)),
        "width": max(1, int(round(out_w))),
        "height": max(1, int(round(out_h))),
    }


def _append_unique(values: List[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _json_for_script(json_text: str) -> str:
    """Keep JSON valid inside an inline script tag."""
    return json_text.replace("</", "<\\/")


# ---------------------------------------------------------------------------
# Static HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Face Gallery</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#111;color:#ddd;font-family:system-ui,sans-serif}
  header{background:#1a1a1a;padding:16px 24px;border-bottom:1px solid #333;
         display:flex;align-items:center;gap:16px;flex-wrap:wrap}
  header h1{font-size:1.2rem;color:#88aaff;white-space:nowrap}
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
  .img-strip img{height:80px;border-radius:4px;border:1px solid #333;
                 cursor:pointer;transition:opacity .2s;object-fit:cover}
  .img-strip img:hover{opacity:.85;border-color:#88aaff}
  .media-wrap{position:relative;display:inline-block;line-height:0}
  .media-wrap>img{display:block}
  .bbox-overlay{position:absolute;inset:0;pointer-events:none}
  .bbox{position:absolute;outline:2px solid rgba(50,220,50,.95);
        background:rgba(50,220,50,.06);box-shadow:0 0 0 1px rgba(0,0,0,.45)}
  .bbox-label{position:absolute;left:0;top:0;transform:translateY(calc(-100% - 4px));
              max-width:min(280px,70vw);overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap;background:rgba(12,18,12,.9);color:#75ff75;
              border:1px solid rgba(50,220,50,.85);border-radius:4px;
              padding:2px 6px;font-size:12px;font-weight:700;line-height:1.25}
  .bbox-label.inside{transform:none;top:2px;left:2px}

  /* lightbox */
  #lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
      z-index:100;align-items:center;justify-content:center;flex-direction:column;gap:12px}
  #lb.open{display:flex}
  #lb-media{max-width:92vw;max-height:86vh;border-radius:6px;
            box-shadow:0 0 0 2px #88aaff;background:#050505}
  #lb-media img{max-width:92vw;max-height:86vh;width:auto;height:auto;border-radius:6px}
  #lb-close{position:fixed;top:16px;right:20px;font-size:2rem;cursor:pointer;
             color:#aaa;line-height:1;background:none;border:none}
  #lb-close:hover{color:#fff}
</style>
</head>
<body>
<header>
  <h1>Face Gallery</h1>
  <input id="search" type="text" placeholder="Keresés / Search…" oninput="filter()">
  <span id="count"></span>
</header>
<div id="persons"></div>

<!-- lightbox -->
<div id="lb">
  <button id="lb-close" onclick="closeLb()">✕</button>
  <div id="lb-media" class="media-wrap">
    <img id="lb-img" src="" alt="">
    <div id="lb-overlay" class="bbox-overlay"></div>
  </div>
</div>

<script>
const PERSONS = __PERSONS_JSON__;
const IMAGES = __IMAGES_JSON__;
let currentImage = null;

function transformBox(box, originalSize, display){
  const sourceW=Number(originalSize.width), sourceH=Number(originalSize.height);
  const displayW=Number(display.width), displayH=Number(display.height);
  if(!sourceW||!sourceH||!displayW||!displayH)return null;
  const scaleX=displayW/sourceW;
  const scaleY=displayH/sourceH;
  return {
    left:display.offsetX+Number(box.x)*scaleX,
    top:display.offsetY+Number(box.y)*scaleY,
    width:Number(box.width)*scaleX,
    height:Number(box.height)*scaleY
  };
}

function displayedImageBox(imgEl, wrapperEl, originalSize){
  const imgRect=imgEl.getBoundingClientRect();
  const wrapRect=wrapperEl.getBoundingClientRect();
  let width=imgRect.width, height=imgRect.height;
  let offsetX=imgRect.left-wrapRect.left;
  let offsetY=imgRect.top-wrapRect.top;
  const sourceW=Number(originalSize.width), sourceH=Number(originalSize.height);
  const fit=getComputedStyle(imgEl).objectFit;
  if(sourceW&&sourceH&&width&&height&&(fit==='contain'||fit==='scale-down'||fit==='cover')){
    const scale=fit==='cover'
      ? Math.max(width/sourceW,height/sourceH)
      : Math.min(width/sourceW,height/sourceH);
    const contentW=sourceW*scale;
    const contentH=sourceH*scale;
    offsetX+=(width-contentW)/2;
    offsetY+=(height-contentH)/2;
    width=contentW;
    height=contentH;
  }
  return {width,height,offsetX,offsetY};
}

function renderOverlay(record){
  const imgEl=document.getElementById('lb-img');
  const overlay=document.getElementById('lb-overlay');
  const wrapper=document.getElementById('lb-media');
  overlay.innerHTML='';
  if(!record||!imgEl.complete||!imgEl.naturalWidth)return;
  const display=displayedImageBox(imgEl,wrapper,{width:record.width,height:record.height});
  record.faces.forEach(face=>{
    const rect=transformBox(face.bbox,{width:record.width,height:record.height},display);
    if(!rect||rect.width<=0||rect.height<=0)return;
    const box=document.createElement('div');
    box.className='bbox';
    box.style.left=rect.left.toFixed(2)+'px';
    box.style.top=rect.top.toFixed(2)+'px';
    box.style.width=rect.width.toFixed(2)+'px';
    box.style.height=rect.height.toFixed(2)+'px';
    if(face.name){
      const label=document.createElement('div');
      label.className='bbox-label';
      if(rect.top<24)label.classList.add('inside');
      label.textContent=face.name;
      box.appendChild(label);
    }
    overlay.appendChild(box);
  });
}

function openLb(imageId){
  const record=IMAGES[imageId];
  if(!record)return;
  currentImage=record;
  const img=document.getElementById('lb-img');
  img.onload=()=>renderOverlay(record);
  img.src=record.src;
  document.getElementById('lb').classList.add('open');
  if(img.complete)requestAnimationFrame(()=>renderOverlay(record));
}

function openThumb(src){
  currentImage=null;
  document.getElementById('lb-overlay').innerHTML='';
  document.getElementById('lb-img').src=src;
  document.getElementById('lb').classList.add('open');
}
function closeLb(){
  document.getElementById('lb').classList.remove('open');
  currentImage=null;
}
document.getElementById('lb').addEventListener('click',function(e){
  if(e.target===this)closeLb();
});
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeLb();});
window.addEventListener('resize',function(){
  if(currentImage&&document.getElementById('lb').classList.contains('open')){
    requestAnimationFrame(()=>renderOverlay(currentImage));
  }
});

function buildCards(){
  const wrap=document.getElementById('persons');
  wrap.innerHTML='';
  PERSONS.forEach(p=>{
    const card=document.createElement('div');
    card.className='person-card';
    card.dataset.name=p.name.toLowerCase();

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
        img.onclick=()=>openThumb('thumbs/'+t);
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
        const rec=IMAGES[imageId];
        if(!rec)return;
        const img=document.createElement('img');
        img.src=rec.src;
        img.title=p.name;
        img.onclick=()=>openLb(imageId);
        strip.appendChild(img);
      });
      card.appendChild(strip);
    }

    wrap.appendChild(card);
  });
  updateCount();
}

function filter(){
  const q=document.getElementById('search').value.toLowerCase().trim();
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
</script>
</body>
</html>
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
<meta charset="UTF-8">
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
  let h='<div class="tip-title">'+(node.src||'\u2014')+'</div>';
  if(node.missing)h+='<div class="tip-warn">\u26a0 Forr\u00e1sf\u00e1jl hi\u00e1nyzik</div>';
  if(node.year)h+='<div><b>\u00c9v:</b> '+node.year+'</div>';
  if(node.location)h+='<div><b>Helysz\u00edn:</b> '+node.location+'</div>';
  if(node.event)h+='<div><b>Esem\u00e9ny:</b> '+node.event+'</div>';
  if(node.notes)h+='<div><b>Megjegyz\u00e9s:</b> '+node.notes+'</div>';
  if(node.faces&&node.faces.length){
    const names=[...new Set(node.faces.map(f=>f.name).filter(Boolean))];
    if(names.length)h+='<div><b>Szem\u00e9lyek:</b> '+names.join(', ')+'</div>';
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
    hdr.innerHTML='<div class="collage-title">'+col.title+'</div>'
                 +(col.date?'<div class="collage-date">'+col.date+'</div>':'');
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
