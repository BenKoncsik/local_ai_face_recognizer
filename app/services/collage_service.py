"""Collage service.

Handles importing Picasa collage XML files into the database,
linking nodes to existing Image records, projecting detected faces
back onto the collage canvas, and generating a modified collage
image where labelled-person nodes are annotated.
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

import cv2

from sqlalchemy.orm import Session

from app.db.models import Collage, CollageNode, Face, Image, Person
from app.services.collage_parser import (
    CollageData,
    CollageNodeData,
    parse_collage_file,
    project_face_to_collage,
)
from app.utils.image_utils import load_image_bgr, save_image_bgr

log = logging.getLogger(__name__)

# Default canvas height used when rendering a collage preview
_DEFAULT_RENDER_HEIGHT = 1000


class CollageService:
    """Business logic for Picasa collage import and management.

    Args:
        session: Active SQLAlchemy session.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_collage(
        self,
        file_path: str | Path,
        search_roots: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> Collage:
        """Import (or re-import) a collage file.

        Args:
            file_path:    Path to the .cxf/.cfx file.
            search_roots: Extra directories to search when resolving
                          Windows-style source paths.
            overwrite:    If ``True``, delete the existing record and
                          re-import.  Otherwise return the existing record.

        Returns:
            The persisted :class:`Collage` ORM object.
        """
        file_path = Path(file_path).resolve()
        src_str = str(file_path)

        existing = (
            self._session.query(Collage)
            .filter(Collage.source_file == src_str)
            .first()
        )
        if existing and not overwrite:
            log.info("Collage already imported (id=%d): %s", existing.id, src_str)
            return existing
        if existing and overwrite:
            self._session.delete(existing)
            self._session.flush()

        data: CollageData = parse_collage_file(file_path, search_roots)
        collage = self._persist_collage(data)
        self._session.flush()

        # Try to link nodes to existing Image records
        self._link_images(collage)
        self._session.commit()

        log.info(
            "Collage imported: id=%d, %d node(s), %d missing",
            collage.id,
            len(collage.nodes),
            sum(1 for n in collage.nodes if n.src_missing),
        )
        return collage

    def _persist_collage(self, data: CollageData) -> Collage:
        collage = Collage(
            source_file=data.source_file,
            collage_uid=data.collage_uid,
            album_title=data.album_title,
            album_date=data.album_date,
            format_width=data.format_width,
            format_height=data.format_height,
            orientation=data.orientation,
            bg_color=data.bg_color,
            spacing=data.spacing,
        )
        self._session.add(collage)
        self._session.flush()  # get collage.id

        for nd in data.nodes:
            node = CollageNode(
                collage_id=collage.id,
                node_uid=nd.node_uid,
                rel_x=nd.rel_x,
                rel_y=nd.rel_y,
                rel_w=nd.rel_w,
                rel_h=nd.rel_h,
                theta=nd.theta,
                scale=nd.scale,
                theme=nd.theme,
                src_raw=nd.src_raw,
                src_resolved=nd.src_resolved,
                src_missing=nd.src_missing,
            )
            # Attempt pre-fill of metadata from filename
            if nd.src_resolved:
                node.year, node.location, node.event_name = _extract_metadata_from_filename(
                    Path(nd.src_resolved).name
                )
            self._session.add(node)

        return collage

    # ------------------------------------------------------------------
    # Linking to Image records
    # ------------------------------------------------------------------

    def _link_images(self, collage: Collage) -> int:
        """Link CollageNode records to existing Image rows by resolved path.

        Returns:
            Number of nodes linked.
        """
        linked = 0
        for node in collage.nodes:
            if node.src_missing or not node.src_resolved:
                continue
            image = (
                self._session.query(Image)
                .filter(Image.file_path == node.src_resolved)
                .first()
            )
            if image:
                node.image_id = image.id
                linked += 1
        log.debug("Linked %d/%d nodes to Image records", linked, len(collage.nodes))
        return linked

    def relink_images(self, collage_id: int) -> int:
        """Re-run image linking for a collage (useful after scanning new images)."""
        collage = self._session.get(Collage, collage_id)
        if not collage:
            raise ValueError(f"Collage id={collage_id} not found")
        linked = self._link_images(collage)
        self._session.commit()
        return linked

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_collages(self) -> List[Collage]:
        return self._session.query(Collage).order_by(Collage.album_title).all()

    def get_collage(self, collage_id: int) -> Optional[Collage]:
        return self._session.get(Collage, collage_id)

    def get_nodes(self, collage_id: int) -> List[CollageNode]:
        return (
            self._session.query(CollageNode)
            .filter(CollageNode.collage_id == collage_id)
            .order_by(CollageNode.id)
            .all()
        )

    def update_node_metadata(
        self,
        node_id: int,
        *,
        year: Optional[str] = None,
        location: Optional[str] = None,
        event_name: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> CollageNode:
        """Update user-editable metadata fields on a node."""
        node = self._session.get(CollageNode, node_id)
        if not node:
            raise ValueError(f"CollageNode id={node_id} not found")
        if year is not None:
            node.year = year or None
        if location is not None:
            node.location = location or None
        if event_name is not None:
            node.event_name = event_name or None
        if notes is not None:
            node.notes = notes or None
        self._session.commit()
        return node

    # ------------------------------------------------------------------
    # Face projection
    # ------------------------------------------------------------------

    def get_faces_for_node(self, node: CollageNode) -> List[Face]:
        """Return all Face records for the source image of *node*."""
        if node.image_id is None:
            return []
        return (
            self._session.query(Face)
            .filter(Face.image_id == node.image_id)
            .all()
        )

    def projected_faces(
        self,
        collage: Collage,
        render_w: int,
        render_h: int,
    ) -> List[Dict]:
        """Return face projections for all nodes of *collage*.

        Each entry::

            {
              "node_id": int,
              "face_id": int,
              "person_name": str | None,
              "person_id": int | None,
              "bbox_collage": (px, py, pw, ph),   # in render_w x render_h space
              "theta": float,                      # node rotation (radians)
              "partial_rotation": bool,            # True if theta != 0 (approx only)
            }

        Args:
            collage:  Collage ORM object (nodes must be loaded).
            render_w: Width of the render canvas (pixels).
            render_h: Height of the render canvas (pixels).
        """
        results = []
        for node in collage.nodes:
            if node.src_missing or node.image_id is None:
                continue
            image = self._session.get(Image, node.image_id)
            if not image or not image.width or not image.height:
                continue

            nd = CollageNodeData(
                rel_x=node.rel_x, rel_y=node.rel_y,
                rel_w=node.rel_w, rel_h=node.rel_h,
                theta=node.theta, scale=node.scale,
            )

            faces = self.get_faces_for_node(node)
            for face in faces:
                bbox_collage = project_face_to_collage(
                    (face.bbox_x, face.bbox_y, face.bbox_w, face.bbox_h),
                    image.width, image.height,
                    nd,
                    render_w, render_h,
                )
                if bbox_collage is None:
                    continue
                person = self._session.get(Person, face.person_id) if face.person_id else None
                results.append({
                    "node_id": node.id,
                    "face_id": face.id,
                    "person_name": person.name if person else None,
                    "person_id": face.person_id,
                    "bbox_collage": bbox_collage,
                    "theta": node.theta,
                    "partial_rotation": abs(node.theta) > 0.001,
                })
        return results

    # ------------------------------------------------------------------
    # Render collage image
    # ------------------------------------------------------------------

    def render_collage_image(
        self,
        collage: Collage,
        render_h: int = _DEFAULT_RENDER_HEIGHT,
        draw_borders: bool = True,
        draw_faces: bool = False,
    ):
        """Render the collage as a numpy (BGR) image.

        Missing source images are replaced by a grey placeholder.

        Args:
            collage:      Collage to render.
            render_h:     Canvas height in pixels.
            draw_borders: Draw white borders around each node.
            draw_faces:   Overlay face bounding boxes (projected from source).

        Returns:
            A ``numpy.ndarray`` (H, W, 3) or ``None`` if dimensions unknown.
        """
        import numpy as np

        cw = collage.format_width or 2858
        ch = collage.format_height or 1000
        scale = render_h / ch
        render_w = int(cw * scale)

        canvas = np.full((render_h, render_w, 3), 40, dtype=np.uint8)

        for node in collage.nodes:
            px = int(node.rel_x * render_w)
            py = int(node.rel_y * render_h)
            pw = max(int(node.rel_w * render_w), 1)
            ph = max(int(node.rel_h * render_h), 1)

            cell = np.full((ph, pw, 3), 70, dtype=np.uint8)

            src_path = node.src_resolved
            if src_path and Path(src_path).exists():
                img = load_image_bgr(src_path)
                if img is not None:
                    cell = _fit_cover(img, pw, ph, node.scale)

            # Paste cell into canvas
            x1, y1 = px, py
            x2, y2 = min(px + pw, render_w), min(py + ph, render_h)
            cw2 = x2 - x1
            ch2 = y2 - y1
            if cw2 > 0 and ch2 > 0:
                canvas[y1:y2, x1:x2] = cell[:ch2, :cw2]

            if draw_borders:
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (255, 255, 255), 1)

            if node.src_missing:
                # Draw a red cross over missing nodes
                cv2.line(canvas, (px, py), (px + pw, py + ph), (60, 60, 200), 2)
                cv2.line(canvas, (px + pw, py), (px, py + ph), (60, 60, 200), 2)

        if draw_faces:
            face_data = self.projected_faces(collage, render_w, render_h)
            for fd in face_data:
                fx, fy, fw, fh = fd["bbox_collage"]
                color = (50, 220, 50)
                cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), color, 2)
                if fd["person_name"]:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale_f = max(0.35, min(0.8, fw / 80))
                    cv2.putText(
                        canvas, fd["person_name"],
                        (fx, max(fy - 4, 12)),
                        font, scale_f, color, 1, cv2.LINE_AA,
                    )

        return canvas

    # ------------------------------------------------------------------
    # Modified collage export (annotated CXF with labelled nodes)
    # ------------------------------------------------------------------

    def export_annotated_collage(
        self,
        collage: Collage,
        output_dir: str | Path,
        render_h: int = _DEFAULT_RENDER_HEIGHT,
    ) -> Path:
        """Write a modified collage image (JPEG) and an updated .cxf file.

        The exported image shows only those nodes whose source images
        have at least one identified person.  Missing source images are
        replaced with a grey placeholder.  A companion .cxf file
        lists only the annotated nodes.

        Args:
            collage:    Collage to export.
            output_dir: Target directory (created if absent).
            render_h:   Height of the output image in pixels.

        Returns:
            Path to the exported JPEG file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        canvas = self.render_collage_image(
            collage, render_h=render_h, draw_borders=True, draw_faces=True
        )

        safe_title = _safe_filename(collage.album_title or f"collage_{collage.id}")
        jpg_path = out / f"{safe_title}_annotated.jpg"
        save_image_bgr(jpg_path, canvas)

        # Write modified CXF listing only nodes with recognised persons
        self._write_annotated_cxf(collage, out / f"{safe_title}_annotated.cxf")

        log.info("Annotated collage exported: %s", jpg_path)
        return jpg_path

    def _write_annotated_cxf(self, collage: Collage, out_path: Path) -> None:
        """Write a minimal .cxf with only person-labelled nodes."""
        root = ET.Element("collage", {
            "version": "2",
            "format": f"{collage.format_width or 2858}:{collage.format_height or 1000}",
            "orientation": collage.orientation or "landscape",
            "theme": "picturegrid",
            "albumUID": collage.collage_uid or "",
        })

        if collage.album_title:
            el = ET.SubElement(root, "albumTitle")
            el.text = collage.album_title
        if collage.album_date:
            el = ET.SubElement(root, "albumDate")
            el.text = collage.album_date

        for node in collage.nodes:
            faces = self.get_faces_for_node(node)
            identified = [f for f in faces if f.person_id is not None]
            if not identified:
                continue

            node_el = ET.SubElement(root, "node", {
                "x": f"{node.rel_x:.6f}",
                "y": f"{node.rel_y:.6f}",
                "w": f"{node.rel_w:.6f}",
                "h": f"{node.rel_h:.6f}",
                "theta": f"{node.theta:.6f}",
                "scale": f"{node.scale:.6f}",
            })
            if node.theme:
                th = ET.SubElement(node_el, "theme")
                th.text = node.theme
            if node.src_raw:
                src = ET.SubElement(node_el, "src")
                src.text = node.src_raw
            if node.node_uid:
                uid = ET.SubElement(node_el, "uid")
                uid.text = node.node_uid

            # Embed persons as comment
            names = ", ".join(
                p.name
                for f in identified
                if f.person_id and (p := self._session.get(Person, f.person_id))
            )
            if names:
                node_el.set("persons", names)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
        log.debug("Annotated CXF written: %s", out_path)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _fit_cover(img, target_w: int, target_h: int, scale_param: float = 100.0):
    """Scale and center-crop *img* to fill (target_w × target_h).

    Mirrors Picasa's cover-fill + zoom logic.
    """
    import numpy as np

    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0 or target_w <= 0 or target_h <= 0:
        return np.full((target_h, target_w, 3), 70, dtype=np.uint8)

    # Cover scale then apply zoom
    cover = max(target_w / iw, target_h / ih)
    zoom = scale_param / 100.0
    eff = cover * zoom

    new_w = max(int(iw * eff), 1)
    new_h = max(int(ih * eff), 1)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    cx = (new_w - target_w) // 2
    cy = (new_h - target_h) // 2
    cx = max(cx, 0)
    cy = max(cy, 0)

    cropped = resized[cy:cy + target_h, cx:cx + target_w]

    # Pad if needed
    ch, cw = cropped.shape[:2]
    if ch < target_h or cw < target_w:
        out = np.full((target_h, target_w, 3), 70, dtype=np.uint8)
        out[:ch, :cw] = cropped
        return out

    return cropped


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe version of *name*."""
    import re
    safe = re.sub(r'[\\/:*?"<>|]', "_", name)
    return safe[:120] or "collage"


def _extract_metadata_from_filename(filename: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Heuristically extract year, location, event from a filename.

    Returns:
        (year, location, event_name) — any element may be None.

    Examples:
        ``"1969 [02 koffer] foto.jpg"``  → year="1969", others None
        ``"2005 Balaton nyaralas.jpg"``  → year="2005", others None
    """
    import re

    year: Optional[str] = None
    location: Optional[str] = None
    event_name: Optional[str] = None

    # Year: 4-digit number starting with 19xx or 20xx
    m = re.search(r"\b(19\d{2}|20\d{2})\b", filename)
    if m:
        year = m.group(1)

    return year, location, event_name
