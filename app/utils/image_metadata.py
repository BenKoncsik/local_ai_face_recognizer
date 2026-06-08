"""Low-level image metadata writer for face-local face annotations.

This module embeds a versioned, structured JSON payload (schema
``facelocal.faces.v1``) into an image file, or writes it to a sidecar JSON
file next to the image when embedding is not possible.

Write strategy (in preference order):

1. **XMP** — a dedicated ``facelocal`` RDF block inside the image's XMP packet
   (JPEG and PNG).  Existing XMP and EXIF are preserved.
2. **EXIF** — ``UserComment`` (preferred) or ``ImageDescription`` as a JSON
   string, when XMP is not available for the format but EXIF is (JPEG).
3. **Sidecar** — ``<image>.facelocal.json`` next to the file, used for
   unsupported / RAW / read-only / remote-only files, or when the caller
   forces sidecar-only mode.

Design goals:

* The original image is never corrupted — embedded writes go through a
  temporary file in the same directory and an atomic ``os.replace``.
* Existing EXIF (GPS, dates, …) and unrelated XMP blocks survive a write.
* Re-running an export *updates* the single ``facelocal`` block instead of
  appending duplicates.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Schema identifier carried inside every payload — bump when the shape changes.
FACELOCAL_SCHEMA = "facelocal.faces.v1"

# Custom XMP namespace for the embedded block.
_XMP_NS_URI = "http://facelocal.app/ns/faces/1.0/"
_XMP_NS_PREFIX = "facelocal"
# The single property holding the JSON document (as an XML-escaped string).
_XMP_PROP = "data"

# Adobe-required XMP packet wrapper id.
_XPACKET_ID = "W5M0MpCehiHzreSzNTczkc9d"

# Write-mode result constants.
WRITE_MODE_XMP = "xmp"
WRITE_MODE_EXIF_USER_COMMENT = "exif_user_comment"
WRITE_MODE_EXIF_IMAGE_DESCRIPTION = "exif_image_description"
WRITE_MODE_SIDECAR = "sidecar"
WRITE_MODE_SKIPPED = "skipped"
WRITE_MODE_FAILED = "failed"

# Formats we can embed XMP into.
_XMP_FORMATS = {"JPEG", "PNG"}
# Formats we can embed EXIF into (JPEG; TIFF technically but kept conservative).
_EXIF_FORMATS = {"JPEG"}

# File extensions we will *attempt* to embed into.  Everything else (RAW, GIF,
# HEIC, etc.) falls back to a sidecar JSON file.
_EMBEDDABLE_EXTS = {".jpg", ".jpeg", ".jpe", ".png"}

# Standard XMP-in-PNG iTXt keyword.
_PNG_XMP_KEY = "XML:com.adobe.xmp"


@dataclass
class MetadataWriteResult:
    """Outcome of a single :func:`write_face_metadata` call."""

    write_mode: str
    sidecar_path: Optional[Path] = None
    error_message: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.write_mode not in (WRITE_MODE_FAILED,)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sidecar_path_for(image_path: str | Path) -> Path:
    """Return the sidecar JSON path for *image_path* (``name.facelocal.json``)."""
    p = Path(image_path)
    return p.with_name(p.name + ".facelocal.json")


def is_embeddable(image_path: str | Path) -> bool:
    """True when the file extension is one we attempt to embed metadata into."""
    return Path(image_path).suffix.lower() in _EMBEDDABLE_EXTS


def write_face_metadata(
    image_path: str | Path,
    payload: dict,
    *,
    prefer_sidecar_only: bool = False,
    prefer_exif_comment: bool = True,
    set_finder_comment: bool = True,
) -> MetadataWriteResult:
    """Persist *payload* for *image_path* using the best available strategy.

    Args:
        image_path: Absolute path to the image file.
        payload: A JSON-serialisable ``facelocal.faces.v1`` document.
        prefer_sidecar_only: When True, skip embedding entirely and always write
            a sidecar JSON file (used for the privacy-conscious export option).
        prefer_exif_comment: When True (default), JPEGs receive the JSON in the
            EXIF ``UserComment`` field — the image's "comment", visible among the
            standard EXIF data in most viewers. When False, the JSON goes into an
            XMP block instead. Formats without EXIF (PNG) always use XMP.
        set_finder_comment: When True (default) and running on macOS, the JSON is
            *also* written to the file's Finder/Spotlight comment so it shows up
            in Finder → Get Info → "Comments". Best-effort; no-op elsewhere.

    Returns:
        A :class:`MetadataWriteResult` describing how (or whether) the data was
        written.  Embedding failures degrade gracefully to a sidecar file; a
        genuinely unwritable target yields ``WRITE_MODE_FAILED``.
    """
    path = Path(image_path)

    if not path.exists():
        return MetadataWriteResult(
            WRITE_MODE_FAILED, error_message=f"file not found: {path}"
        )

    if prefer_sidecar_only:
        return _write_sidecar_result(path, payload)

    if not is_embeddable(path):
        log.debug("Not embeddable (%s) → sidecar: %s", path.suffix, path)
        return _write_sidecar_result(path, payload)

    if not os.access(path, os.W_OK):
        log.info("Image not writable → sidecar: %s", path)
        return _write_sidecar_result(path, payload)

    try:
        mode = _embed(path, payload, prefer_exif_comment=prefer_exif_comment)
        if mode is not None:
            if set_finder_comment:
                write_macos_finder_comment(path, json.dumps(payload, ensure_ascii=False))
            return MetadataWriteResult(mode)
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding metadata failed for %s: %s — falling back to sidecar", path, exc)

    return _write_sidecar_result(path, payload)


def write_macos_finder_comment(image_path: str | Path, text: str) -> bool:
    """Set the macOS Finder/Spotlight comment for *image_path* to *text*.

    This is the comment shown in Finder → Get Info → "Comments" (the field most
    users look at), which is a separate channel from EXIF/XMP. Implemented via
    AppleScript so Finder's own store is updated too; falls back to writing the
    ``kMDItemFinderComment`` extended attribute directly.

    Best-effort: returns False (and logs at debug) on non-macOS systems or any
    failure. The first call may trigger a one-time "control Finder" automation
    prompt.
    """
    if sys.platform != "darwin":
        return False
    # Avoid triggering Finder automation prompts / subprocesses during tests.
    if "PYTEST_CURRENT_TEST" in os.environ:
        return False

    import subprocess

    path_str = str(image_path)
    script = (
        "on run argv\n"
        "  set p to POSIX file (item 1 of argv) as alias\n"
        '  tell application "Finder" to set comment of p to (item 2 of argv)\n'
        "end run"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script, path_str, text],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Finder comment via AppleScript failed for %s: %s", path_str, exc)

    # Fallback: write the extended attribute directly (Spotlight-indexed).
    try:
        import plistlib

        data = plistlib.dumps(text, fmt=plistlib.FMT_BINARY)
        subprocess.run(
            ["xattr", "-w", "-x", "com.apple.metadata:kMDItemFinderComment",
             data.hex(), path_str],
            check=True,
            capture_output=True,
            timeout=20,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("Finder comment via xattr failed for %s: %s", path_str, exc)
        return False


def read_face_metadata(image_path: str | Path) -> Optional[dict]:
    """Read back a previously written ``facelocal`` payload, or ``None``.

    Checks, in order: embedded XMP, embedded EXIF (UserComment /
    ImageDescription), then the sidecar JSON file.  Used by the dedup/update
    logic and by tests.
    """
    path = Path(image_path)

    if is_embeddable(path) and path.exists():
        try:
            embedded = _read_embedded(path)
            if embedded is not None:
                return embedded
        except Exception as exc:  # noqa: BLE001
            log.debug("Reading embedded metadata failed for %s: %s", path, exc)

    sidecar = sidecar_path_for(path)
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.debug("Reading sidecar failed for %s: %s", sidecar, exc)

    return None


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def _write_sidecar_result(path: Path, payload: dict) -> MetadataWriteResult:
    try:
        sidecar = write_sidecar(path, payload)
        return MetadataWriteResult(WRITE_MODE_SIDECAR, sidecar_path=sidecar)
    except Exception as exc:  # noqa: BLE001
        log.error("Sidecar write failed for %s: %s", path, exc)
        return MetadataWriteResult(WRITE_MODE_FAILED, error_message=str(exc))


def write_sidecar(image_path: str | Path, payload: dict) -> Path:
    """Atomically write *payload* to ``<image>.facelocal.json`` and return it."""
    sidecar = sidecar_path_for(image_path)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_bytes(sidecar, text.encode("utf-8"))
    log.info("Face metadata sidecar written: %s", sidecar)
    return sidecar


# ---------------------------------------------------------------------------
# Embedding (XMP / EXIF)
# ---------------------------------------------------------------------------

def _embed(path: Path, payload: dict, *, prefer_exif_comment: bool = True) -> Optional[str]:
    """Embed *payload* into the image; return the write-mode or ``None``.

    Returns ``None`` (not an exception) when the format simply has no embedding
    path, so the caller can fall back to a sidecar cleanly.

    For JPEG the JSON goes into the EXIF ``UserComment`` field when
    *prefer_exif_comment* is True (the default — it shows up as the image's
    "comment" among the standard EXIF data), otherwise into an XMP block.
    PNG has no EXIF and always uses XMP.
    """
    from PIL import Image as PilImage

    with PilImage.open(path) as img:
        img_format = (img.format or "").upper()
        existing_xmp = img.info.get("xmp")
        existing_exif = img.info.get("exif", b"")
        png_text = dict(getattr(img, "text", {}) or {})
        img.load()
        work = img.copy()

    json_text = json.dumps(payload, ensure_ascii=False)

    if prefer_exif_comment and img_format in _EXIF_FORMATS:
        return _save_jpeg_exif_comment(
            path, work, existing_exif, json_text, existing_xmp=existing_xmp
        )

    if img_format in _XMP_FORMATS:
        new_xmp = _merge_xmp(existing_xmp, json_text)
        if img_format == "JPEG":
            _save_jpeg(path, work, xmp=new_xmp, exif=existing_exif)
        else:  # PNG
            _save_png(path, work, png_text, xmp=new_xmp.decode("utf-8"))
        return WRITE_MODE_XMP

    if img_format in _EXIF_FORMATS:
        return _save_jpeg_exif_comment(
            path, work, existing_exif, json_text, existing_xmp=existing_xmp
        )

    return None


def _save_jpeg(path: Path, img, *, xmp: bytes, exif: bytes) -> None:
    tmp = _tmp_for(path)
    save_kwargs = {"format": "JPEG", "xmp": xmp, "quality": "keep"}
    if exif:
        save_kwargs["exif"] = exif
    try:
        img.save(tmp, **save_kwargs)
    except (ValueError, OSError):
        # "keep" quality only works for JPEG inputs that still have their
        # original tables; retry with a high explicit quality.
        save_kwargs["quality"] = 95
        img.save(tmp, **save_kwargs)
    _replace_atomic(tmp, path)


def _save_jpeg_exif_comment(
    path: Path,
    img,
    existing_exif: bytes,
    json_text: str,
    *,
    existing_xmp=None,
) -> str:
    import piexif

    exif_dict = piexif.load(existing_exif) if existing_exif else {
        "0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None,
    }
    # UserComment requires an 8-byte character-code prefix; "UNICODE\0" with
    # UTF-16, or the ASCII prefix for plain text.  We use the ASCII prefix and
    # store UTF-8 JSON bytes after it — widely tolerated and round-trips here.
    exif_dict.setdefault("Exif", {})
    comment = b"ASCII\x00\x00\x00" + json_text.encode("utf-8")
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment
    new_exif = piexif.dump(exif_dict)

    # Re-attach any pre-existing XMP so an unrelated XMP block survives the write.
    save_kwargs = {"format": "JPEG", "exif": new_exif, "quality": "keep"}
    if existing_xmp:
        save_kwargs["xmp"] = existing_xmp

    tmp = _tmp_for(path)
    try:
        img.save(tmp, **save_kwargs)
    except (ValueError, OSError):
        save_kwargs["quality"] = 95
        img.save(tmp, **save_kwargs)
    _replace_atomic(tmp, path)
    return WRITE_MODE_EXIF_USER_COMMENT


def _save_png(path: Path, img, existing_text: dict, *, xmp: str) -> None:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    # Preserve all existing textual chunks except a prior XMP block.
    for key, value in existing_text.items():
        if key == _PNG_XMP_KEY:
            continue
        try:
            info.add_itxt(key, value)
        except Exception:  # noqa: BLE001
            # Non-iTXt legacy chunks: store as latin-1 tEXt where possible.
            try:
                info.add_text(key, value)
            except Exception:  # noqa: BLE001
                log.debug("Skipping unpreservable PNG text chunk %r", key)
    info.add_itxt(_PNG_XMP_KEY, xmp)

    tmp = _tmp_for(path)
    img.save(tmp, format="PNG", pnginfo=info)
    _replace_atomic(tmp, path)


# ---------------------------------------------------------------------------
# XMP packet construction / merging
# ---------------------------------------------------------------------------

def _build_xmp_description(json_text: str) -> str:
    escaped = _xml_escape(json_text)
    return (
        f'<rdf:Description rdf:about="" '
        f'xmlns:{_XMP_NS_PREFIX}="{_XMP_NS_URI}">'
        f"<{_XMP_NS_PREFIX}:{_XMP_PROP}>{escaped}</{_XMP_NS_PREFIX}:{_XMP_PROP}>"
        f"</rdf:Description>"
    )


def _build_xmp_packet(json_text: str) -> bytes:
    description = _build_xmp_description(json_text)
    packet = (
        f'<?xpacket begin="﻿" id="{_XPACKET_ID}"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        f"{description}"
        "</rdf:RDF>"
        "</x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


# Matches a whole <rdf:Description …facelocal…>…</rdf:Description> block, so a
# re-export replaces our block in place instead of duplicating it.
_FACELOCAL_DESC_RE = re.compile(
    r"<rdf:Description\b[^>]*?" + re.escape(_XMP_NS_URI) + r".*?</rdf:Description>",
    re.DOTALL,
)
_RDF_CLOSE_RE = re.compile(r"</rdf:RDF>")


def _merge_xmp(existing_xmp, json_text: str) -> bytes:
    """Return XMP bytes with our facelocal block inserted/updated.

    Preserves any other RDF descriptions in *existing_xmp*.  When there is no
    usable existing packet, a fresh minimal packet is built.
    """
    if not existing_xmp:
        return _build_xmp_packet(json_text)

    if isinstance(existing_xmp, bytes):
        try:
            text = existing_xmp.decode("utf-8")
        except UnicodeDecodeError:
            text = existing_xmp.decode("latin-1", errors="replace")
    else:
        text = str(existing_xmp)

    new_desc = _build_xmp_description(json_text)

    if _FACELOCAL_DESC_RE.search(text):
        merged = _FACELOCAL_DESC_RE.sub(lambda _m: new_desc, text, count=1)
    elif _RDF_CLOSE_RE.search(text):
        merged = _RDF_CLOSE_RE.sub(new_desc + "</rdf:RDF>", text, count=1)
    else:
        # Existing XMP is malformed / has no RDF — start clean rather than risk
        # producing an invalid packet.
        return _build_xmp_packet(json_text)

    return merged.encode("utf-8")


# ---------------------------------------------------------------------------
# Reading embedded payloads back
# ---------------------------------------------------------------------------

_FACELOCAL_PROP_RE = re.compile(
    rf"<{_XMP_NS_PREFIX}:{_XMP_PROP}>(.*?)</{_XMP_NS_PREFIX}:{_XMP_PROP}>",
    re.DOTALL,
)


def _read_embedded(path: Path) -> Optional[dict]:
    from PIL import Image as PilImage

    with PilImage.open(path) as img:
        xmp = img.info.get("xmp")
        exif = img.info.get("exif", b"")
        png_text = dict(getattr(img, "text", {}) or {})

    # XMP (JPEG stores raw bytes in info["xmp"]; PNG keeps it in text chunks).
    candidates = []
    if xmp:
        candidates.append(xmp.decode("utf-8", errors="replace") if isinstance(xmp, bytes) else str(xmp))
    if _PNG_XMP_KEY in png_text:
        candidates.append(png_text[_PNG_XMP_KEY])

    for text in candidates:
        m = _FACELOCAL_PROP_RE.search(text)
        if m:
            try:
                return json.loads(_xml_unescape(m.group(1)))
            except json.JSONDecodeError:
                continue

    # EXIF UserComment / ImageDescription.
    if exif:
        payload = _read_exif_comment(exif)
        if payload is not None:
            return payload

    return None


def _read_exif_comment(exif: bytes) -> Optional[dict]:
    try:
        import piexif

        exif_dict = piexif.load(exif)
    except Exception:  # noqa: BLE001
        return None

    raw = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment)
    if isinstance(raw, bytes) and raw[:8] in (b"ASCII\x00\x00\x00", b"UNICODE\x00"):
        body = raw[8:]
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass

    desc = exif_dict.get("0th", {}).get(piexif.ImageIFD.ImageDescription)
    if isinstance(desc, bytes):
        try:
            return json.loads(desc.decode("utf-8"))
        except Exception:  # noqa: BLE001
            pass

    return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _tmp_for(path: Path) -> Path:
    return path.with_name(path.name + ".facelocal.tmp")


def _replace_atomic(tmp: Path, path: Path) -> None:
    """Replace *path* with *tmp*, tolerating transient Windows file locks.

    ``os.replace`` over an existing file raises ``PermissionError`` on Windows
    whenever another handle is briefly open on the target — the Search indexer,
    antivirus, Explorer's thumbnail cache, or the app's own preview. (POSIX has
    no such restriction, which is why this only bites on Windows.) We retry a few
    times with a short backoff, then fall back to overwriting the file contents
    in place. Only if *all* of that fails does the exception propagate, letting
    the caller degrade to a sidecar JSON file.
    """
    import time

    last_exc: Optional[BaseException] = None
    for attempt in range(6):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as exc:  # Windows: target momentarily locked
            last_exc = exc
            time.sleep(0.08 * (attempt + 1))

    # Last resort: overwrite the original file's bytes in place. This keeps the
    # original inode/handle and succeeds in some lock scenarios os.replace can't.
    try:
        data = tmp.read_bytes()
        with open(path, "r+b") as dst:
            dst.seek(0)
            dst.write(data)
            dst.truncate()
        try:
            tmp.unlink()
        except OSError:
            pass
        log.info("Atomic replace blocked; wrote in place instead: %s", path)
        return
    except Exception as exc:  # noqa: BLE001
        log.warning("In-place fallback failed for %s: %s", path, exc)
        try:
            tmp.unlink()
        except OSError:
            pass
        raise last_exc if last_exc is not None else exc


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _xml_unescape(text: str) -> str:
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&amp;", "&")
    )
