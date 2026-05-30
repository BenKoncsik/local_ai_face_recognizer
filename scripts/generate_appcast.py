#!/usr/bin/env python3
"""Generate a Sparkle-compatible appcast.xml for macOS automatic updates.

The generated file conforms to the Sparkle 2 appcast format:
  https://sparkle-project.org/documentation/publishing/

Usage example (called from the release CI job)::

    python3 scripts/generate_appcast.py \\
        --version 1.2.3 \\
        --build  1.2.3 \\
        --zip    release/Face-Local-macOS-sparkle-1.2.3.zip \\
        --signature MEQCIQDxxx... \\
        --download-url https://github.com/owner/repo/releases/download/v1.2.3/Face-Local-macOS-sparkle-1.2.3.zip \\
        --release-notes-url https://github.com/owner/repo/releases/tag/v1.2.3 \\
        --output docs/appcast.xml
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


def _rfc_2822_now() -> str:
    return datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S %z")


def _zip_length(path: Path) -> int:
    return path.stat().st_size


def _build_xml(
    *,
    title: str,
    app_version: str,
    build_number: str,
    release_notes_url: str,
    download_url: str,
    zip_length: int,
    eddsa_signature: str,
    min_os_version: str,
    pub_date: str,
) -> str:
    # Use a plain string template — avoids xml.etree namespace-prefix gymnastics.
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<rss version="2.0"\n'
        '     xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"\n'
        '     xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        "  <channel>\n"
        f"    <title>{title}</title>\n"
        f"    <link>{download_url}</link>\n"
        f"    <description>Most recent changes with {title}.</description>\n"
        "    <language>en</language>\n"
        "    <item>\n"
        f"      <title>{title} {app_version}</title>\n"
        f"      <sparkle:releaseNotesLink>{release_notes_url}</sparkle:releaseNotesLink>\n"
        f"      <pubDate>{pub_date}</pubDate>\n"
        "      <enclosure\n"
        f'        url="{download_url}"\n'
        f'        sparkle:version="{build_number}"\n'
        f'        sparkle:shortVersionString="{app_version}"\n'
        f'        sparkle:edSignature="{eddsa_signature}"\n'
        f'        sparkle:minimumSystemVersion="{min_os_version}"\n'
        f'        length="{zip_length}"\n'
        '        type="application/octet-stream"/>\n'
        "    </item>\n"
        "  </channel>\n"
        "</rss>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title", default="Face-Local", help="App display name")
    parser.add_argument("--version", required=True, help="Display version, e.g. 1.2.3")
    parser.add_argument("--build", required=True, help="CFBundleVersion / build number")
    parser.add_argument("--zip", required=True, type=Path, help="Path to Sparkle ZIP file")
    parser.add_argument("--signature", required=True, help="EdDSA signature from sign_update")
    parser.add_argument("--download-url", required=True, help="Public download URL for the ZIP")
    parser.add_argument("--release-notes-url", default="", help="URL to release notes page")
    parser.add_argument("--min-os", default="12.0", help="Minimum macOS version string")
    parser.add_argument("--pub-date", default="", help="Publication date (RFC 2822); defaults to now")
    parser.add_argument("--output", required=True, type=Path, help="Output path for appcast.xml")
    args = parser.parse_args()

    zip_path: Path = args.zip
    if not zip_path.exists():
        print(f"ERROR: ZIP not found: {zip_path}", file=sys.stderr)
        sys.exit(1)

    length = _zip_length(zip_path)
    pub_date = args.pub_date or _rfc_2822_now()
    notes_url = args.release_notes_url or args.download_url

    xml = _build_xml(
        title=args.title,
        app_version=args.version,
        build_number=args.build,
        release_notes_url=notes_url,
        download_url=args.download_url,
        zip_length=length,
        eddsa_signature=args.signature,
        min_os_version=args.min_os,
        pub_date=pub_date,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml, encoding="utf-8")

    print(f"appcast.xml written to {args.output}")
    print(f"  version:   {args.version}")
    print(f"  build:     {args.build}")
    print(f"  length:    {length:,} bytes")
    print(f"  signature: {args.signature[:24]}...")
    print(f"  url:       {args.download_url}")


if __name__ == "__main__":
    main()
