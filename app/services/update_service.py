"""GitHub release update checker and downloader."""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_REPO = "HunKonTech/local_ai_face_recognizer"
_API_URL = f"https://api.github.com/repos/{_REPO}/releases/latest"


@dataclass
class ReleaseInfo:
    version: str
    tag: str
    url: str          # browser html url
    asset_name: str
    asset_url: str
    asset_size: int   # bytes


def _parse_version(v: str) -> tuple[int, ...]:
    v = v.lstrip("v")
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", v)
    if not m:
        return (0,)
    return tuple(int(x) for x in m.groups())


def _pick_asset(assets: list[dict]) -> Optional[dict]:
    """Return the best asset for the running OS and architecture, or None."""
    os_platform = sys.platform
    machine = platform.machine().lower()
    is_arm = machine in ("arm64", "aarch64")

    def score(a: dict) -> int:
        n = a["name"].lower()
        if os_platform == "darwin":
            if n.endswith(".dmg") and "macos" in n:
                return 2
            if n.endswith(".zip") and "macos" in n:
                return 1
        elif os_platform == "win32":
            if n.endswith(".exe") and "windows" in n:
                return 2
            if n.endswith(".zip") and "windows" in n:
                return 1
        else:
            # Linux: match architecture
            arch_tag = "arm64" if is_arm else "x64"
            if n.endswith(".deb") and "linux" in n and arch_tag in n:
                return 4
            if n.endswith(".tar.gz") and "linux" in n and arch_tag in n:
                return 3
            # fallback: any linux asset (no arch tag in name)
            if n.endswith(".deb") and "linux" in n:
                return 2
            if n.endswith(".tar.gz") and "linux" in n:
                return 1
        return 0

    ranked = sorted(assets, key=score, reverse=True)
    return ranked[0] if ranked and score(ranked[0]) > 0 else None


def fetch_latest_release() -> Optional[ReleaseInfo]:
    """Query GitHub API for the latest release. Returns None on error."""
    import json
    try:
        req = urllib.request.Request(
            _API_URL,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": "Face-Local-Updater/1"},
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.warning("Update check failed: %s", exc)
        return None

    asset = _pick_asset(data.get("assets", []))
    if not asset:
        log.info("No matching asset found for platform %s", sys.platform)
        return None

    return ReleaseInfo(
        version=data["tag_name"].lstrip("v"),
        tag=data["tag_name"],
        url=data["html_url"],
        asset_name=asset["name"],
        asset_url=asset["browser_download_url"],
        asset_size=asset["size"],
    )


def is_newer(remote_version: str, local_version: str) -> bool:
    return _parse_version(remote_version) > _parse_version(local_version)


def download_asset(
    release: ReleaseInfo,
    progress_cb: Callable[[int, int], None],
) -> Path:
    """Download the release asset to a temp file. Returns the path.

    progress_cb(downloaded_bytes, total_bytes) is called periodically.
    """
    name = release.asset_name
    suffix = ".tar.gz" if name.endswith(".tar.gz") else Path(name).suffix
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix="face-local-update-"
    )
    tmp.close()
    dest = Path(tmp.name)

    req = urllib.request.Request(
        release.asset_url,
        headers={"User-Agent": "Face-Local-Updater/1"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = release.asset_size or int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        chunk = 65536
        with open(dest, "wb") as fh:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                downloaded += len(buf)
                progress_cb(downloaded, total)

    log.info("Downloaded %s → %s", release.asset_name, dest)
    return dest


def apply_update(path: Path) -> None:
    """Apply the downloaded update automatically on all platforms."""
    if sys.platform == "darwin" and path.suffix == ".dmg":
        _apply_macos_dmg(path)
    elif sys.platform == "win32":
        if path.suffix == ".exe":
            _apply_windows_exe(path)
        else:
            _apply_windows_zip(path)
    else:
        if path.suffix == ".deb":
            _apply_linux_deb(path)
        else:
            _apply_linux_targz(path)  # .tar.gz or .gz


def _apply_macos_dmg(dmg_path: Path) -> None:
    """Mount the DMG, replace the running .app, relaunch, quit."""
    import subprocess
    import tempfile
    import textwrap

    # Determine the running .app bundle path
    exe = Path(sys.executable).resolve()
    # sys.executable inside a bundle is  .../Face-Local.app/Contents/MacOS/Face-Local
    app_bundle: Optional[Path] = None
    for parent in exe.parents:
        if parent.suffix == ".app":
            app_bundle = parent
            break

    if app_bundle is None:
        # Not running from a bundle — just open the DMG normally
        subprocess.Popen(["open", str(dmg_path)])
        return

    install_dir = app_bundle.parent  # usually /Applications
    app_name = app_bundle.name       # Face-Local.app

    # Write a helper shell script that:
    #  1. Waits for this process to exit
    #  2. Mounts the DMG
    #  3. Copies the new .app into place
    #  4. Detaches the DMG
    #  5. Relaunches the app
    #  6. Deletes itself
    script = textwrap.dedent(f"""\
        #!/bin/bash
        # Wait for the old process to exit
        PID={os.getpid()}
        while kill -0 "$PID" 2>/dev/null; do sleep 0.5; done

        # Mount DMG
        MOUNT=$(hdiutil attach -nobrowse -noautoopen "{dmg_path}" | \\
                awk '/\\/Volumes\\//' | tail -1 | awk '{{print $NF}}')
        if [ -z "$MOUNT" ]; then exit 1; fi

        # Replace app
        rm -rf "{install_dir}/{app_name}"
        cp -R "$MOUNT/{app_name}" "{install_dir}/{app_name}"

        # Detach
        hdiutil detach "$MOUNT" -quiet

        # Relaunch
        open "{install_dir}/{app_name}"

        # Self-delete
        rm -- "$0"
    """)

    tmp_script = Path(tempfile.mktemp(suffix=".sh", prefix="face-local-update-"))
    tmp_script.write_text(script, encoding="utf-8")
    tmp_script.chmod(0o755)
    subprocess.Popen([str(tmp_script)], close_fds=True)
    sys.exit(0)


def _apply_windows_exe(exe_path: Path) -> None:
    """Run the Inno Setup installer; ShellExecute triggers UAC elevation if needed."""
    import ctypes
    # ShellExecuteW with "runas" properly requests admin rights via UAC prompt.
    # subprocess.Popen does NOT trigger UAC, causing silent failure on most systems.
    params = "/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS"
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", str(exe_path), params, None, 1
    )
    if ret <= 32:
        # ShellExecute returns a value > 32 on success; fall back to direct launch
        import subprocess
        subprocess.Popen(
            [str(exe_path), "/SILENT", "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"],
            close_fds=True,
        )
    sys.exit(0)


def _apply_windows_zip(zip_path: Path) -> None:
    """Extract portable ZIP, replace running dir, relaunch via PowerShell script."""
    import subprocess
    import textwrap

    exe = Path(sys.executable).resolve()
    install_dir = exe.parent  # e.g. C:\Users\...\Face-Local\

    pid = os.getpid()
    script = textwrap.dedent(f"""\
        # Wait for the old process to exit
        while (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{
            Start-Sleep -Milliseconds 500
        }}

        # Extract ZIP on top of install dir (overwrite)
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        $zip  = [System.IO.Compression.ZipFile]::OpenRead('{zip_path}')
        foreach ($entry in $zip.Entries) {{
            $dest = Join-Path '{install_dir}' $entry.FullName
            $destDir = Split-Path $dest
            if (-not (Test-Path $destDir)) {{ New-Item -ItemType Directory -Path $destDir | Out-Null }}
            if ($entry.Name -ne '') {{
                [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
            }}
        }}
        $zip.Dispose()

        # Relaunch
        Start-Process '{exe}'

        # Self-delete
        Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force
    """)

    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".ps1", prefix="face-local-update-"))
    tmp.write_text(script, encoding="utf-8")
    subprocess.Popen(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(tmp)],
        close_fds=True,
        creationflags=0x00000008,  # DETACHED_PROCESS
    )
    sys.exit(0)


def _apply_linux_deb(deb_path: Path) -> None:
    """Install .deb with pkexec, then relaunch."""
    import subprocess
    import textwrap

    exe = Path(sys.executable).resolve()
    pid = os.getpid()

    script = textwrap.dedent(f"""\
        #!/bin/bash
        while kill -0 {pid} 2>/dev/null; do sleep 0.5; done
        pkexec dpkg -i '{deb_path}'
        # Relaunch — the .deb installs to /usr/bin or /opt
        APP=$(command -v face-local 2>/dev/null || echo '{exe}')
        nohup "$APP" &>/dev/null &
        rm -- "$0"
    """)

    tmp = Path(tempfile.mktemp(suffix=".sh", prefix="face-local-update-"))
    tmp.write_text(script, encoding="utf-8")
    tmp.chmod(0o755)
    subprocess.Popen([str(tmp)], close_fds=True)
    sys.exit(0)


def _apply_linux_targz(tgz_path: Path) -> None:
    """Extract .tar.gz, replace running dir, relaunch."""
    import subprocess
    import textwrap

    exe = Path(sys.executable).resolve()
    install_dir = exe.parent
    pid = os.getpid()

    script = textwrap.dedent(f"""\
        #!/bin/bash
        while kill -0 {pid} 2>/dev/null; do sleep 0.5; done

        # Extract over install dir
        tar -xzf '{tgz_path}' -C '{install_dir}' --strip-components=1

        # Relaunch
        nohup '{exe}' &>/dev/null &

        # Cleanup
        rm -f '{tgz_path}'
        rm -- "$0"
    """)

    tmp = Path(tempfile.mktemp(suffix=".sh", prefix="face-local-update-"))
    tmp.write_text(script, encoding="utf-8")
    tmp.chmod(0o755)
    subprocess.Popen([str(tmp)], close_fds=True)
    sys.exit(0)
