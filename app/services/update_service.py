"""GitHub release update checker and downloader."""

from __future__ import annotations

import logging
import os
import platform
import re
import sys
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


def _updater_log_file() -> Path:
    """Return a persistent log file for work that continues after app exit."""
    from app.paths import user_data_dir

    log_dir = user_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "face_local_updater.log"


def _update_download_dir() -> Path:
    """Return an app-owned, writable directory for update downloads.

    Downloading into (and extracting from) a per-user app folder under
    %LOCALAPPDATA% instead of the shared system %TEMP% avoids the Windows
    "Unable to execute file in the temporary directory ... Error 5" failure
    on machines where antivirus, Software Restriction Policies or AppLocker
    block execution from %TEMP%.
    """
    from app.paths import user_data_dir

    d = user_data_dir() / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ps_single_quote(value: Path | str) -> str:
    """Quote a value as a PowerShell single-quoted string."""
    return "'" + str(value).replace("'", "''") + "'"


def download_asset(
    release: ReleaseInfo,
    progress_cb: Callable[[int, int], None],
) -> Path:
    """Download the release asset to a temp file. Returns the path.

    progress_cb(downloaded_bytes, total_bytes) is called periodically.
    """
    name = release.asset_name
    dest = _update_download_dir() / name
    # Remove a stale copy from a previous run so a partial/locked file does
    # not block the write; fall back to a unique name if it cannot be deleted.
    try:
        if dest.exists():
            dest.unlink()
    except OSError:
        suffix = ".tar.gz" if name.endswith(".tar.gz") else Path(name).suffix
        stem = name[: -len(suffix)] if suffix else name
        dest = dest.with_name(f"{stem}-{os.getpid()}{suffix}")
    log.info(
        "Update download started: asset=%s size=%s url=%s dest=%s",
        release.asset_name,
        release.asset_size,
        release.asset_url,
        dest,
    )

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

    actual_size = dest.stat().st_size if dest.exists() else 0
    log.info("Update download finished: asset=%s path=%s bytes=%d", release.asset_name, dest, actual_size)
    return dest


def apply_update(path: Path) -> None:
    """Apply the downloaded update automatically on all platforms."""
    log_file = _updater_log_file()
    log.info(
        "Applying update: platform=%s path=%s suffix=%s updater_log=%s",
        sys.platform,
        path,
        path.suffix,
        log_file,
    )
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

    log_file = _updater_log_file()
    installer_log = log_file.with_name("face_local_installer.log")

    # Redirect the Inno Setup installer's self-extraction away from the system
    # %TEMP%.  When the installer launches it unpacks helper files into the
    # directory named by TMP/TEMP and runs them; on machines where antivirus,
    # SRP or AppLocker block execution from %TEMP% this fails before the wizard
    # appears with "Unable to execute file in the temporary directory ...
    # Error 5".  An app-owned folder under %LOCALAPPDATA% is writable and
    # policy-allowed.  Setting os.environ here propagates to both the
    # ShellExecuteW child and the subprocess fallback below.
    installer_tmp = _update_download_dir() / "installer-tmp"
    installer_tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(installer_tmp)
    os.environ["TEMP"] = str(installer_tmp)

    log.info(
        "Starting Windows installer update: installer=%s updater_log=%s installer_log=%s tmp=%s",
        exe_path,
        log_file,
        installer_log,
        installer_tmp,
    )
    # The installer uses PrivilegesRequired=lowest (per-user install), so no
    # UAC elevation is needed.  "open" lets the installer handle its own
    # privilege requirements without forcing an unnecessary UAC prompt.
    params = f'/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /LOG="{installer_log}"'
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "open", str(exe_path), params, None, 1
    )
    log.info("ShellExecuteW returned %s for installer update", ret)
    if ret <= 32:
        # ShellExecute returns a value > 32 on success; fall back to direct launch.
        import subprocess
        log.error(
            "ShellExecuteW failed with code %s; falling back to direct installer launch",
            ret,
        )
        subprocess.Popen(
            [
                str(exe_path),
                "/SILENT",
                "/CLOSEAPPLICATIONS",
                "/RESTARTAPPLICATIONS",
                f"/LOG={installer_log}",
            ],
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
    log_file = _updater_log_file()
    log.info(
        "Starting Windows ZIP update: zip=%s install_dir=%s exe=%s helper_log=%s pid=%d",
        zip_path,
        install_dir,
        exe,
        log_file,
        pid,
    )

    ps_log = _ps_single_quote(log_file)
    ps_zip = _ps_single_quote(zip_path)
    ps_install_dir = _ps_single_quote(install_dir)
    ps_exe = _ps_single_quote(exe)
    script = textwrap.dedent(f"""\
        $ErrorActionPreference = 'Stop'
        $LogFile = {ps_log}
        function Write-UpdateLog([string]$Message) {{
            $dir = Split-Path -Parent $LogFile
            if (-not (Test-Path $dir)) {{ New-Item -ItemType Directory -Path $dir -Force | Out-Null }}
            $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
            Add-Content -LiteralPath $LogFile -Value "$stamp [updater] $Message"
        }}

        Write-UpdateLog 'Windows ZIP updater helper started.'
        Write-UpdateLog 'Waiting for process {pid} to exit.'

        # Wait for the old process to exit
        while (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{
            Start-Sleep -Milliseconds 500
        }}
        Write-UpdateLog 'Old process exited.'

        try {{
            Write-UpdateLog 'Loading ZIP assembly.'
            Add-Type -AssemblyName System.IO.Compression.FileSystem
            Write-UpdateLog 'Opening ZIP: {zip_path}'
            $zip  = [System.IO.Compression.ZipFile]::OpenRead({ps_zip})
            $count = 0
            foreach ($entry in $zip.Entries) {{
                $dest = Join-Path {ps_install_dir} $entry.FullName
                $destDir = Split-Path $dest
                if (-not (Test-Path $destDir)) {{ New-Item -ItemType Directory -Path $destDir -Force | Out-Null }}
                if ($entry.Name -ne '') {{
                    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
                    $count += 1
                }}
            }}
            $zip.Dispose()
            Write-UpdateLog "ZIP extraction completed. Files written: $count"

            Write-UpdateLog 'Relaunching app: {exe}'
            Start-Process {ps_exe}
            Write-UpdateLog 'Relaunch command started.'
        }} catch {{
            Write-UpdateLog ('ERROR: ' + $_.Exception.ToString())
            throw
        }}

        # Self-delete
        Write-UpdateLog 'Removing updater helper script.'
        Remove-Item -LiteralPath $MyInvocation.MyCommand.Path -Force
    """)

    # Write the helper into the app-owned updates folder rather than %TEMP%, so
    # it survives SRP/AppLocker policies that restrict the system temp dir.
    tmp = _update_download_dir() / f"face-local-update-{pid}.ps1"
    tmp.write_text(script, encoding="utf-8")
    log.info("Windows ZIP updater helper script written: %s", tmp)
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
