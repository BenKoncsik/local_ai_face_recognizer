"""Local gallery server export service.

Generates a self-contained directory you can run on any machine (Windows /
macOS / Linux) with a single command:

    python start.py

Output structure:
    dist/        — Astro static site
    server/      — Node/Express auth server (OTP login, admin panel)
    config.json  — baked admin email, Gmail app-password, seed allowlist
    start.py     — cross-platform launcher (Python 3.8+, no extra packages)
    README.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from app.services.docker_export_service import DockerExportService

_ProgressCb = Callable[[Optional[int], str], None]


class LocalServerExportService(DockerExportService):
    """Like DockerExportService but emits a start.py instead of Dockerfile/k8s."""

    def export_local(
        self,
        target_dir: str,
        admin_email: str,
        gmail_app_password: str,
        allowed_emails_csv: str,
        port: int = 3000,
        progress_callback: Optional[_ProgressCb] = None,
    ) -> Path:
        """Export to *target_dir* and return the path.

        Args:
            target_dir:          Destination directory (created if absent).
            admin_email:         Gmail address used to send OTPs.
            gmail_app_password:  Gmail App Password (16-char, no spaces).
            allowed_emails_csv:  CSV whose first column contains email addresses.
            port:                HTTP port the server listens on.
            progress_callback:   Optional (pct, message) callback.
        """
        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _p(pct: Optional[int], msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        # 1. Build Astro site
        _p(0, "Astro site generálása…")
        self._build_astro(out, progress_callback)

        # 2. Parse allowed emails
        _p(70, "Email-lista olvasása…")
        import json
        allowed = self._parse_csv(allowed_emails_csv)

        # 3. config.json
        _p(72, "Konfig írása…")
        config = {
            "adminEmail": admin_email,
            "gmailAppPassword": gmail_app_password,
            "allowedEmails": allowed,
            "sessionTimeoutMinutes": 30,
            "otpExpiryMinutes": 10,
            "port": port,
        }
        (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

        # 4. Server files (shared with Docker export)
        _p(75, "Szerver fájlok írása…")
        self._write_server(out, port)

        # 5. Cross-platform launcher
        _p(90, "Indítószkript írása…")
        start = out / "start.py"
        start.write_text(_START_PY.replace("__PORT__", str(port)))
        # Make executable on POSIX
        try:
            import stat
            start.chmod(start.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        # 5b. Native launchers that just run start.py (no terminal needed)
        _p(92, "Indító szkriptek írása…")
        sh = out / "start.sh"
        sh.write_text(_START_SH)
        cmd = out / "start.command"  # macOS double-click launcher
        cmd.write_text(_START_SH)
        (out / "start.bat").write_text(_START_BAT)
        try:
            import stat
            for f in (sh, cmd):
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        # 6. README
        _p(95, "README írása…")
        (out / "README.md").write_text(_README_MD.replace("__PORT__", str(port)))

        _p(100, "Kész.")
        return out


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_START_PY = r'''#!/usr/bin/env python3
"""start.py — Galéria szerver indítása (Windows / macOS / Linux).

Használat:
    python start.py [--port 3000] [--no-browser]

Követelmények: Python 3.8+ és Node.js 18+.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_exe(names: list) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def _need_node() -> str:
    node = _find_exe(["node", "node.exe"])
    if node:
        return node
    sys = platform.system()
    hints = {
        "Windows": "Letöltés: https://nodejs.org  (Windows Installer .msi)\n"
                   "vagy: winget install OpenJS.NodeJS",
        "Darwin":  "Telepítés: brew install node\n"
                   "vagy: https://nodejs.org",
    }
    hint = hints.get(sys, "Telepítés: sudo apt install nodejs npm\n"
                          "vagy: https://nodejs.org")
    print(f"\n[HIBA] Node.js nem található.\n{hint}\n")
    raise SystemExit(1)


def _need_npm() -> str:
    npm = _find_exe(["npm", "npm.cmd", "npm.ps1"])
    if npm:
        return npm
    print("[HIBA] npm nem található. Telepítsd a Node.js-t: https://nodejs.org")
    raise SystemExit(1)


def _install_deps(server_dir: Path, npm: str) -> None:
    if (server_dir / "node_modules").exists():
        return
    print("[setup] npm install (csak az első indításkor)…")
    r = subprocess.run([npm, "install", "--omit=dev"], cwd=str(server_dir))
    if r.returncode:
        print("[HIBA] npm install sikertelen.")
        raise SystemExit(r.returncode)


def _wait_for_server(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description="Galéria szerver")
    p.add_argument("--port",       type=int, default=None)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    server_dir  = HERE / "server"
    config_path = HERE / "config.json"

    if not (server_dir / "index.js").exists():
        print("[HIBA] Nem találom a server/index.js fájlt.\n"
              "Futtasd ezt a szkriptet az exportált mappa gyökeréből.")
        raise SystemExit(1)

    port = args.port
    if port is None:
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            port = int(cfg.get("port", __PORT__))
        except Exception:
            port = __PORT__

    node = _need_node()
    npm  = _need_npm()
    _install_deps(server_dir, npm)

    url = f"http://localhost:{port}"
    print(f"\n{'='*50}")
    print(f"  Galéria szerver: {url}")
    print(f"  Admin felület:   {url}/admin")
    print(f"  Leállítás:       Ctrl+C")
    print(f"{'='*50}\n")

    env  = {**os.environ, "PORT": str(port)}
    proc = subprocess.Popen([node, str(server_dir / "index.js")],
                            env=env, cwd=str(HERE))

    if not args.no_browser:
        if _wait_for_server(port):
            webbrowser.open(url)
        else:
            print(f"[!] Szerver nem válaszolt {15}s alatt — nyisd meg: {url}")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nLeállítás…")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    raise SystemExit(proc.returncode or 0)


if __name__ == "__main__":
    main()
'''

_START_SH = r"""#!/usr/bin/env bash
# Galéria szerver indítása macOS / Linux alatt — lefuttatja a start.py-t.
set -e
cd "$(dirname "$0")"

for py in python3 python; do
  if command -v "$py" >/dev/null 2>&1; then
    exec "$py" start.py "$@"
  fi
done

echo "[HIBA] Python 3 nem található."
echo "Telepítés:  macOS -> brew install python   |   Linux -> sudo apt install python3"
read -r -p "Enter a kilépéshez..." _
exit 1
"""

_START_BAT = r"""@echo off
REM Galeria szerver inditasa Windows alatt - lefuttatja a start.py-t.
cd /d "%~dp0"

where py >nul 2>nul && (
  py start.py %*
  goto :end
)
where python >nul 2>nul && (
  python start.py %*
  goto :end
)
where python3 >nul 2>nul && (
  python3 start.py %*
  goto :end
)

echo [HIBA] Python 3 nem talalhato.
echo Telepites: https://www.python.org/downloads/windows/  ^(jelold be: Add Python to PATH^)
pause

:end
"""


_README_MD = """# Galéria — helyi szerver

## Indítás

A legegyszerűbb — dupla kattintás az operációs rendszerednek megfelelő indítóra:

- **macOS:** `start.command`
- **Linux:** `start.sh`
- **Windows:** `start.bat`

Vagy terminálból:

```bash
python start.py
```

Mindegyik ugyanazt csinálja: lefuttatja a `start.py`-t, ami elindítja az
email-belépéses szervert, és megnyitja a böngészőt: `http://localhost:__PORT__`

### Opciók

```bash
python start.py --port 8080       # más port
python start.py --no-browser      # ne nyissa meg a böngészőt
```

## Belépés

1. Add meg az email cím-ed a bejelentkező oldalon
2. Ha szerepelsz az engedélyezett listán, kapsz egy 6 jegyű kódot Gmailben
3. A kóddal beléphetsz — a session 30 perc inaktivitás után jár le

## Admin felület

Elérhető: `http://localhost:__PORT__/admin`
Csak az admin email cím tud belépni (az export-kor megadott Gmail cím).
Az adminon email címeket adhatsz hozzá / törölhetsz futás közben.

## Követelmények

- Python 3.8+
- Node.js 18+
  - Windows: `winget install OpenJS.NodeJS` vagy https://nodejs.org
  - macOS: `brew install node`
  - Linux: `sudo apt install nodejs npm`

Az `npm install` csak az első indításkor fut le automatikusan.
"""
