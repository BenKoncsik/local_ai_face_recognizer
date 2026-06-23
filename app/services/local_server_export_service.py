"""Local gallery server export service.

Generates a self-contained directory you can run on any machine (Windows /
macOS / Linux) with a single command:

    python start.py

Output structure:
    dist/               — Astro static site
    server/             — Node/Express auth server (OTP login, admin panel)
    config.json         — baked admin email, Gmail app-password, seed allowlist
    start.py            — cross-platform launcher (Python 3.8+, no extra packages)
    Caddyfile           — (optional) Caddy reverse proxy + auto Let's Encrypt
    apache-vhost.conf   — (optional) Apache/XAMPP VirtualHost + Let's Encrypt
    setup-https.sh      — (optional) one-shot HTTPS setup script
    README.md
"""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Callable, Optional

from app.services.docker_export_service import DockerExportService

_ProgressCb = Callable[[Optional[int], str], None]

# Supported HTTPS modes:
#   "none"   — HTTP only, no reverse-proxy config generated
#   "caddy"  — Caddyfile + setup-https.sh (Caddy handles 80/443 + ACME)
#   "apache" — apache-vhost.conf + setup-https.sh (XAMPP/Apache + certbot)
HTTPS_MODES = ("none", "caddy", "apache")


class LocalServerExportService(DockerExportService):
    """Like DockerExportService but emits a start.py instead of Dockerfile/k8s."""

    def export_local(
        self,
        target_dir: str,
        admin_email: str,
        gmail_app_password: str,
        allowed_emails_csv: str,
        port: int = 3000,
        domain: str = "",
        https_mode: str = "none",
        progress_callback: Optional[_ProgressCb] = None,
    ) -> Path:
        """Export to *target_dir* and return the path.

        Args:
            target_dir:          Destination directory (created if absent).
            admin_email:         Gmail address used to send OTPs.
            gmail_app_password:  Gmail App Password (16-char, no spaces).
            allowed_emails_csv:  CSV whose first column contains email addresses.
            port:                HTTP port the Node.js server listens on.
            domain:              Public hostname (e.g. "csaladfa.kncsk.hu").
                                 Required when https_mode != "none".
            https_mode:          One of "none", "caddy", "apache".
            progress_callback:   Optional (pct, message) callback.
        """
        import json

        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)
        https_enabled = bool(domain) and https_mode != "none"

        def _p(pct: Optional[int], msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        # 1. Build Astro site
        _p(0, "Astro site generálása…")
        self._build_astro(out, progress_callback)

        # 2. Parse allowed emails
        _p(70, "Email-lista olvasása…")
        allowed = self._parse_csv(allowed_emails_csv)

        # 3. config.json
        _p(72, "Konfig írása…")
        try:
            from app import __version__ as _app_version
        except Exception:
            _app_version = ""
        config: dict = {
            "adminEmail": admin_email,
            "gmailAppPassword": gmail_app_password,
            "allowedEmails": allowed,
            "sessionTimeoutMinutes": 30,
            "otpExpiryMinutes": 10,
            "port": port,
            "appVersion": _app_version,
        }
        if https_enabled:
            config["domain"] = domain
            config["httpsEnabled"] = True
        (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

        # 4. Server files (shared with Docker export)
        _p(75, "Szerver fájlok írása…")
        self._write_server(out, port)

        # 5. Cross-platform launcher
        _p(86, "Indítószkript írása…")
        start = out / "start.py"
        start.write_text(_START_PY.replace("__PORT__", str(port)))
        try:
            start.chmod(start.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        # 5b. Native launchers
        _p(88, "Indító szkriptek írása…")
        sh = out / "start.sh"
        sh.write_text(_START_SH)
        cmd = out / "start.command"  # macOS double-click launcher
        cmd.write_text(_START_SH)
        (out / "start.bat").write_text(_START_BAT)
        try:
            for f in (sh, cmd):
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass

        # 6. HTTPS config (optional)
        if https_enabled:
            _p(92, "HTTPS konfig írása…")
            self._write_https_config(out, domain, port, https_mode)

        # 7. README
        _p(96, "README írása…")
        readme = _README_MD.replace("__PORT__", str(port))
        if https_enabled:
            readme = readme.replace("__HTTPS_SECTION__", _README_HTTPS_SECTION
                                    .replace("__DOMAIN__", domain)
                                    .replace("__PORT__", str(port))
                                    .replace("__HTTPS_MODE__", https_mode))
        else:
            readme = readme.replace("__HTTPS_SECTION__", "")
        (out / "README.md").write_text(readme)

        _p(100, "Kész.")
        return out

    # ------------------------------------------------------------------
    # HTTPS helpers
    # ------------------------------------------------------------------

    def _write_https_config(
        self, out: Path, domain: str, port: int, https_mode: str
    ) -> None:
        def _sub(text: str) -> str:
            return text.replace("__DOMAIN__", domain).replace("__PORT__", str(port))

        if https_mode == "caddy":
            (out / "Caddyfile").write_text(_sub(_CADDYFILE))
            sh = out / "setup-https.sh"
            sh.write_text(_sub(_SETUP_HTTPS_CADDY))
            try:
                sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except Exception:
                pass
            (out / "setup-https.ps1").write_text(_sub(_SETUP_HTTPS_CADDY_PS1))

        elif https_mode == "apache":
            (out / "apache-vhost.conf").write_text(_sub(_APACHE_VHOST))
            sh = out / "setup-https.sh"
            sh.write_text(_sub(_SETUP_HTTPS_APACHE))
            try:
                sh.chmod(sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except Exception:
                pass
            (out / "setup-https.ps1").write_text(_sub(_SETUP_HTTPS_APACHE_PS1))


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# ── HTTPS configs ──────────────────────────────────────────────────────────

_CADDYFILE = """\
# Galéria — Caddy reverse proxy + automatikus Let's Encrypt
# Telepítés: sudo bash setup-https.sh
# Caddy dokumentáció: https://caddyserver.com/docs/

__DOMAIN__ {
    reverse_proxy localhost:__PORT__
}
"""

_SETUP_HTTPS_CADDY = r"""#!/usr/bin/env bash
# setup-https.sh — Caddy telepítése + galéria HTTPS beállítása
# Futtatás: sudo bash setup-https.sh
# Követelmény: Ubuntu/Debian Linux, root jogosultság,
#              __DOMAIN__ DNS-ben már erre a szerverre mutat,
#              80-as és 443-as port szabad (ha XAMPP fut, lásd README.md).
set -euo pipefail

DOMAIN="__DOMAIN__"
PORT="__PORT__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo -e "\033[1;34m[https-setup]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[     ok     ]\033[0m $*"; }
err() { echo -e "\033[1;31m[    hiba    ]\033[0m $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "Root jogosultság szükséges: sudo bash setup-https.sh"

# ── Port-ellenőrzés ───────────────────────────────────────────────────────
for p in 80 443; do
  if ss -tlnp 2>/dev/null | grep -q ":${p} " || netstat -tlnp 2>/dev/null | grep -q ":${p} "; then
    echo ""
    echo "⚠  A ${p}-as port már foglalt."
    echo "   Ha XAMPP/Apache fut ezen a porton, két lehetőség van:"
    echo "   1) XAMPP-ot állíts át más portra (pl. 8080), majd futtasd újra ezt a scriptet."
    echo "   2) Használd helyette az Apache/XAMPP módot: setup-https.sh --apache"
    echo ""
    read -rp "Folytatás ugyanígy? (i/n): " yn
    [[ "$yn" =~ ^[Ii] ]] || exit 0
  fi
done

# ── Caddy telepítése ──────────────────────────────────────────────────────
if ! command -v caddy &>/dev/null; then
  log "Caddy telepítése (Debian/Ubuntu)..."
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | tee /etc/apt/sources.list.d/caddy-stable.list
  apt-get update
  apt-get install -y caddy
  ok "Caddy telepítve"
else
  ok "Caddy már telepítve: $(caddy version)"
fi

# ── Caddyfile másolása ────────────────────────────────────────────────────
cp "$SCRIPT_DIR/Caddyfile" /etc/caddy/Caddyfile
ok "Caddyfile másolva → /etc/caddy/Caddyfile"

# ── Caddy (újra)indítása ──────────────────────────────────────────────────
if systemctl is-active --quiet caddy 2>/dev/null; then
  log "Caddy reload..."
  systemctl reload caddy
else
  log "Caddy engedélyezése és indítása..."
  systemctl enable --now caddy
fi
ok "Caddy fut"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "HTTPS beállítva!"
echo "   URL   : https://${DOMAIN}"
echo "   Admin : https://${DOMAIN}/admin"
echo "   TLS   : automatikus Let's Encrypt (Caddy kezeli, auto-megújítás)"
echo ""
echo "   A galériaszerverre (Node.js) külön kell elindítani:"
echo "     python start.py --no-browser"
echo "   Vagy systemd service-ként (lásd README.md #systemd szekció)."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_APACHE_VHOST = """\
# Galéria — Apache VirtualHost konfig (XAMPP / Apache2 mellé)
# Másolás helye:
#   Ubuntu/Debian Apache2: /etc/apache2/sites-available/gallery-__DOMAIN__.conf
#   XAMPP (Linux):         /opt/lampp/etc/extra/httpd-gallery.conf
#                          + Include-old az /opt/lampp/etc/httpd.conf végére
#
# Let's Encrypt automatikusan hozzáadja a *:443 blokkot:
#   sudo bash setup-https.sh

<VirtualHost *:80>
    ServerName __DOMAIN__

    # Proxy a galériaszerverre (Node.js, localhost:__PORT__)
    ProxyPreserveHost On
    ProxyPass        / http://localhost:__PORT__/
    ProxyPassReverse / http://localhost:__PORT__/

    # WebSocket támogatás
    RewriteEngine On
    RewriteCond %{HTTP:Upgrade} =websocket [NC]
    RewriteRule /(.*) ws://localhost:__PORT__/$1 [P,L]

    # X-Forwarded-* fejlécek küldése a Node.js-nek
    RequestHeader set X-Forwarded-Proto "http"
</VirtualHost>

# A Let's Encrypt tanúsítvány megszerzése után a certbot automatikusan
# hozzáadja a *:443 VirtualHost-ot és beállítja az HTTPS redirect-et.
"""

_SETUP_HTTPS_APACHE = r"""#!/usr/bin/env bash
# setup-https.sh — Apache/XAMPP + Let's Encrypt beállítása a galériaszerverre
# Futtatás: sudo bash setup-https.sh
# Követelmény: Ubuntu/Debian Linux, Apache2 vagy XAMPP telepítve,
#              root jogosultság, __DOMAIN__ DNS-ben erre a szerverre mutat.
set -euo pipefail

DOMAIN="__DOMAIN__"
PORT="__PORT__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo -e "\033[1;34m[https-setup]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[     ok     ]\033[0m $*"; }
err() { echo -e "\033[1;31m[    hiba    ]\033[0m $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "Root jogosultság szükséges: sudo bash setup-https.sh"

# ── Apache mod_proxy engedélyezése (standard Apache2) ────────────────────
if command -v a2enmod &>/dev/null; then
  log "Apache modulok engedélyezése..."
  a2enmod proxy proxy_http rewrite ssl headers 2>/dev/null
  ok "Modulok engedélyezve"
fi

# ── VirtualHost konfig másolása ───────────────────────────────────────────
SITES_AVAILABLE="/etc/apache2/sites-available"
if [[ -d "$SITES_AVAILABLE" ]]; then
  # Standard Apache2 (Ubuntu/Debian)
  CONF="$SITES_AVAILABLE/gallery-${DOMAIN}.conf"
  cp "$SCRIPT_DIR/apache-vhost.conf" "$CONF"
  a2ensite "gallery-${DOMAIN}.conf" 2>/dev/null || true
  ok "VirtualHost engedélyezve: $CONF"
  APACHE_CTL="apache2ctl"
elif [[ -f /opt/lampp/bin/apachectl ]]; then
  # XAMPP
  XAMPP_EXTRA="/opt/lampp/etc/extra"
  mkdir -p "$XAMPP_EXTRA"
  cp "$SCRIPT_DIR/apache-vhost.conf" "$XAMPP_EXTRA/httpd-gallery.conf"
  HTTPD_CONF="/opt/lampp/etc/httpd.conf"
  if ! grep -q "httpd-gallery.conf" "$HTTPD_CONF"; then
    echo "" >> "$HTTPD_CONF"
    echo "# Galéria szerver VirtualHost" >> "$HTTPD_CONF"
    echo "Include /opt/lampp/etc/extra/httpd-gallery.conf" >> "$HTTPD_CONF"
    ok "Include hozzáadva: $HTTPD_CONF"
  fi
  # XAMPP-ban engedélyezni kell a proxy modulokat a httpd.conf-ban
  for mod in mod_proxy.so mod_proxy_http.so mod_ssl.so mod_rewrite.so mod_headers.so; do
    sed -i "s|^#LoadModule ${mod%.*}|LoadModule ${mod%.*}|" /opt/lampp/etc/httpd.conf || true
  done
  ok "XAMPP VirtualHost hozzáadva: $XAMPP_EXTRA/httpd-gallery.conf"
  APACHE_CTL="/opt/lampp/bin/apachectl"
else
  err "Apache2 és XAMPP sem található. Telepítsd az egyiket, majd futtasd újra."
fi

# ── certbot telepítése ────────────────────────────────────────────────────
if ! command -v certbot &>/dev/null; then
  log "certbot telepítése..."
  apt-get install -y certbot python3-certbot-apache
  ok "certbot telepítve"
fi

# ── Apache újraindítása (a certbot HTTP-01 challenge-hez szükséges) ───────
log "Apache újraindítása..."
"$APACHE_CTL" graceful 2>/dev/null || "$APACHE_CTL" restart
ok "Apache fut"

# ── Let's Encrypt tanúsítvány ─────────────────────────────────────────────
log "Let's Encrypt tanúsítvány kérése: ${DOMAIN}"
certbot --apache -d "$DOMAIN" \
  --redirect \
  --agree-tos \
  --register-unsafely-without-email \
  --non-interactive \
  || certbot --apache -d "$DOMAIN" --redirect

ok "Tanúsítvány beszerezve"

# ── Apache újraindítása HTTPS konfiggal ───────────────────────────────────
log "Apache végső újraindítása..."
"$APACHE_CTL" graceful 2>/dev/null || "$APACHE_CTL" restart
ok "Apache fut HTTPS-sel"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "HTTPS beállítva!"
echo "   URL   : https://${DOMAIN}"
echo "   Admin : https://${DOMAIN}/admin"
echo "   TLS   : Let's Encrypt (certbot, auto-megújítás 90 naponta)"
echo ""
echo "   A galériaszerverre (Node.js) külön kell elindítani:"
echo "     python start.py --no-browser"
echo "   Vagy systemd service-ként (lásd README.md #systemd szekció)."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_README_HTTPS_SECTION = """
## HTTPS beállítása (domain: __DOMAIN__)

A Node.js szerver a **__PORT__**-es porton fut;
a 80/443-as portokat a reverse proxy (Caddy vagy Apache/XAMPP) kezeli.

### Linux / macOS

```bash
python start.py --no-browser   # Node.js szerver indítása háttérben

sudo bash setup-https.sh       # HTTPS beállítása (egyszer)
```

### Windows

Nyisd meg a PowerShellt **rendszergazdaként** (jobb klikk → Futtatás rendszergazdaként):

```powershell
python start.py --no-browser   # Node.js szerver indítása háttérben

.\\setup-https.ps1             # HTTPS beállítása (egyszer)
```

XAMPP nem az alapértelmezett `C:\\xampp` mappában van? Add meg a -XamppPath paramétert:

```powershell
.\\setup-https.ps1 -XamppPath D:\\xampp
```

A szkriptek elvégzik:
- Szükséges eszköz telepítése (Caddy: `winget` / Linux: `apt`; Apache mód: win-acme / certbot)
- Reverse proxy konfig beállítása
- Let's Encrypt TLS tanúsítvány megszerzése
- Tanúsítvány **automatikus megújítása** (Caddy: beépített; Apache: systemd timer / Task Scheduler)

### Automatikus indítás

**Linux (systemd):**

```ini
# /etc/systemd/system/gallery.service
[Unit]
Description=Galéria szerver
After=network.target

[Service]
Type=simple
WorkingDirectory=__INSTALL_DIR__
ExecStart=/usr/bin/node __INSTALL_DIR__/server/index.js
Restart=always
RestartSec=5
Environment=PORT=__PORT__

[Install]
WantedBy=multi-user.target
```

```bash
sudo nano /etc/systemd/system/gallery.service   # töltsd ki az __INSTALL_DIR__-t
sudo systemctl enable --now gallery
```

**Windows (Feladatütemező):**

```powershell
$trigger = New-ScheduledTaskTrigger -AtLogOn
$action  = New-ScheduledTaskAction -Execute "node" -Argument "__INSTALL_DIR__\\server\\index.js" `
           -WorkingDirectory "__INSTALL_DIR__"
Register-ScheduledTask -TaskName "GaleriaServer" -Trigger $trigger -Action $action `
  -RunLevel Highest -Force
```

### XAMPP / más Apache aldomének

- **Apache mód**: XAMPP marad a 80/443-on, az `__DOMAIN__` aldomén egy VirtualHost-on
  keresztül proxyzódik a Node.js-re. Nincs port-ütközés.

- **Caddy mód**: A Caddy veszi át a 80/443-at. Ha XAMPP is fut, tedd át belső portra
  (pl. 8080), majd add hozzá a `Caddyfile`-hoz:

  ```
  masik.kncsk.hu {
      reverse_proxy localhost:8080
  }
  ```
"""

_SETUP_HTTPS_CADDY_PS1 = r"""# setup-https.ps1 — Caddy telepítése + galéria HTTPS beállítása (Windows)
# Futtatás: adminisztratori PowerShell-ben: .\setup-https.ps1
# Követelmény: __DOMAIN__ DNS-ben már erre a szerverre mutat,
#              80/443-as port szabad (ha XAMPP fut a 80/443-on, lásd README.md).

param(
    [string]$Domain = "__DOMAIN__",
    [int]   $Port   = __PORT__
)
$ErrorActionPreference = "Stop"

function Log { Write-Host "[https-setup] $args" -ForegroundColor Cyan  }
function Ok  { Write-Host "[     ok     ] $args" -ForegroundColor Green }
function Err { Write-Host "[    hiba    ] $args" -ForegroundColor Red; exit 1 }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Err "Futtasd PowerShell adminisztrátorként (jobb klikk → Futtatás rendszergazdaként)." }

# ── Caddy telepítése ──────────────────────────────────────────────────────
$CaddyDir = "C:\caddy"
$CaddyExe = "$CaddyDir\caddy.exe"

if (-not (Get-Command caddy -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Caddy telepítése (winget)..."
        winget install --id Caddyserver.Caddy --accept-package-agreements --accept-source-agreements
        $CaddyExe = (Get-Command caddy).Source
        Ok "Caddy telepítve (winget)"
    } else {
        Log "Caddy letöltése (közvetlen)..."
        New-Item -ItemType Directory -Force -Path $CaddyDir | Out-Null
        $ZipPath = "$env:TEMP\caddy.zip"
        # Mindig a legfrissebb stable releaset tölti le
        $apiUrl = "https://api.github.com/repos/caddyserver/caddy/releases/latest"
        $tag = (Invoke-RestMethod $apiUrl).tag_name
        $dlUrl = "https://github.com/caddyserver/caddy/releases/download/$tag/caddy_$($tag.TrimStart('v'))_windows_amd64.zip"
        Invoke-WebRequest -Uri $dlUrl -OutFile $ZipPath
        Expand-Archive -Path $ZipPath -DestinationPath $CaddyDir -Force
        $CaddyExe = "$CaddyDir\caddy.exe"
        # PATH-hoz adás
        $machinePath = [Environment]::GetEnvironmentVariable("PATH", "Machine")
        if ($machinePath -notlike "*$CaddyDir*") {
            [Environment]::SetEnvironmentVariable("PATH", "$machinePath;$CaddyDir", "Machine")
        }
        Ok "Caddy letöltve: $CaddyExe"
    }
} else {
    $CaddyExe = (Get-Command caddy).Source
    Ok "Caddy már telepítve: $(& $CaddyExe version)"
}

# ── Tűzfal szabályok ──────────────────────────────────────────────────────
Log "Tűzfal szabályok beállítása (80, 443)..."
foreach ($p in @(80, 443)) {
    $rule = Get-NetFirewallRule -DisplayName "Caddy port $p" -ErrorAction SilentlyContinue
    if (-not $rule) {
        New-NetFirewallRule -DisplayName "Caddy port $p" -Direction Inbound -Protocol TCP -LocalPort $p -Action Allow | Out-Null
    }
}
Ok "Tűzfal kész"

# ── Caddyfile másolása ────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $CaddyDir | Out-Null
Copy-Item "$ScriptDir\Caddyfile" "$CaddyDir\Caddyfile" -Force
Ok "Caddyfile → $CaddyDir\Caddyfile"

# ── Caddy Windows Service regisztrálása ───────────────────────────────────
$svcName = "caddy"
$existingSvc = Get-Service -Name $svcName -ErrorAction SilentlyContinue

if ($existingSvc) {
    Log "Caddy service újraindítása..."
    Restart-Service $svcName
} else {
    Log "Caddy service regisztrálása..."
    Push-Location $CaddyDir
    & $CaddyExe service install
    Pop-Location
    Start-Service $svcName
}
Ok "Caddy service fut"

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Ok "HTTPS beállítva!"
Write-Host "   URL   : https://$Domain"
Write-Host "   Admin : https://$Domain/admin"
Write-Host "   TLS   : automatikus Let's Encrypt (Caddy, auto-megujulas)"
Write-Host ""
Write-Host "   Galeriaszervert kulon kell inditani:"
Write-Host "     python start.py --no-browser"
Write-Host "   Automatikus inditashoz lasd README.md (Feladatutemezo szekciо)."
Write-Host ("=" * 65) -ForegroundColor Green
"""

_SETUP_HTTPS_APACHE_PS1 = r"""# setup-https.ps1 — XAMPP + Let's Encrypt (win-acme) beállítása Windows-on
# Futtatás: adminisztratori PowerShell-ben: .\setup-https.ps1
# Követelmény: XAMPP telepítve, __DOMAIN__ DNS-ben erre a szerverre mutat.

param(
    [string]$Domain    = "__DOMAIN__",
    [int]   $Port      = __PORT__,
    [string]$XamppPath = ""          # pl. -XamppPath D:\xampp, ha nem C:\xampp
)
$ErrorActionPreference = "Stop"

function Log { Write-Host "[https-setup] $args" -ForegroundColor Cyan  }
function Ok  { Write-Host "[     ok     ] $args" -ForegroundColor Green }
function Err { Write-Host "[    hiba    ] $args" -ForegroundColor Red; exit 1 }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Err "Futtasd PowerShell adminisztrátorként (jobb klikk → Futtatás rendszergazdaként)." }

# ── XAMPP keresése ────────────────────────────────────────────────────────
if ($XamppPath -and -not (Test-Path "$XamppPath\apache\conf\httpd.conf")) {
    Err "A megadott XamppPath-on nem található httpd.conf: $XamppPath"
}
if (-not $XamppPath) {
    $candidates = @("C:\xampp", "D:\xampp", "C:\Program Files\xampp", "C:\Program Files (x86)\xampp")
    $XamppPath = $candidates | Where-Object { Test-Path "$_\apache\conf\httpd.conf" } | Select-Object -First 1
    if (-not $XamppPath) { Err "XAMPP nem található. Add meg a -XamppPath paramétert." }
}
Ok "XAMPP: $XamppPath"

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$HttpdConf  = "$XamppPath\apache\conf\httpd.conf"
$ExtraDir   = "$XamppPath\apache\conf\extra"
$VhostConf  = "$ExtraDir\httpd-gallery.conf"
$ApacheCtl  = "$XamppPath\apache\bin\httpd.exe"
$Htdocs     = "$XamppPath\htdocs"

# ── mod_proxy modulok engedélyezése ───────────────────────────────────────
Log "mod_proxy modulok engedélyezése a httpd.conf-ban..."
$content = Get-Content $HttpdConf -Raw
@("mod_proxy","mod_proxy_http","mod_ssl","mod_rewrite","mod_headers") | ForEach-Object {
    # Komment eltávolítása a LoadModule sorról
    $content = $content -replace "(?m)^#(LoadModule\s+${_}_module\b)", '$1'
}
[IO.File]::WriteAllText($HttpdConf, $content)
Ok "Modulok engedélyezve"

# ── VirtualHost konfig másolása ───────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $ExtraDir | Out-Null
Copy-Item "$ScriptDir\apache-vhost.conf" $VhostConf -Force
Ok "VirtualHost konfig → $VhostConf"

# Include hozzáadása httpd.conf-hoz
$includeRelPath = "conf/extra/httpd-gallery.conf"
if ((Get-Content $HttpdConf -Raw) -notlike "*httpd-gallery.conf*") {
    Add-Content $HttpdConf "`r`n# Galeria szerver VirtualHost`r`nInclude $includeRelPath"
    Ok "Include hozzáadva a httpd.conf-hoz"
}

# ── Apache újraindítása ───────────────────────────────────────────────────
Log "Apache újraindítása..."
$apacheSvc = Get-Service -DisplayName "*Apache*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($apacheSvc) {
    Restart-Service $apacheSvc.Name
} else {
    & $ApacheCtl -k restart 2>$null
}
Start-Sleep 2
Ok "Apache fut"

# ── win-acme telepítése ───────────────────────────────────────────────────
$WacsDir = "$env:ProgramData\win-acme"
$WacsExe = Get-ChildItem $WacsDir -Filter "wacs.exe" -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName

if (-not $WacsExe) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "win-acme telepítése (winget)..."
        winget install --id win-acme.win-acme --accept-package-agreements --accept-source-agreements
        $WacsExe = Get-Command wacs -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source
    }
    if (-not $WacsExe) {
        Log "win-acme letöltése..."
        New-Item -ItemType Directory -Force -Path $WacsDir | Out-Null
        $apiUrl = "https://api.github.com/repos/win-acme/win-acme/releases/latest"
        $rel    = Invoke-RestMethod $apiUrl
        $asset  = $rel.assets | Where-Object { $_.name -like "*x64.pluggable.zip" } | Select-Object -First 1
        $zip    = "$env:TEMP\win-acme.zip"
        Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip
        Expand-Archive -Path $zip -DestinationPath $WacsDir -Force
        $WacsExe = Get-ChildItem $WacsDir -Filter "wacs.exe" -Recurse | Select-Object -First 1 -ExpandProperty FullName
        Ok "win-acme letöltve: $WacsExe"
    }
} else {
    Ok "win-acme megtalálva: $WacsExe"
}

# ── Let's Encrypt tanúsítvány (HTTP-01 webroot) ───────────────────────────
$SslDir = "$XamppPath\apache\conf\ssl\$Domain"
New-Item -ItemType Directory -Force -Path $SslDir | Out-Null

Log "Let's Encrypt tanúsítvány kérése: $Domain"
& $WacsExe `
    --target          manual `
    --host            $Domain `
    --validation      selfhosting `
    --store           pemfiles `
    --pemfilespath    $SslDir `
    --accepttos `
    --emailaddress    "" `
    --notaskscheduler:$false

Ok "Tanúsítvány: $SslDir"

# ── SSL VirtualHost hozzáadása ────────────────────────────────────────────
$sslBlock = @"

<VirtualHost *:443>
    ServerName $Domain
    SSLEngine             On
    SSLCertificateFile    "$SslDir\$Domain-crt.pem"
    SSLCertificateKeyFile "$SslDir\$Domain-key.pem"

    ProxyPreserveHost On
    ProxyPass        / http://localhost:$Port/
    ProxyPassReverse / http://localhost:$Port/

    RequestHeader set X-Forwarded-Proto "https"
</VirtualHost>
"@

if ((Get-Content $VhostConf -Raw) -notlike "*:443*") {
    Add-Content $VhostConf $sslBlock
    Ok "HTTPS VirtualHost hozzáadva"
}

# ── Apache végső újraindítása ─────────────────────────────────────────────
if ($apacheSvc) { Restart-Service $apacheSvc.Name } else { & $ApacheCtl -k restart 2>$null }
Ok "Apache újraindítva HTTPS-sel"

Write-Host ""
Write-Host ("=" * 65) -ForegroundColor Green
Ok "HTTPS beállítva!"
Write-Host "   URL   : https://$Domain"
Write-Host "   Admin : https://$Domain/admin"
Write-Host "   TLS   : Let's Encrypt (win-acme, auto-megujulas Task Schedulerrel)"
Write-Host ""
Write-Host "   Galeriaszervert kulon kell inditani:"
Write-Host "     python start.py --no-browser"
Write-Host ("=" * 65) -ForegroundColor Green
"""

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
__HTTPS_SECTION__"""
