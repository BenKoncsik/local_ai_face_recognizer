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
#   "none"       — HTTP only, no reverse-proxy config generated
#   "caddy"      — Caddyfile + setup-https.sh (Caddy handles 80/443 + ACME)
#   "apache"     — apache-vhost.conf + setup-https.sh (XAMPP/Apache + certbot)
#   "cloudflare" — Cloudflare Tunnel (cloudflared): public HTTPS with NO open
#                  ports, works behind CGNAT / mobile internet. No Let's Encrypt.
HTTPS_MODES = ("none", "caddy", "apache", "cloudflare")


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
        smtp_allow_self_signed: bool = False,
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
            "smtpAllowSelfSigned": bool(smtp_allow_self_signed),
        }
        if https_enabled:
            config["domain"] = domain
            config["httpsEnabled"] = True
            config["httpsMode"] = https_mode
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
            section = (_README_TUNNEL_SECTION if https_mode == "cloudflare"
                       else _README_HTTPS_SECTION)
            readme = readme.replace("__HTTPS_SECTION__", section
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

        def _sh(name: str, content: str) -> None:
            p = out / name
            p.write_text(_sub(content))
            try:
                p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            except Exception:
                pass

        if https_mode == "caddy":
            (out / "Caddyfile").write_text(_sub(_CADDYFILE))
            _sh("setup-https.sh",      _SETUP_HTTPS_CADDY)
            _sh("uninstall-https.sh",  _UNINSTALL_HTTPS_CADDY)
            (out / "setup-https.ps1").write_text(_sub(_SETUP_HTTPS_CADDY_PS1))
            (out / "uninstall-https.ps1").write_text(_sub(_UNINSTALL_HTTPS_CADDY_PS1))

        elif https_mode == "apache":
            (out / "apache-vhost.conf").write_text(_sub(_APACHE_VHOST))
            _sh("setup-https.sh",      _SETUP_HTTPS_APACHE)
            _sh("uninstall-https.sh",  _UNINSTALL_HTTPS_APACHE)
            (out / "setup-https.ps1").write_text(_sub(_SETUP_HTTPS_APACHE_PS1))
            (out / "uninstall-https.ps1").write_text(_sub(_UNINSTALL_HTTPS_APACHE_PS1))

        elif https_mode == "cloudflare":
            _sh("setup-tunnel.sh",      _SETUP_TUNNEL_SH)
            _sh("uninstall-tunnel.sh",  _UNINSTALL_TUNNEL_SH)
            (out / "setup-tunnel.ps1").write_text(_sub(_SETUP_TUNNEL_PS1))
            (out / "uninstall-tunnel.ps1").write_text(_sub(_UNINSTALL_TUNNEL_PS1))


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
# setup-https.sh — Caddy telepítése + galéria HTTPS beállítása (Linux / macOS)
# Linux: sudo bash setup-https.sh
# macOS: sudo bash setup-https.sh   (sudo kell a 80/443-hoz)
# Követelmény: __DOMAIN__ DNS-ben már erre a szerverre mutat,
#              80/443-as port szabad (ha XAMPP fut, lásd README.md).
set -euo pipefail

DOMAIN="__DOMAIN__"
PORT="__PORT__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

log() { echo -e "\033[1;34m[https-setup]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[     ok     ]\033[0m $*"; }
err() { echo -e "\033[1;31m[    hiba    ]\033[0m $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "Root jogosultság szükséges: sudo bash setup-https.sh"

# macOS: a brew install nem futhat root-ként — az eredeti felhasználó nevében futtatjuk
REAL_USER="${SUDO_USER:-$USER}"

# ── Port-ellenőrzés ───────────────────────────────────────────────────────
for p in 80 443; do
  if lsof -iTCP:${p} -sTCP:LISTEN -n -P 2>/dev/null | grep -q "." || \
     ss -tlnp 2>/dev/null | grep -q ":${p} "; then
    echo ""
    echo "⚠  A ${p}-as port már foglalt."
    echo "   Ha XAMPP/Apache fut ezen a porton:"
    echo "   1) XAMPP-ot állítsd át más portra (pl. 8080), majd futtasd újra."
    echo "   2) Vagy használd az Apache/XAMPP módot (apache-vhost.conf)."
    echo ""
    read -rp "Folytatás ugyanígy? (i/n): " yn
    [[ "$yn" =~ ^[Ii] ]] || exit 0
  fi
done

# ── Caddy telepítése ──────────────────────────────────────────────────────
if [[ "$OS" == "Darwin" ]]; then
  # brew install nem futhat root-ként → felhasználó nevében
  BREW_BIN="$(sudo -u "$REAL_USER" command -v brew 2>/dev/null || echo "")"
  [[ -z "$BREW_BIN" ]] && err "Homebrew szükséges: https://brew.sh"
  if ! sudo -u "$REAL_USER" "$BREW_BIN" list caddy &>/dev/null 2>&1; then
    log "Caddy telepítése (Homebrew, felhasználó: $REAL_USER)..."
    sudo -u "$REAL_USER" "$BREW_BIN" install caddy
    ok "Caddy telepítve"
  else
    ok "Caddy már telepítve"
  fi
  BREW_PREFIX="$(sudo -u "$REAL_USER" "$BREW_BIN" --prefix)"
else
  if ! command -v caddy &>/dev/null; then
    log "Caddy telepítése (apt)..."
    apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
      | tee /etc/apt/sources.list.d/caddy-stable.list
    apt-get update && apt-get install -y caddy
    ok "Caddy telepítve"
  else
    ok "Caddy már telepítve: $(caddy version)"
  fi
fi

# ── Caddyfile másolása ────────────────────────────────────────────────────
if [[ "$OS" == "Darwin" ]]; then
  CADDY_CFG="$BREW_PREFIX/etc/caddy/Caddyfile"
  mkdir -p "$(dirname "$CADDY_CFG")"
else
  CADDY_CFG="/etc/caddy/Caddyfile"
fi
cp "$SCRIPT_DIR/Caddyfile" "$CADDY_CFG"
ok "Caddyfile → $CADDY_CFG"

# ── Caddy (újra)indítása ──────────────────────────────────────────────────
if [[ "$OS" == "Darwin" ]]; then
  # sudo brew services → LaunchDaemon (root-ként fut, port 80/443 OK)
  "$BREW_PREFIX/bin/brew" services restart caddy
  ok "Caddy fut (launchd, LaunchDaemon)"
else
  if systemctl is-active --quiet caddy 2>/dev/null; then
    systemctl reload caddy
  else
    systemctl enable --now caddy
  fi
  ok "Caddy fut (systemd)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "HTTPS beállítva!"
echo "   URL    : https://${DOMAIN}"
echo "   Admin  : https://${DOMAIN}/admin"
echo "   TLS    : automatikus Let's Encrypt (Caddy, auto-megújítás)"
echo "   Eltávolítás: sudo bash uninstall-https.sh"
echo ""
echo "   Node.js szerver indítása (külön terminálban):"
echo "     python start.py --no-browser"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_APACHE_VHOST = """\
# Galéria — Apache VirtualHost konfig (XAMPP / Apache2 mellé)
# A setup-https.sh / setup-https.ps1 automatikusan a megfelelő helyre másolja.
#
# Kézi másolás helye (ha szükséges):
#   Ubuntu/Debian Apache2:  /etc/apache2/sites-available/gallery-__DOMAIN__.conf
#   XAMPP Linux:            /opt/lampp/etc/extra/httpd-gallery.conf
#   XAMPP macOS:            /Applications/XAMPP/xamppfiles/etc/extra/httpd-gallery.conf
#   Homebrew Apache (macOS):(brew --prefix)/etc/httpd/extra/httpd-gallery.conf
#   XAMPP Windows:          C:\\xampp\\apache\\conf\\extra\\httpd-gallery.conf
#
# Let's Encrypt (certbot) a setup szkript futtatása után hozzáadja a *:443 blokkot.

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
# setup-https.sh — Apache/XAMPP + Let's Encrypt beállítása (Linux / macOS)
# Futtatás: sudo bash setup-https.sh
# Követelmény: Apache2 vagy XAMPP telepítve, root jogosultság,
#              __DOMAIN__ DNS-ben már erre a szerverre mutat.
set -euo pipefail

DOMAIN="__DOMAIN__"
PORT="__PORT__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

log() { echo -e "\033[1;34m[https-setup]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[     ok     ]\033[0m $*"; }
err() { echo -e "\033[1;31m[    hiba    ]\033[0m $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "Root jogosultság szükséges: sudo bash setup-https.sh"

# macOS: brew nem futhat root-ként → eredeti felhasználó nevében
REAL_USER="${SUDO_USER:-$USER}"

# ── Apache / XAMPP helyszín detektálása ───────────────────────────────────
if [[ "$OS" == "Darwin" ]]; then
  # macOS: XAMPP for Mac
  XAMPP_MAC="/Applications/XAMPP/xamppfiles"
  BREW_BIN="$(sudo -u "$REAL_USER" command -v brew 2>/dev/null || echo "")"
  BREW_PREFIX="$( [[ -n "$BREW_BIN" ]] && sudo -u "$REAL_USER" "$BREW_BIN" --prefix || echo /opt/homebrew)"
  if [[ -f "$XAMPP_MAC/etc/httpd.conf" ]]; then
    HTTPD_CONF="$XAMPP_MAC/etc/httpd.conf"
    EXTRA_DIR="$XAMPP_MAC/etc/extra"
    APACHE_CTL="$XAMPP_MAC/bin/apachectl"
    VHOST_FILE="$EXTRA_DIR/httpd-gallery.conf"
    CERTBOT_MODE="standalone"          # XAMPP-hoz nincs apache plugin
    CERTBOT_INSTALL="sudo -u \"$REAL_USER\" \"$BREW_BIN\" install certbot"
    SITES_AVAILABLE=""
  elif [[ -d "$BREW_PREFIX/etc/httpd" ]]; then
    # Homebrew Apache
    HTTPD_CONF="$BREW_PREFIX/etc/httpd/httpd.conf"
    EXTRA_DIR="$BREW_PREFIX/etc/httpd/extra"
    APACHE_CTL="$BREW_PREFIX/bin/apachectl"
    VHOST_FILE="$EXTRA_DIR/httpd-gallery.conf"
    CERTBOT_MODE="standalone"
    CERTBOT_INSTALL="sudo -u \"$REAL_USER\" \"$BREW_BIN\" install certbot"
    SITES_AVAILABLE=""
  else
    err "XAMPP for Mac (/Applications/XAMPP) vagy Homebrew httpd szükséges."
  fi
elif [[ -d "/etc/apache2/sites-available" ]]; then
  # Ubuntu/Debian standard Apache2
  SITES_AVAILABLE="/etc/apache2/sites-available"
  VHOST_FILE="$SITES_AVAILABLE/gallery-${DOMAIN}.conf"
  APACHE_CTL="apache2ctl"
  CERTBOT_MODE="apache"
  CERTBOT_INSTALL="apt-get install -y certbot python3-certbot-apache"
  HTTPD_CONF=""
  EXTRA_DIR=""
elif [[ -f "/opt/lampp/bin/apachectl" ]]; then
  # XAMPP Linux
  HTTPD_CONF="/opt/lampp/etc/httpd.conf"
  EXTRA_DIR="/opt/lampp/etc/extra"
  APACHE_CTL="/opt/lampp/bin/apachectl"
  VHOST_FILE="$EXTRA_DIR/httpd-gallery.conf"
  CERTBOT_MODE="standalone"
  CERTBOT_INSTALL="apt-get install -y certbot"
  SITES_AVAILABLE=""
else
  err "Apache2 és XAMPP sem található. Telepítsd az egyiket, majd futtasd újra."
fi

# ── mod_proxy modulok engedélyezése ──────────────────────────────────────
if [[ -n "$SITES_AVAILABLE" ]] && command -v a2enmod &>/dev/null; then
  log "Apache modulok engedélyezése (a2enmod)..."
  a2enmod proxy proxy_http rewrite ssl headers 2>/dev/null
  ok "Modulok engedélyezve"
elif [[ -n "$HTTPD_CONF" ]]; then
  log "mod_proxy modulok engedélyezése a httpd.conf-ban..."
  SED_I="sed -i"
  [[ "$OS" == "Darwin" ]] && SED_I="sed -i ''"
  for mod in proxy proxy_http ssl rewrite headers; do
    $SED_I "s|^#LoadModule ${mod}_module|LoadModule ${mod}_module|" "$HTTPD_CONF" 2>/dev/null || true
  done
  ok "Modulok engedélyezve"
fi

# ── VirtualHost konfig másolása ───────────────────────────────────────────
[[ -n "$EXTRA_DIR" ]] && mkdir -p "$EXTRA_DIR"
cp "$SCRIPT_DIR/apache-vhost.conf" "$VHOST_FILE"
ok "VirtualHost konfig → $VHOST_FILE"

if [[ -n "$SITES_AVAILABLE" ]]; then
  a2ensite "gallery-${DOMAIN}.conf" 2>/dev/null || true
  ok "VirtualHost site engedélyezve"
elif [[ -n "$HTTPD_CONF" ]] && ! grep -q "httpd-gallery.conf" "$HTTPD_CONF"; then
  printf "\n# Galéria szerver VirtualHost\nInclude %s\n" "$VHOST_FILE" >> "$HTTPD_CONF"
  ok "Include hozzáadva: $HTTPD_CONF"
fi

# ── certbot telepítése ────────────────────────────────────────────────────
if ! command -v certbot &>/dev/null; then
  log "certbot telepítése..."
  eval "$CERTBOT_INSTALL"
  ok "certbot telepítve"
fi

# ── Apache újraindítása ───────────────────────────────────────────────────
log "Apache indítása..."
"$APACHE_CTL" graceful 2>/dev/null || "$APACHE_CTL" start
ok "Apache fut"

# ── Let's Encrypt tanúsítvány ─────────────────────────────────────────────
log "Let's Encrypt tanúsítvány kérése: ${DOMAIN}"
if [[ "$CERTBOT_MODE" == "apache" ]]; then
  # Ubuntu: certbot automatikusan konfigurálja az Apache-t
  certbot --apache -d "$DOMAIN" --redirect --agree-tos \
    --register-unsafely-without-email --non-interactive \
    || certbot --apache -d "$DOMAIN" --redirect
else
  # XAMPP / Homebrew: standalone mód (rövid Apache-leállás)
  log "Apache rövid leállítása a standalone challenge-hez..."
  "$APACHE_CTL" stop 2>/dev/null || true
  certbot certonly --standalone -d "$DOMAIN" --agree-tos \
    --register-unsafely-without-email --non-interactive \
    || certbot certonly --standalone -d "$DOMAIN"
  "$APACHE_CTL" start
  ok "Tanúsítvány → /etc/letsencrypt/live/${DOMAIN}/"

  # SSL VirtualHost blokk hozzáadása
  CERT_DIR="/etc/letsencrypt/live/${DOMAIN}"
  if ! grep -q ":443" "$VHOST_FILE"; then
    cat >> "$VHOST_FILE" <<SSLBLOCK

<VirtualHost *:443>
    ServerName ${DOMAIN}
    SSLEngine             On
    SSLCertificateFile    ${CERT_DIR}/fullchain.pem
    SSLCertificateKeyFile ${CERT_DIR}/privkey.pem

    ProxyPreserveHost On
    ProxyPass        / http://localhost:${PORT}/
    ProxyPassReverse / http://localhost:${PORT}/
    RequestHeader set X-Forwarded-Proto "https"
</VirtualHost>
SSLBLOCK
    ok "HTTPS VirtualHost hozzáadva"
  fi
fi

# ── Apache végső újraindítása ─────────────────────────────────────────────
"$APACHE_CTL" graceful 2>/dev/null || "$APACHE_CTL" restart
ok "Apache fut HTTPS-sel"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "HTTPS beállítva!"
echo "   URL    : https://${DOMAIN}"
echo "   Admin  : https://${DOMAIN}/admin"
echo "   TLS    : Let's Encrypt (certbot, 90 naponta auto-megújítás)"
echo "   Eltávolítás: sudo bash uninstall-https.sh"
echo ""
echo "   Node.js szerver indítása (külön terminálban):"
echo "     python start.py --no-browser"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_UNINSTALL_HTTPS_CADDY = r"""#!/usr/bin/env bash
# uninstall-https.sh — Caddy leállítása és eltávolítása (Linux / macOS)
# Linux: sudo bash uninstall-https.sh
# macOS: sudo bash uninstall-https.sh
set -euo pipefail

DOMAIN="__DOMAIN__"
OS="$(uname -s)"

log() { echo -e "\033[1;34m[uninstall]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[   ok    ]\033[0m $*"; }

[[ $EUID -ne 0 ]] && { echo "sudo szükséges: sudo bash uninstall-https.sh"; exit 1; }

# macOS: brew install/uninstall nem futhat root-ként
REAL_USER="${SUDO_USER:-$USER}"

if [[ "$OS" == "Darwin" ]]; then
  BREW_BIN="$(sudo -u "$REAL_USER" command -v brew 2>/dev/null || echo "")"
  BREW_PREFIX="$( [[ -n "$BREW_BIN" ]] && sudo -u "$REAL_USER" "$BREW_BIN" --prefix || echo /opt/homebrew)"
  log "Caddy leállítása (launchd)..."
  # sudo brew services → LaunchDaemon management (root OK)
  "$BREW_PREFIX/bin/brew" services stop caddy 2>/dev/null || true
  CFG="$BREW_PREFIX/etc/caddy/Caddyfile"
  [[ -f "$CFG" ]] && rm "$CFG" && ok "Caddyfile törölve: $CFG"
  read -rp "Caddy teljesen eltávolítása (brew uninstall)? (i/n): " yn
  [[ "$yn" =~ ^[Ii] ]] && sudo -u "$REAL_USER" "$BREW_BIN" uninstall caddy && ok "Caddy eltávolítva"
else
  log "Caddy leállítása (systemd)..."
  systemctl stop    caddy 2>/dev/null || true
  systemctl disable caddy 2>/dev/null || true
  [[ -f /etc/caddy/Caddyfile ]] && rm /etc/caddy/Caddyfile && ok "Caddyfile törölve"
  read -rp "Caddy teljesen eltávolítása (apt remove)? (i/n): " yn
  [[ "$yn" =~ ^[Ii] ]] && apt-get remove -y caddy && ok "Caddy eltávolítva"
fi

echo ""
ok "Kész — Let's Encrypt tanúsítvány automatikusan lejár 90 nap múlva."
echo "   (Azonnali visszavonás: certbot revoke --cert-name ${DOMAIN})"
"""

_UNINSTALL_HTTPS_APACHE = r"""#!/usr/bin/env bash
# uninstall-https.sh — Apache/XAMPP VirtualHost + Let's Encrypt eltávolítása (Linux / macOS)
# Futtatás: sudo bash uninstall-https.sh
set -euo pipefail

DOMAIN="__DOMAIN__"
OS="$(uname -s)"

log() { echo -e "\033[1;34m[uninstall]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[   ok    ]\033[0m $*"; }
err() { echo -e "\033[1;31m[  hiba   ]\033[0m $*" >&2; exit 1; }

[[ $EUID -ne 0 ]] && err "sudo szükséges: sudo bash uninstall-https.sh"

REAL_USER="${SUDO_USER:-$USER}"

# ── Apache helyszín detektálása ───────────────────────────────────────────
if [[ "$OS" == "Darwin" ]]; then
  BREW_BIN="$(sudo -u "$REAL_USER" command -v brew 2>/dev/null || echo "")"
  BREW_PREFIX="$( [[ -n "$BREW_BIN" ]] && sudo -u "$REAL_USER" "$BREW_BIN" --prefix || echo /opt/homebrew)"
  XAMPP_MAC="/Applications/XAMPP/xamppfiles"
  if [[ -f "$XAMPP_MAC/etc/httpd.conf" ]]; then
    HTTPD_CONF="$XAMPP_MAC/etc/httpd.conf"
    VHOST_FILE="$XAMPP_MAC/etc/extra/httpd-gallery.conf"
    APACHE_CTL="$XAMPP_MAC/bin/apachectl"
    SITES_AVAILABLE=""
  elif [[ -d "$BREW_PREFIX/etc/httpd" ]]; then
    HTTPD_CONF="$BREW_PREFIX/etc/httpd/httpd.conf"
    VHOST_FILE="$BREW_PREFIX/etc/httpd/extra/httpd-gallery.conf"
    APACHE_CTL="$BREW_PREFIX/bin/apachectl"
    SITES_AVAILABLE=""
  else
    err "XAMPP for Mac vagy Homebrew httpd nem található."
  fi
elif [[ -d "/etc/apache2/sites-available" ]]; then
  SITES_AVAILABLE="/etc/apache2/sites-available"
  VHOST_FILE="$SITES_AVAILABLE/gallery-${DOMAIN}.conf"
  APACHE_CTL="apache2ctl"
  HTTPD_CONF=""
elif [[ -f "/opt/lampp/bin/apachectl" ]]; then
  HTTPD_CONF="/opt/lampp/etc/httpd.conf"
  VHOST_FILE="/opt/lampp/etc/extra/httpd-gallery.conf"
  APACHE_CTL="/opt/lampp/bin/apachectl"
  SITES_AVAILABLE=""
else
  err "Apache2 / XAMPP nem található."
fi

# ── VirtualHost konfig eltávolítása ───────────────────────────────────────
if [[ -f "$VHOST_FILE" ]]; then
  [[ -n "$SITES_AVAILABLE" ]] && a2dissite "gallery-${DOMAIN}.conf" 2>/dev/null || true
  rm "$VHOST_FILE"
  ok "VirtualHost konfig törölve: $VHOST_FILE"
fi

if [[ -n "$HTTPD_CONF" ]] && grep -q "httpd-gallery.conf" "$HTTPD_CONF"; then
  SED_I="sed -i"
  [[ "$OS" == "Darwin" ]] && SED_I="sed -i ''"
  $SED_I '/# Galéria szerver VirtualHost/d' "$HTTPD_CONF"
  $SED_I '/httpd-gallery\.conf/d'           "$HTTPD_CONF"
  ok "Include eltávolítva: $HTTPD_CONF"
fi

# ── certbot tanúsítvány ───────────────────────────────────────────────────
if command -v certbot &>/dev/null; then
  read -rp "Let's Encrypt tanúsítvány törlése ($DOMAIN)? (i/n): " yn
  if [[ "$yn" =~ ^[Ii] ]]; then
    certbot delete --cert-name "$DOMAIN" --non-interactive 2>/dev/null || true
    ok "Tanúsítvány törölve"
  fi
fi

# ── Apache újraindítása ───────────────────────────────────────────────────
"$APACHE_CTL" graceful 2>/dev/null || "$APACHE_CTL" restart || true
ok "Apache újraindítva"

echo ""
ok "Kész — a galéria aldomén (${DOMAIN}) le van tiltva."
"""

_UNINSTALL_HTTPS_CADDY_PS1 = r"""# uninstall-https.ps1 — Caddy leállítása és eltávolítása (Windows)
# Futtatás: adminisztrátori PowerShell: .\uninstall-https.ps1
param([string]$Domain = "__DOMAIN__")
$ErrorActionPreference = "Stop"

function Log { Write-Host "[uninstall] $args" -ForegroundColor Cyan  }
function Ok  { Write-Host "[   ok    ] $args" -ForegroundColor Green }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Write-Error "Adminisztrátori PowerShell szükséges."; exit 1 }

# ── Caddy ütemezett feladat leállítása ────────────────────────────────────
$TaskName = "CaddyGallery"
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    Log "Caddy ütemezett feladat leállítása és törlése..."
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Ok "Ütemezett feladat törölve: $TaskName"
}
# Régi (hibás) "caddy" Windows service maradék takarítása
$svc = Get-Service -Name "caddy" -ErrorAction SilentlyContinue
if ($svc) { Stop-Service caddy -Force -ErrorAction SilentlyContinue; & sc.exe delete caddy | Out-Null }
# Háttérben futó Caddy leállítása
$caddy = (Get-Command caddy -ErrorAction SilentlyContinue)?.Source
if ($caddy) { & $caddy stop 2>$null | Out-Null }

# ── Caddyfile törlése ─────────────────────────────────────────────────────
foreach ($p in @("C:\caddy\Caddyfile", "$env:ProgramData\caddy\Caddyfile")) {
    if (Test-Path $p) { Remove-Item $p -Force; Ok "Törölve: $p" }
}

# ── Tűzfal szabályok eltávolítása ────────────────────────────────────────
foreach ($p in @(80, 443)) {
    Remove-NetFirewallRule -DisplayName "Caddy port $p" -ErrorAction SilentlyContinue
}
Ok "Tűzfal szabályok törölve"

# ── Opcionális teljes eltávolítás ─────────────────────────────────────────
$yn = Read-Host "Caddy teljesen eltávolítása (winget)? (i/n)"
if ($yn -match "^[Ii]") {
    winget uninstall --id Caddyserver.Caddy 2>$null
    foreach ($d in @("C:\caddy", "$env:ProgramData\caddy")) {
        Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue
    }
    Ok "Caddy eltávolítva"
}

Write-Host ""
Ok "Kész — Let's Encrypt tanusítvány 90 nap múlva jár le automatikusan."
"""

_UNINSTALL_HTTPS_APACHE_PS1 = r"""# uninstall-https.ps1 — XAMPP VirtualHost + win-acme tanúsítvány eltávolítása (Windows)
# Futtatás: adminisztrátori PowerShell: .\uninstall-https.ps1
param(
    [string]$Domain    = "__DOMAIN__",
    [string]$XamppPath = ""
)
$ErrorActionPreference = "Stop"

function Log { Write-Host "[uninstall] $args" -ForegroundColor Cyan  }
function Ok  { Write-Host "[   ok    ] $args" -ForegroundColor Green }
function Err { Write-Host "[  hiba   ] $args" -ForegroundColor Red; exit 1 }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { Err "Adminisztrátori PowerShell szükséges." }

# ── XAMPP keresése ────────────────────────────────────────────────────────
if (-not $XamppPath) {
    $XamppPath = @("C:\xampp","D:\xampp","C:\Program Files\xampp") |
        Where-Object { Test-Path "$_\apache\conf\httpd.conf" } | Select-Object -First 1
    if (-not $XamppPath) { Err "XAMPP nem található. Add meg: -XamppPath C:\xampp" }
}
Ok "XAMPP: $XamppPath"

$VhostConf = "$XamppPath\apache\conf\extra\httpd-gallery.conf"
$HttpdConf  = "$XamppPath\apache\conf\httpd.conf"

# ── VirtualHost konfig törlése ────────────────────────────────────────────
if (Test-Path $VhostConf) {
    Remove-Item $VhostConf -Force
    Ok "VirtualHost konfig törölve: $VhostConf"
}

# ── Include sor törlése httpd.conf-ból ───────────────────────────────────
if (Test-Path $HttpdConf) {
    $raw = Get-Content $HttpdConf -Raw
    $raw = $raw -replace "(?m)\r?\n# Galeria.*", ""
    $raw = $raw -replace "(?m)\r?\nInclude[^\n]*httpd-gallery\.conf[^\n]*", ""
    [IO.File]::WriteAllText($HttpdConf, $raw)
    Ok "Include eltávolítva: $HttpdConf"
}

# ── win-acme tanúsítvány visszavonása ────────────────────────────────────
$WacsExe = Get-ChildItem "$env:ProgramData\win-acme" -Filter "wacs.exe" -Recurse -ErrorAction SilentlyContinue |
           Select-Object -First 1 -ExpandProperty FullName
if ($WacsExe) {
    $yn = Read-Host "Let's Encrypt tanusítvány visszavonása ($Domain)? (i/n)"
    if ($yn -match "^[Ii]") {
        & $WacsExe --cancel --host $Domain 2>$null
        Ok "Tanúsítvány visszavonva"
    }
}

# ── SSL-könyvtár törlése (opcionális) ────────────────────────────────────
$SslDir = "$XamppPath\apache\conf\ssl\$Domain"
if (Test-Path $SslDir) {
    $yn = Read-Host "SSL cert fájlok törlése ($SslDir)? (i/n)"
    if ($yn -match "^[Ii]") { Remove-Item $SslDir -Recurse -Force; Ok "SSL könyvtár törölve" }
}

# ── Apache újraindítása ───────────────────────────────────────────────────
$apacheSvc = Get-Service -DisplayName "*Apache*" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($apacheSvc) { Restart-Service $apacheSvc.Name }
elseif (Test-Path "$XamppPath\apache\bin\httpd.exe") {
    & "$XamppPath\apache\bin\httpd.exe" -k restart 2>$null
}
Ok "Apache újraindítva"

Write-Host ""
Ok "Kész — a galeria aldomén ($Domain) le van tiltva."
"""

# ── Cloudflare Tunnel (cloudflared) ──────────────────────────────────────────

_SETUP_TUNNEL_SH = r"""#!/usr/bin/env bash
# setup-tunnel.sh — Cloudflare Tunnel beállítása (Linux / macOS)
# Futtatás:  bash setup-tunnel.sh        (NEM kell sudo / root!)
#
# Nyilvános HTTPS a __DOMAIN__ domainre, nyitott portok NÉLKÜL — működik
# CGNAT / mobilnet mögött is. A HTTPS tanúsítványt a Cloudflare kezeli.
# Követelmény: a __DOMAIN__ domén Cloudflare-en legyen (a névszerverek a
#              Cloudflare-re mutassanak).
set -euo pipefail

DOMAIN="__DOMAIN__"
PORT="__PORT__"
TUNNEL_NAME="galeria-__DOMAIN__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS="$(uname -s)"

log() { echo -e "\033[1;34m[tunnel]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[  ok  ]\033[0m $*"; }
err() { echo -e "\033[1;31m[ hiba ]\033[0m $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] && err "NE root-ként futtasd: bash setup-tunnel.sh (sudo nélkül)."

# ── cloudflared telepítése ────────────────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
  if [[ "$OS" == "Darwin" ]]; then
    command -v brew &>/dev/null || err "Homebrew szükséges: https://brew.sh"
    log "cloudflared telepítése (Homebrew)..."
    brew install cloudflared
  elif command -v apt-get &>/dev/null; then
    log "cloudflared telepítése (apt)..."
    sudo mkdir -p --mode=0755 /usr/share/keyrings
    curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
      | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
    echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
      | sudo tee /etc/apt/sources.list.d/cloudflared.list
    sudo apt-get update && sudo apt-get install -y cloudflared
  else
    err "Telepítsd a cloudflared-et: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
  fi
  ok "cloudflared telepítve"
else
  ok "cloudflared már telepítve: $(cloudflared --version 2>&1 | head -1)"
fi

# ── Cloudflare bejelentkezés (böngészőt nyit) ──────────────────────────────
if [[ ! -f "$HOME/.cloudflared/cert.pem" ]]; then
  log "Cloudflare bejelentkezés — a böngészőben válaszd ki a domain zónát ($DOMAIN)..."
  cloudflared tunnel login
fi
ok "Cloudflare hitelesítés kész"

# ── Tunnel létrehozása (ha még nincs) ──────────────────────────────────────
if ! cloudflared tunnel list 2>/dev/null | grep -qw "$TUNNEL_NAME"; then
  log "Tunnel létrehozása: $TUNNEL_NAME"
  cloudflared tunnel create "$TUNNEL_NAME"
else
  ok "Tunnel már létezik: $TUNNEL_NAME"
fi

TUNNEL_ID="$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$2==n{print $1}')"
[[ -z "$TUNNEL_ID" ]] && err "Nem sikerült megállapítani a tunnel ID-t."
CRED_FILE="$HOME/.cloudflared/${TUNNEL_ID}.json"
ok "Tunnel ID: $TUNNEL_ID"

# ── config.yml írása ──────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/cloudflared-config.yml" <<EOF
# Cloudflare Tunnel konfiguráció — automatikusan generálva
tunnel: ${TUNNEL_ID}
credentials-file: ${CRED_FILE}
ingress:
  - hostname: ${DOMAIN}
    service: http://localhost:${PORT}
  - service: http_status:404
EOF
ok "config.yml → $SCRIPT_DIR/cloudflared-config.yml"

# ── DNS route a domainre ───────────────────────────────────────────────────
log "DNS rekord beállítása: ${DOMAIN} → tunnel"
cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN" || \
  log "(A DNS rekord már létezhet — ez rendben van.)"
ok "DNS kész"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Cloudflare Tunnel beállítva!"
echo "   URL   : https://${DOMAIN}"
echo "   Admin : https://${DOMAIN}/admin"
echo "   TLS   : Cloudflare kezeli (nincs nyitott port, CGNAT/mobilnet OK)"
echo ""
echo "   Indítás (Node + tunnel egyben, sudo NÉLKÜL):"
echo "     python3 start.py"
echo "   Eltávolítás: bash uninstall-tunnel.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_UNINSTALL_TUNNEL_SH = r"""#!/usr/bin/env bash
# uninstall-tunnel.sh — Cloudflare Tunnel eltávolítása (Linux / macOS)
# Futtatás: bash uninstall-tunnel.sh   (sudo nélkül)
set -euo pipefail

DOMAIN="__DOMAIN__"
TUNNEL_NAME="galeria-__DOMAIN__"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo -e "\033[1;34m[uninstall]\033[0m $*"; }
ok()  { echo -e "\033[1;32m[   ok    ]\033[0m $*"; }

command -v cloudflared &>/dev/null || { echo "cloudflared nincs telepítve — nincs mit eltávolítani."; exit 0; }

if cloudflared tunnel list 2>/dev/null | grep -qw "$TUNNEL_NAME"; then
  read -rp "Tunnel törlése ($TUNNEL_NAME)? A DNS rekordot is érdemes a Cloudflare felületen törölni. (i/n): " yn
  if [[ "$yn" =~ ^[Ii] ]]; then
    cloudflared tunnel cleanup "$TUNNEL_NAME" 2>/dev/null || true
    cloudflared tunnel delete "$TUNNEL_NAME" 2>/dev/null || true
    ok "Tunnel törölve: $TUNNEL_NAME"
  fi
fi

[[ -f "$SCRIPT_DIR/cloudflared-config.yml" ]] && rm "$SCRIPT_DIR/cloudflared-config.yml" && ok "config.yml törölve"

echo ""
ok "Kész. A ${DOMAIN} DNS (CNAME) rekordot a Cloudflare DNS felületén törölheted, ha kell."
"""

_SETUP_TUNNEL_PS1 = r"""# setup-tunnel.ps1 — Cloudflare Tunnel beállítása (Windows)
# Futtatás:  .\setup-tunnel.ps1     (NEM kell rendszergazda!)
param(
    [string]$Domain = "__DOMAIN__",
    [int]   $Port   = __PORT__
)
$ErrorActionPreference = "Stop"
$TunnelName = "galeria-$Domain"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

function Log { Write-Host "[tunnel] $args" -ForegroundColor Cyan  }
function Ok  { Write-Host "[  ok  ] $args" -ForegroundColor Green }
function Err { Write-Host "[ hiba ] $args" -ForegroundColor Red; exit 1 }

# ── cloudflared telepítése ────────────────────────────────────────────────
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "cloudflared telepítése (winget)..."
        winget install --id Cloudflare.cloudflared --accept-package-agreements --accept-source-agreements
    } else {
        Err "Telepítsd a cloudflared-et: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    }
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User")
    Ok "cloudflared telepítve"
} else {
    Ok "cloudflared már telepítve: $(cloudflared --version)"
}

# ── Cloudflare bejelentkezés ───────────────────────────────────────────────
if (-not (Test-Path "$env:USERPROFILE\.cloudflared\cert.pem")) {
    Log "Cloudflare bejelentkezés — a böngészőben válaszd ki a domain zónát ($Domain)..."
    cloudflared tunnel login
}
Ok "Cloudflare hitelesítés kész"

# ── Tunnel létrehozása ─────────────────────────────────────────────────────
$exists = (cloudflared tunnel list 2>$null | Select-String -Pattern "\b$TunnelName\b")
if (-not $exists) {
    Log "Tunnel létrehozása: $TunnelName"
    cloudflared tunnel create $TunnelName
} else {
    Ok "Tunnel már létezik: $TunnelName"
}

$line = (cloudflared tunnel list 2>$null | Select-String -Pattern "\b$TunnelName\b").Line
$TunnelId = ($line -split "\s+")[0]
if (-not $TunnelId) { Err "Nem sikerült megállapítani a tunnel ID-t." }
$CredFile = "$env:USERPROFILE\.cloudflared\$TunnelId.json"
Ok "Tunnel ID: $TunnelId"

# ── config.yml írása ──────────────────────────────────────────────────────
$cfg = @"
# Cloudflare Tunnel konfiguráció — automatikusan generálva
tunnel: $TunnelId
credentials-file: $($CredFile -replace '\\','/')
ingress:
  - hostname: $Domain
    service: http://localhost:$Port
  - service: http_status:404
"@
[IO.File]::WriteAllText("$ScriptDir\cloudflared-config.yml", $cfg)
Ok "config.yml -> $ScriptDir\cloudflared-config.yml"

# ── DNS route ──────────────────────────────────────────────────────────────
Log "DNS rekord beállítása: $Domain -> tunnel"
cloudflared tunnel route dns $TunnelName $Domain 2>$null
Ok "DNS kész"

Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Green
Ok "Cloudflare Tunnel beallitva!"
Write-Host "   URL   : https://$Domain"
Write-Host "   Admin : https://$Domain/admin"
Write-Host "   Inditas (Node + tunnel egyben): python start.py"
Write-Host "   Eltavolitas: .\uninstall-tunnel.ps1"
Write-Host ("=" * 60) -ForegroundColor Green
"""

_UNINSTALL_TUNNEL_PS1 = r"""# uninstall-tunnel.ps1 — Cloudflare Tunnel eltávolítása (Windows)
# Futtatás: .\uninstall-tunnel.ps1
param([string]$Domain = "__DOMAIN__")
$ErrorActionPreference = "Stop"
$TunnelName = "galeria-$Domain"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path

function Ok { Write-Host "[   ok    ] $args" -ForegroundColor Green }

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared nincs telepítve — nincs mit eltávolítani."; exit 0
}

$exists = (cloudflared tunnel list 2>$null | Select-String -Pattern "\b$TunnelName\b")
if ($exists) {
    $yn = Read-Host "Tunnel törlése ($TunnelName)? (i/n)"
    if ($yn -match "^[Ii]") {
        cloudflared tunnel cleanup $TunnelName 2>$null
        cloudflared tunnel delete  $TunnelName 2>$null
        Ok "Tunnel törölve: $TunnelName"
    }
}

if (Test-Path "$ScriptDir\cloudflared-config.yml") {
    Remove-Item "$ScriptDir\cloudflared-config.yml" -Force
    Ok "config.yml törölve"
}

Write-Host ""
Ok "Kész. A $Domain DNS (CNAME) rekordot a Cloudflare felületén törölheted, ha kell."
"""

_README_TUNNEL_SECTION = """
## HTTPS Cloudflare Tunnellel (domain: __DOMAIN__)

A **Cloudflare Tunnel** nyilvános HTTPS-t ad a domainedre **nyitott portok
nélkül** — működik CGNAT / mobilnet / otthoni router mögött is. A TLS
tanúsítványt a Cloudflare kezeli, nincs Let's Encrypt és nincs port-forward.

**Követelmény:** a `__DOMAIN__` domén a Cloudflare-en legyen (a domain
névszerverei a Cloudflare-re mutassanak — ingyenes Cloudflare-fiók elég).

### Gyors teszt (azonnali, bejelentkezés és domain nélkül)

```bash
python3 start.py --no-browser              # Node szerver külön terminálban
cloudflared tunnel --url http://localhost:__PORT__
```

Ez ad egy ideiglenes `https://valami.trycloudflare.com` címet. Belépéskor a
**6 jegyű kódot** használd (a magic-link e-mail a saját domainre mutat).

### Tartós beállítás a saját domainnel

**Linux / macOS** (NEM kell sudo):

```bash
bash setup-tunnel.sh        # cloudflared telepítése + login + tunnel + DNS
python3 start.py            # Node + tunnel egyben (Ctrl+C mindkettőt leállítja)
```

**Windows** (NEM kell rendszergazda):

```powershell
.\\setup-tunnel.ps1
python start.py
```

A `setup-tunnel.sh/.ps1` egyszer fut le: telepíti a `cloudflared`-et,
bejelentkeztet a Cloudflare-be (böngészőt nyit), létrehozza a tunnelt, beállítja
a `__DOMAIN__` DNS rekordját, és kiír egy `cloudflared-config.yml`-t. Ezután a
`start.py` a Node mellett automatikusan elindítja a tunnelt is.

### Eltávolítás

```bash
# Linux / macOS:
bash uninstall-tunnel.sh
# Windows:
.\\uninstall-tunnel.ps1
```

### Automatikus indítás háttérszolgáltatásként (opcionális)

A `cloudflared` rendszerszolgáltatásként is futhat (újraindítás után is feljön):

```bash
# Linux / macOS:
sudo cloudflared --config "$(pwd)/cloudflared-config.yml" service install
# Windows (admin PowerShell):
cloudflared --config "$PWD\\cloudflared-config.yml" service install
```

> A galéria Node szerverét ekkor is el kell indítani (`python start.py --no-browser`),
> a tunnel csak a publikus elérést adja.
"""

_README_HTTPS_SECTION = """
## HTTPS beállítása (domain: __DOMAIN__)

A Node.js szerver a **__PORT__**-es porton fut belül;
a 80/443-as portokat a reverse proxy (Caddy vagy Apache/XAMPP) kezeli.

### HTTPS szkriptek

| Fájl | Platform | Leírás |
|------|----------|--------|
| `setup-https.sh` | Linux / macOS | HTTPS telepítése és beállítása |
| `setup-https.ps1` | Windows | HTTPS telepítése és beállítása |
| `uninstall-https.sh` | Linux / macOS | Reverse proxy + cert eltávolítása |
| `uninstall-https.ps1` | Windows | Reverse proxy + cert eltávolítása |
| `Caddyfile` *(Caddy mód)* | minden | Caddy konfigurációs fájl |
| `apache-vhost.conf` *(Apache mód)* | minden | Apache/XAMPP VirtualHost konfig |

> **Megjegyzés:** A szkripteket elég **egyszer** futtatni a szerveren.
> A Let's Encrypt tanúsítvány megújítása automatikus.

### Beállítás — Linux / macOS

Két lehetőség:

**A) Minden egy paranccsal** (teszteléshez a legegyszerűbb) — a `start.py`
felhúzza a Node szervert **és** a Caddy-t is. A 80/443-as porthoz root kell:

```bash
sudo bash setup-https.sh         # CSAK ELŐSZÖR: telepíti a Caddy-t
sudo python3 start.py            # Node + Caddy együtt (Ctrl+C mindkettőt leállítja)
```

**B) Caddy háttérszolgáltatásként** (folyamatos üzemhez ajánlott) — a
`setup-https.sh` a Caddy-t rendszerszolgáltatásként indítja, ami újraindításkor
is feljön; a Node szervert külön indítod:

```bash
sudo bash setup-https.sh         # HTTPS beállítása (egyszer, a szerveren)
python start.py --no-browser     # Node.js szerver indítása
```

### Beállítás — Windows

Nyisd meg a PowerShellt **rendszergazdaként** (jobb klikk → Futtatás rendszergazdaként):

```powershell
python start.py --no-browser     # Node.js szerver indítása háttérben
.\\setup-https.ps1               # HTTPS beállítása (egyszer, a szerveren)
```

XAMPP nem az alapértelmezett `C:\\xampp` mappában van?

```powershell
.\\setup-https.ps1 -XamppPath D:\\xampp
```

### Eltávolítás

```bash
# Linux / macOS:
sudo bash uninstall-https.sh

# Windows (admin PowerShell):
.\\uninstall-https.ps1
.\\uninstall-https.ps1 -XamppPath D:\\xampp   # ha nem C:\\xampp
```

Az uninstall script leállítja a proxyt, törli a konfigot, és opcionálisan
visszavonja a Let's Encrypt tanúsítványt.

### Mit csinálnak a szkriptek?

**`setup-https.sh` / `setup-https.ps1`:**
- Telepíti a szükséges eszközt (Caddy mód: winget/brew/apt; Apache mód: win-acme/certbot)
- Beállítja a reverse proxy konfigot (Caddyfile vagy apache-vhost.conf)
- Let's Encrypt TLS tanúsítványt igényel a domainre
- Tanúsítvány **automatikusan megújul** — nincs további tennivaló

**`uninstall-https.sh` / `uninstall-https.ps1`:**
- Leállítja és eltávolítja a reverse proxy service-t (Caddy / Apache vhost)
- Törli a generált konfigurációs fájlokat
- Opcionálisan visszavonja a Let's Encrypt tanúsítványt is

### Eltávolítás

| Platform | Parancs |
|----------|---------|
| Linux / macOS | `sudo bash uninstall-https.sh` |
| Windows | `.\\ uninstall-https.ps1` (admin PowerShell) |

Az uninstall script leállítja a proxyt, törli a konfigot, és opcionálisan
visszavonja a Let's Encrypt tanúsítványt is.

XAMPP más mappában (Windows): `.\\uninstall-https.ps1 -XamppPath D:\\xampp`

### Automatikus indítás (Node.js szerver)

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

**macOS (launchd):**

```bash
sudo tee /Library/LaunchDaemons/hu.galeria.server.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>hu.galeria.server</string>
  <key>ProgramArguments</key><array>
    <string>/usr/local/bin/node</string>
    <string>__INSTALL_DIR__/server/index.js</string>
  </array>
  <key>WorkingDirectory</key><string>__INSTALL_DIR__</string>
  <key>EnvironmentVariables</key><dict>
    <key>PORT</key><string>__PORT__</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
</dict></plist>
EOF
sudo launchctl load /Library/LaunchDaemons/hu.galeria.server.plist
```

**Windows (Feladatütemező):**

```powershell
$trigger = New-ScheduledTaskTrigger -AtStartup
$action  = New-ScheduledTaskAction -Execute "node" `
           -Argument "__INSTALL_DIR__\\server\\index.js" `
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

# A winget frissíti a PATH-t, de a futó munkamenet még a régit látja → frissítjük,
# majd ha az alias még így sem elérhető, megkeressük a winget shim/csomag helyén.
function Resolve-Caddy {
    $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path","User")
    $cmd = Get-Command caddy -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $shim = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\caddy.exe"
    if (Test-Path $shim) { return $shim }
    $pkg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter caddy.exe `
           -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pkg) { return $pkg.FullName }
    return $null
}

if (-not (Get-Command caddy -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Log "Caddy telepítése (winget)..."
        winget install --id Caddyserver.Caddy --accept-package-agreements --accept-source-agreements
        $CaddyExe = Resolve-Caddy
        if (-not $CaddyExe) { Err "Caddy telepítve, de nem található. Indítsd újra a PowerShell-t (rendszergazdaként) és futtasd újra a szkriptet." }
        Ok "Caddy telepítve (winget): $CaddyExe"
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

# ── Caddy indítása ütemezett feladatként ──────────────────────────────────
# A Caddy-nak nincs natív Windows "service" parancsa, ezért egy Scheduled Task
# futtatja SYSTEM-ként, induláskor (így a 80/443 portokat is tudja kötni, és
# újraindítás után magától feljön). A 'caddy run' a feladaton belül fut.
$TaskName = "CaddyGallery"
$CaddyfilePath = "$CaddyDir\Caddyfile"

# Korábbi (hibás) próbálkozások takarítása: régi "caddy" Windows service törlése
$oldSvc = Get-Service -Name "caddy" -ErrorAction SilentlyContinue
if ($oldSvc) { Stop-Service caddy -Force -ErrorAction SilentlyContinue; & sc.exe delete caddy | Out-Null }
# Ha már fut egy Caddy a háttérben (pl. 'caddy start'), állítsuk le, hogy ne ütközzön
& $CaddyExe stop 2>$null | Out-Null

$action    = New-ScheduledTaskAction -Execute $CaddyExe `
             -Argument "run --config `"$CaddyfilePath`"" -WorkingDirectory $CaddyDir
$trigger   = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
             -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Log "Caddy ütemezett feladat regisztrálása ($TaskName)..."
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep 2
Ok "Caddy fut (ütemezett feladat, induláskor automatikusan)"

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


def _is_root() -> bool:
    """True if the process can bind privileged ports (80/443)."""
    if os.name == "nt":
        try:
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            return False
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _start_caddy(here: Path, cfg: dict):
    """Launch Caddy as a child process when HTTPS is enabled.

    Returns the Popen, or None if Caddy should not / cannot be started here
    (HTTP-only export, Caddy missing, already running as a service, or no
    root for the 80/443 ports). In every skip case it prints why.
    """
    if not cfg.get("httpsEnabled"):
        return None
    caddyfile = here / "Caddyfile"
    if not caddyfile.exists():
        return None  # not a Caddy export (e.g. Apache mode)

    domain = cfg.get("domain", "")
    caddy = _find_exe(["caddy", "caddy.exe"])
    if not caddy:
        print("[https] Caddy nincs telepítve — egyelőre csak HTTP fut.")
        print("        Telepítsd és állítsd be egyszer:  sudo bash setup-https.sh")
        return None

    # Caddy already serving (e.g. installed as a brew service / LaunchDaemon)?
    if _port_in_use(443):
        print(f"[https] A 443-as port már foglalt — a Caddy valószínűleg már fut "
              f"(rendszerszolgáltatásként). https://{domain}")
        return None

    if not _is_root():
        hint = ("Indítsd rendszergazdaként:  Jobb klikk a PowerShell-en → "
                "Futtatás rendszergazdaként" if os.name == "nt"
                else "Indítsd így:  sudo python3 start.py")
        print("[https] A 80/443-as porthoz root/rendszergazda jogosultság kell.")
        print(f"        {hint}")
        print("        (Vagy telepítsd háttérszolgáltatásként: sudo bash setup-https.sh)")
        return None

    print(f"[https] Caddy indítása (reverse proxy + Let's Encrypt): https://{domain}")
    try:
        return subprocess.Popen(
            [caddy, "run", "--config", str(caddyfile), "--adapter", "caddyfile"],
            cwd=str(here),
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"[https] Caddy indítása sikertelen: {e}")
        return None


def _start_cloudflared(here: Path, cfg: dict):
    """Launch the Cloudflare Tunnel (cloudflared) when httpsMode == 'cloudflare'.

    Needs NO root and no open ports — works behind CGNAT / mobile internet.
    Returns the Popen, or None (with an explanation) if it should not start.
    """
    if cfg.get("httpsMode") != "cloudflare":
        return None
    domain = cfg.get("domain", "")
    cfg_yml = here / "cloudflared-config.yml"
    if not cfg_yml.exists():
        print("[tunnel] A cloudflared-config.yml hiányzik — előbb futtasd egyszer:")
        print("         bash setup-tunnel.sh   (Windows: .\\setup-tunnel.ps1)")
        return None
    cloudflared = _find_exe(["cloudflared", "cloudflared.exe"])
    if not cloudflared:
        print("[tunnel] cloudflared nincs telepítve — futtasd:  bash setup-tunnel.sh")
        return None

    print(f"[tunnel] Cloudflare Tunnel indítása: https://{domain}")
    try:
        return subprocess.Popen(
            [cloudflared, "tunnel", "--config", str(cfg_yml), "run"],
            cwd=str(here),
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"[tunnel] cloudflared indítása sikertelen: {e}")
        return None


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

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        cfg = {}

    port = args.port
    if port is None:
        try:
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

    # When HTTPS is enabled, bring up the Caddy reverse proxy alongside Node so
    # a single command starts everything. Skips itself (with an explanation) on
    # HTTP-only exports, missing Caddy, an already-running Caddy service, or
    # without root for the 80/443 ports.
    caddy  = _start_caddy(HERE, cfg)
    tunnel = _start_cloudflared(HERE, cfg)

    if not args.no_browser:
        if _wait_for_server(port):
            webbrowser.open(url)
        else:
            print(f"[!] Szerver nem válaszolt {15}s alatt — nyisd meg: {url}")

    def _stop(p) -> None:
        if not p:
            return
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nLeállítás…")
    finally:
        _stop(proc)
        _stop(caddy)
        _stop(tunnel)
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

## Generált fájlok

| Fájl | Leírás |
|------|--------|
| `start.py` | Fő indítószkript (Python 3.8+, cross-platform) |
| `start.sh` / `start.command` | Linux / macOS indító (dupla kattintás) |
| `start.bat` | Windows indító (dupla kattintás) |
| `server/` | Node.js / Express auth szerver (OTP belépés, session) |
| `dist/` | Astro statikus galéria oldal |
| `config.json` | Szerver konfiguráció (email, jelszó, portszám) |
| `data/` | Futásidejű adatok (email-lista módosítások) |

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
