"""Docker + Kubernetes export service.

Builds the Astro static site, then wraps it in a Node/Express auth layer
(OTP via Gmail, session-based access, admin panel) and emits a ready-to-deploy
directory containing:

  dist/          — Astro static output (generated fresh)
  server/        — Express auth server
  config.json    — baked admin email, Gmail app-password, seed allowlist
  Dockerfile
  docker-compose.yml
  k8s/           — Kubernetes manifests (nginx-ingress + cert-manager)
  deploy.sh      — single-script deploy for Linux + Kubernetes
  README.md
"""
from __future__ import annotations

import base64
import csv
import json
import os
import shutil
import textwrap
from pathlib import Path
from typing import Callable, List, Optional

_ProgressCb = Callable[[Optional[int], str], None]


class DockerExportService:
    """Generate a self-contained Docker/K8s export bundle."""

    def __init__(self, session) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        target_dir: str,
        admin_email: str,
        gmail_app_password: str,
        allowed_emails_csv: str,
        domain: str,
        cluster_issuer: str = "letsencrypt-prod",
        port: int = 3000,
        progress_callback: Optional[_ProgressCb] = None,
    ) -> Path:
        """Export to *target_dir* and return the path.

        Args:
            target_dir:          Destination directory (created if absent).
            admin_email:         Gmail address used to send OTPs.
            gmail_app_password:  Gmail App Password (not the account password).
            allowed_emails_csv:  Path to a CSV file whose first column is email.
            domain:              Public hostname for the Ingress (e.g. gallery.example.com).
            cluster_issuer:      cert-manager ClusterIssuer name.
            port:                Container HTTP port.
            progress_callback:   Optional (pct, message) callback.
        """
        out = Path(target_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _p(pct: Optional[int], msg: str) -> None:
            if progress_callback:
                progress_callback(pct, msg)

        # 1. Build Astro site into out/dist
        _p(0, "Astro site generálása…")
        self._build_astro(out, progress_callback)

        # 2. Parse allowed emails
        _p(70, "Email-lista olvasása…")
        allowed = self._parse_csv(allowed_emails_csv)

        # 3. Write config.json (baked secrets)
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

        # 4. Server files
        _p(75, "Szerver fájlok írása…")
        self._write_server(out, port)

        # 5. Docker files
        _p(85, "Dockerfile + Compose írása…")
        self._write_docker(out, port)

        # 6. Kubernetes manifests
        _p(88, "Kubernetes manifesztek írása…")
        self._write_k8s(out, domain, cluster_issuer, port)

        # 7. deploy.sh
        _p(93, "Deploy szkript írása…")
        self._write_deploy_sh(out, domain, cluster_issuer)

        # 8. README
        _p(97, "README írása…")
        self._write_readme(out, domain, port)

        _p(100, "Kész.")
        return out

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def _build_astro(self, out: Path, progress_callback: Optional[_ProgressCb]) -> None:
        from app.services.export_service import ExportService

        def _fwd(pct, msg):
            if progress_callback:
                # Scale 0-65 to leave room for the rest
                progress_callback(int((pct or 0) * 0.65), msg)

        dist_target = out / "dist"
        dist_target.mkdir(parents=True, exist_ok=True)
        ExportService(self._session).export_astro(
            target_dir=str(dist_target),
            # No standalone static-server launchers in dist/: this bundle is
            # served behind the OTP auth layer, and a no-auth run.sh/run.ps1
            # would let anyone bypass the login.
            write_run_scripts=False,
            server_auth=True,
            progress_callback=_fwd,
        )

    @staticmethod
    def _parse_csv(csv_path: str) -> List[str]:
        emails: List[str] = []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    email = row[0].strip().lower()
                    if "@" in email:
                        emails.append(email)
        return emails

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _write_server(self, out: Path, port: int) -> None:
        srv = out / "server"
        srv.mkdir(exist_ok=True)
        views = srv / "views"
        views.mkdir(exist_ok=True)

        (srv / "package.json").write_text(_SERVER_PACKAGE_JSON)
        (srv / "index.js").write_text(_SERVER_INDEX_JS.replace("__PORT__", str(port)))
        (srv / "store.js").write_text(_SERVER_STORE_JS)
        (srv / "auth.js").write_text(_SERVER_AUTH_JS)
        (srv / "mailer.js").write_text(_SERVER_MAILER_JS)
        (views / "login.html").write_text(_LOGIN_HTML)
        (views / "admin.html").write_text(_ADMIN_HTML)

    def _write_docker(self, out: Path, port: int) -> None:
        (out / "Dockerfile").write_text(_DOCKERFILE.replace("__PORT__", str(port)))
        (out / "docker-compose.yml").write_text(
            _DOCKER_COMPOSE.replace("__PORT__", str(port))
        )

    def _write_k8s(
        self, out: Path, domain: str, cluster_issuer: str, port: int
    ) -> None:
        k8s = out / "k8s"
        k8s.mkdir(exist_ok=True)

        # Secret: base64-encode config.json
        config_b64 = base64.b64encode((out / "config.json").read_bytes()).decode()

        files = {
            "namespace.yaml": _K8S_NAMESPACE,
            "secret.yaml": _K8S_SECRET.replace("__CONFIG_B64__", config_b64),
            "pvc.yaml": _K8S_PVC,
            "deployment.yaml": _K8S_DEPLOYMENT.replace("__PORT__", str(port)),
            "service.yaml": _K8S_SERVICE.replace("__PORT__", str(port)),
            "ingress.yaml": _K8S_INGRESS
            .replace("__DOMAIN__", domain)
            .replace("__CLUSTER_ISSUER__", cluster_issuer),
        }
        for name, content in files.items():
            (k8s / name).write_text(content)

    def _write_deploy_sh(self, out: Path, domain: str, cluster_issuer: str) -> None:
        script = _DEPLOY_SH.replace("__DOMAIN__", domain).replace(
            "__CLUSTER_ISSUER__", cluster_issuer
        )
        deploy = out / "deploy.sh"
        deploy.write_text(script)
        deploy.chmod(0o755)

    def _write_readme(self, out: Path, domain: str, port: int) -> None:
        (out / "README.md").write_text(
            _README_MD.replace("__DOMAIN__", domain).replace("__PORT__", str(port))
        )


# ===========================================================================
# Embedded templates
# ===========================================================================

_SERVER_PACKAGE_JSON = """{
  "name": "gallery-auth-server",
  "version": "1.0.0",
  "type": "commonjs",
  "scripts": { "start": "node index.js" },
  "dependencies": {
    "express": "^4.18.2",
    "express-session": "^1.17.3",
    "nodemailer": "^6.9.9",
    "cookie-parser": "^1.4.6"
  }
}
"""

_SERVER_INDEX_JS = r"""'use strict';
const express = require('express');
const session = require('express-session');
const path = require('path');
const fs = require('fs');
const store = require('./store');
const auth = require('./auth');
const mailer = require('./mailer');

const PORT = process.env.PORT || __PORT__;
const DIST = path.join(__dirname, '..', 'dist');
const VIEWS = path.join(__dirname, 'views');

const app = express();
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

const SESSION_MS = store.config.sessionTimeoutMinutes * 60 * 1000;
app.use(session({
  secret: store.config.gmailAppPassword + store.config.adminEmail,
  resave: false,
  saveUninitialized: false,
  rolling: true,          // refresh on every request
  cookie: { maxAge: SESSION_MS, httpOnly: true, sameSite: 'lax' },
}));

// ---- Auth guard ----
function requireAuth(req, res, next) {
  if (req.session && req.session.email) return next();
  if (req.accepts('html')) return res.redirect('/login');
  res.status(401).json({ error: 'unauthorized' });
}
function requireAdmin(req, res, next) {
  if (req.session && req.session.isAdmin) return next();
  res.status(403).json({ error: 'forbidden' });
}

// ---- Public routes ----
app.get('/login', (_req, res) => res.sendFile(path.join(VIEWS, 'login.html')));

app.post('/api/request-code', async (req, res) => {
  const email = (req.body.email || '').trim().toLowerCase();
  if (!email || !email.includes('@')) return res.status(400).json({ error: 'invalid email' });

  if (store.isAllowed(email)) {
    const code = auth.generateOtp();
    store.setOtp(email, code);
    try {
      await mailer.sendOtp(email, code);
    } catch (e) {
      console.error('OTP send failed:', e.message);
    }
  } else {
    // Notify admin silently
    mailer.sendAdminAlert(email).catch(() => {});
  }
  // Always same response to avoid leaking allowlist
  res.json({ ok: true });
});

app.post('/api/verify-code', (req, res) => {
  const email = (req.body.email || '').trim().toLowerCase();
  const code  = (req.body.code  || '').trim();
  if (!store.verifyOtp(email, code)) return res.status(401).json({ error: 'invalid or expired code' });
  req.session.email   = email;
  req.session.isAdmin = store.isAdmin(email);
  res.json({ ok: true, isAdmin: req.session.isAdmin });
});

app.post('/api/logout', (req, res) => {
  req.session.destroy(() => res.json({ ok: true }));
});

// ---- Admin routes ----
app.get('/admin', requireAuth, requireAdmin, (_req, res) =>
  res.sendFile(path.join(VIEWS, 'admin.html')));

app.get('/api/admin/emails', requireAuth, requireAdmin, (_req, res) =>
  res.json({ emails: store.listEmails() }));

app.post('/api/admin/emails', requireAuth, requireAdmin, (req, res) => {
  const email = (req.body.email || '').trim().toLowerCase();
  if (!email || !email.includes('@')) return res.status(400).json({ error: 'invalid' });
  store.addEmail(email);
  res.json({ ok: true });
});

app.delete('/api/admin/emails/:email', requireAuth, requireAdmin, (req, res) => {
  store.removeEmail(decodeURIComponent(req.params.email).toLowerCase());
  res.json({ ok: true });
});

// ---- Protected static site ----
app.use(requireAuth, express.static(DIST));

// Catch-all: serve index of dist (Astro uses directory-based routing)
app.get('*', requireAuth, (req, res) => {
  const candidate = path.join(DIST, req.path, 'index.html');
  if (fs.existsSync(candidate)) return res.sendFile(candidate);
  res.status(404).send('Not found');
});

app.listen(PORT, () => console.log(`Gallery server listening on :${PORT}`));
"""

_SERVER_STORE_JS = r"""'use strict';
const fs = require('fs');
const path = require('path');

const CONFIG_PATH = path.join(__dirname, '..', 'config.json');
const DATA_PATH   = path.join(__dirname, '..', 'data', 'allowlist.json');

const config = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8'));

// Runtime allowlist: seed from config, then overlay persistent additions
const _allowset = new Set(config.allowedEmails.map(e => e.toLowerCase()));

function _loadDisk() {
  try {
    if (fs.existsSync(DATA_PATH)) {
      const extra = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));
      (extra.added   || []).forEach(e => _allowset.add(e.toLowerCase()));
      (extra.removed || []).forEach(e => _allowset.delete(e.toLowerCase()));
    }
  } catch { /* corrupt file — ignore */ }
}

function _saveDisk() {
  const dir = path.dirname(DATA_PATH);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  // Persist full current list (minus the original seed) as two delta arrays
  const original = new Set(config.allowedEmails.map(e => e.toLowerCase()));
  const added   = [..._allowset].filter(e => !original.has(e));
  const removed = [...original].filter(e => !_allowset.has(e));
  fs.writeFileSync(DATA_PATH, JSON.stringify({ added, removed }, null, 2));
}

_loadDisk();

// In-memory OTP store: email -> { code, expiresAt }
const _otps = new Map();

module.exports = {
  config,
  isAllowed:  (email) => _allowset.has(email.toLowerCase()) || email.toLowerCase() === config.adminEmail.toLowerCase(),
  isAdmin:    (email) => email.toLowerCase() === config.adminEmail.toLowerCase(),
  listEmails: ()      => [..._allowset].sort(),
  addEmail(email) { _allowset.add(email.toLowerCase()); _saveDisk(); },
  removeEmail(email) { _allowset.delete(email.toLowerCase()); _saveDisk(); },

  setOtp(email, code) {
    const exp = Date.now() + config.otpExpiryMinutes * 60 * 1000;
    _otps.set(email.toLowerCase(), { code, expiresAt: exp });
  },
  verifyOtp(email, code) {
    const entry = _otps.get(email.toLowerCase());
    if (!entry) return false;
    if (Date.now() > entry.expiresAt) { _otps.delete(email.toLowerCase()); return false; }
    if (entry.code !== code) return false;
    _otps.delete(email.toLowerCase());
    return true;
  },
};
"""

_SERVER_AUTH_JS = r"""'use strict';
function generateOtp() {
  // 6-digit zero-padded
  return String(Math.floor(Math.random() * 1_000_000)).padStart(6, '0');
}
module.exports = { generateOtp };
"""

_SERVER_MAILER_JS = r"""'use strict';
const nodemailer = require('nodemailer');
const store = require('./store');

function _transporter() {
  return nodemailer.createTransport({
    service: 'gmail',
    auth: { user: store.config.adminEmail, pass: store.config.gmailAppPassword },
  });
}

async function sendOtp(to, code) {
  await _transporter().sendMail({
    from: `"Galéria" <${store.config.adminEmail}>`,
    to,
    subject: 'Belépési kód',
    text: `A belépési kódod: ${code}\n\nÉrvényes ${store.config.otpExpiryMinutes} percig.`,
    html: `<p>A belépési kódod: <strong style="font-size:1.5em">${code}</strong></p>
           <p>Érvényes ${store.config.otpExpiryMinutes} percig.</p>`,
  });
}

async function sendAdminAlert(attemptedEmail) {
  await _transporter().sendMail({
    from: `"Galéria" <${store.config.adminEmail}>`,
    to: store.config.adminEmail,
    subject: 'Ismeretlen belépési kísérlet',
    text: `Nem engedélyezett email cím próbált belépni: ${attemptedEmail}`,
    html: `<p>Nem engedélyezett email cím próbált belépni:</p>
           <p><strong>${attemptedEmail}</strong></p>`,
  });
}

module.exports = { sendOtp, sendAdminAlert };
"""

_LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Belépés — Galéria</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: flex; align-items: center;
         justify-content: center; background: #111; color: #eee;
         font-family: system-ui, sans-serif; }
  .card { background: #1e1e1e; border-radius: 12px; padding: 2.5rem 2rem;
          width: min(100%, 380px); box-shadow: 0 8px 32px #0008; }
  h1 { margin: 0 0 1.5rem; font-size: 1.4rem; text-align: center; }
  label { display: block; font-size: .85rem; color: #aaa; margin-bottom: .3rem; }
  input { width: 100%; padding: .65rem .9rem; border: 1px solid #444;
          border-radius: 8px; background: #2a2a2a; color: #eee;
          font-size: 1rem; outline: none; transition: border .2s; }
  input:focus { border-color: #6c9fff; }
  button { width: 100%; margin-top: 1rem; padding: .75rem;
           border: none; border-radius: 8px; background: #3a7bfd;
           color: #fff; font-size: 1rem; cursor: pointer; transition: opacity .2s; }
  button:hover { opacity: .88; }
  .msg { margin-top: 1rem; padding: .65rem; border-radius: 8px;
         font-size: .9rem; text-align: center; display: none; }
  .msg.ok  { background: #1a3a1a; color: #6fcf6f; }
  .msg.err { background: #3a1a1a; color: #cf6f6f; }
  .field { margin-bottom: 1rem; }
  #step2 { display: none; }
</style>
</head>
<body>
<div class="card">
  <h1>📷 Galéria belépés</h1>
  <div id="step1">
    <div class="field">
      <label for="email">Email cím</label>
      <input id="email" type="email" placeholder="te@gmail.com" autocomplete="email">
    </div>
    <button id="reqBtn">Kód kérése</button>
  </div>
  <div id="step2">
    <p style="color:#aaa;font-size:.9rem;margin:.5rem 0 1rem">
      Ha az email cím szerepel a listában, küldtünk egy 6 jegyű kódot.
    </p>
    <div class="field">
      <label for="code">Belépési kód</label>
      <input id="code" type="text" inputmode="numeric" pattern="\d{6}"
             maxlength="6" placeholder="123456" autocomplete="one-time-code">
    </div>
    <button id="verBtn">Belépés</button>
    <button id="backBtn" style="background:#333;margin-top:.5rem">← Vissza</button>
  </div>
  <div id="msg" class="msg"></div>
</div>
<script>
const email$ = document.getElementById('email');
const code$  = document.getElementById('code');
const msg$   = document.getElementById('msg');
const step1  = document.getElementById('step1');
const step2  = document.getElementById('step2');

function showMsg(text, type) {
  msg$.textContent = text;
  msg$.className = 'msg ' + type;
  msg$.style.display = 'block';
}

document.getElementById('reqBtn').onclick = async () => {
  const email = email$.value.trim();
  if (!email) return showMsg('Add meg az email címed.', 'err');
  const r = await fetch('/api/request-code', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email }),
  });
  if (r.ok) { step1.style.display='none'; step2.style.display='block'; msg$.style.display='none'; }
  else showMsg('Hiba történt, próbáld újra.', 'err');
};

document.getElementById('backBtn').onclick = () => {
  step2.style.display='none'; step1.style.display='block'; msg$.style.display='none';
};

document.getElementById('verBtn').onclick = async () => {
  const r = await fetch('/api/verify-code', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email: email$.value.trim(), code: code$.value.trim() }),
  });
  if (r.ok) {
    const data = await r.json();
    showMsg('Sikeres belépés! Átirányítás…', 'ok');
    setTimeout(() => { window.location.href = data.isAdmin ? '/admin' : '/'; }, 800);
  } else {
    showMsg('Érvénytelen vagy lejárt kód.', 'err');
  }
};

email$.addEventListener('keydown', e => e.key==='Enter' && document.getElementById('reqBtn').click());
code$.addEventListener('keydown',  e => e.key==='Enter' && document.getElementById('verBtn').click());
</script>
</body>
</html>
"""

_ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Admin — Galéria</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; background: #111; color: #eee;
         font-family: system-ui, sans-serif; padding: 2rem 1rem; }
  h1 { font-size: 1.4rem; margin-bottom: 1.5rem; }
  .card { background: #1e1e1e; border-radius: 12px; padding: 1.5rem;
          max-width: 560px; margin: 0 auto; box-shadow: 0 4px 20px #0006; }
  .row { display: flex; gap: .5rem; margin-bottom: 1rem; }
  input { flex: 1; padding: .6rem .9rem; border: 1px solid #444; border-radius: 8px;
          background: #2a2a2a; color: #eee; font-size: .95rem; }
  button.add { padding: .6rem 1.2rem; border: none; border-radius: 8px;
               background: #3a7bfd; color: #fff; cursor: pointer; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { display: flex; justify-content: space-between; align-items: center;
       padding: .55rem .5rem; border-bottom: 1px solid #2a2a2a; font-size: .9rem; }
  li:last-child { border: none; }
  .del { background: none; border: 1px solid #555; border-radius: 6px;
         color: #cf6f6f; cursor: pointer; padding: .25rem .6rem; font-size: .8rem; }
  .del:hover { background: #3a1a1a; }
  .gallery-link { display:block; text-align:center; margin-top:1.5rem;
                  color: #6c9fff; text-decoration: none; }
  .msg { margin-top:.75rem; padding:.5rem; border-radius:8px; font-size:.85rem;
         text-align:center; display:none; }
  .msg.ok  { background:#1a3a1a; color:#6fcf6f; display:block; }
  .msg.err { background:#3a1a1a; color:#cf6f6f; display:block; }
</style>
</head>
<body>
<div class="card">
  <h1>⚙️ Admin — engedélyezett email címek</h1>
  <div class="row">
    <input id="newEmail" type="email" placeholder="uj@gmail.com">
    <button class="add" id="addBtn">Hozzáadás</button>
  </div>
  <div id="msg" class="msg"></div>
  <ul id="list"></ul>
  <a class="gallery-link" href="/">← Vissza a galériához</a>
</div>
<script>
async function load() {
  const r = await fetch('/api/admin/emails');
  const { emails } = await r.json();
  const ul = document.getElementById('list');
  ul.innerHTML = emails.map(e =>
    `<li><span>${e}</span><button class="del" data-email="${e}">Törlés</button></li>`
  ).join('');
  ul.querySelectorAll('.del').forEach(b => b.onclick = () => remove(b.dataset.email));
}

async function remove(email) {
  await fetch('/api/admin/emails/' + encodeURIComponent(email), { method: 'DELETE' });
  load();
}

document.getElementById('addBtn').onclick = async () => {
  const email = document.getElementById('newEmail').value.trim().toLowerCase();
  const msg = document.getElementById('msg');
  if (!email || !email.includes('@')) {
    msg.className='msg err'; msg.textContent='Érvénytelen email cím.'; return;
  }
  const r = await fetch('/api/admin/emails', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ email }),
  });
  if (r.ok) {
    msg.className='msg ok'; msg.textContent=`${email} hozzáadva.`;
    document.getElementById('newEmail').value = '';
    load();
  } else {
    msg.className='msg err'; msg.textContent='Hiba történt.';
  }
};

document.getElementById('newEmail').addEventListener('keydown',
  e => e.key==='Enter' && document.getElementById('addBtn').click());

load();
</script>
</body>
</html>
"""

_DOCKERFILE = r"""# syntax=docker/dockerfile:1
FROM node:20-alpine AS deps
WORKDIR /app
COPY server/package.json server/package-lock.json* ./
RUN npm ci --omit=dev

FROM node:20-alpine
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY server/ ./server/
COPY dist/   ./dist/
COPY config.json ./
RUN mkdir -p /app/data

EXPOSE __PORT__
ENV PORT=__PORT__
CMD ["node", "server/index.js"]
"""

_DOCKER_COMPOSE = r"""version: '3.9'
services:
  gallery:
    build: .
    ports:
      - "__PORT__:__PORT__"
    volumes:
      - gallery_data:/app/data
    restart: unless-stopped
volumes:
  gallery_data:
"""

_K8S_NAMESPACE = """apiVersion: v1
kind: Namespace
metadata:
  name: gallery
"""

_K8S_SECRET = """apiVersion: v1
kind: Secret
metadata:
  name: gallery-config
  namespace: gallery
type: Opaque
data:
  config.json: __CONFIG_B64__
"""

_K8S_PVC = """apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: gallery-data
  namespace: gallery
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 100Mi
"""

_K8S_DEPLOYMENT = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: gallery
  namespace: gallery
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gallery
  template:
    metadata:
      labels:
        app: gallery
    spec:
      containers:
        - name: gallery
          image: localhost:5000/gallery:latest
          imagePullPolicy: Always
          ports:
            - containerPort: __PORT__
          env:
            - name: PORT
              value: "__PORT__"
          volumeMounts:
            - name: config
              mountPath: /app/config.json
              subPath: config.json
              readOnly: true
            - name: data
              mountPath: /app/data
          resources:
            requests:
              cpu: 50m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 512Mi
      volumes:
        - name: config
          secret:
            secretName: gallery-config
        - name: data
          persistentVolumeClaim:
            claimName: gallery-data
"""

_K8S_SERVICE = """apiVersion: v1
kind: Service
metadata:
  name: gallery
  namespace: gallery
spec:
  selector:
    app: gallery
  ports:
    - port: 80
      targetPort: __PORT__
"""

_K8S_INGRESS = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gallery
  namespace: gallery
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: __CLUSTER_ISSUER__
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  tls:
    - hosts:
        - __DOMAIN__
      secretName: gallery-tls
  rules:
    - host: __DOMAIN__
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: gallery
                port:
                  number: 80
"""

_DEPLOY_SH = r"""#!/usr/bin/env bash
# deploy.sh — build & deploy gallery to Kubernetes
# Usage: bash deploy.sh [--domain <domain>] [--issuer <cluster-issuer>]
# Requirements on the server: docker, kubectl (configured), optional: k3s
set -euo pipefail

DOMAIN="__DOMAIN__"
CLUSTER_ISSUER="__CLUSTER_ISSUER__"
IMAGE="localhost:5000/gallery:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── helpers ────────────────────────────────────────────────────────────────
log()  { echo -e "\033[1;34m[deploy]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[  ok  ]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ err  ]\033[0m $*" >&2; exit 1; }

# ── parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --domain) DOMAIN="$2"; shift 2 ;;
    --issuer) CLUSTER_ISSUER="$2"; shift 2 ;;
    *) err "Unknown argument: $1" ;;
  esac
done

# ── prereqs ────────────────────────────────────────────────────────────────
for cmd in docker kubectl; do
  command -v "$cmd" &>/dev/null || err "'$cmd' not found in PATH"
done

# ── local registry ─────────────────────────────────────────────────────────
# We run a tiny registry on localhost:5000 so Kubernetes (containerd) can pull
# the image without a remote registry. The container is named 'gallery-registry'.
log "Checking local registry on :5000 …"
if ! docker ps --format '{{.Names}}' | grep -q '^gallery-registry$'; then
  if docker ps -a --format '{{.Names}}' | grep -q '^gallery-registry$'; then
    log "Starting existing registry container …"
    docker start gallery-registry
  else
    log "Creating local registry container …"
    docker run -d --name gallery-registry --restart=unless-stopped \
      -p 127.0.0.1:5000:5000 registry:2
  fi
  sleep 2
fi
ok "Registry ready at localhost:5000"

# For k3s: register the registry so containerd trusts localhost:5000
if command -v k3s &>/dev/null; then
  REG_CFG="/etc/rancher/k3s/registries.yaml"
  if ! grep -q "localhost:5000" "$REG_CFG" 2>/dev/null; then
    log "Registering insecure registry with k3s …"
    sudo mkdir -p "$(dirname "$REG_CFG")"
    sudo tee -a "$REG_CFG" > /dev/null <<'YAML'
mirrors:
  "localhost:5000":
    endpoint:
      - "http://localhost:5000"
YAML
    log "Restarting k3s to apply registry config …"
    sudo systemctl restart k3s
    sleep 10
    ok "k3s restarted"
  fi
fi

# For standard k8s (kubeadm/etc): patch containerd config
if ! command -v k3s &>/dev/null && command -v containerd &>/dev/null; then
  CTR_CFG="/etc/containerd/config.toml"
  if ! grep -q "localhost:5000" "$CTR_CFG" 2>/dev/null; then
    log "Patching containerd to trust localhost:5000 …"
    sudo mkdir -p /etc/containerd/certs.d/localhost:5000
    echo -e '[host."http://localhost:5000"]\n  capabilities = ["pull","resolve","push"]' \
      | sudo tee /etc/containerd/certs.d/localhost:5000/hosts.toml > /dev/null
    sudo systemctl restart containerd
    sleep 5
    ok "containerd restarted"
  fi
fi

# ── build & push ───────────────────────────────────────────────────────────
log "Building Docker image …"
docker build -t "$IMAGE" "$SCRIPT_DIR"
log "Pushing to local registry …"
docker push "$IMAGE"
ok "Image pushed: $IMAGE"

# ── update k8s manifests with actual domain / issuer ──────────────────────
K8S_DIR="$SCRIPT_DIR/k8s"
# Patch ingress inline (sed on the generated file)
sed -i "s|__DOMAIN__|${DOMAIN}|g;s|__CLUSTER_ISSUER__|${CLUSTER_ISSUER}|g" \
  "$K8S_DIR/ingress.yaml" 2>/dev/null || true

# ── apply manifests ────────────────────────────────────────────────────────
log "Applying Kubernetes manifests …"
kubectl apply -f "$K8S_DIR/namespace.yaml"
kubectl apply -f "$K8S_DIR/secret.yaml"
kubectl apply -f "$K8S_DIR/pvc.yaml"
kubectl apply -f "$K8S_DIR/deployment.yaml"
kubectl apply -f "$K8S_DIR/service.yaml"
kubectl apply -f "$K8S_DIR/ingress.yaml"
ok "Manifests applied"

# ── rollout ────────────────────────────────────────────────────────────────
log "Waiting for rollout (max 3 min) …"
kubectl rollout restart deployment/gallery -n gallery 2>/dev/null || true
kubectl rollout status  deployment/gallery -n gallery --timeout=180s
ok "Rollout complete"

# ── summary ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "Gallery deployed!"
echo "   URL : https://${DOMAIN}"
echo "   Admin: https://${DOMAIN}/admin  (admin email + OTP)"
echo "   TLS : cert-manager issuer '${CLUSTER_ISSUER}'"
echo "   Note: first-time TLS cert may take 1-2 minutes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""

_README_MD = """# Gallery — Docker / Kubernetes deploy

## Gyors start (Docker Compose, lokális teszt)

```bash
docker compose up --build
# Megnyit: http://localhost:__PORT__
```

## Kubernetes deploy (saját szerver)

1. Másold fel ezt a mappát a szerverre:
   ```bash
   rsync -az --progress ./ user@server:/opt/gallery/
   ```

2. A szerveren futtasd:
   ```bash
   cd /opt/gallery
   bash deploy.sh
   ```

   Opcionálisan felülírhatod a domain-t / issuer-t:
   ```bash
   bash deploy.sh --domain gallery.example.com --issuer letsencrypt-prod
   ```

3. Kész. A deploy.sh:
   - Elindít egy lokális Docker registry-t (:5000)
   - Beállítja a containerd / k3s registry trust-ot (sudo szükséges)
   - Buildeli és pusholja az image-et
   - Alkalmazza a k8s manifesteket (namespace / secret / pvc / deployment / service / ingress)
   - Megvárja a rollout-ot

## Belépés

- Nyisd meg `https://__DOMAIN__`
- Add meg az email cím-ed → kapsz egy 6 jegyű kódot Gmailben
- Session: 30 perc inaktivitás után jár le (képnézés is aktívitásnak számít)
- Admin felület: `https://__DOMAIN__/admin` (az admin email + OTP)

## Email-lista módosítása (futás közben)

Az admin felületen (`/admin`) bármikor hozzáadhatsz / törölhetsz email
címeket. A változások a `data/allowlist.json` fájlba mentődnek (PVC), és
konténer-újraindítás után is megmaradnak.

## Biztonsági megjegyzés

A `config.json` tartalmazza a Gmail app-jelszót — az image **ne kerüljön
nyilvános Docker registry-be**. A deploy.sh lokális registry-t használ.

## Követelmények

- Docker 20+
- kubectl (beállított kubeconfig)
- nginx-ingress controller
- cert-manager (ClusterIssuer: `__CLUSTER_ISSUER__`)
- k3s esetén: sudo jogosultság a registry-konfig írásához
"""
