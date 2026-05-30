---
name: project-sparkle-updater
description: Sparkle 2 auto-updater integration — architecture, files created, and secrets required
metadata:
  type: project
---

## Sparkle 2 integráció implementálva

**Miért:** A manuális DMG-alapú frissítési rendszert (app/services/update_service.py) kiegészíti egy teljesen automatikus Sparkle 2 alapú macOS frissítés.

**How to apply:** Az app PyInstaller-bundled Python/PySide6 app, nem natív Swift. A Sparkle integráció ezért `SparkleHelper` Swift subprocess-el történik (nem direkt framework link), hogy elkerüljük az NSApplication ütközést.

## Létrehozott/módosított fájlok

- `sparkle/Sources/SparkleHelper/main.swift` — Swift subprocess Sparkle 2 wrapper
- `app/updater/__init__.py` + `app/updater/sparkle_bridge.py` — Python oldal, subprocess launcher
- `assets/entitlements.plist` — Hardened runtime entitlements PyInstaller-hez
- `scripts/generate_appcast.py` — CI szkript appcast.xml generáláshoz
- `docs/appcast.xml` — Stable URL: `https://raw.githubusercontent.com/BenKoncsik/local_ai_face_recognizer/main/docs/appcast.xml`
- `.github/workflows/build-release.yml` — Teljes CI/CD pipeline Sparkle lépésekkel

## Szükséges GitHub Secrets

| Secret | Leírás |
|--------|--------|
| `DEVELOPER_ID_CERT_P12` | Developer ID Application cert (base64 P12) |
| `DEVELOPER_ID_CERT_PASSWORD` | P12 jelszó |
| `APPLE_TEAM_ID` | Apple Developer Team ID |
| `ASC_API_KEY_PRIVATE_KEY` | App Store Connect API key (base64 .p8) |
| `ASC_API_KEY_ID` | ASC API Key ID |
| `ASC_API_KEY_ISSUER_ID` | ASC Issuer ID |
| `SPARKLE_PRIVATE_KEY` | EdDSA private key (base64, Sparkle `generate_keys` output) |
| `SPARKLE_PUBLIC_KEY` | EdDSA public key (bekerül Info.plist SUPublicEDKey-be) |

## Meglévő update_service.py

- Megmarad! macOS DMG, Windows EXE/ZIP, Linux DEB/tar.gz manuális frissítés kezelésére
- A Sparkle bridge ettől teljesen független, nem interferál

## app/main.py integrációhoz szükséges (NEM implementálva, user feladata)

```python
from app.updater import start_background_update_check
# A QApplication létrehozása után, sys.exit(app.exec()) előtt:
start_background_update_check()  # macOS-on indít SparkleHelper subprocess-t
```

## Sparkle kulcs generálás (egyszeri setup)

```bash
# Töltsd le Sparkle 2.6.4-et
curl -fsSL https://github.com/sparkle-project/Sparkle/releases/download/2.6.4/Sparkle-2.6.4.tar.xz | tar xJ
./bin/generate_keys
# → kiírja a public key-t → SPARKLE_PUBLIC_KEY secret
# → elmenti a private key-t → base64 encode → SPARKLE_PRIVATE_KEY secret
```
