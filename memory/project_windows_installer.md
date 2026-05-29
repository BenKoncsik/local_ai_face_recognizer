---
name: windows-installer
description: Windows Inno Setup installer and in-app update flow — technology choice, file locations, upgrade behavior
metadata:
  type: project
---

Windows installer uses **Inno Setup 6** (`scripts/build_windows_installer.iss`).

**Why:** Single `.iss` script, per-user install without UAC, native wizard UI, automatic upgrade detection via AppId GUID. WiX would be the next choice if enterprise GPO deployment is ever needed.

**AppId GUID:** `{6CC9EB5A-2A67-4B10-B8B3-7D9E7C871B6E}` — must never change.

**Install location:** `%LocalAppData%\Programs\Face-Local` (per-user, `PrivilegesRequired=lowest`).

**User data preserved across upgrade/uninstall** (never touched by installer):
- `%AppData%\Face-Local\` — database, Drive mirrors
- `%LocalAppData%\Face-Local\` — logs, updater log
- `%UserProfile%\Documents\localAIFaceRecognizer\settings\settings.ini` — settings INI

**Key installer flags:** `CloseApplications=yes`, `RestartApplications=yes`, `SetupLogging=yes`, `DisableDirPage=auto`, `SetupMutex=FaceLocalSetupMutex`.

**In-app update:** `app/services/update_service._apply_windows_exe()` calls `ShellExecuteW("open", ...)` with `/SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS`. Uses `"open"` (not `"runas"`) because per-user installer needs no elevation.

**How it:** Workflow calls `ISCC.exe /DAppVersion=x.y.z scripts\build_windows_installer.iss` → outputs `release\Face-Local-windows-installer-{version}.exe`. Portable ZIP is also produced as alternative.

**Docs:** `docs/windows-installer.md`
