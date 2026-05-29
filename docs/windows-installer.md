# Windows Installer

## Technology choice: Inno Setup

Inno Setup was chosen over WiX, MSIX, and Squirrel for these reasons:

| Criterion | Inno Setup | WiX | MSIX | Squirrel |
|-----------|-----------|-----|------|----------|
| Setup complexity | Low — single `.iss` script | High — XML + toolchain | Medium — requires signing | Medium — NuGet packaging |
| PyInstaller compatibility | Excellent | Good | Friction (sandboxing) | Good |
| Per-user install (no UAC) | ✅ `PrivilegesRequired=lowest` | Possible | Default | Default |
| Upgrade with same ID | ✅ `AppId` GUID | ✅ `ProductCode` | ✅ | ✅ auto |
| CI without extra tooling | ✅ Chocolatey or pre-installed | Needs WiX SDK | Needs SDK + cert | Needs NuGet |
| Wizard UX | ✅ Modern wizard style | Basic | Store-like | No wizard |

WiX would be the next choice if MSI-level deployment policies are ever needed (e.g., enterprise GPO).

---

## Building the installer locally

### Prerequisites

Install Inno Setup 6 from [jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php) or via Chocolatey:

```powershell
choco install innosetup -y
```

### Steps

```powershell
# 1. Build the PyInstaller bundle (produces dist\Face-Local\)
python scripts/package_app.py --version 0.1.0

# 2. Compile the installer
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DAppVersion=0.1.0 scripts\build_windows_installer.iss
```

The output is `release\Face-Local-windows-installer-0.1.0.exe`.

---

## Release workflow

The GitHub Actions workflow (`build-release.yml`) does this automatically on every push to `main`:

1. `prepare` — derives the next semantic version from `pyproject.toml` and existing git tags.
2. `build-windows` — on the Windows runner:
   - Installs Python dependencies and PyInstaller.
   - Downloads embedding models.
   - Runs `package_app.py` → `dist\Face-Local\`.
   - Creates the portable ZIP: `release\Face-Local-windows-portable-{version}.zip`.
   - Runs Inno Setup → `release\Face-Local-windows-installer-{version}.exe`.
3. `release` — uploads both the `.exe` installer and the `.zip` portable archive to the GitHub release.

The **primary published artifact** for Windows is the `.exe` installer.  
The `.zip` portable archive is also published as an alternative for users who cannot or prefer not to run a setup wizard.

---

## In-app update flow (Windows)

When the user triggers an update from inside the running app:

```
UpdateDialog
  └─ download_asset()     — streams .exe to %TEMP%\face-local-update-*.exe
  └─ apply_update()
       └─ _apply_windows_exe()
            1. Builds Inno Setup flags:
               /SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /LOG="…\face_local_installer.log"
            2. ShellExecuteW("open", installer.exe, flags)
               — no forced elevation; installer handles its own privilege level
            3. sys.exit(0) — app closes itself
  └─ Inno Setup runs:
       • Detects CloseApplications=yes → any remaining Face-Local.exe process is terminated
       • Overwrites application binaries in {app} (ignoreversion flag)
       • /RESTARTAPPLICATIONS → relaunches Face-Local after install completes
```

The installer log is written to:
```
%LocalAppData%\Face-Local\logs\face_local_installer.log
```

The updater helper log (for the download step) is at:
```
%LocalAppData%\Face-Local\logs\face_local_updater.log
```

---

## What is preserved across upgrades and uninstall

The installer only touches the application directory (`{autopf}\Face-Local`, typically
`%LocalAppData%\Programs\Face-Local`).  All user data lives outside this directory and
is never modified by the installer or uninstaller:

| Directory | Contents |
|-----------|----------|
| `%AppData%\Face-Local\` | SQLite database, Google Drive mirrors |
| `%LocalAppData%\Face-Local\` | Log files, updater log |
| `%LocalAppData%\Temp\Face-Local\` | Google Drive download cache (ephemeral) |
| `%UserProfile%\Documents\localAIFaceRecognizer\settings\settings.ini` | Application settings (INI) |

These paths are defined in `app/paths.py` and `app/app_settings.py`.

---

## Upgrade detection

The installer uses the fixed AppId GUID `{6CC9EB5A-2A67-4B10-B8B3-7D9E7C871B6E}`.
Inno Setup writes the current version to:

```
HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\{6CC9EB5A-...}_is1\DisplayVersion
```

On the next install run, `InitializeSetup()` in the `[Code]` section reads this key and
logs whether it is a fresh install or an upgrade.  The directory page is automatically
skipped on upgrade (`DisableDirPage=auto`) so the existing install location is preserved.

---

## Silent install / CI

To install silently (e.g., from a script or test environment):

```powershell
Face-Local-windows-installer-1.2.3.exe /VERYSILENT /NORESTART
```

To install and relaunch automatically (same as the in-app updater):

```powershell
Face-Local-windows-installer-1.2.3.exe /SILENT /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS
```
