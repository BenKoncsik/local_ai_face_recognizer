#define AppName "Face-Local"
#define AppPublisher "Face-Local"
#define AppExeName "Face-Local.exe"
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define SourceDir "..\\dist\\Face-Local"
#define OutputDir "..\\release"

[Setup]
; IMPORTANT: Keep this AppId constant across all releases.
; Changing it breaks upgrade detection and leaves orphaned registry entries.
AppId={{6CC9EB5A-2A67-4B10-B8B3-7D9E7C871B6E}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
; Per-user install: {autopf} maps to %LocalAppData%\Programs — no UAC needed.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Skip the directory page when upgrading (preserves the previous install location).
DisableDirPage=auto
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename=Face-Local-windows-installer-{#AppVersion}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
; Prevent two simultaneous installs or upgrades running at once.
SetupMutex=FaceLocalSetupMutex
; Gracefully close a running Face-Local before overwriting its files.
CloseApplications=yes
CloseApplicationsFilter=Face-Local.exe
; After a silent upgrade (/RESTARTAPPLICATIONS flag) relaunch the app automatically.
RestartApplications=yes
; Write a setup log to %TEMP% — useful for diagnosing failed installs.
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; Flags: unchecked

[Files]
; ignoreversion: overwrite on upgrade without comparing version resources.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "&Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; On explicit uninstall: remove the install directory entirely.
;
; User data directories are intentionally NOT touched here — they survive
; both upgrades and uninstall by design:
;   %AppData%\Face-Local\                         — database, Drive mirrors
;   %LocalAppData%\Face-Local\                    — logs, updater log, cache
;   %LocalAppData%\Temp\Face-Local\               — Drive download cache
;   %UserProfile%\Documents\localAIFaceRecognizer\ — settings INI
Type: filesandordirs; Name: "{app}"

[Code]
// Registry key written by Inno Setup for this installation.
// Used to detect an existing install and log the upgrade path.
const
  UNINSTALL_REGKEY = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{6CC9EB5A-2A67-4B10-B8B3-7D9E7C871B6E}_is1';

function InitializeSetup(): Boolean;
var
  PreviousVersion: String;
begin
  Result := True;
  if RegQueryStringValue(HKEY_CURRENT_USER, UNINSTALL_REGKEY,
                         'DisplayVersion', PreviousVersion) then
    Log('Upgrade detected: installed=' + PreviousVersion + ' -> new={#AppVersion}')
  else
    Log('Fresh installation of {#AppName} {#AppVersion}');
end;
