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
; Always show the directory page so the user can choose or confirm the install location.
DisableDirPage=no
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
  InstallPath: String;
  Msg: String;
begin
  Result := True;
  if RegQueryStringValue(HKEY_CURRENT_USER, UNINSTALL_REGKEY,
                         'DisplayVersion', PreviousVersion) then begin
    Log('Upgrade detected: installed=' + PreviousVersion + ' -> new={#AppVersion}');
    if not RegQueryStringValue(HKEY_CURRENT_USER, UNINSTALL_REGKEY,
                               'InstallLocation', InstallPath) then
      InstallPath := ExpandConstant('{autopf}\{#AppName}');
    Msg := 'A Face-Local egy korábbi verziója már telepítve van:' + #13#10 + #13#10 +
           '  Jelenlegi verzió: ' + PreviousVersion + #13#10 +
           '  Új verzió:        {#AppVersion}' + #13#10 +
           '  Telepítési hely:  ' + InstallPath + #13#10 + #13#10 +
           'A telepítő frissíti az alkalmazást. Folytatja?';
    if MsgBox(Msg, mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end else
    Log('Fresh installation of {#AppName} {#AppVersion}');
end;

// Download mobilefacenet.tflite via PowerShell if the bundled copy is somehow
// absent.  The model is normally embedded by PyInstaller (package_app.py) so
// this only fires as a fallback (e.g. build produced without the model file).
procedure CurStepChanged(CurStep: TSetupStep);
var
  ModelPath, Script: String;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then Exit;
  ModelPath := ExpandConstant('{app}\models\mobilefacenet.tflite');
  if FileExists(ModelPath) then Exit;

  Log('mobilefacenet.tflite not bundled — attempting post-install download');
  Script :=
    '$f=''' + ModelPath + ''';' +
    'New-Item -ItemType Directory -Force -Path (Split-Path $f)|Out-Null;' +
    '$urls=@(' +
      '''https://github.com/MCarlomagno/FaceRecognitionAuth/raw/refs/heads/master/assets/mobilefacenet.tflite'',' +
      '''https://github.com/pb-julian/liteface/raw/main/tflite_models/mobilefacenet.tflite'',' +
      '''https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android/raw/master/app/src/main/assets/mobile_face_net.tflite'');' +
    'foreach($u in $urls){' +
      'try{Invoke-WebRequest -Uri $u -OutFile "$f.tmp" -UseBasicParsing -ErrorAction Stop;' +
           'Move-Item "$f.tmp" $f -Force;break}' +
      'catch{Remove-Item "$f.tmp" -ea si}}';
  Exec('powershell.exe',
       '-NonInteractive -WindowStyle Hidden -Command "' + Script + '"',
       '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  if FileExists(ModelPath) then
    Log('mobilefacenet.tflite downloaded successfully')
  else
    Log('WARNING: mobilefacenet.tflite download failed (Settings > AI csomagok can retry)');
end;
