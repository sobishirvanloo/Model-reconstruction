; MeshBench - Copyright (c) 2026 Sobhan Shirvanlou. All rights reserved.
; Inno Setup script of the Windows installer (after the PyInstaller build):
;     iscc packaging\meshbench.iss
#define AppName "MeshBench"
#define AppVersion "1.0"
#define AppPublisher "Sobhan Shirvanlou"
#define AppUserModelID "SobhanShirvanlou.MeshBench.1.0"

[Setup]
AppId={{6F1C9A52-3E7D-4B8E-9C21-5A7D2E4B9F10}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/sobishirvanloo
AppSupportURL=https://github.com/sobishirvanloo/Model-reconstruction
AppCopyright=© 2026 {#AppPublisher}. All rights reserved.
VersionInfoVersion={#AppVersion}
VersionInfoCompany={#AppPublisher}
VersionInfoCopyright=© 2026 {#AppPublisher}. All rights reserved.
VersionInfoDescription={#AppName} installer
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequiredOverridesAllowed=dialog
LicenseFile=..\LICENSE
SetupIconFile=..\app\resources\meshbench.ico
UninstallDisplayIcon={app}\MeshBench.exe
UninstallDisplayName={#AppName} {#AppVersion}
OutputDir=..\dist
OutputBaseFilename=MeshBench-Setup-{#AppVersion}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\MeshBench\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\MeshBench.exe"; AppUserModelID: "{#AppUserModelID}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\MeshBench.exe"; Tasks: desktopicon; AppUserModelID: "{#AppUserModelID}"

[Run]
Filename: "{app}\MeshBench.exe"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
