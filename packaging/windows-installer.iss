; inno setup script for the windows installer. built by .github/workflows/release.yml:
;   iscc /DMyAppVersion=1.0.0 packaging\windows-installer.iss
; installs per user into %LOCALAPPDATA%\Programs, so no admin rights are needed,
; which is what most hospital computers allow.

#define MyAppName "ClinTAB-ML"
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif

[Setup]
; keep this AppId the same in every release so new versions upgrade the old one
AppId={{27501F89-A07B-4AB2-8AEB-72658A480CAD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Applied Clinical AI Initiative
AppPublisherURL=https://github.com/Applied-Clinical-AI-Initiative/ClinTab-ML
DefaultDirName={localappdata}\Programs\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=ClinTAB-ML-{#MyAppVersion}-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\ClinTAB-ML.exe
LicenseFile=..\LICENSE

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\ClinTAB-ML\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\ClinTAB-ML.exe"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\ClinTAB-ML.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\ClinTAB-ML.exe"; Description: "Start ClinTAB-ML now"; Flags: nowait postinstall skipifsilent
