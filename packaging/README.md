# Desktop app builds

Clinicians use ClinTAB-ML as a normal desktop app. They download one file, open it, and ClinTAB-ML starts in their browser with a small control window. Nothing needs Python, a terminal or admin rights.

| File | What it is |
|------|------------|
| `../clintab_desktop.py` | The launcher the app runs. It starts the server on 127.0.0.1, opens the browser and shows the control window (Open, Open data folder, Quit). `--self-test` runs one full analysis and exits 0 or 1. |
| `clintab.spec` | PyInstaller recipe. Builds `dist/ClinTAB-ML/` (Windows, Linux) or `dist/ClinTAB-ML.app` (macOS). |
| `windows-installer.iss` | Inno Setup script for the Windows installer. It installs per user into `%LOCALAPPDATA%\Programs`, so no admin rights are needed, and adds Start menu and optional desktop shortcuts. |
| `icon.png` | App icon, 1024 px, cut from `assets/CircleACAI.png`. |
| `requirements-build.txt` | Runtime requirements plus PyInstaller and Pillow. |
| `constraints.txt` | Exact versions every installer is built with. Change them on purpose only, then test a build. |

## Releasing

`.github/workflows/release.yml` does all of this on GitHub's Windows, macOS and Linux machines when a release is published:

1. checks that the release tag matches `clintab.__version__` (tag `v1.0.0` for version `1.0.0`)
2. runs the test suite
3. builds the app from the pinned versions in `constraints.txt` (on the Mac, with the numpy and scipy builds that run on macOS 12)
4. runs `--self-test` inside the packaged build, then `--gui-smoke`, which opens the real control window, waits for the server and closes it
5. makes the downloads and attaches them to the release:
   - `ClinTAB-ML-<version>-windows-setup.exe` (installer, the one clinicians should use)
   - `ClinTAB-ML-<version>-windows-portable.zip` (unzip and run, for locked down machines)
   - `ClinTAB-ML-<version>-macos-arm64.dmg` (Apple silicon Macs on macOS 12 or later, drag to Applications)
   - `ClinTAB-ML-<version>-linux-x64.tar.gz`

To try a build without releasing, open the Actions tab on GitHub, pick "Desktop app" and press "Run workflow". The files appear as downloadable artifacts on that run.

For the next version, change `__version__` in `clintab/__init__.py`, `version` and `date-released` in `CITATION.cff`, and add a CHANGELOG entry before tagging.

## Building on your own machine

```bash
pip install -r packaging/requirements-build.txt
pyinstaller packaging/clintab.spec --noconfirm --clean
./dist/ClinTAB-ML/ClinTAB-ML --self-test                    # windows: dist\ClinTAB-ML\ClinTAB-ML.exe
./dist/ClinTAB-ML.app/Contents/MacOS/ClinTAB-ML --self-test   # macos
```

PyInstaller only builds for the system it runs on, so a Windows build needs Windows and a Mac build needs a Mac. That is why the workflow uses GitHub's machines.

## Where the app keeps data

| System | Folder |
|--------|--------|
| Windows | `%LOCALAPPDATA%\ClinTAB-ML` |
| macOS | `~/Library/Application Support/ClinTAB-ML` |
| Linux | `~/.local/share/ClinTAB-ML` |

These are local, unsynced folders on purpose. Documents and Desktop are often synced to OneDrive or iCloud on hospital machines, and uploaded cohorts must not leave the computer. "Open data folder" in the control window opens it, and "Clear all data" in the app deletes every uploaded dataset. Uninstalling the app does not delete this folder.

## When the app won't open

Every start writes a log, `runtime/clintab-desktop.log`, in the data folder (table above). A crash leaves its traceback there too. Two quick checks on a Mac:

```bash
cat ~/Library/Application\ Support/ClinTAB-ML/runtime/clintab-desktop.log
/Applications/ClinTAB-ML.app/Contents/MacOS/ClinTAB-ML        # runs it with output in the terminal
```

On Windows the log is `%LOCALAPPDATA%\ClinTAB-ML\runtime\clintab-desktop.log`. If the app can't start it also says why in its window, or in a system message when the window itself can't open.

## Things to know before giving it to clinicians

- **Unsigned apps get warnings.** Windows SmartScreen shows "Windows protected your PC" (More info, then Run anyway). macOS blocks an app from an unidentified developer; on macOS 15 and later the user has to go to System Settings, Privacy & Security, and press "Open Anyway". Many hospital machines block unsigned software outright. Signing removes this: a code signing certificate for Windows, and an Apple Developer account (about 99 USD a year) to sign and notarize the Mac app. Both can be added to the workflow as repository secrets later.
- **Talk to hospital IT early.** They usually need to approve or whitelist new software, and the per-user installer and the portable zip are the options most likely to be allowed.
- **Intel Macs** are not built by default. GitHub's `macos-latest` machines are Apple silicon. An Intel build needs an Intel runner added to the matrix.
- **Size.** The app is about 300 MB installed, most of it NumPy, SciPy, scikit-learn and statsmodels.
- **Model files run code when loaded.** Clinicians should only load `.pkl` models they made themselves or got from someone they trust.
- **Performance.** Inside the packaged app, scikit-learn uses threads instead of worker processes (worker processes can't be started from a packaged app reliably). Results are identical; very large grid searches can be somewhat slower than running from source.
