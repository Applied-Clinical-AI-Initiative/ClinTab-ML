# pyinstaller recipe for the double-click desktop app.
#
#   pip install -r packaging/requirements-build.txt
#   pyinstaller packaging/clintab.spec --noconfirm --clean
#
# output: dist/ClinTAB-ML/ (windows, linux) or dist/ClinTAB-ML.app (macos).
# a folder build ("onedir") starts in a few seconds, a single-file build would
# unpack ~400 MB of numpy/scipy/sklearn on every launch.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, ROOT)
from clintab import __version__  # noqa: E402

ICON = os.path.join(ROOT, "packaging", "icon.png")   # pillow turns it into .ico / .icns

datas = [
    (os.path.join(ROOT, "static"), "static"),
    (os.path.join(ROOT, "Data", "sample_clinical.csv"), "Data"),
    (os.path.join(ROOT, "LICENSE"), "."),
]
datas += collect_data_files("lifelines")
for pkg in ("scikit-learn", "imbalanced-learn", "lifelines", "statsmodels", "pandas", "numpy", "scipy", "flask"):
    datas += copy_metadata(pkg)

hiddenimports = collect_submodules("imblearn") + collect_submodules("lifelines")

a = Analysis(
    [os.path.join(ROOT, "clintab_desktop.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6", "PySide2", "PySide6", "gunicorn"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ClinTAB-ML",
    console=False,          # no terminal window for clinicians
    icon=ICON,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="ClinTAB-ML", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ClinTAB-ML.app",
        icon=ICON,
        bundle_identifier="org.appliedclinicalai.clintabml",
        version=__version__,
        info_plist={
            "CFBundleDisplayName": "ClinTAB-ML",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            # scikit-learn's wheels need macos 12, and the workflow installs the
            # numpy and scipy builds that run there too (not the macos 14 ones)
            "LSMinimumSystemVersion": "12.0",
        },
    )
