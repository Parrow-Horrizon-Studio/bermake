# PyInstaller spec for the Windows build (M7.9, spec 2.2). Run it through
# tools/package_windows.py, which sets the two environment variables below.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH)
MESA_DIR = Path(os.environ["BERMAKE_MESA_DIR"])
VERSION_FILE = os.environ["BERMAKE_VERSION_FILE"]

# The application imports QtCore, QtGui, QtWidgets, QtOpenGLWidgets and QtSvg.
# Everything else PySide6 ships stays behind; the 100 MB ceiling in
# tools/package_windows.py catches anything that slips through.
UNUSED = [
    "tkinter",
    "PySide6.Qt3DCore",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtMultimedia",
    "PySide6.QtNetwork",
    "PySide6.QtPdf",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
]

a = Analysis(
    [str(SPEC_DIR / "entry.py")],
    binaries=[
        (str(MESA_DIR / "opengl32sw.dll"), "mesa"),
        (str(MESA_DIR / "libgallium_wgl.dll"), "mesa"),
    ],
    # Icons and shaders, plus the dist-info importlib.metadata reads for the
    # smoke test's version check. Compiled modules are binaries, not data.
    datas=collect_data_files("bermake", excludes=["**/*.pyd"]) + copy_metadata("bermake"),
    hiddenimports=collect_submodules("bermake"),
    excludes=UNUSED,
)

# The Mesa DLLs belong only in mesa/. PySide6's own opengl32sw.dll is Mesa
# 11.2.2 (OpenGL 3.0) and cannot run Bermake's shaders; the pinned one in mesa/
# replaces it (spec 2.4.1). Dependency analysis of the pinned loader also
# collects a second libgallium_wgl.dll (59 MB) beside the executable, which
# nothing loads.
MESA_FILES = {"opengl32sw.dll", "libgallium_wgl.dll"}

# Excluding the unused Python modules above does not stop PySide6's hook
# collecting every Qt plugin, and plugins drag whole Qt libraries in as
# binary dependencies: the virtual keyboard input context brings Qt6Quick,
# Qt6Qml and Qt6Network, and the PDF image format brings Qt6Pdf. So the Qt
# libraries are limited to what the five modules link against, and the
# plugins to the ones Bermake uses: the windows platform, the SVG and JPEG
# image formats (JPEG textures decode through QImage), and the Windows 11
# widget style, without which Qt falls back to the classic Windows look.
QT_LIBRARIES = {
    "qt6core.dll",
    "qt6gui.dll",
    "qt6widgets.dll",
    "qt6opengl.dll",
    "qt6openglwidgets.dll",
    "qt6svg.dll",
}
QT_PLUGINS = {
    "platforms/qwindows.dll",
    "imageformats/qsvg.dll",
    "imageformats/qjpeg.dll",
    "styles/qmodernwindowsstyle.dll",
}


def _keep(entry):
    dest = Path(entry[0])
    name = dest.name.lower()
    if name in MESA_FILES:
        return dest.parent.as_posix() == "mesa"
    if name.startswith("qt6") and name.endswith(".dll"):
        return name in QT_LIBRARIES
    parts = [part.lower() for part in dest.parts]
    if parts[:2] == ["pyside6", "plugins"]:
        return "/".join(parts[2:]) in QT_PLUGINS
    return True


a.binaries = [entry for entry in a.binaries if _keep(entry)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Bermake",
    console=False,
    version=VERSION_FILE,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Bermake", upx=False)
