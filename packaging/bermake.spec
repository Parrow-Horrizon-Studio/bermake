# PyInstaller spec for the Windows build (M7.9, spec 2.2). Run it through
# tools/package_windows.py, which sets the two environment variables below.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

SPEC_DIR = Path(SPECPATH)
MESA_DIR = Path(os.environ["BERMAKE_MESA_DIR"])
VERSION_FILE = os.environ["BERMAKE_VERSION_FILE"]

# The application imports QtCore, QtGui, QtWidgets, QtOpenGLWidgets and QtSvg.
# Excluding the other PySide6 modules keeps their Python bindings out, but
# not the Qt libraries and plugins PySide6's hook collects regardless: the
# QT_LIBRARIES and QT_PLUGINS allow-lists below decide which of those ship.
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

# OpenSSL (libssl-3.dll, libcrypto-3.dll) arrives only through _ssl and
# _hashlib. Bermake makes no network connections. The standard library reaches
# ssl only lazily (logging.handlers' HTTP and SMTP handlers, urllib), always
# inside `except ImportError`, and hashlib, hmac and random fall back to
# Python's built-in hash modules when _hashlib is absent.
NO_OPENSSL = ["ssl", "_ssl", "_hashlib"]

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
    excludes=UNUSED + NO_OPENSSL,
)

# The Mesa DLLs belong only in mesa/. PySide6's own opengl32sw.dll is Mesa
# 11.2.2 (OpenGL 3.0) and cannot run Bermake's shaders; the pinned one in mesa/
# replaces it (spec 2.4.1). Dependency analysis of the pinned loader also
# collects a second libgallium_wgl.dll (59 MB) beside the executable, which
# nothing loads.
MESA_FILES = {"opengl32sw.dll", "libgallium_wgl.dll"}

# PySide6's hook collects every Qt plugin, and plugins drag whole Qt libraries
# in as binary dependencies: the virtual keyboard input context brings
# Qt6Quick, Qt6Qml and Qt6Network, and the PDF image format brings Qt6Pdf. So
# the Qt libraries are limited to what the five modules link against, and the
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

# Collected but never loaded. PyOpenGL's hook bundles its whole DLLS folder
# (freeglut and GLE, for OpenGL.GLUT and OpenGL.GLE, which Bermake does not
# import), and the vc10 builds of those are the only users of MSVCR100.dll.
# Bermake loads no Qt translations.
UNUSED_PREFIXES = ("opengl/dlls/", "pyside6/translations/")
UNUSED_NAMES = {"msvcr100.dll"}


def _dest(entry):
    return Path(entry[0]).as_posix().lower()


def _qt_plugin(dest):
    prefix = "pyside6/plugins/"
    return dest[len(prefix) :] if dest.startswith(prefix) else None


def _keep(entry):
    dest = _dest(entry)
    name = dest.rsplit("/", 1)[-1]
    if dest.startswith(UNUSED_PREFIXES) or name in UNUSED_NAMES:
        return False
    if name in MESA_FILES:
        return dest.startswith("mesa/")
    if name.startswith("qt6") and name.endswith(".dll"):
        return name in QT_LIBRARIES
    plugin = _qt_plugin(dest)
    if plugin is not None:
        return plugin in QT_PLUGINS
    return True


# Fail loudly if an allow-list entry matches nothing, so a PySide6 upgrade
# that renames a library or plugin cannot silently drop it from the bundle.
_collected = {_dest(entry) for entry in a.binaries}
_names = {dest.rsplit("/", 1)[-1] for dest in _collected}
_plugins = {_qt_plugin(dest) for dest in _collected} - {None}
_missing = sorted((QT_LIBRARIES - _names) | (QT_PLUGINS - _plugins))
if _missing:
    raise SystemExit(f"bermake.spec: allow-listed Qt files were not collected: {_missing}")

a.binaries = [entry for entry in a.binaries if _keep(entry)]
a.datas = [entry for entry in a.datas if _keep(entry)]

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
