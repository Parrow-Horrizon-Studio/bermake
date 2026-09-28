"""Compatibility rendering: drawing through the bundled Mesa (M7.9, spec 2.4.1).

Two switches, and both must happen before anything imports OpenGL.GL and
before QApplication exists:

1. PyOpenGL must call the same DLL Qt uses. Its Windows platform loads
   `opengl32` by name, which is always the system driver, so without this
   every GL call fails once Qt has switched to Mesa. Loading the bundled
   loader by full path also makes Windows resolve libgallium_wgl.dll from the
   same folder, and Qt's own later load of `opengl32sw.dll` by name returns
   the module already in the process.
2. Qt.AA_UseSoftwareOpenGL, which Qt honours only before QApplication.

Mesa then picks its own driver: Direct3D 12 on the real GPU where available,
llvmpipe on the CPU otherwise. Setting GALLIUM_DRIVER=llvmpipe forces the
latter.
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from bermake.diagnostics.launch import COMPAT_FLAG

MESA_DIR_NAME = "mesa"
MESA_LOADER = "opengl32sw.dll"
MESA_DRIVER = "libgallium_wgl.dll"

# python/bermake/diagnostics/compat_rendering.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def locate_mesa_dir(*, frozen: bool, bundle_dir: Path | None, repo_root: Path) -> Path | None:
    """The folder holding both Mesa DLLs, or None if either is missing."""
    if frozen:
        if bundle_dir is None:
            return None
        candidate = bundle_dir / MESA_DIR_NAME
    else:
        candidate = repo_root / "build" / MESA_DIR_NAME
    if (candidate / MESA_LOADER).is_file() and (candidate / MESA_DRIVER).is_file():
        return candidate
    return None


def default_mesa_dir() -> Path | None:
    frozen = bool(getattr(sys, "frozen", False))
    meipass = getattr(sys, "_MEIPASS", None)
    return locate_mesa_dir(
        frozen=frozen,
        bundle_dir=Path(meipass) if meipass else None,
        repo_root=_REPO_ROOT,
    )


def _redirect_pyopengl(loader: Path) -> None:
    import OpenGL.platform

    OpenGL.platform.PLATFORM.GL = ctypes.WinDLL(str(loader))


def _set_software_attribute() -> None:
    from PySide6.QtCore import QCoreApplication, Qt

    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)


def enable_compatibility_rendering(
    mesa_dir: Path,
    *,
    set_gl_library: Callable[[Path], None] | None = None,
    set_attribute: Callable[[], None] | None = None,
) -> None:
    if "OpenGL.GL" in sys.modules:
        raise RuntimeError(
            "compatibility rendering must be enabled before OpenGL.GL is imported; "
            "PyOpenGL has already bound the system driver"
        )
    (set_gl_library or _redirect_pyopengl)(mesa_dir / MESA_LOADER)
    (set_attribute or _set_software_attribute)()


def compatibility_rendering_active() -> bool:
    from PySide6.QtCore import QCoreApplication, Qt

    return QCoreApplication.testAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL)


def relaunch_command(
    *, frozen: bool, executable: str, argv: Sequence[str], orig_argv: Sequence[str]
) -> list[str]:
    """The command that started this process, minus the one-launch flag.

    From a checkout the executable is the Python interpreter, so its own
    arguments (-m bermake.app, or a console-script launcher) come from
    sys.orig_argv. The flag is stripped so the stored preference governs the
    new process.
    """
    arguments = argv[1:] if frozen else orig_argv[1:]
    return [executable, *(a for a in arguments if a != COMPAT_FLAG)]


def _start_detached(program: str, arguments: list[str]) -> bool:
    from PySide6.QtCore import QProcess

    started, _pid = QProcess.startDetached(program, arguments)
    return bool(started)


def relaunch(
    *,
    command: list[str] | None = None,
    start_detached: Callable[[str, list[str]], bool] | None = None,
) -> bool:
    if command is None:
        command = relaunch_command(
            frozen=bool(getattr(sys, "frozen", False)),
            executable=sys.executable,
            argv=sys.argv,
            orig_argv=sys.orig_argv,
        )
    return (start_detached or _start_detached)(command[0], command[1:])
