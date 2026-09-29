"""The OpenGL check that runs before the main window opens (M7.9, final review I1).

On a machine whose driver offers only OpenGL 1.x (Remote Desktop, Windows
Sandbox, a virtual machine without a graphics driver), Qt's own startup test
gives up on the desktop driver and loads `opengl32sw.dll` by name. The
packaged build keeps its Mesa in `mesa/`, which is not on the DLL search
path, so Qt gets no context at all: the viewport's initializeGL never runs,
its check never fires, and the tester would see neither the explanation nor
the offer. This check asks Qt for a context first, using nothing but Qt, so
both still reach the tester. The viewport keeps its own check for shader and
setup failures on a context that is new enough.

The decision is pure (preflight_problem) and the sequence around it is
injectable (run_gl_preflight), so both are tested without a GPU; only
probe_gl touches Qt's OpenGL.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from bermake.diagnostics.gl_check import GlInfo, evaluate_gl

logger = logging.getLogger(__name__)

GL_VERSION = 0x1F02
GL_RENDERER = 0x1F01

NO_CONTEXT_MESSAGE = (
    "Bermake could not find a usable OpenGL driver on this computer: no OpenGL context "
    "could be created. This is common in virtual machines, over Remote Desktop, and "
    "with missing or very old graphics drivers."
)


class PreflightOutcome(Enum):
    """What run_gl_preflight did, so the caller can tell the three apart.

    OK: nothing was wrong, or the check itself could not run; the viewport's
    own check still follows and will report any problem it finds.
    REPORTED: the tester was shown the problem and Bermake carries on
    anyway. The viewport must not show the same OpenGL version problem a
    second time when the window's own context turns out to be the same one.
    RELAUNCHED: a new Bermake was started with compatibility rendering, so
    this one must exit without building a window.
    """

    OK = "ok"
    REPORTED = "reported"
    RELAUNCHED = "relaunched"


@dataclass(frozen=True)
class PreflightProblem:
    message: str
    offer_restart: bool


def preflight_problem(
    info: GlInfo | None, *, mesa_available: bool, compat_active: bool
) -> PreflightProblem | None:
    """What to tell the tester before the window opens, or None if nothing.

    `info` is None when no context could be created at all. The restart is
    offered only where it can help: Mesa is there to switch to, and this
    process is not already using it.
    """
    if info is None:
        message = NO_CONTEXT_MESSAGE
    else:
        # No shader outcome yet: shaders compile in the viewport's own check.
        verdict = evaluate_gl(info, None)
        if verdict.ok:
            return None
        message = verdict.message
    return PreflightProblem(message, mesa_available and not compat_active)


def _gl_string(functions, name: int) -> str:
    value = functions.glGetString(name)
    return value if value else "unknown"


def probe_gl() -> GlInfo | None:
    """Ask Qt for a context with the default format, as the viewport will.

    Needs a QApplication. Returns None if no context can be created or made
    current. Uses Qt's own GL functions rather than PyOpenGL, so it calls the
    same driver Qt chose and imports nothing from OpenGL.
    """
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext

    context = QOpenGLContext()
    if not context.create():
        return None
    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        surface.destroy()
        return None
    try:
        fmt = context.format()
        functions = context.functions()
        return GlInfo(
            version=(fmt.majorVersion(), fmt.minorVersion()),
            version_string=_gl_string(functions, GL_VERSION),
            renderer=_gl_string(functions, GL_RENDERER),
        )
    finally:
        context.doneCurrent()
        surface.destroy()


def run_gl_preflight(
    *,
    mesa_available: bool,
    compat_active: bool,
    prompt: Callable[[str, bool], bool],
    store_preference: Callable[[], None],
    relaunch: Callable[[], bool],
    probe: Callable[[], GlInfo | None] = probe_gl,
) -> PreflightOutcome:
    """Check OpenGL before the window exists. RELAUNCHED means a new Bermake
    was started with compatibility rendering, so main() returns without
    building the window; REPORTED means the tester has already been shown the
    problem (see PreflightOutcome).

    `prompt(message, offer_restart)` is the same dialog the window uses and
    returns True for the restart. Declining, or a problem with no restart on
    offer, continues to the window as before. A probe that raises is logged
    and skipped rather than stopping startup: the viewport's own check still
    follows.
    """
    try:
        info = probe()
    except Exception:
        logger.exception("the OpenGL preflight check failed; continuing without it")
        return PreflightOutcome.OK
    if info is None:
        logger.error("OpenGL preflight: no context could be created")
    else:
        logger.info(
            "OpenGL preflight: %d.%d, %s (%s)",
            info.version[0],
            info.version[1],
            info.version_string,
            info.renderer,
        )
    problem = preflight_problem(info, mesa_available=mesa_available, compat_active=compat_active)
    if problem is None:
        return PreflightOutcome.OK
    logger.error("OpenGL unavailable before the main window opened: %s", problem.message)
    if not prompt(problem.message, problem.offer_restart) or not problem.offer_restart:
        return PreflightOutcome.REPORTED
    store_preference()
    if relaunch():
        logger.info("restarting with compatibility rendering from the startup check")
        return PreflightOutcome.RELAUNCHED
    logger.error("failed to relaunch Bermake for compatibility rendering; continuing")
    return PreflightOutcome.REPORTED
