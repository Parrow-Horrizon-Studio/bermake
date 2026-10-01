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
import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from bermake.diagnostics.gl_check import MIN_GL_VERSION, GlInfo, evaluate_gl

logger = logging.getLogger(__name__)

GL_VERSION = 0x1F02
GL_RENDERER = 0x1F01
GL_COLOR_BUFFER_BIT = 0x4000
GL_TRIANGLES = 0x0004
GL_FLOAT = 0x1406

# The draw test: a red triangle on a blue 32x32 image. Pure channel values
# survive sRGB and precision differences between drivers.
DRAW_TEST_SIZE = 32
DRAW_TEST_TOLERANCE = 8
_BLUE = (0, 0, 255)
_RED = (255, 0, 0)
# The triangle covers the centre and no corner.
_CENTRE_PIXEL = (DRAW_TEST_SIZE // 2, DRAW_TEST_SIZE // 2)
_CORNER_PIXEL = (1, 1)

# The vertices go through a buffer on attribute 0 with a VAO bound, the way
# every viewport draw does, so the test fails only where the viewport would. A
# compatibility-profile driver may skip a draw whose attribute 0 is not enabled.
_DRAW_TRIANGLE = struct.pack("6f", -0.5, -0.5, 0.5, -0.5, 0.0, 0.5)
_DRAW_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""
_DRAW_FRAGMENT_SHADER = """#version 330 core
out vec4 colour;
void main() {
    colour = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

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
        # The draw test's failure counts as a setup error; the viewport's own
        # shaders compile later, in its own check.
        verdict = evaluate_gl(info, info.draw_error)
        if verdict.ok:
            return None
        message = verdict.message
    return PreflightProblem(message, mesa_available and not compat_active)


def _gl_string(functions, name: int) -> str:
    value = functions.glGetString(name)
    return value if value else "unknown"


def _near(value: tuple[int, int, int], expected: tuple[int, int, int]) -> bool:
    return all(
        abs(got - want) <= DRAW_TEST_TOLERANCE for got, want in zip(value, expected, strict=True)
    )


def draw_test_problem(corner: tuple[int, int, int], centre: tuple[int, int, int]) -> str | None:
    """What is wrong with the draw test's read-back image, or None if it is right.

    Expected: blue at the corner, where nothing was drawn, and red at the
    centre, where the triangle is.
    """
    if _near(corner, _BLUE) and _near(centre, _RED):
        return None
    return (
        "a test image came back wrong (expected red on blue, "
        f"read rgb({centre[0]}, {centre[1]}, {centre[2]}) at the centre and "
        f"rgb({corner[0]}, {corner[1]}, {corner[2]}) at the corner)"
    )


def _first_line(text: str, fallback: str) -> str:
    lines = text.strip().splitlines()
    return lines[0] if lines else fallback


def _rgb_at(image, pixel: tuple[int, int]) -> tuple[int, int, int]:
    colour = image.pixelColor(pixel[0], pixel[1])
    return (colour.red(), colour.green(), colour.blue())


def _draw_test(context) -> str | None:
    """Draw a red triangle on a blue offscreen image and read it back.

    Needs `context` current. Returns the driver's failure as text, or None if
    the image came back right. Qt classes only, so no OpenGL module is
    imported. Every GL object is released before returning, while the context
    is still current.
    """
    from PySide6.QtOpenGL import (
        QOpenGLBuffer,
        QOpenGLFramebufferObject,
        QOpenGLShader,
        QOpenGLShaderProgram,
        QOpenGLVertexArrayObject,
    )

    size = DRAW_TEST_SIZE
    functions = context.functions()
    fbo = QOpenGLFramebufferObject(size, size)
    program = QOpenGLShaderProgram()
    vao = QOpenGLVertexArrayObject()
    buffer = QOpenGLBuffer()
    try:
        if not fbo.isValid():
            return "could not create an offscreen image"
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Vertex, _DRAW_VERTEX_SHADER
        ):
            return _first_line(program.log(), "the test shader would not compile")
        if not program.addShaderFromSourceCode(
            QOpenGLShader.ShaderTypeBit.Fragment, _DRAW_FRAGMENT_SHADER
        ):
            return _first_line(program.log(), "the test shader would not compile")
        if not program.link():
            return _first_line(program.log(), "the test shader would not link")
        vao.create()
        fbo.bind()
        program.bind()
        vao.bind()
        buffer.create()
        buffer.bind()
        buffer.allocate(_DRAW_TRIANGLE, len(_DRAW_TRIANGLE))
        program.enableAttributeArray(0)
        program.setAttributeBuffer(0, GL_FLOAT, 0, 2)
        functions.glViewport(0, 0, size, size)
        functions.glClearColor(0.0, 0.0, 1.0, 1.0)
        functions.glClear(GL_COLOR_BUFFER_BIT)
        functions.glDrawArrays(GL_TRIANGLES, 0, 3)
        functions.glFinish()
        buffer.release()
        vao.release()
        program.release()
        fbo.release()
        image = fbo.toImage()

        return draw_test_problem(_rgb_at(image, _CORNER_PIXEL), _rgb_at(image, _CENTRE_PIXEL))
    finally:
        if buffer.isCreated():
            buffer.destroy()
        if vao.isCreated():
            vao.destroy()
        program.removeAllShaders()
        del program, buffer, vao, fbo


def _guarded_draw_test(draw: Callable[[], str | None], *, strict: bool = False) -> str | None:
    """Run the draw step. A failure the driver reports comes back as text; an
    unexpected exception is our own bug, not the driver's, so it is logged and
    does not count against the driver (which would push the tester onto the
    slow renderer for nothing). `strict` lets such an exception propagate, for
    the smoke test, which must tell "drew correctly" from "crashed"."""
    try:
        return draw()
    except Exception:
        if strict:
            raise
        logger.exception(
            "the OpenGL draw test could not run because of an error in Bermake, "
            "not the driver; it is being treated as passed"
        )
        return None


def probe_gl(strict: bool = False) -> GlInfo | None:
    """Ask Qt for a context with the default format, as the viewport will.

    Needs a QApplication. Returns None if no context can be created or made
    current. Uses Qt's own GL functions rather than PyOpenGL, so it calls the
    same driver Qt chose and imports nothing from OpenGL. A context that
    reports a good version also has to draw a small test image: some drivers
    claim OpenGL 4.6 and draw nothing (Windows Sandbox with the host's GPU
    driver), and the reported version alone cannot tell. With `strict`, an
    unexpected error in the draw step propagates instead of being logged and
    ignored.
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
        version = (fmt.majorVersion(), fmt.minorVersion())
        # Only a context that claims to be new enough is worth drawing on; an
        # old one already gets its own message.
        draw_error = (
            _guarded_draw_test(lambda: _draw_test(context), strict=strict)
            if version >= MIN_GL_VERSION
            else None
        )
        return GlInfo(
            version=version,
            version_string=_gl_string(functions, GL_VERSION),
            renderer=_gl_string(functions, GL_RENDERER),
            draw_error=draw_error,
        )
    finally:
        context.doneCurrent()
        surface.destroy()


def _log_draw_outcome(info: GlInfo) -> None:
    if info.version < MIN_GL_VERSION:
        logger.info("OpenGL preflight draw test: not run, the version is too low")
    elif info.draw_error is None:
        logger.info("OpenGL preflight draw test: passed")
    else:
        logger.error("OpenGL preflight draw test: failed, %s", info.draw_error)


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
        _log_draw_outcome(info)
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
