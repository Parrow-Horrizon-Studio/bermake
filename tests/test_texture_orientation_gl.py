"""M7.5b Task 6 regression: a texture must not render vertically mirrored.

The manual visual pass found every texture drawn upside down on the surface --
left/right correct, top/bottom swapped -- and not one of 2211 tests saw it,
because every fixture in the milestone was symmetric under a vertical flip.
tests/test_texture_cache.py holds the seam (what row order reaches
glTexImage2D). This file holds the thing the user actually complained about:
the colour authored at the TOP of the image appears at the TOP of the wall.

It needs a real GL context, so it SKIPS rather than fails wherever one cannot
be created (a headless runner, QT_QPA_PLATFORM=offscreen, a machine with no
GPU). A skip here is not a pass: the seam tests are the ones that run
everywhere, and this one is what proves the seam's convention is the right way
round on real hardware.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

_W = _H = 192

# Vertically asymmetric, and deliberately so: red across the top half, blue
# across the bottom. A fixture that cannot tell an image from its mirror is
# worth less than it looks -- that is the entire lesson of this defect.
_TOP = (220, 30, 30)
_BOTTOM = (30, 30, 220)


def _png(w: int, h: int, rgba: list[int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _top_red_bottom_blue() -> bytes:
    """32 x 32: the top half red, the bottom half blue.

    Big enough that bilinear filtering's blend band across the colour boundary
    is a couple of screen pixels rather than a third of the wall, which is what
    lets the assertions below be about whole regions.
    """
    n = 32
    rows = [[*_TOP, 255] * n for _ in range(n // 2)]
    rows += [[*_BOTTOM, 255] * n for _ in range(n // 2)]
    return _png(n, n, [c for row in rows for c in row])


@pytest.fixture
def gl_context(qapp):
    """A current 3.3 core context on an offscreen surface, or a skip.

    Takes pytest-qt's `qapp` rather than building its own QGuiApplication: a
    process may hold exactly one application object, and creating a second (or
    a non-widget one before the widget tests run) would break every qtbot
    fixture in the suite.
    """
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)

    surface = QOffscreenSurface()
    surface.setFormat(fmt)
    surface.create()
    if not surface.isValid():
        pytest.skip("no usable offscreen surface on this platform")

    ctx = QOpenGLContext()
    ctx.setFormat(fmt)
    if not ctx.create() or not ctx.makeCurrent(surface):
        pytest.skip("no OpenGL 3.3 core context available here")
    try:
        yield ctx
    finally:
        ctx.doneCurrent()


def _fbo(GL):
    fbo = int(GL.glGenFramebuffers(1))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    color = int(GL.glGenTextures(1))
    GL.glBindTexture(GL.GL_TEXTURE_2D, color)
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, _W, _H, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
    )
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glFramebufferTexture2D(
        GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, color, 0
    )
    depth = int(GL.glGenRenderbuffers(1))
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, _W, _H)
    GL.glFramebufferRenderbuffer(
        GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, depth
    )
    if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
        pytest.skip("could not build a complete offscreen framebuffer")
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return fbo


# The wall stands at y = _WALL_Y, between the camera and the origin, so the
# grid and the coloured world axes are all BEHIND it and depth-culled. They
# matter: the Z axis is a blue vertical line and the X axis a red horizontal
# one, and classifying those as texture would answer this test's own question
# with the axis colours. The sampled window is additionally clamped well inside
# the wall's footprint, and the assertion demands that a band be OVERWHELMINGLY
# one colour, so a window that drifted off the wall onto the background fails
# loudly rather than quietly measuring scenery.
_WALL_Y = -1.0
_Z0, _Z1 = 0.4, 2.4


def _wall(scene):
    """A 2 x 2 quad standing upright, facing -Y."""
    v = [
        scene.add_vertex(np.array([-1.0, _WALL_Y, _Z0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, _WALL_Y, _Z0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, _WALL_Y, _Z1], dtype=np.float32)),
        scene.add_vertex(np.array([-1.0, _WALL_Y, _Z1], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_the_top_of_the_image_renders_at_the_top_of_the_wall(gl_context, monkeypatch):
    from OpenGL import GL

    from bermake.model.model import Model
    from bermake.scene.scene import Side, TexturePlacement
    from bermake.viewport import scene_renderer as sr
    from bermake.viewport.camera import Camera

    # A core profile rejects glLineWidth above 1.0, which the renderer sets for
    # its overlays. Pre-existing, unrelated to orientation, and only reachable
    # because this drives the renderer outside the app's own surface format.
    monkeypatch.setattr(sr.GL, "glLineWidth", lambda *_a: None, raising=False)

    _fbo(GL)
    GL.glViewport(0, 0, _W, _H)

    model = Model()
    scene = model.root.mesh
    face = _wall(scene)
    data = _top_red_bottom_blue()
    tex = model.textures.add("wall.png", data, "png", 32, 32, False)
    mat = model.materials.add_custom("Wall", (1.0, 1.0, 1.0))
    # texture_size 2.0 makes the 2-unit-tall wall exactly one whole copy...
    model.materials.edit(mat.id, texture_id=tex.id, texture_size=(2.0, 2.0))
    # BOTH sides, so which way the loop wound is not a variable in this test.
    scene.set_face_material(face, mat.id, Side.FRONT)
    scene.set_face_material(face, mat.id, Side.BACK)
    # ...but UVs are projected from the face CENTROID, so without this the
    # copy straddles the tile seam: the wall's lower half would show the
    # image's upper half and vice versa, which looks exactly like the mirroring
    # this test is hunting and would make the fixture answer its own question
    # backwards. The half-tile shift aligns v = 0 with the foot of the wall and
    # v = 1 with its top (verified corner by corner before this was written).
    for side in (Side.FRONT, Side.BACK):
        scene.set_face_placement(face, TexturePlacement(offset_u=0.5, offset_v=0.5), side)

    mid_z = 0.5 * (_Z0 + _Z1)
    camera = Camera(
        position=np.array([0.0, _WALL_Y - 5.0, mid_z], dtype=np.float32),
        target=np.array([0.0, _WALL_Y, mid_z], dtype=np.float32),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )

    renderer = sr.SceneRenderer()
    renderer.initialize_gl()
    renderer.resize(_W, _H)
    renderer.render(camera, model)
    GL.glFinish()

    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    buf = GL.glReadPixels(0, 0, _W, _H, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
    # GL reads bottom-up; flip so row 0 is the TOP of the screen.
    img = np.frombuffer(buf, dtype=np.uint8).reshape(_H, _W, 3)[::-1].astype(np.int32)

    # A window comfortably inside the wall's footprint (it spans roughly the
    # central 48% of the frame at this camera distance).
    lo, hi = int(_H * 0.35), int(_H * 0.65)
    window = img[lo:hi, int(_W * 0.35) : int(_W * 0.65)]

    # Classify by hue, not by value: the wall is lit, so nothing matches the
    # authored bytes exactly, but a red texel stays red-dominant.
    r, g, b = window[:, :, 0], window[:, :, 1], window[:, :, 2]
    reddish = (r > b + 25) & (r > g + 25)
    bluish = (b > r + 25) & (b > g + 25)

    # Row 0 is the top of the screen. Quarters rather than halves, so the few
    # pixels of bilinear blend across the colour boundary are never sampled.
    q = window.shape[0] // 4
    red_at_top = float(reddish[:q].mean())
    blue_at_bottom = float(bluish[-q:].mean())

    assert red_at_top > 0.9 and blue_at_bottom > 0.9, (
        f"the texture is vertically mirrored: the image's TOP is red and its "
        f"BOTTOM blue, but only {red_at_top:.0%} of the wall's upper band is "
        f"red and {blue_at_bottom:.0%} of its lower band is blue "
        f"(got {float(bluish[:q].mean()):.0%} blue on top, "
        f"{float(reddish[-q:].mean()):.0%} red on the bottom). A photograph "
        f"painted on this wall comes out upside down."
    )
