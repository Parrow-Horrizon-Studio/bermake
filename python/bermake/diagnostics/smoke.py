"""`Bermake.exe --smoke-test <report.json>` (M7.9, spec 2.6).

Runs inside the built executable, since that is what ships, and writes one
JSON entry per check. The build script compares the resource counts against
the source tree, so a file left out of the bundle fails the build without
anyone maintaining a number by hand.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bermake.diagnostics.compat_rendering import default_mesa_dir, enable_compatibility_rendering

_SQUARE = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 3.0, 0.0), (0.0, 3.0, 0.0))


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, object] = field(default_factory=dict)


def _square_model():
    import numpy as np

    from bermake.model.model import Model

    model = Model()
    ids = [model.root.mesh.add_vertex(np.array(p, dtype=np.float32)) for p in _SQUARE]
    model.root.mesh.add_face_from_loop(ids)
    return model


def check_imports() -> CheckResult:
    from importlib.metadata import version as distribution_version

    import bermake
    from bermake import _core

    declared = distribution_version("bermake")
    compiled = _core.version()
    ok = declared == compiled == bermake.__version__
    return CheckResult("imports", ok, f"metadata {declared}, compiled {compiled}")


def check_resources() -> CheckResult:
    from importlib.resources import files

    from PySide6.QtGui import QColor

    from bermake.ui.icons import available_icon_stems, icon_pixmap
    from bermake.viewport.scene_renderer import _load_shader_source

    stems = sorted(available_icon_stems())
    blank = [s for s in stems if icon_pixmap(s, 24, QColor("black")).isNull()]
    shader_dir = files("bermake.viewport") / "shaders"
    shaders = sorted(entry.name for entry in shader_dir.iterdir() if entry.is_file())
    empty = [name for name in shaders if not _load_shader_source(name).strip()]
    ok = bool(stems) and bool(shaders) and not blank and not empty
    detail = f"blank icons: {blank}, empty shaders: {empty}" if not ok else ""
    return CheckResult("resources", ok, detail, {"icons": len(stems), "shaders": len(shaders)})


def check_document_roundtrip(tmp_dir: Path) -> CheckResult:
    from bermake.document import DocumentSettings
    from bermake.io import load_document, save_document
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import RenderStyle

    path = Path(tmp_dir) / "smoke.berm"
    save_document(path, _square_model(), Camera(), DocumentSettings(), RenderStyle())
    mesh = load_document(path).model.root.mesh
    faces = len(list(mesh.faces_iter()))
    points = sorted(tuple(round(float(c), 4) for c in v.position) for v in mesh.vertices_iter())
    ok = faces == 1 and points == sorted(_SQUARE)
    return CheckResult(
        "document",
        ok,
        "" if ok else f"faces {faces}, points {points}",
        {"faces": faces, "vertices": len(points)},
    )


def check_rendering(width: int = 64, height: int = 64) -> CheckResult:
    """Compile every shader and draw into an offscreen framebuffer, twice: an
    empty model, then the square. Passes only if the two frames differ."""
    import numpy as np
    from OpenGL import GL
    from PySide6.QtGui import QOffscreenSurface, QOpenGLContext

    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.scene_renderer import SceneRenderer

    context = QOpenGLContext()
    if not context.create():
        return CheckResult("rendering", False, "could not create an OpenGL context")
    surface = QOffscreenSurface()
    surface.setFormat(context.format())
    surface.create()
    if not context.makeCurrent(surface):
        return CheckResult("rendering", False, "could not make the context current")
    try:
        version = GL.glGetString(GL.GL_VERSION).decode("utf-8", errors="replace")
        renderer = GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace")
        data: dict[str, object] = {"gl_version": version, "renderer": renderer}
        match = re.match(r"(\d+)\.(\d+)", version)
        numeric = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        if "Mesa" not in version or numeric < (3, 3):
            return CheckResult("rendering", False, f"not the bundled Mesa: {version}", data)

        framebuffer = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, framebuffer)
        colour, depth = GL.glGenRenderbuffers(2)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, colour)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, width, height)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, colour
        )
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH24_STENCIL8, width, height)
        GL.glFramebufferRenderbuffer(
            GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, depth
        )
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            return CheckResult("rendering", False, "offscreen framebuffer incomplete", data)

        scene_renderer = SceneRenderer()
        scene_renderer.initialize_gl()
        scene_renderer.resize(width, height)
        camera = Camera()
        camera.aspect = width / height

        def frame(model) -> np.ndarray:
            scene_renderer.render(camera, model)
            GL.glFinish()
            pixels = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
            return np.frombuffer(pixels, dtype=np.uint8).reshape(-1, 4).copy()

        # The background and grid alone already give more than one colour, so
        # only the difference from an empty model proves the model was drawn
        # (final review M2).
        empty = frame(Model())
        drawn = frame(_square_model())
        data["distinct_colours"] = len(np.unique(drawn, axis=0))
        changed = changed_pixels(empty, drawn)
        data["model_pixels"] = changed
        ok = changed > 0
        detail = "" if ok else "the model frame is identical to the empty frame"
        return CheckResult("rendering", ok, detail, data)
    finally:
        context.doneCurrent()


def changed_pixels(before, after) -> int:
    """How many RGBA pixels differ between two frames of the same size."""
    import numpy as np

    before = np.asarray(before, dtype=np.uint8).reshape(-1, 4)
    after = np.asarray(after, dtype=np.uint8).reshape(-1, 4)
    if before.shape != after.shape:
        raise ValueError(f"frames differ in size: {before.shape} and {after.shape}")
    return int(np.any(before != after, axis=1).sum())


def _guarded(name: str, check: Callable[[], CheckResult]) -> CheckResult:
    try:
        return check()
    except Exception:
        return CheckResult(name, False, traceback.format_exc())


def run_smoke(
    report_path: Path, *, include_rendering: bool = True, mesa_dir: Path | None = None
) -> int:
    results: list[CheckResult] = []
    mesa = mesa_dir if mesa_dir is not None else default_mesa_dir()
    render = include_rendering and mesa is not None
    if render:
        # Before QApplication and before OpenGL.GL is imported (spec 2.4.1).
        enable_compatibility_rendering(mesa)

    from PySide6.QtWidgets import QApplication

    # Held until the checks finish; the `del` below makes the lifetime explicit.
    app = QApplication.instance() or QApplication([sys.argv[0]])

    with tempfile.TemporaryDirectory() as scratch:
        checks: list[tuple[str, Callable[[], CheckResult]]] = [
            ("imports", check_imports),
            ("resources", check_resources),
            ("document", lambda: check_document_roundtrip(Path(scratch))),
        ]
        results.extend(_guarded(name, check) for name, check in checks)
    if include_rendering:
        if render:
            results.append(_guarded("rendering", check_rendering))
        else:
            results.append(CheckResult("rendering", False, "Mesa not found"))

    ok = all(result.ok for result in results)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"ok": ok, "checks": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    del app
    return 0 if ok else 1
