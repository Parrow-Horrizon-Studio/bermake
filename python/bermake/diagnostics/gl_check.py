"""Is this OpenGL context good enough for Bermake? (M7.9, spec 2.4)

Every shader is `#version 330 core`, so 3.3 is the floor. Pure, so the whole
decision is testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_GL_VERSION = (3, 3)


@dataclass(frozen=True)
class GlInfo:
    version: tuple[int, int]
    version_string: str
    renderer: str


@dataclass(frozen=True)
class GlVerdict:
    ok: bool
    message: str = ""


def evaluate_gl(info: GlInfo, shader_error: str | None) -> GlVerdict:
    need = f"{MIN_GL_VERSION[0]}.{MIN_GL_VERSION[1]}"
    if info.version < MIN_GL_VERSION:
        got = f"{info.version[0]}.{info.version[1]}"
        return GlVerdict(
            False,
            f"Bermake needs OpenGL {need} or newer, but this computer's graphics driver "
            f"provides OpenGL {got} ({info.renderer}). This is common in virtual machines, "
            "over Remote Desktop, and with older graphics drivers.",
        )
    if shader_error is not None:
        lines = shader_error.strip().splitlines()
        first = lines[0] if lines else "unknown error"
        return GlVerdict(
            False,
            f"This graphics driver ({info.renderer}) could not compile Bermake's "
            f"viewport shaders: {first}",
        )
    return GlVerdict(True)
