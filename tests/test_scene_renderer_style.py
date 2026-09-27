from __future__ import annotations

from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.viewport.scene_renderer import SceneRenderer


def test_renderer_defaults_to_shaded_no_xray():
    r = SceneRenderer()
    assert r._render_style == RenderStyle()
    assert r._render_style.face_style is FaceStyle.SHADED
    assert r._render_style.xray is False


def test_set_render_style_stores_value():
    r = SceneRenderer()
    r.set_render_style(RenderStyle(FaceStyle.WIREFRAME, xray=True))
    assert r._render_style.face_style is FaceStyle.WIREFRAME
    assert r._render_style.xray is True


def test_set_render_style_stores_a_decoupled_copy():
    r = SceneRenderer()
    original = RenderStyle(FaceStyle.SHADED, xray=False)
    r.set_render_style(original)
    # Mutating the caller's object must NOT change the renderer's stored snapshot.
    original.face_style = FaceStyle.WIREFRAME
    original.xray = True
    assert r._render_style.face_style is FaceStyle.SHADED
    assert r._render_style.xray is False
