from __future__ import annotations

from bermake.viewport.scene_renderer import _PHONG_UNIFORMS, _load_shader_source


def test_phong_uniforms_tuple_includes_u_alpha():
    assert "u_alpha" in _PHONG_UNIFORMS


def test_phong_fragment_declares_and_uses_u_alpha():
    src = _load_shader_source("phong.frag")
    assert "uniform float u_alpha;" in src
    assert "u_alpha" in src.split("frag_color")[1]  # used in the output statement
