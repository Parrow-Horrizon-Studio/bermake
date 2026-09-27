from __future__ import annotations

import pytest
from bermake.viewport.render_style import (
    FACE_STYLE_TABLE,
    MONO_COLOR,
    XRAY_ALPHA,
    FaceShading,
    FaceStyle,
    FaceUniforms,
    PhongMaterial,
    RenderStyle,
    face_uniforms,
    resolve_face_pass,
)


def test_render_style_defaults_to_shaded_no_xray():
    rs = RenderStyle()
    assert rs.face_style is FaceStyle.SHADED
    assert rs.xray is False


def test_face_style_table_covers_all_styles():
    assert set(FACE_STYLE_TABLE) == set(FaceStyle)


def test_face_style_table_draw_faces_and_shading():
    assert FACE_STYLE_TABLE[FaceStyle.WIREFRAME].draw_faces is False
    assert FACE_STYLE_TABLE[FaceStyle.WIREFRAME].shading is None
    assert FACE_STYLE_TABLE[FaceStyle.HIDDEN_LINE].shading is FaceShading.FLAT_BG
    assert FACE_STYLE_TABLE[FaceStyle.MONOCHROME].shading is FaceShading.UNIFORM
    assert FACE_STYLE_TABLE[FaceStyle.SHADED].shading is FaceShading.LIT
    for style in (FaceStyle.HIDDEN_LINE, FaceStyle.MONOCHROME, FaceStyle.SHADED):
        assert FACE_STYLE_TABLE[style].draw_faces is True


_BG = (0.15, 0.15, 0.18)
_MAT = PhongMaterial(
    ambient=(0.10, 0.10, 0.11),
    diffuse=(0.65, 0.65, 0.70),
    specular=(0.10, 0.10, 0.10),
    shininess=16.0,
)


def test_face_uniforms_lit_uses_material_opaque():
    fu = face_uniforms(FaceShading.LIT, bg=_BG, material=_MAT, xray=False)
    assert isinstance(fu, FaceUniforms)
    assert fu.diffuse == _MAT.diffuse
    assert fu.ambient == _MAT.ambient
    assert fu.specular == _MAT.specular
    assert fu.shininess == _MAT.shininess
    assert fu.alpha == 1.0


def test_face_uniforms_uniform_uses_mono_diffuse():
    fu = face_uniforms(FaceShading.UNIFORM, bg=_BG, material=_MAT, xray=False)
    assert fu.diffuse == MONO_COLOR
    assert fu.ambient == _MAT.ambient        # keeps material ambient → still "lit"
    assert fu.specular == _MAT.specular
    assert fu.alpha == 1.0


def test_face_uniforms_flat_bg_is_unlit_background_fill():
    fu = face_uniforms(FaceShading.FLAT_BG, bg=_BG, material=_MAT, xray=False)
    assert fu.ambient == _BG                 # output == background (unlit)
    assert fu.diffuse == (0.0, 0.0, 0.0)
    assert fu.specular == (0.0, 0.0, 0.0)
    assert fu.shininess == _MAT.shininess
    assert fu.alpha == 1.0


def test_face_uniforms_xray_lowers_alpha_for_every_shading():
    for shading in (FaceShading.LIT, FaceShading.UNIFORM, FaceShading.FLAT_BG):
        fu = face_uniforms(shading, bg=_BG, material=_MAT, xray=True)
        assert fu.alpha == XRAY_ALPHA


_DIM_AMBIENT = (0.30, 0.30, 0.31)
_DIM_DIFFUSE = (0.40, 0.40, 0.42)
_DIM_ALPHA = 0.35
_DIM_ALPHA_FLOOR = 0.25


def _resolve(style, dimmed=False, material_alpha=1.0):
    return resolve_face_pass(
        style, dimmed=dimmed, bg=_BG, material=_MAT,
        dim_ambient=_DIM_AMBIENT, dim_diffuse=_DIM_DIFFUSE, dim_alpha=_DIM_ALPHA,
        dim_alpha_floor=_DIM_ALPHA_FLOOR, material_alpha=material_alpha,
    )


def test_resolve_shaded_default_is_opaque_depth_writing():
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=False))
    assert rp.draw_faces is True
    assert rp.diffuse == _MAT.diffuse
    assert rp.alpha == 1.0
    assert rp.blend is False
    assert rp.depth_write is True


def test_resolve_wireframe_skips_face_pass():
    rp = _resolve(RenderStyle(FaceStyle.WIREFRAME))
    assert rp.draw_faces is False


def test_resolve_xray_enables_blend_and_disables_depth_write():
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=True))
    assert rp.alpha == XRAY_ALPHA
    assert rp.blend is True
    assert rp.depth_write is False


def test_resolve_dim_only_matches_legacy_035_alpha():
    # dim (non-active context), opaque material, X-Ray off: table row 1.
    # max(1.0 * 0.35, 0.25) == 0.35 -- above the floor, so this reproduces the
    # M4e dim look untouched: dim colors + 0.35 alpha + blend, depth writes
    # preserved. Also a regression guard: a broken floor (e.g. one that
    # clamps upward instead of only raising a too-low value) would move this.
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=False), dimmed=True)
    assert rp.ambient == _DIM_AMBIENT
    assert rp.diffuse == _DIM_DIFFUSE
    assert rp.alpha == _DIM_ALPHA
    assert rp.blend is True
    assert rp.depth_write is True


def test_resolve_dim_and_xray_compose_through_the_floor():
    # Table row 3 (dimmed X-Ray). This test used to assert `alpha == 0.35`,
    # pinning the cap M7.5b (#107) shipped with (min(fu.alpha, dim_alpha)).
    # That cap fixed the reported defect (a translucent MATERIAL compounding
    # with the dim pass, 0.4 -> effective 0.14) but had a side effect: X-Ray's
    # alpha is exactly XRAY_ALPHA == dim_alpha == 0.35, so
    # min(0.35, 0.35) == 0.35, identical to undimmed X-Ray. Dimmed and
    # undimmed X-Ray geometry became distinguishable only by colour, losing
    # an alpha cue.
    #
    # The floored product restores it: max(0.35 * 0.35, 0.25) == 0.25, fainter
    # than plain X-Ray's 0.35. It also still isn't the pre-#107 defect,
    # because 0.25 is well above the 0.1225 a plain product would give here.
    #
    # Kills a plain product (0.35 * 0.35 == 0.1225 != 0.25) and a plain cap
    # (min(0.35, 0.35) == 0.35 != 0.25) -- see the red-probe outputs in the
    # v0.7.2 mapping-fix report for both failures captured live.
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=True), dimmed=True)
    assert rp.alpha == pytest.approx(0.25)
    assert rp.alpha < XRAY_ALPHA  # dimmed X-Ray is fainter than plain X-Ray again
    assert rp.alpha > XRAY_ALPHA * _DIM_ALPHA  # not back to the pre-#107 vanishing product
    assert rp.ambient == _DIM_AMBIENT  # the dim cue still lands
    assert rp.depth_write is False


def test_resolve_dim_translucent_material_floors_rather_than_vanishes():
    # Table row 2: a 0.4 translucent material, dimmed, X-Ray off -- the exact
    # scenario #107 reported. max(0.4 * 0.35, 0.25) == 0.25.
    #
    # Kills a plain product (0.4 * 0.35 == 0.14, the near-invisible defect
    # #107 fixed) and kills a plain cap (min(0.4, 0.35) == 0.35, which would
    # make a 0.4 material read as MORE opaque once dimmed than it started).
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=False), dimmed=True, material_alpha=0.4)
    assert rp.alpha == pytest.approx(0.25)
    assert rp.alpha < _DIM_ALPHA  # still fainter than a dimmed opaque face


def test_resolve_dim_deeply_translucent_material_is_floored():
    # Table row 4: a 0.1 material, dimmed, X-Ray off. max(0.1*0.35, 0.25) ==
    # 0.25 -- the floor does its job of preventing a near-vanish (the raw
    # product would be 0.035).
    #
    # Kills a plain product (0.1 * 0.35 == 0.035, effectively invisible) and
    # kills a plain cap (min(0.1, 0.35) == 0.1, still nearly invisible since
    # the cap never raises an already-low alpha).
    rp = _resolve(RenderStyle(FaceStyle.SHADED, xray=False), dimmed=True, material_alpha=0.1)
    assert rp.alpha == pytest.approx(0.25)
