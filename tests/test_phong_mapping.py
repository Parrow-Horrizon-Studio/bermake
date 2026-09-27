"""M7.5a Task 2: PBR parameters mapped onto the Blinn-Phong uniforms."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from bermake.viewport.render_style import (
    _DIELECTRIC_F0,
    _MAX_SHININESS,
    _MIN_SHININESS,
    BACK_DEFAULT_COLOR,
    FaceShading,
    FaceStyle,
    RenderStyle,
    face_uniforms,
    phong_material_for,
    resolve_face_pass,
)

RED = (0.8, 0.1, 0.1)


def test_a_dielectric_keeps_its_full_diffuse():
    m = phong_material_for(RED, metallic=0.0)
    assert m.diffuse == pytest.approx(RED)


def test_a_metal_has_no_diffuse_and_takes_its_specular_from_base_color():
    m = phong_material_for(RED, metallic=1.0)
    assert m.diffuse == pytest.approx((0.0, 0.0, 0.0))
    assert m.specular == pytest.approx(RED)


def test_a_dielectric_specular_is_the_standard_f0_not_the_base_color():
    m = phong_material_for(RED, metallic=0.0)
    assert m.specular == pytest.approx((0.04, 0.04, 0.04))
    # and is emphatically not the base color, which a naive mapping would use
    assert m.specular != pytest.approx(RED)


def test_metallic_interpolates_between_the_two():
    half = phong_material_for(RED, metallic=0.5)
    assert half.diffuse == pytest.approx(tuple(c * 0.5 for c in RED))
    assert half.specular == pytest.approx(tuple(0.04 * 0.5 + c * 0.5 for c in RED))


def test_lower_roughness_gives_a_tighter_highlight():
    smooth = phong_material_for(RED, roughness=0.1)
    rough = phong_material_for(RED, roughness=0.9)
    assert smooth.shininess > rough.shininess


def test_shininess_stays_inside_its_declared_bounds():
    # Renamed in M7.5b (#107): the values did not move, but the reason did.
    # There is no clamp any more. 256 ** (1 - roughness) lands inside
    # [_MIN_SHININESS, _MAX_SHININESS] by construction, so what this now
    # guards is that the curve's own range still matches the constants the
    # rest of the codebase reasons about (see test_phong_material_for.py).
    # The dead zone the old clamp created is covered by
    # test_shininess_spans_the_slider_with_no_dead_zone below.
    assert phong_material_for(RED, roughness=0.0001).shininess <= 256.0
    assert phong_material_for(RED, roughness=1.0).shininess >= 1.0


def test_ambient_tracks_base_color_and_no_longer_fades_with_metallic():
    # The expectation moved in M7.5b (#107), and it is the whole point of the
    # fix, not a number nudged to make a suite green.
    #
    # This test used to assert `metal.ambient == (0, 0, 0)`, pinning
    # `ambient = base * factor * (1 - metallic)`. That formula is right for
    # real PBR -- a metal has no diffuse lobe, and its appearance comes from
    # the environment it reflects -- but this renderer has one directional
    # light and no environment term, so there was nothing left to carry the
    # metal's colour and metallic 1.0 rendered pure black off the highlight.
    # Ambient is the only stand-in for environment light available here, and a
    # metal's colour IS its reflectance, so the metal keeps its ambient.
    #
    # Dielectrics are untouched: the factor that was dropped is 1.0 at
    # metallic 0, so `dielectric.ambient` is bit-identical to v0.7.1's.
    dielectric = phong_material_for(RED, metallic=0.0)
    metal = phong_material_for(RED, metallic=1.0)
    assert dielectric.ambient == pytest.approx(tuple(c * 0.55 for c in RED))
    assert metal.ambient == pytest.approx(dielectric.ambient)


def test_material_alpha_multiplies_into_the_resolved_alpha():
    m = phong_material_for(RED)
    fu = face_uniforms(
        FaceShading.LIT, bg=(0.1, 0.1, 0.1), material=m, xray=False, material_alpha=0.25
    )
    assert fu.alpha == pytest.approx(0.25)


def test_material_alpha_and_xray_compose():
    m = phong_material_for(RED)
    fu = face_uniforms(
        FaceShading.LIT, bg=(0.1, 0.1, 0.1), material=m, xray=True, material_alpha=0.5
    )
    # XRAY_ALPHA is 0.35; a translucent material under X-Ray is more transparent
    # than either alone, so the two multiply rather than one winning.
    assert fu.alpha == pytest.approx(0.35 * 0.5)


def test_a_translucent_material_blends_and_stops_writing_depth():
    m = phong_material_for(RED)
    r = resolve_face_pass(
        RenderStyle(face_style=FaceStyle.SHADED),
        dimmed=False,
        bg=(0.1, 0.1, 0.1),
        material=m,
        dim_ambient=(0.0, 0.0, 0.0),
        dim_diffuse=(0.0, 0.0, 0.0),
        dim_alpha=1.0,
        dim_alpha_floor=0.0,
        material_alpha=0.4,
    )
    assert r.blend is True
    assert r.depth_write is False


def test_an_opaque_material_still_writes_depth():
    m = phong_material_for(RED)
    r = resolve_face_pass(
        RenderStyle(face_style=FaceStyle.SHADED),
        dimmed=False,
        bg=(0.1, 0.1, 0.1),
        material=m,
        dim_ambient=(0.0, 0.0, 0.0),
        dim_diffuse=(0.0, 0.0, 0.0),
        dim_alpha=1.0,
        dim_alpha_floor=0.0,
        material_alpha=1.0,
    )
    assert r.blend is False
    assert r.depth_write is True


def test_the_back_default_is_distinct_from_the_front_default():
    from bermake.model.material import _DEFAULT_SWATCH_COLOR

    assert BACK_DEFAULT_COLOR != _DEFAULT_SWATCH_COLOR
    # and distinguishable, not a near-identical shade
    delta = sum(abs(a - b) for a, b in zip(BACK_DEFAULT_COLOR, _DEFAULT_SWATCH_COLOR))
    assert delta > 0.2


# --- M7.5b (#107): the extremes, asserted in RENDERED units ----------------
#
# The defect this section exists to kill shipped under ten passing tests,
# every one of which restated the formula it was testing. Terms alone cannot
# tell you whether a slider does anything: a dielectric's specular is pinned
# at _DIELECTRIC_F0 = 0.04 == 10/255, so roughness could reshape the highlight
# lobe as much as it liked and still never move a pixel by more than 10. The
# tests below therefore shade the material and assert on the result.
#
# _shade transcribes viewport/shaders/phong.frag's main(), and the two agree
# where it matters: sweeping roughness over the probe sphere gives a maximum
# spread of 74/255 through this model and 74/255 through a real offscreen GL
# render of the same scene (v0.7.2 report, section 4). The shader's own
# structure -- that every uniform exists, is set, and is read -- is already
# pinned by tests/test_two_sided_shading.py, so this transcription cannot
# silently drift away from a shader that changed shape.

_PROBE = (0.55, 0.35, 0.20)  # deliberately non-clipping: a delta is a real delta
_ROUGH_SWEEP = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_METAL_SWEEP = (0.0, 0.25, 0.5, 0.75, 1.0)


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _shade(material, normal, view_dir):
    """phong.frag main(), transcribed. Returns one RGB pixel in 0..255."""
    from bermake.viewport.scene_renderer import _LIGHT_COLOR, _LIGHT_DIR

    n, lv, v = _unit(normal), _unit(_LIGHT_DIR), _unit(view_dir)
    r = lv - 2.0 * np.dot(lv, n) * n
    diff = max(float(np.dot(n, -lv)), 0.0)
    spec = max(float(np.dot(r, v)), 0.0) ** material.shininess
    color = (
        np.array(material.ambient)
        + np.array(material.diffuse) * diff * np.array(_LIGHT_COLOR)
        + np.array(material.specular) * spec * np.array(_LIGHT_COLOR)
    )
    return np.round(np.clip(color, 0.0, 1.0) * 255.0)


def _rendered(material, samples=40):
    """Every visible pixel of a lit sphere, as an (N, 3) array in 0..255.

    A sphere rather than a quad because one flat face samples exactly one
    dot(N, -L): the highlight lobe, the terminator and the shadow side all
    have to be in frame for "what does this slider do" to be answerable.
    """
    cam = np.array([5.0, -5.5, 4.0])
    centre = np.array([0.0, 0.0, 1.4])
    normals = []
    for i in range(samples):
        z = -1.0 + 2.0 * (i + 0.5) / samples
        rr = float(np.sqrt(max(0.0, 1.0 - z * z)))
        for j in range(samples):
            th = 2.0 * np.pi * j / samples
            normals.append((rr * np.cos(th), rr * np.sin(th), z))
    normals = np.array(normals)
    view = cam - (centre + 1.2 * normals)
    visible = (view * normals).sum(axis=1) > 0.0
    return np.array(
        [_shade(material, n, v) for n, v in zip(normals[visible], view[visible], strict=True)]
    )


def test_a_full_metal_is_not_black():
    # THE reported symptom. Through the real renderer, v0.7.1's metallic 1.0
    # measured min-per-channel [0 0 0]: every fragment outside the highlight
    # lobe was literally black.
    #
    # Asserting on the DARKEST pixel, not the mean, is what kills it. A mean
    # over a sphere that still carries a specular highlight is comfortably
    # non-zero even when the entire body of the object is black, so a
    # mean-based assertion would pass against the exact bug.
    metal = _rendered(phong_material_for(_PROBE, metallic=1.0))
    darkest = metal.min(axis=0)
    assert (darkest >= 16).all(), f"metal bottoms out at {darkest}"
    # And it is the metal's OWN colour down there, not a grey floor bolted on:
    # the probe is orange, so the channels must stay ordered R > G > B.
    assert darkest[0] > darkest[1] > darkest[2]


def test_the_metallic_sweep_darkens_monotonically_without_reaching_black():
    # Metals SHOULD read darker than dielectrics under one light -- v0.7.1 was
    # right about the direction and wrong only about the floor. Kills a fix
    # that buys "not black" by flattening the slider's response, trading one
    # unusable extreme for a dead control.
    means = [_rendered(phong_material_for(_PROBE, metallic=m)).mean() for m in _METAL_SWEEP]
    assert means == sorted(means, reverse=True), means
    assert means[0] - means[-1] > 20, "metallic must still visibly do something"
    for m in _METAL_SWEEP:
        assert _rendered(phong_material_for(_PROBE, metallic=m)).min() > 0


def test_roughness_moves_more_than_the_whole_dielectric_specular_budget():
    # v0.7.1's roughness changed only the highlight LOBE WIDTH. A dielectric's
    # specular is _DIELECTRIC_F0 == 0.04 == 10/255, so 10/255 was a hard
    # ceiling on how far roughness could move any pixel no matter what the
    # lobe did -- and the reported measurement hit exactly 10, which was that
    # ceiling rather than a coincidence.
    #
    # Any fix that still only reshapes the lobe is capped at the same 10 and
    # fails here. The threshold is four times the ceiling and well under the
    # 74/255 actually measured, so it guards the class of defect without
    # pinning the tuning.
    frames = np.stack([_rendered(phong_material_for(_PROBE, roughness=r)) for r in _ROUGH_SWEEP])
    spread = frames.max(axis=0) - frames.min(axis=0)
    ceiling = round(_DIELECTRIC_F0 * 255)  # == 10, v0.7.1's measured maximum
    assert spread.max() > 4 * ceiling, f"roughness spread only {spread.max()}/255"


def test_roughness_redistributes_light_instead_of_dimming_the_object():
    # The cheap way to pass the previous test is to scale every term with
    # roughness, which buys a big pixel delta by turning the slider into a
    # brightness knob. Kills that: the MEAN over the probe must stay put while
    # individual pixels move a long way, and the movement must carry opposite
    # signs at the two ends of the tonal range.
    smooth = _rendered(phong_material_for(_PROBE, roughness=0.0))
    rough = _rendered(phong_material_for(_PROBE, roughness=1.0))
    delta = rough - smooth

    assert abs(delta.mean()) < 0.25 * np.abs(delta).max(), (
        f"mean moved {delta.mean():.1f} against a max of {np.abs(delta).max():.0f}"
    )
    luminance = smooth.sum(axis=1)
    darkest = luminance <= np.percentile(luminance, 25)
    brightest = luminance >= np.percentile(luminance, 75)
    assert delta[darkest].mean() > 10, "rough must lift the shadow side"
    assert delta[brightest].mean() < -10, "rough must pull the lit side down"


def test_the_default_dielectric_is_untouched_by_the_roughness_remap():
    # The whole-document regression guard. Roughness defaults to 0.5, so this
    # is what every unpainted back face and most painted faces resolve to; if
    # it moved, every existing document would change appearance on upgrade.
    # The remap is centred on 0.5 precisely so both scales are exactly 1.0
    # here. Kills a remap centred anywhere else -- on 0.0, say, which is the
    # natural thing to write and would restyle every model in existence.
    m = phong_material_for(RED, metallic=0.0, roughness=0.5)
    assert m.ambient == pytest.approx(tuple(c * 0.55 for c in RED))
    assert m.diffuse == pytest.approx(RED)
    assert m.specular == pytest.approx((_DIELECTRIC_F0,) * 3)


def test_shininess_spans_the_slider_with_no_dead_zone():
    # v0.7.1 used the spec's Blinn-Phong equivalence 2 / roughness**4 - 2,
    # which clamps to _MAX_SHININESS for EVERY roughness below 0.28: the first
    # 28% of the slider produced one identical value. `assert smooth > rough`
    # on two far-apart samples passes happily against that, which is how it
    # shipped. Sampling the whole slider and demanding STRICT monotonicity is
    # what catches a flat region anywhere in it.
    values = [phong_material_for(RED, roughness=i / 100.0).shininess for i in range(101)]
    assert all(a > b for a, b in pairwise(values)), "shininess has a flat region"
    assert values[0] == pytest.approx(_MAX_SHININESS)
    assert values[-1] == pytest.approx(_MIN_SHININESS)
    for roughness, expected in ((0.25, 64.0), (0.5, 16.0), (0.75, 4.0)):
        assert phong_material_for(RED, roughness=roughness).shininess == pytest.approx(expected)


# --- M7.5b (#107): the dim pass must not compound with translucency --------


def _dimmed(material_alpha, *, xray=False, dim_alpha=0.35, dim_alpha_floor=0.25):
    return resolve_face_pass(
        RenderStyle(face_style=FaceStyle.SHADED, xray=xray),
        dimmed=True,
        bg=(0.1, 0.1, 0.1),
        material=phong_material_for(RED),
        dim_ambient=(0.3, 0.3, 0.31),
        dim_diffuse=(0.4, 0.4, 0.42),
        dim_alpha=dim_alpha,
        dim_alpha_floor=dim_alpha_floor,
        material_alpha=material_alpha,
    )


def test_dimming_a_translucent_face_is_floored_not_vanished():
    # The reported symptom: `alpha = fu.alpha * dim_alpha` turned a 0.4
    # material into an effective 0.14 once its group stopped being the active
    # context -- very nearly invisible. Dimming says "not the thing you are
    # editing"; it is not licensed to delete the geometry.
    #
    # The fix is no longer "don't multiply" (that was the #107 cap, reverted
    # -- see test_xray_and_dim_compose_through_the_floor below for why); it is
    # "multiply, but floor the result": max(0.4 * 0.35, 0.25) == 0.25.
    assert _dimmed(0.4).alpha == pytest.approx(0.25)
    assert _dimmed(0.4).alpha > 0.4 * 0.35  # the plain product, named explicitly


def test_dimming_still_makes_an_opaque_face_recede():
    # An opaque face must still be pulled down to dim_alpha: max(1.0 * 0.35,
    # 0.25) == 0.35, above the floor, so this is unaffected by it.
    assert _dimmed(1.0).alpha == pytest.approx(0.35)
    assert _dimmed(1.0).blend is True


def test_dimming_a_barely_visible_face_is_floored_not_left_alone():
    # Renamed from test_dimming_never_makes_a_barely_visible_face_more_solid:
    # that name asserted the cap's behaviour (min(0.1, 0.35) == 0.1, alpha
    # untouched), which is no longer the rule. A deliberately ghostly 0.1
    # material dimmed now floors to 0.25 -- more opaque than 0.1, but still
    # well under a dimmed opaque face's 0.35, and nowhere near the plain
    # product's near-invisible 0.035.
    assert _dimmed(0.1).alpha == pytest.approx(0.25)
    assert _dimmed(0.1).alpha < 0.35  # still fainter than a dimmed opaque face


def test_xray_and_dim_compose_through_the_floor():
    # Renamed from test_xray_and_dim_still_compose_through_the_cap.
    # material_alpha=0.5 under X-Ray gives fu.alpha == 0.35 * 0.5 == 0.175,
    # already below dim_alpha (0.35) -- under the old cap this composed
    # untouched (min(0.175, 0.35) == 0.175), which is what that test name
    # described. Under the floored product it is pulled back up:
    # max(0.175 * 0.35, 0.25) == 0.25, since 0.175 * 0.35 == 0.06125 is below
    # the floor. The floor's job is exactly this: stop a doubly-attenuated
    # (translucent + X-Ray + dim) face from vanishing.
    assert _dimmed(0.5, xray=True).alpha == pytest.approx(0.25)
    assert _dimmed(0.5, xray=True).alpha > 0.35 * 0.5 * 0.35  # the plain product
