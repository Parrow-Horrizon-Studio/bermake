from __future__ import annotations

from bermake.viewport.render_style import (
    _DIELECTRIC_F0,
    _MAX_SHININESS,
    _MIN_SHININESS,
)


def test_dielectric_default_stays_consistent_with_the_renderer_hand_tuned_default():
    """M7.5a Task 2 correction 6 guard, rewritten.

    Before this task, `_DEFAULT_SPECULAR`/`_DEFAULT_SHININESS` were fixed
    constants in render_style.py that literally equaled scene_renderer's
    `_MATERIAL_SPECULAR`/`_MATERIAL_SHININESS`, and this test pinned that
    exact equality so the two files could not drift apart silently.

    After this task, `phong_material_for`'s specular/shininess are *derived*
    from metallic/roughness instead of being fixed constants, so that
    equality is gone by construction: a dielectric (metallic=0.0) at the
    function's default roughness (0.5) produces specular
    `(_DIELECTRIC_F0,) * 3` == (0.04, 0.04, 0.04), which does not match the
    renderer's hand-tuned `_MATERIAL_SPECULAR` (0.10, 0.10, 0.10). That
    divergence is intentional: M7.5a Task 6 keeps
    scene_renderer._DEFAULT_MATERIAL hand-tuned and untouched for unpainted
    front faces, while painted materials go through the derivation.

    The derived shininess at that same point was 30.0 when this docstring was
    written, under `2 / roughness**4 - 2`. M7.5b (#107) replaced that curve
    with `_MAX_SHININESS ** (1 - roughness)` -- the old one clamped flat for
    every roughness below 0.28 -- and the new one happens to give exactly
    16.0 at roughness 0.5, which IS `_MATERIAL_SHININESS`. That coincidence
    is worth naming so nobody reads it as the constants having been re-tied
    together: nothing enforces it, the two still come from different places,
    and the assertion below is deliberately a range check rather than an
    equality so it keeps holding either way.

    What still has to hold, and is worth guarding: the dielectric F0 this
    module uses is a physically conservative floor, strictly below the
    renderer's hand-tuned specular brightness, and the renderer's hand-tuned
    shininess still falls inside the range our roughness-driven clamp can
    produce. If either constant moves out of that relationship -- e.g. F0
    creeping above a real dielectric's reflectance, or the renderer's
    shininess drifting outside what the clamp permits -- the assumption that
    "painted dielectric faces and the hand-tuned default look like the same
    family of material" has quietly broken, and this test catches it.
    """
    from bermake.viewport.scene_renderer import _MATERIAL_SHININESS, _MATERIAL_SPECULAR

    assert _DIELECTRIC_F0 < _MATERIAL_SPECULAR[0]
    assert _MIN_SHININESS <= _MATERIAL_SHININESS <= _MAX_SHININESS
