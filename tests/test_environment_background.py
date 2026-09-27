"""The background reaching both of its sinks: glClearColor and Hidden Line's fill."""

import pytest
from bermake.model.material import MaterialLibrary
from bermake.viewport.environment import PLAIN_WHITE, STUDIO
from bermake.viewport.face_batches import FaceBatch
from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.viewport.scene_renderer import resolve_batch_sides


def test_hidden_line_on_a_white_environment_fills_faces_white():
    """The site a background change is most likely to miss.

    Hidden Line resolves to FLAT_BG: faces are filled with the BACKGROUND
    colour, not with a material colour. A change that reaches glClearColor but
    not resolve_batch_sides passes every other test in this milestone and still
    draws dark grey faces on a white page.

    Note WHICH term carries it: FLAT_BG is an unlit fill, so resolve_face_pass
    puts the background in AMBIENT and zeroes diffuse and specular
    (render_style.py, the FaceShading.FLAT_BG branch). Asserting on diffuse here
    would pass for any background at all, since diffuse is always (0, 0, 0).

    Discriminates: revert the `bg=` argument at the resolve_face_pass call to the
    old _BG_COLOR[:3] and this fails, returning Studio's (0.15, 0.15, 0.18).
    """
    lib = MaterialLibrary()
    batch = FaceBatch(front_material_id=0, back_material_id=0, first=0, count=3)
    front, _ = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.HIDDEN_LINE),
        dimmed=False,
        translucent_ids=frozenset(),
        bg=PLAIN_WHITE.background,
    )
    assert front.ambient == pytest.approx((1.0, 1.0, 1.0))
    assert front.diffuse == pytest.approx((0.0, 0.0, 0.0))


def test_hidden_line_follows_the_studio_background_too():
    """The same call with the dark environment, so the test above is not vacuous.

    Two backgrounds through one code path is the whole point: a hardcoded white
    would satisfy the test above on its own.
    """
    lib = MaterialLibrary()
    batch = FaceBatch(front_material_id=0, back_material_id=0, first=0, count=3)
    front, _ = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.HIDDEN_LINE),
        dimmed=False,
        translucent_ids=frozenset(),
        bg=STUDIO.background,
    )
    assert front.ambient == pytest.approx((0.15, 0.15, 0.18))


def test_bg_is_required_not_defaulted():
    """resolve_face_pass already requires bg; this restores the symmetry.

    A caller that omitted it would silently get the dark Hidden Line fill and no
    error, which is the failure mode the function's own docstring argues against
    for translucent_ids.
    """
    lib = MaterialLibrary()
    batch = FaceBatch(front_material_id=0, back_material_id=0, first=0, count=3)
    with pytest.raises(TypeError):
        resolve_batch_sides(batch, lib, RenderStyle(), dimmed=False, translucent_ids=frozenset())
