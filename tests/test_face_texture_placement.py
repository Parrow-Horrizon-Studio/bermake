"""M7.5b Task 3: independent front and back texture placement."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import DEFAULT_PLACEMENT, Scene, Side, TexturePlacement


def _square(scene, z=0.0):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, z], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_an_unadjusted_face_reports_the_identity_placement():
    s = Scene()
    f = _square(s)
    assert s.face_placement(f) == DEFAULT_PLACEMENT
    assert s.face_placement(f, Side.BACK) == DEFAULT_PLACEMENT


def test_placing_the_front_leaves_the_back_alone():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(offset_u=0.5), Side.FRONT)
    assert s.face_placement(f, Side.FRONT).offset_u == 0.5
    assert s.face_placement(f, Side.BACK) == DEFAULT_PLACEMENT


def test_placing_the_back_leaves_the_front_alone():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(rotation=1.25), Side.BACK)
    assert s.face_placement(f, Side.BACK).rotation == 1.25
    assert s.face_placement(f, Side.FRONT) == DEFAULT_PLACEMENT


def test_the_two_sides_hold_different_placements_at_once():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.FRONT)
    s.set_face_placement(f, TexturePlacement(scale=3.0), Side.BACK)
    assert (s.face_placement(f).scale, s.face_placement(f, Side.BACK).scale) == (2.0, 3.0)


def test_side_defaults_to_front():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(offset_v=0.25))
    assert s.face_placement(f, Side.FRONT).offset_v == 0.25
    assert s.face_placement(f, Side.BACK) == DEFAULT_PLACEMENT


def test_setting_the_identity_clears_the_entry_rather_than_storing_it():
    # The sidecars hold only ADJUSTED faces, mirroring how painting Default
    # clears a material entry. Task 7 serializes these dicts directly, so a
    # stored identity would write noise for every face the user ever touched
    # and then reset.
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(scale=2.0))
    assert s.faces_with_placement() == [(f, Side.FRONT)]
    s.set_face_placement(f, DEFAULT_PLACEMENT)
    assert s.faces_with_placement() == []


def test_clear_face_placement_removes_only_the_named_side():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.FRONT)
    s.set_face_placement(f, TexturePlacement(scale=3.0), Side.BACK)
    s.clear_face_placement(f, Side.FRONT)
    assert s.face_placement(f, Side.FRONT) == DEFAULT_PLACEMENT
    assert s.face_placement(f, Side.BACK).scale == 3.0


def test_faces_with_placement_finds_both_sides():
    s = Scene()
    a = _square(s)
    b = _square(s, z=1.0)
    s.set_face_placement(a, TexturePlacement(scale=2.0), Side.FRONT)
    s.set_face_placement(b, TexturePlacement(scale=2.0), Side.BACK)
    assert sorted(s.faces_with_placement()) == sorted([(a, Side.FRONT), (b, Side.BACK)])


def test_faces_with_placement_reports_one_face_adjusted_on_both_sides():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.FRONT)
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.BACK)
    # set(), not sorted(): both pairs share face id f, and Side (a plain Enum,
    # not IntEnum/comparable) has no ordering, so sorted() raises TypeError
    # even on the literal expected list.
    assert set(s.faces_with_placement()) == {(f, Side.FRONT), (f, Side.BACK)}


def test_clear_resets_both_sides():
    # Deliberately checks the ORIGINAL face id through the public probe rather
    # than a newly built face. HalfEdgeMesh.clear() does not reset the face-id
    # counter, so a new face is never a key in either dict and an assertion
    # about it passes even when clear() emptied nothing. M7.5a shipped exactly
    # that test and it could not fail.
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.FRONT)
    s.set_face_placement(f, TexturePlacement(scale=3.0), Side.BACK)
    s.clear()
    assert s.faces_with_placement() == []


def test_placing_marks_the_scene_render_dirty():
    # UVs are baked into the vertex buffer in Task 4, so a placement change has
    # to reach the renderer as a re-upload.
    s = Scene()
    f = _square(s)
    s.mark_clean()
    s.set_face_placement(f, TexturePlacement(scale=2.0), Side.BACK)
    assert s.dirty is True
