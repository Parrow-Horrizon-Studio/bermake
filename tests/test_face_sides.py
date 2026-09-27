"""M7.5a Task 3: independent front and back material slots."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene, Side


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


def test_painting_the_front_leaves_the_back_untouched():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    assert s.face_material(f, Side.FRONT) == 3
    assert s.face_material(f, Side.BACK) == 0   # the discriminating half


def test_painting_the_back_leaves_the_front_untouched():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 5, Side.BACK)
    assert s.face_material(f, Side.BACK) == 5
    assert s.face_material(f, Side.FRONT) == 0


def test_the_two_sides_hold_different_materials_at_once():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    s.set_face_material(f, 5, Side.BACK)
    assert (s.face_material(f, Side.FRONT), s.face_material(f, Side.BACK)) == (3, 5)


def test_side_defaults_to_front_for_every_accessor():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3)
    assert s.face_material(f) == 3
    assert s.face_material(f, Side.BACK) == 0
    s.clear_face_material(f)
    assert s.face_material(f) == 0


def test_painting_default_clears_only_the_named_side():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    s.set_face_material(f, 5, Side.BACK)
    s.set_face_material(f, 0, Side.FRONT)      # Default clears
    assert s.face_material(f, Side.FRONT) == 0
    assert s.face_material(f, Side.BACK) == 5  # the back survives


def test_face_triangle_materials_reports_per_side():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    s.set_face_material(f, 5, Side.BACK)
    front = s.face_triangle_materials(Side.FRONT)
    back = s.face_triangle_materials(Side.BACK)
    assert front.shape == back.shape
    assert set(front.tolist()) == {3}
    assert set(back.tolist()) == {5}


def test_faces_with_material_finds_both_sides():
    s = Scene()
    a = _square(s, z=0.0)
    b = _square(s, z=1.0)
    s.set_face_material(a, 7, Side.FRONT)
    s.set_face_material(b, 7, Side.BACK)
    s.set_face_material(b, 9, Side.FRONT)
    assert sorted(s.faces_with_material(7)) == sorted([(a, Side.FRONT), (b, Side.BACK)])


def test_faces_with_material_is_empty_for_an_unused_material():
    s = Scene()
    _square(s)
    assert s.faces_with_material(42) == []


def test_clear_resets_both_sides():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    s.set_face_material(f, 5, Side.BACK)
    s.clear()
    assert s.faces_with_material(3) == []
    assert s.faces_with_material(5) == []


def test_faces_with_material_finds_both_sides_on_the_same_face():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 7, Side.FRONT)
    s.set_face_material(f, 7, Side.BACK)
    assert set(s.faces_with_material(7)) == {(f, Side.FRONT), (f, Side.BACK)}


def test_painting_the_back_marks_the_scene_render_dirty():
    s = Scene()
    f = _square(s)
    s.mark_clean()
    s.set_face_material(f, 5, Side.BACK)
    assert s.dirty is True
