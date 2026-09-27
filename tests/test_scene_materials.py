from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene


def _two_face_scene():
    """Two triangular faces sharing an edge: fa (lower id) then fb."""
    s = Scene()
    a = [
        s.add_vertex(np.array([0.0, 0.0, 0.0])),
        s.add_vertex(np.array([1.0, 0.0, 0.0])),
        s.add_vertex(np.array([0.0, 1.0, 0.0])),
    ]
    fa = s.add_face_from_loop(a)
    b = [a[1], s.add_vertex(np.array([1.0, 1.0, 0.0])), a[2]]
    fb = s.add_face_from_loop(b)
    return s, fa, fb


def test_face_material_defaults_to_zero():
    s, fa, fb = _two_face_scene()
    assert s.face_material(fa) == 0
    assert s.face_material(fb) == 0


def test_set_and_get_face_material():
    s, fa, fb = _two_face_scene()
    s.set_face_material(fa, 3)
    assert s.face_material(fa) == 3
    assert s.face_material(fb) == 0


def test_set_default_id_clears():
    s, fa, _ = _two_face_scene()
    s.set_face_material(fa, 3)
    s.set_face_material(fa, 0)
    assert s.face_material(fa) == 0


def test_clear_face_material():
    s, fa, _ = _two_face_scene()
    s.set_face_material(fa, 5)
    s.clear_face_material(fa)
    assert s.face_material(fa) == 0


def test_paint_marks_render_dirty_and_mark_clean_clears():
    s, fa, _ = _two_face_scene()
    s.mark_clean()
    assert s.dirty is False
    s.set_face_material(fa, 2)
    assert s.dirty is True
    s.mark_clean()
    assert s.dirty is False


def test_face_triangle_materials_aligns_one_to_one_with_buffer():
    s, _fa, fb = _two_face_scene()
    s.set_face_material(fb, 7)
    positions, _ = s.face_triangle_buffer()
    tri_mats = s.face_triangle_materials()
    assert tri_mats.shape[0] * 3 == positions.shape[0]
    assert tri_mats.tolist() == [0, 7]   # fa(default) then fb(7), ascending id order


def test_clear_resets_face_materials():
    s, fa, fb = _two_face_scene()
    s.set_face_material(fa, 3)
    s.set_face_material(fb, 7)
    s.clear()
    # sidecar is empty after clear; new faces (which reuse ids from 0) are unpainted
    assert s.face_material(fa) == 0
    assert s.face_material(fb) == 0
    assert s.face_triangle_materials().tolist() == []
