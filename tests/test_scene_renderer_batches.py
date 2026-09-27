from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene, Side
from bermake.viewport.face_batches import plan_face_batches


def _two_face_scene():
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


def test_unpainted_scene_yields_single_default_batch():
    s, _, _ = _two_face_scene()
    plan = plan_face_batches(s.face_triangle_materials(), s.face_triangle_materials(Side.BACK))
    assert len(plan.opaque) == 1
    assert plan.translucent == []
    assert plan.opaque[0].front_material_id == 0
    assert plan.opaque[0].back_material_id == 0
    assert plan.opaque[0].first == 0
    # identity reorder => byte-identical draw path
    assert plan.vertex_order.tolist() == list(range(plan.vertex_order.shape[0]))


def test_painted_scene_splits_into_per_material_batches():
    s, _fa, fb = _two_face_scene()
    s.set_face_material(fb, 7)
    plan = plan_face_batches(s.face_triangle_materials(), s.face_triangle_materials(Side.BACK))
    assert [b.front_material_id for b in plan.opaque] == [0, 7]
    assert sum(b.count for b in plan.opaque) == plan.vertex_order.shape[0]
