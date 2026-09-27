from __future__ import annotations

from bermake.viewport.face_batches import FaceBatch, plan_face_batches


def test_empty_returns_no_batches():
    plan = plan_face_batches([], [])
    assert plan.vertex_order.tolist() == []
    assert plan.opaque == []
    assert plan.translucent == []


def test_single_material_one_batch_identity_order():
    plan = plan_face_batches([0, 0, 0], [0, 0, 0])  # 3 triangles, all Default
    assert plan.vertex_order.tolist() == list(range(9))  # identity over 9 vertices
    assert plan.opaque == [FaceBatch(front_material_id=0, back_material_id=0, first=0, count=9)]


def test_default_only_collapses_to_one_default_batch():
    plan = plan_face_batches([0, 0, 0, 0], [0, 0, 0, 0])
    assert plan.opaque == [FaceBatch(front_material_id=0, back_material_id=0, first=0, count=12)]
    assert plan.vertex_order.tolist() == list(range(12))  # identity -> byte-identical path


def test_interleaved_materials_grouped_and_contiguous():
    # triangles: mat 2, mat 0, mat 2, mat 0  -> grouped 0,0 then 2,2
    plan = plan_face_batches([2, 0, 2, 0], [0, 0, 0, 0])
    assert plan.opaque == [
        FaceBatch(front_material_id=0, back_material_id=0, first=0, count=6),
        FaceBatch(front_material_id=2, back_material_id=0, first=6, count=6),
    ]
    # mat-0 tris are originals 1 and 3 (verts 3,4,5 and 9,10,11), then mat-2.
    assert plan.vertex_order.tolist() == [3, 4, 5, 9, 10, 11, 0, 1, 2, 6, 7, 8]


def test_vertex_order_is_a_valid_permutation():
    plan = plan_face_batches([5, 1, 5, 1, 9], [0, 0, 0, 0, 0])
    assert sorted(plan.vertex_order.tolist()) == list(range(15))
