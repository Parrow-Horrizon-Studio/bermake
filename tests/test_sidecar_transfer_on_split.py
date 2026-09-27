"""Every face-keyed sidecar survives the destroy-and-rebuild inside split_edge.

Before M7.5c these all failed: HalfEdgeMesh::split_edge removes both incident
faces and rebuilds them with new ids, and nothing carried the sidecars across.
"""

import numpy as np

from bermake.commands.scene_commands import SplitEdgeCommand
from bermake.scene.scene import Scene, Side, TexturePlacement


def _painted_quad(scene, material_id=7):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    f = scene.add_face_from_loop(ids)
    scene.set_face_material(f, material_id, Side.FRONT)
    return f, ids


def test_material_survives_a_split():
    s = Scene()
    f, ids = _painted_quad(s)
    e = s.edge_between(ids[0], ids[1])
    res = s.split_edge(e, 0.25)
    assert res is not None and res.face_a is not None
    assert s.face_material(res.face_a, Side.FRONT) == 7


def test_back_material_survives_a_split():
    s = Scene()
    f, ids = _painted_quad(s)
    s.set_face_material(f, 3, Side.BACK)
    e = s.edge_between(ids[0], ids[1])
    res = s.split_edge(e, 0.25)
    assert s.face_material(res.face_a, Side.BACK) == 3


def test_placement_survives_a_split():
    s = Scene()
    f, ids = _painted_quad(s)
    s.set_face_placement(f, TexturePlacement(0.25, 0.5, 2.0, 0.75), Side.FRONT)
    e = s.edge_between(ids[0], ids[1])
    res = s.split_edge(e, 0.25)
    got = s.face_placement(res.face_a, Side.FRONT)
    assert (got.offset_u, got.offset_v, got.scale, got.rotation) == (0.25, 0.5, 2.0, 0.75)


def test_stored_uvs_gain_one_lerped_corner():
    s = Scene()
    f, ids = _painted_quad(s)
    # UVs parallel to the loop, which is ids[0..3] in order.
    s.set_face_uvs(f, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Side.FRONT)
    e = s.edge_between(ids[0], ids[1])
    res = s.split_edge(e, 0.25)

    new_loop = s.face_loop(res.face_a)
    uvs = s.face_uvs(res.face_a, Side.FRONT)
    assert uvs is not None
    assert uvs.shape == (5, 2)
    inserted_at = new_loop.index(res.vertex)
    # 0.25 from (0,0) toward (1,0). NOT 0.5: a midpoint implementation and a
    # copy-the-neighbour implementation both pass a t=0.5 test.
    np.testing.assert_allclose(uvs[inserted_at], [0.25, 0.0], atol=1e-6)


def test_the_old_faces_entries_are_left_in_place_for_undo():
    s = Scene()
    f, ids = _painted_quad(s)
    e = s.edge_between(ids[0], ids[1])
    s.split_edge(e, 0.25)
    # The dead face keeps its entry; ids are never reused so this is safe, and
    # restore_face on undo relies on it.
    assert s._face_materials_front[f] == 7


def test_undo_then_redo_keeps_the_material_on_the_live_face():
    s = Scene()
    f, ids = _painted_quad(s)
    e = s.edge_between(ids[0], ids[1])
    cmd = SplitEdgeCommand(e, 0.25)

    cmd.do(s)
    assert any(s.face_material(fc.id, Side.FRONT) == 7 for fc in s.faces_iter())

    cmd.undo(s)
    assert s.face_material(f, Side.FRONT) == 7

    cmd.do(s)
    assert any(s.face_material(fc.id, Side.FRONT) == 7 for fc in s.faces_iter())


def test_a_boundary_edge_split_transfers_only_the_one_real_face():
    s = Scene()
    f, ids = _painted_quad(s)
    e = s.edge_between(ids[0], ids[1])
    res = s.split_edge(e, 0.25)
    # A lone quad's edges are all boundary, so one side has no face at all.
    assert res.face_b is None
    assert s.face_material(res.face_a, Side.FRONT) == 7


def test_an_interior_edge_split_transfers_both_faces_independently():
    # Two quads sharing a vertical edge, both coplanar with normal +Z:
    #
    #   D(0,1)---C(1,1)---G(2,1)
    #    |    f1   |   f2   |
    #   A(0,0)---B(1,0)---E(2,0)
    #
    # f1's loop is A,B,C,D so it traverses the shared edge B->C. f2's loop is
    # B,E,G,C so it traverses the shared edge as ...,C,(back to)B, i.e. C->B:
    # the opposite direction. A manifold half-edge mesh requires this (the
    # two half-edges along a shared edge always run opposite ways), and it is
    # exactly the condition transfer_uvs_across_split's t vs 1-t branch
    # exists for.
    s = Scene()
    a = s.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    b = s.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    c = s.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    d = s.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    e_vert = s.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    g = s.add_vertex(np.array([2, 1, 0], dtype=np.float32))

    f1 = s.add_face_from_loop([a, b, c, d])
    f2 = s.add_face_from_loop([b, e_vert, g, c])

    # Confirm the shared edge is genuinely traversed in opposite directions
    # by the two faces before this split, which is the precondition for the
    # branch under test.
    assert list(s.face_loop(f1)) == [a, b, c, d]
    assert list(s.face_loop(f2)) == [b, e_vert, g, c]

    s.set_face_material(f1, 7, Side.FRONT)
    s.set_face_material(f2, 11, Side.FRONT)
    # Distinct UV layouts per face so a crossed pairing or a t/1-t mix-up
    # produces a visibly wrong number rather than passing by symmetry.
    s.set_face_uvs(f1, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Side.FRONT)
    s.set_face_uvs(f2, [(10.0, 0.0), (12.0, 0.0), (12.0, 1.0), (10.0, 1.0)], Side.FRONT)

    shared_edge = s.edge_between(b, c)
    edge = s.edge(shared_edge)
    assert (edge.v1_id, edge.v2_id) == (b, c)  # va=b, vb=c

    res = s.split_edge(shared_edge, 0.25)
    assert res.face_b is not None

    # f1's new loop is A,B,W,C,D: W sits between B(=va) and C(=vb), so the
    # split runs the SAME way as the loop (va-then-vb) and the inserted UV
    # is lerped at t=0.25 straight from B's UV (1.0, 0.0) toward C's UV
    # (1.0, 1.0): (1.0, 0.0) + 0.25 * (0.0, 1.0) = (1.0, 0.25).
    loop_a = s.face_loop(res.face_a)
    uvs_a = s.face_uvs(res.face_a, Side.FRONT)
    assert s.face_material(res.face_a, Side.FRONT) == 7
    np.testing.assert_allclose(uvs_a[loop_a.index(res.vertex)], [1.0, 0.25], atol=1e-6)

    # f2's new loop is B,E,G,C,W: W sits between C(=vb) and B(=va), so the
    # split runs the OPPOSITE way from the loop (vb-then-va) and the inserted
    # UV is lerped at 1-t=0.75 from C's UV (10.0, 1.0) toward B's UV
    # (10.0, 0.0): (10.0, 1.0) + 0.75 * (0.0, -1.0) = (10.0, 0.25).
    loop_b = s.face_loop(res.face_b)
    uvs_b = s.face_uvs(res.face_b, Side.FRONT)
    assert s.face_material(res.face_b, Side.FRONT) == 11
    np.testing.assert_allclose(uvs_b[loop_b.index(res.vertex)], [10.0, 0.25], atol=1e-6)
