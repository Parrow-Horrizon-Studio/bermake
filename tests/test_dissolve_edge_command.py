"""DissolveEdgeCommand: do, undo, redo round-trips."""

from __future__ import annotations

import numpy as np
from bermake.commands.scene_commands import DissolveEdgeCommand
from bermake.scene.scene import Scene


def _two_quads_sharing_edge(scene: Scene) -> tuple[int, int, int]:
    """Returns (f1, f2, shared_edge_id) — identical setup to test_scene_dissolve."""
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    v5 = scene.add_vertex(np.array([2, 1, 0], dtype=np.float32))
    scene.add_edge(v0, v1)
    e_shared = scene.add_edge(v1, v2)
    scene.add_edge(v2, v3)
    scene.add_edge(v3, v0)
    scene.add_edge(v1, v4)
    scene.add_edge(v4, v5)
    scene.add_edge(v5, v2)
    f1 = scene.add_face_from_loop([v0, v1, v2, v3])
    f2 = scene.add_face_from_loop([v1, v4, v5, v2])
    return f1, f2, e_shared


def test_do_removes_shared_edge_and_merges_faces():
    scene = Scene()
    f1, f2, e_shared = _two_quads_sharing_edge(scene)

    cmd = DissolveEdgeCommand(e_shared)
    cmd.do(scene)

    # The two source faces are gone; one merged face has 6 vertices.
    live_faces = [f.id for f in scene.faces_iter()]
    assert f1 not in live_faces
    assert f2 not in live_faces
    assert len(live_faces) == 1
    assert len(scene.face(live_faces[0]).loop_vertex_ids) == 6


def test_undo_restores_both_original_faces():
    scene = Scene()
    _, _, e_shared = _two_quads_sharing_edge(scene)
    pre_face_count = sum(1 for _ in scene.faces_iter())
    pre_edge_count = sum(1 for _ in scene.edges_iter())

    cmd = DissolveEdgeCommand(e_shared)
    cmd.do(scene)
    cmd.undo(scene)

    post_face_count = sum(1 for _ in scene.faces_iter())
    post_edge_count = sum(1 for _ in scene.edges_iter())
    assert post_face_count == pre_face_count
    assert post_edge_count == pre_edge_count


def test_do_returns_none_op_on_boundary_edge():
    """Command on a boundary edge does nothing; undo also does nothing.
    The undo stack must stay consistent (no exceptions)."""
    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    scene.add_edge(v0, v1)
    scene.add_edge(v1, v2)
    scene.add_edge(v2, v0)
    f = scene.add_face_from_loop([v0, v1, v2])
    e_boundary = scene.face_edges(f)[0]

    cmd = DissolveEdgeCommand(e_boundary)
    cmd.do(scene)    # should not raise
    cmd.undo(scene)  # should not raise

    # Mesh unchanged.
    assert sum(1 for _ in scene.faces_iter()) == 1
    assert scene.face(f).loop_vertex_ids == (v0, v1, v2)


def test_do_undo_redo_round_trip():
    scene = Scene()
    _, _, e_shared = _two_quads_sharing_edge(scene)

    cmd = DissolveEdgeCommand(e_shared)
    cmd.do(scene)

    cmd.undo(scene)
    cmd.do(scene)  # redo

    # After redo the mesh must be back to the MERGED state: exactly one live
    # face, a hexagon. Redo re-resolves the (now-changed) shared edge id from
    # the captured vertex pair, so it genuinely re-merges (not a no-op).
    live_faces = [f.id for f in scene.faces_iter()]
    assert len(live_faces) == 1
    merged_id_after_redo = live_faces[0]
    assert scene._mesh.face_is_live(merged_id_after_redo)
    assert len(scene.face(merged_id_after_redo).loop_vertex_ids) == 6


def test_do_undo_redo_double_cycle():
    """do -> undo -> do -> undo round-trips twice, exercising the redo edge
    re-resolution and the _merged_face_id overwrite together."""
    scene = Scene()
    _, _, e_shared = _two_quads_sharing_edge(scene)
    pre_face_count = sum(1 for _ in scene.faces_iter())
    pre_edge_count = sum(1 for _ in scene.edges_iter())

    cmd = DissolveEdgeCommand(e_shared)
    cmd.do(scene)
    cmd.undo(scene)
    cmd.do(scene)    # redo
    cmd.undo(scene)  # undo again

    assert sum(1 for _ in scene.faces_iter()) == pre_face_count
    assert sum(1 for _ in scene.edges_iter()) == pre_edge_count


def test_undo_inside_composite_with_sibling_addface():
    """Regression: a DissolveEdgeCommand that consumes a face created by a
    sibling AddFaceCommand in the same composite must restore that face to its
    ORIGINAL id, so the sibling's undo (remove_face by cached id) succeeds.
    Reproduces the Task 8 atomic-undo bug."""
    from bermake.commands.command import CompositeCommand
    from bermake.commands.scene_commands import AddFaceCommand

    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    v5 = scene.add_vertex(np.array([2, 1, 0], dtype=np.float32))
    for a, b in [(v0, v1), (v1, v2), (v2, v3), (v3, v0), (v1, v4), (v4, v5), (v5, v2)]:
        scene.add_edge(a, b)
    f1 = scene.add_face_from_loop([v0, v1, v2, v3])
    pre_faces = sum(1 for _ in scene.faces_iter())
    pre_edges = sum(1 for _ in scene.edges_iter())

    comp = CompositeCommand(name="seam-merge-repro")
    add_f2 = AddFaceCommand((v1, v4, v5, v2))
    add_f2.do(scene)
    comp.children.append(add_f2)
    shared = next(
        e
        for e in scene.face_edges(f1)
        if {scene.edge(e).v1_id, scene.edge(e).v2_id} == {v1, v2}
    )
    dissolve = DissolveEdgeCommand(shared)
    dissolve.do(scene)
    comp.children.append(dissolve)

    # One atomic undo must not raise and must round-trip face/edge counts.
    comp.undo(scene)
    assert sum(1 for _ in scene.faces_iter()) == pre_faces
    assert sum(1 for _ in scene.edges_iter()) == pre_edges


def test_redo_with_no_live_edge_between_vertices_does_not_fabricate_one():
    """Regression for #26: the redo branch used to re-resolve the edge id via
    scene._mesh.add_halfedge_pair(va, vb), a MUTATING call that creates the
    pair when absent. If the captured vertex pair is ever not joined by a
    live edge when redo runs, that old code would silently fabricate a new
    boundary edge (dissolve_edge would then no-op on it, but the fabricated
    edge stays behind). The fix must instead recognize the missing pair and
    no-op cleanly, creating nothing.

    We force the redo branch directly (rather than reconstructing the full
    do/undo/sibling-removal sequence that produces it in practice) by setting
    the command's captured-state fields by hand, on two vertices with no edge
    between them at all.
    """
    scene = Scene()
    va = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    vb = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    cmd = DissolveEdgeCommand(0xFFFFFFFF)  # edge id irrelevant; unused by the redo branch
    cmd._captured_f1 = ()  # any non-None value routes do() into the redo branch
    cmd._shared_verts = (va, vb)

    cmd.do(scene)  # exercises the redo branch with an unjoined vertex pair

    assert cmd._was_noop is True
    assert list(scene.edges_iter()) == []
