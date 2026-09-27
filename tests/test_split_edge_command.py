"""SplitEdgeCommand: do, undo, redo round-trips + composite-sibling safety."""

from __future__ import annotations

import numpy as np
from bermake.commands.command import CompositeCommand
from bermake.commands.scene_commands import AddEdgeCommand, SplitEdgeCommand
from bermake.scene.scene import Scene


def _two_quads(scene: Scene):
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    v5 = scene.add_vertex(np.array([2, 1, 0], dtype=np.float32))
    for a, b in [(v0, v1), (v1, v2), (v2, v3), (v3, v0), (v1, v4), (v4, v5), (v5, v2)]:
        scene.add_edge(a, b)
    scene.add_face_from_loop([v0, v1, v2, v3])
    scene.add_face_from_loop([v1, v4, v5, v2])
    e_shared = scene.add_edge(v1, v2)  # idempotent → existing id
    return e_shared, (v1, v2)


def test_do_splits_and_grows_both_faces():
    scene = Scene()
    e, _ = _two_quads(scene)
    cmd = SplitEdgeCommand(e, 0.5)
    cmd.do(scene)
    faces = list(scene.faces_iter())
    assert len(faces) == 2
    assert all(len(f.loop_vertex_ids) == 5 for f in faces)


def test_undo_restores_counts():
    scene = Scene()
    e, _ = _two_quads(scene)
    pf = sum(1 for _ in scene.faces_iter())
    pe = sum(1 for _ in scene.edges_iter())
    pv = sum(1 for _ in scene.vertices_iter())
    cmd = SplitEdgeCommand(e, 0.5)
    cmd.do(scene)
    cmd.undo(scene)
    assert sum(1 for _ in scene.faces_iter()) == pf
    assert sum(1 for _ in scene.edges_iter()) == pe
    assert sum(1 for _ in scene.vertices_iter()) == pv


def test_do_undo_redo_double_cycle():
    scene = Scene()
    e, _ = _two_quads(scene)
    pf = sum(1 for _ in scene.faces_iter())
    pe = sum(1 for _ in scene.edges_iter())
    cmd = SplitEdgeCommand(e, 0.5)
    cmd.do(scene)
    cmd.undo(scene)
    cmd.do(scene)    # redo
    cmd.undo(scene)  # undo again
    assert sum(1 for _ in scene.faces_iter()) == pf
    assert sum(1 for _ in scene.edges_iter()) == pe


def test_invalid_t_is_noop():
    scene = Scene()
    e, _ = _two_quads(scene)
    cmd = SplitEdgeCommand(e, 0.0)
    cmd.do(scene)    # no-op, must not raise
    cmd.undo(scene)  # no-op, must not raise
    assert sum(1 for _ in scene.faces_iter()) == 2


def test_redo_keeps_new_vertex_id_stable_for_sibling():
    """A sibling AddEdgeCommand that connects to the new vertex must survive an
    undo/redo of the whole composite — i.e. redo must restore the SAME vertex id."""
    scene = Scene()
    e, (v1, v2) = _two_quads(scene)
    anchor = scene.add_vertex(np.array([0.5, -1.0, 0.0], dtype=np.float32))

    comp = CompositeCommand(name="line-onto-edge")
    split = SplitEdgeCommand(e, 0.5)
    split.do(scene)
    comp.children.append(split)
    w = split.new_vertex_id
    assert w is not None
    e_cmd = AddEdgeCommand(anchor, w)
    e_cmd.do(scene)
    comp.children.append(e_cmd)

    comp.undo(scene)
    comp.do(scene)  # redo
    assert scene._mesh.vertex_is_live(w)
    reconnected = scene.add_edge(anchor, w)  # idempotent lookup
    assert scene._mesh.edge_is_live(reconnected)
