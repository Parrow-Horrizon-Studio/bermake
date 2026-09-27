"""Tests for AddVertex / AddEdge / AddFace / Remove* / ClearScene commands."""

from __future__ import annotations

import numpy as np


def _three_vertex_scene():
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return s, v0, v1, v2


def test_add_vertex_command_round_trip():
    from bermake.commands.scene_commands import AddVertexCommand
    from bermake.scene import Scene

    s = Scene()
    pos = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    cmd = AddVertexCommand(pos)

    cmd.do(s)
    assert len(list(s.vertices_iter())) == 1

    cmd.undo(s)
    assert len(list(s.vertices_iter())) == 0

    cmd.do(s)
    assert len(list(s.vertices_iter())) == 1


def test_add_edge_command_round_trip():
    from bermake.commands.scene_commands import AddEdgeCommand
    from bermake.scene import Scene

    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    cmd = AddEdgeCommand(v0, v1)

    cmd.do(s)
    assert len(list(s.edges_iter())) == 1

    cmd.undo(s)
    assert len(list(s.edges_iter())) == 0


def test_add_face_command_round_trip():
    from bermake.commands.scene_commands import AddFaceCommand
    from bermake.scene import Scene

    s, v0, v1, v2 = _three_vertex_scene()
    s.add_edge(v0, v1); s.add_edge(v1, v2); s.add_edge(v2, v0)
    cmd = AddFaceCommand((v0, v1, v2))

    cmd.do(s)
    assert len(list(s.faces_iter())) == 1

    cmd.undo(s)
    assert len(list(s.faces_iter())) == 0


def test_remove_face_command_round_trip():
    from bermake.commands.scene_commands import AddFaceCommand, RemoveFaceCommand
    from bermake.scene import Scene

    s, v0, v1, v2 = _three_vertex_scene()
    s.add_edge(v0, v1); s.add_edge(v1, v2); s.add_edge(v2, v0)
    add = AddFaceCommand((v0, v1, v2))
    add.do(s)
    f = next(iter(s.faces_iter())).id

    remove = RemoveFaceCommand(f)
    remove.do(s)
    assert len(list(s.faces_iter())) == 0

    remove.undo(s)
    assert len(list(s.faces_iter())) == 1
    assert next(iter(s.faces_iter())).id == f


def test_remove_edge_command_round_trip():
    from bermake.commands.scene_commands import AddEdgeCommand, RemoveEdgeCommand
    from bermake.scene import Scene

    s, v0, v1, _ = _three_vertex_scene()
    add = AddEdgeCommand(v0, v1)
    add.do(s)
    e = next(iter(s.edges_iter())).id

    remove = RemoveEdgeCommand(e)
    remove.do(s)
    assert len(list(s.edges_iter())) == 0

    remove.undo(s)
    assert len(list(s.edges_iter())) == 1


def test_remove_vertex_command_round_trip():
    from bermake.commands.scene_commands import RemoveVertexCommand
    from bermake.scene import Scene

    s = Scene()
    v = s.add_vertex(np.array([1.0, 2.0, 0.0], dtype=np.float32))

    remove = RemoveVertexCommand(v)
    remove.do(s)
    assert len(list(s.vertices_iter())) == 0

    remove.undo(s)
    assert len(list(s.vertices_iter())) == 1
    restored = next(iter(s.vertices_iter()))
    assert restored.id == v
    np.testing.assert_array_equal(restored.position, np.array([1.0, 2.0, 0.0], dtype=np.float32))


def test_clear_scene_command_captures_and_restores():
    from bermake.commands.scene_commands import ClearSceneCommand
    from bermake.scene import Scene

    s, v0, v1, v2 = _three_vertex_scene()
    s.add_edge(v0, v1); s.add_edge(v1, v2); s.add_edge(v2, v0)
    s.add_face_from_loop((v0, v1, v2))
    assert len(list(s.vertices_iter())) == 3
    assert len(list(s.edges_iter())) == 3
    assert len(list(s.faces_iter())) == 1

    cmd = ClearSceneCommand()
    cmd.do(s)
    assert len(list(s.vertices_iter())) == 0
    assert len(list(s.edges_iter())) == 0
    assert len(list(s.faces_iter())) == 0

    cmd.undo(s)
    # All IDs restored.
    verts = list(s.vertices_iter())
    edges = list(s.edges_iter())
    faces = list(s.faces_iter())
    assert len(verts) == 3
    assert len(edges) == 3
    assert len(faces) == 1
    assert {v.id for v in verts} == {v0, v1, v2}


def test_add_vertex_command_undo_leaves_a_deduped_vertex_alone():
    # Scene.add_vertex is idempotent on exact position, so a command aimed at
    # an occupied position resolves onto the EXISTING vertex. It never created
    # it, so undo must not delete it out from under whoever owns it.
    from bermake.commands.scene_commands import AddVertexCommand

    s, v0, _v1, _v2 = _three_vertex_scene()
    before = {v.id for v in s.vertices_iter()}

    cmd = AddVertexCommand(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    cmd.do(s)
    assert cmd._vertex_id == v0  # deduped onto the pre-existing vertex
    assert {v.id for v in s.vertices_iter()} == before

    cmd.undo(s)
    assert {v.id for v in s.vertices_iter()} == before, (
        "undo deleted a pre-existing vertex the command never created"
    )

    # do -> undo -> redo -> undo: still a clean no-op in both directions.
    cmd.do(s)
    assert {v.id for v in s.vertices_iter()} == before
    cmd.undo(s)
    assert {v.id for v in s.vertices_iter()} == before


def test_add_edge_command_undo_leaves_a_deduped_edge_alone():
    from bermake.commands.scene_commands import AddEdgeCommand

    s, v0, v1, _v2 = _three_vertex_scene()
    e0 = s.add_edge(v0, v1)
    before = {e.id for e in s.edges_iter()}

    cmd = AddEdgeCommand(v0, v1)
    cmd.do(s)
    assert cmd._edge_id == e0  # deduped onto the pre-existing edge
    assert {e.id for e in s.edges_iter()} == before

    cmd.undo(s)
    assert {e.id for e in s.edges_iter()} == before, (
        "undo deleted a pre-existing edge the command never created"
    )

    cmd.do(s)
    assert {e.id for e in s.edges_iter()} == before
    cmd.undo(s)
    assert {e.id for e in s.edges_iter()} == before


def test_add_vertex_and_edge_commands_still_own_what_they_actually_create():
    # The ownership guard must not turn a genuine creation into a no-op:
    # do -> undo -> redo -> undo on freshly-minted geometry still round-trips,
    # id-preserving, exactly as before.
    from bermake.commands.scene_commands import AddEdgeCommand, AddVertexCommand

    s, v0, _v1, _v2 = _three_vertex_scene()
    v_cmd = AddVertexCommand(np.array([9.0, 9.0, 9.0], dtype=np.float32))
    v_cmd.do(s)
    new_v = v_cmd._vertex_id
    assert new_v not in (v0,)
    e_cmd = AddEdgeCommand(v0, new_v)
    e_cmd.do(s)
    new_e = e_cmd._edge_id

    e_cmd.undo(s)
    v_cmd.undo(s)
    assert new_v not in {v.id for v in s.vertices_iter()}
    assert new_e not in {e.id for e in s.edges_iter()}

    v_cmd.do(s)  # redo restores the SAME ids
    e_cmd.do(s)
    assert v_cmd._vertex_id == new_v
    assert e_cmd._edge_id == new_e
    assert new_v in {v.id for v in s.vertices_iter()}
    assert new_e in {e.id for e in s.edges_iter()}

    e_cmd.undo(s)
    v_cmd.undo(s)
    assert new_v not in {v.id for v in s.vertices_iter()}
    assert new_e not in {e.id for e in s.edges_iter()}
