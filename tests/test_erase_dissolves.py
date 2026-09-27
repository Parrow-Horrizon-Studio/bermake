"""Eraser tool: a coplanar interior seam merges instead of cascading (M7.6a).

Splitting a face divides it in two (Tasks 1-5). Before this task the obvious
way to undo that by hand -- erase the line just drawn -- destroyed BOTH
halves and left a hole, because the Eraser cascaded to every incident face
unconditionally. These tests pin the new decision: a coplanar interior edge
dissolves its two faces into one; anything else (boundary edge, crease,
multi-shared edge) keeps the old cascade.
"""

from __future__ import annotations

import numpy as np
from bermake.commands import CommandStack, CompositeCommand
from bermake.scene import Scene
from bermake.scene.scene import Side
from bermake.tools import ToolContext
from bermake.tools.erase_tool import EraserTool


def _make(scene: Scene, stack: CommandStack) -> EraserTool:
    tool = EraserTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    return tool


def _open_stroke(tool: EraserTool) -> None:
    # Mirrors the start of EraserTool.on_mouse_press, minus the mouse pick --
    # _erase_edge is a plain method that only needs an open stroke.
    tool._stroke = CompositeCommand(name="Erase")
    tool._erased = set()


def _close_stroke(tool: EraserTool, scene: Scene, stack: CommandStack) -> None:
    # Mirrors EraserTool.on_mouse_release.
    if tool._stroke is not None and tool._stroke.children:
        stack.push_executed(tool._stroke, scene)
    tool._stroke = None
    tool._erased = set()


def _two_coplanar_quads() -> tuple[Scene, int]:
    """Two unit quads on the XY plane sharing edge (v1, v2)."""
    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v4 = s.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    v5 = s.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    s.add_face_from_loop([v0, v1, v2, v3])
    s.add_face_from_loop([v1, v4, v5, v2])
    e_shared = s.edge_between(v1, v2)
    assert e_shared is not None
    return s, e_shared


def _three_coplanar_quads() -> tuple[Scene, int, int]:
    """Three unit quads in a row on the XY plane, sharing (v1,v2) and (v4,v5)."""
    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v4 = s.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    v5 = s.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    v6 = s.add_vertex(np.array([3.0, 0.0, 0.0], dtype=np.float32))
    v7 = s.add_vertex(np.array([3.0, 1.0, 0.0], dtype=np.float32))
    s.add_face_from_loop([v0, v1, v2, v3])
    s.add_face_from_loop([v1, v4, v5, v2])
    s.add_face_from_loop([v4, v6, v7, v5])
    e1 = s.edge_between(v1, v2)
    e2 = s.edge_between(v4, v5)
    assert e1 is not None and e2 is not None
    return s, e1, e2


def _two_creased_quads() -> tuple[Scene, int]:
    """Two quads sharing an edge at a right-angle fold (NOT coplanar)."""
    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v4 = s.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32))
    v5 = s.add_vertex(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    s.add_face_from_loop([v0, v1, v2, v3])
    s.add_face_from_loop([v1, v4, v5, v2])
    e_shared = s.edge_between(v1, v2)
    assert e_shared is not None
    return s, e_shared


def test_erasing_a_coplanar_interior_seam_merges_the_two_faces():
    # Discriminates against the pre-M7.6a Eraser, which cascades every
    # interior edge unconditionally: that implementation removes BOTH faces
    # (face count drops by 2, edge count drops sharply) instead of merging
    # them into one (face count drops by exactly 1).
    s, e_shared = _two_coplanar_quads()
    stack = CommandStack()
    tool = _make(s, stack)
    f0 = len(list(s.faces_iter()))

    _open_stroke(tool)
    tool._erase_edge(e_shared)
    _close_stroke(tool, s, stack)

    assert len(list(s.faces_iter())) == f0 - 1
    assert not s.edge_is_live(e_shared)


def test_erasing_a_non_coplanar_crease_still_deletes_both_faces():
    # Discriminates against an implementation that dissolves ANY interior
    # edge with exactly two incident faces without checking coplanarity --
    # that would merge this crease into one (non-planar) face instead of
    # cascading, silently producing a face the kernel's planar assumption
    # does not hold for.
    s, e_shared = _two_creased_quads()
    stack = CommandStack()
    tool = _make(s, stack)
    f0 = len(list(s.faces_iter()))

    _open_stroke(tool)
    tool._erase_edge(e_shared)
    _close_stroke(tool, s, stack)

    assert len(list(s.faces_iter())) == f0 - 2


def test_erasing_a_boundary_edge_still_cascades():
    # Discriminates against an implementation that only handles the
    # two-face case and mishandles (crashes on, or fails to remove) a
    # boundary edge once the dissolve check is added in front of the
    # cascade loop.
    s = Scene()
    a = s.add_vertex(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    b = s.add_vertex(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    c = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = s.add_vertex(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    s.add_face_from_loop((a, b, c, d))
    e_boundary = s.edge_between(a, b)
    assert e_boundary is not None
    stack = CommandStack()
    tool = _make(s, stack)
    f0 = len(list(s.faces_iter()))

    _open_stroke(tool)
    tool._erase_edge(e_boundary)
    _close_stroke(tool, s, stack)

    assert len(list(s.faces_iter())) == f0 - 1
    assert not s.edge_is_live(e_boundary)


def test_a_drag_mixing_a_dissolve_and_a_cascade_is_one_undo_step():
    # Discriminates against a dissolve path that pushes DissolveEdgeCommand
    # onto the command stack directly (or otherwise outside self._stroke)
    # instead of appending it into the same CompositeCommand as the cascade
    # commands -- that would split one drag into two undo steps.
    s, e_shared = _two_coplanar_quads()
    # A second, unrelated boundary edge on the same scene to erase in the
    # same drag.
    v6 = s.add_vertex(np.array([0.0, 2.0, 0.0], dtype=np.float32))
    v7 = s.add_vertex(np.array([1.0, 2.0, 0.0], dtype=np.float32))
    e_boundary = s.add_edge(v6, v7)
    stack = CommandStack()
    tool = _make(s, stack)
    v0, e0, f0 = (
        len(list(s.vertices_iter())),
        len(list(s.edges_iter())),
        len(list(s.faces_iter())),
    )

    _open_stroke(tool)
    tool._erase_edge(e_shared)
    tool._erase_edge(e_boundary)
    _close_stroke(tool, s, stack)

    assert f0 - 1 == len(list(s.faces_iter()))
    assert stack.can_undo
    stack.undo()
    assert (
        len(list(s.vertices_iter())),
        len(list(s.edges_iter())),
        len(list(s.faces_iter())),
    ) == (v0, e0, f0)


def test_chained_dissolves_in_one_stroke_survive_undo_redo_undo():
    # Discriminates against DissolveEdgeCommand's redo minting a FRESH merged
    # face id every time (Scene.dissolve_edge has no id-preserving variant)
    # while undo always restores the two parents to their ORIGINAL ids.
    # Chaining two dissolves in one Eraser stroke -- an ordinary "split a
    # face twice, then drag-erase both seams" gesture, the headline use case
    # this task exists to serve -- makes the second command's undo/redo
    # id-bookkeeping depend on the first's merged-face id staying stable
    # across a redo. Before the id-preserving fix, the second Ctrl+Z after a
    # Ctrl+Y raises KeyError from HalfEdgeMesh.remove_face on an
    # already-dead face id.
    s, e1, e2 = _three_coplanar_quads()
    stack = CommandStack()
    tool = _make(s, stack)
    v0, ed0, f0 = (
        len(list(s.vertices_iter())),
        len(list(s.edges_iter())),
        len(list(s.faces_iter())),
    )

    _open_stroke(tool)
    tool._erase_edge(e1)
    tool._erase_edge(e2)
    _close_stroke(tool, s, stack)
    assert len(list(s.faces_iter())) == f0 - 2  # three quads merged into one

    stack.undo()
    assert (
        len(list(s.vertices_iter())),
        len(list(s.edges_iter())),
        len(list(s.faces_iter())),
    ) == (v0, ed0, f0)

    stack.redo()
    assert len(list(s.faces_iter())) == f0 - 2

    stack.undo()  # this is the undo that used to crash
    assert (
        len(list(s.vertices_iter())),
        len(list(s.edges_iter())),
        len(list(s.faces_iter())),
    ) == (v0, ed0, f0)


def test_erasing_an_edge_shared_by_two_faces_falls_back_to_the_cascade():
    # A square split along an interior chain [v0, m, v2] gives two coplanar
    # children that share TWO edges (v0-m and m-v2), not one.
    # Scene.dissolve_edge's kernel refuses this (halfedge.cpp's
    # shared-edge-count guard), so EraserTool._try_dissolve must read that
    # refusal back and let the caller cascade instead. Discriminates against
    # dropping the `scene.edge_is_live(e_id)` readback in `_try_dissolve`:
    # without it, a refused DissolveEdgeCommand is trusted as a success and
    # the Eraser silently does nothing, leaving the edge and both faces
    # untouched.
    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([2.0, 2.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 2.0, 0.0], dtype=np.float32))
    fid = s.add_face_from_loop([v0, v1, v2, v3])
    m = s.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    s.add_edge(v0, m)
    s.add_edge(m, v2)
    out = s.split_face(fid, [v0, m, v2])
    assert out is not None
    e_multi = s.edge_between(v0, m)
    assert e_multi is not None

    stack = CommandStack()
    tool = _make(s, stack)
    f0 = len(list(s.faces_iter()))

    _open_stroke(tool)
    tool._erase_edge(e_multi)
    _close_stroke(tool, s, stack)

    assert len(list(s.faces_iter())) == f0 - 2
    assert not s.edge_is_live(e_multi)


def test_split_then_erase_round_trip_through_the_eraser_restores_material():
    # Fix round 1 (Minor 6): test_split_then_erase_round_trip_restores_the
    # _parents_material in test_dissolve_sidecars.py calls
    # Scene.dissolve_edge directly despite its name, so nothing actually
    # drove the headline gesture -- split a face, then erase the seam with
    # the Eraser tool -- end to end. This one goes through
    # EraserTool._erase_edge, the real path a user's drag takes.
    s = Scene()
    v0 = s.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = s.add_vertex(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    v2 = s.add_vertex(np.array([2.0, 2.0, 0.0], dtype=np.float32))
    v3 = s.add_vertex(np.array([0.0, 2.0, 0.0], dtype=np.float32))
    fid = s.add_face_from_loop([v0, v1, v2, v3])
    s.set_face_material(fid, 7, Side.FRONT)
    s.add_edge(v0, v2)
    out = s.split_face(fid, [v0, v2])
    assert out is not None
    e_shared = s.edge_between(v0, v2)
    assert e_shared is not None

    stack = CommandStack()
    tool = _make(s, stack)
    f0 = len(list(s.faces_iter()))

    _open_stroke(tool)
    tool._erase_edge(e_shared)
    _close_stroke(tool, s, stack)

    faces = list(s.faces_iter())
    assert len(faces) == f0 - 1 == 1
    merged = faces[0].id
    assert set(s.face_loop(merged)) == {v0, v1, v2, v3}
    assert s.face_material(merged, Side.FRONT) == 7
