"""Generated meshes enter a Scene as ordinary commands (M7.4 Task 10).

`build_mesh_into_scene` is the missing link between Task 9's primitive
generators (which each return a standalone `HalfEdgeMesh`) and Task 11's
tools (which need that mesh living inside an editable, undoable `Scene`).
These tests are written to catch a plausible BROKEN adapter, not just a
completely absent one:

- a count-only assertion would pass against an adapter that welds every
  vertex to the origin, or that winds every face backwards -- so counts are
  always paired with geometry (positions, loop order, edge totals).
- undo is exercised through a real `CommandStack`, from a scene that
  already has geometry in it, so "back to empty" cannot pass by accident
  and a broken adapter that pushes more than one undo step is caught by a
  single `undo()` call not fully reverting.
- the transform test uses translation AND rotation and checks a named
  vertex against its exact expected position, so an adapter that ignores
  `transform` (identity default is easy to accidentally always take) or one
  that applies it row-vector instead of column-vector both fail.
"""

from __future__ import annotations

import numpy as np
import pytest

import bermake._core as core
from bermake.commands import CommandStack, CompositeCommand
from bermake.scene.mesh_builder import build_mesh_into_scene
from bermake.scene.scene import Scene


def _make_box():
    return core.make_box(2.0, 2.0, 2.0)


def _add_seed_triangle(scene: Scene) -> None:
    """Non-empty starting state, so undo tests can't pass by accident."""
    a = scene.add_vertex(np.array([100.0, 100.0, 100.0], dtype=np.float32))
    b = scene.add_vertex(np.array([101.0, 100.0, 100.0], dtype=np.float32))
    c = scene.add_vertex(np.array([100.0, 101.0, 100.0], dtype=np.float32))
    scene.add_face_from_loop([a, b, c])


def _cyclic_match(expected: list, actual: tuple) -> bool:
    """True if `actual` is `expected` read starting from some index, same
    direction -- i.e. the same loop, not reversed and not reordered."""
    n = len(expected)
    if len(actual) != n:
        return False
    actual = list(actual)
    for start in range(n):
        if actual == expected[start:] + expected[:start]:
            return True
    return False


# --- basic shape -------------------------------------------------------


def test_a_box_becomes_eight_vertices_six_faces_twelve_edges():
    scene = Scene()
    build_mesh_into_scene(_make_box(), scene)
    assert len(list(scene.vertices_iter())) == 8
    assert len(list(scene.faces_iter())) == 6
    # 12, not more: two faces sharing a source edge must share ONE scene
    # edge. An adapter that skipped the edge_between dedup check would add
    # each face's 4 edges independently and land on more than 12 (fewer
    # would collide/raise, since add_edge on a true duplicate pair is not
    # what edge_between guards against here -- it is duplicate REQUESTS
    # for the same pair that this test catches).
    assert len(list(scene.edges_iter())) == 12


def test_the_commands_are_already_executed():
    scene = Scene()
    cmds = build_mesh_into_scene(_make_box(), scene)
    assert cmds
    # Already applied BEFORE any push/execute by a caller -- this is the
    # entire reason push_executed (not execute) is the right call for
    # Task 11 to make.
    assert len(list(scene.faces_iter())) == 6


def test_face_loops_are_quads_in_source_order_not_triangulated_or_reordered():
    mesh = _make_box()
    scene = Scene()
    build_mesh_into_scene(mesh, scene)

    # Map every destination vertex's position back to a source vertex id,
    # via the exact float32 match the adapter itself relies on.
    pos_to_src: dict[tuple, int] = {}
    v = mesh.next_live_vertex(0)
    while v != mesh.INVALID_ID:
        pos_to_src[tuple(np.float32(x) for x in mesh.vertex_position(v))] = v
        v = mesh.next_live_vertex(v + 1)

    dst_faces_by_src_positions = []
    for face in scene.faces_iter():
        assert len(face.loop_vertex_ids) == 4, "a box face must stay a quad, not a triangle pair"
        src_loop = [pos_to_src[tuple(np.float32(x) for x in scene.vertex(vid).position)]
                    for vid in face.loop_vertex_ids]
        dst_faces_by_src_positions.append(src_loop)

    # Every source face loop must appear, in the SAME cyclic order (winding
    # preserved), among the destination faces translated back to source ids.
    f = mesh.next_live_face(0)
    while f != mesh.INVALID_ID:
        expected = list(mesh.face_loop_vertices(f))
        assert any(_cyclic_match(expected, tuple(actual)) for actual in dst_faces_by_src_positions), (
            f"source face loop {expected} not found (in order) among {dst_faces_by_src_positions}"
        )
        f = mesh.next_live_face(f + 1)


# --- undo ----------------------------------------------------------------


def test_undoing_via_the_command_stack_is_one_step_and_exact():
    scene = Scene()
    _add_seed_triangle(scene)
    before_v = len(list(scene.vertices_iter()))
    before_f = len(list(scene.faces_iter()))
    before_e = len(list(scene.edges_iter()))

    cmds = build_mesh_into_scene(_make_box(), scene)
    assert len(list(scene.vertices_iter())) == before_v + 8
    assert len(list(scene.faces_iter())) == before_f + 6

    composite = CompositeCommand(name="Insert Box", children=list(cmds))
    stack = CommandStack()
    stack.push_executed(composite, scene)

    undone = stack.undo()
    assert undone
    # A SINGLE undo() call must fully revert: this is what "one undo step"
    # for the whole insertion means, and it is the sharpest check available
    # -- an adapter that returned commands in an order undo() can't cleanly
    # reverse (e.g. a vertex removed while an edge still references it)
    # would raise or leave stale geometry here, not just miscount.
    assert len(list(scene.vertices_iter())) == before_v
    assert len(list(scene.faces_iter())) == before_f
    assert len(list(scene.edges_iter())) == before_e

    # The seed triangle was added directly (not through a command), so the
    # stack now has nothing left for this insertion -- a second undo must
    # report there is nothing more to undo.
    assert not stack.undo()


def test_undo_stack_has_no_further_entries_after_the_single_undo():
    scene = Scene()
    cmds = build_mesh_into_scene(_make_box(), scene)
    composite = CompositeCommand(name="Insert Box", children=list(cmds))
    stack = CommandStack()
    stack.push_executed(composite, scene)
    assert stack.undo()
    assert not stack.undo(), "the whole box must be ONE undo step, not several"


def test_manually_undoing_every_returned_command_in_reverse_empties_the_scene():
    scene = Scene()
    cmds = build_mesh_into_scene(_make_box(), scene)
    for c in reversed(cmds):
        c.undo(scene)
    assert list(scene.faces_iter()) == []
    assert list(scene.edges_iter()) == []
    assert list(scene.vertices_iter()) == []


# --- transform -------------------------------------------------------------


def test_transform_is_column_vector_and_moves_a_named_vertex_exactly():
    # 90-degree rotation about Z (column-vector: (x, y, z) -> (-y, x, z)),
    # then translate by (10, 0, 0) -- matching sweep_stations' convention
    # (p_world = M @ [p; 1]) so Task 11 can feed it a station transform
    # unchanged. A row-vector implementation, or one that ignores rotation
    # (translation-only), lands this vertex somewhere else and fails below.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transform[:3, 3] = [10.0, 0.0, 0.0]

    mesh = _make_box()  # vertex 2 sits at (1, 1, 0) -- see setup below
    assert tuple(mesh.vertex_position(2)) == (1.0, 1.0, 0.0)

    scene = Scene()
    build_mesh_into_scene(mesh, scene, transform=transform)

    # Expected: R @ (1, 1, 0) = (-1, 1, 0); + translation = (9, 1, 0).
    expected = np.array([9.0, 1.0, 0.0], dtype=np.float32)
    positions = [scene.vertex(v.id).position for v in scene.vertices_iter()]
    assert any(np.allclose(p, expected, atol=1e-5) for p in positions), (
        f"expected a vertex at {expected}, got {positions}"
    )
    # And the untransformed position must NOT appear -- guards against an
    # adapter that silently ignores `transform`.
    untouched = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    assert not any(np.allclose(p, untouched, atol=1e-5) for p in positions)


def test_no_transform_leaves_positions_unchanged():
    mesh = _make_box()
    scene = Scene()
    build_mesh_into_scene(mesh, scene)
    positions = {tuple(np.round(v.position, 4)) for v in scene.vertices_iter()}
    src_positions = set()
    v = mesh.next_live_vertex(0)
    while v != mesh.INVALID_ID:
        src_positions.add(tuple(np.round(np.asarray(mesh.vertex_position(v), dtype=np.float32), 4)))
        v = mesh.next_live_vertex(v + 1)
    assert positions == src_positions


# --- unusual shapes: not every mesh has the same face/vertex ratio a box does


@pytest.mark.parametrize(
    "mesh_factory,expected_vertices,expected_faces",
    [
        (lambda: core.make_cylinder(1.0, 1.0, 3), None, 5),  # 3 sides + top + bottom
        (lambda: core.make_cone(1.0, 1.0, 3), None, 4),  # 3 sides + base
    ],
)
def test_other_generators_also_walk_cleanly(mesh_factory, expected_vertices, expected_faces):
    mesh = mesh_factory()
    scene = Scene()
    cmds = build_mesh_into_scene(mesh, scene)
    assert cmds
    assert len(list(scene.faces_iter())) == expected_faces
    for c in reversed(cmds):
        c.undo(scene)
    assert list(scene.faces_iter()) == []
    assert list(scene.vertices_iter()) == []


# --- atomicity on a mid-walk failure ---------------------------------------


def test_a_degenerate_transform_leaves_the_scene_exactly_as_found():
    # A transform with a zero-scaled Z axis collapses a box's top loop onto
    # its bottom loop (matching x/y, now matching z too), welding each top
    # vertex onto its corresponding bottom vertex via Scene.add_vertex's
    # exact-match dedup. A side face's loop then has two adjacent entries
    # that resolve to the SAME scene vertex id, so Scene.add_edge is asked
    # for a self-loop and raises -- mid-walk, after some vertices (and
    # possibly some faces) have already been committed to `scene`.
    #
    # This is exactly the shape of input Task 11 will produce: a UI-supplied
    # dimension of 0, or a scale derived from two coincident picked points,
    # both reach this same zero-scaled-axis transform.
    scene = Scene()
    _add_seed_triangle(scene)
    before_v = len(list(scene.vertices_iter()))
    before_e = len(list(scene.edges_iter()))
    before_f = len(list(scene.faces_iter()))

    transform = np.eye(4, dtype=np.float64)
    transform[2, 2] = 0.0

    with pytest.raises(ValueError):
        build_mesh_into_scene(_make_box(), scene, transform=transform)

    # The failed call must leave the scene exactly as it found it -- not
    # merely "smaller than a full box" -- so a caller that catches
    # ValueError can trust nothing was left behind.
    assert len(list(scene.vertices_iter())) == before_v
    assert len(list(scene.edges_iter())) == before_e
    assert len(list(scene.faces_iter())) == before_f
