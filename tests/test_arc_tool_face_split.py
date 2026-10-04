"""ArcTool: drawing an arc across a face divides it (M7.6a task 5, fix round 1).

Mirrors tests/test_line_tool_face_split.py. Before this file, chain_cuts_face
and SplitFaceCommand were wired into ArcTool._commit_polyline with no
automated coverage at all -- every existing test_arc_tool.py case draws into
an empty scene, so the split branch was never entered.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.commands.scene_commands import SplitEdgeCommand
from bermake.geometry.transforms import mat_translate
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.tools.arc_tool import ArcTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind, SnapResult


def _snap(kind, pos, **kw):
    return SnapResult(
        kind=kind,
        world_position=np.array(pos, dtype=np.float32),
        axis=kw.get("axis"),
        vertex_id=kw.get("vertex_id"),
        label=kw.get("label", ""),
        edge_id=kw.get("edge_id"),
        face_id=kw.get("face_id"),
        edge_t=kw.get("edge_t"),
    )


def _make_tool(scene, model=None):
    stack = CommandStack()
    tool = ArcTool()
    tool.activate(
        ToolContext(
            scene=scene,
            command_stack=stack,
            camera=None,
            widget_size_provider=None,
            model=model,
        )
    )
    return tool, stack


def _quad(scene, origin=(0.0, 0.0, 0.0), size=4.0):
    ox, oy, oz = origin
    v = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [
            (ox, oy, oz),
            (ox + size, oy, oz),
            (ox + size, oy + size, oz),
            (ox, oy + size, oz),
        ]
    ]
    return scene.add_face_from_loop(v), v


def _draw_arc_across(tool, start, end, bulge):
    """Three clicks: chord start, chord end, bulge point -- the ArcTool
    commits on the third click."""
    tool.on_mouse_press(None, _snap(SnapKind.ENDPOINT, start, vertex_id=None))
    tool.on_mouse_press(None, _snap(SnapKind.ENDPOINT, end, vertex_id=None))
    tool.on_mouse_press(None, _snap(SnapKind.GRID, bulge))


def test_an_arc_drawn_across_a_face_splits_it():
    scene = Scene()
    fid, _v = _quad(scene)
    tool, stack = _make_tool(scene)

    # Chord along the diagonal, small bulge kept well inside the quad so the
    # whole 12-segment arc stays strictly interior between the two corners.
    _draw_arc_across(tool, (0, 0, 0), (4, 4, 0), (2.2, 1.8, 0))

    assert sum(1 for _ in scene.faces_iter()) == 2
    assert not scene._mesh.face_is_live(fid)
    assert stack.can_undo


def test_one_undo_after_a_splitting_arc_returns_to_one_face():
    """Same same-undo-step guarantee as LineTool: the split must be folded
    into the arc's own composite, not pushed as a second undo entry."""
    scene = Scene()
    fid, _v = _quad(scene)
    tool, stack = _make_tool(scene)

    _draw_arc_across(tool, (0, 0, 0), (4, 4, 0), (2.2, 1.8, 0))
    assert sum(1 for _ in scene.faces_iter()) == 2
    edges_after_split = sum(1 for _ in scene.edges_iter())

    stack.undo()

    assert sum(1 for _ in scene.faces_iter()) == 1
    assert scene._mesh.face_is_live(fid)
    # The arc's own edges must be gone too, not just the split half of the
    # gesture (see the equivalent LineTool undo test for why this is the
    # assertion that actually discriminates a correctly-folded undo step).
    assert sum(1 for _ in scene.edges_iter()) < edges_after_split
    assert stack.can_undo is False


def _on_edge(scene, a, b, pos, t):
    """A MIDPOINT/ON_EDGE snap on the edge joining vertices a and b."""
    e = scene.edge_between(a, b)
    assert e is not None
    kind = SnapKind.MIDPOINT if t == 0.5 else SnapKind.ON_EDGE
    return _snap(kind, pos, edge_id=e, edge_t=t)


def _draw_arc_snaps(tool, start_snap, end_snap, bulge):
    tool.on_mouse_press(None, start_snap)
    tool.on_mouse_press(None, end_snap)
    tool.on_mouse_press(None, _snap(SnapKind.GRID, bulge))


def _counts(scene):
    return (
        sum(1 for _ in scene.faces_iter()),
        sum(1 for _ in scene.edges_iter()),
        sum(1 for _ in scene.vertices_iter()),
    )


def _live_vertices_at(scene, **fixed):
    """Live vertex ids whose position matches every given axis (x=, y=, z=)."""
    axis = {"x": 0, "y": 1, "z": 2}
    out = []
    for v in scene.vertices_iter():
        if all(abs(float(v.position[axis[k]]) - val) < 1e-6 for k, val in fixed.items()):
            out.append(v.id)
    return out


# An arc is 13 points: 2 chord ends and 11 interior vertices.
_ARC_INTERIOR = 11


def test_an_arc_whose_endpoints_land_on_edge_interiors_splits_both_edges_and_the_face():
    """Was the pinned limitation (#122 part 2): both chord ends on edge
    INTERIORS (the midpoints of two opposite sides), the way most arcs are
    drawn. Each end now splits its edge, so the arc runs between two loop
    vertices and divides the face."""
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, _stack = _make_tool(scene)
    bottom = scene.edge_between(v[0], v[1])
    top = scene.edge_between(v[2], v[3])

    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (2, 0, 0), 0.5),
        _on_edge(scene, v[2], v[3], (2, 4, 0), 0.5),
        (3.0, 2.0, 0),
    )

    assert not scene.edge_is_live(bottom)
    assert not scene.edge_is_live(top)
    assert not scene._mesh.face_is_live(fid)
    faces, edges, verts = _counts(scene)
    assert faces == 2
    # 4 corners + 2 split vertices + the arc's interior; no free vertex.
    assert verts == 4 + 2 + _ARC_INTERIOR
    # 4 sides + 2 extra from the splits + 12 arc segments.
    assert edges == 4 + 2 + 12
    loops = [set(f.loop_vertex_ids) for f in scene.faces_iter()]
    for vid in _live_vertices_at(scene, x=2.0, z=0.0, y=0.0) + _live_vertices_at(
        scene, x=2.0, z=0.0, y=4.0
    ):
        assert all(vid in loop for loop in loops)  # a mid vertex borders both halves


def test_an_arc_from_a_corner_to_a_mid_edge_splits_the_edge_and_the_face():
    scene = Scene()
    _fid, v = _quad(scene, size=4.0)
    tool, _stack = _make_tool(scene)
    top = scene.edge_between(v[2], v[3])

    _draw_arc_snaps(
        tool,
        _snap(SnapKind.ENDPOINT, (0, 0, 0), vertex_id=v[0]),
        _on_edge(scene, v[2], v[3], (2, 4, 0), 0.5),
        (1.4, 1.8, 0),
    )

    assert not scene.edge_is_live(top)
    faces, edges, verts = _counts(scene)
    assert faces == 2
    assert verts == 4 + 1 + _ARC_INTERIOR
    assert edges == 4 + 1 + 12


def test_mid_edge_ends_split_their_edges_even_when_the_arc_cuts_no_face():
    """D6: the edges split whether or not a face is cut. The arc bows out of
    the quad, so it divides nothing, but its ends still become real vertices
    of the bottom edge instead of free vertices lying on it."""
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)
    bottom = scene.edge_between(v[0], v[1])

    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (1, 0, 0), 0.25),
        _on_edge(scene, v[0], v[1], (3, 0, 0), 0.75),
        (2.0, -1.0, 0),
    )

    assert not scene.edge_is_live(bottom)
    faces, edges, verts = _counts(scene)
    assert faces == 1  # still one face, though its loop now has the two new vertices
    (face,) = list(scene.faces_iter())
    assert len(face.loop_vertex_ids) == 6
    assert verts == 4 + 2 + _ARC_INTERIOR
    assert edges == 4 + 2 + 12
    stack.undo()
    assert _counts(scene) == (1, 4, 4)
    assert scene._mesh.face_is_live(fid)


def _assert_bottom_is_a_chain(scene):
    """Along y = 0 the bottom side is v0 - a - b - v1: vertices at x = 0, 1, 3, 4
    joined by three consecutive edges, so no gap and no T-junction."""
    ids = sorted(
        _live_vertices_at(scene, y=0.0, z=0.0),
        key=lambda i: float(scene.vertex(i).position[0]),
    )
    assert [float(scene.vertex(i).position[0]) for i in ids] == [0.0, 1.0, 3.0, 4.0]
    for a, b in pairwise(ids):
        assert scene.edge_between(a, b) is not None


def _draw_both_ends_on_the_bottom_edge(first_t, second_t, bulge):
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)
    bottom = scene.edge_between(v[0], v[1])
    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (4 * first_t, 0, 0), first_t),
        _on_edge(scene, v[0], v[1], (4 * second_t, 0, 0), second_t),
        bulge,
    )
    return scene, stack, fid, v, bottom


def test_both_ends_on_the_same_edge_resolve_the_second_against_the_new_halves():
    """Review Focus 5. The first end's split retires the edge id, so the second
    end is re-resolved against the two new sub-edges: both splits happen (the
    bottom becomes three edges), nothing raises, and the cap the arc encloses
    is cut off as its own face."""
    scene, _stack, fid, _v, bottom = _draw_both_ends_on_the_bottom_edge(0.25, 0.75, (2.0, 1.0, 0))

    assert not scene.edge_is_live(bottom)
    faces, edges, verts = _counts(scene)
    assert faces == 2
    assert not scene._mesh.face_is_live(fid)
    assert verts == 4 + 2 + _ARC_INTERIOR
    assert edges == 4 + 2 + 12
    # Along y = 0 the bottom is now v0 - a - b - v1: three edges, no gap and no T-junction.
    _assert_bottom_is_a_chain(scene)


def test_both_ends_on_the_same_edge_in_reverse_order_also_split_both():
    """The second end lies on the FIRST sub-edge here, not the second."""
    scene, _stack, _fid, _v, bottom = _draw_both_ends_on_the_bottom_edge(0.75, 0.25, (2.0, 1.0, 0))

    assert not scene.edge_is_live(bottom)
    faces, edges, verts = _counts(scene)
    assert faces == 2
    assert verts == 4 + 2 + _ARC_INTERIOR
    assert edges == 4 + 2 + 12
    _assert_bottom_is_a_chain(scene)


def test_one_undo_removes_the_arc_both_edge_splits_and_the_face_split_and_redo_restores_them():
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)
    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (2, 0, 0), 0.5),
        _on_edge(scene, v[2], v[3], (2, 4, 0), 0.5),
        (3.0, 2.0, 0),
    )
    after = _counts(scene)
    assert after[0] == 2

    stack.undo()
    assert _counts(scene) == (1, 4, 4)
    assert scene._mesh.face_is_live(fid)
    assert tuple(scene.face(fid).loop_vertex_ids) == tuple(v)
    assert stack.can_undo is False

    stack.redo()
    assert _counts(scene) == after
    assert not scene._mesh.face_is_live(fid)

    stack.undo()
    assert _counts(scene) == (1, 4, 4)


def test_one_undo_also_covers_two_splits_of_the_same_edge():
    scene, stack, fid, _v, _bottom = _draw_both_ends_on_the_bottom_edge(0.25, 0.75, (2.0, 1.0, 0))
    after = _counts(scene)

    stack.undo()
    assert _counts(scene) == (1, 4, 4)
    assert scene._mesh.face_is_live(fid)

    stack.redo()
    assert _counts(scene) == after


def test_an_arc_that_does_not_cross_a_face_leaves_face_count_unchanged():
    scene = Scene()
    fid, _v = _quad(scene)
    tool, _stack = _make_tool(scene)

    # Drawn entirely away from the quad.
    _draw_arc_across(tool, (10, 0, 0), (12, 0, 0), (11, 1, 0))

    assert sum(1 for _ in scene.faces_iter()) == 1
    assert scene._mesh.face_is_live(fid)


def test_an_arc_split_reaches_through_a_non_identity_world_transform():
    """Drawn inside a translated group: the snap positions this test passes
    are WORLD-space (the convention every tool uses), while the quad's own
    vertices live in the group's LOCAL frame -- build_open_polyline's
    world->local conversion has to run and land on the same local vertices
    for chain_cuts_face to find anything to split at all."""
    translation = [5.0, 0.0, 0.0]
    wt = mat_translate(translation)
    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    scene = grp_def.mesh
    fid, _v = _quad(scene)

    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)

    tool, stack = _make_tool(scene, model=model)
    # LOCAL (0,0,0)/(4,4,0) -> WORLD (5,0,0)/(9,4,0).
    _draw_arc_across(tool, (5, 0, 0), (9, 4, 0), (7.2, 1.8, 0))

    assert sum(1 for _ in scene.faces_iter()) == 2
    assert not scene._mesh.face_is_live(fid)
    assert stack.can_undo


def test_a_mid_edge_arc_split_reaches_through_a_non_identity_world_transform():
    """The edge_t on a snap is a fraction of the edge, so it is frame
    independent; the second end's re-resolution against the sub-edges is not,
    because closest_point_on_edge works in the group's local frame."""
    wt = mat_translate([5.0, 0.0, 0.0])
    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    scene = grp_def.mesh
    fid, v = _quad(scene)
    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)
    tool, _stack = _make_tool(scene, model=model)

    # Both ends on the bottom edge, in WORLD coordinates (local x + 5).
    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (6, 0, 0), 0.25),
        _on_edge(scene, v[0], v[1], (8, 0, 0), 0.75),
        (7.0, 1.0, 0),
    )

    faces, edges, verts = _counts(scene)
    assert faces == 2
    assert not scene._mesh.face_is_live(fid)
    assert verts == 4 + 2 + _ARC_INTERIOR
    assert edges == 4 + 2 + 12


def test_an_end_off_the_drawing_plane_splits_no_edge_it_never_reaches():
    """Start on the ground, end on the midpoint of a box-top edge 1 m up. The
    plane is the ground's, so the arc stays on it and its end is the projection
    of that midpoint. The raised edge is not where the arc ends, so it must not
    split, and the arc is still created."""
    scene = Scene()
    _fid, v = _quad(scene, origin=(0.0, 0.0, 1.0), size=4.0)
    tool, stack = _make_tool(scene)
    top_front = scene.edge_between(v[0], v[1])

    _draw_arc_snaps(
        tool,
        _snap(SnapKind.GRID, (1, -2, 0)),
        _on_edge(scene, v[0], v[1], (2, 0, 1), 0.5),
        (2.0, -1.0, 0),
    )

    assert scene.edge_is_live(top_front)
    faces, edges, verts = _counts(scene)
    assert faces == 1
    assert verts == 4 + 13  # the arc's own 13 points; nothing added on the raised edge
    assert edges == 4 + 12
    assert stack.can_undo
    stack.undo()
    assert _counts(scene) == (1, 4, 4)


def _wall_and_top(scene):
    """A front wall (plane y = 0) and a top (plane z = 4) sharing the edge
    (0,0,4)-(4,0,4)."""
    pts = [(0, 0, 0), (4, 0, 0), (4, 0, 4), (0, 0, 4)]
    wall_v = [scene.add_vertex(np.array(p, dtype=np.float32)) for p in pts]
    wall = scene.add_face_from_loop(wall_v)
    top_v = [
        wall_v[3],
        wall_v[2],
        scene.add_vertex(np.array((4, 4, 4), dtype=np.float32)),
        scene.add_vertex(np.array((0, 4, 4), dtype=np.float32)),
    ]
    top = scene.add_face_from_loop(top_v)
    return wall, wall_v, top, top_v


def test_a_start_on_a_shared_edge_does_not_split_the_far_edge_of_the_other_face():
    """The cursor on a shared edge can report either face. Here it reports the
    front wall, so the plane is the wall's (y = 0). The end snap is on the top's
    far edge (y = 4), which is off that plane, so that edge must stay whole."""
    scene = Scene()
    wall, wall_v, top, top_v = _wall_and_top(scene)
    tool, _stack = _make_tool(scene)
    shared = scene.edge_between(wall_v[2], wall_v[3])
    far = scene.edge_between(top_v[2], top_v[3])
    assert shared is not None
    assert far is not None
    before = _counts(scene)

    start = _snap(SnapKind.MIDPOINT, (2, 0, 4), edge_id=shared, edge_t=0.5, face_id=wall)
    end = _snap(SnapKind.ON_EDGE, (3, 4, 4), edge_id=far, edge_t=0.75, face_id=top)
    _draw_arc_snaps(tool, start, end, (2.5, 0.0, 3.0))

    assert scene.edge_is_live(far)
    assert not scene.edge_is_live(shared)  # the start is on the plane, so it did split
    verts = _counts(scene)[2]
    assert verts == before[2] + 1 + 12  # the start's split vertex and the arc's other 12


def test_both_ends_still_join_their_split_vertices_far_from_the_origin():
    """Inside a group translated far from the origin, the plane round-trip's
    float32 rounding exceeds the join tolerance. The arc's ends are set onto
    the resolved vertices, so both still join and the face is split."""
    wt = mat_translate([523.17, 311.41, 0.0])
    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    scene = grp_def.mesh
    fid, v = _quad(scene, size=4.0)
    grp_inst = model.new_instance(grp_def, wt)
    model.root.children.append(grp_inst)
    model.enter(grp_inst)
    tool, _stack = _make_tool(scene, model=model)
    ox, oy = 523.17, 311.41

    _draw_arc_snaps(
        tool,
        _on_edge(scene, v[0], v[1], (ox + 1.3, oy, 0), 0.325),
        _on_edge(scene, v[2], v[3], (ox + 2.9, oy + 4, 0), 0.275),
        (ox + 2.6, oy + 2.0, 0),
    )

    faces, edges, verts = _counts(scene)
    assert faces == 2
    assert not scene._mesh.face_is_live(fid)
    assert verts == 4 + 2 + _ARC_INTERIOR
    assert edges == 4 + 2 + 12


def test_a_snap_whose_edge_died_mid_gesture_is_a_plain_point_and_the_commit_does_not_raise():
    """An undo between the presses and the bulge can remove the edge a stored
    snap named. The commit used to raise KeyError after the start's split had
    already run outside the undo stack, and again on every later click."""
    scene = Scene()
    fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)
    top = scene.edge_between(v[2], v[3])
    split = SplitEdgeCommand(top, 0.5)
    split.do(scene)
    stack.push_executed(split, scene)
    half = scene.edge_between(split.new_vertex_id, v[3])
    assert half is not None

    # Start on the bottom edge, end on one half of the split top edge ...
    tool.on_mouse_press(None, _on_edge(scene, v[0], v[1], (2, 0, 0), 0.5))
    tool.on_mouse_press(None, _snap(SnapKind.ON_EDGE, (3, 4, 0), edge_id=half, edge_t=0.5))
    # ... then the user undoes the top split, which retires that half.
    stack.undo()
    assert not scene.edge_is_live(half)
    tool.on_mouse_press(None, _snap(SnapKind.GRID, (3.0, 2.0, 0)))  # the bulge

    assert not tool.has_active_gesture
    assert stack.can_undo
    # Nothing ran outside the undo stack: one undo returns to the bare quad.
    stack.undo()
    assert _counts(scene) == (1, 4, 4)
    assert scene._mesh.face_is_live(fid)
    assert stack.can_undo is False


def test_a_snap_whose_vertex_died_mid_gesture_is_a_plain_point_too():
    scene = Scene()
    _fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)

    tool.on_mouse_press(None, _snap(SnapKind.ENDPOINT, (0, 0, 0), vertex_id=9999))
    tool.on_mouse_press(None, _snap(SnapKind.ENDPOINT, (4, 4, 0), vertex_id=v[2]))
    tool.on_mouse_press(None, _snap(SnapKind.GRID, (2.2, 1.8, 0)))

    assert not tool.has_active_gesture
    assert stack.can_undo
    assert _counts(scene)[0] == 2


def test_an_edge_snap_that_moved_since_its_press_splits_nothing_it_no_longer_meets():
    """Undoing a Move between two arc clicks leaves a live edge that is no
    longer where the snap recorded it. The old check only caught dead edges,
    so the commit split that edge at the parameter's new, unrelated point."""
    scene = Scene()
    _fid, v = _quad(scene, size=4.0)
    tool, stack = _make_tool(scene)
    bottom = scene.edge_between(v[0], v[1])

    # Press the start on the bottom edge's midpoint, (2, 0, 0).
    tool.on_mouse_press(None, _on_edge(scene, v[0], v[1], (2, 0, 0), 0.5))
    # The bottom edge then moves to y = -1 (what undoing a Move does).
    scene.set_vertex_position(v[0], np.array([0.0, -1.0, 0.0], dtype=np.float32))
    scene.set_vertex_position(v[1], np.array([4.0, -1.0, 0.0], dtype=np.float32))
    tool.on_mouse_press(None, _snap(SnapKind.ENDPOINT, (4, 4, 0), vertex_id=v[2]))
    tool.on_mouse_press(None, _snap(SnapKind.GRID, (2.2, 1.8, 0)))

    assert not tool.has_active_gesture
    assert scene.edge_is_live(bottom)
    assert _live_vertices_at(scene, x=2.0, y=-1.0, z=0.0) == []
    assert stack.can_undo
    # The arc added only its own vertices and edge: one undo is the bare quad.
    stack.undo()
    assert _counts(scene) == (1, 4, 4)
    assert stack.can_undo is False


def test_an_end_is_never_pinned_onto_a_vertex_that_is_far_from_it():
    """A snap whose edge_t is degenerate (a no-op split) falls back to the
    host edge's nearest endpoint. Pinning the arc's end onto that vertex would
    drag it metres along the plane, so the pin only applies within float
    rounding: the arc still starts where it was drawn. Since the moved-edge
    check in `_is_stale` such a snap (edge_t 0.0 but the point a metre from
    that end) is stale and never reaches the pin; the pin's own tolerance is
    kept as a second line of defence."""
    scene = Scene()
    _fid, v = _quad(scene, size=4.0)
    tool, _stack = _make_tool(scene)
    bottom = scene.edge_between(v[0], v[1])

    start = _snap(SnapKind.ON_EDGE, (1, 0, 0), edge_id=bottom, edge_t=0.0)
    _draw_arc_snaps(tool, start, _snap(SnapKind.GRID, (3, 3, 0)), (1.0, 2.5, 0))

    assert len(_live_vertices_at(scene, x=1.0, y=0.0, z=0.0)) == 1
