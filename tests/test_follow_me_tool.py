"""The Follow Me tool (M7.4 Task 8)."""

from __future__ import annotations

import numpy as np

from bermake.geometry.transforms import apply_mat
from bermake.tools.sweep_support import sweep_stations


def _profile_and_path(window):
    """A 1x1 profile at the origin, and an L-shaped 3-point path."""
    scene = window._model.active_context.mesh
    p = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]
    profile = scene.add_face_from_loop(p)
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.0, 5.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([5.0, 5.0, 0.0], dtype=np.float32))
    e1 = scene.add_edge(a, b)
    e2 = scene.add_edge(b, c)
    return profile, [e1, e2]


def _tool(window):
    return window._tool_manager._tools_by_id["follow_me"]


def test_sweeping_a_profile_along_a_path_adds_geometry(main_window):
    profile, path = _profile_and_path(main_window)
    before = len(list(main_window._model.active_context.mesh.faces_iter()))
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=path)
    tool._commit_sweep(profile)
    after = len(list(main_window._model.active_context.mesh.faces_iter()))
    assert after > before


def test_a_sweep_is_one_undo_step(main_window):
    profile, path = _profile_and_path(main_window)
    depth = len(main_window._command_stack._undo)
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=path)
    tool._commit_sweep(profile)
    assert len(main_window._command_stack._undo) == depth + 1


def test_an_unordered_edge_selection_is_still_walked_into_a_chain(main_window):
    # The selection is a set. Reversing it must not change the result.
    profile, path = _profile_and_path(main_window)
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    ordered = tool._order_path(list(reversed(path)))
    assert ordered is not None
    assert len(ordered) == 3  # three vertices for two edges, open
    assert tool._path_closed is False


def test_a_forked_edge_selection_is_refused(main_window):
    scene = main_window._model.active_context.mesh
    hub = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    arms = [
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]
    forked = [scene.add_edge(hub, a) for a in arms]
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    assert tool._order_path(forked) is None


def test_two_disjoint_runs_are_refused_even_though_neither_forks(main_window):
    # Two disjoint closed triangles: every vertex has degree 2 and there are
    # zero degree-one vertices overall, EXACTLY the signature _order_path
    # otherwise takes as "a single closed loop". A broken implementation
    # that classifies open/closed by degree counts alone, without first
    # checking that everything selected is one connected run, would
    # misread this as a valid (if strange) closed path and walk only one
    # triangle -- silently dropping the other -- instead of refusing the
    # whole ambiguous selection.
    scene = main_window._model.active_context.mesh
    a0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    a1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    a2 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    b0 = scene.add_vertex(np.array([9.0, 0.0, 0.0], dtype=np.float32))
    b1 = scene.add_vertex(np.array([10.0, 0.0, 0.0], dtype=np.float32))
    b2 = scene.add_vertex(np.array([9.0, 1.0, 0.0], dtype=np.float32))
    edges = [
        scene.add_edge(a0, a1),
        scene.add_edge(a1, a2),
        scene.add_edge(a2, a0),
        scene.add_edge(b0, b1),
        scene.add_edge(b1, b2),
        scene.add_edge(b2, b0),
    ]
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    assert tool._order_path(edges) is None


def test_a_refused_sweep_reports_and_emits_nothing(main_window):
    scene = main_window._model.active_context.mesh
    profile, path = _profile_and_path(main_window)
    before = len(list(scene.faces_iter()))
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=[path[0]])  # single edge, no corner
    # Force refusal by handing the tool a fork.
    hub = scene.add_vertex(np.array([9.0, 9.0, 9.0], dtype=np.float32))
    arms = [
        scene.add_vertex(np.array([10.0, 9.0, 9.0], dtype=np.float32)),
        scene.add_vertex(np.array([9.0, 10.0, 9.0], dtype=np.float32)),
        scene.add_vertex(np.array([9.0, 9.0, 10.0], dtype=np.float32)),
    ]
    main_window._selection.replace(edges=[scene.add_edge(hub, a) for a in arms])
    tool._commit_sweep(profile)
    # Refused: no new faces beyond the vertices/edges the fixture added.
    assert len(list(scene.faces_iter())) == before


def test_a_fork_refusal_reports_a_message_in_the_status_bar(main_window):
    profile, path = _profile_and_path(main_window)
    scene = main_window._model.active_context.mesh
    hub = scene.add_vertex(np.array([9.0, 9.0, 9.0], dtype=np.float32))
    arms = [
        scene.add_vertex(np.array([10.0, 9.0, 9.0], dtype=np.float32)),
        scene.add_vertex(np.array([9.0, 10.0, 9.0], dtype=np.float32)),
        scene.add_vertex(np.array([9.0, 9.0, 10.0], dtype=np.float32)),
    ]
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=[scene.add_edge(hub, a) for a in arms])
    main_window._status_bar.set_message("")
    tool._commit_sweep(profile)
    assert main_window._status_bar._message != ""


def test_a_too_tight_corner_is_refused_via_sweep_stations_and_reported(main_window):
    # Same proportions test_sweep_stations.py proves raise SweepRefused: a
    # 10-unit-wide profile cannot survive a hairpin after a 1-unit segment.
    # This exercises the OTHER refusal path (a real corner, not a fork) --
    # a tool that only ever catches forks and never SweepRefused would pass
    # every test above but fail this one.
    scene = main_window._model.active_context.mesh
    p = [
        scene.add_vertex(np.array([-5.0, -5.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([5.0, -5.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([5.0, 5.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([-5.0, 5.0, 0.0], dtype=np.float32)),
    ]
    profile = scene.add_face_from_loop(p)
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([0.02, 0.01, 0.0], dtype=np.float32))
    edges = [scene.add_edge(a, b), scene.add_edge(b, c)]

    before_faces = len(list(scene.faces_iter()))
    depth = len(main_window._command_stack._undo)

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    main_window._status_bar.set_message("")
    tool._commit_sweep(profile)

    assert len(list(scene.faces_iter())) == before_faces
    assert len(main_window._command_stack._undo) == depth
    assert "corner" in main_window._status_bar._message.lower()


def test_a_sharp_but_survivable_corner_still_sweeps(main_window):
    # The other direction of the refusal test: a real (not fork) corner
    # that is NOT too tight must go through. Reusing the fixture's mild
    # 90-degree / 1x1-profile corner would not probe this at all -- Task 7's
    # refusal criterion (SweepRefused: a profile vertex's required
    # slide-to-miter-plane distance exceeds the adjoining segment's own
    # length) is nowhere near tripped by a 90-degree bend on a 5-unit
    # segment, so a sloppy or over-conservative refusal threshold would
    # never be exercised by it.
    #
    # This fixture instead bends two 5-unit segments by 168.5 degrees (i.e.
    # only 11.5 degrees short of the path folding straight back on itself),
    # against the same 1x1 profile square used elsewhere in this file. For
    # this profile/segment-length pair, the required slide distance is
    # ~99.3% of the 5-unit segment length -- independently verified by
    # replicating _mitered_station's own slide formula outside the tool:
    # the refusal boundary for this exact profile and segment length sits
    # at a 168.579-degree bend (slide == 100.00% of the segment), so 168.5
    # degrees clears it by well under a tenth of a degree, sitting at 99.3%
    # of the way to refusal. A correct implementation accepts this; a
    # sloppy or over-conservative threshold (e.g. a fixed angle cutoff, or
    # one that doesn't scale with segment length) would very plausibly
    # refuse it.
    scene = main_window._model.active_context.mesh
    p = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 1.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]
    profile = scene.add_face_from_loop(p)
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.0, 5.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([0.9968396721, 0.1003764769, 0.0], dtype=np.float32))
    path = [scene.add_edge(a, b), scene.add_edge(b, c)]

    before = len(list(scene.faces_iter()))
    depth = len(main_window._command_stack._undo)

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=path)
    tool._commit_sweep(profile)

    assert len(list(scene.faces_iter())) > before
    assert len(main_window._command_stack._undo) == depth + 1


def test_a_closed_path_sweeps_into_a_lathe_with_no_caps(main_window):
    # Closed paths differ from open ones in station count (Task 7) and in
    # needing a seam back to the start instead of end caps. A broken
    # implementation that always caps (or that forgets to close the seam,
    # leaving a gap) would produce the wrong face count here even though
    # every open-path test above still passes.
    scene = main_window._model.active_context.mesh
    profile, _unused_path = _profile_and_path(main_window)
    p0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    p1 = scene.add_vertex(np.array([4.0, 0.0, 0.0], dtype=np.float32))
    p2 = scene.add_vertex(np.array([4.0, 4.0, 0.0], dtype=np.float32))
    p3 = scene.add_vertex(np.array([0.0, 4.0, 0.0], dtype=np.float32))
    edges = [
        scene.add_edge(p0, p1),
        scene.add_edge(p1, p2),
        scene.add_edge(p2, p3),
        scene.add_edge(p3, p0),
    ]
    before = len(list(scene.faces_iter()))
    before_vertices = len(list(scene.vertices_iter()))
    depth = len(main_window._command_stack._undo)

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    tool._commit_sweep(profile)

    # 4 profile vertices x 4 path edges = 16 new side quads, no caps, minus
    # the one removed source face. Face count alone can't tell "welded onto
    # the original loop" apart from "left a duplicate coincident ring" --
    # both shapes have 16 quads. What pins the weld is the vertex count:
    # only 3 of the 4 loft segments mint a fresh ring (12 new vertices);
    # the segment that closes the seam back to the start must land on the
    # profile's OWN original loop vertices (Scene.add_vertex is idempotent
    # on exact float32 match) rather than minting a 4th, merely-coincident
    # ring, which would show up here as +16 instead of +12.
    assert len(list(scene.faces_iter())) == before + 15
    assert len(list(scene.vertices_iter())) == before_vertices + 12
    assert len(main_window._command_stack._undo) == depth + 1


def test_reversing_a_closed_selection_still_finds_the_cycle(main_window):
    scene = main_window._model.active_context.mesh
    p0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    p1 = scene.add_vertex(np.array([4.0, 0.0, 0.0], dtype=np.float32))
    p2 = scene.add_vertex(np.array([4.0, 4.0, 0.0], dtype=np.float32))
    edges = [
        scene.add_edge(p0, p1),
        scene.add_edge(p1, p2),
        scene.add_edge(p2, p0),
    ]
    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    ordered = tool._order_path(list(reversed(edges)))
    assert ordered is not None
    assert len(ordered) == 3
    assert tool._path_closed is True


def test_an_asymmetric_profile_lands_correctly_after_a_right_angle_turn(main_window):
    # A square profile or a straight path hides a wrong rotation through
    # symmetry -- and the fixtures above use exactly that shape. A scalene
    # right triangle swept around a real 90-degree corner has no such
    # symmetry: a bug that never rotates the profile, mirrors it, or applies
    # the station transform with the wrong (row- vs column-vector)
    # convention lands its vertices somewhere a correct implementation
    # never does. The expected position is derived independently, from the
    # same public functions the tool is supposed to be wiring together
    # (sweep_stations + apply_mat), so this exercises the WIRING between
    # _order_path/_commit_sweep and those functions, not the maths itself
    # (already covered by test_sweep_stations.py).
    scene = main_window._model.active_context.mesh
    p = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([3.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ]
    profile_pts = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    profile = scene.add_face_from_loop(p)

    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.0, 5.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([5.0, 5.0, 0.0], dtype=np.float32))
    edges = [scene.add_edge(a, b), scene.add_edge(b, c)]
    path_points = np.array(
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float64
    )

    expected_final = apply_mat(
        profile_pts, sweep_stations(profile_pts, path_points, closed=False)[-1]
    )

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    tool._commit_sweep(profile)

    live_positions = [v.position for v in scene.vertices_iter()]
    for expected_pt in expected_final:
        assert any(
            np.allclose(pos, expected_pt, atol=1e-3) for pos in live_positions
        ), f"expected a swept vertex near {expected_pt}, found none"


# ---------------------------------------------------------------------------
# The lathe: a profile held OFF the path axis (spec 1.6). Everything above
# uses a profile sitting on the path start, where a station that recentres
# the profile onto the path is indistinguishable from one that carries its
# offset along. These two fixtures put the profile 3 units to one side, which
# is the whole point of the feature and the only configuration that can see
# the difference.
# ---------------------------------------------------------------------------

# Asymmetric trapezoid in the x = 0 plane, 3 units off the path in +Y. No
# mirror symmetry in either in-plane axis, so a wrong orientation shows up as
# a wrong position rather than cancelling out.
_LATHE_PROFILE = [
    (0.0, 3.0, 0.0),
    (0.0, 4.0, 0.0),
    (0.0, 4.0, 1.0),
    (0.0, 3.0, 2.0),
]


def _lathe_profile_face(scene):
    ids = [scene.add_vertex(np.array(p, dtype=np.float32)) for p in _LATHE_PROFILE]
    return scene.add_face_from_loop(ids), np.array(_LATHE_PROFILE, dtype=np.float64)


def _has_vertex_at(scene, point, atol=1e-4):
    return any(np.allclose(v.position, point, atol=atol) for v in scene.vertices_iter())


def test_an_offset_profile_sweeps_into_a_lathe_not_onto_the_path_centreline(main_window):
    # The first tube segment is the one Follow Me builds from the SOURCE
    # loop as drawn, so it is the segment where a station that recentres
    # the profile onto the path does its worst damage: the source loop
    # stays 3 units off-axis while the ring it lofts to lands on the
    # centreline, and the segment is a gross lateral skew.
    #
    # Expected ring positions are derived from the corner geometry, not
    # observed: the miter plane at (5, 0, 0) bisecting +X and +Y is
    # x + y == 5, and a 90-degree corner slides each profile vertex along
    # the segments without changing its distance from the path axis, so
    # the ring keeps the profile's own y = 3..4 and z = 0..2 and takes
    # x = 5 - y.
    scene = main_window._model.active_context.mesh
    profile, _pts = _lathe_profile_face(scene)
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([5.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([5.0, 5.0, 0.0], dtype=np.float32))
    edges = [scene.add_edge(a, b), scene.add_edge(b, c)]

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    tool._commit_sweep(profile)

    for expected in [(2.0, 3.0, 0.0), (1.0, 4.0, 0.0), (1.0, 4.0, 1.0), (2.0, 3.0, 2.0)]:
        assert _has_vertex_at(scene, expected), (
            f"first-segment ring vertex missing at {expected}: the profile's own "
            "offset from the path was not carried into the sweep"
        )
    # The source loop is untouched for an open path (station 0 is the
    # identity), so it must still be exactly where the user drew it.
    for original in _LATHE_PROFILE:
        assert _has_vertex_at(scene, original)


def test_a_closed_path_materialises_the_mitered_seam_cross_section(main_window):
    # Finding 3: for a closed path, station 0 IS the seam miter and must be
    # applied. Skipping it terminates the closing segment on the profile as
    # drawn -- a cross-section whose plane is perpendicular to its own
    # direction of travel at the seam, so the last segment's quads cross
    # each other. Face and vertex counts are identical either way, which is
    # exactly why the shipped closed-path test could not see it.
    #
    # The seam plane bisects the closing segment (0, -1, 0) and the first
    # (1, 0, 0) through the path's start, i.e. x == y; the profile drawn in
    # the x = 0 plane at y = 3..4 is sheared onto it at (3, 3) and (4, 4).
    scene = main_window._model.active_context.mesh
    profile, _pts = _lathe_profile_face(scene)
    p = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([12.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([12.0, 12.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 12.0, 0.0], dtype=np.float32)),
    ]
    edges = [scene.add_edge(p[i], p[(i + 1) % 4]) for i in range(4)]

    before_faces = len(list(scene.faces_iter()))
    before_vertices = len(list(scene.vertices_iter()))

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    tool._commit_sweep(profile)

    seam = [(3.0, 3.0, 0.0), (4.0, 4.0, 0.0), (4.0, 4.0, 1.0), (3.0, 3.0, 2.0)]
    for expected in seam:
        assert _has_vertex_at(scene, expected), (
            f"seam ring vertex missing at {expected}: the closing segment is "
            "terminating on the un-mitered profile"
        )
    # The seam ring is the source loop MOVED onto the miter plane, not a
    # fifth ring minted beside it, so nothing is left behind at the raw
    # profile positions and the tube gains no duplicate ring.
    for original in _LATHE_PROFILE:
        assert not _has_vertex_at(scene, original), (
            f"un-mitered profile vertex {original} survives the closed sweep"
        )
    # Same counts as the shipped closed-path test: 4 rings x 4 vertices,
    # three of them minted (+12) and the fourth being the moved source
    # loop, and 16 side quads less the removed source face.
    assert len(list(scene.faces_iter())) == before_faces + 15
    assert len(list(scene.vertices_iter())) == before_vertices + 12


def test_tool_identity(main_window):
    tool = _tool(main_window)
    assert tool.id == "follow_me"
    assert tool.shortcut == ""


def test_arming_from_its_action_checks_and_sets_the_cursor(qtbot, main_window):
    # Follow Me is the first tool to ship with no shortcut at all (M7.4
    # Task 5's registry re-key is what makes that legal) -- confirm the
    # toolbar/menu checked state and viewport cursor sync for it, end to
    # end, exactly as they would for a shortcut-carrying tool.
    from bermake.ui import cursors

    main_window._activate("follow_me")
    assert main_window._actions["tool_follow_me"].isChecked()
    hotspot = main_window._viewport.cursor().hotSpot()
    assert (hotspot.x(), hotspot.y()) == cursors.ARROW_HOTSPOT


def test_undoing_a_sweep_restores_the_path_it_was_swept_along(main_window):
    # The ordinary SketchUp gesture puts a profile corner ON the path's start
    # point, so the fixture's profile vertex (0,0,0) and the path's start
    # vertex are literally the same scene vertex. sweep_stations maps
    # path_points[0] -> path_points[i] exactly, so the lofted rings land on
    # the path's own vertices and Scene.add_vertex/add_edge dedup onto them.
    # Undo must not treat that borrowed geometry as its own and delete the
    # user's path. Assert the full vertex and edge SETS, not just counts --
    # a count check passes while ids silently churn.
    scene = main_window._model.active_context.mesh
    profile, path = _profile_and_path(main_window)
    before_verts = {v.id for v in scene.vertices_iter()}
    before_edges = {e.id for e in scene.edges_iter()}

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=path)
    tool._commit_sweep(profile)
    assert len(list(scene.vertices_iter())) > len(before_verts)

    assert main_window._command_stack.undo()
    assert {v.id for v in scene.vertices_iter()} == before_verts, (
        "undoing the sweep deleted vertices it did not create"
    )
    assert {e.id for e in scene.edges_iter()} == before_edges, (
        "undoing the sweep deleted edges it did not create"
    )


def test_undoing_a_closed_sweep_restores_the_scene_it_started_from(main_window):
    # The closed (lathe) path deliberately welds its seam ring back onto the
    # profile's OWN loop, so every seam-segment vertex and edge command dedups
    # onto geometry it did not create -- and undoing those removals used to
    # tear the source face's own boundary out from under it, raising
    # "remove_edge: edge N still bordered by a face" mid-unwind. Ownership
    # tracking makes the seam commands leave the borrowed ids alone, so the
    # whole gesture unwinds cleanly: same vertices, edges, faces AND positions.
    scene = main_window._model.active_context.mesh
    profile, _unused = _profile_and_path(main_window)
    p0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    p1 = scene.add_vertex(np.array([4.0, 0.0, 0.0], dtype=np.float32))
    p2 = scene.add_vertex(np.array([4.0, 4.0, 0.0], dtype=np.float32))
    p3 = scene.add_vertex(np.array([0.0, 4.0, 0.0], dtype=np.float32))
    edges = [
        scene.add_edge(p0, p1),
        scene.add_edge(p1, p2),
        scene.add_edge(p2, p3),
        scene.add_edge(p3, p0),
    ]
    before_verts = {v.id: tuple(v.position) for v in scene.vertices_iter()}
    before_edges = {e.id for e in scene.edges_iter()}
    before_faces = {f.id for f in scene.faces_iter()}

    tool = _tool(main_window)
    tool.activate(main_window._tool_context())
    main_window._selection.replace(edges=edges)
    tool._commit_sweep(profile)

    assert main_window._command_stack.undo()
    assert {v.id: tuple(v.position) for v in scene.vertices_iter()} == before_verts
    assert {e.id for e in scene.edges_iter()} == before_edges
    assert {f.id for f in scene.faces_iter()} == before_faces
