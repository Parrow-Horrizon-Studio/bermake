"""The Offset tool (M7.4 Task 6)."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.tools.offset_tool import OffsetTool


def _square_face(window, size=4.0):
    scene = window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([size, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([size, size, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, size, 0.0], dtype=np.float32)),
    ]
    return scene.add_face_from_loop(v)


def _rectangle_face(window, width, height):
    # An asymmetric (non-square) rectangle: its bounding box would equal its
    # own vertex set under a naive "just move the bbox" offset just as much
    # as a square's would, but a wrong axis-scale bug (e.g. offsetting width
    # and height by different, swapped amounts) is caught here where it
    # would not be on a square.
    scene = window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([width, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([width, height, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, height, 0.0], dtype=np.float32)),
    ]
    return scene.add_face_from_loop(v)


def _arm(window, fid):
    window._activate("offset")
    tool = window._tool_manager._tools_by_id["offset"]
    tool._arm_face(fid)
    return tool


def test_offsetting_a_face_replaces_it_with_a_ring_and_an_inner_face(main_window):
    fid = _square_face(main_window)
    before = len(list(main_window._model.active_context.mesh.faces_iter()))
    tool = _arm(main_window, fid)
    tool._commit_offset(0.5)
    after = len(list(main_window._model.active_context.mesh.faces_iter()))
    # Source face gone, four ring quads plus one inner face.
    assert after == before - 1 + 5


def test_the_inner_face_is_smaller_than_the_source_for_a_positive_distance(main_window):
    # A broken implementation that flips the sign convention (offsets
    # outward for a positive distance, growing the inner loop instead of
    # shrinking it) would still produce the right face COUNT above, so that
    # test alone cannot catch a sign error. This asserts the direction by
    # checking the actual inner-face area, not just "some face changed" --
    # note the ring's four trapezoids are NOT candidates to confuse with the
    # inner face: on this 6x2 rectangle the two short-edge trapezoids are
    # smaller than the inner face (area 0.75 vs 5.0), so a naive
    # "smallest face" heuristic would silently check the wrong face; this
    # instead locates the inner face by its area matching the expected
    # inset exactly.
    fid = _rectangle_face(main_window, width=6.0, height=2.0)
    scene = main_window._model.active_context.mesh
    tool = _arm(main_window, fid)
    tool._commit_offset(0.5)

    def _area(face_id):
        loop = scene.face_loop(face_id)
        pts = np.array([scene.vertex(v).position for v in loop], dtype=np.float64)
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    areas = [_area(f.id) for f in scene.faces_iter()]
    source_area = 6.0 * 2.0
    # Inset by 0.5 on all sides of a 6x2 rect -> 5x1 -> area 5.0. An outward
    # (sign-flipped) offset would instead produce a 7x3 outer face (area 21),
    # which would not match this.
    assert any(area == pytest.approx(5.0 * 1.0, abs=1e-3) for area in areas)
    assert all(area < source_area for area in areas)


def _planar_area(scene, face_id):
    """Shoelace area of a face lying in the z = 0 plane."""
    loop = scene.face_loop(face_id)
    pts = np.array([scene.vertex(v).position for v in loop], dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def test_an_outward_offset_keeps_the_source_face_and_adds_no_outer_face(main_window):
    # Negative distance is fully reachable and intended: the cursor outside
    # the boundary makes _signed_distance_to_boundary return negative, and
    # offset_polygon documents negative as outward. This tool used to treat
    # both signs alike -- remove the source face, face the new loop -- which
    # outward means removing the face over the region the user did NOT
    # touch (an unfaced hole) and then covering the WHOLE 6x6 outer loop
    # with a face that overlaps all four ring quads: coplanar overlapping
    # geometry and z-fighting on top of the hole.
    #
    # Correct outward behaviour is the mirror of inward's question set: the
    # source face survives because it still covers its own region, the ring
    # is added, and the outer loop gets no face at all.
    fid = _square_face(main_window, size=4.0)
    scene = main_window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    depth = len(main_window._command_stack._undo)

    tool = _arm(main_window, fid)
    applied = tool._commit_offset(-1.0)
    # Outward never collapses, so nothing is clamped.
    assert applied == pytest.approx(-1.0)

    live_face_ids = {f.id for f in scene.faces_iter()}
    assert fid in live_face_ids, "the source face was removed, leaving an unfaced hole"
    assert len(live_face_ids) == before + 4, "expected exactly four new ring quads"

    areas = sorted(_planar_area(scene, f) for f in live_face_ids)
    # 4x4 source face kept, plus four ring trapezoids of (36 - 16) / 4 = 5.
    assert areas == pytest.approx([5.0, 5.0, 5.0, 5.0, 16.0], abs=1e-6)
    # The decisive one: nothing covers the full 6x6 outer loop.
    assert all(area != pytest.approx(36.0, abs=1e-6) for area in areas)

    # Half-edge integrity across the join. A half-edge belongs to at most
    # one face and the mesh overwrites silently rather than raising, so a
    # ring quad wound the same way round as the surviving source face would
    # quietly steal the source face's own boundary half-edge and leave the
    # shared edge with one face instead of two.
    for e in scene.face_edges(fid):
        f_a, f_b = scene.edge_faces(e)
        assert f_a is not None and f_b is not None, (
            f"boundary edge {e} of the source face is bordered by one face, not two"
        )

    assert len(main_window._command_stack._undo) == depth + 1
    main_window._command_stack.undo()
    assert len(list(scene.faces_iter())) == before


def test_an_offset_is_one_undo_step(main_window):
    fid = _rectangle_face(main_window, width=6.0, height=2.0)
    depth = len(main_window._command_stack._undo)
    tool = _arm(main_window, fid)
    tool._commit_offset(0.5)
    assert len(main_window._command_stack._undo) == depth + 1
    main_window._command_stack.undo()
    assert len(list(main_window._model.active_context.mesh.faces_iter())) == 1


def test_the_drag_clamps_rather_than_collapsing_the_face(main_window):
    # A 4x4 face collapses at 2.0. Asking for 10 must stop strictly short of
    # 2.0 (fix round, Task 6 review Finding 1: offset_polygon's clamp now
    # backs off from the exact collapse point, since a distance exactly at
    # it produces a degenerate, coincident-vertex loop), not produce
    # inverted geometry.
    fid = _square_face(main_window, size=4.0)
    tool = _arm(main_window, fid)
    applied = tool._commit_offset(10.0)
    assert applied < 2.0
    assert applied == pytest.approx(2.0, rel=1e-2)


def test_full_collapse_at_the_clamp_limit_does_not_crash(main_window):
    # offset_polygon's clamp used to be the distance at which an offset edge
    # reaches EXACTLY zero length -- for a square that meant all four
    # corners landed on the same point, and Scene.add_vertex welds
    # coincident positions to one vertex id, so a naive re-use of
    # loft_between_loops's ring-closing edge asked for a self-loop and
    # raised. The fix (Task 6 review Finding 1) moved the non-degeneracy
    # guarantee into offset_polygon itself: it now always backs off short of
    # that point, so this exercises the same "very large offset request"
    # scenario as a plain regression check -- it must still commit cleanly
    # as one undo step through the ordinary loft_between_loops path, with no
    # special-casing needed at this layer.
    fid = _square_face(main_window, size=4.0)
    depth = len(main_window._command_stack._undo)
    tool = _arm(main_window, fid)
    applied = tool._commit_offset(10.0)
    assert applied < 2.0
    assert applied == pytest.approx(2.0, rel=1e-2)
    assert len(main_window._command_stack._undo) == depth + 1

    scene = main_window._model.active_context.mesh
    # Source face gone, four ring quads plus one (now non-degenerate) inner
    # face -- not zero faces, and no exception.
    assert len(list(scene.faces_iter())) >= 1


def test_a_small_distance_is_not_clamped(main_window):
    # The clamped-distance path needs both a case that genuinely clamps
    # (above) and one that genuinely does not -- an implementation that
    # always reports a clamp would fail this one.
    fid = _square_face(main_window, size=4.0)
    tool = _arm(main_window, fid)
    applied = tool._commit_offset(0.5)
    assert applied == pytest.approx(0.5)


def test_typed_value_drives_the_offset(main_window):
    from bermake.units import Units

    fid = _square_face(main_window)
    tool = _arm(main_window, fid)
    assert tool.apply_typed_value("500 mm", Units()) is True


def test_offset_has_no_option_bar(main_window):
    # SketchUp's Offset has no options, so arming it must not steal the tab.
    main_window._properties_dock.show_tab("entity_info")
    main_window._activate("offset")
    assert main_window._properties_dock.current_tab_id == "entity_info"


def test_offset_tool_id_and_shortcut():
    tool = OffsetTool()
    assert tool.id == "offset"
    assert tool.shortcut == "F"


def test_status_text_reports_the_applied_distance_not_the_requested_one(main_window):
    # Fix round (Task 6 review, Finding 2): the ghost overlay already stops
    # growing at the collapse limit, but the status bar used to keep showing
    # the raw drag distance -- a number that no longer matched the geometry.
    # Past a 4x4 square's ~2.0 limit, the text must reflect what committing
    # NOW would actually apply.
    from bermake.tools.sweep_support import offset_polygon
    from bermake.units import format_length

    fid = _square_face(main_window, size=4.0)
    tool = _arm(main_window, fid)
    tool._current_distance = 10.0

    _, expected_applied = offset_polygon(
        tool._armed_face_points, tool._armed_face_normal, tool._current_distance
    )
    assert expected_applied < 10.0  # sanity: this scenario really does clamp

    expected_text = f"offset: {format_length(expected_applied, main_window._doc.units)}"
    assert tool.status_text == expected_text
    # A test that would fail if the tool reported the requested distance
    # instead: the raw 10.0 must not appear anywhere in the status text.
    raw_text = f"offset: {format_length(10.0, main_window._doc.units)}"
    assert tool.status_text != raw_text
