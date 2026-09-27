"""The four primitive tools (M7.4 Task 11)."""

from __future__ import annotations

import numpy as np
import pytest

IDS = ["box", "cylinder", "cone", "sphere"]

# Exact face count each tool adds for a 2x2x2 commit at its default
# segments/rings (24 and 12 respectively -- see primitive_tool.py). These
# values differ per tool by construction (box is always 6; the others
# depend on segments/rings), so an assertion against them can tell a broken
# implementation that ignores which subclass is active (e.g. one that always
# builds a box) apart from a correct one -- unlike a bare "some geometry
# appeared" check, which any of the four would satisfy identically.
_EXPECTED_FACES_ADDED = {
    "box": 6,
    "cylinder": 26,  # 24 sides + top + bottom
    "cone": 25,  # 24 sides + base
    "sphere": 288,  # 2*24 pole fans + (12 - 2)*24 interior bands
}


@pytest.mark.parametrize("tool_id", IDS)
def test_each_primitive_is_registered_without_a_shortcut(main_window, tool_id):
    tool = main_window._tool_manager._tools_by_id[tool_id]
    assert tool.shortcut in ("", None)


@pytest.mark.parametrize("tool_id", IDS)
def test_placing_a_primitive_adds_geometry_in_one_undo_step(main_window, tool_id):
    scene = main_window._model.active_context.mesh
    before_faces = len(list(scene.faces_iter()))
    depth = len(main_window._command_stack._undo)
    tool = main_window._tool_manager._tools_by_id[tool_id]
    tool.activate(main_window._tool_context())
    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)
    assert len(list(scene.faces_iter())) > before_faces
    assert len(main_window._command_stack._undo) == depth + 1


@pytest.mark.parametrize("tool_id", IDS)
def test_undo_removes_the_whole_primitive(main_window, tool_id):
    scene = main_window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    tool = main_window._tool_manager._tools_by_id[tool_id]
    tool.activate(main_window._tool_context())
    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)
    main_window._command_stack.undo()
    assert len(list(scene.faces_iter())) == before


@pytest.mark.parametrize("tool_id", IDS)
def test_arming_a_primitive_switches_to_tool_settings(main_window, tool_id):
    main_window._properties_dock.show_tab("entity_info")
    main_window._activate(tool_id)
    assert main_window._properties_dock.current_tab_id == "tool_settings"


def test_segment_count_reaches_the_generated_geometry(main_window):
    scene = main_window._model.active_context.mesh
    tool = main_window._tool_manager._tools_by_id["cylinder"]
    tool.activate(main_window._tool_context())
    tool.segments = 6
    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)
    # Six sides plus two caps.
    assert len(list(scene.faces_iter())) == 8


def test_add_bar_refuses_a_duplicate_key(main_window):
    # Deferred as Minor in M7.3 with three call sites. There are now seven.
    page = main_window._tool_settings_page
    from PySide6.QtWidgets import QLabel

    with pytest.raises(KeyError):
        page.add_bar("box", QLabel("second"))


# ---- Test discrimination beyond the baseline above ----------------------


@pytest.mark.parametrize("tool_id", IDS)
def test_each_primitive_adds_its_own_distinct_face_count(main_window_with_square, tool_id):
    # main_window_with_square starts from ONE pre-existing face, not an
    # empty scene, so "count went up" and "back to the starting count" can't
    # pass by accident, and the exact expected count differs per tool_id --
    # a box is not a cylinder, and this is the assertion that can tell.
    window = main_window_with_square
    scene = window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    tool = window._tool_manager._tools_by_id[tool_id]
    tool.activate(window._tool_context())
    # Placed away from the fixture's own unit square at (0,0)-(1,1): calling
    # _commit_primitive directly bypasses the drag gesture that would
    # otherwise set this, and an un-set (None) base_center places the
    # primitive at the origin, exactly overlapping the fixture's square.
    tool._base_center = np.array([10.0, 10.0, 0.0])
    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)
    added = len(list(scene.faces_iter())) - before
    assert added == _EXPECTED_FACES_ADDED[tool_id]


@pytest.mark.parametrize("tool_id", IDS)
def test_undo_returns_to_the_pre_existing_non_empty_scene(main_window_with_square, tool_id):
    window = main_window_with_square
    scene = window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    assert before > 0  # guards this test against a silently-empty fixture
    tool = window._tool_manager._tools_by_id[tool_id]
    tool.activate(window._tool_context())
    # Same placement note as above: away from the fixture's square so undo
    # removes a clean, unwelded primitive rather than tripping over a
    # vertex shared with surviving geometry.
    tool._base_center = np.array([10.0, 10.0, 0.0])
    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)
    window._command_stack.undo()
    assert len(list(scene.faces_iter())) == before


def test_segments_below_the_floor_are_clamped_not_raised(main_window):
    # bermake._core.make_cylinder raises ValueError below segments=3 (M7.4
    # Task 9). Setting the tool's own attribute below that floor must not
    # let that exception reach the caller -- it should be clamped before
    # _make_mesh ever calls the generator.
    scene = main_window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    tool = main_window._tool_manager._tools_by_id["cylinder"]
    tool.activate(main_window._tool_context())
    tool.segments = 1
    assert tool.segments == 3  # clamped, not stored as-is

    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)  # must not raise

    assert len(list(scene.faces_iter())) - before == 5  # 3 sides + 2 caps


def test_rings_below_the_floor_are_clamped_not_raised(main_window):
    # Same floor-clamp contract, but for the sphere's second option field
    # (make_sphere raises ValueError below rings=2).
    scene = main_window._model.active_context.mesh
    before = len(list(scene.faces_iter()))
    tool = main_window._tool_manager._tools_by_id["sphere"]
    tool.activate(main_window._tool_context())
    tool.rings = 0
    tool.segments = 4
    assert tool.rings == 2  # clamped, not stored as-is

    tool._commit_primitive(width=2.0, depth_=2.0, height=2.0)  # must not raise

    # 2 pole fans * 4 segments, no interior bands.
    assert len(list(scene.faces_iter())) - before == 8


@pytest.mark.parametrize("tool_id", IDS)
def test_a_degenerate_dimension_raises_and_leaves_the_scene_untouched(
    main_window_with_square, tool_id
):
    # The gesture guards (`_MIN_FOOTPRINT` in `_commit_footprint`,
    # `_MIN_HEIGHT` in `on_mouse_press`) keep a zero width/depth/height from
    # ever reaching `_commit_primitive` through normal interaction -- there
    # is no shortcut, typed-value path, or other reachable way for a user to
    # arrive here with a degenerate dimension. `_commit_primitive` itself,
    # called directly (as every test in this file already does, bypassing
    # the gesture), has no such guard.
    #
    # Confirmed by hand: `make_box`/`make_cylinder`/`make_cone`/`make_sphere`
    # (M7.4 Task 9) raise `ValueError` themselves the moment a dimension
    # collapses two vertices onto each other (a self-loop edge), before
    # `_commit_primitive` ever calls `build_mesh_into_scene` -- so no
    # command is built and nothing is pushed to the stack. Since a bare
    # `ValueError` here can never reach an actual user (the path is
    # unreachable through the UI), letting it propagate is the honest
    # choice: catching and "reporting" it would only paper over a bug in
    # the gesture guards above, were one ever introduced. This test pins
    # that decision and the atomicity it depends on -- both would fail if a
    # future change (e.g. building the mesh in two steps, or pushing before
    # validating) leaked a partial primitive into the scene or the undo
    # stack on this path.
    window = main_window_with_square
    scene = window._model.active_context.mesh
    before_vertices = len(list(scene.vertices_iter()))
    before_edges = len(list(scene.edges_iter()))
    before_faces = len(list(scene.faces_iter()))
    depth = len(window._command_stack._undo)
    tool = window._tool_manager._tools_by_id[tool_id]
    tool.activate(window._tool_context())
    # Away from the fixture's own square, same as the tests above -- not
    # load-bearing here (the generator raises before any welding would
    # happen), but kept consistent so this test isn't the odd one out.
    tool._base_center = np.array([10.0, 10.0, 0.0])

    with pytest.raises(ValueError):
        tool._commit_primitive(width=0.0, depth_=2.0, height=2.0)

    assert len(list(scene.vertices_iter())) == before_vertices
    assert len(list(scene.edges_iter())) == before_edges
    assert len(list(scene.faces_iter())) == before_faces
    assert len(window._command_stack._undo) == depth


def test_arming_box_from_its_action_checks_the_toolbar_and_sets_the_cursor(main_window):
    # All four primitives are shortcut-less (M7.4 Task 5's id-keyed registry
    # is what makes that legal); _activate's toolbar/cursor sync must not
    # depend on a shortcut existing.
    main_window._activate("box")
    assert main_window._actions["tool_box"].isChecked()
    assert not main_window._viewport.cursor().pixmap().isNull()


def test_the_sphere_bar_writes_segments_and_rings_to_its_tool(main_window):
    main_window._activate("sphere")
    bar = main_window._sphere_options_bar

    bar._segments_spin.setValue(9)
    bar._rings_spin.setValue(5)

    assert main_window._sphere_tool.segments == 9
    assert main_window._sphere_tool.rings == 5


def test_the_box_bar_has_no_numeric_fields(main_window):
    # Box's footprint and height come entirely from the drag gesture, not a
    # typed value -- unlike the other three, its bar carries neither field.
    bar = main_window._box_options_bar
    assert bar._segments_spin is None
    assert bar._rings_spin is None
