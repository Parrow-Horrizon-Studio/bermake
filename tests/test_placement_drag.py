"""M7.5b Task 12: dragging a texture across a face."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Side, TexturePlacement


def test_a_drag_shifts_the_offset_of_the_face_under_the_cursor(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    tool = win._paint_tool

    tool.begin_placement_drag(f, Side.FRONT)
    tool.update_placement_drag(du=0.25, dv=-0.5)
    tool.end_placement_drag()

    p = scene.face_placement(f)
    assert p.offset_u == 0.25
    assert p.offset_v == -0.5


def test_one_drag_is_one_undo_step(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    tool = win._paint_tool
    depth = len(win._command_stack._undo)

    tool.begin_placement_drag(f, Side.FRONT)
    for _ in range(5):
        tool.update_placement_drag(du=0.1, dv=0.0)
    tool.end_placement_drag()

    assert len(win._command_stack._undo) == depth + 1
    win._command_stack.undo()
    assert scene.face_placement(f) == TexturePlacement()


def test_the_drag_accumulates_from_the_placement_it_started_with(main_window_with_square):
    # Starting from an already-adjusted face must offset FROM that value, not
    # from the identity, or the texture jumps on the first mouse move.
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    scene.set_face_placement(f, TexturePlacement(offset_u=1.0))
    tool = win._paint_tool

    tool.begin_placement_drag(f, Side.FRONT)
    tool.update_placement_drag(du=0.5, dv=0.0)
    tool.end_placement_drag()
    assert scene.face_placement(f).offset_u == 1.5


def test_the_drag_edits_the_side_it_was_given(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    tool = win._paint_tool

    tool.begin_placement_drag(f, Side.BACK)
    tool.update_placement_drag(du=0.25, dv=0.0)
    tool.end_placement_drag()
    assert scene.face_placement(f, Side.BACK).offset_u == 0.25
    assert scene.face_placement(f, Side.FRONT) == TexturePlacement()


def test_a_drag_that_moved_nothing_pushes_nothing(main_window_with_square):
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    tool = win._paint_tool
    depth = len(win._command_stack._undo)

    tool.begin_placement_drag(f, Side.FRONT)
    tool.end_placement_drag()
    assert len(win._command_stack._undo) == depth


def test_the_tool_reports_an_active_gesture_during_the_drag(main_window_with_square):
    # M7.5a shipped Paint with has_active_gesture hardcoded False, and review
    # found a right-click mid-stroke popped the context menu instead of
    # cancelling. Same hazard here.
    win = main_window_with_square
    scene = win._model.active_context.mesh
    f = next(iter(scene.faces_iter())).id
    tool = win._paint_tool

    assert tool.has_active_gesture is False
    tool.begin_placement_drag(f, Side.FRONT)
    assert tool.has_active_gesture is True
    tool.end_placement_drag()
    assert tool.has_active_gesture is False
