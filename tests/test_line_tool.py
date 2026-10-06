"""Unit tests for the Line tool — three-branch click logic + ESC + < 3 close."""

from __future__ import annotations

import numpy as np


def _endpoint_snap(world, vertex_id: int):
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.ENDPOINT,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=vertex_id,
        label="Endpoint",
    )


def _grid_snap(world):
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Grid",
    )


def test_line_tool_first_click_adds_one_vertex():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _grid_snap((1.0, 0.0, 0.0)))  # type: ignore[arg-type]

    assert len(list(scene.vertices_iter())) == 1


def test_line_tool_branch_3_new_vertex_creates_edge():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((1.0, 0.0, 0.0)))  # type: ignore[arg-type]

    assert len(list(scene.vertices_iter())) == 2
    assert len(list(scene.edges_iter())) == 1
    assert len(list(scene.faces_iter())) == 0


def test_line_tool_loop_close_creates_face():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    # Click 1 — first vertex at (0,0,0).
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    first_vid = next(iter(scene.vertices_iter())).id
    # Clicks 2, 3 — extend the polyline.
    tool.on_mouse_press(None, _grid_snap((2.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 2.0, 0.0)))  # type: ignore[arg-type]
    # Click 4 — endpoint-snap onto the FIRST vertex → loop close.
    tool.on_mouse_press(None, _endpoint_snap((0.0, 0.0, 0.0), vertex_id=first_vid))  # type: ignore[arg-type]

    assert len(list(scene.vertices_iter())) == 3
    assert len(list(scene.edges_iter())) == 3
    assert len(list(scene.faces_iter())) == 1


def test_line_tool_branch_2_extend_to_existing_vertex():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    # Pre-existing vertex (e.g. drawn by a previous gesture or Rectangle).
    pre_vid = scene.add_vertex(np.array([5.0, 0.0, 0.0], dtype=np.float32))

    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _endpoint_snap((5.0, 0.0, 0.0), vertex_id=pre_vid))  # type: ignore[arg-type]

    # Only 2 vertices (the new one + the pre-existing one), 1 edge, no face.
    assert len(list(scene.vertices_iter())) == 2
    assert len(list(scene.edges_iter())) == 1
    assert len(list(scene.faces_iter())) == 0


def test_line_tool_close_with_fewer_than_three_vertices_ignored():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    # Click 1 — first vertex at (0,0,0).
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    first_vid = next(iter(scene.vertices_iter())).id
    # Click 2 — endpoint snap back onto first vertex (only 1 vertex in gesture).
    # Should be ignored — no face, no extra edge.
    tool.on_mouse_press(None, _endpoint_snap((0.0, 0.0, 0.0), vertex_id=first_vid))  # type: ignore[arg-type]

    assert len(list(scene.faces_iter())) == 0


def test_line_tool_has_active_gesture_reflects_state():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    assert tool.has_active_gesture is False

    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    assert tool.has_active_gesture is True

    first_vid = next(iter(scene.vertices_iter())).id
    tool.on_mouse_press(None, _grid_snap((2.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 2.0, 0.0)))  # type: ignore[arg-type]
    # Close the loop — should reset gesture.
    tool.on_mouse_press(None, _endpoint_snap((0.0, 0.0, 0.0), vertex_id=first_vid))  # type: ignore[arg-type]
    assert tool.has_active_gesture is False


def test_line_tool_esc_cancels_visible_gesture():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((1.0, 0.0, 0.0)))  # type: ignore[arg-type]

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    # Per spec §4.5 / §5.6: ESC clears the visible gesture state but does NOT
    # un-add already-committed vertices/edges.
    assert tool.overlay().rubber_band_segments.shape == (0, 3)


def test_line_tool_pushes_composite_at_loop_close():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))

    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    first_vid = next(iter(scene.vertices_iter())).id
    tool.on_mouse_press(None, _grid_snap((2.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 2.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _endpoint_snap((0.0, 0.0, 0.0), vertex_id=first_vid))  # type: ignore[arg-type]

    assert stack.can_undo
    assert len(list(scene.faces_iter())) == 1

    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.edges_iter())) == 0
    assert len(list(scene.faces_iter())) == 0


def test_line_tool_esc_mid_gesture_rolls_back_committed_geometry():
    """The M2 §5.6 #3 elimination test — ESC fully reverses in-progress mutations."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))

    # 3 clicks — verts + edges committed to scene as the gesture progresses.
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((1.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((1.0, 1.0, 0.0)))  # type: ignore[arg-type]
    assert len(list(scene.vertices_iter())) == 3
    assert len(list(scene.edges_iter())) == 2

    # ESC mid-gesture rolls back everything committed during this gesture.
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.edges_iter())) == 0
    # Nothing pushed to undo stack (the composite was discarded, not committed).
    assert not stack.can_undo


# ---------------------------------------------------------------------------
# Enter / Return — finish open polyline (M3d Task 12b)
# ---------------------------------------------------------------------------


def test_line_tool_enter_finishes_and_commits_open_polyline():
    """Test A — Enter commits an open 2-segment polyline as one undoable unit."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))

    # 3 clicks → 3 vertices, 2 edges, NO loop closure.
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 2.0, 0.0)))  # type: ignore[arg-type]

    assert tool.has_active_gesture is True
    assert len(list(scene.vertices_iter())) == 3
    assert len(list(scene.edges_iter())) == 2

    # Press Enter — should finish the gesture.
    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    # Tool returns to idle.
    assert tool.has_active_gesture is False

    # The polyline was registered as one undoable unit.
    assert stack.can_undo

    # A single undo removes ALL the polyline geometry.
    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.edges_iter())) == 0


def test_line_tool_enter_with_only_seed_discards():
    """Test B — Enter with only the start point placed discards it (nothing to commit)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))

    # Only the seed vertex placed — no edges yet.
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    assert tool.has_active_gesture is True

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    # Tool returns to idle.
    assert tool.has_active_gesture is False
    # Seed vertex was discarded.
    assert len(list(scene.vertices_iter())) == 0
    # Nothing pushed to undo stack.
    assert stack.can_undo is False


def _two_segment_chain():
    """A Line tool mid-chain: three clicks, two edges, nothing on the stack yet."""
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _grid_snap((2.0, 2.0, 0.0)))  # type: ignore[arg-type]
    assert stack.can_undo is False
    return tool, scene, stack


def test_line_tool_deactivate_mid_chain_commits_one_undo_step():
    """#144: switching tools keeps the drawn edges, as one undoable step."""
    tool, scene, stack = _two_segment_chain()

    tool.deactivate()

    assert tool.has_active_gesture is False
    assert len(list(scene.edges_iter())) == 2
    assert stack.can_undo
    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.edges_iter())) == 0


def test_line_tool_deactivate_with_only_seed_discards():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    scene = Scene()
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    tool.on_mouse_press(None, _grid_snap((0.0, 0.0, 0.0)))  # type: ignore[arg-type]

    tool.deactivate()

    assert len(list(scene.vertices_iter())) == 0
    assert stack.can_undo is False


def test_line_tool_commit_pending_changes_commits_and_ends_the_chain():
    tool, scene, stack = _two_segment_chain()

    tool.commit_pending_changes()

    assert tool.has_active_gesture is False
    assert stack.can_undo
    stack.undo()
    assert len(list(scene.edges_iter())) == 0


def test_line_tool_commit_pending_changes_when_idle_does_nothing():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.line_tool import LineTool

    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=Scene(), command_stack=stack))

    tool.commit_pending_changes()

    assert stack.can_undo is False
