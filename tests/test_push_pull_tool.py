"""PushPullTool — state machine + depth metric + composite-building tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent


# ---- helpers ---------------------------------------------------------------


def _make_event(pos=(100.0, 100.0), button=Qt.MouseButton.LeftButton, kind=QMouseEvent.Type.MouseMove):
    return QMouseEvent(
        kind,
        QPointF(*pos),
        QPointF(*pos),
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )


def _make_tool_with_unit_rect():
    """Spin up a fresh Scene + PushPullTool with one unit-rect face at z=0.

    The tool is activated against a context with a mock camera + widget sizer
    that we can drive directly.
    """
    from bermake.scene import Scene
    from bermake.tools.push_pull_tool import PushPullTool
    from bermake.tools.tool import ToolContext

    scene = Scene()
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    f = scene.add_face_from_loop([v0, v1, v2, v3])

    # Mock camera that ray_from_screen always returns "directly down from (0.5, 0.5, 5)".
    camera = MagicMock()
    camera.ray_from_screen.return_value = (
        np.array([0.5, 0.5, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )

    cmd_stack = MagicMock()

    tool = PushPullTool()
    tool.activate(
        ToolContext(
            scene=scene,
            command_stack=cmd_stack,
            camera=camera,
            widget_size_provider=lambda: (800, 600),
        )
    )
    return tool, scene, f, camera, cmd_stack


# ---- identity ---------------------------------------------------------------


class TestPushPullIdentity:
    def test_name_and_shortcut(self):
        from bermake.tools.push_pull_tool import PushPullTool

        tool = PushPullTool()
        assert tool.name == "Push/Pull"
        assert tool.shortcut == "P"


# ---- IDLE / HOVERING -------------------------------------------------------


class TestPushPullHovering:
    def test_starts_in_idle_state(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        assert tool.has_active_gesture is False
        overlay = tool.overlay()
        assert overlay.face_fill_polygons == []

    def test_mouse_move_over_face_transitions_to_hovering(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        # The mock camera ray hits the rectangle face.
        tool.on_mouse_move(_make_event(), snap=MagicMock())
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 1
        loop = overlay.face_fill_polygons[0]
        assert loop.shape == (4, 3)
        # Should be the rectangle's 4 corners at z=0 (any order).
        np.testing.assert_allclose(
            sorted(map(tuple, loop.tolist())),
            sorted([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]),
            atol=1e-5,
        )
        assert tool.has_active_gesture is False

    def test_mouse_move_off_face_transitions_back_to_idle(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=MagicMock())
        assert tool.overlay().face_fill_polygons != []
        # Move OFF — camera ray now misses.
        camera.ray_from_screen.return_value = (
            np.array([5.0, 5.0, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=MagicMock())
        assert tool.overlay().face_fill_polygons == []

    def test_status_text_is_none_in_idle(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        assert tool.status_text is None

    def test_status_text_is_none_in_hovering(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=MagicMock())
        assert tool.status_text is None


class TestPushPullArmingAndDepth:
    """HOVERING → DRAGGING transition + line-line CPA depth metric."""

    def test_click_in_hovering_arms_the_face(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        # Enter HOVERING
        tool.on_mouse_move(_make_event(), snap=None)
        # Click to arm
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        assert tool.has_active_gesture is True
        assert tool.status_text == "depth: 0.000"

    def test_click_in_idle_does_nothing(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        # Camera ray misses the rectangle.
        camera.ray_from_screen.return_value = (
            np.array([5.0, 5.0, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)  # IDLE
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        assert tool.has_active_gesture is False

    def test_depth_increases_as_camera_ray_aims_above_face(self):
        """Line-line CPA: if the camera ray's closest approach to the normal
        line (face_center, +Z) is at z=2, depth should be 2."""
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        # Hover + arm.
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        # Now move: rotate the camera so its ray aims at (0.5, 0.5, 2).
        # A horizontal ray at z=2 with direction +X, origin (-3, 0.5, 2):
        camera.ray_from_screen.return_value = (
            np.array([-3.0, 0.5, 2.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        # CPA between this horizontal ray and the +Z normal line through
        # (0.5, 0.5, 0) gives t = 2.0 on the normal line. So depth = 2.0.
        assert tool.status_text == "depth: 2.000"

    def test_depth_clamps_to_zero_on_negative_drag(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        # Horizontal ray at z = -3 (below the source face).
        camera.ray_from_screen.return_value = (
            np.array([-3.0, 0.5, -3.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        assert tool.status_text == "depth: 0.000"

    def test_depth_frozen_when_view_parallel_to_normal(self):
        """Camera looking straight down — ray direction == -normal — the depth
        metric's denominator goes to ~0; depth should NOT update."""
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        # First move: drive depth to a non-zero value.
        camera.ray_from_screen.return_value = (
            np.array([-3.0, 0.5, 2.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        assert tool.status_text == "depth: 2.000"
        # Second move: ray collinear with normal (degenerate case).
        camera.ray_from_screen.return_value = (
            np.array([0.5, 0.5, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        # Depth should be FROZEN at the previous value (2.0), not reset to 0.
        assert tool.status_text == "depth: 2.000"


class TestPushPullCancel:
    def test_esc_in_dragging_clears_state_without_committing(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        tool, scene, f, camera, cmd_stack = _make_tool_with_unit_rect()
        # Hover + arm + drag.
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        camera.ray_from_screen.return_value = (
            np.array([-3.0, 0.5, 2.0], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        assert tool.has_active_gesture is True
        # ESC
        esc = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        tool.on_key_press(esc)
        # No command pushed.
        cmd_stack.push_executed.assert_not_called()
        # State reset.
        assert tool.has_active_gesture is False

    def test_second_click_below_threshold_cancels(self):
        tool, scene, f, camera, cmd_stack = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        # Don't move; depth stays at 0. Second click should cancel.
        tool.on_mouse_press(_make_event(kind=QMouseEvent.Type.MouseButtonPress), snap=None)
        cmd_stack.push_executed.assert_not_called()
        assert tool.has_active_gesture is False

    def test_esc_in_hovering_or_idle_is_noop_for_the_tool(self):
        """The two-stage ESC behavior (deactivating the tool) is owned by
        MainWindow. The tool itself should treat ESC in non-DRAGGING as a no-op
        (it should not crash, it should not clear state)."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent

        tool, scene, f, camera, cmd_stack = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)  # HOVERING
        assert tool._state.name == "HOVERING"
        esc = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        tool.on_key_press(esc)
        # Still HOVERING — the tool itself doesn't deactivate; MainWindow does.
        assert tool._state.name == "HOVERING"
