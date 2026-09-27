"""ToolOverlay / ToolContext / Tool ABC extensions for M3b."""

from __future__ import annotations

import numpy as np
import pytest


def test_tool_overlay_face_fill_defaults_to_empty_list_and_ghost_rgba():
    from bermake.tools.tool import ToolOverlay

    overlay = ToolOverlay(
        rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
        rubber_band_color=(1.0, 1.0, 1.0),
        snap_marker_position=None,
        snap_marker_color=(1.0, 1.0, 1.0),
    )
    assert overlay.face_fill_polygons == []
    assert overlay.face_fill_color == (0.4, 0.7, 1.0, 0.15)


def test_tool_overlay_accepts_explicit_face_fill_polygons():
    from bermake.tools.tool import ToolOverlay

    poly = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    overlay = ToolOverlay(
        rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
        rubber_band_color=(1.0, 1.0, 1.0),
        snap_marker_position=None,
        snap_marker_color=(1.0, 1.0, 1.0),
        face_fill_polygons=[poly],
        face_fill_color=(1.0, 0.0, 0.0, 0.3),
    )
    assert len(overlay.face_fill_polygons) == 1
    np.testing.assert_array_equal(overlay.face_fill_polygons[0], poly)
    assert overlay.face_fill_color == (1.0, 0.0, 0.0, 0.3)


def test_tool_context_camera_and_widget_size_provider_default_to_none():
    from bermake.tools.tool import ToolContext

    ctx = ToolContext(scene=object())
    assert ctx.camera is None
    assert ctx.widget_size_provider is None


def test_tool_context_can_carry_camera_and_widget_size_provider():
    from bermake.tools.tool import ToolContext

    fake_camera = object()
    sizer = lambda: (640, 480)
    ctx = ToolContext(
        scene=object(),
        command_stack=None,
        camera=fake_camera,
        widget_size_provider=sizer,
    )
    assert ctx.camera is fake_camera
    assert ctx.widget_size_provider is sizer
    assert ctx.widget_size_provider() == (640, 480)


def test_tool_status_text_default_is_none():
    """Existing M2 / M3a tools that don't override status_text should return None."""
    from bermake.tools import RectangleTool

    tool = RectangleTool()
    assert tool.status_text is None
