"""Smoke tests: the new drawing tools are registered and key-activatable."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import Qt


@pytest.fixture
def main_window(qtbot):  # noqa: ANN001
    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_new_tools_registered(main_window):  # noqa: ANN001
    mgr = main_window._tool_manager
    assert mgr.activate_by_shortcut("C")
    assert mgr.active.name == "Circle"
    assert mgr.activate_by_shortcut("G")
    assert mgr.active.name == "Polygon"
    assert mgr.activate_by_shortcut("A")
    assert mgr.active.name == "Arc"


def test_arrow_keys_forward_to_active_polygon_gesture(main_window):  # noqa: ANN001
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    mgr = main_window._tool_manager
    mgr.activate_by_shortcut("G")
    tool = mgr.active
    snap = SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="t",
    )
    tool.on_mouse_press(None, snap)  # begin gesture
    main_window._on_tool_key(Qt.Key.Key_Up)
    assert tool._sides == 7


def test_arrow_keys_inert_without_active_gesture(main_window):  # noqa: ANN001
    mgr = main_window._tool_manager
    mgr.activate_by_shortcut("G")
    tool = mgr.active
    # No gesture started → forwarding must be a no-op (sides stay default 6).
    main_window._on_tool_key(Qt.Key.Key_Up)
    assert tool._sides == 6
