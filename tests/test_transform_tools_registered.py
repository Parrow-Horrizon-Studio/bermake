"""Move/Rotate/Scale are registered and bound to M/Q/S."""

from __future__ import annotations

from bermake.tools import MoveTool, RotateTool, ScaleTool


def test_tool_shortcuts():
    assert MoveTool().shortcut == "M"
    assert RotateTool().shortcut == "Q"
    assert ScaleTool().shortcut == "S"
    assert MoveTool().name == "Move"
    assert RotateTool().name == "Rotate"
    assert ScaleTool().name == "Scale"


def test_main_window_registers_transform_tools(qtbot):
    from bermake.ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    assert win._tool_manager.activate_by_shortcut("M")
    assert win._tool_manager.activate_by_shortcut("Q")
    assert win._tool_manager.activate_by_shortcut("S")
