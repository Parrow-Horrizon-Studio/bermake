"""MainWindow integration: WallTool registered under W + options bar (M7a, Task 5)."""

from __future__ import annotations

from bermake.tools.wall_tool import WallTool
from bermake.ui.main_window import MainWindow


def test_wall_tool_registered_with_w(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._tool_manager.activate_by_shortcut("W")
    assert isinstance(w._tool_manager.active, WallTool)


def test_options_bar_visible_only_for_wall(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w._tool_manager.activate_by_shortcut("W")
    w._refresh_tool_options()                 # the hook MainWindow calls on tool switch
    assert w._wall_options_bar.isVisibleTo(w)
    w._tool_manager.activate_by_shortcut("L")  # line tool
    w._refresh_tool_options()
    assert not w._wall_options_bar.isVisibleTo(w)


def test_arming_a_tool_with_settings_does_not_reopen_a_closed_panel(qtbot):
    # Spec 1.6: arming a tool with settings SWITCHES the Tool Settings tab; it
    # must not also un-hide a panel the user deliberately closed.
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w._properties_dock.hide()
    assert w._properties_dock.isHidden()

    w._tool_manager.activate_by_shortcut("W")
    w._refresh_tool_options()

    assert w._properties_dock.current_tab_id == "tool_settings"
    assert w._properties_dock.isHidden()
