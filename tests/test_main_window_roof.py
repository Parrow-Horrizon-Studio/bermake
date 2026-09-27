from __future__ import annotations

from bermake.tools.roof_tool import RoofTool
from bermake.ui.main_window import MainWindow


def test_roof_tool_registered_with_o(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._tool_manager.activate_by_shortcut("O")
    assert isinstance(w._tool_manager.active, RoofTool)


def test_o_key_shortcut_registered(qtbot):
    # M7.2 Task 10: O is no longer a raw QShortcut -- it is the tool_roof
    # action's registry shortcut, wired to fire at window scope by
    # build_shortcuts() (added to w.actions()).
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._actions["tool_roof"].shortcut().toString().upper() == "O"
    assert w._actions["tool_roof"] in w.actions()


def test_roof_options_bar_visible_only_for_tool(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w.show()
    w._tool_manager.activate_by_shortcut("O")
    w._refresh_tool_options()
    assert w._roof_options_bar.isVisibleTo(w)
    w._tool_manager.activate_by_shortcut("L")  # line tool
    w._refresh_tool_options()
    assert not w._roof_options_bar.isVisibleTo(w)
