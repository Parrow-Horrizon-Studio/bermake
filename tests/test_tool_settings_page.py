"""The Tool Settings tab hosts the three option bars (M7.3 Task 12)."""

from __future__ import annotations

from bermake.ui.tool_settings_page import ToolSettingsPage


def test_an_unknown_key_shows_the_empty_page(qtbot):
    page = ToolSettingsPage()
    qtbot.addWidget(page)
    assert page.current_key is None
    assert page.current_text() == ToolSettingsPage.EMPTY_TEXT


def test_show_bar_switches_to_a_registered_bar(qtbot):
    from PySide6.QtWidgets import QLabel

    page = ToolSettingsPage()
    qtbot.addWidget(page)
    bar = QLabel("wall")
    page.add_bar("wall", bar)

    page.show_bar("wall")

    assert page.current_key == "wall"
    assert page.current_widget() is bar


def test_show_bar_none_returns_to_empty(qtbot):
    from PySide6.QtWidgets import QLabel

    page = ToolSettingsPage()
    qtbot.addWidget(page)
    page.add_bar("wall", QLabel("wall"))
    page.show_bar("wall")

    page.show_bar(None)

    assert page.current_key is None


def test_arming_the_wall_tool_shows_its_bar_and_focuses_the_tab(main_window):
    main_window._properties_dock.show_tab("entity_info")

    main_window._activate("wall")

    assert main_window._properties_dock.current_tab_id == "tool_settings"
    assert main_window._tool_settings_page.current_key == "wall"


def test_arming_a_tool_without_settings_shows_the_empty_page(main_window):
    main_window._activate("wall")
    assert main_window._tool_settings_page.current_key == "wall"

    main_window._activate("line")

    assert main_window._tool_settings_page.current_key is None


def test_arming_a_tool_without_settings_does_not_steal_the_tab(main_window):
    # Only tools that HAVE settings pull the panel over; otherwise every tool
    # switch would yank you out of whatever you were inspecting.
    main_window._properties_dock.show_tab("entity_info")

    main_window._activate("line")

    assert main_window._properties_dock.current_tab_id == "entity_info"


def test_each_tool_with_settings_gets_its_own_bar(main_window):
    for tool_id, key in (("wall", "wall"), ("door_window", "opening"), ("roof", "roof")):
        main_window._activate(tool_id)
        assert main_window._tool_settings_page.current_key == key, tool_id


def test_arming_the_opening_tool_shows_its_bar_and_focuses_the_tab(main_window):
    main_window._properties_dock.show_tab("entity_info")

    main_window._activate("door_window")

    assert main_window._properties_dock.current_tab_id == "tool_settings"
    assert main_window._tool_settings_page.current_key == "opening"


def test_arming_the_roof_tool_shows_its_bar_and_focuses_the_tab(main_window):
    main_window._properties_dock.show_tab("entity_info")

    main_window._activate("roof")

    assert main_window._properties_dock.current_tab_id == "tool_settings"
    assert main_window._tool_settings_page.current_key == "roof"


def test_the_reparented_wall_bar_still_writes_to_its_tool(main_window):
    # The reparenting risk, stated as a test: the bar moved from a QVBoxLayout
    # into a QStackedWidget, and its binding to the tool must survive.
    main_window._activate("wall")
    bar = main_window._wall_options_bar

    bar._thickness_edit.setText("250 mm")
    bar._thickness_edit.editingFinished.emit()

    assert main_window._wall_tool.thickness == 0.25


def test_the_reparented_roof_bar_still_writes_to_its_tool(main_window):
    main_window._activate("roof")
    bar = main_window._roof_options_bar

    bar._slope_edit.setText("40")
    bar._slope_edit.editingFinished.emit()

    assert main_window._roof_tool.slope == 40.0


def test_the_reparented_opening_bar_still_writes_to_its_tool(main_window):
    # Field/attribute confirmed from python/bermake/ui/opening_options_bar.py:
    # `_width_edit` -> `_commit(self._width_edit, "width")` -> DoorWindowTool.width
    # (metres; parsed through parse_length, same as the Wall bar's thickness).
    main_window._activate("door_window")
    bar = main_window._opening_options_bar

    bar._width_edit.setText("1200 mm")
    bar._width_edit.editingFinished.emit()

    assert main_window._opening_tool.width == 1.2


def test_the_option_bars_left_the_central_column(main_window):
    central = main_window.centralWidget()
    for bar in (
        main_window._wall_options_bar,
        main_window._opening_options_bar,
        main_window._roof_options_bar,
    ):
        assert not central.isAncestorOf(bar)
