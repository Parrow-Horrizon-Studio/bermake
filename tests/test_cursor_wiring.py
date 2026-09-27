"""The viewport cursor follows the armed tool (M7.2 Task 12)."""

from __future__ import annotations

import pytest

from bermake.ui import cursors


def test_arming_a_crosshair_tool_sets_the_crosshair_hotspot(qtbot, main_window):
    main_window._activate("line")
    hotspot = main_window._viewport.cursor().hotSpot()
    assert (hotspot.x(), hotspot.y()) == cursors.CROSSHAIR_HOTSPOT


def test_arming_an_arrow_tool_sets_the_arrow_hotspot(qtbot, main_window):
    main_window._activate("select")
    hotspot = main_window._viewport.cursor().hotSpot()
    assert (hotspot.x(), hotspot.y()) == cursors.ARROW_HOTSPOT


def test_switching_tools_switches_the_cursor(qtbot, main_window):
    main_window._activate("select")
    first = main_window._viewport.cursor().pixmap().toImage()
    main_window._activate("circle")
    second = main_window._viewport.cursor().pixmap().toImage()
    assert first != second


def test_the_cursor_is_set_on_the_viewport_not_the_window(qtbot, main_window):
    # Setting it on the window would leak the tool cursor over the docks,
    # menu bar, and option bars.
    main_window._activate("line")
    assert main_window._viewport.testAttribute.__self__ is main_window._viewport
    assert main_window._viewport.cursor().hotSpot() != main_window.cursor().hotSpot()


def test_every_tool_can_be_armed_without_raising(qtbot, main_window):
    from bermake.ui.actions import ACTIONS, TOOL_GROUP

    for spec in ACTIONS:
        if spec.group == TOOL_GROUP:
            main_window._activate(spec.handler_arg)
            assert not main_window._viewport.cursor().pixmap().isNull(), spec.id


def test_arming_a_shortcutless_tool_checks_its_action_and_sets_the_cursor(qtbot, main_window):
    """M7.4 Task 5 Finding 1: the toolbar/cursor sync in _activate must not
    depend on the tool having a keyboard shortcut. Five of M7.4's six new
    tools ship with none (spec D9), and arming one from its toolbar button
    or menu entry -- which reaches _activate exactly the way this test does,
    via handler_arg -- must still check the action and set the cursor, not
    just update ToolManager state.
    """
    from bermake.tools.line_tool import LineTool

    class _ShortcutlessLine(LineTool):
        """Same id as the shipped Line tool, but no shortcut to key off of."""

        @property
        def shortcut(self) -> str:
            return ""

    main_window._tool_manager.register(_ShortcutlessLine())
    main_window._activate("line")

    assert main_window._actions["tool_line"].isChecked()
    hotspot = main_window._viewport.cursor().hotSpot()
    assert (hotspot.x(), hotspot.y()) == cursors.CROSSHAIR_HOTSPOT


def test_the_wired_cursor_is_composed_at_the_viewports_device_pixel_ratio(
    qtbot, main_window, monkeypatch
):
    # A window can move between monitors with different scaling, so the
    # ratio must be read from the widget itself at activation time rather
    # than baked in as a global 1.0 (M7.2 Task 12 -- closing the Task 5 gap
    # where cursor_for always composed at dpr=1.0).
    monkeypatch.setattr(main_window._viewport, "devicePixelRatioF", lambda: 2.0)

    main_window._activate("line")

    pixmap = main_window._viewport.cursor().pixmap()
    assert pixmap.devicePixelRatio() == pytest.approx(2.0)
    assert pixmap.width() == cursors.CURSOR_SIZE * 2


def test_escape_disarms_the_toolbar_button_and_the_cursor(qtbot, main_window):
    """Arming checks the action and sets the cursor; disarming must undo both.

    Otherwise the toolbar keeps a button depressed and the viewport keeps a
    tool cursor for a tool that is no longer active -- the exact ambiguity
    the toolbars were added to remove.
    """
    window = main_window
    window._activate("line")
    assert window._actions["tool_line"].isChecked()
    armed = window._viewport.cursor().hotSpot()

    window._on_escape()

    assert window._viewport.tool_manager.active is None
    assert not window._actions["tool_line"].isChecked()
    assert window._viewport.cursor().hotSpot() != armed
