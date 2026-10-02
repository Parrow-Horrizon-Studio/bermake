"""Registry-driven menus and shortcuts (M7.2 Task 10)."""

from __future__ import annotations

from bermake.ui import actions
from PySide6.QtGui import QAction


def test_every_declared_action_became_a_qaction(qtbot, main_window):
    built = main_window._actions
    for spec in actions.ACTIONS:
        assert spec.id in built, f"{spec.id} was never built"
        assert isinstance(built[spec.id], QAction)


def test_tools_menu_lists_all_eighteen_tools(qtbot, main_window):
    tools_menu = next(m for m in main_window.menuBar().actions() if m.text() == "Tools")
    entries = [a for a in tools_menu.menu().actions() if not a.isSeparator()]
    assert len(entries) == 24


def test_labels_come_from_the_registry(qtbot, main_window):
    assert main_window._actions["tool_push_pull"].text() == "Push/Pull"
    assert main_window._actions["file_save_as"].text() == "Save As…"


def test_shortcuts_come_from_the_registry(qtbot, main_window):
    action = main_window._actions["tool_wall"]
    assert action.shortcut().toString().upper() == "W"


def test_redo_carries_both_bindings(qtbot, main_window):
    bindings = {s.toString().upper() for s in main_window._actions["edit_redo"].shortcuts()}
    assert bindings == {"CTRL+Y", "CTRL+SHIFT+Z"}


def test_activating_a_tool_action_arms_that_tool(qtbot, main_window):
    main_window._actions["tool_circle"].trigger()
    assert main_window._tool_manager.active.shortcut.upper() == "C"


def test_face_style_action_applies_the_style(qtbot, main_window):
    from bermake.viewport.render_style import FaceStyle

    main_window._actions["view_style_wireframe"].trigger()
    assert main_window._render_style.face_style is FaceStyle.WIREFRAME


def test_units_action_applies_the_unit(qtbot, main_window):
    main_window._actions["units_metric_cm"].trigger()
    # _set_units_metric("cm") must have been called with its handler_arg.
    assert "cm" in repr(main_window._doc.units).lower()


def test_unit_labels_use_parentheses_not_dashes(qtbot, main_window):
    labels = {
        spec_id: main_window._actions[spec_id].text()
        for spec_id in (
            "units_metric_m",
            "units_metric_cm",
            "units_metric_mm",
            "units_imperial",
        )
    }
    assert labels == {
        "units_metric_m": "Metric (m)",
        "units_metric_cm": "Metric (cm)",
        "units_metric_mm": "Metric (mm)",
        "units_imperial": "Imperial (architectural)",
    }


def test_xray_toggle_passes_the_checked_state(qtbot, main_window):
    action = main_window._actions["view_xray"]
    action.setChecked(True)
    assert main_window._render_style.xray is True
    action.setChecked(False)
    assert main_window._render_style.xray is False


def test_tool_actions_are_mutually_exclusive(qtbot, main_window):
    main_window._actions["tool_line"].trigger()
    main_window._actions["tool_circle"].trigger()
    checked = [
        spec.id
        for spec in actions.ACTIONS
        if spec.group == actions.TOOL_GROUP and main_window._actions[spec.id].isChecked()
    ]
    assert checked == ["tool_circle"]


def test_every_handler_name_exists_on_the_window_class(qtbot):
    from bermake.ui.main_window import MainWindow

    missing = sorted({s.handler for s in actions.ACTIONS if not hasattr(MainWindow, s.handler)})
    assert missing == []
