"""View menu entries focus Properties tabs (M7.3 Task 11)."""

from __future__ import annotations

import pytest
from bermake.ui import actions


@pytest.mark.parametrize(
    "action_id,tab_id",
    [("view_materials", "material"), ("view_tags", "tags"), ("view_scenes", "scenes")],
)
def test_the_action_is_declared(action_id, tab_id):
    spec = next((s for s in actions.ACTIONS if s.id == action_id), None)
    assert spec is not None, f"{action_id} is not declared"
    del tab_id


@pytest.mark.parametrize(
    "action_id", ["view_materials", "view_tags", "view_scenes"]
)
def test_the_action_carries_no_icon(action_id):
    # test_icon_assets asserts exactly 29 declared icons; these are menu
    # entries, never toolbar buttons, so adding one would break that count
    # for no benefit.
    spec = next(s for s in actions.ACTIONS if s.id == action_id)
    assert spec.icon is None


def test_the_three_entries_are_in_the_view_menu():
    view = next(m for m in actions.MENUS if m.title == "View")
    for action_id in ("view_materials", "view_tags", "view_scenes"):
        assert action_id in view.action_ids


@pytest.mark.parametrize(
    "action_id,tab_id",
    [("view_materials", "material"), ("view_tags", "tags"), ("view_scenes", "scenes")],
)
def test_triggering_the_action_focuses_its_tab(main_window, action_id, tab_id):
    main_window._properties_dock.show_tab("tool_settings")

    main_window._actions[action_id].trigger()

    assert main_window._properties_dock.current_tab_id == tab_id


def test_the_pages_are_installed_in_the_panel(main_window):
    dock = main_window._properties_dock
    dock.show_tab("material")
    assert dock.current_page() is main_window._materials_page
    dock.show_tab("tags")
    assert dock.current_page() is main_window._tags_page
    dock.show_tab("scenes")
    assert dock.current_page() is main_window._scenes_page


def test_the_old_docks_are_gone(main_window):
    from PySide6.QtWidgets import QDockWidget

    remaining = {
        d.objectName() for d in main_window.findChildren(QDockWidget)
    }
    assert "materials_dock" not in remaining
    assert "tags_dock" not in remaining
    assert "scenes_dock" not in remaining
    assert "properties_dock" in remaining


def test_the_pages_are_plain_widgets(main_window):
    from PySide6.QtWidgets import QDockWidget

    for page in (
        main_window._materials_page,
        main_window._tags_page,
        main_window._scenes_page,
    ):
        assert not isinstance(page, QDockWidget)
