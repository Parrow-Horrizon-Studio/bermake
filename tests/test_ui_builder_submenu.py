"""A SubmenuSpec inside a MenuSpec builds a nested QMenu (M7.12, File > Autosave)."""

from __future__ import annotations

from bermake.ui import actions, ui_builder
from PySide6.QtWidgets import QMainWindow

AUTOSAVE_IDS = (
    "file_autosave_off",
    "file_autosave_1",
    "file_autosave_5",
    "file_autosave_10",
    "file_autosave_30",
)


def _submenu(menu, title):
    for entry in menu.actions():
        if entry.menu() is not None and entry.text() == title:
            return entry.menu()
    return None


def test_builder_turns_a_submenu_spec_into_a_nested_menu(qtbot, monkeypatch):
    """Exercised on a bare window with a made-up table, so the builder's
    handling (including a separator inside the submenu) is pinned on its own."""
    window = QMainWindow()
    qtbot.addWidget(window)
    ui_builder.build_all_actions(window)
    monkeypatch.setattr(
        actions,
        "MENUS",
        (
            actions.MenuSpec(
                "Things",
                (
                    "file_new",
                    actions.SubmenuSpec("Nested", ("file_open", None, "file_save")),
                    "file_save_as",
                ),
            ),
        ),
    )

    menus = ui_builder.build_menubar(window)

    things = menus["Things"]
    top = things.actions()
    assert top[0] is window._actions["file_new"]
    assert top[1].menu() is not None and top[1].text() == "Nested"
    assert top[2] is window._actions["file_save_as"]
    nested = top[1].menu().actions()
    assert nested[0] is window._actions["file_open"]
    assert nested[1].isSeparator()
    assert nested[2] is window._actions["file_save"]


def test_file_menu_holds_the_autosave_submenu(qtbot, main_window):
    autosave = _submenu(main_window._file_menu, "Autosave")
    assert autosave is not None
    entries = autosave.actions()
    assert entries == [main_window._actions[i] for i in AUTOSAVE_IDS]
    assert [a.text() for a in entries] == [
        "Off",
        "Every Minute",
        "Every 5 Minutes",
        "Every 10 Minutes",
        "Every 30 Minutes",
    ]
    groups = {a.actionGroup() for a in entries}
    assert len(groups) == 1
    (group,) = groups
    assert group is not None and group.isExclusive()
    assert all(a.isCheckable() for a in entries)
    assert group.actions() == entries  # nothing else shares the group


def test_autosave_submenu_follows_save_as(qtbot, main_window):
    top = main_window._file_menu.actions()
    save_as = top.index(main_window._actions["file_save_as"])
    assert top[save_as + 1].menu() is not None
    assert top[save_as + 1].text() == "Autosave"
