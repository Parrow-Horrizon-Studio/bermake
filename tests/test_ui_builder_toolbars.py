"""The seven toolbars (M7.2 Task 11)."""

from __future__ import annotations

from bermake.ui import actions


def test_all_seven_toolbars_exist(qtbot, main_window):
    assert set(main_window._toolbars) == {spec.id for spec in actions.TOOLBARS}
    assert len(main_window._toolbars) == 7


def test_every_toolbar_has_an_object_name_so_state_can_persist(qtbot, main_window):
    # saveState() silently drops toolbars without an object name.
    for toolbar_id, toolbar in main_window._toolbars.items():
        assert toolbar.objectName() == toolbar_id


def test_toolbar_contents_match_the_registry(qtbot, main_window):
    for spec in actions.TOOLBARS:
        built = main_window._toolbars[spec.id].actions()
        expected_labels = [
            None if aid is None else actions.action_by_id(aid).label for aid in spec.action_ids
        ]
        actual_labels = [None if a.isSeparator() else a.text() for a in built]
        assert actual_labels == expected_labels, spec.id


def test_toolbar_buttons_carry_icons(qtbot, main_window):
    for toolbar in main_window._toolbars.values():
        for action in toolbar.actions():
            if not action.isSeparator():
                assert not action.icon().isNull(), action.text()


def test_toolbar_actions_are_the_same_objects_the_menus_use(qtbot, main_window):
    # The whole point of the registry: toolbars must share the QActions the
    # menus already built, not construct parallel ones with the same label.
    for spec in actions.TOOLBARS:
        toolbar = main_window._toolbars[spec.id]
        for action_id, built_action in zip(
            (aid for aid in spec.action_ids if aid is not None),
            (a for a in toolbar.actions() if not a.isSeparator()),
            strict=True,
        ):
            assert built_action is main_window._actions[action_id]


def test_clicking_a_toolbar_tool_arms_it_and_checks_only_that_button(qtbot, main_window):
    main_window._actions["tool_roof"].trigger()

    assert main_window._tool_manager.active.shortcut.upper() == "O"
    checked = [
        spec.id
        for spec in actions.ACTIONS
        if spec.group == actions.TOOL_GROUP and main_window._actions[spec.id].isChecked()
    ]
    assert checked == ["tool_roof"]


def test_keyboard_activation_updates_the_toolbar_button(qtbot, main_window):
    # Arming by shortcut must check the toolbar button too, or the toolbar
    # would lie about which tool is active.
    main_window._activate("rectangle")
    assert main_window._actions["tool_rectangle"].isChecked()


def test_toolbars_submenu_has_one_checkbox_per_toolbar(qtbot, main_window):
    # Look up the submenu through main_window's own persistent attributes
    # rather than re-deriving it from menuBar().actions() -> .menu(): PySide6
    # frees the underlying C++ QMenu when a transient wrapper for a *parent*
    # menu is discarded, even while another Python name still points at one
    # of its children -- so re-deriving and discarding the View menu here
    # would silently invalidate the Toolbars submenu fetched through it.
    toolbars_action = next(a for a in main_window._view_menu.actions() if a.text() == "Toolbars")
    assert toolbars_action.menu() is main_window._toolbars_menu
    labels = {a.text() for a in main_window._toolbars_menu.actions() if not a.isSeparator()}
    for spec in actions.TOOLBARS:
        assert spec.title in labels


def test_reset_toolbars_lives_in_the_menu_bar_not_a_toolbar(qtbot, main_window):
    # The escape hatch: restoreState() can hide every toolbar, but it cannot
    # touch the menu bar, so Reset Toolbars must be reachable from there (see
    # the test above for why this reads main_window._toolbars_menu directly
    # instead of re-deriving the submenu from the menu bar).
    assert main_window._actions["view_reset_toolbars"] in main_window._toolbars_menu.actions()
    for toolbar in main_window._toolbars.values():
        assert main_window._actions["view_reset_toolbars"] not in toolbar.actions()


def test_reset_toolbars_restores_a_hidden_toolbar(qtbot, main_window):
    # isVisible() reflects the whole ancestor chain, including whether the
    # (never-shown, per this fixture) top-level window itself is on screen,
    # so it would read False here regardless of what reset does. isHidden()
    # is the toolbar's own explicit visibility flag -- the thing Reset
    # Toolbars actually controls -- matching the convention already used by
    # tests/test_window_state.py::test_toolbar_visibility_survives_a_round_trip.
    main_window._toolbars["drawing"].setVisible(False)
    assert main_window._toolbars["drawing"].isHidden()

    main_window._on_reset_toolbars()

    assert not main_window._toolbars["drawing"].isHidden()


def test_every_handler_name_now_exists(qtbot):
    from bermake.ui.main_window import MainWindow

    missing = sorted({s.handler for s in actions.ACTIONS if not hasattr(MainWindow, s.handler)})
    assert missing == []


def test_a_missing_icon_asset_does_not_stop_the_window_from_building(qtbot, monkeypatch):
    """A damaged install should cost one glyph, not the whole application.

    icon() stays strict so a typo'd stem fails a test; build_action
    degrades so a quarantined or truncated asset at runtime leaves a
    labelled, working, icon-less button instead of an unhandled KeyError
    out of MainWindow.__init__.
    """
    from bermake.ui import icons as icons_module
    from bermake.ui.main_window import MainWindow

    real = icons_module.icon

    def _one_missing(stem, color=None):
        if stem == "tool_line":
            raise KeyError(stem)
        return real(stem, color)

    monkeypatch.setattr("bermake.ui.ui_builder.icon", _one_missing)

    window = MainWindow()
    qtbot.addWidget(window)

    assert window._actions["tool_line"].icon().isNull()
    assert window._actions["tool_line"].text() == "Line"
    assert not window._actions["tool_rectangle"].icon().isNull()
