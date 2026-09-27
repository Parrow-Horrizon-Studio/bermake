"""Turning ActionSpecs into Qt widgets (M7.2).

This is the only module that converts the registry into QActions, so the
wiring rules live in exactly one place:

  checkable and ungrouped -> connect `toggled`, handler receives the bool
  handler_arg is set      -> connect `triggered`, handler receives that value
  otherwise               -> connect `triggered`, handler receives nothing

Built QActions are stored on the window as `_actions: dict[str, QAction]` so
later code (toolbars, context menus, enablement) can look one up by id
instead of searching menus by label.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QMainWindow, QMenu, QToolBar

from bermake.ui import actions
from bermake.ui.icons import icon

ACTION_REGISTRY_ATTR = "_actions"


def connect_action(action: QAction, window: QMainWindow, spec: actions.ActionSpec) -> None:
    """Wire one QAction to its MainWindow handler.

    `view_reset_toolbars` names a handler (`_on_reset_toolbars`) that does not
    exist on MainWindow yet -- it lands with the toolbars in the next task.
    Its QAction must still be built (the registry-wide "every action became a
    QAction" test requires it), so a missing handler is tolerated here and the
    action is simply left unconnected rather than raising.
    """
    handler = getattr(window, spec.handler, None)
    if handler is None:
        return

    if spec.checkable and spec.group is None:
        action.toggled.connect(handler)
        return

    if spec.handler_arg is not None:
        action.triggered.connect(lambda _checked=False, arg=spec.handler_arg: handler(arg))
        return

    action.triggered.connect(lambda _checked=False: handler())


def build_action(
    window: QMainWindow, spec: actions.ActionSpec, groups: dict[str, QActionGroup]
) -> QAction:
    """Create, configure, and connect the QAction for one spec."""
    action = QAction(spec.label, window)
    action.setToolTip(spec.effective_tooltip)
    action.setStatusTip(spec.effective_tooltip)

    if spec.icon is not None:
        try:
            action.setIcon(icon(spec.icon, window.palette().windowText().color()))
        except KeyError:
            # icon() is deliberately strict so a typo'd stem fails a test
            # rather than shipping a blank button. But at runtime the only
            # way here is a damaged install -- a quarantined or truncated
            # asset -- and refusing to build the action would take the whole
            # window down over one missing glyph. Degrade to a labelled,
            # working, icon-less action instead.
            logging.getLogger(__name__).warning(
                "icon asset %r is missing; %r will render without one", spec.icon, spec.id
            )

    keys = [k for k in (spec.shortcut, *spec.extra_shortcuts) if k]
    if len(keys) == 1:
        action.setShortcut(QKeySequence(keys[0]))
    elif keys:
        action.setShortcuts([QKeySequence(k) for k in keys])

    if spec.checkable:
        action.setCheckable(True)
    if spec.group is not None:
        group = groups.get(spec.group)
        if group is None:
            group = QActionGroup(window)
            group.setExclusive(True)
            groups[spec.group] = group
        action.setActionGroup(group)

    connect_action(action, window, spec)
    return action


def build_all_actions(window: QMainWindow) -> dict[str, QAction]:
    """Build every declared action once and stash them on the window."""
    groups: dict[str, QActionGroup] = {}
    built = {spec.id: build_action(window, spec, groups) for spec in actions.ACTIONS}
    setattr(window, ACTION_REGISTRY_ATTR, built)
    window._action_groups = groups
    return built


def build_menubar(window: QMainWindow) -> dict[str, QMenu]:
    """Populate the menu bar from MENUS. Returns the menus by title so the
    caller can append dynamic entries (dock toggles, the Toolbars submenu)."""
    built = getattr(window, ACTION_REGISTRY_ATTR)
    menubar = window.menuBar()
    menus: dict[str, QMenu] = {}

    for menu_spec in actions.MENUS:
        menu = menubar.addMenu(menu_spec.title)
        for action_id in menu_spec.action_ids:
            if action_id is None:
                menu.addSeparator()
            else:
                menu.addAction(built[action_id])
        menus[menu_spec.title] = menu

    return menus


def build_toolbars(window: QMainWindow) -> dict[str, QToolBar]:
    """Create the declared toolbars and add them to the window.

    Every toolbar gets setObjectName(spec.id): QMainWindow.saveState() silently
    skips toolbars without an object name, so omitting this makes layout
    persistence fail invisibly. Toolbars reference the same QAction objects the
    menus were built from (looked up in `_actions` by id) rather than building
    parallel ones -- that identity is what lets one QAction's state (checked,
    enabled, icon) show up consistently in both surfaces.
    """
    built = getattr(window, ACTION_REGISTRY_ATTR)
    toolbars: dict[str, QToolBar] = {}

    for spec in actions.TOOLBARS:
        toolbar = QToolBar(spec.title, window)
        toolbar.setObjectName(spec.id)
        toolbar.setIconSize(QSize(24, 24))
        for action_id in spec.action_ids:
            if action_id is None:
                toolbar.addSeparator()
            else:
                toolbar.addAction(built[action_id])
        window.addToolBar(toolbar)
        toolbars[spec.id] = toolbar

    return toolbars


def build_shortcuts(window: QMainWindow) -> None:
    """Make every action's shortcut fire regardless of focus.

    QActions only respond to their shortcut while they are reachable from the
    focused widget's window; adding them to the window itself with a
    WindowShortcut context is what makes the single-letter tool keys work
    while the viewport has focus.
    """
    for action in getattr(window, ACTION_REGISTRY_ATTR).values():
        if action.shortcuts():
            action.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            window.addAction(action)
