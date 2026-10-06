"""Registry invariants (M7.2 Task 1). Imports no Qt — runs headless."""

from __future__ import annotations

import bermake.ui.actions as A


def test_module_imports_no_qt():
    import sys

    # Importing actions must not drag PySide6 in. Check the module's own globals
    # rather than sys.modules, which other tests may have populated.
    for value in vars(A).values():
        module = getattr(value, "__module__", "") or ""
        assert not module.startswith("PySide6"), f"{value!r} comes from Qt"
    assert "bermake.ui.actions" in sys.modules


def test_action_ids_are_unique():
    ids = [spec.id for spec in A.ACTIONS]
    assert len(ids) == len(set(ids))


def test_no_duplicate_shortcuts():
    seen: dict[str, str] = {}
    for spec in A.ACTIONS:
        for key in filter(None, (spec.shortcut, *spec.extra_shortcuts)):
            assert key not in seen, f"{key} used by both {seen[key]} and {spec.id}"
            seen[key] = spec.id


def test_icon_stem_equals_id_when_present():
    for spec in A.ACTIONS:
        if spec.icon is not None:
            assert spec.icon == spec.id


def test_every_toolbar_id_resolves():
    for toolbar in A.TOOLBARS:
        for action_id in filter(None, toolbar.action_ids):
            A.action_by_id(action_id)  # raises KeyError if unknown


def _menu_action_ids(entries):
    """Every action id in a menu, descending into its submenus (M7.12)."""
    for entry in entries:
        if isinstance(entry, A.SubmenuSpec):
            yield from _menu_action_ids(entry.action_ids)
        elif entry is not None:
            yield entry


def test_every_menu_id_resolves():
    for menu in A.MENUS:
        for action_id in _menu_action_ids(menu.action_ids):
            A.action_by_id(action_id)


def test_autosave_entries_are_declared_exactly():
    """File > Autosave (M7.12, spec 4.4): five checkable, exclusive entries."""
    expected = [
        ("file_autosave_off", "Off", 0),
        ("file_autosave_1", "Every Minute", 1),
        ("file_autosave_5", "Every 5 Minutes", 5),
        ("file_autosave_10", "Every 10 Minutes", 10),
        ("file_autosave_30", "Every 30 Minutes", 30),
    ]
    assert A.AUTOSAVE_GROUP == "autosave"
    grouped = [s for s in A.ACTIONS if s.group == A.AUTOSAVE_GROUP]
    assert [(s.id, s.label, s.handler_arg) for s in grouped] == expected
    for spec in grouped:
        assert spec.checkable is True
        assert spec.handler == "_set_autosave_interval"


def test_autosave_submenu_sits_after_save_as_before_the_separator():
    file_menu = next(m for m in A.MENUS if m.title == "File")
    entries = list(file_menu.action_ids)
    at = entries.index("file_save_as")
    submenu = entries[at + 1]
    assert submenu == A.SubmenuSpec(
        "Autosave",
        (
            "file_autosave_off",
            "file_autosave_1",
            "file_autosave_5",
            "file_autosave_10",
            "file_autosave_30",
        ),
    )
    # Still in the Save block: a separator comes before the import entries.
    # (Not pinned to at + 2, because Revert to Saved joins this block next.)
    assert None in entries[at + 2 : entries.index("file_import_obj")]


def test_every_context_menu_id_resolves():
    for ids in A.CONTEXT_MENUS.values():
        for action_id in filter(None, ids):
            A.action_by_id(action_id)


def test_every_toolbar_action_has_an_icon():
    for toolbar in A.TOOLBARS:
        for action_id in filter(None, toolbar.action_ids):
            assert A.action_by_id(action_id).icon is not None


def test_all_eighteen_tools_are_declared():
    tools = [s for s in A.ACTIONS if s.group == A.TOOL_GROUP]
    assert len(tools) == 24
    for spec in tools:
        assert spec.checkable is True
        assert spec.cursor is not None
        assert spec.icon is not None
        assert spec.handler == "_activate"
        # M7.4 Task 5: handler_arg dispatches by tool id, not by shortcut, so
        # a tool without a shortcut (M7.4's primitives, Follow Me) can still
        # be armed from a menu or toolbar. (The id/handler_arg cross-check
        # against real tool classes lives in
        # test_tool_manager_ids.py::test_every_shipped_tool_has_a_unique_id_matching_its_action
        # -- asserting it again here against spec.id, itself built from the
        # same action id by the same _tool() call, would prove nothing.)
        assert spec.id.startswith("tool_")


def test_grouped_actions_are_checkable():
    for spec in A.ACTIONS:
        if spec.group is not None:
            assert spec.checkable, f"{spec.id} is grouped but not checkable"


def test_action_by_id_raises_on_unknown():
    import pytest

    with pytest.raises(KeyError):
        A.action_by_id("no_such_action")


def test_tooltip_defaults_to_label_and_shortcut():
    spec = A.action_by_id("tool_line")
    assert spec.effective_tooltip == "Line (L)"
    spec = A.action_by_id("file_import_obj")
    assert spec.effective_tooltip == "Import OBJ…"
