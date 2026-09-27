"""Context-menu composition (M7.2 Task 13). Pure -- no Qt needed."""

from __future__ import annotations

import pytest

from bermake.ui.actions import ContextTarget
from bermake.ui.context_menu import context_menu_ids, is_enabled


def _ids(target, **kwargs):
    kwargs.setdefault("in_group", False)
    kwargs.setdefault("has_selection", True)
    return [i for i in context_menu_ids(target, **kwargs) if i is not None]


def test_face_menu_offers_paint():
    assert "edit_paint_selection" in _ids(ContextTarget.FACE)


def test_edge_menu_does_not_offer_paint():
    # Materials are per-face; painting an edge is meaningless.
    assert "edit_paint_selection" not in _ids(ContextTarget.EDGE)


def test_assign_tag_is_not_a_declared_action_anywhere():
    # It is a dynamic submenu built from the live TagLibrary (Task 14), and
    # M5c shipped instances-only tagging (#69).
    for target in ContextTarget:
        assert "edit_assign_tag" not in _ids(target)


def test_instance_menu_offers_the_group_operations():
    ids = _ids(ContextTarget.INSTANCE)
    assert {"edit_edit_group", "edit_explode", "edit_make_unique"} <= set(ids)


def test_annotation_menu_offers_erase_and_retype():
    assert _ids(ContextTarget.ANNOTATION) == ["edit_erase", "edit_label_text"]


def test_empty_menu_offers_selection_and_view_commands():
    ids = _ids(ContextTarget.EMPTY)
    assert {"edit_select_all", "edit_select_none", "view_zoom_extents"} <= set(ids)


def test_every_returned_id_is_a_declared_action():
    from bermake.ui.actions import action_by_id

    for target in ContextTarget:
        for action_id in _ids(target):
            action_by_id(action_id)


def test_separators_are_preserved():
    assert None in context_menu_ids(
        ContextTarget.FACE, in_group=False, has_selection=True
    )


@pytest.mark.parametrize("target", list(ContextTarget))
def test_every_target_returns_a_non_empty_menu(target):
    assert _ids(target), f"{target} produces an empty menu"


def test_close_group_is_disabled_at_the_root():
    assert not is_enabled(
        "edit_close_group",
        target=ContextTarget.EMPTY,
        in_group=False,
        is_component=False,
        has_selection=False,
    )


def test_close_group_is_enabled_inside_a_group():
    assert is_enabled(
        "edit_close_group",
        target=ContextTarget.EMPTY,
        in_group=True,
        is_component=False,
        has_selection=False,
    )


def test_make_unique_is_disabled_for_a_plain_group():
    assert not is_enabled(
        "edit_make_unique",
        target=ContextTarget.INSTANCE,
        in_group=False,
        is_component=False,
        has_selection=True,
    )


def test_make_unique_is_enabled_for_a_component():
    assert is_enabled(
        "edit_make_unique",
        target=ContextTarget.INSTANCE,
        in_group=False,
        is_component=True,
        has_selection=True,
    )


def test_select_none_is_disabled_with_an_empty_selection():
    assert not is_enabled(
        "edit_select_none",
        target=ContextTarget.EMPTY,
        in_group=False,
        is_component=False,
        has_selection=False,
    )


def test_ordinary_entries_are_always_enabled():
    assert is_enabled(
        "edit_erase",
        target=ContextTarget.FACE,
        in_group=False,
        is_component=False,
        has_selection=True,
    )


def test_select_same_material_is_disabled_with_nothing_selected(qtbot):
    """Mutant 3 (Step 8): enable select_same_material unconditionally in
    _add_select_submenu and this is the test that catches it. Same
    material needs a seed face; with no selection at all there is nothing
    to seed from."""
    from bermake.ui.context_menu import build_context_menu
    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w._selection.clear()
    menu = build_context_menu(w, ContextTarget.EMPTY, None)
    submenu = None
    for action in menu.actions():
        candidate = action.menu()
        if candidate is not None and candidate.title() == "Select":
            submenu = candidate
            break
    assert submenu is not None
    same_material = next(a for a in submenu.actions() if a.text() == "All with Same Material")
    assert not same_material.isEnabled()


def test_the_select_submenu_appears_on_every_target(qtbot):
    from bermake.ui.context_menu import build_context_menu
    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    for target in ContextTarget:
        menu = build_context_menu(w, target, None)
        titles = [a.menu().title() for a in menu.actions() if a.menu() is not None]
        assert "Select" in titles, target
