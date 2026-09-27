# tests/test_tags_page.py
from __future__ import annotations

import pytest
from bermake.model.tag import TagLibrary
from bermake.ui.tags_page import TagsPage
from PySide6.QtCore import Qt


@pytest.fixture
def lib():
    return TagLibrary()


def test_page_lists_untagged_first(qtbot, lib):
    lib.add("Walls")
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    assert dock._list.count() == 2
    assert dock._list.item(0).text() == "Untagged"


def test_checkbox_toggles_visibility_and_emits(qtbot, lib):
    walls = lib.add("Walls")
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    item = dock._list.item(1)                       # the Walls row
    with qtbot.waitSignal(dock.visibility_changed, timeout=500):
        item.setCheckState(Qt.CheckState.Unchecked)
    assert lib.is_visible(walls.id) is False


def test_selecting_row_changes_active_and_emits(qtbot, lib):
    walls = lib.add("Walls")
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    with qtbot.waitSignal(dock.active_tag_changed, timeout=500) as blocker:
        dock._list.setCurrentRow(1)
    assert dock.active_tag_id == walls.id
    assert blocker.args[0] == walls.id


def test_add_tag_grows_list(qtbot, lib):
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    n = dock._list.count()
    dock._on_add()
    assert dock._list.count() == n + 1


def test_assign_emits(qtbot, lib):
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    with qtbot.waitSignal(dock.assign_to_selection_requested, timeout=500):
        dock._on_assign()


def test_rename_via_item_edit_updates_library(qtbot, lib):
    walls = lib.add("Walls")
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    dock._list.item(1).setText("Exterior")
    assert lib.get(walls.id).name == "Exterior"


def test_untagged_item_not_editable(qtbot, lib):
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    assert not (dock._list.item(0).flags() & Qt.ItemFlag.ItemIsEditable)


def test_empty_rename_is_restored(qtbot, lib):
    walls = lib.add("Walls")
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    dock._list.item(1).setText("")
    assert lib.get(walls.id).name == "Walls"
    assert dock._list.item(1).text() == "Walls"


def test_set_selection_tag_label(qtbot, lib):
    dock = TagsPage(lib)
    qtbot.addWidget(dock)
    dock.set_selection_tag("Walls")
    assert dock._selection_label.text() == "Selection: Walls"
    dock.set_selection_tag(None)
    assert dock._selection_label.text() == "Selection: —"
