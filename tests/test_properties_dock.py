"""The Properties dock shell: splitter, tab strip, page stack (M7.3 Task 8)."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from bermake.ui.panels import PROPERTIES_TABS
from bermake.ui.properties_dock import PropertiesDock


def test_the_dock_has_a_persistence_object_name(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    # QMainWindow.saveState() silently drops docks without an object name.
    assert dock.objectName() == PropertiesDock.OBJECT_NAME
    assert dock.objectName() != ""


def test_one_button_per_declared_tab(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    assert sorted(dock.tab_ids()) == sorted(t.id for t in PROPERTIES_TABS)


def test_every_tab_button_carries_an_icon(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    for tab_id in dock.tab_ids():
        assert not dock.tab_button(tab_id).icon().isNull(), tab_id


def test_the_first_tab_is_current_on_construction(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    assert dock.current_tab_id == PROPERTIES_TABS[0].id


def test_show_tab_switches_the_stack(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)

    dock.show_tab("scenes")

    assert dock.current_tab_id == "scenes"
    assert dock.tab_button("scenes").isChecked()
    assert not dock.tab_button("tool_settings").isChecked()


def test_clicking_a_tab_button_switches_the_stack(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)

    dock.tab_button("tags").click()

    assert dock.current_tab_id == "tags"


def test_show_tab_ignores_an_unknown_id(qtbot):
    # Defensive: a stale saved id or a typo must not take the panel down.
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    before = dock.current_tab_id

    dock.show_tab("no_such_tab")

    assert dock.current_tab_id == before


def test_set_page_replaces_the_placeholder(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    page = QLabel("real page")

    dock.set_page("entity_info", page)
    dock.show_tab("entity_info")

    assert dock.current_page() is page


def test_set_page_called_twice_shows_the_second_widget(qtbot):
    # Task 11 installs three real pages through set_page after Task 8's own
    # placeholder-replacement already occupies the slot -- untested until now,
    # even though set_page's remove-then-insert-at-old-index arithmetic was
    # traced correct by hand during Task 8's review.
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    first = QLabel("first page")
    second = QLabel("second page")

    dock.set_page("entity_info", first)
    dock.set_page("entity_info", second)
    dock.show_tab("entity_info")

    assert dock.current_page() is second
    assert dock._stack.indexOf(first) == -1


def test_set_outliner_installs_the_widget_above_the_tabs(qtbot):
    dock = PropertiesDock()
    qtbot.addWidget(dock)
    tree = QLabel("tree")

    dock.set_outliner(tree)

    assert dock.outliner() is tree
