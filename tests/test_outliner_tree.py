"""The Outliner tree widget, fed synthetic rows (M7.3 Task 9)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView

from bermake.model.model_queries import OutlinerRow
from bermake.ui.outliner_tree import OutlinerTree


def _row(instance_id, depth=0, label="Group", **kwargs):
    defaults = dict(
        is_component=False,
        hidden=False,
        inherited_hidden=False,
        tag_hidden=False,
        on_active_path=False,
    )
    defaults.update(kwargs)
    return OutlinerRow(instance_id=instance_id, depth=depth, label=label, **defaults)


def _tree(qtbot, rows=(), root_label="Model"):
    tree = OutlinerTree()
    qtbot.addWidget(tree)
    tree.set_rows(rows, root_label)
    return tree


def test_an_empty_model_shows_only_the_root(qtbot):
    tree = _tree(qtbot)
    assert tree.topLevelItemCount() == 1
    assert tree.topLevelItem(0).text(0) == "Model"


def test_the_root_label_comes_from_the_caller(qtbot):
    # The root is not an OutlinerRow -- it has no Instance -- so the widget
    # renders it from a label the caller supplies.
    tree = _tree(qtbot, root_label="Drawing")
    assert tree.topLevelItem(0).text(0) == "Drawing"


def test_rows_nest_by_depth(qtbot):
    tree = _tree(qtbot, [_row(1, depth=0, label="Outer"), _row(2, depth=1, label="Inner")])
    root = tree.topLevelItem(0)

    assert root.childCount() == 1
    outer = root.child(0)
    assert outer.text(0) == "Outer"
    assert outer.childCount() == 1
    assert outer.child(0).text(0) == "Inner"


def test_siblings_stay_siblings(qtbot):
    tree = _tree(qtbot, [_row(1, depth=0), _row(2, depth=1), _row(3, depth=0)])
    root = tree.topLevelItem(0)
    assert root.childCount() == 2


def test_item_for_finds_a_row_by_instance_id(qtbot):
    tree = _tree(qtbot, [_row(7, label="Roof")])
    assert tree.item_for(7).text(0) == "Roof"
    assert tree.item_for(999) is None


def test_selecting_a_row_emits_its_instance_id(qtbot):
    tree = _tree(qtbot, [_row(5)])
    with qtbot.waitSignal(tree.instance_clicked) as blocker:
        tree.setCurrentItem(tree.item_for(5))
    assert blocker.args == [5]


def test_set_selected_ids_does_not_emit(qtbot):
    # Called on every selection change, including mid box-drag: it must not
    # bounce a signal back at the caller that just set the selection, and it
    # must not rebuild -- isSelected() alone can't tell a real in-place select
    # apart from a rebuild-then-reselect, so identity is asserted too.
    tree = _tree(qtbot, [_row(5)])
    item = tree.item_for(5)
    seen = []
    tree.instance_clicked.connect(seen.append)

    tree.set_selected_ids({5})

    assert tree.item_for(5).isSelected()
    assert tree.item_for(5) is item
    assert seen == []


def test_set_selected_ids_clears_a_previous_highlight(qtbot):
    tree = _tree(qtbot, [_row(1), _row(2)])
    item1 = tree.item_for(1)
    item2 = tree.item_for(2)
    tree.set_selected_ids({1})

    tree.set_selected_ids({2})

    assert not tree.item_for(1).isSelected()
    assert tree.item_for(2).isSelected()
    # Same objects throughout -- a rebuild-then-reselect would swap these out
    # for new QTreeWidgetItems and still satisfy the isSelected() checks above.
    assert tree.item_for(1) is item1
    assert tree.item_for(2) is item2


def test_rebuilding_rows_does_not_emit_a_selection_signal(qtbot):
    # A prior selection is essential here: with nothing selected, Qt emits no
    # itemSelectionChanged during a rebuild regardless of the _applying guard,
    # so that case can't tell a real guard from a missing one.
    tree = _tree(qtbot, [_row(1)])
    tree.setCurrentItem(tree.item_for(1))
    seen = []
    tree.instance_clicked.connect(seen.append)

    tree.set_rows([_row(1), _row(2)], "Model")

    assert seen == []


def test_clicking_the_eye_column_emits_the_new_hidden_state(qtbot):
    tree = _tree(qtbot, [_row(3, hidden=False)])
    item = tree.item_for(3)

    with qtbot.waitSignal(tree.hide_toggled) as blocker:
        tree.itemClicked.emit(item, OutlinerTree.EYE_COLUMN)

    assert blocker.args == [3, True]


def test_clicking_the_eye_on_a_hidden_row_asks_to_unhide(qtbot):
    tree = _tree(qtbot, [_row(3, hidden=True)])
    item = tree.item_for(3)

    with qtbot.waitSignal(tree.hide_toggled) as blocker:
        tree.itemClicked.emit(item, OutlinerTree.EYE_COLUMN)

    assert blocker.args == [3, False]


def test_clicking_the_label_column_does_not_toggle_visibility(qtbot):
    tree = _tree(qtbot, [_row(3)])
    seen = []
    tree.hide_toggled.connect(lambda *a: seen.append(a))

    tree.itemClicked.emit(tree.item_for(3), OutlinerTree.LABEL_COLUMN)

    assert seen == []


def test_double_click_emits_activated(qtbot):
    tree = _tree(qtbot, [_row(9)])
    with qtbot.waitSignal(tree.instance_activated) as blocker:
        tree.itemDoubleClicked.emit(tree.item_for(9), OutlinerTree.LABEL_COLUMN)
    assert blocker.args == [9]


def test_editing_a_label_requests_a_rename(qtbot):
    tree = _tree(qtbot, [_row(4, label="Wall")])
    item = tree.item_for(4)

    with qtbot.waitSignal(tree.rename_requested) as blocker:
        item.setText(OutlinerTree.LABEL_COLUMN, "North Wing")

    assert blocker.args == [4, "North Wing"]


def test_an_inherited_hidden_row_is_greyed_but_keeps_its_own_flag(qtbot):
    # The eye must still show "visible" -- the row's own hidden is False, and
    # clicking it would hide it for real, which is different from what its
    # ancestor is doing.
    tree = _tree(qtbot, [_row(1, hidden=True), _row(2, depth=1, inherited_hidden=True)])
    child = tree.item_for(2)

    assert child.data(OutlinerTree.LABEL_COLUMN, Qt.ItemDataRole.UserRole + 1) is False
    assert tree.is_dimmed(child)


def test_a_tag_hidden_row_is_greyed(qtbot):
    tree = _tree(qtbot, [_row(1, tag_hidden=True)])
    assert tree.is_dimmed(tree.item_for(1))


def test_a_plain_row_is_not_greyed(qtbot):
    tree = _tree(qtbot, [_row(1)])
    assert not tree.is_dimmed(tree.item_for(1))


def test_the_active_path_is_expanded(qtbot):
    tree = _tree(
        qtbot, [_row(1, on_active_path=True), _row(2, depth=1, label="Inner")]
    )
    assert tree.item_for(1).isExpanded()


def test_a_component_row_gets_a_different_icon_than_a_group(qtbot):
    tree = _tree(qtbot, [_row(1, is_component=False), _row(2, is_component=True)])
    group_icon = tree.item_for(1).icon(OutlinerTree.LABEL_COLUMN)
    component_icon = tree.item_for(2).icon(OutlinerTree.LABEL_COLUMN)
    assert not group_icon.isNull()
    assert not component_icon.isNull()
    assert group_icon.cacheKey() != component_icon.cacheKey()


# --- extra coverage: three-level nesting (M7.3 Task 9 gap closure) --------
#
# Task 5's row derivation shipped with a reviewer-confirmed coverage gap:
# nothing tested depth or inherited_hidden at three or more levels
# (grandparent -> parent -> grandchild). That gap lives in
# model_queries.outliner_rows (see tests/test_outliner_rows.py) -- this widget
# only renders whatever inherited_hidden Task 5 already computed for each row;
# _apply_visibility is a per-row `inherited_hidden or tag_hidden` OR with no
# reference to tree position or ancestor state, so it has nothing depth-
# dependent to exercise. A three-level "does dimming reach a grandchild" test
# here would combine two facts already proven independently -- structural
# nesting (below) and the OR itself (test_an_inherited_hidden_row_is_greyed...
# above) -- without adding coverage, so it was removed rather than kept
# alongside a redundant assertion.


def test_the_tree_only_supports_single_selection(qtbot):
    # ExtendedSelection advertised a multi-select the rest of the pipeline
    # could not honour: _on_selection_changed emits one instance id, and
    # MainWindow replaces the whole Selection with it, so a shift-click range
    # visually snapped back to one row. SingleSelection is honest about the
    # contract actually shipped; a full id set is M7.4 scope.
    tree = _tree(qtbot, [_row(1), _row(2)])
    assert tree.selectionMode() == QAbstractItemView.SelectionMode.SingleSelection


def test_rows_nest_three_levels_deep(qtbot):
    tree = _tree(
        qtbot,
        [
            _row(1, depth=0, label="Grandparent"),
            _row(2, depth=1, label="Parent"),
            _row(3, depth=2, label="Grandchild"),
        ],
    )
    root = tree.topLevelItem(0)

    grandparent = root.child(0)
    assert grandparent.text(0) == "Grandparent"
    assert grandparent.childCount() == 1

    parent = grandparent.child(0)
    assert parent.text(0) == "Parent"
    assert parent.childCount() == 1

    grandchild = parent.child(0)
    assert grandchild.text(0) == "Grandchild"
    assert grandchild.childCount() == 0
