"""The Outliner drives the model and follows the viewport (M7.3 Task 10)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from bermake.ui.outliner_tree import OutlinerTree


def _square(window):
    scene = window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def _group(window):
    window._on_select_all()
    window._on_make_group()
    window._selection.clear()
    return window._model.root.children[-1]


def test_the_window_owns_a_properties_dock_and_an_outliner(main_window):
    assert main_window._properties_dock is not None
    assert isinstance(main_window._outliner, OutlinerTree)
    assert main_window._properties_dock.outliner() is main_window._outliner


def test_making_a_group_adds_an_outliner_row(main_window):
    _square(main_window)
    assert main_window._outliner.item_for(0) is None

    instance = _group(main_window)

    assert main_window._outliner.item_for(instance.id) is not None


def test_undo_removes_the_row_again(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._command_stack.undo()

    assert main_window._outliner.item_for(instance.id) is None


def test_clicking_a_row_selects_that_instance(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._outliner.instance_clicked.emit(instance.id)

    assert main_window._selection.instances == {instance.id}


def test_clicking_a_nested_row_re_roots_the_active_path(main_window):
    # Selection is only meaningful inside the active context, so selecting a
    # nested row must move the context to that row's parent chain -- otherwise
    # the id lands in a selection the active context cannot see.
    #
    # Two back-to-back _group() calls do NOT nest: _on_make_group bails out
    # unless edges or faces (not instances) are selected, so a second call
    # with only the first group's instance selected creates nothing and
    # returns that SAME instance. Genuine nesting requires entering the
    # first group and adding fresh geometry before grouping again -- the
    # same recipe tests/test_outliner_rows.py uses for its nesting cases.
    _square(main_window)
    outer = _group(main_window)
    main_window._model.enter(outer)
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    inner = main_window._model.active_context.children[-1]
    main_window._selection.clear()
    main_window._model.exit_one()

    main_window._outliner.instance_clicked.emit(inner.id)

    assert [i.id for i in main_window._model.active_path] == [outer.id]
    assert main_window._selection.instances == {inner.id}


def test_clicking_a_root_level_row_exits_back_to_the_root(main_window):
    _square(main_window)
    inner = _group(main_window)
    outer = _group(main_window)
    main_window._model.active_path = [outer]

    main_window._outliner.instance_clicked.emit(outer.id)

    assert main_window._model.active_path == []
    del inner


def test_double_clicking_a_row_enters_that_instance(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._outliner.instance_activated.emit(instance.id)

    assert [i.id for i in main_window._model.active_path] == [instance.id]


def test_the_eye_toggle_hides_through_an_undoable_command(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._outliner.hide_toggled.emit(instance.id, True)

    assert instance.hidden is True
    main_window._command_stack.undo()
    assert instance.hidden is False


def test_renaming_a_row_goes_through_an_undoable_command(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._outliner.rename_requested.emit(instance.id, "North Wing")

    assert instance.name == "North Wing"
    main_window._command_stack.undo()
    assert instance.name == ""


def test_a_viewport_selection_highlights_the_row(main_window):
    _square(main_window)
    instance = _group(main_window)

    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()

    assert main_window._outliner.item_for(instance.id).isSelected()


def test_selection_sync_does_not_rebuild_the_tree(main_window):
    # Rows must not be rebuilt on selection change: it fires many times a
    # second during a box-select drag.
    _square(main_window)
    instance = _group(main_window)
    rebuilds = []
    original = main_window._outliner.set_rows
    main_window._outliner.set_rows = lambda *a, **k: (rebuilds.append(1), original(*a, **k))[1]

    main_window._selection.replace(instances=[instance.id])
    main_window._refresh_selection_status()

    assert rebuilds == []


def test_a_stale_row_id_is_ignored(main_window):
    # An id that no longer resolves must be a no-op, not a crash.
    _square(main_window)
    _group(main_window)

    main_window._outliner.instance_clicked.emit(9999)
    main_window._outliner.instance_activated.emit(9999)
    main_window._outliner.hide_toggled.emit(9999, True)
    main_window._outliner.rename_requested.emit(9999, "x")

    assert main_window._model.active_path == []


def test_toggling_tag_visibility_rebuilds_the_outliner(main_window):
    # Spec 1.7: "Tag visibility toggled -> full rebuild (tag_hidden moved)".
    # visibility_changed was wired only to a viewport repaint + dirty-marking,
    # which left tag_hidden -- and the row's dimming -- stale.
    _square(main_window)
    instance = _group(main_window)
    tag = main_window._model.tags.add("Hidden Tag")
    instance.tag_id = tag.id
    main_window._rebuild_outliner()
    assert not main_window._outliner.is_dimmed(main_window._outliner.item_for(instance.id))

    main_window._model.tags.set_visible(tag.id, False)
    main_window._tags_page.visibility_changed.emit()

    assert main_window._outliner.is_dimmed(main_window._outliner.item_for(instance.id))


def test_clicking_the_eye_column_does_not_re_root_the_active_context(main_window, qtbot):
    # itemSelectionChanged fires (from the mouse press) before itemClicked --
    # so toggling visibility on a nested row used to yank the user into that
    # row's parent context a moment before the eye toggle itself ran.
    #
    # A real (not synthesized-via-signal) click is required to reproduce this
    # -- itemSelectionChanged only fires ahead of itemClicked as a side effect
    # of QAbstractItemView's actual mousePressEvent handling. The window must
    # also be shown: a QTreeWidget that has never been shown/laid out does
    # not deliver synthetic QTest mouse events to its items at all, fix or
    # no fix, which would make this test pass for the wrong reason.
    main_window.show()
    _square(main_window)
    outer = _group(main_window)
    main_window._model.enter(outer)
    _square(main_window)
    main_window._on_select_all()
    main_window._on_make_group()
    inner = main_window._model.active_context.children[-1]
    main_window._selection.clear()
    main_window._model.exit_one()
    main_window._rebuild_outliner()

    tree = main_window._outliner
    tree.item_for(outer.id).setExpanded(True)
    point = tree.visualRect(tree.indexFromItem(tree.item_for(inner.id), tree.EYE_COLUMN)).center()

    qtbot.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=point)

    assert main_window._model.active_path == []


def test_double_click_activation_rebuilds_the_outliner_once(main_window):
    # _on_active_context_changed (called from _on_outliner_activated) already
    # rebuilds the tree -- an extra explicit rebuild after it just repopulated
    # the same tree a second time for nothing.
    _square(main_window)
    instance = _group(main_window)
    calls = []
    original = main_window._rebuild_outliner
    main_window._rebuild_outliner = lambda: (calls.append(1), original())[1]

    main_window._on_outliner_activated(instance.id)

    assert calls == [1]


def test_a_no_op_rename_restores_the_items_canonical_label(main_window):
    # No command runs on a no-op rename, so nothing used to rebuild the tree
    # -- the row kept whatever raw text (padding whitespace included) the
    # user had just typed.
    _square(main_window)
    instance = _group(main_window)
    main_window._outliner.rename_requested.emit(instance.id, "Wall")
    item = main_window._outliner.item_for(instance.id)

    item.setText(0, "Wall  ")  # same name, extra whitespace the user typed

    assert main_window._outliner.item_for(instance.id).text(0) == "Wall"
