"""Regression: Esc must exit a group/component edit context even when the
selection is empty (entering a group clears the selection). The bug was that
MainWindow._on_escape only forwarded Esc to the tool when it had an "active
gesture" (box-select or a non-empty selection); inside a group with nothing
selected it deactivated the tool instead of exiting — a dead-end."""

from __future__ import annotations

import numpy as np
import pytest

from bermake.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _make_and_enter_group(win: MainWindow):
    s = win._model.root.mesh
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    c = s.add_vertex(np.array([0, 1, 0], np.float32))
    f = s.add_face_from_loop([a, b, c])
    win._selection.replace(faces=[f])
    win._on_make_group()  # creates the group instance + selects it
    inst = win._model.root.children[0]
    win._model.enter(inst)
    win._selection.clear()  # entering a group clears the selection
    win._rebuild_tool_context()
    return inst


def test_escape_inside_group_empty_selection_exits(win):
    win._tool_manager.activate_by_shortcut("Space")  # SelectTool
    inst = _make_and_enter_group(win)
    assert win._model.active_path == [inst]
    assert win._selection.is_empty()

    win._on_escape()

    # The fix: Esc exits one level instead of deactivating the tool.
    assert win._model.active_path == []
    assert win._model.active_context is win._model.root


def test_escape_at_root_empty_selection_deactivates_tool(win):
    # Regression guard: at the root with no gesture/selection, Esc still
    # deactivates the active tool (the pre-existing behavior is preserved).
    win._tool_manager.activate_by_shortcut("Space")
    win._selection.clear()
    assert win._model.active_path == []

    win._on_escape()

    assert win._tool_manager.active is None
