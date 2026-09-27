"""MainWindow wiring for selection: registration, shared selection, Delete,
clear-on-undo, status count."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.model.annotation import Dimension


@pytest.fixture
def win(qtbot):
    from bermake.ui.main_window import MainWindow
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_select_and_eraser_registered(win):
    mgr = win._tool_manager
    assert mgr.activate_by_shortcut("Space")
    assert mgr.active.name == "Select"
    assert mgr.activate_by_shortcut("E")
    assert mgr.active.name == "Eraser"


def test_selection_is_shared_with_viewport(win):
    assert win._viewport.selection is win._selection


def _quad(win):
    s = win.scene
    a = s.add_vertex(np.array([-1, -1, 0], dtype=np.float32))
    b = s.add_vertex(np.array([1, -1, 0], dtype=np.float32))
    c = s.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    d = s.add_vertex(np.array([-1, 1, 0], dtype=np.float32))
    fid = s.add_face_from_loop((a, b, c, d))
    return s, fid


def test_delete_selection_removes_face_and_is_undoable(win):
    scene, fid = _quad(win)
    f0 = len(list(scene.faces_iter()))
    win._selection.replace(faces=[fid])
    win._on_delete_selection()
    assert len(list(scene.faces_iter())) == f0 - 1
    assert win._selection.is_empty()
    win._command_stack.undo()
    assert len(list(scene.faces_iter())) == f0


def test_delete_selected_edge_cascades_face(win):
    scene, fid = _quad(win)
    e = next(iter(scene.edges_iter())).id
    f0 = len(list(scene.faces_iter()))
    win._selection.replace(edges=[e])
    win._on_delete_selection()
    assert len(list(scene.faces_iter())) == f0 - 1


# M7.1 T9 (#46): this test used to assert that undo ALWAYS empties the
# selection, encoding the old blanket `self._selection.clear()` bug. Under
# the prune contract (_on_after_undo_redo keeps ids still live in the active
# context, drops the rest), the assertion is wrong: RemoveFaceCommand.undo
# restores the face with its ORIGINAL id, so `fid` is live again by the time
# the post-undo hook runs, and a selection re-pointed at that same id must
# survive -- not be wiped.
def test_undo_of_delete_keeps_selection_for_the_restored_entity(win):
    scene, fid = _quad(win)
    win._selection.replace(faces=[fid])
    win._on_delete_selection()
    # Simulate re-pointing the selection at the same id before undo runs.
    win._selection.replace(faces=[fid])
    win._command_stack.undo()
    assert fid in win._selection.faces


def test_empty_selection_delete_is_noop(win):
    scene, fid = _quad(win)
    f0 = len(list(scene.faces_iter()))
    win._on_delete_selection()
    assert len(list(scene.faces_iter())) == f0
    assert not win._command_stack.can_undo


# ---------------------------------------------------------------------------
# M7d Task 11: closes a real gap -- before this task, an annotation-only
# selection fell through the instance/edge/face branches above (all of which
# found nothing to do) and _on_delete_selection unconditionally cleared the
# selection anyway, silently dropping the annotation with no undo record.
# ---------------------------------------------------------------------------

def test_delete_removes_annotation_only_selection_and_undo_restores(win):
    ann = Dimension(
        win._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    win._model.active_context.annotations.append(ann)
    win._selection.replace(annotations=[ann.id])

    win._on_delete_selection()

    assert win._model.active_context.annotations == []
    assert win._selection.is_empty()

    assert win._command_stack.undo()
    assert len(win._model.active_context.annotations) == 1


# ---------------------------------------------------------------------------
# Fix wave: a mixed annotation + geometry/instance selection must delete as
# ONE undo-stack entry, so a single Ctrl+Z restores everything at once.
# ---------------------------------------------------------------------------

def test_delete_mixed_annotation_and_face_selection_is_one_undo(win):
    scene, fid = _quad(win)
    f0 = len(list(scene.faces_iter()))
    ann = Dimension(
        win._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    win._model.active_context.annotations.append(ann)
    win._selection.replace(faces=[fid], annotations=[ann.id])

    win._on_delete_selection()

    assert len(list(scene.faces_iter())) == f0 - 1
    assert win._model.active_context.annotations == []
    assert win._selection.is_empty()
    assert len(win._command_stack._undo) == 1, "one delete must be one undo-stack entry"

    assert win._command_stack.undo()

    assert len(list(scene.faces_iter())) == f0
    assert len(win._model.active_context.annotations) == 1


def test_delete_mixed_annotation_and_instance_selection_is_one_undo(win):
    d = win._model.new_definition("Box", is_group=False)
    inst = win._model.new_instance(d)
    win._model.active_context.children.append(inst)
    ann = Dimension(
        win._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    win._model.active_context.annotations.append(ann)
    win._selection.replace(instances=[inst.id], annotations=[ann.id])

    win._on_delete_selection()

    assert inst not in win._model.active_context.children
    assert win._model.active_context.annotations == []
    assert win._selection.is_empty()
    assert len(win._command_stack._undo) == 1, "one delete must be one undo-stack entry"

    assert win._command_stack.undo()

    assert inst in win._model.active_context.children
    assert len(win._model.active_context.annotations) == 1
