"""Tests for Edit-menu handlers: Make Group, Make Component, Explode, Make Unique,
and instance deletion via the delete key."""

from __future__ import annotations

import numpy as np
import pytest

from bermake.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def _add_triangle(win):
    """Add a triangle face to the root mesh and return the face id."""
    s = win._model.root.mesh
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    c = s.add_vertex(np.array([0, 1, 0], np.float32))
    return s.add_face_from_loop([a, b, c])


# ---------------------------------------------------------------------------
# Make Group
# ---------------------------------------------------------------------------

def test_make_group_action_creates_instance_from_selection(win):
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    win._on_make_group()
    assert len(win._model.root.children) == 1
    assert win._model.root.children[0].definition.is_group is True
    # Undo restores loose geometry:
    win._command_stack.undo()
    assert len(win._model.root.children) == 0
    assert len(list(win._model.root.mesh.faces_iter())) == 1


def test_make_group_noop_when_no_entity_selection(win):
    """Handler is a no-op (no crash) when nothing is selected."""
    win._on_make_group()
    assert len(win._model.root.children) == 0


def test_make_group_selection_replaced_with_instance(win):
    """After grouping, the selection should contain the new instance."""
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    win._on_make_group()
    assert len(win._selection.instances) == 1


# ---------------------------------------------------------------------------
# Make Component
# ---------------------------------------------------------------------------

def test_make_component_action_creates_non_group_instance(win, monkeypatch):
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    monkeypatch.setattr(win, "_prompt_component_name", lambda default: "MyComp")
    win._on_make_component()
    assert len(win._model.root.children) == 1
    child_def = win._model.root.children[0].definition
    assert child_def.is_group is False
    assert child_def.name == "MyComp"


def test_make_component_aborts_when_dialog_cancelled(win, monkeypatch):
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    monkeypatch.setattr(win, "_prompt_component_name", lambda default: None)
    win._on_make_component()
    # Nothing should have been created.
    assert len(win._model.root.children) == 0


def test_make_component_noop_when_no_entity_selection(win, monkeypatch):
    monkeypatch.setattr(win, "_prompt_component_name", lambda default: "X")
    win._on_make_component()
    assert len(win._model.root.children) == 0


# ---------------------------------------------------------------------------
# Explode
# ---------------------------------------------------------------------------

def test_explode_round_trip(win):
    """Make Group then Explode restores the loose face in the parent mesh."""
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    win._on_make_group()
    assert len(win._model.root.children) == 1

    # Select the created instance and explode it.
    inst = win._model.root.children[0]
    win._selection.replace(instances=[inst.id])
    win._on_explode()
    assert len(win._model.root.children) == 0
    assert len(list(win._model.root.mesh.faces_iter())) == 1


def test_explode_noop_when_no_instance_selected(win):
    """Explode is a no-op when no instance is selected."""
    win._on_explode()  # Should not raise.


# ---------------------------------------------------------------------------
# Instance delete
# ---------------------------------------------------------------------------

def test_delete_instance_removes_it(win):
    """Delete key with a selected instance removes the instance."""
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    win._on_make_group()
    inst = win._model.root.children[0]

    win._selection.replace(instances=[inst.id])
    win._on_delete_selection()
    assert len(win._model.root.children) == 0


def test_delete_instance_undoable(win):
    """Deleting an instance via keyboard is undoable."""
    f = _add_triangle(win)
    win._selection.replace(faces=[f])
    win._on_make_group()
    inst = win._model.root.children[0]

    win._selection.replace(instances=[inst.id])
    win._on_delete_selection()
    assert len(win._model.root.children) == 0

    win._command_stack.undo()
    assert len(win._model.root.children) == 1


# ---------------------------------------------------------------------------
# Final review I5: a bare selected vertex must not ride into the group
# ---------------------------------------------------------------------------


def _two_quads(win):
    """Two disjoint unit quads in the root mesh, and their ids.

    Disjoint on purpose: a vertex of the second quad is a vertex belonging to
    no selected edge or face when only the first quad is selected, which is
    exactly the state a window box-select produces (it takes every vertex
    whose point falls in the rect while taking only fully-enclosed faces).
    """
    s = win._model.root.mesh
    first = [
        s.add_vertex(np.array([0, 0, 0], np.float32)),
        s.add_vertex(np.array([1, 0, 0], np.float32)),
        s.add_vertex(np.array([1, 1, 0], np.float32)),
        s.add_vertex(np.array([0, 1, 0], np.float32)),
    ]
    second = [
        s.add_vertex(np.array([5, 0, 0], np.float32)),
        s.add_vertex(np.array([6, 0, 0], np.float32)),
        s.add_vertex(np.array([6, 1, 0], np.float32)),
        s.add_vertex(np.array([5, 1, 0], np.float32)),
    ]
    return s.add_face_from_loop(first), s.add_face_from_loop(second), first, second


def test_make_group_ignores_a_bare_vertex_from_other_geometry(win):
    """The bug: both handlers gate on `sel.edges or sel.faces`, so a face plus
    an unrelated bare vertex passed the gate and handed that vertex to
    MakeGroupCommand. The command copied it into the group and then failed to
    remove it from the parent (`remove_vertex` raises while an edge still uses
    it, and group_commands swallows that), leaving a duplicated, orphaned
    vertex: a 5-vertex group mesh for a 4-corner quad."""
    face_a, _face_b, _va, vb = _two_quads(win)
    win._selection.replace(faces=[face_a], vertices=[vb[0]])
    win._on_make_group()

    group_mesh = win._model.root.children[0].definition.mesh
    assert len(list(group_mesh.vertices_iter())) == 4
    # The other quad is untouched in the parent: still four corners, one face.
    parent = win._model.root.mesh
    assert len(list(parent.faces_iter())) == 1
    assert len(list(parent.vertices_iter())) == 4


def test_make_component_ignores_a_bare_vertex_from_other_geometry(win, monkeypatch):
    face_a, _face_b, _va, vb = _two_quads(win)
    monkeypatch.setattr(win, "_prompt_component_name", lambda default: "C")
    win._selection.replace(faces=[face_a], vertices=[vb[0]])
    win._on_make_component()

    comp_mesh = win._model.root.children[0].definition.mesh
    assert len(list(comp_mesh.vertices_iter())) == 4


def test_make_group_still_takes_a_vertex_that_belongs_to_the_selection(win):
    """The fix drops BARE vertices only. A vertex of a selected face was
    already going into the group by way of that face's loop and must keep
    going, whether or not it is also selected in its own right."""
    face_a, _face_b, va, _vb = _two_quads(win)
    win._selection.replace(faces=[face_a], vertices=[va[0]])
    win._on_make_group()

    group_mesh = win._model.root.children[0].definition.mesh
    assert len(list(group_mesh.vertices_iter())) == 4
    assert len(list(group_mesh.faces_iter())) == 1
