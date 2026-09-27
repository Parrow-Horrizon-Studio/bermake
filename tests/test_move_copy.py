"""Tests for Move-copy (Ctrl during Move → new instance) — M4e Task 18.

Unit-level: CreateInstanceCommand places a second instance of the same definition
at the translated position while the original stays put.

Tool-level: MoveTool in instance-mode with Ctrl held at release creates a new
instance, leaves the original unchanged, updates the selection, and is undoable.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.commands.instance_commands import CreateInstanceCommand
from bermake.geometry.transforms import mat_translate
from bermake.model.model import Model
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _press(x=0.0, y=0.0):
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _release_ctrl(x=0.0, y=0.0):
    """Release event with Ctrl held."""
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
    )


def _release(x=0.0, y=0.0):
    """Release event without Ctrl."""
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind,
        world_position=np.asarray(pos, np.float32),
        axis=None, vertex_id=None, edge_id=None, edge_t=None,
    )


def _make_model_with_component():
    """Return (model, inst) — a non-group component placed at the origin."""
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    d.mesh.add_vertex(np.array([1, 1, 1], np.float32))
    inst = m.new_instance(d)
    m.root.children.append(inst)
    return m, inst


def _ctx(model: Model, stack: CommandStack, selection: Selection) -> ToolContext:
    return ToolContext(
        scene=model.active_scene,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=selection,
        model=model,
    )


# ---------------------------------------------------------------------------
# Unit-level: CreateInstanceCommand round-trip
# ---------------------------------------------------------------------------

def test_move_copy_adds_instance_of_same_definition():
    """CreateInstanceCommand at translated position leaves original in place."""
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    base = m.new_instance(d)
    m.root.children.append(base)

    cmd = CreateInstanceCommand(m.root, d, mat_translate([5, 0, 0]) @ base.transform)
    cmd.do(m)

    assert len(m.root.children) == 2
    assert all(child.definition is d for child in m.root.children)
    assert np.allclose(m.root.children[1].transform[:3, 3], [5, 0, 0])


def test_move_copy_unit_undo():
    """Undo of CreateInstanceCommand removes the copy; original untouched."""
    m = Model()
    d = m.new_definition("Chair", is_group=False)
    base = m.new_instance(d)
    m.root.children.append(base)
    original_transform = base.transform.copy()

    cmd = CreateInstanceCommand(m.root, d, mat_translate([5, 0, 0]) @ base.transform)
    cmd.do(m)
    assert len(m.root.children) == 2

    cmd.undo(m)
    assert len(m.root.children) == 1
    assert m.root.children[0] is base
    assert np.allclose(base.transform, original_transform)


# ---------------------------------------------------------------------------
# Tool-level: MoveTool Move-copy gesture
# ---------------------------------------------------------------------------

def test_move_copy_tool_creates_second_instance(qtbot):
    """Ctrl-held at release: two instances of same definition, original unchanged."""
    m, inst = _make_model_with_component()
    original_transform = inst.transform.copy()

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release_ctrl(5, 0), _snap([5, 0, 0]))

    # Two instances now exist
    assert len(m.root.children) == 2
    # Both share the same definition
    assert all(child.definition is inst.definition for child in m.root.children)
    # Original is unchanged
    assert np.allclose(inst.transform, original_transform, atol=1e-9)
    # New instance is at x=5
    new_inst = next(c for c in m.root.children if c is not inst)
    assert np.allclose(new_inst.transform[:3, 3], [5, 0, 0], atol=1e-5)


def test_move_copy_tool_undo_removes_copy(qtbot):
    """Undo after Move-copy removes the new instance; original untouched."""
    m, inst = _make_model_with_component()
    original_transform = inst.transform.copy()

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release_ctrl(5, 0), _snap([5, 0, 0]))

    assert len(m.root.children) == 2
    assert stack.can_undo

    stack.undo()

    assert len(m.root.children) == 1
    assert m.root.children[0] is inst
    assert np.allclose(inst.transform, original_transform, atol=1e-9)


def test_move_copy_tool_selection_updated_to_new_instance(qtbot):
    """After Move-copy, the selection holds the new instance, not the original."""
    m, inst = _make_model_with_component()

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release_ctrl(5, 0), _snap([5, 0, 0]))

    new_inst = next(c for c in m.root.children if c is not inst)
    assert new_inst.id in sel.instances
    assert inst.id not in sel.instances


def test_move_copy_without_ctrl_does_not_copy(qtbot):
    """Without Ctrl, normal Move leaves only one instance (regression)."""
    m, inst = _make_model_with_component()

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release(5, 0), _snap([5, 0, 0]))

    assert len(m.root.children) == 1
    assert np.allclose(inst.transform[:3, 3], [5, 0, 0], atol=1e-5)


def test_move_copy_zero_delta_does_not_crash(qtbot):
    """Ctrl-held but no drag (zero delta) creates a coincident duplicate without crashing."""
    m, inst = _make_model_with_component()

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    # release at same position — zero delta
    tool.on_mouse_release(_release_ctrl(0, 0), _snap([0, 0, 0]))

    assert len(m.root.children) == 2
    # Both at origin
    for child in m.root.children:
        assert np.allclose(child.transform[:3, 3], [0, 0, 0], atol=1e-9)


def test_move_copy_keeps_a_co_selected_annotation_selected(qtbot):
    """Move-copy replaced the whole Selection, silently dropping a co-selected
    annotation instead of copying, moving, or reporting it (#96)."""
    from bermake.model.annotation import Dimension

    m, inst = _make_model_with_component()
    ann = Dimension(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.2, 0.0))
    m.active_context.annotations.append(ann)

    sel = Selection()
    sel.replace(instances={inst.id}, annotations={ann.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release_ctrl(5, 0), _snap([5, 0, 0]))

    assert ann.id in sel.annotations, "move-copy dropped the co-selected annotation"
