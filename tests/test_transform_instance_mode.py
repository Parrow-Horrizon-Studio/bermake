"""Tests: Move / Rotate / Scale operate on a selected instance (M4e T17).

Unit-level: verify TransformInstanceCommand composes the delta correctly.
Tool-level: drive each tool's gesture with an instance selected and assert
the instance's transform changed (and undo restores).
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.commands.instance_commands import TransformInstanceCommand
from bermake.geometry.transforms import mat_translate, mat_rotate, mat_scale
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.scale_tool import ScaleTool
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


def _release(x=0.0, y=0.0):
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


def _make_model_with_group() -> tuple[Model, object]:
    """Return (model, inst) where inst is a group placed at the origin."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    # Give it some geometry so local_aabb() is non-None
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
# Unit-level: TransformInstanceCommand round-trip
# ---------------------------------------------------------------------------

def test_transform_instance_command_translate():
    """A translate delta applied to an instance transform composes correctly."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)

    delta = mat_translate([3, 0, 0])
    cmd = TransformInstanceCommand(inst, delta @ inst.transform)
    cmd.do(m)
    assert np.allclose(inst.transform[:3, 3], [3, 0, 0])


def test_transform_instance_command_undo():
    """Undo restores the original transform."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)

    original = inst.transform.copy()
    delta = mat_translate([5, 2, 1])
    cmd = TransformInstanceCommand(inst, delta @ inst.transform)
    cmd.do(m)
    cmd.undo(m)
    assert np.allclose(inst.transform, original)


def test_transform_instance_command_rotate():
    """A 90° Z-rotate delta changes the rotation part of the transform."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)

    center = np.array([0, 0, 0], np.float64)
    axis = np.array([0, 0, 1], np.float64)
    delta = mat_rotate(center, axis, math.pi / 2)
    cmd = TransformInstanceCommand(inst, delta @ inst.transform)
    cmd.do(m)
    # After 90° Z-rotation about origin, the X column of R should be ≈ [0,1,0]
    assert np.allclose(inst.transform[:3, 0], [0, 1, 0], atol=1e-6)


def test_transform_instance_command_scale():
    """A 2× scale on X composes into the transform."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)

    anchor = np.array([0, 0, 0])
    factors = np.array([2.0, 1.0, 1.0])
    delta = mat_scale(anchor, factors)
    cmd = TransformInstanceCommand(inst, delta @ inst.transform)
    cmd.do(m)
    # Scale in X: the [0,0] element of the transform should be 2
    assert np.allclose(inst.transform[0, 0], 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Tool-level: MoveTool instance-mode
# ---------------------------------------------------------------------------

def test_move_instance_mode_moves_instance(qtbot):
    """MoveTool with an instance selected moves the instance, not vertices."""
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([3, 0, 0]))
    tool.on_mouse_release(_release(), _snap([3, 0, 0]))

    assert np.allclose(inst.transform[:3, 3], [3, 0, 0], atol=1e-5)
    assert stack.can_undo


def test_move_instance_mode_undo(qtbot):
    """Undo after moving an instance restores the original transform."""
    m, inst = _make_model_with_group()
    original = inst.transform.copy()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()
    tool.activate(_ctx(m, stack, sel))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release(), _snap([5, 0, 0]))

    stack.undo()
    assert np.allclose(inst.transform, original, atol=1e-9)


def test_move_entity_mode_unchanged(qtbot):
    """With only entities selected, MoveTool still moves vertices (regression)."""
    from bermake.scene.scene import Scene
    s = Scene()
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([1, 0, 0], np.float32))
    e = s.add_edge(a, b)

    sel = Selection()
    sel.replace(edges=[e])
    stack = CommandStack()
    m = Model()
    tool = MoveTool()
    # Use a minimal ToolContext with scene (entity mode)
    ctx = ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
    )
    tool.activate(ctx)
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_release(_release(), _snap([0, 0, 3]))
    assert np.allclose(s.vertex(a).position, [0, 0, 3], atol=1e-5)
    assert np.allclose(s.vertex(b).position, [1, 0, 3], atol=1e-5)


def test_move_instance_mode_apply_typed_value(qtbot):
    """apply_typed_value in instance-mode applies the typed distance along the drag direction."""
    from bermake.units import Units
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = MoveTool()

    ctx = ToolContext(
        scene=m.active_scene,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
        units_provider=lambda: Units(),
    )
    tool.activate(ctx)
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_move(_press(), _snap([1, 0, 0]))  # drag direction = +X

    result = tool.apply_typed_value("10m", Units())
    assert result is True
    # Should have moved 10 units along +X
    assert np.allclose(inst.transform[:3, 3], [10, 0, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# Tool-level: RotateTool instance-mode
# ---------------------------------------------------------------------------

def test_rotate_instance_mode_rotates_instance(qtbot, monkeypatch):
    """RotateTool with an instance selected rotates the instance's transform."""
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = RotateTool()
    tool.activate(_ctx(m, stack, sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))

    # 3-click: center -> start -> end (90° around Z)
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))  # center
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))  # start dir = +X
    tool.on_mouse_press(_press(), _snap([0, 1, 0]))  # end dir = +Y -> +90

    assert stack.can_undo
    # After 90° Z-rotation about origin, X column ≈ [0, 1, 0]
    assert np.allclose(inst.transform[:3, 0], [0, 1, 0], atol=1e-4)


def test_rotate_instance_mode_undo(qtbot, monkeypatch):
    """Undo after rotating an instance restores the original transform."""
    m, inst = _make_model_with_group()
    original = inst.transform.copy()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = RotateTool()
    tool.activate(_ctx(m, stack, sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))
    tool.on_mouse_press(_press(), _snap([0, 1, 0]))

    stack.undo()
    assert np.allclose(inst.transform, original, atol=1e-9)


def test_rotate_entity_mode_unchanged(qtbot, monkeypatch):
    """With only entities selected, RotateTool still rotates vertices (regression)."""
    s = Scene()
    a = s.add_vertex(np.array([1, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    e = s.add_edge(a, b)

    sel = Selection()
    sel.replace(edges=[e])
    stack = CommandStack()
    m = Model()
    ctx = ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
    )
    tool = RotateTool()
    tool.activate(ctx)
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))  # center
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))  # start
    tool.on_mouse_press(_press(), _snap([0, 1, 0]))  # 90°

    assert np.allclose(s.vertex(a).position, [0, 1, 0], atol=1e-4)
    assert np.allclose(s.vertex(b).position, [0, 2, 0], atol=1e-4)


def test_rotate_instance_mode_apply_typed_value(qtbot, monkeypatch):
    """apply_typed_value in rotate instance-mode emits TransformInstanceCommand."""
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = RotateTool()
    tool.activate(_ctx(m, stack, sel))
    monkeypatch.setattr(tool, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))

    tool.on_mouse_press(_press(), _snap([0, 0, 0]))  # center
    tool.on_mouse_press(_press(), _snap([1, 0, 0]))  # start dir = +X
    # Move slightly to establish direction (positive)
    tool.on_mouse_move(_press(), _snap([0, 1, 0]))

    result = tool.apply_typed_value("90", None)
    assert result is True
    # 90° Z-rotation: X column ≈ [0, 1, 0]
    assert np.allclose(inst.transform[:3, 0], [0, 1, 0], atol=1e-4)


# ---------------------------------------------------------------------------
# Tool-level: ScaleTool instance-mode
# ---------------------------------------------------------------------------

def test_scale_instance_mode_scales_instance(qtbot, monkeypatch):
    """ScaleTool with an instance selected scales the instance's transform."""
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = ScaleTool()
    tool.activate(_ctx(m, stack, sel))

    # Manually inject a grip (corner at [1,1,1], anchor at [0,0,0])
    from bermake.tools.transform_support import GripSpec
    grip = GripSpec(
        position=np.array([1, 1, 1], np.float32),
        opposite=np.array([0, 0, 0], np.float32),
        axes=(0, 1, 2),
    )
    monkeypatch.setattr(tool, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(tool, "_cursor_world", lambda ev: np.array([2, 2, 2], np.float32))

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    rel = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tool.on_mouse_press(press, None)
    tool.on_mouse_release(rel, None)

    assert stack.can_undo
    # The transform should have been modified (scale applied)
    # Diagonal scale by 2x from origin: transform[i,i] should be 2
    assert not np.allclose(inst.transform, np.eye(4), atol=1e-5)


def test_scale_instance_mode_undo(qtbot, monkeypatch):
    """Undo after scaling an instance restores the original transform."""
    m, inst = _make_model_with_group()
    original = inst.transform.copy()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = ScaleTool()
    tool.activate(_ctx(m, stack, sel))

    from bermake.tools.transform_support import GripSpec
    grip = GripSpec(
        position=np.array([1, 1, 1], np.float32),
        opposite=np.array([0, 0, 0], np.float32),
        axes=(0, 1, 2),
    )
    monkeypatch.setattr(tool, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(tool, "_cursor_world", lambda ev: np.array([2, 2, 2], np.float32))

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    rel = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tool.on_mouse_press(press, None)
    tool.on_mouse_release(rel, None)
    stack.undo()
    assert np.allclose(inst.transform, original, atol=1e-9)


def test_scale_entity_mode_unchanged(qtbot, monkeypatch):
    """With only entities selected, ScaleTool still scales vertices (regression)."""
    s = Scene()
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    c = s.add_vertex(np.array([2, 2, 0], np.float32))
    d = s.add_vertex(np.array([0, 2, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])

    sel = Selection()
    sel.replace(faces=[f])
    stack = CommandStack()
    m = Model()
    ctx = ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        selection=sel,
        model=m,
    )
    tool = ScaleTool()
    tool.activate(ctx)

    from bermake.tools.transform_support import GripSpec
    grip = GripSpec(
        position=np.array([2, 2, 0], np.float32),
        opposite=np.array([0, 0, 0], np.float32),
        axes=(0, 1),
    )
    monkeypatch.setattr(tool, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(tool, "_cursor_world", lambda ev: np.array([4, 4, 0], np.float32))

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    rel = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tool.on_mouse_press(press, None)
    tool.on_mouse_release(rel, None)
    assert np.allclose(s.vertex(c).position, [4, 4, 0])
    assert np.allclose(s.vertex(b).position, [4, 0, 0])


def test_scale_instance_world_aabb(qtbot):
    """_instance_world_aabb returns the correct world bounding box for an instance."""
    m = Model()
    d = m.new_definition("G", is_group=True)
    # local AABB: [0,0,0] to [1,1,1]
    d.mesh.add_vertex(np.array([0, 0, 0], np.float32))
    d.mesh.add_vertex(np.array([1, 1, 1], np.float32))
    # Translate instance by [5, 0, 0]
    from bermake.geometry.transforms import mat_translate
    inst = m.new_instance(d, mat_translate([5, 0, 0]))
    m.root.children.append(inst)

    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = ScaleTool()
    tool.activate(_ctx(m, stack, sel))

    # The world AABB should be [5,0,0] to [6,1,1]
    assert tool._lo is not None
    assert np.allclose(tool._lo, [5, 0, 0], atol=1e-5)
    assert np.allclose(tool._hi, [6, 1, 1], atol=1e-5)


def test_scale_instance_mode_apply_typed_value(qtbot, monkeypatch):
    """apply_typed_value in scale instance-mode emits TransformInstanceCommand."""
    m, inst = _make_model_with_group()
    sel = Selection()
    sel.replace(instances={inst.id})
    stack = CommandStack()
    tool = ScaleTool()
    tool.activate(_ctx(m, stack, sel))

    from bermake.tools.transform_support import GripSpec
    grip = GripSpec(
        position=np.array([1, 1, 1], np.float32),
        opposite=np.array([0, 0, 0], np.float32),
        axes=(0, 1, 2),
    )
    monkeypatch.setattr(tool, "_pick_grip", lambda ev: grip)
    monkeypatch.setattr(tool, "_cursor_world", lambda ev: np.array([2, 2, 2], np.float32))

    press = QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(0, 0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    tool.on_mouse_press(press, None)

    result = tool.apply_typed_value("3.0", None)
    assert result is True
    assert stack.can_undo
    # Not identity anymore
    assert not np.allclose(inst.transform, np.eye(4), atol=1e-5)
