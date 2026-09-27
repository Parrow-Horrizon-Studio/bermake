"""Unit tests for the Wall tool — chaining, transform-aware conversion, VCB length."""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.model.model import Model
from bermake.tools.tool import ToolContext
from bermake.tools.wall_tool import WallTool


def _ctx(model, stack):
    return ToolContext(
        scene=model.active_scene,
        command_stack=stack,
        model=model,
        units_provider=lambda: None,
    )


def _snap(x, y, z=0.0):
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.ON_FACE,
        world_position=np.array([x, y, z], dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Face",
    )


def test_two_clicks_commit_one_wall_and_chain():
    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(0, 0))  # anchor
    assert tool.has_active_gesture
    assert len(model.active_context.children) == 0
    tool.on_mouse_press(None, _snap(4, 0))  # commit wall #1, chain
    assert len(model.active_context.children) == 1
    tool.on_mouse_press(None, _snap(4, 3))  # commit wall #2 from (4,0)
    assert len(model.active_context.children) == 2
    assert stack.can_undo


def test_escape_ends_chain_without_removing_committed_walls():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(0, 0))
    tool.on_mouse_press(None, _snap(4, 0))

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    assert not tool.has_active_gesture
    assert len(model.active_context.children) == 1  # committed wall stays


def test_thickness_height_drive_geometry():
    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.thickness = 0.3
    tool.height = 3.0
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(0, 0))
    tool.on_mouse_press(None, _snap(2, 0))
    wall = model.active_context.children[-1].definition
    zs = [v.position[2] for v in wall.mesh.vertices_iter()]
    ys = [v.position[1] for v in wall.mesh.vertices_iter()]
    assert max(zs) == 3.0
    assert max(ys) - min(ys) == 0.3


def test_apply_typed_value_commits_wall_at_typed_length():
    from bermake.units import Units

    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(0, 0))  # anchor at origin
    tool.on_mouse_move(None, _snap(1, 0))  # establishes cursor direction (+X)

    assert tool.apply_typed_value("5", Units()) is True
    assert len(model.active_context.children) == 1
    wall = model.active_context.children[-1].definition
    xs = [v.position[0] for v in wall.mesh.vertices_iter()]
    assert max(xs) - min(xs) == 5.0
    assert tool.has_active_gesture  # chained — ready for the next segment


def test_second_immediate_typed_value_does_not_fire_backward():
    """Regression: after a typed-length commit, _preview_tip must refresh to
    the new anchor (mirroring LineTool). Without that, a SECOND immediately
    typed length (no mouse move in between) computes its direction from the
    stale OLD tip against the NEW anchor and can fire the wall backward."""
    from bermake.units import Units

    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(0, 0))  # anchor at origin
    tool.on_mouse_move(None, _snap(1, 0))  # establishes cursor direction (+X)

    assert tool.apply_typed_value("4", Units()) is True
    assert len(model.active_context.children) == 1
    wall = model.active_context.children[-1].definition
    xs = [v.position[0] for v in wall.mesh.vertices_iter()]
    assert min(xs) == 0.0
    assert max(xs) == 4.0

    # No further mouse move: the preview tip should now equal the new anchor
    # (4,0,0), so a second immediately-typed length has zero direction and
    # must be rejected — not silently fire a second wall backward.
    assert tool.apply_typed_value("2", Units()) is False
    assert len(model.active_context.children) == 1


def test_locally_degenerate_vertical_segment_pushes_no_command():
    """Regression: on_mouse_press's world-space guard (norm < 1e-6) includes
    z, so two points differing only in z pass it. After _to_local_ground
    zeroes z, start == end in the active context's ground plane, wall_box
    returns empty geometry, and CreateWallCommand.do() adds nothing — but
    without the local-ground guard, execute() would still push an empty,
    undoable no-op command onto the stack."""
    model = Model()
    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(2, 3, 0))  # anchor
    tool.on_mouse_press(None, _snap(2, 3, 5))  # same local x,y, different z

    assert len(model.active_context.children) == 0
    assert not stack.can_undo


def test_world_to_local_conversion_inside_translated_group():
    """A group translated +10 in X: clicking world (10,0,0) -> (14,0,0) must
    build the wall from LOCAL (0,0,0) -> (4,0,0), not the doubled world offset."""
    from bermake.geometry.transforms import mat_translate

    model = Model()
    d = model.new_definition("G", is_group=True)
    inst = model.new_instance(d, mat_translate([10.0, 0.0, 0.0]))
    model.root.children.append(inst)
    model.enter(inst)
    assert not np.allclose(model.active_world_transform, np.eye(4))

    stack = CommandStack()
    tool = WallTool()
    tool.activate(_ctx(model, stack))
    tool.on_mouse_press(None, _snap(10, 0))
    tool.on_mouse_press(None, _snap(14, 0))

    wall = model.active_context.children[-1].definition
    xs = [v.position[0] for v in wall.mesh.vertices_iter()]
    assert min(xs) == 0.0
    assert max(xs) == 4.0
