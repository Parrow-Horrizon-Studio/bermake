"""Unit tests for the Rectangle tool."""

from __future__ import annotations

import numpy as np
import pytest


def _snap_at(world):
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    return SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Grid",
    )


def test_rectangle_tool_idle_overlay_is_empty():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    tool = RectangleTool()
    tool.activate(ToolContext(scene=Scene()))
    overlay = tool.overlay()
    assert overlay.rubber_band_segments.shape == (0, 3)


def test_rectangle_tool_first_click_starts_drag():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    # Scene is still empty until the second click commits.
    assert len(list(scene.vertices_iter())) == 0


def test_rectangle_tool_two_clicks_commit_four_verts_four_edges_one_face():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _snap_at((3.0, 2.0, 0.0)))  # type: ignore[arg-type]

    assert len(list(scene.vertices_iter())) == 4
    assert len(list(scene.edges_iter())) == 4
    assert len(list(scene.faces_iter())) == 1


def test_rectangle_tool_zero_area_drops_gesture():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _snap_at((1.0, 1.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _snap_at((1.0, 1.0, 0.0)))  # type: ignore[arg-type]

    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.faces_iter())) == 0


def test_rectangle_tool_has_active_gesture_reflects_state():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    assert tool.has_active_gesture is False

    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    assert tool.has_active_gesture is True

    tool.on_mouse_press(None, _snap_at((1.0, 1.0, 0.0)))  # type: ignore[arg-type]  # commit
    assert tool.has_active_gesture is False


def test_rectangle_tool_esc_cancels_mid_drag():
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]

    ev = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    tool.on_key_press(ev)

    overlay = tool.overlay()
    assert overlay.rubber_band_segments.shape == (0, 3)
    assert len(list(scene.vertices_iter())) == 0


def test_rectangle_tool_pushes_composite_to_command_stack():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    stack = CommandStack()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack))
    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _snap_at((3.0, 2.0, 0.0)))  # type: ignore[arg-type]

    assert stack.can_undo
    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.faces_iter())) == 0

    stack.redo()
    assert len(list(scene.vertices_iter())) == 4
    assert len(list(scene.faces_iter())) == 1


@pytest.mark.parametrize(
    "second_corner",
    [
        (3.0, 2.0, 0.0),    # up-right
        (3.0, -2.0, 0.0),   # down-right
        (-3.0, 2.0, 0.0),   # up-left
        (-3.0, -2.0, 0.0),  # down-left
    ],
)
def test_rectangle_face_normal_always_points_up(second_corner):
    """A ground-plane rectangle must always have a +Z (upward) face normal,
    regardless of which diagonal the second corner is dragged toward — so
    push/pull extrudes upward consistently. Regression for the un-normalized
    winding bug where down-right / up-left drags produced a -Z (downward)
    normal and push/pull went the wrong way."""
    from bermake.scene import Scene
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    scene = Scene()
    tool = RectangleTool()
    tool.activate(ToolContext(scene=scene))
    tool.on_mouse_press(None, _snap_at((0.0, 0.0, 0.0)))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _snap_at(second_corner))  # type: ignore[arg-type]

    face = next(iter(scene.faces_iter()))
    normal = scene.face_normal(face.id)
    assert normal[2] > 0.99, (
        f"Rectangle dragged to {second_corner} has normal {normal}; "
        f"expected +Z (up) so push/pull extrudes upward."
    )


def test_rectangle_commits_local_z_zero_under_rotated_translated_context(group_factory):
    """#108: RectangleTool must resolve both corners in the ACTIVE CONTEXT's
    local frame before building the loop, exactly the fix PrimitiveTool got
    in M7.5a (see test_primitive_footprint_plane.py). A translation-only
    fixture cannot prove this: a wrong world z=0, converted to local, stays
    a wrong-but-consistent z under pure translation. Under a ROTATION, that
    same wrong z feeds through the inverse transform and corrupts local x
    and y too, so this fixture rotates the active context (90 degrees about
    world X) as well as translating it.
    """
    import math

    from bermake.geometry.transforms import apply_mat, mat_compose, mat_rotate, mat_translate
    from bermake.model.model import Model
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool

    model = Model()
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    scene.add_face_from_loop(v)
    inst = group_factory(model)
    # Rotate 90 degrees about world X (local +Z now points along world +Y),
    # then translate: the group's own local floor no longer coincides with
    # world Z=0 on any axis.
    rot = mat_rotate([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], math.radians(90.0))
    trans = mat_translate([5.0, 0.0, 7.0])
    inst.transform = mat_compose(rot, trans)
    model.enter(inst)

    inner_scene = model.active_context.mesh
    before_ids = {vv.id for vv in inner_scene.vertices_iter()}

    tool = RectangleTool()
    tool.activate(ToolContext(scene=inner_scene, model=model))

    wt = model.active_world_transform
    # Well clear of the pre-existing square (local (0,0,0)-(1,1,0)) so no
    # corner collides with an existing vertex (add_vertex is idempotent on
    # an exact position match).
    local_first = np.array([5.0, 5.0, 0.0], dtype=np.float64)
    local_second = np.array([8.0, 7.0, 0.0], dtype=np.float64)
    # World-space corners a real vertex/edge snap onto the group's own
    # (rotated + translated) floor would report.
    world_first = apply_mat(local_first, wt)[0]
    world_second = apply_mat(local_second, wt)[0]

    tool.on_mouse_press(None, _snap_at(world_first))  # type: ignore[arg-type]
    tool.on_mouse_press(None, _snap_at(world_second))  # type: ignore[arg-type]

    new_ids = {vv.id for vv in inner_scene.vertices_iter()} - before_ids
    assert len(new_ids) == 4, "expected exactly 4 new (committed) vertices"

    committed_local_xy = set()
    for vid in new_ids:
        pos = inner_scene.vertex(vid).position
        assert round(float(pos[2]), 4) == 0.0, (
            f"vertex {vid} local z={pos[2]!r}; expected local z=0 "
            f"(a wrong world z=0 would have corrupted this under the fix's "
            f"absence)"
        )
        committed_local_xy.add((round(float(pos[0]), 4), round(float(pos[1]), 4)))

    assert committed_local_xy == {(5.0, 5.0), (8.0, 5.0), (8.0, 7.0), (5.0, 7.0)}, (
        f"committed local x/y {committed_local_xy} do not match the drawn "
        f"footprint; a wrong z fed through the rotated inverse transform "
        f"would corrupt local x/y as well as z"
    )


def test_measurement_text_reports_local_width_and_height_under_a_rotated_context(group_factory):
    """The readout is the number the VCB invites the user to type back.

    `_current_size` alone read raw world x and y while `overlay`,
    `apply_typed_value` and `_commit_rect` all resolve through
    `world_to_local_point`. In a context rotated 90 degrees about Z, local
    +X is world +Y, so a 2-by-4 local drag was announced as "4 m x 2 m" and
    typing that back would have built a 4-by-2 rectangle instead.
    """
    import math

    from bermake.geometry.transforms import apply_mat, mat_compose, mat_rotate, mat_translate
    from bermake.model.model import Model
    from bermake.tools import ToolContext
    from bermake.tools.rectangle_tool import RectangleTool
    from bermake.units import Units

    model = Model()
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    scene.add_face_from_loop(v)
    inst = group_factory(model)
    rot = mat_rotate([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], math.radians(90.0))
    trans = mat_translate([5.0, 0.0, 0.0])
    inst.transform = mat_compose(rot, trans)
    model.enter(inst)

    inner_scene = model.active_context.mesh
    tool = RectangleTool()
    tool.activate(ToolContext(scene=inner_scene, model=model, units_provider=Units))

    wt = model.active_world_transform
    world_first = apply_mat(np.array([0.0, 0.0, 0.0], dtype=np.float64), wt)[0]
    world_second = apply_mat(np.array([2.0, 4.0, 0.0], dtype=np.float64), wt)[0]

    tool.on_mouse_press(None, _snap_at(world_first))  # type: ignore[arg-type]
    tool.on_mouse_move(None, _snap_at(world_second))  # type: ignore[arg-type]

    width, height = tool._current_size()
    assert round(width, 6) == 2.0
    assert round(height, 6) == 4.0
    assert tool.measurement_text == "2 m x 4 m"
