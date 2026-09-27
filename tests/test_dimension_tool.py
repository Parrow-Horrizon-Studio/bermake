"""Unit tests for the Dimension tool (M7d) — three-click gesture, world-space
overlay preview, local-frame storage, and the perpendicular offset contract."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.command_stack import CommandStack
from bermake.model.model import Model
from bermake.tools.dimension_tool import DimensionTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind


class _Snap:
    def __init__(self, x, y, z=0.0):
        self.kind = SnapKind.ON_FACE
        self.world_position = np.array([x, y, z], dtype=np.float64)


def _ctx(model, stack):
    return ToolContext(
        scene=model.active_scene, command_stack=stack, model=model, camera=None,
        widget_size_provider=lambda: (100, 100), units_provider=lambda: None,
    )


def _make(tool, ax, ay, bx, by, ox, oy):
    tool.on_mouse_press(None, _Snap(ax, ay))
    tool.on_mouse_press(None, _Snap(bx, by))
    tool.on_mouse_move(None, _Snap(ox, oy))
    tool.on_mouse_press(None, _Snap(ox, oy))


def test_three_clicks_create_one_dimension():
    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))
    _make(tool, 0.0, 0.0, 4.0, 0.0, 2.0, -1.0)
    anns = model.active_context.annotations
    assert len(anns) == 1
    assert anns[0].kind == "dimension"
    assert anns[0].p1 == (0.0, 0.0, 0.0)
    assert anns[0].p2 == (4.0, 0.0, 0.0)


def test_offset_is_perpendicular_to_the_measured_axis():
    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))
    # third click is off to the side AND along the axis; only the perpendicular
    # component may survive
    _make(tool, 0.0, 0.0, 4.0, 0.0, 3.5, -1.0)
    off = model.active_context.annotations[0].offset
    assert abs(off[0]) < 1e-9      # along-axis component removed
    assert abs(off[1] + 1.0) < 1e-9


def test_degenerate_second_click_creates_nothing():
    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))
    _make(tool, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0)
    assert model.active_context.annotations == []


def test_escape_cancels_the_gesture():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                Qt.KeyboardModifier.NoModifier))
    assert tool.has_active_gesture is False


# --- additional coverage required beyond the brief -------------------------


def test_escape_leaves_no_partial_state_and_tool_stays_usable():
    """Escape mid-gesture (after BOTH p1 and p2 are set, not just p1) must
    create nothing, clear all partial state, and leave the tool ready to
    complete a fresh dimension right afterward — not stuck or contaminated
    by the cancelled points."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_mouse_press(None, _Snap(4.0, 0.0))
    tool.on_mouse_move(None, _Snap(2.0, -1.0))

    tool.on_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                Qt.KeyboardModifier.NoModifier))

    assert tool.has_active_gesture is False
    assert model.active_context.annotations == []
    segments = tool.overlay().rubber_band_segments
    assert segments.shape[0] == 0   # no residual rubber band either

    # tool must still work for the next dimension: no leftover p1/p2/cursor
    _make(tool, 10.0, 0.0, 10.0, 5.0, 11.0, 2.0)
    anns = model.active_context.annotations
    assert len(anns) == 1
    assert anns[0].p1 == pytest.approx((10.0, 0.0, 0.0))
    assert anns[0].p2 == pytest.approx((10.0, 5.0, 0.0))


def test_offset_removes_along_axis_component_for_non_axis_aligned_dimension():
    """A dimension whose p1->p2 axis is not aligned to any single world axis:
    the stored offset must be exactly the perpendicular part of
    (third_click - midpoint), with the along-axis contamination genuinely
    zeroed out -- verified against an independently-computed projection."""
    model = Model()
    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))

    p1 = np.array([1.0, -2.0, 0.5])
    p2 = np.array([6.0, 1.0, 4.5])          # non-axis-aligned 3D segment
    axis = p2 - p1
    axis_unit = axis / np.linalg.norm(axis)
    mid = (p1 + p2) / 2.0

    raw = np.array([1.0, 2.0, -0.5])        # arbitrary perturbation
    contamination = axis_unit * 3.7          # deliberately added along-axis noise
    third = mid + raw + contamination
    expected_offset = raw - axis_unit * float(np.dot(raw, axis_unit))

    tool.on_mouse_press(None, _Snap(p1[0], p1[1], p1[2]))
    tool.on_mouse_press(None, _Snap(p2[0], p2[1], p2[2]))
    tool.on_mouse_move(None, _Snap(third[0], third[1], third[2]))
    tool.on_mouse_press(None, _Snap(third[0], third[1], third[2]))

    # NOTE: at the document root, world_to_local_point's identity-transform
    # branch round-trips through float32 (an existing, out-of-scope property
    # of bermake.viewport.picking shared by every drawing tool), so this
    # tolerance is float32-scale rather than float64 exact.
    offset = np.array(model.active_context.annotations[0].offset)
    assert float(np.dot(offset, axis_unit)) == pytest.approx(0.0, abs=1e-5)
    assert offset == pytest.approx(expected_offset, abs=1e-5)


def _entered_rotated_group(model):
    """Enter a group rotated 90deg CCW about Z and translated (10,20,0):
    world = R @ local + t, with R mapping local (lx,ly) -> world offset
    (-ly, lx). Chosen as an exact rotation (no float trig residue worth
    guarding against) so the arithmetic below is easy to hand-verify."""
    grp = model.new_definition("G", is_group=True)
    ang = np.pi / 2.0
    c, s = np.cos(ang), np.sin(ang)
    tf = np.array([
        [c, -s, 0.0, 10.0],
        [s, c, 0.0, 20.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    inst = model.new_instance(grp, tf)
    model.root.children.append(inst)
    model.enter(inst)
    return inst


def test_overlay_preview_is_in_world_space_inside_entered_transformed_group():
    """Regression guard for the M7-class overlay-frame bug: the renderer draws
    ToolOverlay.rubber_band_segments in WORLD space with no model matrix
    applied, so the live preview must be built in world space even while
    entered into a translated+rotated group -- not left in local space."""
    model = Model()
    _entered_rotated_group(model)
    assert not np.allclose(model.active_world_transform, np.eye(4))

    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))

    p1_world = (10.0, 20.0, 0.0)   # the group's local origin
    p2_world = (10.0, 24.0, 0.0)   # local (4,0,0) -> world offset (0,4,0)
    third_world = (7.0, 22.0, 0.0)  # local (2,3,0) -> world offset (-3,2,0)

    tool.on_mouse_press(None, _Snap(*p1_world))
    tool.on_mouse_press(None, _Snap(*p2_world))
    tool.on_mouse_move(None, _Snap(*third_world))

    segments = tool.overlay().rubber_band_segments
    assert segments.shape[0] == 8   # main line + 2 extension lines + dim line

    # main measured segment: exactly the clicked WORLD points (no transform
    # should ever touch these two anchors).
    assert segments[0] == pytest.approx(np.array(p1_world), abs=1e-5)
    assert segments[1] == pytest.approx(np.array(p2_world), abs=1e-5)

    # the offset dimension line (last pair) is the piece that must be
    # converted from the local perpendicular back into WORLD space.
    expected_d1 = np.array([7.0, 20.0, 0.0])
    expected_d2 = np.array([7.0, 24.0, 0.0])
    assert segments[6] == pytest.approx(expected_d1, abs=1e-5)
    assert segments[7] == pytest.approx(expected_d2, abs=1e-5)


def test_dimension_created_inside_entered_group_stores_local_coords():
    """The committed Dimension's p1/p2/offset must be CONTEXT-LOCAL, and
    mapping them back through the active world transform must reproduce the
    exact WORLD points/line that were drawn (same numbers as the overlay
    test above) -- overlay-in-world and storage-in-local must agree."""
    model = Model()
    _entered_rotated_group(model)

    tool = DimensionTool()
    tool.activate(_ctx(model, CommandStack()))

    p1_world = (10.0, 20.0, 0.0)
    p2_world = (10.0, 24.0, 0.0)
    third_world = (7.0, 22.0, 0.0)
    tool.on_mouse_press(None, _Snap(*p1_world))
    tool.on_mouse_press(None, _Snap(*p2_world))
    tool.on_mouse_move(None, _Snap(*third_world))
    tool.on_mouse_press(None, _Snap(*third_world))

    dim = model.active_context.annotations[0]
    assert dim.p1 == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert dim.p2 == pytest.approx((4.0, 0.0, 0.0), abs=1e-5)
    assert dim.offset == pytest.approx((0.0, 3.0, 0.0), abs=1e-5)

    wt = model.active_world_transform
    p1_back = (wt @ np.append(np.array(dim.p1), 1.0))[:3]
    p2_back = (wt @ np.append(np.array(dim.p2), 1.0))[:3]
    offset_back = wt[:3, :3] @ np.array(dim.offset)

    assert p1_back == pytest.approx(np.array(p1_world), abs=1e-5)
    assert p2_back == pytest.approx(np.array(p2_world), abs=1e-5)
    assert (np.array(p1_world) + offset_back) == pytest.approx(
        np.array([7.0, 20.0, 0.0]), abs=1e-5
    )
    assert (np.array(p2_world) + offset_back) == pytest.approx(
        np.array([7.0, 24.0, 0.0]), abs=1e-5
    )
