"""VCB completions (#55): Circle segment count, signed Rotate angle, per-axis
Scale factors. Each test establishes the tool's real `apply_typed_value`
precondition (the guard state each tool checks before it will parse typed
text) rather than weakening the assertion.
"""

from __future__ import annotations

import types

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.circle_tool import CircleTool
from bermake.tools.rotate_tool import RotateTool, _Stage
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.tool import ToolContext
from bermake.tools.transform_support import GripSpec
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()


def _press():
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(p):
    return types.SimpleNamespace(
        kind=SnapKind.ON_FACE,
        world_position=np.asarray(p, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
        face_id=None,
    )


def _ctx(s, stack):
    return ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        units_provider=lambda: U,
    )


# ---------------------------------------------------------------------------
# #55a — Circle segment count ("12s")
# ---------------------------------------------------------------------------


def test_circle_accepts_a_segment_count(qtbot):
    """CircleTool.apply_typed_value guards on `_state == DRAWING and _plane
    is not None` (circle_tool.py) -- established here the same way the
    existing test_circle_typed_radius does: activate + press to place the
    center, which resolves the drawing plane and enters DRAWING."""
    s = Scene()
    stack = CommandStack()
    tool = CircleTool()
    tool.activate(_ctx(s, stack))
    tool.on_mouse_press(_press(), _snap([0, 0, 0]))  # center placed -> DRAWING

    before = tool.segments
    assert tool.apply_typed_value("12s", U) is True
    assert tool.segments == 12 and tool.segments != before
    # Mirrors polygon's "Ns" completion: setting the count keeps the gesture
    # open (it does not commit geometry), same as PolygonTool.
    assert tool.has_active_gesture


# ---------------------------------------------------------------------------
# #55b — Signed Rotate angle ("-30")
# ---------------------------------------------------------------------------


def _rotate_typed_under_cw_sweep(typed: str) -> float:
    """Resolve a typed Rotate angle under a fixed CLOCKWISE (-90 deg) sweep and
    return the signed result in degrees. RotateTool.apply_typed_value guards on
    `_stage == _Stage.HAVE_START` (rotate_tool.py), so establish that plus a
    start/cur direction pair that sweeps -90 deg (start +X, swept to -Y)."""
    tool = RotateTool()
    tool._stage = _Stage.HAVE_START
    tool._normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    tool._start_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tool._cur_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)  # swept -90 deg
    assert tool.apply_typed_value(typed, U) is True
    return tool.angle_degrees


def test_rotate_negative_angle_is_the_opposite_of_positive():
    """#55: a negative typed angle rotates the OTHER way, i.e. -deg is the exact
    opposite of +deg under the SAME sweep -- for either sweep sign. This is
    checked under a clockwise (-90 deg) sweep, the case where a literal
    `radians(deg)` for negatives (ignoring the sweep) collapses "-30" and "30"
    to the same rotation. A relative-opposite assertion catches that collision,
    where a bare `angle_degrees < 0` would not (both are negative under a CW
    sweep)."""
    pos = _rotate_typed_under_cw_sweep("30")
    neg = _rotate_typed_under_cw_sweep("-30")
    assert pos != 0.0
    assert neg == -pos, f"-30 must rotate opposite to 30 under the same sweep (got {neg} vs {pos})"


# ---------------------------------------------------------------------------
# #55c — Per-axis Scale factors ("2,1,1")
# ---------------------------------------------------------------------------


def test_scale_accepts_per_axis_factors():
    """ScaleTool.apply_typed_value guards on `_active is None` (scale_tool.py).
    Following test_scale_tool_cursor_plane.py's fixture convention: build a
    GripSpec and set tool._active directly on a fresh (never-activated) tool
    rather than going through activate()."""
    tool = ScaleTool()
    grip = GripSpec(
        position=np.array([1.0, 0.0, 0.0], np.float32),
        opposite=np.array([0.0, 0.0, 0.0], np.float32),
        axes=(0,),
    )
    tool._active = grip

    assert tool.apply_typed_value("2,1,1", U) is True
    # Real attribute name is `_factor_vec` (private; ScaleTool has no public
    # `factor_vec` property -- unlike CircleTool.segments/RotateTool.angle_degrees
    # which this task adds, this one is intentionally the internal name).
    assert tuple(round(float(f), 3) for f in tool._factor_vec) == (2.0, 1.0, 1.0)
