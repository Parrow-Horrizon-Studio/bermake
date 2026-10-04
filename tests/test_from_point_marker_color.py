"""#123: the From-Point snap marker takes the colour of the axis it radiates along.

Decision D1: the marker only, never the rubber band. Bermake draws no
inference line, so colouring the line being drawn would read as an axis lock.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
from bermake.tools.arc_tool import ArcTool
from bermake.tools.circle_tool import CircleTool
from bermake.tools.dimension_tool import DimensionTool
from bermake.tools.line_tool import LineTool
from bermake.tools.line_tool import _State as _LineState
from bermake.tools.polygon_tool import PolygonTool
from bermake.tools.primitive_tool import BoxTool, ConeTool, CylinderTool, SphereTool
from bermake.tools.rectangle_tool import RectangleTool
from bermake.tools.roof_tool import RoofTool
from bermake.tools.tape_measure_tool import TapeMeasureTool
from bermake.tools.text_tool import TextTool
from bermake.tools.wall_tool import WallTool
from bermake.viewport.snap_engine import (
    AXIS_COLORS,
    MARKER_COLOR_BY_KIND,
    SnapKind,
    SnapResult,
    marker_color,
)

_FALLBACK = (0.5, 0.5, 0.5)

# The eleven marker consumers named in the task (the four primitives share one
# base class and one call site). The source scan below catches a twelfth.
_TOOL_CLASSES = [
    ArcTool,
    CircleTool,
    DimensionTool,
    LineTool,
    PolygonTool,
    BoxTool,
    CylinderTool,
    ConeTool,
    SphereTool,
    RectangleTool,
    RoofTool,
    TapeMeasureTool,
    TextTool,
    WallTool,
]


def _snap(kind: SnapKind, axis: int | None) -> SnapResult:
    return SnapResult(
        kind=kind,
        world_position=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        axis=axis,
        vertex_id=None,
        label="",
    )


@pytest.fixture
def make_snap():
    return _snap


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_from_point_marker_takes_the_axis_colour(axis, make_snap):
    snap = make_snap(SnapKind.FROM_POINT, axis=axis)
    assert marker_color(snap, (0.85, 0.85, 0.85)) == AXIS_COLORS[axis]


def test_from_point_without_an_axis_falls_back(make_snap):
    snap = make_snap(SnapKind.FROM_POINT, axis=None)
    assert marker_color(snap, _FALLBACK) == _FALLBACK


def test_the_three_axis_colours_are_distinct_and_x_red_y_green_z_blue():
    assert len(set(AXIS_COLORS)) == 3
    r, g, b = AXIS_COLORS
    assert r[0] > r[1] and r[0] > r[2]
    assert g[1] > g[0] and g[1] > g[2]
    assert b[2] > b[0] and b[2] > b[1]


@pytest.mark.parametrize("kind", [k for k in SnapKind if k is not SnapKind.FROM_POINT])
def test_other_kinds_keep_their_per_kind_colour_even_with_an_axis(kind, make_snap):
    snap = make_snap(kind, axis=1)
    assert marker_color(snap, _FALLBACK) == MARKER_COLOR_BY_KIND.get(kind, _FALLBACK)


def _marker_colour_of(tool_cls, snap) -> tuple[float, float, float]:
    tool = tool_cls()
    tool.on_mouse_move(None, snap)
    return tuple(tool.overlay().snap_marker_color)


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_every_tool_overlay_shows_the_axis_colour_for_from_point(tool_cls, axis, make_snap):
    colour = _marker_colour_of(tool_cls, make_snap(SnapKind.FROM_POINT, axis=axis))
    assert colour == AXIS_COLORS[axis]


@pytest.mark.parametrize("tool_cls", _TOOL_CLASSES, ids=lambda c: c.__name__)
def test_every_tool_overlay_keeps_other_kind_colours(tool_cls, make_snap):
    colour = _marker_colour_of(tool_cls, make_snap(SnapKind.ENDPOINT, axis=1))
    assert colour == MARKER_COLOR_BY_KIND[SnapKind.ENDPOINT]


def test_from_point_does_not_colour_the_line_rubber_band(make_snap):
    # D1: marker only. The Line tool already colours its rubber band for
    # AXIS_LOCK; a From-Point must leave it neutral.
    tool = LineTool()
    tool._state = _LineState.DRAWING
    tool._preview_tip = np.zeros(3)
    tool.on_mouse_move(None, make_snap(SnapKind.FROM_POINT, axis=1))
    assert tuple(tool.overlay().rubber_band_color) != AXIS_COLORS[1]


def test_no_tool_looks_the_marker_colour_up_by_kind_alone():
    """A new marker consumer must go through `marker_color`, not the raw dict.

    Static, so a twelfth tool is caught without being listed above.
    """
    tools_dir = pathlib.Path(__file__).resolve().parents[1] / "python" / "bermake" / "tools"
    offenders = [
        p.name
        for p in sorted(tools_dir.glob("*.py"))
        if re.search(r"MARKER_COLOR_BY_KIND\s*\.\s*get", p.read_text(encoding="utf-8"))
    ]
    assert offenders == []
