"""Tests for the M7.6b Task 9 on-canvas measurement readout.

Covers Tool.measurement_text on the tools that already track a live numeric
value (Line, Rectangle, Rotate), the "no gesture -> None" default, and
draw_plan.plan_cursor_readout's viewport clamping. Length assertions go
through format_length(expected_m, units) rather than a hardcoded string, so
these pin behaviour rather than the formatter's current choices.
"""

from __future__ import annotations

import types

import numpy as np
from bermake.annotations.draw_plan import CHAR_W_PX, FONT_PX, plan_cursor_readout
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.line_tool import LineTool
from bermake.tools.rectangle_tool import RectangleTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.tool import ToolContext
from bermake.units import Units, UnitSystem, format_length
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

_METRIC = Units()
_IMPERIAL = Units(system=UnitSystem.IMPERIAL)


def _press():
    return QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(0, 0),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind,
        world_position=np.asarray(pos, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
    )


def _ctx(s, stack, units, sel=None):
    return ToolContext(
        scene=s,
        command_stack=stack,
        camera=None,
        widget_size_provider=lambda: (800, 600),
        units_provider=lambda: units,
        selection=sel,
    )


def _line_mid_gesture(units):
    s = Scene()
    stack = CommandStack()
    t = LineTool()
    t.activate(_ctx(s, stack, units))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))
    t.on_mouse_move(_press(), _snap([3, 0, 0]))
    return t


def test_line_tool_measurement_metric_document(qtbot):
    t = _line_mid_gesture(_METRIC)
    assert t.measurement_text == format_length(3.0, _METRIC)


def test_line_tool_measurement_imperial_document(qtbot):
    t = _line_mid_gesture(_IMPERIAL)
    assert t.measurement_text == format_length(3.0, _IMPERIAL)


def test_rectangle_tool_measurement_reports_w_x_h(qtbot):
    s = Scene()
    stack = CommandStack()
    t = RectangleTool()
    t.activate(_ctx(s, stack, _METRIC))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))
    t.on_mouse_move(_press(), _snap([4, 2, 0]))
    expected = f"{format_length(4.0, _METRIC)} x {format_length(2.0, _METRIC)}"
    assert t.measurement_text == expected


def test_rotate_tool_measurement_reports_degrees(qtbot, monkeypatch):
    s = Scene()
    a = s.add_vertex(np.array([1, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    e = s.add_edge(a, b)
    sel = Selection()
    sel.replace(edges=[e])
    stack = CommandStack()
    t = RotateTool()
    t.activate(_ctx(s, stack, _METRIC, sel))
    monkeypatch.setattr(t, "_pick_plane_normal", lambda ev: np.array([0, 0, 1], np.float32))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))  # center
    t.on_mouse_press(_press(), _snap([1, 0, 0]))  # start direction
    t.on_mouse_move(_press(), _snap([0, 1, 0]))  # sweep to +Y, 90 deg
    assert t.measurement_text == "90 deg"


def test_no_active_gesture_reports_none(qtbot):
    s = Scene()
    stack = CommandStack()
    t = LineTool()
    t.activate(_ctx(s, stack, _METRIC))
    assert t.measurement_text is None


def test_cursor_readout_stays_inside_viewport_near_right_edge():
    width, height = 800.0, 600.0
    text = "3.500 m"
    cursor_px = (width - 5.0, 300.0)
    plan = plan_cursor_readout(text, cursor_px, width, height)

    assert plan.annotation_id == -1  # sentinel: never a real, pickable annotation
    text_w = max(len(text), 1) * CHAR_W_PX
    assert plan.texts, "expected a text draw for the readout"
    for draw in plan.texts:
        assert draw.align == "left"
        assert draw.x >= 0.0
        assert draw.x + text_w <= width


def test_cursor_readout_stays_inside_viewport_near_top_edge():
    width, height = 800.0, 600.0
    text = "3.500 m"
    cursor_px = (400.0, 5.0)
    plan = plan_cursor_readout(text, cursor_px, width, height)

    assert plan.annotation_id == -1  # sentinel: never a real, pickable annotation
    text_h = FONT_PX
    assert plan.texts, "expected a text draw for the readout"
    for draw in plan.texts:
        assert draw.align == "left"
        assert draw.y - text_h >= 0.0
        assert draw.y <= height
