from __future__ import annotations

import types

import numpy as np
from bermake.scene.scene import Scene
from bermake.tools.tape_measure_tool import TapeMeasureTool
from bermake.tools.tool import ToolContext
from bermake.units import Units
from bermake.viewport.snap_engine import SnapKind
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

U = Units()


def _press():
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(0, 0),
                       Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier)


def _snap(p):
    return types.SimpleNamespace(kind=SnapKind.ENDPOINT, world_position=np.asarray(p, np.float32),
                                 axis=None, vertex_id=None, edge_id=None, edge_t=None)


def _ctx(s):
    return ToolContext(scene=s, command_stack=None, camera=None,
                       widget_size_provider=lambda: (800, 600), units_provider=lambda: U)


def test_distance_readout(qtbot):
    s = Scene()
    t = TapeMeasureTool()
    t.activate(_ctx(s))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))
    t.on_mouse_press(_press(), _snap([3, 4, 0]))   # 3-4-5 → distance 5
    assert "5" in (t.status_text or "")
    assert t.shortcut == "T" and t.name == "Tape Measure"


def test_measure_only_no_mutation(qtbot):
    s = Scene()
    before = sum(1 for _ in s.vertices_iter())
    t = TapeMeasureTool()
    t.activate(_ctx(s))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))
    t.on_mouse_press(_press(), _snap([1, 0, 0]))
    assert sum(1 for _ in s.vertices_iter()) == before


def test_esc_resets(qtbot):
    from PySide6.QtGui import QKeyEvent
    s = Scene()
    t = TapeMeasureTool()
    t.activate(_ctx(s))
    t.on_mouse_press(_press(), _snap([0, 0, 0]))
    t.on_key_press(QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
    ))
    assert not t.has_active_gesture


def _snap_none():
    return types.SimpleNamespace(
        kind=SnapKind.NONE,
        world_position=np.zeros(3, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
    )


def test_snap_marker_surfaced(qtbot):
    """on_mouse_move with a real snap kind populates overlay snap_marker_*."""
    s = Scene()
    t = TapeMeasureTool()
    t.activate(_ctx(s))

    pos = [1.0, 2.0, 3.0]
    t.on_mouse_move(None, _snap(pos))
    ov = t.overlay()
    assert ov.snap_marker_position is not None
    np.testing.assert_allclose(ov.snap_marker_position, np.asarray(pos, np.float32))
    assert ov.snap_marker_kind == int(SnapKind.ENDPOINT)

    # After a NONE snap the marker clears.
    t.on_mouse_move(None, _snap_none())
    ov2 = t.overlay()
    assert ov2.snap_marker_position is None
