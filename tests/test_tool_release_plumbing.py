"""Plumbing: on_mouse_release default, ToolOverlay.box_rect, ToolContext.selection,
and viewport LMB-release forwarding."""

from __future__ import annotations

import numpy as np


def test_tool_default_on_mouse_release_is_noop():
    from bermake.tools.tool import Tool, ToolOverlay

    class _Min(Tool):
        @property
        def name(self): return "Min"
        @property
        def shortcut(self): return "Z"
        @property
        def id(self): return "min"
        @property
        def has_active_gesture(self): return False
        def activate(self, ctx): pass
        def deactivate(self): pass
        def overlay(self): return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
            rubber_band_color=(1, 1, 1), snap_marker_position=None,
            snap_marker_color=(1, 1, 1),
        )
        @property
        def anchor_or_none(self): return None

    _Min().on_mouse_release(None, None)  # must not raise


def test_tool_overlay_box_rect_defaults_none():
    from bermake.tools.tool import ToolOverlay

    o = ToolOverlay(
        rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
        rubber_band_color=(1, 1, 1), snap_marker_position=None,
        snap_marker_color=(1, 1, 1),
    )
    assert o.box_rect is None
    assert isinstance(o.box_rect_color, tuple)


def test_tool_context_has_selection_field():
    from bermake.tools.tool import ToolContext

    ctx = ToolContext(scene=object())
    assert ctx.selection is None
    ctx2 = ToolContext(scene=object(), selection="sel")
    assert ctx2.selection == "sel"


def test_viewport_forwards_lmb_release_to_active_tool(qtbot):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from bermake.model import Model
    from bermake.viewport.viewport_widget import ViewportWidget

    calls = []

    class _Recorder:
        @property
        def anchor_or_none(self): return None
        # Abstract on Tool, so every real tool has it; _snap_for_event reads
        # it to spot the end of a gesture (M7.6b whole-branch fix, I1).
        @property
        def has_active_gesture(self): return False
        def on_mouse_release(self, event, snap):
            calls.append(("release", event.position().x()))

    class _Mgr:
        def __init__(self): self.active = _Recorder()

    vw = ViewportWidget(model=Model(), tool_manager=_Mgr())
    qtbot.addWidget(vw)
    ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(120.0, 50.0),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    vw.mouseReleaseEvent(ev)
    assert calls and calls[0][0] == "release" and calls[0][1] == 120.0
