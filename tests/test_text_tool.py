"""Unit tests for the Text tool (M7d) — two-click gesture, overridable text
prompt, world-space overlay preview, and local-frame storage."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.command_stack import CommandStack
from bermake.model.model import Model
from bermake.tools.text_tool import TextTool
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


def _tool(model, answer="Load-bearing"):
    tool = TextTool()
    tool.prompt_text = lambda default="": answer
    tool.activate(_ctx(model, CommandStack()))
    return tool


def test_two_clicks_plus_prompt_create_a_label():
    model = Model()
    tool = _tool(model)
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_mouse_press(None, _Snap(2.0, 2.0))
    anns = model.active_context.annotations
    assert len(anns) == 1
    assert anns[0].kind == "label"
    assert anns[0].anchor == (0.0, 0.0, 0.0)
    assert anns[0].text_pos == (2.0, 2.0, 0.0)
    assert anns[0].text == "Load-bearing"


def test_cancelled_prompt_creates_nothing():
    model = Model()
    tool = _tool(model, answer=None)
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_mouse_press(None, _Snap(2.0, 2.0))
    assert model.active_context.annotations == []


def test_blank_text_creates_nothing():
    model = Model()
    tool = _tool(model, answer="   ")
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_mouse_press(None, _Snap(2.0, 2.0))
    assert model.active_context.annotations == []


def test_escape_cancels_the_gesture():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    model = Model()
    tool = _tool(model)
    tool.on_mouse_press(None, _Snap(0.0, 0.0))
    tool.on_key_press(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                Qt.KeyboardModifier.NoModifier))
    assert tool.has_active_gesture is False
    assert model.active_context.annotations == []
    segments = tool.overlay().rubber_band_segments
    assert segments.shape[0] == 0

    # tool must still work for a fresh label right afterward
    tool.on_mouse_press(None, _Snap(5.0, 5.0))
    tool.on_mouse_press(None, _Snap(6.0, 6.0))
    anns = model.active_context.annotations
    assert len(anns) == 1
    assert anns[0].anchor == pytest.approx((5.0, 5.0, 0.0))
    assert anns[0].text_pos == pytest.approx((6.0, 6.0, 0.0))


def _entered_rotated_scaled_group(model):
    """Enter a group rotated 90deg CCW about Z, non-uniformly scaled
    (sx=2, sy=0.5, sz=1), and translated (10,20,0): world = R @ (S @ local)
    + t. Chosen so local (0,0,0) -> world (10,20,0) and local (4,0,0) ->
    world (10,28,0) -- easy to hand-verify, and (unlike DimensionTool's
    rotation-only regression test) exercises non-uniform scale too."""
    grp = model.new_definition("G", is_group=True)
    ang = np.pi / 2.0
    c, s = np.cos(ang), np.sin(ang)
    rot = np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = np.diag([2.0, 0.5, 1.0])
    tf = np.eye(4)
    tf[:3, :3] = rot @ scale
    tf[:3, 3] = [10.0, 20.0, 0.0]
    inst = model.new_instance(grp, tf)
    model.root.children.append(inst)
    model.enter(inst)
    return inst


def test_label_created_inside_entered_rotated_scaled_group_world_and_local_agree():
    """Regression guard for the M7-class overlay/storage frame bug, extended
    beyond DimensionTool's rotation-only case to a NON-UNIFORMLY SCALED
    group too: (a) the live rubber-band preview must be built in WORLD space
    even while entered into a rotated+scaled group -- the renderer draws
    ToolOverlay.rubber_band_segments in world space with no model matrix
    applied; (b) the committed Label's LOCAL anchor/text_pos must map back
    through the active world transform to the exact WORLD points clicked."""
    model = Model()
    _entered_rotated_scaled_group(model)
    assert not np.allclose(model.active_world_transform, np.eye(4))

    tool = TextTool()
    tool.prompt_text = lambda default="": "Beam"
    tool.activate(_ctx(model, CommandStack()))

    anchor_world = (10.0, 20.0, 0.0)   # the group's local origin
    text_world = (10.0, 28.0, 0.0)     # local (4,0,0)

    tool.on_mouse_press(None, _Snap(*anchor_world))
    tool.on_mouse_move(None, _Snap(*text_world))

    # (a) preview overlay: WORLD-space leader line from anchor to cursor.
    segments = tool.overlay().rubber_band_segments
    assert segments.shape == (2, 3)
    assert segments[0] == pytest.approx(np.array(anchor_world), abs=1e-5)
    assert segments[1] == pytest.approx(np.array(text_world), abs=1e-5)

    tool.on_mouse_press(None, _Snap(*text_world))

    anns = model.active_context.annotations
    assert len(anns) == 1
    label = anns[0]
    assert label.anchor == pytest.approx((0.0, 0.0, 0.0), abs=1e-5)
    assert label.text_pos == pytest.approx((4.0, 0.0, 0.0), abs=1e-5)

    # (b) mapping the stored LOCAL coordinates back through the active world
    # transform must reproduce the exact WORLD points that were clicked.
    wt = model.active_world_transform
    anchor_back = (wt @ np.append(np.array(label.anchor), 1.0))[:3]
    text_pos_back = (wt @ np.append(np.array(label.text_pos), 1.0))[:3]
    assert anchor_back == pytest.approx(np.array(anchor_world), abs=1e-5)
    assert text_pos_back == pytest.approx(np.array(text_world), abs=1e-5)
