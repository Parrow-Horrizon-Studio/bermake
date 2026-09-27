from __future__ import annotations

import numpy as np
import pytest
from bermake.model.material import Material
from bermake.scene.scene import Side, TexturePlacement
from bermake.tools import paint_tool as paint_tool_mod
from bermake.tools.paint_tool import PaintTool
from bermake.tools.tool import ToolContext
from PySide6.QtCore import QPointF, Qt


class _FakeScene:
    def __init__(self):
        self._mats: dict[tuple[int, Side], int] = {}
        self._placements: dict[tuple[int, Side], TexturePlacement] = {}

    def face_material(self, fid, side=Side.FRONT):
        return self._mats.get((fid, side), 0)

    def set_face_material(self, fid, mid, side=Side.FRONT):
        if mid == 0:
            self._mats.pop((fid, side), None)
        else:
            self._mats[(fid, side)] = mid

    def clear_face_material(self, fid, side=Side.FRONT):
        self._mats.pop((fid, side), None)

    def face_normal(self, fid):
        # Fixed +Z normal; paired with _FakeCamera's ray this resolves FRONT
        # (Task 10's side_for_ray is exercised directly and via
        # test_paint_stroke.py -- this double only needs a stable side so
        # the click/no-op/alt-click behaviour above it stays testable).
        return np.array([0.0, 0.0, 1.0])

    def face_center(self, fid):
        # M7.5b Task 12 fix round 1: placement-drag plane test only needs a
        # fixed origin -- the plane through world (0, 0, 0) with the +Z
        # normal above.
        return np.array([0.0, 0.0, 0.0])

    def face_placement(self, fid, side=Side.FRONT):
        return self._placements.get((fid, side), TexturePlacement())

    def set_face_placement(self, fid, placement, side=Side.FRONT):
        if placement == TexturePlacement():
            self._placements.pop((fid, side), None)
        else:
            self._placements[(fid, side)] = placement


class _FakeCamera:
    def ray_from_screen(self, x, y, w, h):
        # Looking straight down at the +Z-normal face from above: FRONT.
        # The origin now tracks the cursor pixel (not just a fixed point) so
        # a placement drag's ray-plane hit moves with the cursor (M7.5b Task
        # 12 fix round 1) -- existing side-resolution tests only ever
        # consult `direction` (via side_for_ray), so this changes nothing
        # for them.
        return np.array([float(x), float(y), 5.0]), np.array([0.0, 0.0, -1.0])


class _FakeStack:
    def __init__(self):
        self.pushed: list = []

    def push_executed(self, cmd, target):
        self.pushed.append((cmd, target))


class _Event:
    def __init__(self, alt=False, shift=False, pos=(10.0, 10.0), left_down=True):
        self._alt = alt
        self._shift = shift
        self._pos = pos
        self._left_down = left_down

    def position(self):
        return QPointF(*self._pos)

    def modifiers(self):
        mods = Qt.KeyboardModifier.NoModifier
        if self._alt:
            mods |= Qt.KeyboardModifier.AltModifier
        if self._shift:
            mods |= Qt.KeyboardModifier.ShiftModifier
        return mods

    def buttons(self):
        return Qt.MouseButton.LeftButton if self._left_down else Qt.MouseButton.NoButton


def _tool(monkeypatch, scene, stack, active_mat, pick=7):
    monkeypatch.setattr(
        paint_tool_mod, "pick_selectable",
        lambda *a, **k: ("face", pick) if pick is not None else None,
    )
    captured: dict = {}
    ctx = ToolContext(
        scene=scene,
        command_stack=stack,
        camera=_FakeCamera(),
        widget_size_provider=lambda: (100, 100),
        model=None,
        active_material_provider=lambda: active_mat,
        set_active_material=lambda mid: captured.__setitem__("sampled", mid),
    )
    t = PaintTool()
    t.activate(ctx)
    return t, captured


RED = Material(3, "Brick Red", (0.70, 0.27, 0.22))


def test_paint_pushes_command_and_applies(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    t, _ = _tool(monkeypatch, scene, stack, RED, pick=7)
    # A plain click is a one-face stroke: press paints it live, release is
    # what pushes the (single-child) CompositeCommand for undo.
    t.on_mouse_press(_Event(), snap=None)
    assert scene.face_material(7) == 3  # visible immediately, before release
    t.on_mouse_release(_Event(), snap=None)
    assert len(stack.pushed) == 1


def test_alt_click_samples_without_command(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    scene.set_face_material(7, 5)
    t, captured = _tool(monkeypatch, scene, stack, RED, pick=7)
    t.on_mouse_press(_Event(alt=True), snap=None)
    t.on_mouse_release(_Event(alt=True), snap=None)
    assert captured["sampled"] == 5
    assert stack.pushed == []
    assert scene.face_material(7) == 5


def test_no_op_when_material_unchanged(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    scene.set_face_material(7, 3)
    t, _ = _tool(monkeypatch, scene, stack, RED, pick=7)
    t.on_mouse_press(_Event(), snap=None)
    t.on_mouse_release(_Event(), snap=None)
    # Repainting a face with its current material is a no-op guard inside the
    # stroke: press paints nothing, so release finds an empty stroke and
    # pushes nothing (not just "press alone touches nothing").
    assert stack.pushed == []


def test_miss_does_nothing(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    t, _ = _tool(monkeypatch, scene, stack, RED, pick=None)
    t.on_mouse_press(_Event(), snap=None)
    t.on_mouse_release(_Event(), snap=None)
    assert stack.pushed == []


def test_shortcut_is_b():
    assert PaintTool().shortcut == "B"


# --- Shift-drag placement, driven end to end via mouse events -------------
#
# M7.5b Task 12 fix round 1: begin/update/end_placement_drag are exercised
# directly by tests/test_placement_drag.py, but the pixel -> UV conversion
# wired up in on_mouse_press/_move/_release (_begin_placement_drag_projection,
# _placement_plane_hit, _pixel_delta_to_uv) had no coverage at all and
# shipped with an inverted sign, caught by hand in self-review rather than by
# a failing test. `model=None` here means _begin_placement_drag_projection's
# `if self._model is not None` guard skips the materials-library lookup
# entirely (falling back to texture_size (1, 1)) -- no materials-library
# double or GL context needed, contrary to the earlier assumption that this
# path was too heavy to unit test.


def test_shift_drag_shifts_placement_via_mouse_events(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    t, _ = _tool(monkeypatch, scene, stack, RED, pick=7)

    t.on_mouse_press(_Event(shift=True, pos=(0.0, 0.0)), snap=None)
    assert t.has_active_gesture is True

    # plane_basis for the +Z face normal above gives u_axis=(0,-1,0) and
    # v_axis=(1,0,0), and _pixel_delta_to_uv negates the projected delta (a
    # grabbed point must slide toward +u/+v on screen, not away from it) --
    # so +Y cursor motion increases offset_u, and +X cursor motion DECREASES
    # offset_v. Three moves, in different directions, to cover both axes'
    # sign and accumulation across several calls.
    t.on_mouse_move(_Event(shift=True, pos=(0.0, 5.0)), snap=None)  # dy=+5 -> du=+5
    assert scene.face_placement(7).offset_u == pytest.approx(5.0)  # live, before release
    assert scene.face_placement(7).offset_v == pytest.approx(0.0)

    t.on_mouse_move(_Event(shift=True, pos=(5.0, 5.0)), snap=None)  # dx=+5 -> dv=-5
    assert scene.face_placement(7).offset_u == pytest.approx(5.0)
    assert scene.face_placement(7).offset_v == pytest.approx(-5.0)

    t.on_mouse_move(_Event(shift=True, pos=(5.0, 10.0)), snap=None)  # dy=+5 -> du+=5
    assert scene.face_placement(7).offset_u == pytest.approx(10.0)
    assert scene.face_placement(7).offset_v == pytest.approx(-5.0)

    t.on_mouse_release(_Event(shift=True, pos=(5.0, 10.0)), snap=None)
    assert t.has_active_gesture is False
    assert len(stack.pushed) == 1  # one drag, one undo step -- even mid-drag
    assert scene.face_placement(7).offset_u == pytest.approx(10.0)
    assert scene.face_placement(7).offset_v == pytest.approx(-5.0)


def test_shift_drag_with_no_motion_pushes_nothing(monkeypatch):
    scene, stack = _FakeScene(), _FakeStack()
    t, _ = _tool(monkeypatch, scene, stack, RED, pick=7)

    t.on_mouse_press(_Event(shift=True, pos=(0.0, 0.0)), snap=None)
    t.on_mouse_release(_Event(shift=True, pos=(0.0, 0.0)), snap=None)

    assert stack.pushed == []
    assert scene.face_placement(7) == TexturePlacement()
