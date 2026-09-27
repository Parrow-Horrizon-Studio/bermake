"""The Door/Window placement tool (M7b).

Pick a wall face; a framed door/window Component is placed flush to it,
upright, floor-anchored (window at a sill height), horizontally following the
cursor. Identical openings share one Component. The wall is not cut.
"""

from __future__ import annotations

import itertools

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.opening_commands import PlaceOpeningCommand
from bermake.geometry.opening import opening_placement_transform
from bermake.tools.tool import Tool, ToolContext, ToolOverlay

_NEUTRAL = (0.85, 0.85, 0.85)


class DoorWindowTool(Tool):
    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._command_stack = None
        self._camera = None
        self._size_provider = None
        self._preview = None  # (transform 4x4) or None
        self.kind = "door"
        self.width = 0.9  # meters
        self.height = 2.1
        self.sill = 0.0
        self.depth = 0.1

    @property
    def name(self) -> str:
        return "Door/Window"

    @property
    def shortcut(self) -> str:
        return "D"

    @property
    def id(self) -> str:
        return "door_window"

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._camera = ctx.camera
        self._size_provider = ctx.widget_size_provider
        self._preview = None

    def deactivate(self) -> None:
        self._preview = None

    def _viewport_size(self) -> tuple[int, int]:
        if self._size_provider is None:
            return (1, 1)
        return self._size_provider()

    def _cursor(self, event: QMouseEvent) -> tuple[float, float]:
        pos = event.position()
        return float(pos.x()), float(pos.y())

    def _sill_for_kind(self) -> float:
        return 0.0 if self.kind == "door" else self.sill

    def _resolve(self, event: QMouseEvent):
        if self._model is None or self._camera is None:
            return None
        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
        hit = self._model.pick_face_local(origin, direction)
        if hit is None:
            return None
        point, normal = hit
        return opening_placement_transform(point, normal, self._sill_for_kind())

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        self._preview = self._resolve(event)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        transform = self._resolve(event)
        if transform is None:
            return
        cmd = PlaceOpeningCommand(
            self.kind,
            self.width,
            self.height,
            self.depth,
            transform,
            self._model.active_context,
        )
        self._command_stack.execute(cmd, self._model)

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        value = parse_length(text, units)
        if value is None or value <= 0:
            return False
        self.width = value
        return True

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._preview = None

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        if self._preview is not None:
            hx = self.width / 2.0
            corners = np.array(
                [
                    [-hx, 0.0, 0.0],
                    [hx, 0.0, 0.0],
                    [hx, 0.0, self.height],
                    [-hx, 0.0, self.height],
                ],
                dtype=np.float64,
            )
            w = self._model.active_world_transform
            world = [(w @ self._preview @ np.append(c, 1.0))[:3] for c in corners]
            loop = [*world, world[0]]
            segs = []
            for a, b in itertools.pairwise(loop):
                segs.append(a)
                segs.append(b)
            segments = np.array(segs, dtype=np.float32)
        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_NEUTRAL,
            snap_marker_position=None,
            snap_marker_color=_NEUTRAL,
            snap_marker_kind=0,
        )

    @property
    def has_active_gesture(self) -> bool:
        return False

    @property
    def anchor_or_none(self):
        return None

    @property
    def status_text(self):
        return None
