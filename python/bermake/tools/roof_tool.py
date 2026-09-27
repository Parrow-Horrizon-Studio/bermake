"""The Roof placement tool (M7c).

Draw a rectangle footprint on the active drawing plane; a parametric
Gable/Hip/Shed roof (from the options row) is baked as a "Roof" group over it,
ridge auto-aligned to the longer edge. Up/Down arrows rotate the ridge 90°.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.roof_commands import CreateRoofCommand
from bermake.geometry.roof import _MAX_SLOPE_DEG, _rot_z, roof_solid
from bermake.geometry.transforms import mat_invert
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.snap_engine import MARKER_COLOR_BY_KIND, SnapKind

_NEUTRAL = (0.85, 0.85, 0.85)


class RoofTool(Tool):
    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._command_stack = None
        self._first: np.ndarray | None = None  # world, incl. base z0
        self._preview: np.ndarray | None = None  # world, x/y at z0
        self._flip_quarters = 0
        self._snap_pos: np.ndarray | None = None
        self._snap_color: tuple[float, float, float] = _NEUTRAL
        self._snap_kind = 0
        self.kind = "gable"
        self.slope = 30.0  # degrees

    @property
    def name(self) -> str:
        return "Roof"

    @property
    def shortcut(self) -> str:
        return "O"

    @property
    def id(self) -> str:
        return "roof"

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    # ---- geometry of the placement -------------------------------------
    def _dims_and_transform(self, second_world):
        """Return (w, d, transform_local, m_world) for the current footprint,
        or None if the footprint is degenerate (zero area)."""
        z0 = float(self._first[2])
        x0, y0 = float(self._first[0]), float(self._first[1])
        x1, y1 = float(second_world[0]), float(second_world[1])
        width_x = abs(x1 - x0)
        width_y = abs(y1 - y0)
        if width_x < 1e-9 or width_y < 1e-9:
            return None
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        base_q = 0 if width_y >= width_x else 1  # ridge along the longer edge
        q = (base_q + self._flip_quarters) % 4
        if q % 2 == 0:
            w, d = width_x, width_y
        else:
            w, d = width_y, width_x
        m_world = _rot_z(np.pi / 2.0 * q)
        m_world[:3, 3] = [cx, cy, z0]
        wt = self._model.active_world_transform if self._model is not None else None
        if wt is None:
            transform_local = m_world
        else:
            transform_local = mat_invert(wt) @ m_world
        return w, d, transform_local, m_world

    def _commit(self, second_world) -> None:
        dims = self._dims_and_transform(second_world)
        if dims is None:
            self._reset()
            return
        w, d, transform_local, _m_world = dims
        cmd = CreateRoofCommand(
            self.kind, w, d, self.slope, transform_local, self._model.active_context
        )
        self._command_stack.execute(cmd, self._model)
        self._reset()

    # ---- events ---------------------------------------------------------
    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            self._snap_pos = None
            self._snap_kind = 0
            return
        self._snap_pos = np.asarray(snap.world_position, np.float64).copy()
        self._snap_color = MARKER_COLOR_BY_KIND.get(snap.kind, _NEUTRAL)
        self._snap_kind = int(snap.kind)
        if self._first is not None:
            z0 = float(self._first[2])
            self._preview = np.array(
                [float(snap.world_position[0]), float(snap.world_position[1]), z0],
                dtype=np.float64,
            )

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            return
        pt = np.asarray(snap.world_position, np.float64)
        if self._first is None:
            self._first = pt.copy()
            self._preview = pt.copy()
            return
        second = np.array([float(pt[0]), float(pt[1]), float(self._first[2])], np.float64)
        self._commit(second)

    def on_key_press(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._reset()
        elif key == Qt.Key.Key_Up:
            self._flip_quarters = (self._flip_quarters + 1) % 4
        elif key == Qt.Key.Key_Down:
            self._flip_quarters = (self._flip_quarters - 1) % 4

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_angle

        value = parse_angle(text)
        if value is None or value <= 0.0 or value > _MAX_SLOPE_DEG:
            return False
        self.slope = value
        return True

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        if self._first is not None and self._preview is not None:
            dims = self._dims_and_transform(self._preview)
            if dims is not None:
                w, d, _transform_local, m_world = dims
                verts, faces = roof_solid(self.kind, w, d, self.slope)
                if verts:
                    world = [(m_world @ np.append(np.array(v), 1.0))[:3] for v in verts]
                    segs = []
                    for f in faces:
                        n = len(f)
                        for i in range(n):
                            segs.append(world[f[i]])
                            segs.append(world[f[(i + 1) % n]])
                    segments = np.array(segs, dtype=np.float32)
        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_NEUTRAL,
            snap_marker_position=self._snap_pos.copy() if self._snap_pos is not None else None,
            snap_marker_color=self._snap_color,
            snap_marker_kind=self._snap_kind,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._first is not None

    @property
    def consumes_arrow_keys(self) -> bool:
        """Up/Down cycle the flip-quarters count while drawing."""
        return True

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return self._first.copy() if self._first is not None else None

    @property
    def status_text(self) -> str | None:
        return None

    def _reset(self) -> None:
        self._first = None
        self._preview = None
        self._snap_pos = None
        self._snap_kind = 0
        self._flip_quarters = 0  # ridge orientation is per-gesture, not sticky
        self._snap_color = _NEUTRAL
