"""The Wall drawing tool (M7a).

Chaining polyline of baked solid-box walls. Click to start; each later click
commits one wall (CreateWallCommand) and chains. Esc/Enter ends the chain.
Thickness/height are tool settings (meters) driven by the WallOptionsBar.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.wall_commands import CreateWallCommand
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import world_to_local_point
from bermake.viewport.snap_engine import SnapKind, marker_color

_NEUTRAL = (0.85, 0.85, 0.85)


class WallTool(Tool):
    """Click → click → click chained walls. Mirrors LineTool's gesture shape,
    but each committed segment is its own undoable `CreateWallCommand` (a
    baked solid box) rather than shared vertex/edge topology."""

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._command_stack = None
        self._anchor: np.ndarray | None = None  # world-space start of current segment
        self._preview_tip: np.ndarray | None = None  # world-space cursor
        self._snap_pos: np.ndarray | None = None
        self._snap_color: tuple[float, float, float] = _NEUTRAL
        self._snap_kind: int = 0
        self.thickness: float = 0.1  # meters
        self.height: float = 2.4  # meters

    @property
    def name(self) -> str:
        return "Wall"

    @property
    def shortcut(self) -> str:
        return "W"

    @property
    def id(self) -> str:
        return "wall"

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _to_local_ground(self, world_pt) -> np.ndarray:
        local = np.asarray(world_to_local_point(world_pt, self._world_transform()), np.float64)
        local[2] = 0.0  # base sits on the context ground plane
        return local

    def _commit(self, endpoint_world) -> None:
        start = self._to_local_ground(self._anchor)
        end = self._to_local_ground(endpoint_world)
        if float(np.linalg.norm(end - start)) < 1e-9:
            # Degenerate in the active context's ground plane (e.g. the two
            # points differ only in height): wall_box would return empty, so
            # skip rather than push a no-op command onto the undo stack.
            return
        cmd = CreateWallCommand(start, end, self.thickness, self.height, self._model.active_context)
        self._command_stack.execute(cmd, self._model)
        self._anchor = np.asarray(endpoint_world, np.float32).copy()
        self._preview_tip = np.asarray(endpoint_world, np.float32).copy()

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            self._snap_pos = None
            self._snap_kind = 0
            return
        self._snap_pos = np.asarray(snap.world_position, np.float32).copy()
        self._snap_color = marker_color(snap, _NEUTRAL)
        self._snap_kind = int(snap.kind)
        if self._anchor is not None:
            self._preview_tip = np.asarray(snap.world_position, np.float32).copy()

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            return
        pt = np.asarray(snap.world_position, np.float32)
        if self._anchor is None:
            self._anchor = pt.copy()
            self._preview_tip = pt.copy()
            return
        if float(np.linalg.norm(pt - self._anchor)) < 1e-6:
            return  # degenerate click
        self._commit(pt)

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if self._anchor is None or self._preview_tip is None:
            return False
        length = parse_length(text, units)
        if length is None or length <= 0:
            return False
        direction = np.asarray(self._preview_tip, np.float64) - np.asarray(self._anchor, np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return False
        endpoint = np.asarray(self._anchor, np.float64) + direction / norm * length
        self._commit(endpoint.astype(np.float32))
        return True

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self._reset()

    def overlay(self) -> ToolOverlay:
        if self._anchor is not None and self._preview_tip is not None:
            segments = np.array([self._anchor, self._preview_tip], dtype=np.float32)
        else:
            segments = np.zeros((0, 3), dtype=np.float32)
        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_NEUTRAL,
            snap_marker_position=self._snap_pos.copy() if self._snap_pos is not None else None,
            snap_marker_color=self._snap_color,
            snap_marker_kind=self._snap_kind,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._anchor is not None

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return self._anchor.copy() if self._anchor is not None else None

    @property
    def status_text(self) -> str | None:
        return None

    def _reset(self) -> None:
        self._anchor = None
        self._preview_tip = None
        self._snap_pos = None
        self._snap_kind = 0
