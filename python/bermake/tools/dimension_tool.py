"""The linear Dimension tool (M7d), shortcut "I".

Three clicks: snap p1, snap p2, then a third click whose perpendicular
component (relative to the p1->p2 axis) becomes the dimension line's offset.

Frame handling (the two things this project has repeatedly got wrong):
- The live rubber-band preview is built entirely in WORLD space, because the
  renderer draws `ToolOverlay.rubber_band_segments` in world space with no
  model matrix applied. `_p1_world`/`_p2_world`/`_cursor_world` are therefore
  kept as world-space snaps for as long as the gesture is open.
- The committed `Dimension`'s p1/p2/offset are CONTEXT-LOCAL (see
  `bermake.model.annotation.Dimension`), so every point is converted via
  `world_to_active_local` at the moment it is written to storage -- not
  before. The offset's perpendicular decomposition happens in that same
  local frame, since it is a local vector once stored.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.annotation_commands import CreateAnnotationCommand
from bermake.model.annotation import Dimension
from bermake.tools.annotation_support import NEUTRAL_PREVIEW_COLOR, world_to_active_local
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.snap_engine import MARKER_COLOR_BY_KIND, SnapKind

_EPS = 1e-9


def _perpendicular_offset(p1, p2, point):
    """The component of (point - midpoint) perpendicular to the p1->p2 axis,
    or None if p1 and p2 coincide (no axis to be perpendicular to). Works in
    whatever frame p1/p2/point already share (local or world) -- callers are
    responsible for keeping that frame consistent."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length < _EPS:
        return None
    axis_unit = axis / length
    mid = (p1 + p2) / 2.0
    raw = np.asarray(point, dtype=np.float64) - mid
    return raw - axis_unit * float(np.dot(raw, axis_unit))


class DimensionTool(Tool):
    """Click (p1) -> click (p2) -> click (offset): one linear dimension."""

    def __init__(self) -> None:
        self._model = None
        self._command_stack = None
        self._p1_world: np.ndarray | None = None
        self._p2_world: np.ndarray | None = None
        self._cursor_world: np.ndarray | None = None
        self._snap_color: tuple[float, float, float] = NEUTRAL_PREVIEW_COLOR
        self._snap_kind = 0

    @property
    def name(self) -> str:
        return "Dimension"

    @property
    def shortcut(self) -> str:
        return "I"

    @property
    def id(self) -> str:
        return "dimension"

    def activate(self, ctx: ToolContext) -> None:
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _offset_to_world(self, offset_local: np.ndarray) -> np.ndarray:
        """A LOCAL offset *vector* (no translation) -> WORLD, for drawing the
        live preview's offset dimension line at the correct world position."""
        wt = self._world_transform()
        offset_local = np.asarray(offset_local, dtype=np.float64)
        if wt is None:
            return offset_local
        return np.asarray(wt, dtype=np.float64)[:3, :3] @ offset_local

    def _local_offset_towards(self, world_point) -> np.ndarray | None:
        """Perpendicular local offset for a candidate world point, given the
        gesture's current p1_world/p2_world (both already snapped)."""
        p1_local = world_to_active_local(self._model, self._p1_world)
        p2_local = world_to_active_local(self._model, self._p2_world)
        point_local = world_to_active_local(self._model, world_point)
        return _perpendicular_offset(p1_local, p2_local, point_local)

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            self._cursor_world = None
            self._snap_kind = 0
            return
        self._cursor_world = np.asarray(snap.world_position, dtype=np.float64).copy()
        self._snap_color = MARKER_COLOR_BY_KIND.get(snap.kind, NEUTRAL_PREVIEW_COLOR)
        self._snap_kind = int(snap.kind)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            return
        pt_world = np.asarray(snap.world_position, dtype=np.float64).copy()

        if self._p1_world is None:
            self._p1_world = pt_world
            return

        if self._p2_world is None:
            if float(np.linalg.norm(pt_world - self._p1_world)) < _EPS:
                self._reset()  # degenerate measurement: zero-length axis
                return
            self._p2_world = pt_world
            return

        offset_local = self._local_offset_towards(pt_world)
        if offset_local is None:
            self._reset()
            return

        p1_local = world_to_active_local(self._model, self._p1_world)
        p2_local = world_to_active_local(self._model, self._p2_world)
        dim = Dimension(
            self._model.new_annotation_id(),
            tuple(float(v) for v in p1_local),
            tuple(float(v) for v in p2_local),
            tuple(float(v) for v in offset_local),
        )
        self._command_stack.execute(
            CreateAnnotationCommand(dim, self._model.active_context), self._model
        )
        self._reset()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset()

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        if self._p1_world is not None and self._p2_world is not None:
            pts = [self._p1_world, self._p2_world]
            if self._cursor_world is not None:
                offset_local = self._local_offset_towards(self._cursor_world)
                if offset_local is not None:
                    offset_world = self._offset_to_world(offset_local)
                    d1 = self._p1_world + offset_world
                    d2 = self._p2_world + offset_world
                    pts = [
                        self._p1_world,
                        self._p2_world,  # measured segment
                        self._p1_world,
                        d1,  # extension line 1
                        self._p2_world,
                        d2,  # extension line 2
                        d1,
                        d2,  # dimension line
                    ]
            segments = np.array(pts, dtype=np.float32).reshape(-1, 3)
        elif self._p1_world is not None and self._cursor_world is not None:
            segments = np.array([self._p1_world, self._cursor_world], dtype=np.float32).reshape(
                -1, 3
            )
        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=NEUTRAL_PREVIEW_COLOR,
            snap_marker_position=(
                self._cursor_world.copy() if self._cursor_world is not None else None
            ),
            snap_marker_color=self._snap_color,
            snap_marker_kind=self._snap_kind,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._p1_world is not None

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None

    @property
    def status_text(self) -> str | None:
        return None

    def _reset(self) -> None:
        self._p1_world = None
        self._p2_world = None
        self._cursor_world = None
        self._snap_color = NEUTRAL_PREVIEW_COLOR
        self._snap_kind = 0
