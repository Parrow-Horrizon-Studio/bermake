"""The Rectangle drawing tool.

Two-corner gesture: first click sets the first corner, second click commits
an axis-aligned rectangle on the active context's local ground plane (local
Z=0). ESC cancels mid-drag.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.scene_commands import (
    AddEdgeCommand,
    AddFaceCommand,
    AddVertexCommand,
)
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import world_to_local_point
from bermake.viewport.snap_engine import MARKER_COLOR_BY_KIND


class _State(Enum):
    IDLE = 0
    DRAGGING = 1


_NEUTRAL_COLOR = (0.85, 0.85, 0.85)


class RectangleTool(Tool):
    @property
    def name(self) -> str:
        return "Rectangle"

    @property
    def shortcut(self) -> str:
        return "R"

    @property
    def id(self) -> str:
        return "rectangle"

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._units_provider = None
        self._state = _State.IDLE
        self._first_corner: np.ndarray | None = None
        self._preview_corner: np.ndarray | None = None
        self._snap_marker_pos: np.ndarray | None = None
        self._snap_marker_color: tuple[float, float, float] = _NEUTRAL_COLOR
        self._snap_marker_kind: int = 0
        self._composite: CompositeCommand | None = None
        self._command_stack = None  # populated in activate()

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene  # type: ignore[assignment]
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._units_provider = ctx.units_provider
        self._reset_gesture()

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def deactivate(self) -> None:
        self._reset_gesture()

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if snap.kind == SnapKind.NONE:
            self._snap_marker_pos = None
            self._snap_marker_kind = 0
            return
        self._snap_marker_pos = snap.world_position.copy()
        self._snap_marker_color = MARKER_COLOR_BY_KIND.get(snap.kind, _NEUTRAL_COLOR)
        self._snap_marker_kind = int(snap.kind)
        if self._state == _State.DRAGGING:
            self._preview_corner = snap.world_position.copy()

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if snap.kind == SnapKind.NONE:
            return

        if self._state == _State.IDLE:
            self._first_corner = snap.world_position.copy()
            self._preview_corner = snap.world_position.copy()
            self._state = _State.DRAGGING
            return

        # DRAGGING — commit or drop
        assert self._first_corner is not None
        second = snap.world_position
        if np.array_equal(second, self._first_corner):
            self._reset_gesture()
            return
        self._commit_rect(second)

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() != Qt.Key.Key_Escape:
            return
        if self._composite is not None:
            # We haven't built a composite mid-drag in Rectangle (it commits
            # atomically on second click), so there is nothing to roll back.
            self._composite = None
        self._reset_gesture()

    def overlay(self) -> ToolOverlay:
        if (
            self._state == _State.DRAGGING
            and self._first_corner is not None
            and self._preview_corner is not None
        ):
            # Resolve in the same frame _commit_rect will actually use -- the
            # active context's local ground plane (local Z=0) -- rather than a
            # literal world Z=0, then transform back to world for display.
            # Inside a group translated or rotated in z, local Z=0 is not
            # world Z=0, so a hardcoded 0.0 here would draw the rubber band on
            # a different plane than where the rectangle is about to land.
            wt = self._world_transform()
            p0 = world_to_local_point(self._first_corner, wt)
            p1 = world_to_local_point(self._preview_corner, wt)
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            segments = np.array(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                    [x0, y1, 0.0],
                    [x0, y0, 0.0],
                ],
                dtype=np.float64,
            )
            if wt is not None and not is_identity_transform(wt):
                segments = apply_mat(segments, wt)
            segments = segments.astype(np.float32)
        else:
            segments = np.zeros((0, 3), dtype=np.float32)

        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_NEUTRAL_COLOR,
            snap_marker_position=(
                self._snap_marker_pos.copy() if self._snap_marker_pos is not None else None
            ),
            snap_marker_color=self._snap_marker_color,
            snap_marker_kind=self._snap_marker_kind,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._state == _State.DRAGGING

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None  # Rectangle tool doesn't drive axis-lock

    def _current_size(self) -> tuple[float, float] | None:
        """(width, height) of the live rectangle, or None with no drag yet.

        Reused by measurement_text -- the same two magnitudes
        apply_typed_value parses out of a typed "<w>x<h>" string."""
        if (
            self._state != _State.DRAGGING
            or self._first_corner is None
            or self._preview_corner is None
        ):
            return None
        # LOCAL (active-context) magnitudes, the same frame apply_typed_value,
        # _commit_rect and overlay all resolve in. Reading raw world x/y
        # instead swapped the two numbers under a context rotated 90 degrees
        # about Z, so the readout invited the user to type back a value that
        # would have built a different rectangle than the one on screen.
        wt = self._world_transform()
        p0 = world_to_local_point(self._first_corner, wt)
        p1 = world_to_local_point(self._preview_corner, wt)
        width = abs(float(p1[0]) - float(p0[0]))
        height = abs(float(p1[1]) - float(p0[1]))
        return width, height

    @property
    def measurement_text(self) -> str | None:
        size = self._current_size()
        if size is None:
            return None
        width, height = size
        if self._units_provider is not None:
            from bermake.units import format_length

            units = self._units_provider()
            return f"{format_length(width, units)} x {format_length(height, units)}"
        return f"{width:.3f} x {height:.3f}"

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if (
            self._state != _State.DRAGGING
            or self._first_corner is None
            or self._preview_corner is None
        ):
            return False
        parts = text.replace("*", "x").replace("X", "x").split("x")
        if len(parts) != 2:
            return False
        w = parse_length(parts[0], units)
        h = parse_length(parts[1], units)
        if w is None or h is None or w <= 0 or h <= 0:
            return False
        # Resolve in local (active-context) space, matching _commit_rect and
        # overlay: a literal world z=0 second corner would land on a
        # different plane than the drag itself under a rotated/translated
        # active context.
        wt = self._world_transform()
        p0 = world_to_local_point(self._first_corner, wt)
        p1 = world_to_local_point(self._preview_corner, wt)
        fx, fy = float(p0[0]), float(p0[1])
        sx = 1.0 if float(p1[0]) >= fx else -1.0
        sy = 1.0 if float(p1[1]) >= fy else -1.0
        local_second = np.array([fx + sx * w, fy + sy * h, 0.0], dtype=np.float64)
        if wt is not None and not is_identity_transform(wt):
            second = apply_mat(local_second, wt)[0]
        else:
            second = local_second.astype(np.float32)
        self._commit_rect(second)
        return True

    # ---- internal -------------------------------------------------------
    def _commit_rect(self, second) -> None:
        """Normalize the two corners and commit a rectangle to the scene.

        Resolves both corners in the active context's local frame first, then
        builds the loop at local z=0 -- matching PrimitiveTool's frame
        convergence. Converting a literal world z=0 corner to local (the old
        order) sinks the rectangle onto the wrong plane whenever the active
        context is translated or rotated in z.
        """
        assert self._first_corner is not None
        wt = self._world_transform()
        p0 = world_to_local_point(self._first_corner, wt)
        p1 = world_to_local_point(second, wt)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])

        # Normalize to a canonical CCW-from-above winding (min -> max on each
        # axis) so the face normal always points +Z (up), regardless of which
        # diagonal the user dragged the second corner toward. Without this, a
        # down-right / up-left drag yields a -Z normal and push/pull would
        # extrude the rectangle downward instead of up.
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)

        composite = CompositeCommand(name="Draw Rectangle")
        s = self._scene  # type: ignore[assignment]
        local_corners = [
            np.array([xlo, ylo, 0.0], dtype=np.float32),
            np.array([xhi, ylo, 0.0], dtype=np.float32),
            np.array([xhi, yhi, 0.0], dtype=np.float32),
            np.array([xlo, yhi, 0.0], dtype=np.float32),
        ]
        v_cmds = [AddVertexCommand(p) for p in local_corners]
        for c in v_cmds:
            c.do(s)
            composite.children.append(c)
        vids = [c.vertex_id for c in v_cmds]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            e_cmd = AddEdgeCommand(vids[a], vids[b])
            e_cmd.do(s)
            composite.children.append(e_cmd)
        f_cmd = AddFaceCommand(tuple(vids))
        f_cmd.do(s)
        composite.children.append(f_cmd)

        if self._command_stack is not None:
            self._command_stack.push_executed(composite, self._scene)
        self._reset_gesture()

    def _reset_gesture(self) -> None:
        self._state = _State.IDLE
        self._first_corner = None
        self._preview_corner = None
        self._snap_marker_pos = None
        self._snap_marker_kind = 0
        self._composite = None
