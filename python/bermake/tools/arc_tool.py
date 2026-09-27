"""The 2-Point Arc drawing tool.

Three clicks: start, end (defines the chord and the drawing plane via the first
snap), then a bulge point setting the bow. Commits an open 12-segment polyline
(no face). A near-semicircle bulge snaps to an exact half-circle. ESC cancels.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.scene_commands import SplitFaceCommand
from bermake.geometry import arc_2pt, semicircle_snap
from bermake.tools.shape_support import (
    build_open_polyline,
    chain_cuts_face,
    polyline_segments,
    resolve_drawing_plane,
)
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.snap_engine import MARKER_COLOR_BY_KIND

_NEUTRAL_COLOR = (0.85, 0.85, 0.85)
_MIN_CHORD = 1e-4
_SEGMENTS = 12
# Shared read-only constant: the arc start is always the plane origin (0, 0).
# Frozen writeable=False so any accidental in-place write by a callee fails loudly
# instead of silently corrupting this module-global.
_ORIGIN_UV = np.zeros(2)
_ORIGIN_UV.flags.writeable = False


class _State(Enum):
    IDLE = 0
    PLACING_END = 1
    PLACING_BULGE = 2


class ArcTool(Tool):
    @property
    def name(self) -> str:
        return "Arc"

    @property
    def shortcut(self) -> str:
        return "A"

    @property
    def id(self) -> str:
        return "arc"

    def __init__(self) -> None:
        self._scene = None
        self._command_stack = None
        self._model = None
        self._units_provider = None
        self._state = _State.IDLE
        self._plane = None
        self._start: np.ndarray | None = None  # world
        self._end_uv: np.ndarray | None = None
        self._cursor_uv: np.ndarray | None = None  # live projected cursor (preview)
        self._snap_marker_pos: np.ndarray | None = None
        self._snap_marker_color: tuple[float, float, float] = _NEUTRAL_COLOR
        self._snap_marker_kind = 0

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
        if self._plane is not None and self._state != _State.IDLE:
            self._cursor_uv = self._plane.project(snap.world_position)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if snap.kind == SnapKind.NONE:
            return
        s = self._scene  # type: ignore[assignment]

        if self._state == _State.IDLE:
            self._plane = resolve_drawing_plane(snap, s)
            self._start = snap.world_position.copy()
            self._cursor_uv = _ORIGIN_UV.copy()
            self._state = _State.PLACING_END
            return

        if self._plane is None:
            self._reset_gesture()
            return

        if self._state == _State.PLACING_END:
            end_uv = self._plane.project(snap.world_position)
            if float(np.linalg.norm(end_uv)) < _MIN_CHORD:
                return  # end coincides with start — keep waiting
            self._end_uv = end_uv
            self._cursor_uv = end_uv.copy()
            self._state = _State.PLACING_BULGE
            return

        # PLACING_BULGE → commit
        if self._end_uv is None:
            self._reset_gesture()
            return
        bulge_uv = semicircle_snap(
            _ORIGIN_UV, self._end_uv, self._plane.project(snap.world_position)
        )
        pts_uv = arc_2pt(_ORIGIN_UV, self._end_uv, bulge_uv, _SEGMENTS)
        if len(pts_uv) < 2:
            return
        world = self._plane.to_world(pts_uv).astype(np.float32)
        self._commit_polyline(world)
        self._reset_gesture()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset_gesture()

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        if self._plane is not None and self._cursor_uv is not None:
            if self._state == _State.PLACING_END:
                world = self._plane.to_world(np.stack([_ORIGIN_UV, self._cursor_uv])).astype(
                    np.float32
                )
                segments = polyline_segments(world, closed=False)
            elif self._state == _State.PLACING_BULGE and self._end_uv is not None:
                bulge_uv = semicircle_snap(_ORIGIN_UV, self._end_uv, self._cursor_uv)
                pts_uv = arc_2pt(_ORIGIN_UV, self._end_uv, bulge_uv, _SEGMENTS)
                world = self._plane.to_world(pts_uv).astype(np.float32)
                segments = polyline_segments(world, closed=False)
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
        return self._state != _State.IDLE

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        if self._state != _State.IDLE and self._start is not None:
            return self._start.copy()
        return None

    @property
    def status_text(self) -> str | None:
        if self._state == _State.PLACING_END:
            return "Pick arc end"
        if self._state == _State.PLACING_BULGE:
            return "Drag the bulge"
        return None

    @property
    def measurement_text(self) -> str | None:
        """The live chord length while placing the end point.

        Reuses the same norm(cursor_uv) apply_typed_value's PLACING_END
        branch already computes to turn a typed length into the chord.
        PLACING_BULGE has no equivalent already-tracked scalar (the bulge
        preview only ever produces a 2D point, never a live sagitta
        magnitude), so it reports nothing rather than invent one.
        """
        if self._state != _State.PLACING_END or self._cursor_uv is None:
            return None
        length = float(np.linalg.norm(np.asarray(self._cursor_uv, np.float64)))
        if self._units_provider is not None:
            from bermake.units import format_length

            return format_length(length, self._units_provider())
        return f"{length:.3f}"

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if self._plane is None:
            return False
        val = parse_length(text, units)
        if val is None or val <= 0:
            return False
        if self._state == _State.PLACING_END and self._cursor_uv is not None:
            d = np.asarray(self._cursor_uv, np.float64)
            norm = float(np.linalg.norm(d))
            if norm < _MIN_CHORD:
                return False
            self._end_uv = (d / norm * val).astype(np.float64)
            self._cursor_uv = self._end_uv.copy()
            self._state = _State.PLACING_BULGE
            return True
        if self._state == _State.PLACING_BULGE and self._end_uv is not None:
            # Bulge point = chord midpoint + sagitta * unit-perpendicular, on the
            # side the cursor is currently on.
            mid = (_ORIGIN_UV + self._end_uv) / 2.0
            chord = self._end_uv - _ORIGIN_UV
            perp = np.array([-chord[1], chord[0]], np.float64)
            perp /= np.linalg.norm(perp) + 1e-12
            side = 1.0
            if self._cursor_uv is not None and float(np.dot(self._cursor_uv - mid, perp)) < 0:
                side = -1.0
            bulge_uv = mid + side * val * perp
            pts_uv = arc_2pt(_ORIGIN_UV, self._end_uv, bulge_uv, _SEGMENTS)
            if len(pts_uv) < 2:
                self._reset_gesture()
                return False
            world = self._plane.to_world(pts_uv).astype(np.float32)
            self._commit_polyline(world)
            self._reset_gesture()
            return True
        return False

    def _commit_polyline(self, world: np.ndarray) -> None:
        """Commit the drawn arc as one undo step, and split whichever single
        face (if any) the chord crosses -- the same face-split check
        LineTool applies on its own gesture-end paths (M7.6a task 5).

        build_open_polyline hands back the resolved vertex-id chain along
        with the composite, so chain_cuts_face can be asked about it
        directly -- no separate position lookup needed, and none of
        build_open_polyline's own world->local conversion needs repeating
        here. The split command is appended to the SAME composite, before
        push_executed, so one undo reverses both the arc and the split.
        """
        s = self._scene
        result = build_open_polyline(
            s, world, name="Draw Arc", world_transform=self._world_transform()
        )
        if result is None:
            return
        composite, chain = result
        # KNOWN LIMITATION (branch review Finding 3): unlike LineTool, ArcTool
        # has no `_vertex_for_snap` edge-splitting step, so an endpoint that
        # merely lands on an edge's INTERIOR is never turned into a loop
        # vertex first -- build_open_polyline only reuses a vertex already
        # within _COINCIDENT_EPS and otherwise drops a free-floating one that
        # is not part of any face's boundary loop. chain_cuts_face requires
        # both chain ends to already be loop vertices, so an arc whose
        # endpoints land mid-edge (the common case) never splits, even
        # though the identical corner-to-corner gesture does. See
        # tests/test_arc_tool_face_split.py for the pinned-behaviour test;
        # do not "fix" this here without reading Finding 3's rationale in
        # the M7.6a final-fix-wave review first.
        fid = chain_cuts_face(s, chain)
        if fid is not None:
            split_cmd = SplitFaceCommand(fid, chain)
            split_cmd.do(s)
            composite.children.append(split_cmd)
        if self._command_stack is not None:
            self._command_stack.push_executed(composite, self._scene)

    def _reset_gesture(self) -> None:
        self._state = _State.IDLE
        self._plane = None
        self._start = None
        self._end_uv = None
        self._cursor_uv = None
        self._snap_marker_pos = None
        self._snap_marker_kind = 0
        self._snap_marker_color = _NEUTRAL_COLOR
