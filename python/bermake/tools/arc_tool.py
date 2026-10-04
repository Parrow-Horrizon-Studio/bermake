"""The 2-Point Arc drawing tool.

Three clicks: start, end (defines the chord and the drawing plane via the first
snap), then a bulge point setting the bow. Commits an open 12-segment polyline
(no face). A near-semicircle bulge snaps to an exact half-circle. ESC cancels.
"""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.scene_commands import SplitFaceCommand
from bermake.geometry import arc_2pt, semicircle_snap
from bermake.tools.shape_support import (
    build_open_polyline,
    chain_cuts_face,
    polyline_segments,
    resolve_drawing_plane,
    vertex_for_snap,
)
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import world_to_local_point
from bermake.viewport.snap_engine import SnapKind, marker_color

_NEUTRAL_COLOR = (0.85, 0.85, 0.85)
_MIN_CHORD = 1e-4
_SEGMENTS = 12
# Shared read-only constant: the arc start is always the plane origin (0, 0).
# Frozen writeable=False so any accidental in-place write by a callee fails loudly
# instead of silently corrupting this module-global.
_ORIGIN_UV = np.zeros(2)
_ORIGIN_UV.flags.writeable = False


def _resolves_to_topology(snap) -> bool:
    """True when `snap` names an existing vertex or an edge interior.

    Only those are resolved through vertex_for_snap at commit. Any other snap
    is left to build_open_polyline, which reuses a coincident vertex or adds
    one, so a plain click never adds a second vertex on top of an existing one.
    """
    if snap.kind == SnapKind.ENDPOINT and snap.vertex_id is not None:
        return True
    return (
        snap.edge_id is not None
        and snap.edge_t is not None
        and snap.kind in (SnapKind.MIDPOINT, SnapKind.ON_EDGE, SnapKind.INTERSECTION)
    )


# How far off the drawing plane a snap may sit and still count as a point the
# arc reaches. 1e-4 still missed corner snaps at 2000 m (float32 snap rounding).
_PLANE_TOL = 1e-3
# Pinning an arc end onto its resolved vertex moves it by float rounding only.
_PIN_TOL = 1e-3


def _on_plane(plane, snap) -> bool:
    """True when the snap's point lies on the arc's drawing plane.

    The plane comes from the START snap, and the end is projected onto it. An
    end that was projected away from its edge (it lies off the plane) is not
    on that edge, so it must not split it.
    """
    d = np.asarray(snap.world_position, np.float64) - plane.origin
    return abs(float(d @ plane.normal)) <= _PLANE_TOL


def _is_stale(scene, snap, world_transform) -> bool:
    """True when the snap names an edge or vertex that no longer exists, or
    an edge that has since moved away from the point the snap recorded.

    A snap is stored at its press and resolved up to two clicks later, and an
    undo in between can remove what it named or move it (undoing a Move).
    """
    if snap.edge_id is not None and not scene.edge_is_live(snap.edge_id):
        return True
    if snap.edge_id is not None and snap.edge_t is not None:
        at = scene.point_on_edge(snap.edge_id, snap.edge_t)
        here = world_to_local_point(snap.world_position, world_transform)
        if float(np.linalg.norm(at - here)) > _PIN_TOL:
            return True
    if snap.vertex_id is not None:
        try:
            scene.vertex(snap.vertex_id)
        except KeyError:
            return True
    return False


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
        # The snaps the chord's two ends were clicked on. Kept so the commit can
        # split an edge an end landed on (#122). None for a typed chord length.
        self._start_snap = None
        self._end_snap = None
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
        if snap.kind == SnapKind.NONE:
            self._snap_marker_pos = None
            self._snap_marker_kind = 0
            return
        self._snap_marker_pos = snap.world_position.copy()
        self._snap_marker_color = marker_color(snap, _NEUTRAL_COLOR)
        self._snap_marker_kind = int(snap.kind)
        if self._plane is not None and self._state != _State.IDLE:
            self._cursor_uv = self._plane.project(snap.world_position)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            return
        s = self._scene  # type: ignore[assignment]

        if self._state == _State.IDLE:
            self._plane = resolve_drawing_plane(snap, s)
            self._start = snap.world_position.copy()
            self._start_snap = snap
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
            self._end_snap = snap
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
            self._end_snap = None  # a typed chord end is not on any snap
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
        """Commit the drawn arc as one undo step.

        The step holds, in order: a split of each edge a chord end landed on
        (#122, as LineTool does), the arc's vertices and edges, and the split
        of whichever single face (if any) the chord then crosses (M7.6a). One
        undo reverses all of it.

        build_open_polyline hands back the resolved vertex-id chain along
        with the composite, so chain_cuts_face can be asked about it
        directly. The face split is appended to the SAME outer composite,
        before push_executed.
        """
        s = self._scene
        outer = CompositeCommand(name="Draw Arc")
        # Resolve both ends first. An end that lands on an edge's interior
        # splits that edge (D6: always, even when no face ends up cut), so the
        # end is a real loop vertex and build_open_polyline reuses it because
        # the arc's first and last points coincide with it.
        # A snap whose edge or vertex died since its press (an undo mid-gesture)
        # is a plain point: it splits nothing. Checked for both before any split
        # runs, because the start's own split may retire the end's edge later
        # (the same-edge case below handles that one).
        start_snap = self._start_snap
        end_snap = self._end_snap
        wt = self._world_transform()
        if start_snap is not None and _is_stale(s, start_snap, wt):
            start_snap = None
        if end_snap is not None and _is_stale(s, end_snap, wt):
            end_snap = None
        # The start is the plane's origin, so it is on the plane by construction.
        # An end off the plane was projected onto it and is not where its edge is.
        plane = self._plane
        if end_snap is not None and not _on_plane(plane, end_snap):
            end_snap = None
        host_ends = None
        start_vid = None
        end_vid = None
        if start_snap is not None and _resolves_to_topology(start_snap):
            if start_snap.edge_id is not None and s.edge_is_live(start_snap.edge_id):
                e = s.edge(start_snap.edge_id)
                host_ends = (e.v1_id, e.v2_id)
            start_vid = vertex_for_snap(self._tool_context(), start_snap, outer)
        if end_snap is not None and _resolves_to_topology(end_snap):
            if (
                host_ends is not None
                and start_vid is not None
                and end_snap.edge_id == start_snap.edge_id
                and not s.edge_is_live(end_snap.edge_id)
            ):
                end_snap = self._reresolve_on_sub_edges(end_snap, host_ends, start_vid)
            end_vid = vertex_for_snap(self._tool_context(), end_snap, outer)
        # Build in the scene's local frame with each end set exactly onto the
        # vertex it resolved to, so the join never depends on float rounding of
        # the plane round-trip (it missed far from the origin). An end is only
        # moved when it is already there to within float rounding, so the arc
        # is never bent out of its plane.
        local = np.array([world_to_local_point(p, wt) for p in world], dtype=np.float32)
        for idx, vid in ((0, start_vid), (-1, end_vid)):
            if vid is None:
                continue
            pos = s.vertex(vid).position
            if float(np.linalg.norm(local[idx] - pos)) <= _PIN_TOL:
                local[idx] = pos
        result = build_open_polyline(s, local, name="Draw Arc", world_transform=None)
        if result is None:
            for cmd in reversed(outer.children):
                cmd.undo(s)
            return
        composite, chain = result
        outer.children.append(composite)
        fid = chain_cuts_face(s, chain)
        if fid is not None:
            split_cmd = SplitFaceCommand(fid, chain)
            split_cmd.do(s)
            outer.children.append(split_cmd)
        if self._command_stack is not None:
            self._command_stack.push_executed(outer, self._scene)

    def _tool_context(self) -> ToolContext:
        return ToolContext(scene=self._scene, model=self._model)

    def _reresolve_on_sub_edges(self, snap, host_ends, split_vid):
        """`snap` re-aimed at whichever half of a just-split edge holds it.

        Both chord ends on one edge: the first end's split retired the host
        edge's id and left two halves, (a, split_vid) and (split_vid, b). The
        second end is looked up again among those halves with
        closest_point_on_edge, so it splits the right one at the right t.
        """
        s = self._scene
        pos = np.asarray(
            world_to_local_point(snap.world_position, self._world_transform()), np.float32
        )
        best = None
        for other in host_ends:
            e_id = s.edge_between(other, split_vid)
            if e_id is None:
                continue
            point, t = s.closest_point_on_edge(e_id, pos)
            d = float(np.linalg.norm(point - pos))
            if best is None or d < best[0]:
                best = (d, e_id, t)
        return snap if best is None else replace(snap, edge_id=best[1], edge_t=best[2])

    def _reset_gesture(self) -> None:
        self._state = _State.IDLE
        self._plane = None
        self._start = None
        self._start_snap = None
        self._end_snap = None
        self._end_uv = None
        self._cursor_uv = None
        self._snap_marker_pos = None
        self._snap_marker_kind = 0
        self._snap_marker_color = _NEUTRAL_COLOR
