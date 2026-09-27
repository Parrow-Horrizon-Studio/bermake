"""The Offset tool -- SketchUp-style face-boundary offset.

Two-click gesture, mirroring PushPullTool's shape: arm a face on click,
track a signed distance on move, commit on the second click. The sign
convention is established by Task 4's sweep_support tests: positive
distance offsets inward, negative outward. This tool derives that sign
from which side of the face boundary the cursor sits on (inside == inward
== positive).

No option bar: SketchUp's Offset has no settings, so arming it must not
steal the Tool Settings tab the way Wall/Door-Window/Roof do.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.scene_commands import (
    AddFaceCommand,
    RemoveFaceCommand,
)
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.tools.sweep_support import loft_between_loops, offset_polygon, seam_merge
from bermake.tools.tool import Tool, ToolContext, ToolOverlay

# Visual constants (RGBA), matching PushPullTool's palette.
_HOVER_FILL_COLOR = (0.40, 0.70, 1.00, 0.20)  # light blue
_GHOST_FILL_COLOR = (0.40, 0.70, 1.00, 0.15)  # light blue, fainter

_MIN_COMMIT_DISTANCE = 1e-3  # world units; |distance| below this is a cancel


class _State(Enum):
    IDLE = 0
    HOVERING = 1
    DRAGGING = 2


class OffsetTool(Tool):
    """SketchUp-style face-boundary offset tool."""

    def __init__(self) -> None:
        self._scene = None
        self._command_stack = None
        self._camera = None
        self._widget_size_provider = None
        self._units_provider = None
        self._model = None

        self._state: _State = _State.IDLE

        # HOVERING data
        self._hovered_face_id: int | None = None

        # DRAGGING data (set when entering DRAGGING; cleared on exit)
        self._armed_face_id: int | None = None
        self._armed_face_loop: list[int] = []
        self._armed_face_normal: np.ndarray | None = None
        self._armed_face_points: np.ndarray | None = None
        self._current_distance: float = 0.0

    # ---- Tool ABC ------------------------------------------------------

    @property
    def name(self) -> str:
        return "Offset"

    @property
    def shortcut(self) -> str:
        return "F"

    @property
    def id(self) -> str:
        return "offset"

    @property
    def has_active_gesture(self) -> bool:
        return self._state == _State.DRAGGING

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None  # Offset doesn't drive axis-lock.

    def _applied_distance(self) -> float | None:
        """The distance committing NOW would actually apply, or None while
        not dragging.

        offset_polygon clamps a distance that would collapse the face; this
        returns what the ghost overlay already stops growing at, not the raw
        (possibly past-the-limit) drag distance. Reused by both status_text
        and measurement_text so the two never disagree."""
        if self._state != _State.DRAGGING:
            return None
        applied = self._current_distance
        if self._armed_face_points is not None and self._armed_face_normal is not None:
            _, applied = offset_polygon(
                self._armed_face_points, self._armed_face_normal, self._current_distance
            )
        return applied

    @property
    def status_text(self) -> str | None:
        applied = self._applied_distance()
        if applied is None:
            return None
        if self._units_provider is not None:
            from bermake.units import format_length

            return f"offset: {format_length(applied, self._units_provider())}"
        return f"offset: {applied:.3f}"

    @property
    def measurement_text(self) -> str | None:
        applied = self._applied_distance()
        if applied is None:
            return None
        if self._units_provider is not None:
            from bermake.units import format_length

            return format_length(applied, self._units_provider())
        return f"{applied:.3f}"

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._command_stack = ctx.command_stack
        self._camera = ctx.camera
        self._widget_size_provider = ctx.widget_size_provider
        self._units_provider = ctx.units_provider
        self._model = ctx.model
        self._reset_to_idle()

    def deactivate(self) -> None:
        self._reset_to_idle()

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    # ---- Event handlers -----------------------------------------------

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if self._state == _State.DRAGGING:
            self._update_distance_from_event(event)
            return
        # IDLE / HOVERING -- per-frame ray-pick.
        hit = self._pick_face_under_cursor(event)
        if hit is None:
            self._state = _State.IDLE
            self._hovered_face_id = None
        else:
            self._state = _State.HOVERING
            self._hovered_face_id = hit.face_id

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if self._state == _State.IDLE:
            return  # clicking empty space is a no-op
        if self._state == _State.HOVERING:
            # invariant: HOVERING implies hovered_face_id set
            assert self._hovered_face_id is not None
            self._arm_face(self._hovered_face_id)
            return
        # DRAGGING: commit if |distance| >= min threshold, else cancel.
        if abs(self._current_distance) >= _MIN_COMMIT_DISTANCE:
            self._commit_offset(self._current_distance)
        self._reset_to_idle()
        # After the gesture ends, immediately re-pick under the current cursor so we
        # transition to HOVERING (or IDLE) cleanly.
        hit = self._pick_face_under_cursor(event)
        if hit is not None:
            self._state = _State.HOVERING
            self._hovered_face_id = hit.face_id

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() != Qt.Key.Key_Escape:
            return
        if self._state == _State.DRAGGING:
            # Cancel -- no command pushed, scene was never mutated during DRAGGING.
            self._reset_to_idle()
            return
        # ESC in IDLE/HOVERING is owned by MainWindow's two-stage logic; no-op here.

    def overlay(self) -> ToolOverlay:
        polygons: list[np.ndarray] = []
        color = _HOVER_FILL_COLOR

        if self._state == _State.HOVERING and self._hovered_face_id is not None:
            polygons = [self._loop_world_coords(self._hovered_face_id)]
            color = _HOVER_FILL_COLOR
        elif self._state == _State.DRAGGING and self._armed_face_id is not None:
            polygons = self._build_ghost_polygons()
            color = _GHOST_FILL_COLOR

        # Lift polygons from local (active-context) coords to WORLD so the
        # renderer can draw them at identity (consistent with every other tool).
        if polygons:
            wt = self._world_transform()
            if wt is not None and not is_identity_transform(wt):
                polygons = [
                    apply_mat(np.asarray(p, np.float64), wt).astype(np.float32) for p in polygons
                ]

        return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
            rubber_band_color=(0.85, 0.85, 0.85),
            snap_marker_position=None,
            snap_marker_color=(0.85, 0.85, 0.85),
            snap_marker_kind=0,
            face_fill_polygons=polygons,
            face_fill_color=color,
        )

    # ---- Helpers -------------------------------------------------------

    def _pick_face_under_cursor(self, event: QMouseEvent):
        """Return RayMeshHit | None for the cursor position in `event`."""
        if self._camera is None or self._widget_size_provider is None or self._scene is None:
            return None
        pos = event.position()
        width, height = self._widget_size_provider()
        origin, direction = self._camera.ray_from_screen(
            float(pos.x()), float(pos.y()), int(width), int(height)
        )
        from bermake.viewport.picking import ray_into_local

        origin, direction = ray_into_local(origin, direction, self._world_transform())
        return self._scene.ray_pick_face(origin, direction)

    def _loop_world_coords(self, face_id: int) -> np.ndarray:
        """Return the face's boundary loop as an (N, 3) float32 ndarray."""
        assert self._scene is not None, "_loop_world_coords requires an active scene"
        loop_ids = self._scene.face_loop(face_id)
        coords = np.zeros((len(loop_ids), 3), dtype=np.float32)
        for i, vid in enumerate(loop_ids):
            v = self._scene.vertex(vid)
            coords[i] = v.position
        return coords

    def _arm_face(self, face_id: int) -> None:
        """Cache the source face's data and enter DRAGGING."""
        assert self._scene is not None
        self._armed_face_id = face_id
        self._armed_face_loop = list(self._scene.face_loop(face_id))
        self._armed_face_normal = self._scene.face_normal(face_id)
        self._armed_face_points = np.array(
            [self._scene.vertex(v).position for v in self._armed_face_loop],
            dtype=np.float64,
        )
        self._current_distance = 0.0
        self._state = _State.DRAGGING

    def _update_distance_from_event(self, event: QMouseEvent) -> None:
        """Intersect the cursor ray with the armed face's plane, then set
        `_current_distance` to the signed distance from that point to the
        polygon boundary: positive inside the face (inward offset),
        negative outside (outward) -- sweep_support.offset_polygon's own
        convention. Holds the previous distance if the ray is ~parallel to
        the plane (degenerate case) or the intersection is behind the
        camera."""
        if self._camera is None or self._widget_size_provider is None:
            return
        assert self._armed_face_normal is not None
        assert self._armed_face_points is not None

        pos = event.position()
        width, height = self._widget_size_provider()
        origin, direction = self._camera.ray_from_screen(
            float(pos.x()), float(pos.y()), int(width), int(height)
        )
        from bermake.viewport.picking import ray_into_local

        origin, direction = ray_into_local(origin, direction, self._world_transform())
        d_norm = float(np.linalg.norm(direction))
        if d_norm < 1e-9:
            return
        d_hat = direction / d_norm
        normal = self._armed_face_normal.astype(np.float64)

        denom = float(np.dot(d_hat, normal))
        if abs(denom) < 1e-9:
            return  # ray ~parallel to the plane; distance frozen
        p0 = self._armed_face_points[0]
        t = float(np.dot(p0 - origin.astype(np.float64), normal)) / denom
        if t < 0.0:
            return  # plane intersection is behind the camera; distance frozen
        point = origin.astype(np.float64) + t * d_hat

        self._current_distance = _signed_distance_to_boundary(
            point, self._armed_face_points, normal
        )

    def _build_ghost_polygons(self) -> list[np.ndarray]:
        """Return [armed_face_loop, ghost_offset_loop] in world coords."""
        assert self._armed_face_id is not None
        assert self._armed_face_normal is not None
        assert self._armed_face_points is not None

        outer = self._armed_face_points.astype(np.float32)
        offset_pts, _clamped = offset_polygon(
            self._armed_face_points, self._armed_face_normal, self._current_distance
        )
        inner = np.asarray(offset_pts, dtype=np.float32)
        return [outer, inner]

    def _commit_offset(self, distance: float) -> float:
        """Build the offset CompositeCommand and push it. Returns the
        distance actually applied, which is clamped when the requested one
        would collapse the face.

        offset_polygon's clamp (fix round, Task 6 review Finding 1) always
        lands strictly short of collapse: every returned edge has positive
        length and no two vertices coincide. That guarantee lives entirely
        in offset_polygon itself, so this method needs no special-casing --
        the offset loop always goes through the shared sweep layer exactly
        like Push/Pull's destination loop does.

        The two SIGNS, however, are genuinely different operations and each
        needs its own answer to three questions -- does the source face
        survive, does the new loop get a face, and is there a seam to
        merge:

        * Inward (positive): the new loop lies inside the source face, so
          the source face is replaced. Remove it, ring the annulus, and
          face the inner loop.
        * Outward (negative): the new loop lies outside the source face,
          which therefore still covers its own region and must be KEPT.
          The outer loop gets no face at all -- one would overlap every
          ring quad, coplanar and z-fighting. Removing the source face and
          facing the outer loop instead, as this method used to do
          unconditionally, produced exactly that overlap plus an unfaced
          hole where the source face had been.
        """
        scene = self._scene
        loop = list(scene.face_loop(self._armed_face_id))
        normal = scene.face_normal(self._armed_face_id)
        pts = np.array([scene.vertex(v).position for v in loop], dtype=np.float64)

        offset_pts, applied = offset_polygon(pts, normal, distance)
        outward = applied < 0.0

        # Seam candidates must be captured BEFORE the source face is
        # removed, since removal invalidates the boundary. There is nothing
        # to merge on the outward side: these same edges now separate the
        # SURVIVING source face from a ring quad coplanar with it, so
        # dissolving them would merge the ring straight back into the
        # source face and undo the offset. (Whether the new outer loop
        # should merge against a pre-existing coplanar neighbour is a
        # separate question neither sign answers today.)
        candidate_seam_edges = [] if outward else list(scene.face_edges(self._armed_face_id))

        composite = CompositeCommand(name="Offset")
        if not outward:
            rm = RemoveFaceCommand(self._armed_face_id)
            rm.do(scene)
            composite.children.append(rm)

        # Outward hands both loops to the loft REVERSED. Every ring quad
        # shares an edge with the surviving source face, and a half-edge
        # belongs to at most one face, so the quad has to traverse that
        # shared edge opposite to the way the source face's own boundary
        # does. Reversing both loops does exactly that, and it also lands
        # the ring quads' normals on the same side as the source face's
        # instead of facing away from it.
        result = loft_between_loops(
            scene,
            list(reversed(loop)) if outward else loop,
            offset_pts[::-1] if outward else offset_pts,
            cap_start=False,
            cap_end=False,
        )
        composite.children.extend(result.commands)

        if not outward:
            inner = AddFaceCommand(tuple(result.dst_vertex_ids))
            inner.do(scene)
            composite.children.append(inner)

        composite.children.extend(seam_merge(scene, candidate_seam_edges))
        self._command_stack.push_executed(composite, scene)
        return applied

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if self._state != _State.DRAGGING or self._armed_face_id is None:
            return False
        distance = parse_length(text, units)
        if distance is None or abs(distance) < _MIN_COMMIT_DISTANCE:
            return False
        self._commit_offset(float(distance))
        self._reset_to_idle()
        return True

    def _reset_to_idle(self) -> None:
        self._state = _State.IDLE
        self._hovered_face_id = None
        self._armed_face_id = None
        self._armed_face_loop = []
        self._armed_face_normal = None
        self._armed_face_points = None
        self._current_distance = 0.0


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """An orthonormal (e1, e2) basis spanning the plane perpendicular to `normal`."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - normal * np.dot(helper, normal)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    return e1, e2


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _point_in_polygon_2d(p: np.ndarray, poly: np.ndarray) -> bool:
    """Standard even-odd ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]):
            x_at_p = (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1
            if p[0] < x_at_p:
                inside = not inside
    return inside


def _signed_distance_to_boundary(point: np.ndarray, loop: np.ndarray, normal: np.ndarray) -> float:
    """Distance from `point` (on the polygon's plane) to the nearest boundary
    edge of `loop`, positive when `point` is inside the polygon, negative
    when outside -- matching offset_polygon's inward-positive convention."""
    e1, e2 = _plane_basis(normal)
    origin = loop[0]

    def project(v: np.ndarray) -> np.ndarray:
        rel = v - origin
        return np.array([float(np.dot(rel, e1)), float(np.dot(rel, e2))])

    p2 = project(point)
    poly2 = np.array([project(v) for v in loop])

    n = len(poly2)
    min_dist = min(_point_segment_distance(p2, poly2[i], poly2[(i + 1) % n]) for i in range(n))
    return min_dist if _point_in_polygon_2d(p2, poly2) else -min_dist
