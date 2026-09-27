"""The Scale tool (S) - bounding-box gizmo.

On activate the selection's AABB + grips are computed. Press picks the nearest
grip (screen space); its opposite is the anchor (Ctrl -> AABB centre). Drag
computes per-axis factors (corner = uniform along the diagonal; edge/face =
per-axis; Shift = uniform across driven axes). Zero-extent driven axes stay 1.
Release commits one TransformVerticesCommand. Preview is overlay-only.

Instance-mode (M4e §7.3): if ctx.selection.instances is non-empty, the gizmo
AABB is derived from the union of each selected instance's world bounding box
(local_aabb transformed by active_world_transform @ inst.transform). Commit
emits TransformInstanceCommand(s) via mat_scale(anchor, factor_vec).
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.scene_commands import TransformVerticesCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.geometry.transforms import scale as scale_pts
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.tools.transform_support import (
    GripSpec,
    grip_specs,
    selection_aabb,
    selection_vertices,
)
from bermake.viewport.picking import world_to_local_point

_GRIP_PX = 9.0
_GRIP_COLOR = (0.20, 0.75, 0.35)
_ACTIVE_COLOR = (1.0, 0.85, 0.10)
_BOX_COLOR = (0.20, 0.55, 0.95)
_EPS = 1e-3


def _closest_point_on_axis(p0, axis_dir, origin, direction):
    """Point on the line (p0 + t*axis_dir) closest to the ray (origin + s*direction)."""
    u = np.asarray(axis_dir, np.float64)
    d = np.asarray(direction, np.float64)
    w0 = np.asarray(p0, np.float64) - np.asarray(origin, np.float64)
    a, b, c = float(u @ u), float(u @ d), float(d @ d)
    dd, e = float(u @ w0), float(d @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-12:  # ray parallel to the axis
        return np.asarray(p0, np.float64)
    t = (b * e - c * dd) / denom
    return np.asarray(p0, np.float64) + u * t


class ScaleTool(Tool):
    @property
    def name(self) -> str:
        return "Scale"

    @property
    def shortcut(self) -> str:
        return "S"

    @property
    def id(self) -> str:
        return "scale"

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._stack = None
        self._selection = None
        self._camera = None
        self._size_provider = None
        self._units_provider = None
        self._lo = None
        self._hi = None
        self._grips: list[GripSpec] = []
        self._vertex_ids: list[int] = []
        self._orig: dict[int, np.ndarray] = {}
        self._active: GripSpec | None = None
        self._anchor = np.zeros(3, np.float32)
        self._factor_vec = np.ones(3, np.float32)
        # instance-mode state
        self._instance_mode = False
        self._instances: list = []
        # True when grips are expressed in LOCAL scene coords (entity-mode);
        # False when they are already in WORLD coords (instance-mode).
        self._grips_are_local = True

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _lift_local_to_world(self, points: np.ndarray) -> np.ndarray:
        """Lift LOCAL points to WORLD coordinates via the active world transform.

        None/identity-safe: `_world_transform()` returns None if `_model` is
        unset, and `is_identity_transform` treats None the same as the
        identity -- so at the root context (or with no model at all) points
        come back unchanged rather than raising. Extracted (#60) from the
        three call sites that duplicated this pattern: `_grip_world_pos`,
        the box-segment lift in `overlay()`, and the anchor lift in
        `_cursor_world`.
        """
        pts = np.asarray(points, np.float64).reshape(-1, 3)
        wt = self._world_transform()
        if is_identity_transform(wt):
            return pts
        return apply_mat(pts, wt)

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._model = ctx.model
        self._stack = ctx.command_stack
        self._selection = ctx.selection
        self._camera = ctx.camera
        self._size_provider = ctx.widget_size_provider
        self._units_provider = ctx.units_provider
        self._rebuild_box()

    def deactivate(self) -> None:
        self._reset_drag()
        self._lo = self._hi = None
        self._grips = []
        self._instance_mode = False
        self._instances = []

    def _rebuild_box(self) -> None:
        self._reset_drag()
        if (
            self._selection is not None
            and not self._selection.is_empty()
            and self._selection.instances
        ):
            # Instance mode: compute AABB from world bboxes of selected instances.
            # Grips end up in WORLD space.
            self._instance_mode = True
            self._grips_are_local = False
            self._instances = self._resolve_instances()
            box = self._instance_world_aabb()
        else:
            self._instance_mode = False
            self._grips_are_local = True
            self._instances = []
            self._vertex_ids = (
                selection_vertices(self._scene, self._selection)
                if self._selection is not None and not self._selection.is_empty()
                else []
            )
            box = selection_aabb(self._scene, self._vertex_ids)

        if box is None:
            self._lo = self._hi = None
            self._grips = []
            return
        self._lo, self._hi = box
        self._grips = grip_specs(self._lo, self._hi)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._grips:
            return
        grip = self._pick_grip(event)
        if grip is None:
            return
        if not self._instance_mode:
            self._orig = {v: self._scene.vertex(v).position.copy() for v in self._vertex_ids}
        self._active = grip
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._anchor = (
            ((self._lo + self._hi) * 0.5).astype(np.float32) if ctrl else grip.opposite.copy()
        )
        self._factor_vec = np.ones(3, np.float32)

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if self._active is None:
            return
        cursor = self._cursor_world(event)
        if cursor is None:
            return
        uniform = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        extent = (self._hi - self._lo).astype(np.float32)
        self._factor_vec = self._factor_vec_for(self._active, self._anchor, cursor, extent, uniform)

    def on_mouse_release(self, event: QMouseEvent, snap) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._active is None:
            return
        cursor = self._cursor_world(event)
        if cursor is not None:
            uniform = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            extent = (self._hi - self._lo).astype(np.float32)
            self._factor_vec = self._factor_vec_for(
                self._active, self._anchor, cursor, extent, uniform
            )
        if self._instance_mode:
            self._commit_instance_scale(self._anchor, self._factor_vec)
        else:
            ids = self._vertex_ids
            if ids:
                pts = np.array([self._orig[v] for v in ids], np.float32)
                new = scale_pts(pts, self._anchor, self._factor_vec)
                moves = {v: (self._orig[v], new[i]) for i, v in enumerate(ids)}
                cmd = TransformVerticesCommand(moves)
                if not cmd.is_empty() and self._stack is not None:
                    self._stack.execute(cmd, self._scene)
        self._reset_drag()
        self._rebuild_box()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset_drag()

    def apply_typed_value(self, text, units) -> bool:
        if self._active is None:
            return False
        t = text.strip()
        if "," in t:
            # Per-axis factors, e.g. "2,1,1" scales only X. Values map
            # positionally onto (X, Y, Z); a trailing axis left unspecified
            # keeps its default of 1 (no scale).
            parts = [p.strip() for p in t.split(",")]
            try:
                values = [float(p) for p in parts]
            except ValueError:
                return False
            if not values or any(v <= 0 for v in values):
                return False
            out = np.ones(3, np.float32)
            for i, v in enumerate(values[:3]):
                out[i] = v
        else:
            try:
                factor = float(t)
            except (ValueError, AttributeError):
                return False
            if factor <= 0:
                return False
            out = np.ones(3, np.float32)
            extent = (self._hi - self._lo).astype(np.float32)
            for ax in self._active.axes:
                if abs(float(extent[ax])) > 1e-9:
                    out[ax] = factor

        if self._instance_mode:
            self._commit_instance_scale(self._anchor, out)
        else:
            ids = self._vertex_ids
            pts = np.array([self._orig[v] for v in ids], np.float32)
            new = scale_pts(pts, self._anchor, out)
            moves = {v: (self._orig[v], new[i]) for i, v in enumerate(ids)}
            from bermake.commands.scene_commands import TransformVerticesCommand

            cmd = TransformVerticesCommand(moves)
            if not cmd.is_empty() and self._stack is not None:
                self._stack.execute(cmd, self._scene)

        self._reset_drag()
        self._rebuild_box()
        # _reset_drag() (called again inside _rebuild_box()) zeroes
        # _factor_vec back to ones; restore the applied value afterward so
        # callers/tests can read back what was actually typed, mirroring how
        # PolygonTool's _sides survives _reset_gesture().
        self._factor_vec = out
        return True

    def overlay(self) -> ToolOverlay:
        polylines: list = []
        markers: list = []
        if self._lo is not None and self._hi is not None:
            preview_lo, preview_hi = self._lo, self._hi
            if self._active is not None:
                corners = scale_pts(
                    np.array([self._lo, self._hi], np.float32), self._anchor, self._factor_vec
                )
                preview_lo = np.minimum(corners[0], corners[1]).astype(np.float32)
                preview_hi = np.maximum(corners[0], corners[1]).astype(np.float32)
            # Box segments: in entity-mode lo/hi are LOCAL; lift to world if needed.
            box_segs_local = self._box_segments(preview_lo, preview_hi)
            if self._grips_are_local:
                box_segs = self._lift_local_to_world(box_segs_local.reshape(-1, 3)).reshape(-1, 3)
            else:
                box_segs = box_segs_local
            polylines.append((box_segs, _BOX_COLOR, 1.5))
            for g in self._grips:
                is_active = self._active is not None and np.allclose(
                    g.position, self._active.position
                )
                color = _ACTIVE_COLOR if is_active else _GRIP_COLOR
                # Markers must be in WORLD space; lift local grips to world.
                world_pos = self._grip_world_pos(g)
                markers.append((world_pos, _GRIP_PX, color))
        return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), np.float32),
            rubber_band_color=(1, 1, 1),
            snap_marker_position=None,
            snap_marker_color=(1, 1, 1),
            world_polylines=polylines,
            screen_markers=markers,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._active is not None

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None

    def _factor_text(self) -> str | None:
        """The live per-axis scale factors, formatted, or None with no
        active grip. Reused by both status_text (prefixed prose) and
        measurement_text (the bare value -- factors are dimensionless
        ratios, not lengths, so format_length does not apply here)."""
        if self._active is None:
            return None
        f = self._factor_vec
        return f"({f[0]:.2f}, {f[1]:.2f}, {f[2]:.2f})"

    @property
    def status_text(self) -> str | None:
        if self._selection is None or self._selection.is_empty():
            return "Select geometry first"
        factor_text = self._factor_text()
        if factor_text is not None:
            return f"Scale {factor_text}"
        return "Scale: drag a handle"

    @property
    def measurement_text(self) -> str | None:
        return self._factor_text()

    # ---- factor math (pure; unit-tested) ----
    def _factors(self, grip, anchor, cursor, extent, uniform):
        """Test-facing alias; delegates to _factor_vec_for."""
        return self._factor_vec_for(grip, anchor, cursor, extent, uniform)

    def _factor_vec_for(self, grip: GripSpec, anchor, cursor, extent, uniform: bool) -> np.ndarray:
        anchor = np.asarray(anchor, np.float32)
        cursor = np.asarray(cursor, np.float32)
        extent = np.asarray(extent, np.float32)
        out = np.ones(3, np.float32)
        driven = [ax for ax in grip.axes if abs(float(extent[ax])) > 1e-9]
        if len(grip.axes) == 3 and len(driven) >= 2:
            diag = grip.position - anchor
            dlen = float(np.linalg.norm(diag))
            if dlen > 1e-9:
                proj = float(np.dot(cursor - anchor, diag)) / dlen
                fac = max(proj / dlen, _EPS)
                for ax in driven:
                    out[ax] = fac
            return out
        for ax in driven:
            denom = float(grip.position[ax] - anchor[ax])
            if abs(denom) < 1e-9:
                continue
            out[ax] = max((float(cursor[ax]) - float(anchor[ax])) / denom, _EPS)
        if uniform and driven:
            f = float(np.max([out[ax] for ax in driven]))
            for ax in driven:
                out[ax] = f
        return out

    # ---- instance helpers ----

    def _resolve_instances(self) -> list:
        """Resolve selected instance ids to Instance objects from the active context."""
        if self._model is None or self._selection is None:
            return []
        inst_ids = self._selection.instances
        return [inst for inst in self._model.active_context.children if inst.id in inst_ids]

    def _instance_world_aabb(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Union of world bounding boxes for all selected instances."""
        if not self._instances or self._model is None:
            return None
        world0 = self._model.active_world_transform
        all_pts: list[np.ndarray] = []
        for inst in self._instances:
            local_box = inst.definition.local_aabb()
            if local_box is None:
                continue
            lo, hi = local_box
            # 8 corners of the local AABB
            corners = np.array(
                [
                    [lo[0], lo[1], lo[2]],
                    [hi[0], lo[1], lo[2]],
                    [hi[0], hi[1], lo[2]],
                    [lo[0], hi[1], lo[2]],
                    [lo[0], lo[1], hi[2]],
                    [hi[0], lo[1], hi[2]],
                    [hi[0], hi[1], hi[2]],
                    [lo[0], hi[1], hi[2]],
                ],
                dtype=np.float64,
            )
            world_mat = world0 @ inst.transform
            # transform corners: (8,4) @ (4,4).T -> (8,4)
            hom = np.hstack([corners, np.ones((8, 1))])
            world_corners = (hom @ world_mat.T)[:, :3].astype(np.float32)
            all_pts.append(world_corners)
        if not all_pts:
            return None
        pts = np.vstack(all_pts)
        return pts.min(axis=0).astype(np.float32), pts.max(axis=0).astype(np.float32)

    def _commit_instance_scale(self, anchor: np.ndarray, factor_vec: np.ndarray) -> None:
        """Emit TransformInstanceCommand(s) for the scale gesture."""
        from bermake.commands.command import CompositeCommand
        from bermake.commands.instance_commands import TransformInstanceCommand
        from bermake.geometry.transforms import mat_scale

        delta_mat = mat_scale(anchor, factor_vec)
        cmds = [
            TransformInstanceCommand(inst, delta_mat @ inst.transform) for inst in self._instances
        ]
        if not cmds:
            return
        if len(cmds) == 1:
            cmd = cmds[0]
        else:
            cmd = CompositeCommand(name="Scale Instances", children=cmds)
        if self._stack is not None and self._model is not None:
            self._stack.execute(cmd, self._model)

    # ---- screen picking (camera-dependent; stubbed in unit tests) ----

    def _grip_world_pos(self, g: GripSpec) -> np.ndarray:
        """Return the world position of a grip.

        In entity-mode grips are LOCAL; lift them to world via the active
        world transform.  In instance-mode they are already world.
        When the world transform is identity (root context) this is a no-op.
        """
        if not self._grips_are_local:
            return g.position
        return self._lift_local_to_world(np.asarray(g.position, np.float64).reshape(1, 3))[0]

    def _pick_grip(self, event) -> GripSpec | None:
        if self._camera is None or self._size_provider is None:
            return None
        w, h = self._size_provider()
        pos = event.position()
        best, best_d = None, _GRIP_PX * 1.6
        for g in self._grips:
            world_pos = self._grip_world_pos(g)
            proj = self._camera.world_to_screen(world_pos, w, h)
            if proj is None:
                continue
            sx, sy, _d = proj
            dist = ((sx - pos.x()) ** 2 + (sy - pos.y()) ** 2) ** 0.5
            if dist < best_d:
                best, best_d = g, dist
        return best

    def _cursor_world(self, event):
        """Cursor ray intersected with a camera-facing plane through the grip,
        or (for single-axis grips) projected onto the grip's axis line.

        Falls back to None if no camera.

        In entity-mode the grips are LOCAL.  We use the grip's WORLD position as
        the plane/axis anchor (p0) so the projection is stable in screen space,
        then convert the resulting world-space cursor back to LOCAL so the factor
        math (which compares cursor to LOCAL grips/anchor) operates in the right
        frame. In instance-mode grips are already world; the cursor stays world.
        """
        if self._camera is None or self._size_provider is None:
            return None
        w, h = self._size_provider()
        pos = event.position()
        origin, direction = self._camera.ray_from_screen(pos.x(), pos.y(), w, h)
        # Plane/axis anchor: world position of the active grip (or anchor).
        if self._active is not None:
            p0 = self._grip_world_pos(self._active)
        else:
            # self._anchor is always in the grips' own frame; lift if local.
            if self._grips_are_local:
                p0 = self._lift_local_to_world(np.asarray(self._anchor, np.float64).reshape(1, 3))[
                    0
                ]
            else:
                p0 = np.asarray(self._anchor, np.float32)

        if self._active is not None and len(self._active.axes) == 1:
            # Single-axis grip (a face grip; edge/corner grips drive >1 axis
            # and fall to the plane branch below): drag along the axis line
            # itself rather than an arbitrary plane, so the grip tracks the
            # cursor exactly instead of sliding off-axis.
            ax = self._active.axes[0]
            axis_dir = np.zeros(3, np.float64)
            axis_dir[ax] = 1.0
            if self._grips_are_local:
                wt = self._world_transform()
                if not is_identity_transform(wt):
                    # Rotate the local basis vector into world space using the
                    # world transform's linear part (its ax-th column), the
                    # same pattern picking.ray_into_local uses for directions.
                    axis_dir = np.asarray(wt, np.float64)[:3, ax]
            cursor_world = _closest_point_on_axis(p0, axis_dir, origin, direction).astype(
                np.float32
            )
        else:
            # Corner/uniform or edge grips (and the no-active-grip case): a
            # plane through p0 facing the camera is correct in every
            # orientation and can never be edge-on to the cursor ray.
            view_dir = np.asarray(self._camera.position, np.float64) - np.asarray(
                self._camera.target, np.float64
            )
            view_norm = float(np.linalg.norm(view_dir))
            if view_norm < 1e-9:
                return None
            n = (view_dir / view_norm).astype(np.float32)
            denom = float(np.dot(direction, n))
            if abs(denom) < 1e-9:
                return None
            t = float(np.dot(p0 - origin, n)) / denom
            if t <= 0:
                return None
            cursor_world = (origin + t * direction).astype(np.float32)

        # In entity-mode convert the world cursor back to LOCAL so the factor
        # math (comparing to LOCAL grips/anchor) remains correct.
        if self._grips_are_local:
            wt = self._world_transform()
            return world_to_local_point(cursor_world, wt).astype(np.float32)
        return cursor_world

    def _box_segments(self, lo, hi) -> np.ndarray:
        lo = np.asarray(lo, np.float32)
        hi = np.asarray(hi, np.float32)
        c = [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        segs = np.empty((2 * len(edges), 3), np.float32)
        for i, (u, v) in enumerate(edges):
            segs[2 * i] = c[u]
            segs[2 * i + 1] = c[v]
        return segs

    def _reset_drag(self) -> None:
        self._active = None
        self._anchor = np.zeros(3, np.float32)
        self._factor_vec = np.ones(3, np.float32)
        self._orig = {}
