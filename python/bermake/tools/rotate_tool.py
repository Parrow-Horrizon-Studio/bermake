"""The Rotate tool (Q) - auto-tilt protractor.

Three clicks: center -> start direction -> angle. The protractor plane is the
plane of the face under the cursor when the center is placed (else the ground
plane). Up/Down arrows cycle a forced axis (X/Y/Z/auto) overriding the inferred
normal. The swept angle snaps to 15 degrees. One TransformVerticesCommand on commit.

Instance-mode (M4e §7.3): if ctx.selection.instances is non-empty when the center
is placed, the tool composes the rotate delta into each instance's 4x4 transform and
emits TransformInstanceCommand(s) instead of TransformVerticesCommand.

Transform-awareness (M4e fix): snap/pick quantities arrive in WORLD space.
When editing INSIDE a moved group/component the active context has a non-identity
world transform.  The face-pick ray, rotation center, and rotation axis must all be
converted from world to the LOCAL frame before operating on local mesh vertices.
The OVERLAY (protractor) always renders in world space — it keeps the world-space
center and normal and must NOT use the locally-converted values.
At the root context (identity) every conversion is a no-op.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.scene_commands import TransformVerticesCommand
from bermake.geometry.transforms import is_identity_transform, mat_invert, rotate
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.tools.transform_support import selection_vertices
from bermake.viewport.picking import ray_into_local, world_to_local_point

_ANGLE_SNAP_RAD = math.radians(15.0)
_DISK_COLOR_RGBA = (0.30, 0.55, 0.95, 0.18)
_DISK_OUTLINE = (0.30, 0.55, 0.95)
_RAY_COLOR = (0.95, 0.80, 0.20)
_AXES = (
    np.array([1, 0, 0], np.float32),
    np.array([0, 1, 0], np.float32),
    np.array([0, 0, 1], np.float32),
)


class _Stage(Enum):
    IDLE = 0
    HAVE_CENTER = 1
    HAVE_START = 2


class RotateTool(Tool):
    @property
    def name(self) -> str:
        return "Rotate"

    @property
    def shortcut(self) -> str:
        return "Q"

    @property
    def id(self) -> str:
        return "rotate"

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._stack = None
        self._selection = None
        self._camera = None
        self._size_provider = None
        self._units_provider = None
        self._stage = _Stage.IDLE
        self._center = np.zeros(3, np.float32)
        self._normal = np.array([0, 0, 1], np.float32)
        self._inferred_normal = np.array([0, 0, 1], np.float32)
        self._start_dir = np.array([1, 0, 0], np.float32)
        self._cur_dir = np.array([1, 0, 0], np.float32)
        self._forced_axis: int | None = None
        self._vertex_ids: list[int] = []
        self._orig: dict[int, np.ndarray] = {}
        # instance-mode state
        self._instance_mode = False
        self._instances: list = []
        # Last signed angle (degrees) resolved via apply_typed_value. NOT reset
        # by _reset() -- it's a read-back for callers/tests, like Polygon's
        # persisted _sides, not part of the in-gesture state machine.
        self._angle_degrees = 0.0

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _world_vec_to_local(self, world_vec: np.ndarray) -> np.ndarray:
        """Convert a world-space DIRECTION/VECTOR into the active context's local frame.

        Only the inverse 3x3 block is applied (translation does not affect vectors).
        Returns the vector unchanged at identity/None (root context).
        """
        wt = self._world_transform()
        if is_identity_transform(wt):
            return world_vec.astype(np.float32)
        inv3 = mat_invert(np.asarray(wt, np.float64))[:3, :3]
        return (inv3 @ np.asarray(world_vec, np.float64)).astype(np.float32)

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._model = ctx.model
        self._stack = ctx.command_stack
        self._selection = ctx.selection
        self._camera = ctx.camera
        self._size_provider = ctx.widget_size_provider
        self._units_provider = ctx.units_provider
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._selection is None or self._selection.is_empty() or snap.kind == SnapKind.NONE:
            return
        p = np.asarray(snap.world_position, np.float32)

        if self._stage == _Stage.IDLE:
            # Determine mode at gesture start (center placement)
            self._instance_mode = bool(self._selection.instances)
            if self._instance_mode:
                self._instances = self._resolve_instances()
                if not self._instances:
                    self._instance_mode = False
                    # fall through to entity mode
            if not self._instance_mode:
                self._vertex_ids = selection_vertices(self._scene, self._selection)
                if not self._vertex_ids:
                    return
                self._orig = {v: self._scene.vertex(v).position.copy() for v in self._vertex_ids}
            self._center = p.copy()
            self._inferred_normal = np.asarray(self._pick_plane_normal(event), np.float32)
            self._normal = self._effective_normal(self._inferred_normal)
            self._stage = _Stage.HAVE_CENTER
            return

        if self._stage == _Stage.HAVE_CENTER:
            d = self._project_to_plane(p - self._center)
            if float(np.linalg.norm(d)) < 1e-6:
                return
            self._start_dir = d / np.linalg.norm(d)
            self._cur_dir = self._start_dir.copy()
            self._stage = _Stage.HAVE_START
            return

        angle = self._swept_angle(p)
        if self._instance_mode:
            self._commit_instance_rotate(angle)
        else:
            moves = self._compute_moves(angle)
            cmd = TransformVerticesCommand(moves)
            if not cmd.is_empty() and self._stack is not None:
                self._stack.execute(cmd, self._scene)
        self._reset()

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if self._stage != _Stage.HAVE_START or snap.kind == SnapKind.NONE:
            return
        d = self._project_to_plane(np.asarray(snap.world_position, np.float32) - self._center)
        if float(np.linalg.norm(d)) >= 1e-6:
            self._cur_dir = d / np.linalg.norm(d)

    def on_key_press(self, event: QKeyEvent) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._reset()
            return
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            order = [None, 0, 1, 2]
            cur = order.index(self._forced_axis)
            self._forced_axis = order[(cur + 1) % len(order)]
            if self._stage != _Stage.IDLE:
                self._normal = self._effective_normal(self._inferred_normal)

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_angle

        if self._stage != _Stage.HAVE_START:
            return False
        deg = parse_angle(text)
        if deg is None:
            return False
        # The typed magnitude follows the sweep direction, and an explicit
        # "-" flips it. Because this same signed formula applies to every
        # input, -deg is always the opposite rotation of +deg under the same
        # sweep, for either sweep sign (#55).
        sign = 1.0 if self._swept_angle_from_cur() >= 0 else -1.0
        angle = sign * math.radians(deg)

        if self._instance_mode:
            self._commit_instance_rotate(angle)
            self._reset()
            self._angle_degrees = math.degrees(angle)
            return True

        moves = self._compute_moves(angle)
        from bermake.commands.scene_commands import TransformVerticesCommand

        cmd = TransformVerticesCommand(moves)
        if not cmd.is_empty() and self._stack is not None:
            self._stack.execute(cmd, self._scene)
        self._reset()
        self._angle_degrees = math.degrees(angle)
        return True

    def overlay(self) -> ToolOverlay:
        fills: list = []
        polylines: list = []
        if self._stage in (_Stage.HAVE_CENTER, _Stage.HAVE_START):
            r = self._disk_radius()
            disk = self._disk_loop(radius=r)
            fills.append(disk)
            polylines.append((self._loop_to_segments(disk), _DISK_OUTLINE, 1.5))
            if self._stage == _Stage.HAVE_START:
                start_seg = np.array([self._center, self._center + self._start_dir * r], np.float32)
                cur_seg = np.array([self._center, self._center + self._cur_dir * r], np.float32)
                polylines.append((start_seg, _DISK_OUTLINE, 1.5))
                polylines.append((cur_seg, _RAY_COLOR, 2.0))
        return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), np.float32),
            rubber_band_color=(1, 1, 1),
            snap_marker_position=None,
            snap_marker_color=(1, 1, 1),
            face_fill_polygons=fills,
            face_fill_color=_DISK_COLOR_RGBA,
            world_polylines=polylines,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._stage != _Stage.IDLE

    @property
    def consumes_arrow_keys(self) -> bool:
        """Up/Down cycle the forced rotation axis."""
        return True

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return self._center.copy() if self._stage != _Stage.IDLE else None

    @property
    def angle_degrees(self) -> float:
        """Signed angle (degrees) last resolved via apply_typed_value."""
        return self._angle_degrees

    @property
    def status_text(self) -> str | None:
        if self._selection is None or self._selection.is_empty():
            return "Select geometry first"
        if self._stage == _Stage.IDLE:
            return "Rotate: click the center"
        if self._stage == _Stage.HAVE_CENTER:
            return "Rotate: click the start direction"
        if self._units_provider is not None:
            from bermake.units import format_angle

            return f"Rotate: {format_angle(math.degrees(self._swept_angle_from_cur()))} (15° snap)"
        return f"Rotate: {math.degrees(self._swept_angle_from_cur()):.0f} deg (15 deg snap)"

    @property
    def measurement_text(self) -> str | None:
        """The live swept angle in ASCII degrees, or None before HAVE_START.

        Matches the status_text fallback's "12 deg" shape (rotate_tool.py's
        own convention) rather than format_angle's degree glyph -- angles in
        the numeric readout are always plain ASCII."""
        if self._stage != _Stage.HAVE_START:
            return None
        return f"{math.degrees(self._swept_angle_from_cur()):.0f} deg"

    # ---- internal ----

    def _resolve_instances(self) -> list:
        """Resolve selected instance ids to Instance objects from the active context."""
        if self._model is None or self._selection is None:
            return []
        inst_ids = self._selection.instances
        return [inst for inst in self._model.active_context.children if inst.id in inst_ids]

    def _commit_instance_rotate(self, angle: float) -> None:
        """Emit TransformInstanceCommand(s) for the rotate gesture.

        self._center and self._normal are in WORLD space.  Instance transforms live
        in the ACTIVE CONTEXT's LOCAL frame, so convert center and normal before
        building the matrix.  At identity (root context) this is a no-op.
        """
        from bermake.commands.command import CompositeCommand
        from bermake.commands.instance_commands import TransformInstanceCommand
        from bermake.geometry.transforms import mat_rotate
        from bermake.viewport.picking import world_to_local_point

        local_center = world_to_local_point(self._center, self._world_transform())
        local_normal = self._world_vec_to_local(self._normal)
        ln = float(np.linalg.norm(local_normal))
        if ln > 1e-9:
            local_normal = local_normal / ln
        delta_mat = mat_rotate(local_center, local_normal, angle)
        cmds = [
            TransformInstanceCommand(inst, delta_mat @ inst.transform) for inst in self._instances
        ]
        if not cmds:
            return
        if len(cmds) == 1:
            cmd = cmds[0]
        else:
            cmd = CompositeCommand(name="Rotate Instances", children=cmds)
        if self._stack is not None and self._model is not None:
            self._stack.execute(cmd, self._model)

    def _pick_plane_normal(self, event) -> np.ndarray:
        """Normal of the face under the cursor (in WORLD space), or +Z if none/no camera.

        The ray is converted to the active context's LOCAL frame before the face
        pick (mirrors push_pull_tool.py).  The returned normal is always in WORLD
        space so the caller can use it directly for the overlay protractor.
        """
        if self._camera is None or self._size_provider is None or self._scene is None:
            return np.array([0, 0, 1], np.float32)
        w, h = self._size_provider()
        pos = event.position()
        origin, direction = self._camera.ray_from_screen(pos.x(), pos.y(), w, h)
        # Convert ray to local frame for the face-pick (scene mesh is in local coords).
        local_origin, local_direction = ray_into_local(origin, direction, self._world_transform())
        hit = self._scene.ray_pick_face(local_origin, local_direction)
        if hit is None:
            return np.array([0, 0, 1], np.float32)
        try:
            # face_normal returns a vector in LOCAL space; lift it to world.
            local_n = np.asarray(self._scene.face_normal(hit.face_id), np.float32)
            wt = self._world_transform()
            if is_identity_transform(wt):
                return local_n
            # Normals transform by the inverse-transpose of the 3x3 block.
            inv3_t = mat_invert(np.asarray(wt, np.float64))[:3, :3].T
            world_n = (inv3_t @ local_n.astype(np.float64)).astype(np.float32)
            ln = float(np.linalg.norm(world_n))
            return (world_n / ln) if ln > 1e-9 else np.array([0, 0, 1], np.float32)
        except (KeyError, ValueError):
            return np.array([0, 0, 1], np.float32)

    def _effective_normal(self, inferred: np.ndarray) -> np.ndarray:
        if self._forced_axis is not None:
            return _AXES[self._forced_axis].copy()
        n = np.asarray(inferred, np.float32)
        ln = float(np.linalg.norm(n))
        return (n / ln).astype(np.float32) if ln > 1e-9 else np.array([0, 0, 1], np.float32)

    def _project_to_plane(self, v: np.ndarray) -> np.ndarray:
        n = self._normal
        return (v - n * float(np.dot(v, n))).astype(np.float32)

    def _swept_angle(self, world_point: np.ndarray) -> float:
        d = self._project_to_plane(np.asarray(world_point, np.float32) - self._center)
        if float(np.linalg.norm(d)) < 1e-9:
            return 0.0
        self._cur_dir = d / np.linalg.norm(d)
        return self._swept_angle_from_cur()

    def _swept_angle_from_cur(self) -> float:
        s, c, n = self._start_dir, self._cur_dir, self._normal
        ang = math.atan2(float(np.dot(np.cross(s, c), n)), float(np.dot(s, c)))
        return round(ang / _ANGLE_SNAP_RAD) * _ANGLE_SNAP_RAD

    def _compute_moves(self, angle: float) -> dict:
        """Compute new vertex positions for a rotation by `angle`.

        self._center and self._normal are in WORLD space (used by the overlay).
        The local mesh vertices need them converted to the active context's LOCAL
        frame before the rotation math.  At identity this is a no-op.
        """
        ids = self._vertex_ids
        pts = np.array([self._orig[v] for v in ids], np.float32)
        # Convert world center → local point.
        local_center = world_to_local_point(self._center, self._world_transform())
        # Convert world normal → local vector (inverse 3x3, then renormalize).
        local_normal = self._world_vec_to_local(self._normal)
        ln = float(np.linalg.norm(local_normal))
        if ln > 1e-9:
            local_normal = local_normal / ln
        new = rotate(pts, local_center, local_normal, angle)
        return {v: (self._orig[v], new[i]) for i, v in enumerate(ids)}

    def _disk_radius(self) -> float:
        if self._instance_mode:
            # Use a unit radius for instance mode (no orig vertices)
            return 1.0
        if not self._orig:
            return 1.0
        pts = np.array(list(self._orig.values()), np.float32)
        # self._center is WORLD; pts are LOCAL → convert center to local before
        # computing distances so the protractor radius is correct inside a
        # translated/rotated group.
        from bermake.viewport.picking import world_to_local_point

        local_center = world_to_local_point(self._center, self._world_transform())
        r = float(np.max(np.linalg.norm(pts - local_center, axis=1)))
        return max(r, 0.5)

    def _disk_loop(self, radius: float, segments: int = 48) -> np.ndarray:
        n = self._normal
        if abs(n[0]) < 0.9:
            ref = np.array([1, 0, 0], np.float32)
        else:
            ref = np.array([0, 1, 0], np.float32)
        u = np.cross(n, ref)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        loop = np.empty((segments, 3), np.float32)
        for i in range(segments):
            t = 2 * math.pi * i / segments
            loop[i] = self._center + radius * (math.cos(t) * u + math.sin(t) * v)
        return loop

    @staticmethod
    def _loop_to_segments(loop: np.ndarray) -> np.ndarray:
        n = loop.shape[0]
        segs = np.empty((2 * n, 3), np.float32)
        for i in range(n):
            segs[2 * i] = loop[i]
            segs[2 * i + 1] = loop[(i + 1) % n]
        return segs

    def _reset(self) -> None:
        self._stage = _Stage.IDLE
        self._center = np.zeros(3, np.float32)
        self._normal = np.array([0, 0, 1], np.float32)
        self._inferred_normal = np.array([0, 0, 1], np.float32)
        self._start_dir = np.array([1, 0, 0], np.float32)
        self._cur_dir = np.array([1, 0, 0], np.float32)
        self._forced_axis = None
        self._vertex_ids = []
        self._orig = {}
        self._instance_mode = False
        self._instances = []
