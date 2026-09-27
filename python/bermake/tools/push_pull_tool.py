"""The Push/Pull tool — SketchUp-style face extrusion.

Three-state machine:
    IDLE     — no face under cursor; no overlay.
    HOVERING — cursor over a live face; that face highlighted (light blue).
    DRAGGING — face armed by click; cursor moves drive depth via line-line CPA;
               ghost prism rendered.

Click in HOVERING arms the face. Click in DRAGGING commits (or cancels if
depth < 1e-3). ESC in DRAGGING cancels. ESC in HOVERING / IDLE is handled
by MainWindow's two-stage ESC (it deactivates the tool).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.scene_commands import RemoveFaceCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.tools.sweep_support import loft_between_loops, seam_merge
from bermake.tools.tool import Tool, ToolContext, ToolOverlay

# Visual constants (RGBA).
_HOVER_FILL_COLOR = (0.40, 0.70, 1.00, 0.20)  # light blue
_ARMED_FILL_COLOR = (0.20, 0.50, 0.95, 0.40)  # darker blue
_GHOST_FILL_COLOR = (0.40, 0.70, 1.00, 0.15)  # light blue, fainter

_MIN_COMMIT_DEPTH = 1e-3  # world units; below this is treated as cancel
_DEGENERATE_VIEW_EPSILON = 1e-4  # |1 - (d·n)²| below this freezes depth


class _State(Enum):
    IDLE = 0
    HOVERING = 1
    DRAGGING = 2


class PushPullTool(Tool):
    """SketchUp-style face extrusion tool."""

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
        self._armed_face_center: np.ndarray | None = None
        self._current_depth: float = 0.0

    # ---- Tool ABC ------------------------------------------------------

    @property
    def name(self) -> str:
        return "Push/Pull"

    @property
    def shortcut(self) -> str:
        return "P"

    @property
    def id(self) -> str:
        return "push_pull"

    @property
    def has_active_gesture(self) -> bool:
        return self._state == _State.DRAGGING

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None  # Push/Pull doesn't drive axis-lock.

    def _depth_text(self) -> str | None:
        """The live extrusion depth, formatted, or None while not dragging.

        Reused by both status_text (prefixed prose) and measurement_text
        (the bare value)."""
        if self._state != _State.DRAGGING:
            return None
        if self._units_provider is not None:
            from bermake.units import format_length

            return format_length(self._current_depth, self._units_provider())
        return f"{self._current_depth:.3f}"

    @property
    def status_text(self) -> str | None:
        depth_text = self._depth_text()
        if depth_text is None:
            return None
        return f"depth: {depth_text}"

    @property
    def measurement_text(self) -> str | None:
        return self._depth_text()

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
            self._update_depth_from_event(event)
            return
        # IDLE / HOVERING — per-frame ray-pick.
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
        # DRAGGING: commit if depth >= min threshold, else cancel.
        if self._current_depth >= _MIN_COMMIT_DEPTH:
            self._commit_extrusion()
        # Task 11 fills near-zero cancel + post-commit re-hover.
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
            # Cancel — no command pushed, scene was never mutated during DRAGGING
            # (M3b uses overlay-only preview, not scene mutation).
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
        # At root the world transform is identity → this is a no-op.
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
        self._armed_face_loop = self._scene.face_loop(face_id)
        self._armed_face_normal = self._scene.face_normal(face_id)
        self._armed_face_center = self._scene.face_center(face_id)
        self._current_depth = 0.0
        self._state = _State.DRAGGING

    def _update_depth_from_event(self, event: QMouseEvent) -> None:
        """Update self._current_depth via line-line CPA between the camera ray
        and the line (face_center, +normal). Holds the previous depth if the
        view is ~parallel to the normal (degenerate case)."""
        if self._camera is None or self._widget_size_provider is None:
            return
        assert self._armed_face_normal is not None
        assert self._armed_face_center is not None

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
        n_hat = self._armed_face_normal  # already unit
        c = self._armed_face_center

        b = float(np.dot(d_hat, n_hat))
        denom = 1.0 - b * b
        if abs(denom) < _DEGENERATE_VIEW_EPSILON:
            return  # depth frozen
        w = origin.astype(np.float32) - c
        e = float(np.dot(n_hat, w))
        d_param = float(np.dot(d_hat, w))
        t = (e - b * d_param) / denom
        self._current_depth = max(0.0, t)

    def _build_ghost_polygons(self) -> list[np.ndarray]:
        """Return [armed_face_loop, ghost_top, *ghost_sides] all in world coords."""
        assert self._armed_face_id is not None
        assert self._armed_face_normal is not None
        assert self._armed_face_center is not None
        assert self._scene is not None

        source_loop_xyz = self._loop_world_coords(self._armed_face_id)  # (N, 3)
        n = self._armed_face_normal
        depth = self._current_depth
        top_loop_xyz = source_loop_xyz + depth * n[np.newaxis, :]

        polygons: list[np.ndarray] = [source_loop_xyz, top_loop_xyz]
        # Side polygons (one per source edge): (V_i, V_{i+1}, V'_{i+1}, V'_i).
        n_verts = source_loop_xyz.shape[0]
        for i in range(n_verts):
            j = (i + 1) % n_verts
            side = np.stack(
                [
                    source_loop_xyz[i],
                    source_loop_xyz[j],
                    top_loop_xyz[j],
                    top_loop_xyz[i],
                ]
            ).astype(np.float32)
            polygons.append(side)
        return polygons

    def _should_add_bottom_cap(self, src_face_id: int) -> bool:
        """True iff every boundary edge of the source face has only one
        incident half-edge — meaning the source was standalone (Case 1) and
        adding a bottom cap will close the prism manifold-correctly without
        creating a non-manifold edge."""
        assert self._scene is not None
        for e in self._scene.face_edges(src_face_id):
            if not self._scene.edge_is_boundary(e):
                return False
        return True

    def _commit_extrusion(self) -> None:
        """Build the extrusion CompositeCommand and push it to the command stack."""
        assert self._armed_face_id is not None
        assert self._armed_face_loop, "armed loop must be populated"
        assert self._armed_face_normal is not None
        assert self._scene is not None

        # Capture M3c "is this a standalone source?" check BEFORE we remove
        # the source face. Determines whether we add a bottom cap (Case 1) or
        # leave it open (Case 2 — to be handled by seam-merge in Task 8).
        is_standalone = self._should_add_bottom_cap(self._armed_face_id)

        # M3c: capture the OLD source face's boundary edge ids BEFORE removal,
        # so the seam-merge pass can re-visit them after the new side faces
        # have populated each edge's second half-edge slot.
        candidate_seam_edges = list(self._scene.face_edges(self._armed_face_id))

        scene = self._scene
        loop = self._armed_face_loop
        normal = self._armed_face_normal.astype(np.float32)
        depth = float(self._current_depth)

        composite = CompositeCommand(name="Push/Pull")

        # 1. Remove source face.
        rm = RemoveFaceCommand(self._armed_face_id)
        rm.do(scene)
        composite.children.append(rm)

        # 2-7. Top vertices, vertical + top boundary edges, side faces, top
        # face, and (Case 1 only) the reversed-source bottom cap — all in one
        # loft. is_standalone drives cap_start; the far end is always capped.
        dst = [scene.vertex(v).position + depth * normal for v in loop]
        result = loft_between_loops(scene, loop, dst, cap_start=is_standalone, cap_end=True)
        composite.children.extend(result.commands)

        # 8. (M3c) Seam-merge pass — dissolve OLD-source-boundary edges whose
        # two incident faces (parent's old side + new prism's side) are coplanar.
        # Single-pass over the OLD source face's boundary edges only, per design
        # doc §3.1 decision 6. (For Case 1 the candidate edges border the side +
        # bottom cap, which are NOT coplanar, so this no-ops correctly.)
        composite.children.extend(seam_merge(scene, candidate_seam_edges))

        if self._command_stack is not None:
            self._command_stack.push_executed(composite, self._scene)

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if self._state != _State.DRAGGING or self._armed_face_id is None:
            return False
        depth = parse_length(text, units)
        if depth is None or depth < _MIN_COMMIT_DEPTH:
            return False
        self._current_depth = float(depth)
        self._commit_extrusion()
        self._reset_to_idle()
        return True

    def _reset_to_idle(self) -> None:
        self._state = _State.IDLE
        self._hovered_face_id = None
        self._armed_face_id = None
        self._armed_face_loop = []
        self._armed_face_normal = None
        self._armed_face_center = None
        self._current_depth = 0.0
