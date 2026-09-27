"""The Paint tool (B).

Click a face to apply the active material to the side you clicked; drag to
paint every face the cursor crosses as one undo step. Alt-click samples
(eyedropper) the clicked side's material as the new active material.
Painting the Default material removes paint.

Shift-drag on a face instead shifts that face side's texture placement
(offset_u / offset_v) live, as a convenience over Task 10's numeric
Properties fields -- one drag is one undo step, matching the paint stroke
(M7.5b Task 12).

Side resolution follows model.pick_face_local (M7.4 #92): the face normal is
transformed by the inverse-transpose of the world transform's linear block,
not the linear block itself, so a non-uniformly scaled group still picks the
correct side.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.material_commands import PaintFaceCommand, SetFacePlacementCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert
from bermake.scene.scene import Side, TexturePlacement
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import pick_selectable
from bermake.viewport.uv_projection import plane_basis

_HOVER_ALPHA = 0.45
_NEUTRAL_COLOR = (0.85, 0.85, 0.85)
_PLACEMENT_DRAG_MODIFIER = Qt.KeyboardModifier.ShiftModifier


def side_for_ray(world_normal, ray_direction) -> Side:
    """Which side of a face a ray hits.

    A ray travelling against the normal strikes the front. `world_normal` must
    already be in world space, transformed by the inverse-transpose of the
    world matrix's linear block (see model.pick_face_local, fixed in M7.4 #92):
    the plain linear block gives the wrong side under a non-uniform scale.
    """
    return Side.FRONT if float(np.dot(world_normal, ray_direction)) < 0.0 else Side.BACK


class PaintTool(Tool):
    @property
    def name(self) -> str:
        return "Paint"

    @property
    def shortcut(self) -> str:
        return "B"

    @property
    def id(self) -> str:
        return "paint"

    def __init__(self) -> None:
        self._scene = None
        self._camera = None
        self._size_provider = None
        self._command_stack = None
        self._model = None
        self._active_material_provider = None
        self._set_active_material = None
        self._on_placement_committed = None
        self._hovered_face: int | None = None
        self._stroke_commands: list = []
        self._stroke_painted: set[tuple[int, Side]] = set()
        self._stroke_active: bool = False
        self._stroke_material_id: int | None = None
        self._init_placement_drag_state()

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._camera = ctx.camera
        self._size_provider = ctx.widget_size_provider
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._active_material_provider = ctx.active_material_provider
        self._set_active_material = ctx.set_active_material
        self._on_placement_committed = ctx.on_placement_committed
        self._hovered_face = None
        self._stroke_commands = []
        self._stroke_painted = set()
        self._stroke_active = False
        self._stroke_material_id = None
        self._init_placement_drag_state()

    def deactivate(self) -> None:
        # Roll back any in-progress (un-committed) paint stroke so switching
        # tools mid-drag never leaves un-undoable scene mutations, mirroring
        # EraserTool.deactivate.
        if self._stroke_active and self._stroke_commands and self._scene is not None:
            for cmd in reversed(self._stroke_commands):
                cmd.undo(self._scene)
        if self._drag_active and self._drag_start is not None and self._scene is not None:
            # Same reasoning: the live updates already wrote intermediate
            # placements straight into the scene, so switching tools
            # mid-drag must put the face side back the way it found it.
            self._scene.set_face_placement(self._drag_face_id, self._drag_start, self._drag_side)
        self._hovered_face = None
        self._stroke_commands = []
        self._stroke_painted = set()
        self._stroke_active = False
        self._stroke_material_id = None
        self._init_placement_drag_state()

    def _init_placement_drag_state(self) -> None:
        # Drag-scoped state, tested directly via begin/update/end_placement_drag.
        self._drag_active: bool = False
        self._drag_face_id: int | None = None
        self._drag_side: Side | None = None
        self._drag_start: TexturePlacement | None = None
        self._drag_current: TexturePlacement | None = None
        self._drag_du_acc: float = 0.0
        self._drag_dv_acc: float = 0.0
        # Pixel -> UV conversion cache, populated by the mouse wiring only
        # (begin_placement_drag itself needs none of this).
        self._drag_plane_point: np.ndarray | None = None
        self._drag_plane_normal: np.ndarray | None = None
        self._drag_u_axis: np.ndarray | None = None
        self._drag_v_axis: np.ndarray | None = None
        self._drag_local_from_world: np.ndarray | None = None
        self._drag_su: float = 1.0
        self._drag_sv: float = 1.0
        self._drag_scale: float = 1.0
        self._drag_prev_hit: np.ndarray | None = None

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _viewport_size(self) -> tuple[int, int]:
        return self._size_provider() if self._size_provider is not None else (1, 1)

    def _cursor(self, event: QMouseEvent) -> tuple[float, float]:
        pos = event.position()
        return (float(pos.x()), float(pos.y()))

    def _pick_face(self, event: QMouseEvent) -> int | None:
        hit = pick_selectable(
            self._cursor(event),
            self._viewport_size(),
            self._camera,
            self._scene,
            world_transform=self._world_transform(),
        )
        return hit[1] if hit is not None and hit[0] == "face" else None

    def _world_face_normal(self, f_id: int) -> np.ndarray:
        """Face `f_id`'s local normal, carried into world space.

        Mirrors model.pick_face_local (M7.4 #92): the inverse-transpose of
        the world transform's linear block, not the linear block itself, so
        a non-uniformly scaled group still resolves the correct side/plane.
        Shared by `_resolve_side` and the placement-drag plane test below --
        a separate tool would have had to duplicate this.
        """
        normal = np.asarray(self._scene.face_normal(f_id), dtype=np.float64)
        wt = self._world_transform()
        if not is_identity_transform(wt):
            w_inv = mat_invert(np.asarray(wt, dtype=np.float64))
            normal = w_inv[:3, :3].T @ normal
            length = float(np.linalg.norm(normal))
            if length > 1e-12:
                normal = normal / length
        return normal

    def _resolve_side(self, event: QMouseEvent, f_id: int) -> Side:
        """Which side of face `f_id` the cursor ray strikes, in world space."""
        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        _origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
        normal = self._world_face_normal(f_id)
        return side_for_ray(normal, np.asarray(direction, dtype=np.float64))

    def _active_material(self):
        if self._active_material_provider is None:
            return None
        return self._active_material_provider()

    def _begin_stroke(self, material_id: int) -> None:
        self._stroke_commands = []
        self._stroke_painted = set()
        self._stroke_active = True
        self._stroke_material_id = material_id

    def _paint_during_stroke(self, f_id: int, side: Side, material_id: int) -> None:
        if (f_id, side) in self._stroke_painted:
            return
        if self._scene.face_material(f_id, side) == material_id:
            self._stroke_painted.add((f_id, side))
            return
        cmd = PaintFaceCommand(f_id, material_id, side)
        cmd.do(self._scene)  # eager, so the paint appears live
        self._stroke_commands.append(cmd)
        self._stroke_painted.add((f_id, side))

    def _end_stroke(self) -> None:
        if self._stroke_commands and self._command_stack is not None:
            composite = CompositeCommand(name="Paint", children=self._stroke_commands)
            self._command_stack.push_executed(composite, self._scene)
        self._stroke_commands = []
        self._stroke_painted = set()
        self._stroke_active = False
        self._stroke_material_id = None

    # --- Placement drag (M7.5b Task 12) ------------------------------------
    #
    # Shift-drag over a face shifts that face side's offset_u/offset_v live,
    # as a convenience over Task 10's numeric Properties fields -- same
    # TexturePlacement, same commands.SetFacePlacementCommand, just a
    # different way to reach it. The three methods below are the tested
    # surface; on_mouse_press/_move/_release below just wire the modifier and
    # convert cursor motion to UV deltas.

    def begin_placement_drag(self, f_id: int, side: Side) -> None:
        """Start a placement drag: capture the face, side, and the placement
        the face started with.

        Every update is built from THIS start plus the accumulated delta
        (not from `TexturePlacement()`), or the texture would jump on the
        first mouse move whenever the face already had a placement.
        """
        self._drag_active = True
        self._drag_face_id = f_id
        self._drag_side = side
        self._drag_start = self._scene.face_placement(f_id, side)
        self._drag_current = self._drag_start
        self._drag_du_acc = 0.0
        self._drag_dv_acc = 0.0

    def update_placement_drag(self, du: float, dv: float) -> None:
        """Accumulate a UV delta and apply it to the scene directly, live."""
        if not self._drag_active or self._drag_start is None:
            return
        self._drag_du_acc += du
        self._drag_dv_acc += dv
        self._drag_current = replace(
            self._drag_start,
            offset_u=self._drag_start.offset_u + self._drag_du_acc,
            offset_v=self._drag_start.offset_v + self._drag_dv_acc,
        )
        self._scene.set_face_placement(self._drag_face_id, self._drag_current, self._drag_side)

    def end_placement_drag(self) -> None:
        """Push one SetFacePlacementCommand for the whole drag via
        push_executed -- CommandStack.execute would re-run do() and double
        apply, since update_placement_drag already mutated the scene.

        Pushes nothing if the drag never actually moved anything.
        """
        if (
            self._drag_active
            and self._drag_current != self._drag_start
            and self._command_stack is not None
        ):
            # The scene currently holds the final placement (already applied
            # live, above). Put the face side back to its start value so
            # SetFacePlacementCommand.do -- which only captures "before" on
            # its first call -- captures the true original instead of the
            # already-applied final value.
            self._scene.set_face_placement(self._drag_face_id, self._drag_start, self._drag_side)
            cmd = SetFacePlacementCommand(self._drag_face_id, self._drag_current, self._drag_side)
            cmd.do(self._scene)
            self._command_stack.push_executed(cmd, self._scene)
            if self._on_placement_committed is not None:
                # M7.5b Task 12 fix round 1: fired only here, on commit --
                # never per mouse-move -- so the Properties panel's Task 10
                # placement fields don't go stale after a drag on the
                # currently-selected face without re-reading on every frame.
                self._on_placement_committed()
        self._init_placement_drag_state()

    def _begin_placement_drag_projection(self, f_id: int, side: Side, event: QMouseEvent) -> None:
        """Cache what `_pixel_delta_to_uv` needs to convert cursor motion.

        `plane_basis` gives the same local (u, v) axes uv_projection bakes
        into the render, and the material's `texture_size` is the same
        divisor `project_corners` uses -- so a pixel delta lands in the same
        tile units Task 10's numeric fields already edit. `_world_face_normal`
        (shared with `_resolve_side`) supplies both the plane to raycast
        against and the plane-test normal, so this duplicates none of the
        inverse-transpose maths.
        """
        normal_local = np.asarray(self._scene.face_normal(f_id), dtype=np.float64)
        self._drag_u_axis, self._drag_v_axis = plane_basis(normal_local)

        material_id = self._scene.face_material(f_id, side)
        su, sv = 1.0, 1.0
        if self._model is not None:
            su, sv = self._model.materials.get(material_id).texture_size
        self._drag_su = float(su) or 1.0
        self._drag_sv = float(sv) or 1.0
        self._drag_scale = (float(self._drag_start.scale) or 1.0) if self._drag_start else 1.0

        center_local = np.asarray(self._scene.face_center(f_id), dtype=np.float64)
        wt = self._world_transform()
        if is_identity_transform(wt):
            self._drag_plane_point = center_local
            self._drag_local_from_world = None
        else:
            wt_arr = np.asarray(wt, dtype=np.float64)
            self._drag_plane_point = apply_mat(center_local.reshape(1, 3), wt_arr)[0]
            self._drag_local_from_world = mat_invert(wt_arr)[:3, :3]
        self._drag_plane_normal = self._world_face_normal(f_id)

        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
        self._drag_prev_hit = self._placement_plane_hit(
            np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        )

    def _placement_plane_hit(
        self, ray_origin: np.ndarray, ray_direction: np.ndarray
    ) -> np.ndarray | None:
        """World-space point where the cursor ray meets the dragged face's plane."""
        denom = float(np.dot(ray_direction, self._drag_plane_normal))
        if abs(denom) < 1e-9:
            return None  # ray parallel to the face -- no well-defined hit
        t = float(np.dot(self._drag_plane_point - ray_origin, self._drag_plane_normal)) / denom
        if t <= 0.0:
            return None  # behind the camera
        return ray_origin + ray_direction * t

    def _pixel_delta_to_uv(self, event: QMouseEvent) -> tuple[float, float]:
        """This mouse move's screen delta, converted to a UV (tile) delta.

        Re-raycasts the cursor against the dragged face's plane (world
        space) rather than working in raw screen pixels, so the conversion
        stays exact regardless of camera distance, zoom, or the face's
        angle to the view -- the world-space point under the cursor is what
        moved, and by how much. That world delta is carried back into local
        space by the inverse of the world transform's linear block (vectors,
        so translation cancels), then projected onto the face's own (u, v)
        basis and divided by `texture_size * scale`, matching
        `uv_projection.project_corners` and `apply_placement`'s contract
        that a larger scale makes a fixed offset delta move the texture
        less.

        Negated: `apply_placement` computes `base_uv/scale + offset`, so
        increasing offset_u samples the image further along +u for a FIXED
        surface point -- which looks like the image sliding toward -u on the
        surface. Grabbing a point and dragging it to +u (the cursor's actual
        motion) must slide the image toward +u instead, which takes the
        opposite sign on the offset delta.
        """
        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
        hit = self._placement_plane_hit(
            np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        )
        if hit is None or self._drag_prev_hit is None:
            self._drag_prev_hit = hit
            return 0.0, 0.0
        world_delta = hit - self._drag_prev_hit
        self._drag_prev_hit = hit
        local_delta = (
            self._drag_local_from_world @ world_delta
            if self._drag_local_from_world is not None
            else world_delta
        )
        du = -float(np.dot(local_delta, self._drag_u_axis)) / (self._drag_su * self._drag_scale)
        dv = -float(np.dot(local_delta, self._drag_v_axis)) / (self._drag_sv * self._drag_scale)
        return du, dv

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if self._drag_active:
            self._hovered_face = self._drag_face_id
            if event.buttons() & Qt.MouseButton.LeftButton:
                du, dv = self._pixel_delta_to_uv(event)
                if du or dv:
                    self.update_placement_drag(du, dv)
            return
        f_id = self._pick_face(event)
        self._hovered_face = f_id
        if (
            self._stroke_active
            and f_id is not None
            and self._stroke_material_id is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            side = self._resolve_side(event, f_id)
            self._paint_during_stroke(f_id, side, self._stroke_material_id)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        f_id = self._pick_face(event)
        if f_id is None or self._scene is None:
            return
        side = self._resolve_side(event, f_id)
        if event.modifiers() & _PLACEMENT_DRAG_MODIFIER:
            self.begin_placement_drag(f_id, side)
            self._begin_placement_drag_projection(f_id, side, event)
            return
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            # Eyedropper: sample the clicked side's material. Not a mutation.
            if self._set_active_material is not None:
                self._set_active_material(self._scene.face_material(f_id, side))
            return
        mat = self._active_material()
        if mat is None:
            return
        self._begin_stroke(mat.id)
        self._paint_during_stroke(f_id, side, mat.id)

    def on_mouse_release(self, event: QMouseEvent, snap) -> None:
        if self._drag_active:
            self.end_placement_drag()
            return
        self._end_stroke()

    def overlay(self) -> ToolOverlay:
        fills: list[np.ndarray] = []
        mat = self._active_material()
        tint = mat.base_color if mat is not None else _NEUTRAL_COLOR
        if self._hovered_face is not None and self._scene is not None:
            try:
                from bermake.geometry.transforms import apply_mat

                wt = self._world_transform()
                use_wt = not is_identity_transform(wt)
                wt_arr = np.asarray(wt, dtype=np.float64) if use_wt else None

                def _to_world(local_pos: np.ndarray) -> np.ndarray:
                    if not use_wt:
                        return local_pos
                    return apply_mat(local_pos.reshape(1, 3), wt_arr)[0]

                loop = self._scene.face_loop(self._hovered_face)
                fills.append(
                    np.array(
                        [
                            _to_world(np.asarray(self._scene.vertex(v).position, dtype=np.float32))
                            for v in loop
                        ],
                        dtype=np.float32,
                    )
                )
            except KeyError:
                pass
        return ToolOverlay(
            rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
            rubber_band_color=_NEUTRAL_COLOR,
            snap_marker_position=None,
            snap_marker_color=_NEUTRAL_COLOR,
            snap_marker_kind=0,
            face_fill_polygons=fills,
            face_fill_color=(tint[0], tint[1], tint[2], _HOVER_ALPHA),
        )

    @property
    def has_active_gesture(self) -> bool:
        # Mirrors EraserTool.has_active_gesture: True only for a live stroke
        # or placement drag, not for a mere hover -- so a right-click mid-drag
        # is suppressed as a cancel (ViewportWidget.contextMenuEvent) instead
        # of popping the context menu, and Esc/tool-switch semantics stay
        # consistent with the sibling drag tools. M7.5a shipped this
        # hardcoded False; M7.5b Task 12 must not repeat that for the new
        # drag.
        return self._stroke_active or self._drag_active

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None

    @property
    def status_text(self) -> str | None:
        mat = self._active_material()
        name = mat.name if mat is not None else "Default"
        return f"Paint: {name} · Alt-click to sample"
