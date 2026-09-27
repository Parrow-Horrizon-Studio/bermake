"""The Move tool (M) — point-to-point translation of the selection.

Press snaps a grab point and captures the selection's vertices + their
original positions. Drag computes delta = destination - grab (axis-lock is
provided by the SnapEngine, which the viewport calls with anchor_or_none =
the grab point). Release commits one TransformVerticesCommand. The mesh is
never mutated until release, so Esc/deactivate simply resets.

Instance-mode (M4e §7.3): if ctx.selection.instances is non-empty at press,
the tool composes the gesture delta into each instance's 4x4 transform and
emits TransformInstanceCommand(s) instead of TransformVerticesCommand.

Transform-awareness (M4e fix): snap quantities arrive in WORLD space.
When editing INSIDE a moved group/component the active context has a
non-identity world transform.  We must convert world vectors/points into
the active context's LOCAL frame before applying them to local mesh vertices
or local instance transforms.  At the root context (identity) every
conversion is a no-op so existing behaviour is fully preserved.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.annotation_commands import MoveAnnotationsCommand
from bermake.commands.scene_commands import TransformVerticesCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert, translate
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.tools.transform_support import selection_vertices

_NEUTRAL = (0.85, 0.85, 0.85)
_GHOST = (0.30, 0.65, 1.0)


class MoveTool(Tool):
    @property
    def name(self) -> str:
        return "Move"

    @property
    def shortcut(self) -> str:
        return "M"

    @property
    def id(self) -> str:
        return "move"

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._stack = None
        self._selection = None
        self._units_provider = None
        self._dragging = False
        self._grab: np.ndarray | None = None
        self._delta = np.zeros(3, dtype=np.float32)
        self._vertex_ids: list[int] = []
        self._orig: dict[int, np.ndarray] = {}
        # instance-mode state
        self._instance_mode = False
        self._instances: list = []  # list of Instance objects

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _world_vec_to_local(self, world_vec: np.ndarray) -> np.ndarray:
        """Convert a world-space DIRECTION/VECTOR into the active context's local frame.

        Translation has no effect on vectors — only the 3x3 rotation/scale block
        of the inverse is applied.  Returns the vector unchanged when the active
        transform is None or identity (root context).
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
        self._units_provider = ctx.units_provider
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._selection is None or self._selection.is_empty():
            return
        if snap.kind == SnapKind.NONE:
            return

        # Determine mode at gesture start
        self._instance_mode = bool(self._selection.instances)

        if self._instance_mode:
            self._instances = self._resolve_instances()
            if not self._instances:
                self._instance_mode = False
                return
            self._grab = np.asarray(snap.world_position, np.float32).copy()
            self._delta = np.zeros(3, dtype=np.float32)
            self._dragging = True
        else:
            self._vertex_ids = selection_vertices(self._scene, self._selection)
            # M7d Task 12: an annotation-only selection has no vertices to
            # grab, but the gesture must still open so its Move can commit.
            if not self._vertex_ids and not self._selection.annotations:
                return
            self._orig = {v: self._scene.vertex(v).position.copy() for v in self._vertex_ids}
            self._grab = np.asarray(snap.world_position, np.float32).copy()
            self._delta = np.zeros(3, dtype=np.float32)
            self._dragging = True

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if not self._dragging or self._grab is None or snap.kind == SnapKind.NONE:
            return
        self._delta = (np.asarray(snap.world_position, np.float32) - self._grab).astype(np.float32)

    def on_mouse_release(self, event: QMouseEvent, snap) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        from bermake.viewport.snap_engine import SnapKind

        if (
            snap is not None
            and getattr(snap, "world_position", None) is not None
            and snap.kind != SnapKind.NONE
            and self._grab is not None
        ):
            dest = np.asarray(snap.world_position, np.float32)
            self._delta = (dest - self._grab).astype(np.float32)

        if self._instance_mode:
            ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._commit_instance_move(self._delta, move_copy=ctrl)
        else:
            self._commit_vertex_and_annotation_move(self._delta)
        self._reset()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset()

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if not self._dragging or self._grab is None:
            return False
        dist = parse_length(text, units)
        if dist is None:
            return False
        norm = float(np.linalg.norm(self._delta))
        if norm < 1e-9:
            return False
        direction = (self._delta / norm).astype(np.float32)
        typed_delta = direction * dist

        if self._instance_mode:
            self._commit_instance_move(typed_delta)
        else:
            self._commit_vertex_and_annotation_move(typed_delta)
        self._reset()
        return True

    def overlay(self) -> ToolOverlay:
        polylines: list = []
        segs = np.zeros((0, 3), dtype=np.float32)
        marker = None
        if self._dragging and self._grab is not None and self._scene is not None:
            if not self._instance_mode:
                ghost = self._ghost_segments()
                if ghost.shape[0] >= 2:
                    polylines.append((ghost, _GHOST, 2.0))
            segs = np.array([self._grab, self._grab + self._delta], dtype=np.float32)
            marker = (self._grab + self._delta).astype(np.float32)
        return ToolOverlay(
            rubber_band_segments=segs,
            rubber_band_color=_NEUTRAL,
            snap_marker_position=marker,
            snap_marker_color=_GHOST,
            world_polylines=polylines,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._dragging

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return self._grab.copy() if (self._dragging and self._grab is not None) else None

    def _move_distance_text(self) -> str | None:
        """The live drag distance, formatted, or None while not dragging.

        Reused by both status_text (prefixed prose) and measurement_text
        (the bare value)."""
        if not self._dragging:
            return None
        dist = float(np.linalg.norm(self._delta))
        if self._units_provider is not None:
            from bermake.units import format_length

            return format_length(dist, self._units_provider())
        return f"{dist:.3f}"

    @property
    def status_text(self) -> str | None:
        if self._selection is None or self._selection.is_empty():
            return "Select geometry first"
        distance_text = self._move_distance_text()
        if distance_text is not None:
            return f"Move {distance_text}"
        return "Move: pick a grab point"

    @property
    def measurement_text(self) -> str | None:
        return self._move_distance_text()

    # ---- internal ----

    def _resolve_instances(self) -> list:
        """Resolve selected instance ids to Instance objects from the active context."""
        if self._model is None or self._selection is None:
            return []
        inst_ids = self._selection.instances
        return [inst for inst in self._model.active_context.children if inst.id in inst_ids]

    def _commit_instance_move(self, delta: np.ndarray, move_copy: bool = False) -> None:
        """Emit TransformInstanceCommand(s) or CreateInstanceCommand(s) for the
        current instance selection.

        If *move_copy* is True the originals stay put and new instances are created at the
        translated transform (Ctrl-during-Move behaviour).

        `delta` arrives in WORLD space.  When the active context has a non-identity world
        transform (i.e. we are editing INSIDE a moved group), instance transforms are
        expressed in the ACTIVE CONTEXT's LOCAL frame, so we must convert the world delta
        to that frame (inverse 3x3 block) before building the translation matrix.
        At the root context (identity) the conversion is a no-op.
        """
        from bermake.commands.command import CompositeCommand
        from bermake.commands.instance_commands import (
            CreateInstanceCommand,
            TransformInstanceCommand,
        )
        from bermake.geometry.transforms import mat_translate

        # Convert world delta → local frame of the active context.
        local_delta = self._world_vec_to_local(delta)
        delta_mat = mat_translate(local_delta)

        if move_copy:
            if self._model is None:
                return
            parent = self._model.active_context
            created_cmds = [
                CreateInstanceCommand(parent, inst.definition, delta_mat @ inst.transform)
                for inst in self._instances
            ]
            if not created_cmds:
                return
            if len(created_cmds) == 1:
                cmd = created_cmds[0]
            else:
                cmd = CompositeCommand(name="Move-copy", children=created_cmds)
            if self._stack is not None:
                self._stack.execute(cmd, self._model)
            # Update selection to the newly created instances. Selection.replace
            # zeroes every field not passed, so a co-selected annotation (which
            # is neither copied nor moved by move-copy -- it just stays where
            # it is) must have its id threaded through explicitly here, or it
            # is silently dropped from the selection (#96).
            if self._selection is not None:
                new_ids = [c.created_instance.id for c in created_cmds]
                self._selection.replace(
                    instances=new_ids, annotations=set(self._selection.annotations)
                )
        else:
            cmds = [
                TransformInstanceCommand(inst, delta_mat @ inst.transform)
                for inst in self._instances
            ]
            # M7d Task 12: fold a co-selected annotation move into this SAME
            # composite -- one Move gesture must be one undo entry, so this
            # reuses local_delta above verbatim rather than recomputing it.
            has_ann_sel = self._selection is not None and self._selection.annotations
            if has_ann_sel and self._model is not None:
                cmds.append(
                    MoveAnnotationsCommand(
                        list(self._selection.annotations),
                        tuple(float(x) for x in local_delta),
                        self._model.active_context,
                    )
                )
            if not cmds:
                return
            if len(cmds) == 1:
                cmd = cmds[0]
            else:
                cmd = CompositeCommand(name="Move Instances", children=cmds)
            if self._stack is not None and self._model is not None:
                self._stack.execute(cmd, self._model)

    def _commit_vertex_and_annotation_move(self, delta: np.ndarray) -> None:
        """Compose the vertex move and any selected-annotation move into a
        SINGLE undo entry.

        Annotations can be co-selected with geometry, so one Move gesture
        must be one undo step: an undo must restore BOTH. `delta` arrives in
        WORLD space; `local_delta` (the active context's LOCAL frame) is
        computed once here and reused verbatim for the mesh vertices and for
        MoveAnnotationsCommand -- never recomputed independently.
        """
        from bermake.commands.command import CompositeCommand

        local_delta = self._world_vec_to_local(delta)
        children: list = []

        ids = self._vertex_ids
        if ids:
            pts = np.array([self._orig[v] for v in ids], np.float32)
            new = translate(pts, local_delta)
            moves = {v: (self._orig[v], new[i]) for i, v in enumerate(ids)}
            vertex_cmd = TransformVerticesCommand(moves)
            if not vertex_cmd.is_empty():
                children.append(vertex_cmd)

        has_ann_sel = self._selection is not None and self._selection.annotations
        if has_ann_sel and self._model is not None:
            children.append(
                MoveAnnotationsCommand(
                    list(self._selection.annotations),
                    tuple(float(x) for x in local_delta),
                    self._model.active_context,
                )
            )

        if not children or self._stack is None:
            return
        if len(children) == 1:
            cmd = children[0]
        else:
            cmd = CompositeCommand(name="Move", children=children)
        self._stack.execute(cmd, self._scene)

    def _ghost_segments(self) -> np.ndarray:
        """Selection edges + face loops as world segments, translated by delta.

        Local mesh vertices are transformed to world (via active_world_transform)
        before adding the WORLD delta, so the ghost renders at the correct world
        location even when editing inside a moved group/component.
        """
        s = self._scene
        sel = self._selection
        pts: list[list[float]] = []

        wt = self._world_transform()
        use_wt = not is_identity_transform(wt)
        wt_arr = np.asarray(wt, np.float64) if use_wt else None

        def _to_world(local_pos):
            if not use_wt:
                return np.asarray(local_pos, np.float32)
            return apply_mat(np.asarray(local_pos, np.float64).reshape(1, 3), wt_arr)[0]

        def seg(p0, p1):
            q0 = _to_world(p0) + self._delta
            q1 = _to_world(p1) + self._delta
            pts.append([float(q0[0]), float(q0[1]), float(q0[2])])
            pts.append([float(q1[0]), float(q1[1]), float(q1[2])])

        for e_id in sel.edges:
            try:
                e = s.edge(e_id)
            except KeyError:
                continue
            seg(s.vertex(e.v1_id).position, s.vertex(e.v2_id).position)
        for f_id in sel.faces:
            try:
                loop = s.face_loop(f_id)
            except KeyError:
                continue
            n = len(loop)
            for i in range(n):
                seg(s.vertex(loop[i]).position, s.vertex(loop[(i + 1) % n]).position)
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), np.float32)

    def _reset(self) -> None:
        self._dragging = False
        self._grab = None
        self._delta = np.zeros(3, dtype=np.float32)
        self._vertex_ids = []
        self._orig = {}
        self._instance_mode = False
        self._instances = []
