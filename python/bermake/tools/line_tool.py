"""The Line drawing tool.

Click → click → click polyline. Snapping back onto the first vertex of the
gesture closes the loop and creates a face (provided ≥ 3 vertices exist).
Snapping onto some other existing vertex extends the polyline to it.
Otherwise, a new vertex is created at the snapped position.

Each click changes the scene at once, but the chain reaches the undo stack as one
command only on Enter, a double-click or loop closure. ESC rolls the unfinished
chain back. Until then the tool reports `holds_uncommitted_changes`.
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
    SplitFaceCommand,
)
from bermake.tools.shape_support import chain_cuts_face, vertex_for_snap
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import world_to_local_point
from bermake.viewport.snap_engine import AXIS_COLORS, marker_color


class _State(Enum):
    IDLE = 0
    DRAWING = 1


_NEUTRAL_COLOR = (0.85, 0.85, 0.85)


class LineTool(Tool):
    @property
    def name(self) -> str:
        return "Line"

    @property
    def shortcut(self) -> str:
        return "L"

    @property
    def id(self) -> str:
        return "line"

    def __init__(self) -> None:
        self._scene = None
        self._model = None
        self._units_provider = None
        self._state = _State.IDLE
        self._gesture_vertex_ids: list[int] = []
        self._preview_tip: np.ndarray | None = None
        self._rubber_band_color: tuple[float, float, float] = _NEUTRAL_COLOR
        self._snap_marker_pos: np.ndarray | None = None
        self._snap_marker_color: tuple[float, float, float] = _NEUTRAL_COLOR
        self._snap_marker_kind: int = 0
        self._composite: CompositeCommand | None = None
        self._command_stack = None

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
        self._snap_marker_color = marker_color(snap, _NEUTRAL_COLOR)
        self._snap_marker_kind = int(snap.kind)
        if self._state == _State.DRAWING:
            self._preview_tip = snap.world_position.copy()
            if snap.kind == SnapKind.AXIS_LOCK and snap.axis is not None:
                self._rubber_band_color = AXIS_COLORS[snap.axis]
            else:
                self._rubber_band_color = _NEUTRAL_COLOR

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if snap.kind == SnapKind.NONE:
            return
        s = self._scene  # type: ignore[assignment]

        if self._state == _State.IDLE:
            self._composite = CompositeCommand(name="Draw Line")
            vid = vertex_for_snap(self._tool_context(), snap, self._composite)
            self._gesture_vertex_ids = [vid]
            self._state = _State.DRAWING
            self._preview_tip = snap.world_position.copy()
            return

        assert self._composite is not None
        tip_vid = self._gesture_vertex_ids[-1]
        first_vid = self._gesture_vertex_ids[0]

        # Branch 1 — loop closure (snap back onto the first vertex with ≥3 points).
        if (
            snap.kind == SnapKind.ENDPOINT
            and snap.vertex_id == first_vid
            and len(self._gesture_vertex_ids) >= 3
        ):
            e_cmd = AddEdgeCommand(tip_vid, first_vid)
            e_cmd.do(s)
            self._composite.children.append(e_cmd)
            f_cmd = AddFaceCommand(tuple(self._gesture_vertex_ids))
            f_cmd.do(s)
            self._composite.children.append(f_cmd)
            if self._command_stack is not None:
                self._command_stack.push_executed(self._composite, self._scene)
            self._reset_gesture()
            return

        # Branches 2/3 — extend to a resolved vertex (reuse / split / new).
        children = self._composite.children
        before = len(children)
        vid = vertex_for_snap(self._tool_context(), snap, self._composite)
        if vid == tip_vid:
            # Degenerate: clicked the current tip. Take back whatever the
            # resolution just did.
            for cmd in reversed(children[before:]):
                cmd.undo(s)
            del children[before:]
            return
        e_cmd = AddEdgeCommand(tip_vid, vid)
        e_cmd.do(s)
        self._composite.children.append(e_cmd)
        self._gesture_vertex_ids.append(vid)

    def apply_typed_value(self, text, units) -> bool:
        from bermake.units import parse_length

        if (
            self._state != _State.DRAWING
            or self._preview_tip is None
            or not self._gesture_vertex_ids
        ):
            return False
        length = parse_length(text, units)
        if length is None or length <= 0:
            return False
        s = self._scene
        # anchor_local is in the active context's local space.
        # _preview_tip is always world. Convert anchor local→world to get the
        # direction purely in world space, then convert the world target local.
        anchor_local = np.asarray(s.vertex(self._gesture_vertex_ids[-1]).position, np.float32)
        wt = self._world_transform()
        from bermake.geometry.transforms import apply_mat, is_identity_transform

        if is_identity_transform(wt):
            anchor_world = anchor_local
        else:
            anchor_world = apply_mat(
                anchor_local.astype(np.float64).reshape(1, 3), np.asarray(wt, dtype=np.float64)
            )[0].astype(np.float32)
        direction = np.asarray(self._preview_tip, np.float32) - anchor_world
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return False
        target_world = (anchor_world + (direction / norm) * length).astype(np.float32)
        target_local = world_to_local_point(target_world, wt)
        from bermake.commands.scene_commands import AddEdgeCommand, AddVertexCommand

        assert self._composite is not None
        v_cmd = AddVertexCommand(target_local)
        v_cmd.do(s)
        self._composite.children.append(v_cmd)
        new_vid = v_cmd.vertex_id
        e_cmd = AddEdgeCommand(self._gesture_vertex_ids[-1], new_vid)
        e_cmd.do(s)
        self._composite.children.append(e_cmd)
        self._gesture_vertex_ids.append(new_vid)
        self._preview_tip = target_world.copy()
        return True

    def on_mouse_double_click(self, event: QMouseEvent, snap) -> None:
        """Third gesture-end path, alongside Enter (below) and loop closure
        (branch 1 of on_mouse_press). Finishes the open polyline exactly like
        Enter, including the face-split check: Qt's own double-click delivers
        a press first, and that press either extends the polyline or (when it
        lands back on the current tip, as a double-click's second click
        normally does) is a no-op, so by the time this fires the gesture
        state is whatever a single Enter press would also see.

        Requires at least one committed segment (>= 2 gesture vertices).
        Unlike Enter, double-click was previously a no-op inherited from the
        Tool base class regardless of gesture state, so a double-click with
        only the seed vertex placed must stay a no-op here too rather than
        start discarding gestures a plain double-click never used to touch.
        """
        if (
            self._state != _State.DRAWING
            or self._composite is None
            or len(self._gesture_vertex_ids) < 2
        ):
            return
        self._finish_open_polyline()

    def on_key_press(self, event: QKeyEvent) -> None:
        key = event.key()
        s = self._scene  # type: ignore[assignment]

        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # Finish the open polyline: register it as one undoable unit and end
            # the gesture (ready to start a new line). Matches SketchUp/CAD Enter.
            # This is a split path (see module note above on_mouse_double_click);
            # loop closure (branch 1 of on_mouse_press) and Escape (below) never
            # split.
            if self._state == _State.DRAWING and self._composite is not None:
                self._finish_open_polyline()
            return

        if key != Qt.Key.Key_Escape:
            return
        # ESC mid-gesture: roll back the in-progress composite. Never a split.
        if self._composite is not None:
            self._composite.undo(s)
            self._composite = None
        self._reset_gesture()

    def _finish_open_polyline(self) -> None:
        """Shared tail of the Enter and double-click gesture-end paths.

        Registers the open polyline as one undoable unit; if the drawn chain
        divides exactly one face (chain_cuts_face), the split is executed and
        appended to this same composite BEFORE push_executed, so undoing the
        line also undoes the split in a single step (M7.6a task 5).
        """
        assert self._composite is not None
        s = self._scene  # type: ignore[assignment]
        if len(self._gesture_vertex_ids) >= 2 and self._composite.children:
            fid = chain_cuts_face(s, self._gesture_vertex_ids)
            if fid is not None:
                split_cmd = SplitFaceCommand(fid, self._gesture_vertex_ids)
                split_cmd.do(s)
                self._composite.children.append(split_cmd)
            if self._command_stack is not None:
                self._command_stack.push_executed(self._composite, self._scene)
        else:
            # Only the start point was placed, nothing to commit; discard.
            self._composite.undo(s)
        self._composite = None
        self._reset_gesture()

    def overlay(self) -> ToolOverlay:
        s = self._scene  # type: ignore[assignment]
        if (
            self._state == _State.DRAWING
            and s is not None
            and self._preview_tip is not None
            and self._gesture_vertex_ids
        ):
            anchor_local = s.vertex(self._gesture_vertex_ids[-1]).position
            wt = self._world_transform()
            from bermake.geometry.transforms import apply_mat, is_identity_transform

            if is_identity_transform(wt):
                anchor_world = anchor_local
            else:
                anchor_world = apply_mat(
                    np.asarray(anchor_local, dtype=np.float64).reshape(1, 3),
                    np.asarray(wt, dtype=np.float64),
                )[0]
            segments = np.array(
                [
                    [float(anchor_world[0]), float(anchor_world[1]), float(anchor_world[2])],
                    [
                        float(self._preview_tip[0]),
                        float(self._preview_tip[1]),
                        float(self._preview_tip[2]),
                    ],
                ],
                dtype=np.float32,
            )
        else:
            segments = np.zeros((0, 3), dtype=np.float32)

        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=self._rubber_band_color,
            snap_marker_position=(
                self._snap_marker_pos.copy() if self._snap_marker_pos is not None else None
            ),
            snap_marker_color=self._snap_marker_color,
            snap_marker_kind=self._snap_marker_kind,
        )

    @property
    def holds_uncommitted_changes(self) -> bool:
        """True once a click of this chain has changed the model.

        Each click executes its AddVertex/AddEdge commands at once but keeps
        them in `_composite`, which reaches the undo stack only on Enter, a
        double-click or loop closure; Escape undoes it. So a chain holds
        changes whenever the composite has any child. A chain whose clicks
        all reused existing vertices has no child and holds nothing.
        """
        return self._composite is not None and bool(self._composite.children)

    @property
    def has_active_gesture(self) -> bool:
        return self._state == _State.DRAWING

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        s = self._scene  # type: ignore[assignment]
        if self._state != _State.DRAWING or s is None or not self._gesture_vertex_ids:
            return None
        anchor = np.asarray(s.vertex(self._gesture_vertex_ids[-1]).position, np.float32).copy()
        from bermake.geometry.transforms import apply_mat, is_identity_transform

        wt = self._world_transform()
        if is_identity_transform(wt):
            return anchor
        return apply_mat(anchor.astype(np.float64).reshape(1, 3), np.asarray(wt, dtype=np.float64))[
            0
        ]

    def _gesture_length(self) -> float | None:
        """Distance from the last placed vertex to the live preview tip, or
        None with no in-progress segment. Reused by measurement_text -- the
        same value apply_typed_value's committed segment length replaces."""
        anchor = self.anchor_or_none
        if anchor is None or self._preview_tip is None:
            return None
        return float(np.linalg.norm(np.asarray(self._preview_tip, np.float64) - anchor))

    @property
    def measurement_text(self) -> str | None:
        length = self._gesture_length()
        if length is None:
            return None
        if self._units_provider is not None:
            from bermake.units import format_length

            return format_length(length, self._units_provider())
        return f"{length:.3f}"

    # ---- internal -------------------------------------------------------
    def _tool_context(self) -> ToolContext:
        return ToolContext(scene=self._scene, model=self._model)

    def _reset_gesture(self) -> None:
        self._state = _State.IDLE
        self._gesture_vertex_ids = []
        self._preview_tip = None
        self._rubber_band_color = _NEUTRAL_COLOR
        self._snap_marker_pos = None
        self._snap_marker_kind = 0
        self._composite = None
