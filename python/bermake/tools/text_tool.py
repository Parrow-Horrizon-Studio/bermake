"""The Text (leader label) tool (M7d), shortcut "N".

Two clicks: anchor, then the text position -- then a QInputDialog asks for
the text. Cancelled or blank input creates nothing.

Frame handling (mirrors DimensionTool -- see `bermake.tools.annotation_support`,
the shared helper both tools use so they cannot drift out of sync):
- The live rubber-band preview (a leader line from anchor to cursor) is built
  entirely in WORLD space, because the renderer draws
  `ToolOverlay.rubber_band_segments` in world space with no model matrix
  applied. `_anchor_world` / `_cursor_world` are kept as world-space snaps
  for as long as the gesture is open.
- The committed `Label`'s anchor/text_pos are CONTEXT-LOCAL (see
  `bermake.model.annotation.Label`), so each point is converted via
  `world_to_active_local` only at the moment it is written to storage.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.commands.annotation_commands import CreateAnnotationCommand
from bermake.model.annotation import Label
from bermake.tools.annotation_support import NEUTRAL_PREVIEW_COLOR, world_to_active_local
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.snap_engine import SnapKind, marker_color


class TextTool(Tool):
    """Click (anchor) -> click (text position) -> type text: one leader label."""

    def __init__(self) -> None:
        self._model = None
        self._command_stack = None
        self._anchor_world: np.ndarray | None = None
        self._cursor_world: np.ndarray | None = None
        self._snap_color: tuple[float, float, float] = NEUTRAL_PREVIEW_COLOR
        self._snap_kind = 0

    @property
    def name(self) -> str:
        return "Text"

    @property
    def shortcut(self) -> str:
        return "N"

    @property
    def id(self) -> str:
        return "text"

    def activate(self, ctx: ToolContext) -> None:
        self._command_stack = ctx.command_stack
        self._model = ctx.model
        self._reset()

    def deactivate(self) -> None:
        self._reset()

    def prompt_text(self, default: str = "") -> str | None:
        """Ask the user for the label text. Overridable for testing."""
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(None, "Text", "Label:", text=default)
        return text if ok else None

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            self._cursor_world = None
            self._snap_kind = 0
            return
        self._cursor_world = np.asarray(snap.world_position, dtype=np.float64).copy()
        self._snap_color = marker_color(snap, NEUTRAL_PREVIEW_COLOR)
        self._snap_kind = int(snap.kind)

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if snap.kind == SnapKind.NONE:
            return
        pt_world = np.asarray(snap.world_position, dtype=np.float64).copy()

        if self._anchor_world is None:
            self._anchor_world = pt_world
            return

        text = self.prompt_text("")
        if text is None or not text.strip():
            self._reset()
            return

        anchor_local = world_to_active_local(self._model, self._anchor_world)
        text_pos_local = world_to_active_local(self._model, pt_world)
        label = Label(
            self._model.new_annotation_id(),
            tuple(float(v) for v in anchor_local),
            tuple(float(v) for v in text_pos_local),
            text.strip(),
        )
        self._command_stack.execute(
            CreateAnnotationCommand(label, self._model.active_context), self._model
        )
        self._reset()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._reset()

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        if self._anchor_world is not None and self._cursor_world is not None:
            segments = np.array([self._anchor_world, self._cursor_world], dtype=np.float32).reshape(
                -1, 3
            )
        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=NEUTRAL_PREVIEW_COLOR,
            snap_marker_position=(
                self._cursor_world.copy() if self._cursor_world is not None else None
            ),
            snap_marker_color=self._snap_color,
            snap_marker_kind=self._snap_kind,
        )

    @property
    def has_active_gesture(self) -> bool:
        return self._anchor_world is not None

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None

    @property
    def status_text(self) -> str | None:
        return None

    def _reset(self) -> None:
        self._anchor_world = None
        self._cursor_world = None
        self._snap_color = NEUTRAL_PREVIEW_COLOR
        self._snap_kind = 0
