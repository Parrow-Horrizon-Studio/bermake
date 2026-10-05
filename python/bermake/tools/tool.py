"""Tool abstraction for drawing/modeling tools.

A Tool owns a small state machine, receives Qt events from the active
ViewportWidget, mutates the Scene, and emits a per-frame ToolOverlay
containing transient preview geometry (rubber-band, snap marker).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtGui import QKeyEvent, QMouseEvent


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Handed to Tool.activate(); gives the tool a handle to the live Scene,
    CommandStack, Camera, and a viewport-size accessor."""

    scene: object
    command_stack: object = None  # M3a-introduced — bermake.commands.CommandStack
    camera: object = None  # M3b-introduced — bermake.viewport.camera.Camera
    # M3b-introduced — callable () -> tuple[int, int] returning (width, height)
    widget_size_provider: object = None
    selection: object = None  # M4b — bermake.selection.Selection (shared)
    units_provider: object = None  # M4d — callable () -> bermake.units.Units (or None)
    # M4e — bermake.model.Model (the scene graph); used for active transform + enter/exit
    model: object = None
    # M4e — callable () -> None; rebuilds the tool context after the active
    # editing context changes
    request_context_rebuild: object = None
    active_material_provider: object = None  # M5b — callable () -> Material (active material)
    set_active_material: object = None  # M5b — callable (int) -> None (eyedropper -> dock)
    # M7.5b Task 12 fix round 1 — callable () -> None; fired once when
    # PaintTool.end_placement_drag() commits a drag, so the Properties
    # panel's Task 10 placement fields don't go stale after a Shift-drag on
    # the currently-selected face. Fired on commit only, never per mouse-move.
    on_placement_committed: object = None
    # M7.6b Task 7 fix round 1 -- callable () -> bool, mirroring
    # ViewportWidget.show_guides. Picking tools (Select, Erase) hold the
    # Model, not the viewport, so this is how a hidden guide's kind reaches
    # them without a new Model<->viewport coupling; None (a bare test
    # ToolContext) reads as "guides visible", matching the viewport's own
    # default.
    show_guides_provider: object = None
    # M7.6c -- callable () -> bool, mirroring ViewportWidget.select_vertices.
    # Vertex picking is opt-in and off by default, so None (a bare test
    # ToolContext) reads as "vertices not pickable", matching the viewport's
    # own default and keeping every pre-M7.6c test meaningful.
    select_vertices_provider: object = None


@dataclass(frozen=True, slots=True)
class ToolOverlay:
    """Transient preview geometry rebuilt every frame by the active tool."""

    rubber_band_segments: np.ndarray  # shape (2*N, 3), float32
    rubber_band_color: tuple[float, float, float]
    snap_marker_position: np.ndarray | None
    snap_marker_color: tuple[float, float, float]
    # SnapKind value (0=NONE/no marker); stored as int to avoid circular import
    snap_marker_kind: int = 0

    # M3b: filled face overlays (hover-highlight / armed face / ghost prism faces).
    face_fill_polygons: list[np.ndarray] = field(default_factory=list)
    # List of (N, 3) float32 world-space vertex loops. Renderer earcut-triangulates
    # each at draw time.

    face_fill_color: tuple[float, float, float, float] = (0.4, 0.7, 1.0, 0.15)
    # RGBA. Default is M3b's "ghost prism" color (light blue, 15% alpha).

    # M4b: screen-space box-select rectangle (pixels: x0,y0,x1,y1) or None.
    box_rect: tuple[float, float, float, float] | None = None
    box_rect_color: tuple[float, float, float] = (0.30, 0.55, 0.95)

    # M7.6c (#39): crossing mode draws dashed, matching SketchUp. Colour
    # still distinguishes the two modes; the dash is additive.
    box_rect_dashed: bool = False

    # M4c: generic gizmo primitives (transform tools).
    # world_polylines: list of (segments (2*N, 3) float32, rgb, width) drawn in
    # world space via the line shader. screen_markers: list of (world_pos (3,),
    # size_px, rgb) drawn as screen-space outlined squares (scale grips).
    world_polylines: list = field(default_factory=list)
    screen_markers: list = field(default_factory=list)

    # M7d Task 10b: annotation hover highlight. The id of the single
    # annotation currently under the cursor (SelectTool only), or None.
    # Surfaced here rather than via a new channel: _paint_annotations reads it
    # off the same per-frame ToolOverlay that paintGL already asks the active
    # tool for to render edge/face/instance hover.
    hovered_annotation_id: int | None = None


class Tool(ABC):
    """Base class for all M2+ tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def shortcut(self) -> str: ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable identity used for dispatch (ToolManager.register/activate_by_id).

        Unlike shortcut, this is never empty -- it is the tool's registry key,
        so two tools sharing no keyboard shortcut still resolve to distinct
        entries (M7.4 Task 5).
        """

    @property
    @abstractmethod
    def has_active_gesture(self) -> bool:
        """True if the tool is in the middle of a multi-click gesture.

        MainWindow uses this to decide whether ESC should cancel the gesture
        (forward to tool.on_key_press) or deactivate the tool entirely
        (ToolManager.deactivate_current).
        """

    @abstractmethod
    def activate(self, ctx: ToolContext) -> None: ...

    @abstractmethod
    def deactivate(self) -> None: ...

    @property
    def holds_uncommitted_changes(self) -> bool:
        """True while the Model/Scene holds a temporary change this tool made
        that a cancel would roll back (M7.12, #77).

        Autosave reads the model while the user works, so it must not run
        while this is True: it would save half-finished geometry that Escape
        is about to remove. Committed commands do not count, and neither does
        anything the tool only draws in its overlay. The default is False;
        only a tool that writes to the model during a gesture, ahead of the
        command that finishes it, overrides this. The per-tool audit behind
        the overrides is pinned by tests/test_tool_uncommitted_changes.py.
        """
        return False

    @property
    def consumes_arrow_keys(self) -> bool:
        """True if this tool binds Up/Down itself during a gesture.

        M7.6b gives the arrow keys to inference locking. Left and Right are free
        in every tool, but Up and Down were already taken by PolygonTool (sides),
        RoofTool (flip) and RotateTool (axis cycling), so those three keep them
        and forgo the blue-axis and edge locks. Declared here rather than as a
        tool-id list in MainWindow, which would silently drift.
        """
        return False

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        """Default: do nothing. Tools override as needed."""
        return None

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        """Default: do nothing."""
        return None

    def on_mouse_release(self, event: QMouseEvent, snap) -> None:
        """Default: do nothing. Tools that need drag-release (e.g. box-select)
        override this."""
        return None

    def on_mouse_double_click(self, event: QMouseEvent, snap) -> None:
        """Default: do nothing. SelectTool overrides to enter groups/components."""
        return None

    def on_mouse_triple_click(self, event: QMouseEvent, snap) -> None:
        """Default: do nothing. SelectTool overrides to select everything
        connected to the entity under the cursor.

        Qt sends no triple-click event. ViewportWidget synthesizes this one
        from a ClickRuns counter, so the third press arrives BOTH here and at
        on_mouse_press; this hook is additive, not a replacement.
        """
        return None

    def on_key_press(self, event: QKeyEvent) -> None:
        """Default: do nothing."""
        return None

    def apply_typed_value(self, text: str, units) -> bool:
        """Apply a typed VCB value to the in-progress gesture.

        Returns True if the value was consumed (the tool re-resolved + committed
        or advanced its gesture), else False. Default: not supported.
        """
        return False

    @abstractmethod
    def overlay(self) -> ToolOverlay: ...

    @property
    @abstractmethod
    def anchor_or_none(self) -> np.ndarray | None:
        """Rubber-band anchor used by the SnapEngine for axis-lock."""

    @property
    def status_text(self) -> str | None:
        """Optional third text segment for the status bar.

        Default None means this tool contributes nothing extra to the status
        bar beyond `<name> · <snap>`. PushPullTool overrides this to show the
        current extrusion depth during DRAGGING.
        """
        return None

    @property
    def measurement_text(self) -> str | None:
        """The numeric value for the Measurements box and the cursor readout.

        Deliberately distinct from `status_text`, which is prose for the prompt
        area. This is a bare value safe to drop into a small box beside the
        cursor: lengths through `bermake.units.format_length`, angles as
        "12 deg" matching rotate_tool.py:266, two-axis values joined with " x ".
        Default None means the tool contributes no readout.
        """
        return None
