"""PrimitiveTool base and the four primitive tools (M7.4 Task 11).

Task 9 added four mesh generators to the C++ kernel (`bermake._core.make_box`
/ `make_cylinder` / `make_cone` / `make_sphere`), each returning a standalone
`HalfEdgeMesh` sized exactly to the caller's dimensions -- no scaling
transform is needed to reach the requested size, only a translation to place
it. Task 10's `build_mesh_into_scene` walks a generated mesh into a `Scene`
as ordinary, undoable commands. This module is the tool layer on top: it
owns the gesture that collects width/depth/height from the user and the four
thin subclasses that pick which generator to call.

Gesture, two stages:
  1. DRAGGING_FOOTPRINT -- exactly RectangleTool's own shape: first click
     sets one corner on the ground plane, the cursor previews the opposite
     corner, second click commits a footprint (width, depth).
  2. DRAGGING_HEIGHT -- exactly PushPullTool's own shape: the cursor now
     drives a single extrusion-style distance (here, height above the
     footprint) via the same camera-ray / line line-CPA math PushPullTool
     uses against its face normal, except the line is anchored at the
     footprint's center along local +Z rather than a face's own normal.
     A click commits; ESC at any point in either stage cancels with no
     scene mutation, since nothing is applied until the final click.

Round primitives (cylinder, cone, sphere) need a single radius, not a
width/depth pair, so `_make_mesh` derives one from the dragged footprint:
inscribed in it (`min(width, depth) / 2`) for cylinder/cone, so the base
never spills past what was dragged. The sphere has no independent
"height" of its own -- it is one radius in every direction -- so it uses
`min(width, depth, height) / 2`, the smallest of the three dragged
extents, so the sphere always fits inside the box the user dragged rather
than spilling past whichever extent was smallest.

Segments/rings floors: Task 9 made an out-of-range `segments`/`rings`
raise a catchable `ValueError` instead of segfaulting the interpreter, but
a UI that lets a user type "2" and then calls straight through would still
show them a traceback. `segments`/`rings` are ordinary properties here
whose setters clamp to the floor (`SEGMENTS_FLOOR` / `RINGS_FLOOR`) --
`PrimitiveOptionsBar`'s spin boxes enforce the same floor as their
minimum, so the two paths (typed value, direct attribute set) agree.
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake._core import make_box, make_cone, make_cylinder, make_sphere
from bermake.commands import CompositeCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.scene.mesh_builder import build_mesh_into_scene
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.viewport.picking import ray_into_local, world_to_local_point

# bermake._core's make_cylinder/make_cone/make_sphere floors (M7.4 Task 9).
SEGMENTS_FLOOR = 3
RINGS_FLOOR = 2

_DEFAULT_SEGMENTS = 24  # matches the generators' own C++-side defaults
_DEFAULT_RINGS = 12

_MIN_FOOTPRINT = 1e-3  # world units; a footprint smaller than this cancels
_MIN_HEIGHT = 1e-3  # world units; below this, the second click cancels
_DEGENERATE_VIEW_EPSILON = 1e-4  # matches PushPullTool's own degenerate-view guard

_NEUTRAL_COLOR = (0.85, 0.85, 0.85)
_GHOST_FILL_COLOR = (0.40, 0.70, 1.00, 0.15)  # matches Push/Pull's ghost-prism color


class _State(Enum):
    IDLE = 0
    DRAGGING_FOOTPRINT = 1
    DRAGGING_HEIGHT = 2


class PrimitiveTool(Tool):
    """Shared two-stage gesture and commit path for the four primitives.

    Subclasses supply `name`, `id`, and `_make_mesh`; nothing about the
    gesture, the parameter floors, or how the result reaches the scene
    differs between them.
    """

    def __init__(self) -> None:
        self._scene = None
        self._command_stack = None
        self._camera = None
        self._widget_size_provider = None
        self._units_provider = None
        self._model = None

        self._segments = _DEFAULT_SEGMENTS
        self._rings = _DEFAULT_RINGS

        self._state: _State = _State.IDLE

        # DRAGGING_FOOTPRINT data
        self._first_corner: np.ndarray | None = None
        self._preview_corner: np.ndarray | None = None
        self._snap_marker_pos: np.ndarray | None = None
        self._snap_marker_color: tuple[float, float, float] = _NEUTRAL_COLOR
        self._snap_marker_kind: int = 0

        # DRAGGING_HEIGHT data
        self._width: float = 0.0
        self._depth: float = 0.0
        self._base_center: np.ndarray | None = None  # local coords, Z=0
        self._footprint_local: np.ndarray | None = None  # (4, 3) local coords
        self._current_height: float = 0.0

    # ---- option fields, shared by every subclass (unused ones are simply
    # never surfaced by that subclass's option bar) --------------------

    @property
    def segments(self) -> int:
        return self._segments

    @segments.setter
    def segments(self, value: int) -> None:
        self._segments = max(SEGMENTS_FLOOR, int(value))

    @property
    def rings(self) -> int:
        return self._rings

    @rings.setter
    def rings(self, value: int) -> None:
        self._rings = max(RINGS_FLOOR, int(value))

    # ---- Tool ABC ------------------------------------------------------

    @property
    def shortcut(self) -> str:
        return ""  # No shortcut ships with any primitive tool (M7.4 Task 5, spec D9).

    @property
    def has_active_gesture(self) -> bool:
        return self._state != _State.IDLE

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None  # Primitive tools don't drive axis-lock.

    @property
    def status_text(self) -> str | None:
        if (
            self._state == _State.DRAGGING_FOOTPRINT
            and self._first_corner is not None
            and self._preview_corner is not None
        ):
            w = abs(float(self._preview_corner[0]) - float(self._first_corner[0]))
            d = abs(float(self._preview_corner[1]) - float(self._first_corner[1]))
            return self._format_pair(w, d)
        if self._state == _State.DRAGGING_HEIGHT:
            return self._format_single("height", self._current_height)
        return None

    def _format_pair(self, w: float, d: float) -> str:
        if self._units_provider is not None:
            from bermake.units import format_length

            u = self._units_provider()
            return f"{format_length(w, u)} x {format_length(d, u)}"
        return f"{w:.3f} x {d:.3f}"

    def _format_single(self, label: str, value: float) -> str:
        if self._units_provider is not None:
            from bermake.units import format_length

            return f"{label}: {format_length(value, self._units_provider())}"
        return f"{label}: {value:.3f}"

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

    # ---- Event handlers -------------------------------------------------

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if self._state == _State.DRAGGING_HEIGHT:
            self._update_height_from_event(event)
            return

        from bermake.viewport.snap_engine import SnapKind, marker_color

        if snap.kind == SnapKind.NONE:
            self._snap_marker_pos = None
            self._snap_marker_kind = 0
            return
        self._snap_marker_pos = snap.world_position.copy()
        self._snap_marker_color = marker_color(snap, _NEUTRAL_COLOR)
        self._snap_marker_kind = int(snap.kind)
        if self._state == _State.DRAGGING_FOOTPRINT:
            self._preview_corner = snap.world_position.copy()

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        from bermake.viewport.snap_engine import SnapKind

        if self._state == _State.IDLE:
            if snap.kind == SnapKind.NONE:
                return
            self._first_corner = snap.world_position.copy()
            self._preview_corner = snap.world_position.copy()
            self._state = _State.DRAGGING_FOOTPRINT
            return

        if self._state == _State.DRAGGING_FOOTPRINT:
            if snap.kind == SnapKind.NONE:
                return
            second = snap.world_position
            if np.array_equal(second, self._first_corner):
                self._reset_to_idle()
                return
            self._commit_footprint(second)
            return

        # DRAGGING_HEIGHT: commit if height clears the min threshold, else cancel.
        if self._current_height >= _MIN_HEIGHT:
            self._commit_primitive(
                width=self._width, depth_=self._depth, height=self._current_height
            )
        self._reset_to_idle()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() != Qt.Key.Key_Escape:
            return
        if self._state != _State.IDLE:
            # Cancel -- no command pushed, scene was never mutated in either
            # dragging stage (the commit only happens on the final click).
            self._reset_to_idle()

    def overlay(self) -> ToolOverlay:
        segments = np.zeros((0, 3), dtype=np.float32)
        polygons: list[np.ndarray] = []

        if (
            self._state == _State.DRAGGING_FOOTPRINT
            and self._first_corner is not None
            and self._preview_corner is not None
        ):
            # Resolve in the same frame _commit_footprint will actually use --
            # the active context's local ground plane (local Z=0) -- rather
            # than a literal world Z=0, then transform back to world for
            # display. Inside a group translated in Z, local Z=0 is not world
            # Z=0, so a hardcoded 0.0 here would draw the rubber band on a
            # different plane than where the primitive is about to land.
            wt = self._world_transform()
            p0 = world_to_local_point(self._first_corner, wt)
            p1 = world_to_local_point(self._preview_corner, wt)
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            segments = np.array(
                [
                    [x0, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y0, 0.0],
                    [x1, y1, 0.0],
                    [x1, y1, 0.0],
                    [x0, y1, 0.0],
                    [x0, y1, 0.0],
                    [x0, y0, 0.0],
                ],
                dtype=np.float64,
            )
            if wt is not None and not is_identity_transform(wt):
                segments = apply_mat(segments, wt)
            segments = segments.astype(np.float32)
        elif self._state == _State.DRAGGING_HEIGHT:
            polygons = self._build_ghost_polygons()
            wt = self._world_transform()
            if wt is not None and not is_identity_transform(wt):
                polygons = [
                    apply_mat(np.asarray(p, np.float64), wt).astype(np.float32) for p in polygons
                ]

        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_NEUTRAL_COLOR,
            snap_marker_position=(
                self._snap_marker_pos.copy() if self._snap_marker_pos is not None else None
            ),
            snap_marker_color=self._snap_marker_color,
            snap_marker_kind=self._snap_marker_kind,
            face_fill_polygons=polygons,
            face_fill_color=_GHOST_FILL_COLOR,
        )

    # ---- Helpers -------------------------------------------------------

    def _commit_footprint(self, second) -> None:
        """Normalize the two dragged corners into local width/depth/center
        and enter DRAGGING_HEIGHT, or cancel back to IDLE if the footprint
        is degenerate (a zero-area drag -- e.g. a straight-line drag)."""
        assert self._first_corner is not None
        wt = self._world_transform()
        p0 = world_to_local_point(self._first_corner, wt)
        p1 = world_to_local_point(second, wt)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        xlo, xhi = min(x0, x1), max(x0, x1)
        ylo, yhi = min(y0, y1), max(y0, y1)
        width, depth = xhi - xlo, yhi - ylo
        if width < _MIN_FOOTPRINT or depth < _MIN_FOOTPRINT:
            self._reset_to_idle()
            return

        self._width = width
        self._depth = depth
        self._base_center = np.array([(xlo + xhi) / 2.0, (ylo + yhi) / 2.0, 0.0], dtype=np.float64)
        self._footprint_local = np.array(
            [[xlo, ylo, 0.0], [xhi, ylo, 0.0], [xhi, yhi, 0.0], [xlo, yhi, 0.0]],
            dtype=np.float64,
        )
        self._current_height = 0.0
        self._state = _State.DRAGGING_HEIGHT

    def _update_height_from_event(self, event: QMouseEvent) -> None:
        """Line-line CPA between the camera ray and the vertical line
        (base_center, +local Z) -- the same math PushPullTool uses against
        a face's own normal, anchored here at the footprint's center along
        local +Z instead. Holds the previous height if the view is
        ~parallel to that line (degenerate case)."""
        if self._camera is None or self._widget_size_provider is None or self._base_center is None:
            return
        pos = event.position()
        width, height = self._widget_size_provider()
        origin, direction = self._camera.ray_from_screen(
            float(pos.x()), float(pos.y()), int(width), int(height)
        )
        origin, direction = ray_into_local(origin, direction, self._world_transform())
        d_norm = float(np.linalg.norm(direction))
        if d_norm < 1e-9:
            return
        d_hat = direction / d_norm
        n_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        c = self._base_center

        b = float(np.dot(d_hat, n_hat))
        denom = 1.0 - b * b
        if abs(denom) < _DEGENERATE_VIEW_EPSILON:
            return  # height frozen
        w = origin.astype(np.float64) - c
        e = float(np.dot(n_hat, w))
        d_param = float(np.dot(d_hat, w))
        t = (e - b * d_param) / denom
        self._current_height = max(0.0, t)

    def _build_ghost_polygons(self) -> list[np.ndarray]:
        """Return [base, top, *sides] -- a bounding-prism preview shared by
        all four primitives. What each generates internally (a box's flat
        faces, a cylinder's round ones, ...) is exactly what's about to be
        created; the bounding box the user dragged is what's still being
        adjusted (the height), so that's what's shown moving."""
        assert self._footprint_local is not None
        base = self._footprint_local.astype(np.float32)
        top = base + np.array([0.0, 0.0, self._current_height], dtype=np.float32)
        polygons: list[np.ndarray] = [base, top]
        n = base.shape[0]
        for i in range(n):
            j = (i + 1) % n
            side = np.stack([base[i], base[j], top[j], top[i]]).astype(np.float32)
            polygons.append(side)
        return polygons

    def _placement_transform(self) -> np.ndarray:
        """A pure translation to the dragged footprint's center. No scale is
        needed: `_make_mesh` calls its generator with the actual requested
        dimensions, not a unit shape meant to be scaled up afterward."""
        t = np.eye(4, dtype=np.float64)
        if self._base_center is not None:
            t[:3, 3] = self._base_center
        return t

    def _commit_primitive(self, *, width: float, depth_: float, height: float) -> None:
        """Generate this tool's mesh at the given size and insert it into
        the scene as one undo step.

        `depth_` carries a trailing underscore because `depth` is already
        the extrusion term Push/Pull uses elsewhere in this package -- two
        different meanings under one name in the same package would be a
        trap.

        The gesture (`_commit_footprint` / the DRAGGING_HEIGHT threshold in
        `on_mouse_press`) keeps a degenerate dimension from ever reaching
        this method during normal use. Called directly with one anyway
        (bypassing the gesture, as in a test), it can still raise -- from
        `_make_mesh`'s generator, or from `build_mesh_into_scene` finding
        coincident vertices where the mesh collapsed on one axis. Either
        way, nothing is left in the scene: `build_mesh_into_scene` rolls
        back whatever it already applied before re-raising, and in the
        generator-raises case nothing was ever built or pushed in the
        first place.

        This method deliberately lets that `ValueError` propagate rather
        than catching and reporting it: there is no reachable path from the
        UI to a degenerate dimension (the gesture guards above are the only
        way width/depth/height are ever set before a real commit), so a
        user can never actually see this exception. Catching it here would
        only mask a bug in those guards, were one ever introduced --
        `test_a_degenerate_dimension_raises_and_leaves_the_scene_untouched`
        in `tests/test_primitive_tools.py` pins this choice.
        """
        scene = self._scene
        mesh = self._make_mesh(width=width, depth_=depth_, height=height)
        composite = CompositeCommand(name=self.name)
        composite.children.extend(
            build_mesh_into_scene(mesh, scene, transform=self._placement_transform())
        )
        self._command_stack.push_executed(composite, scene)

    def _reset_to_idle(self) -> None:
        self._state = _State.IDLE
        self._first_corner = None
        self._preview_corner = None
        self._snap_marker_pos = None
        self._snap_marker_kind = 0
        self._width = 0.0
        self._depth = 0.0
        self._base_center = None
        self._footprint_local = None
        self._current_height = 0.0

    @abstractmethod
    def _make_mesh(self, *, width: float, depth_: float, height: float):
        """Call this tool's `bermake._core.make_*` generator and return the
        resulting `HalfEdgeMesh`. `width`/`depth_` are the dragged
        footprint's extents; `height` is the dragged height."""


class BoxTool(PrimitiveTool):
    """Draw a box: the dragged footprint and height map directly onto
    `make_box`'s width/depth/height -- no derived radius, unlike the three
    round primitives below."""

    @property
    def name(self) -> str:
        return "Box"

    @property
    def id(self) -> str:
        return "box"

    def _make_mesh(self, *, width: float, depth_: float, height: float):
        return make_box(width=width, depth=depth_, height=height)


class CylinderTool(PrimitiveTool):
    """Draw a cylinder: its circular base is inscribed in the dragged
    footprint (`min(width, depth) / 2`), so it never spills past what was
    dragged even when the drag wasn't square."""

    @property
    def name(self) -> str:
        return "Cylinder"

    @property
    def id(self) -> str:
        return "cylinder"

    def _make_mesh(self, *, width: float, depth_: float, height: float):
        radius = min(width, depth_) / 2.0
        return make_cylinder(radius=radius, height=height, segments=self.segments)


class ConeTool(PrimitiveTool):
    """Draw a cone. Same inscribed-radius rule as CylinderTool."""

    @property
    def name(self) -> str:
        return "Cone"

    @property
    def id(self) -> str:
        return "cone"

    def _make_mesh(self, *, width: float, depth_: float, height: float):
        radius = min(width, depth_) / 2.0
        return make_cone(radius=radius, height=height, segments=self.segments)


class SphereTool(PrimitiveTool):
    """Draw a sphere. A sphere has one radius, not an independent
    width/depth/height, so this uses the smallest of the three dragged
    extents -- the sphere then always fits inside the box that was dragged,
    rather than spilling past whichever extent was smallest."""

    @property
    def name(self) -> str:
        return "Sphere"

    @property
    def id(self) -> str:
        return "sphere"

    def _make_mesh(self, *, width: float, depth_: float, height: float):
        radius = min(width, depth_, height) / 2.0
        return make_sphere(radius=radius, rings=self.rings, segments=self.segments)
