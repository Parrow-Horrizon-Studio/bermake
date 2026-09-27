"""The Follow Me tool -- SketchUp-style profile sweep along a path.

Unlike Push/Pull and Offset, the path is chosen BEFORE this tool is armed:
Select preselects a chain of edges, the tool is armed (no shortcut -- M7.4
Task 5's id-keyed registry is what makes that legal), and the single click
that follows names the profile face. Preselecting the path first is what
makes the gesture unambiguous, and matches SketchUp's own Follow Me.

Path ordering is this tool's own job (`_order_path`): the selection arrives
as an unordered set of edge ids, which has to be walked into a chain before
`sweep_stations` (M7.4 Task 7) can do anything with it. A selection that
does not form a single unbranched chain -- a fork, or two disjoint runs --
is refused with a status-bar message, the same way `sweep_stations` refuses
a corner too tight for the profile (`SweepRefused`): both are reported, not
raised past this tool, and neither mutates the scene.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
from PySide6.QtGui import QMouseEvent

from bermake.commands import CompositeCommand
from bermake.commands.scene_commands import RemoveFaceCommand, TransformVerticesCommand
from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.tools.sweep_support import SweepRefused, loft_between_loops, sweep_stations
from bermake.tools.tool import Tool, ToolContext, ToolOverlay

_HOVER_FILL_COLOR = (0.40, 0.70, 1.00, 0.20)  # light blue, matching Push/Pull's hover
_PATH_COLOR = (0.95, 0.55, 0.15)  # orange -- distinct from the profile hover highlight

_REFUSAL_MESSAGE = "Select one unbranched path, then click a face."


class _State(Enum):
    IDLE = 0
    HOVERING = 1


class FollowMeTool(Tool):
    """Sweep a preselected path's profile face along it."""

    def __init__(self) -> None:
        self._scene = None
        self._command_stack = None
        self._camera = None
        self._widget_size_provider = None
        self._model = None
        self._selection = None
        self._status_bar = None

        self._state: _State = _State.IDLE
        self._hovered_face_id: int | None = None

        # Side effect of a successful _order_path call: whether the chain it
        # found wraps back on itself. sweep_stations needs to know before
        # _commit_sweep can call it.
        self._path_closed: bool = False

    # ---- Tool ABC ------------------------------------------------------

    @property
    def name(self) -> str:
        return "Follow Me"

    @property
    def shortcut(self) -> str:
        return ""  # No shortcut ships with this tool (M7.4 Task 5, spec D9).

    @property
    def id(self) -> str:
        return "follow_me"

    @property
    def has_active_gesture(self) -> bool:
        return False  # A single click commits; there is no gesture for ESC to cancel.

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None  # Follow Me doesn't drive axis-lock.

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._command_stack = ctx.command_stack
        self._camera = ctx.camera
        self._widget_size_provider = ctx.widget_size_provider
        self._model = ctx.model
        self._selection = ctx.selection
        self._reset_to_idle()

    def deactivate(self) -> None:
        self._reset_to_idle()

    def set_status_bar(self, status_bar) -> None:
        """Wired once by MainWindow, outside ToolContext.

        Every other tool reports through a MainWindow handler method, which
        already has the status bar in scope. Follow Me is the first tool
        that has to report a refusal from inside its own commit path (a fork
        in the preselection, or a corner too tight for the profile), so it
        needs a standing reference of its own rather than borrowing one from
        a call it doesn't make.
        """
        self._status_bar = status_bar

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    # ---- Event handlers -----------------------------------------------

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        hit = self._pick_face_under_cursor(event)
        if hit is None:
            self._state = _State.IDLE
            self._hovered_face_id = None
        else:
            self._state = _State.HOVERING
            self._hovered_face_id = hit.face_id

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        if self._state != _State.HOVERING or self._hovered_face_id is None:
            return  # clicking empty space is a no-op
        self._commit_sweep(self._hovered_face_id)
        self._reset_to_idle()
        # Re-pick immediately so repeated sweeps don't need an intervening
        # mouse move to re-arm hovering, matching Push/Pull and Offset.
        hit = self._pick_face_under_cursor(event)
        if hit is not None:
            self._state = _State.HOVERING
            self._hovered_face_id = hit.face_id

    def overlay(self) -> ToolOverlay:
        polygons: list[np.ndarray] = []
        if self._state == _State.HOVERING and self._hovered_face_id is not None:
            polygons = [self._loop_world_coords(self._hovered_face_id)]

        segments = self._preselected_path_segments()

        wt = self._world_transform()
        if wt is not None and not is_identity_transform(wt):
            if polygons:
                polygons = [
                    apply_mat(np.asarray(p, np.float64), wt).astype(np.float32) for p in polygons
                ]
            if len(segments):
                segments = apply_mat(segments, wt).astype(np.float32)

        return ToolOverlay(
            rubber_band_segments=segments,
            rubber_band_color=_PATH_COLOR,
            snap_marker_position=None,
            snap_marker_color=(0.85, 0.85, 0.85),
            snap_marker_kind=0,
            face_fill_polygons=polygons,
            face_fill_color=_HOVER_FILL_COLOR,
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

    def _preselected_path_segments(self) -> np.ndarray:
        """The raw (unordered) preselected edges, as line segments for the
        overlay. Drawn as-is -- the overlay only needs to show what is
        selected, not what `_order_path` will make of it."""
        if self._scene is None or self._selection is None:
            return np.zeros((0, 3), dtype=np.float32)
        edges = [e for e in self._selection.edges if self._scene.edge_is_live(e)]
        if not edges:
            return np.zeros((0, 3), dtype=np.float32)
        segments = np.zeros((2 * len(edges), 3), dtype=np.float32)
        for i, e in enumerate(edges):
            edge = self._scene.edge(e)
            segments[2 * i] = self._scene.vertex(edge.v1_id).position
            segments[2 * i + 1] = self._scene.vertex(edge.v2_id).position
        return segments

    def _order_path(self, edge_ids: Sequence[int]) -> np.ndarray | None:
        """Walk an unordered edge selection into an ordered chain of points.

        Builds an adjacency map from each edge's two vertices, then walks
        it. A vertex touching three or more of the selected edges is a
        fork: refused (`None`). Two disjoint runs would each individually
        look fork-free, so connectivity is checked separately -- a single
        connected component must cover every selected edge. Within that
        component, exactly two degree-one vertices means an open path
        (walked end to end); zero means the chain closes on itself (walked
        from an arbitrary vertex all the way around). Anything else (e.g. a
        single isolated vertex from a self-loop edge, which `Scene.add_edge`
        already forbids, or some other degree pattern) is refused too.

        Sets `self._path_closed` as a side effect of a successful walk, since
        `sweep_stations` needs to know before `_commit_sweep` can call it.
        Returns the ordered chain's vertex positions (float64 (N, 3)) --
        exactly the shape `sweep_stations` wants as `path_points` -- or
        `None` for anything refused above.
        """
        edge_ids = [e for e in edge_ids if self._scene.edge_is_live(e)]
        if not edge_ids:
            return None

        adjacency: dict[int, list[int]] = {}
        for e in edge_ids:
            edge = self._scene.edge(e)
            adjacency.setdefault(edge.v1_id, []).append(edge.v2_id)
            adjacency.setdefault(edge.v2_id, []).append(edge.v1_id)

        if any(len(neighbors) > 2 for neighbors in adjacency.values()):
            return None  # a vertex touches 3+ selected edges: a fork

        # A single connected component must cover every selected edge -- two
        # disjoint runs would otherwise each pass the fork check on its own.
        start_probe = next(iter(adjacency))
        seen = {start_probe}
        stack = [start_probe]
        while stack:
            v = stack.pop()
            for n in adjacency[v]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        if seen != set(adjacency):
            return None  # more than one run

        degree_one = [v for v, neighbors in adjacency.items() if len(neighbors) == 1]
        if len(degree_one) == 2:
            self._path_closed = False
            start = degree_one[0]
        elif len(degree_one) == 0:
            self._path_closed = True
            start = start_probe
        else:
            return None  # degenerate: neither a simple path nor a simple cycle

        total = len(adjacency)
        order = [start]
        prev = None
        current = start
        while len(order) < total:
            candidates = [v for v in adjacency[current] if v != prev]
            if not candidates:
                return None  # unreachable given the checks above; defensive only
            nxt = candidates[0]
            order.append(nxt)
            prev, current = current, nxt

        return np.array([self._scene.vertex(v).position for v in order], dtype=np.float64)

    def _commit_sweep(self, profile_face_id: int) -> None:
        """Sweep `profile_face_id` along the current preselected path.

        Refuses (status bar message, no scene mutation) rather than raising
        for either failure mode: a preselection that doesn't walk into a
        single unbranched chain, or a corner `sweep_stations` finds too
        tight for this profile.
        """
        scene = self._scene
        path = self._order_path(list(self._selection.edges))
        if path is None:
            self._report(_REFUSAL_MESSAGE)
            return

        loop = list(scene.face_loop(profile_face_id))
        profile_pts = np.array([scene.vertex(v).position for v in loop], dtype=np.float64)

        try:
            stations = sweep_stations(profile_pts, path, closed=self._path_closed)
        except SweepRefused as exc:
            self._report(str(exc))
            return

        composite = CompositeCommand(name="Follow Me")
        rm = RemoveFaceCommand(profile_face_id)
        rm.do(scene)
        composite.children.append(rm)

        # Station 0 differs by path kind, and sweep_stations' relative-frame
        # contract is what makes the difference expressible at all.
        #
        # Open path: station 0 is the identity, so the profile as drawn IS
        # the first cross-section. It is left exactly where the user put it
        # (matching SketchUp), and only stations 1..N-1 mint new rings.
        #
        # Closed path: station 0 is the seam miter and must be applied, or
        # the closing segment terminates on a cross-section that does not
        # lie on the seam's own miter plane (its quads then cross each
        # other). The seam ring is materialised by MOVING the source loop's
        # own vertices onto it rather than minting a fresh ring: the source
        # face is being consumed either way, and reusing its vertices is
        # what keeps the tube free of both a stranded un-mitered wire loop
        # and a duplicate ring coincident with the seam.
        seam_pts = None
        if self._path_closed:
            seam_pts = apply_mat(profile_pts, stations[0])
            move = TransformVerticesCommand(
                {
                    vid: (scene.vertex(vid).position.copy(), seam_pts[i])
                    for i, vid in enumerate(loop)
                }
            )
            if not move.is_empty():
                move.do(scene)
                composite.children.append(move)

        current = loop
        last = len(stations) - 1
        for i in range(1, len(stations)):
            dst = apply_mat(profile_pts, stations[i])
            result = loft_between_loops(
                scene,
                current,
                dst,
                cap_start=(i == 1 and not self._path_closed),
                cap_end=(i == last and not self._path_closed),
            )
            composite.children.extend(result.commands)
            current = result.dst_vertex_ids

        if self._path_closed:
            # Close the tube back onto the seam ring built above -- the
            # profile's OWN original boundary loop, now sitting on the seam
            # miter plane. loft_between_loops mints destination vertices via
            # AddVertexCommand, which Scene.add_vertex makes idempotent on
            # exact float32 match (see scene.py's module docstring and
            # test_scene.py::test_add_vertex_is_idempotent_on_exact_match),
            # and Scene.set_vertex_position keeps that dedup index current
            # when a vertex moves. Handing back the SAME float32 array the
            # move wrote therefore welds this seam onto the source loop's
            # own vertex ids instead of minting a coincident duplicate ring
            # -- no separate stitching helper needed.
            result = loft_between_loops(
                scene,
                current,
                seam_pts,
                cap_start=False,
                cap_end=False,
            )
            composite.children.extend(result.commands)

        self._command_stack.push_executed(composite, scene)

    def _report(self, message: str) -> None:
        if self._status_bar is not None:
            self._status_bar.set_message(message)

    def _reset_to_idle(self) -> None:
        self._state = _State.IDLE
        self._hovered_face_id = None
