"""The Select tool (Spacebar).

Hover pre-highlights the entity under the cursor. Click replaces the selection;
Shift-click toggles; clicking empty space clears. Box-select (drag a rectangle)
is added in M4b Task 8. Esc clears the selection.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent

from bermake.annotations.picking import pick_annotation
from bermake.tools.tool import Tool, ToolContext, ToolOverlay
from bermake.units import Units
from bermake.viewport.picking import pick_selectable

_HOVER_EDGE_COLOR = (0.45, 0.70, 1.00)
_HOVER_FILL_COLOR = (0.40, 0.70, 1.00, 0.18)
_NEUTRAL_COLOR = (0.85, 0.85, 0.85)
_BOX_WINDOW_COLOR = (0.25, 0.50, 0.95)  # left->right, enclose-only
_BOX_CROSSING_COLOR = (0.15, 0.65, 0.30)  # right->left, touch
_DRAG_THRESHOLD_PX = 4.0
_HOVER_BBOX_COLOR = (0.60, 0.78, 1.00)  # Task 15: lighter blue for hover silhouette bbox


class SelectTool(Tool):
    @property
    def name(self) -> str:
        return "Select"

    @property
    def shortcut(self) -> str:
        return "Space"

    @property
    def id(self) -> str:
        return "select"

    def __init__(self) -> None:
        self._scene = None
        self._camera = None
        self._size_provider = None
        self._selection = None
        self._model = None
        self._stack = None  # M7d Task 12 — bermake.commands.CommandStack (or None)
        self._units_provider = None  # M7d — callable () -> bermake.units.Units (or None)
        self._request_rebuild = None  # M4e — callable () -> None
        self._show_guides_provider = None  # M7.6b Task 7 fix round 1 -- callable () -> bool
        self._select_vertices_provider = None  # M7.6c Task 7 -- callable () -> bool
        self._hovered: tuple[str, int] | None = None
        self._hovered_instance = None  # M4e — Instance | None (for Task 15 silhouette)
        self._hovered_annotation: int | None = None  # M7d — annotation id under the cursor
        self._press_px: tuple[float, float] | None = None
        self._is_box = False
        self._box_rect: tuple[float, float, float, float] | None = None
        self._box_window = True  # True = L->R window, False = R->L crossing
        # M4e — set after double-click to eat the trailing release
        self._suppress_next_release = False

    def activate(self, ctx: ToolContext) -> None:
        self._scene = ctx.scene
        self._camera = ctx.camera
        self._size_provider = ctx.widget_size_provider
        self._selection = ctx.selection
        self._model = ctx.model
        self._stack = ctx.command_stack
        self._units_provider = ctx.units_provider
        self._request_rebuild = ctx.request_context_rebuild
        self._show_guides_provider = ctx.show_guides_provider
        self._select_vertices_provider = ctx.select_vertices_provider
        self._hovered = None
        self._hovered_instance = None
        self._hovered_annotation = None
        self._press_px = None
        self._is_box = False
        self._box_rect = None
        self._suppress_next_release = False

    def _world_transform(self):
        return self._model.active_world_transform if self._model is not None else None

    def _select_vertices(self) -> bool:
        """M7.6c: None reads as False, the viewport's own default, so a bare
        test ToolContext keeps pre-M7.6c picking."""
        provider = self._select_vertices_provider
        return bool(provider()) if provider is not None else False

    def _units(self) -> Units:
        # M7d: every other units provider in this codebase (wall/opening/roof
        # options bars, the annotation painter) always yields a real Units
        # object -- None is not a value format_length expects.
        return self._units_provider() if self._units_provider is not None else Units()

    def _pick_annotation(self, cx: float, cy: float, w: int, h: int) -> int | None:
        """M7d: annotation hit-test scoped to the active context, mirroring how
        pick_selectable/pick_instance already scope to it.

        Fix round 1: a hidden guide (View > Guides off) must not be pickable
        either -- `show_guides_provider` is None readable as "visible",
        matching ViewportWidget.show_guides's own default, so a bare test
        ToolContext with no provider wired keeps today's behaviour."""
        if self._model is None or self._camera is None:
            return None
        show_guides = (
            self._show_guides_provider() if self._show_guides_provider is not None else True
        )
        return pick_annotation(
            (cx, cy),
            self._model.active_context.annotations,
            self._model.active_world_transform,
            self._camera,
            w,
            h,
            self._units(),
            show_guides=show_guides,
        )

    def deactivate(self) -> None:
        self._hovered = None
        self._reset_press()

    def _viewport_size(self) -> tuple[int, int]:
        if self._size_provider is None:
            return (1, 1)
        return self._size_provider()

    def _cursor(self, event: QMouseEvent) -> tuple[float, float]:
        pos = event.position()
        return (float(pos.x()), float(pos.y()))

    def on_mouse_move(self, event: QMouseEvent, snap) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton and self._press_px is not None:
            cx, cy = self._cursor(event)
            px, py = self._press_px
            if (
                self._is_box
                or abs(cx - px) >= _DRAG_THRESHOLD_PX
                or abs(cy - py) >= _DRAG_THRESHOLD_PX
            ):
                self._is_box = True
                self._box_rect = (px, py, cx, cy)
                self._box_window = (cx - px) >= 0.0
            return
        # Final review I2: hover honours View > Select Vertices for the same
        # reason the click pick does -- pre-highlight is the promise the
        # click then keeps, so the two must pick the same entity.
        self._hovered = pick_selectable(
            self._cursor(event),
            self._viewport_size(),
            self._camera,
            self._scene,
            world_transform=self._world_transform(),
            select_vertices=self._select_vertices(),
        )
        # M7d: also track the hovered annotation (drawn on top, so hover-picked first)
        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        self._hovered_annotation = self._pick_annotation(cx, cy, w, h)
        # M4e: also track hovered instance for Task 15 silhouette rendering
        if self._model is not None and self._camera is not None:
            cx, cy = self._cursor(event)
            w, h = self._viewport_size()
            origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
            self._hovered_instance = self._model.pick_instance(origin, direction)
        else:
            self._hovered_instance = None

    def on_mouse_press(self, event: QMouseEvent, snap) -> None:
        self._press_px = self._cursor(event)
        self._is_box = False
        self._box_rect = None

    def on_mouse_release(self, event: QMouseEvent, snap) -> None:
        # M4e: suppress the trailing release after a double-click enter
        if self._suppress_next_release:
            self._suppress_next_release = False
            self._reset_press()
            return
        if self._selection is None:
            self._reset_press()
            return
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._is_box and self._box_rect is not None:
            from bermake.viewport.picking import entities_in_box

            mode = "window" if self._box_window else "crossing"
            edges, faces, vertices = entities_in_box(
                self._box_rect,
                mode,
                self._viewport_size(),
                self._camera,
                self._scene,
                world_transform=self._world_transform(),
                select_vertices=self._select_vertices(),
            )
            if shift:
                self._selection.add(edges=edges, faces=faces, vertices=vertices)
            else:
                self._selection.replace(edges=edges, faces=faces, vertices=vertices)
        else:
            cx, cy = self._cursor(event)
            w, h = self._viewport_size()
            # M7d: annotations draw on top, so they win the click before the
            # instance/geometry pick below ever runs.
            ann_id = self._pick_annotation(cx, cy, w, h)
            if ann_id is not None:
                if shift:
                    self._selection.toggle_annotation(ann_id)
                else:
                    self._selection.replace(annotations=[ann_id])
                self._reset_press()
                return
            # M4e: try instance pick first (only if we have a model + camera)
            if self._model is not None and self._camera is not None:
                origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
                inst = self._model.pick_instance(origin, direction)
                if inst is not None:
                    if shift:
                        self._selection.toggle_instance(inst.id)
                    else:
                        self._selection.replace(instances=[inst.id])
                    self._reset_press()
                    return
            # Fall through to entity pick
            # Final review I2: the single-click pick has to honour View >
            # Select Vertices too. Without the flag a plain click on a corner
            # selected the edge while a double-click at the same pixel
            # selected the vertex, so the mode was reachable by drag and by
            # double-click but not by the most obvious gesture of all.
            hit = pick_selectable(
                self._cursor(event),
                self._viewport_size(),
                self._camera,
                self._scene,
                world_transform=self._world_transform(),
                select_vertices=self._select_vertices(),
            )
            if hit is None:
                if not shift:
                    # M4e: inside a group, empty-click exits one level; at root clear selection
                    if self._model is not None and self._model.active_path:
                        self._exit_one()
                    else:
                        self._selection.clear()
            elif hit[0] == "vertex":
                # Final review I2: an explicit branch, not a fall-through.
                # The `else` below used to be the face branch, so a
                # ("vertex", id) hit would have put a vertex id into
                # `selection.faces` -- ids are per-kind here, so that id
                # names a real and unrelated face often enough to matter.
                if shift:
                    self._selection.toggle_vertex(hit[1])
                else:
                    self._selection.replace(vertices=[hit[1]])
            elif hit[0] == "edge":
                if shift:
                    self._selection.toggle_edge(hit[1])
                else:
                    self._selection.replace(edges=[hit[1]])
            else:
                if shift:
                    self._selection.toggle_face(hit[1])
                else:
                    self._selection.replace(faces=[hit[1]])
        self._reset_press()

    def _reset_press(self) -> None:
        self._press_px = None
        self._is_box = False
        self._box_rect = None
        self._box_window = True

    def on_mouse_double_click(self, event: QMouseEvent, snap) -> None:
        """Double-click a label to reopen its text prompt; a dimension has no
        stored text and does nothing. Otherwise double-click an instance to
        enter it (group/component open for editing). Failing both, double-click
        raw geometry to smart-select it with its immediate neighbours.

        The order is load-bearing and matches SketchUp: a group is entered by
        double-clicking it, and raw geometry is what you double-click once you
        are inside. Putting the geometry branch first would make a group
        impossible to enter wherever a face sits under the cursor, which is
        everywhere.

        Only the instance branch needs `self._model`; `_pick_annotation`
        already tolerates a None model (a bare test ToolContext has none),
        and the geometry branch needs neither, so the method is no longer
        gated on model up front.
        """
        if self._camera is None:
            return
        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        # M7d Task 12: annotations draw on top, so they win the double-click
        # before the instance pick below ever runs (same order as on_mouse_release).
        ann_id = self._pick_annotation(cx, cy, w, h)
        if ann_id is not None:
            self._edit_annotation_text(ann_id)
            return
        if self._model is not None:
            origin, direction = self._camera.ray_from_screen(cx, cy, w, h)
            inst = self._model.pick_instance(origin, direction)
            if inst is not None:
                self._model.enter(inst)
                self._enter_or_exit_cleanup()
                self._suppress_next_release = True
                return
        self._smart_select(event, cx, cy, w, h)

    def _smart_select(self, event: QMouseEvent, cx: float, cy: float, w: int, h: int) -> None:
        """M7.6c: a double-click on raw geometry selects the entity plus its
        immediate neighbours, one dimension up and down.

        Face gives the face and its bounding edges; edge gives the edge and
        its adjacent faces. Nothing under the cursor leaves the selection
        alone rather than clearing it, because a double-click that missed is
        far more often a mis-aim than an intent to deselect.
        """
        from bermake.selection_ops import adjacent_faces, bounding_edges, incident_edges

        hit = self._pick_geometry(cx, cy, w, h)
        if hit is None:
            return
        kind, ent_id = hit
        vertices: set[int] = set()
        if kind == "face":
            faces = {int(ent_id)}
            edges = bounding_edges(self._scene, faces)
        elif kind == "edge":
            edges = {int(ent_id)}
            faces = adjacent_faces(self._scene, edges)
        elif kind == "vertex":
            # Final review C1: every branch now computes and falls through to
            # the one shared call, rather than the vertex branch applying and
            # returning on its own. That uniformity is what lets the trailing
            # release be suppressed by a single assignment inside
            # `_apply_smart_selection`.
            vertices = {int(ent_id)}
            edges = incident_edges(self._scene, vertices)
            faces = set()
        else:
            return
        self._apply_smart_selection(event, edges=edges, faces=faces, vertices=vertices)

    def on_mouse_triple_click(self, event: QMouseEvent, snap) -> None:
        """M7.6c: select everything connected to the entity under the cursor.

        Scoped to the active context's mesh, like every other pick here: the
        flood neither descends into a nested instance nor escapes to the
        parent.
        """
        if self._camera is None or self._scene is None:
            return
        from bermake.selection_ops import connected_component

        cx, cy = self._cursor(event)
        w, h = self._viewport_size()
        hit = self._pick_geometry(cx, cy, w, h)
        if hit is None:
            return
        verts, edges, faces = connected_component(self._scene, self._seed_vertices(hit))
        if not self._select_vertices():
            verts = set()
        if not (verts or edges or faces):
            # Final review M9: an isolated vertex (live, but with no incident
            # edge -- a state a bare vertex reaches when its edge is removed)
            # floods to three empty sets, and applying that would wipe the
            # selection. A triple-click that found nothing to flood leaves the
            # selection alone, matching what a double-click that missed does.
            return
        self._apply_smart_selection(event, edges=edges, faces=faces, vertices=verts)

    def _pick_geometry(self, cx: float, cy: float, w: int, h: int):
        """pick_selectable against the active context, or None."""
        if self._scene is None or self._camera is None:
            return None
        return pick_selectable(
            (cx, cy),
            (w, h),
            self._camera,
            self._scene,
            world_transform=self._world_transform(),
            select_vertices=self._select_vertices(),
        )

    def _seed_vertices(self, hit) -> set[int]:
        """The vertices a flood starts from, for whichever kind was picked."""
        kind, ent_id = hit
        try:
            if kind == "face":
                return {int(v) for v in self._scene.face_loop(ent_id)}
            if kind == "edge":
                e = self._scene.edge(ent_id)
                return {int(e.v1_id), int(e.v2_id)}
            if kind == "vertex":
                return {int(ent_id)}
        except KeyError:
            return set()
        return set()

    def _apply_smart_selection(
        self, event: QMouseEvent, *, edges, faces, vertices=frozenset()
    ) -> None:
        """Replace, or add when Shift is held, matching single-click and
        box-select.

        Final review C1: this also eats the gesture's trailing release, the
        way the instance-enter branch of `on_mouse_double_click` already
        does. Qt's real double-click sequence is Press, Release, DblClick,
        Release and `ViewportWidget.mouseReleaseEvent` dispatches
        `on_mouse_release` on every one of those, so without this the release
        after a smart-select ran the ordinary single-click pick and replaced
        the selection smart-select had just built, making the whole feature
        invisible. The triple-click's own trailing release is the same story.

        Setting it here rather than at each call site is deliberate: this is
        the single point every smart-select path funnels through, and it
        fires only when a selection was actually applied, so a double-click
        that missed still leaves the trailing release to behave normally.
        """
        if self._selection is None:
            return
        self._suppress_next_release = True
        if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._selection.add(edges=edges, faces=faces, vertices=vertices)
        else:
            self._selection.replace(edges=edges, faces=faces, vertices=vertices)

    def prompt_text(self, default: str = "") -> str | None:
        """Ask the user for a label's new text. Overridable for testing
        (mirrors TextTool.prompt_text so it can be stubbed the same way)."""
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(None, "Text", "Label:", text=default)
        return text if ok else None

    def _edit_annotation_text(self, ann_id: int) -> None:
        """M7d Task 12: reopen the prompt pre-filled for a label and commit
        an EditLabelTextCommand. A dimension's text is derived from its
        geometry and has nothing stored to edit, so this is a no-op."""
        ann = None
        for a in self._model.active_context.annotations:
            if a.id == ann_id:
                ann = a
                break
        if ann is None or getattr(ann, "kind", None) != "label":
            return
        text = self.prompt_text(ann.text)
        if text is None or not text.strip():
            return
        if self._stack is None:
            return
        from bermake.commands.annotation_commands import EditLabelTextCommand

        self._stack.execute(
            EditLabelTextCommand(ann_id, text.strip(), self._model.active_context),
            self._model,
        )

    def _enter_or_exit_cleanup(self) -> None:
        """Clear selection and trigger a context rebuild after enter/exit."""
        if self._selection is not None:
            self._selection.clear()
        if self._request_rebuild is not None:
            self._request_rebuild()

    def _exit_one(self) -> None:
        """Exit one level of group nesting and rebuild the tool context."""
        if self._model is not None:
            self._model.exit_one()
            self._enter_or_exit_cleanup()

    def on_key_press(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            # M4e: Esc inside a group exits one level; at root it clears selection
            if self._model is not None and self._model.active_path:
                self._exit_one()
            elif self._selection is not None:
                self._selection.clear()

    def overlay(self) -> ToolOverlay:
        box_rect = self._box_rect if self._is_box else None
        box_color = _BOX_WINDOW_COLOR if self._box_window else _BOX_CROSSING_COLOR
        segs = np.zeros((0, 3), dtype=np.float32)
        fills: list[np.ndarray] = []
        if not self._is_box and self._hovered is not None and self._scene is not None:
            kind, ent_id = self._hovered
            # Final review I2: hover can now report a vertex, and the face
            # arm below used to be a bare `else`, so a vertex id would have
            # been handed to `face_loop`. Ids are allocated per kind, so that
            # id usually names a real face and the hover highlight would have
            # lit an unrelated polygon. Naming the face arm explicitly means
            # a vertex (and any future kind) simply draws no preview here --
            # the selected-vertex glyph pass is the only vertex chrome.
            if kind == "edge":
                try:
                    e = self._scene.edge(ent_id)
                    p1 = np.asarray(self._scene.vertex(e.v1_id).position, dtype=np.float32)
                    p2 = np.asarray(self._scene.vertex(e.v2_id).position, dtype=np.float32)
                    segs = np.array([p1, p2], dtype=np.float32)
                except KeyError:
                    pass
            elif kind == "face":
                try:
                    from bermake.geometry.transforms import apply_mat, is_identity_transform

                    wt = self._world_transform()
                    use_wt = wt is not None and not is_identity_transform(wt)
                    wt_arr = np.asarray(wt, dtype=np.float64) if use_wt else None

                    def _to_world_sel(local_pos: np.ndarray) -> np.ndarray:
                        if not use_wt:
                            return local_pos
                        return apply_mat(local_pos.reshape(1, 3), wt_arr)[0]

                    loop = self._scene.face_loop(ent_id)
                    fills = [
                        np.array(
                            [
                                _to_world_sel(
                                    np.asarray(self._scene.vertex(v).position, dtype=np.float32)
                                )
                                for v in loop
                            ],
                            dtype=np.float32,
                        )
                    ]
                except KeyError:
                    pass

        # Task 15: hover silhouette — draw the hovered instance's bbox as a
        # lighter-blue world polyline so it renders via the existing overlay path.
        world_polylines: list = []
        if not self._is_box and self._hovered_instance is not None and self._model is not None:
            aabb = self._hovered_instance.definition.local_aabb()
            if aabb is not None:
                from bermake.viewport.scene_renderer import aabb_world_edges

                lo, hi = aabb
                active_world = self._model.active_world_transform
                world_t = active_world @ self._hovered_instance.transform
                bbox_segs = aabb_world_edges(lo, hi, world_t)
                world_polylines.append((bbox_segs, _HOVER_BBOX_COLOR, 1.5))

        return ToolOverlay(
            rubber_band_segments=segs,
            rubber_band_color=_HOVER_EDGE_COLOR,
            snap_marker_position=None,
            snap_marker_color=_NEUTRAL_COLOR,
            snap_marker_kind=0,
            face_fill_polygons=fills,
            face_fill_color=_HOVER_FILL_COLOR,
            box_rect=box_rect,
            box_rect_color=box_color,
            box_rect_dashed=not self._box_window,
            world_polylines=world_polylines,
            hovered_annotation_id=self._hovered_annotation if not self._is_box else None,
        )

    @property
    def has_active_gesture(self) -> bool:
        if self._is_box:
            return True
        return self._selection is not None and not self._selection.is_empty()

    @property
    def is_box_selecting(self) -> bool:
        """True only while a box-select drag is actually in progress.

        has_active_gesture is deliberately broader -- it also reports True
        for a merely non-empty selection, which is what lets Esc clear one.
        Callers that mean "mid gesture in the click sense" want this.
        """
        return self._is_box

    @property
    def anchor_or_none(self) -> np.ndarray | None:
        return None

    @property
    def status_text(self) -> str | None:
        return None
