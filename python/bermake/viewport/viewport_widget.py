"""The 3D viewport widget — drives the scene + snap engine + active tool.

Owns a Camera (Python/numpy) and a SceneRenderer (GL resources). Translates
Qt mouse events into:
  * MMB drag         -> camera orbit (unchanged from M1)
  * Shift + MMB drag -> camera pan   (unchanged from M1)
  * Scroll wheel     -> camera zoom  (unchanged from M1)
  * LMB / cursor-move (when a tool is active) -> snap + delegate to tool
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent, QOpenGLContext, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from bermake.diagnostics.gl_check import MIN_GL_VERSION, GlInfo, evaluate_gl
from bermake.geometry.transforms import apply_mat, is_identity_transform, mat_invert
from bermake.tools.select_tool import _HOVER_EDGE_COLOR, SelectTool
from bermake.units import Units, format_coordinates
from bermake.viewport.camera import Camera
from bermake.viewport.inference import InferenceState, _default_now_ms
from bermake.viewport.scene_renderer import SceneRenderer
from bermake.viewport.snap_engine import SnapEngine, SnapKind

logger = logging.getLogger(__name__)

# M7.6b Task 7: annotation `kind`s that View > Guides hides/shows. Kept in
# sync with (but not imported from) annotation_painter._GUIDE_KINDS -- that
# one governs pen choice, this one governs whether a guide is painted at all.
_GUIDE_KINDS = frozenset({"guide", "guide_point"})


def _gl_string(name: int) -> str:
    value = GL.glGetString(name)
    return value.decode("utf-8", errors="replace") if value else "unknown"


class ViewportWidget(QOpenGLWidget):
    """The 3D viewport. Renders scene + active tool overlay; routes mouse events."""

    # M7.2 Task 14: emitted on a right-click release with the event's widget-local
    # pixel position; MainWindow resolves it into a right-click menu.
    context_menu_requested = Signal(int, int)

    # M7.9: emitted (from the event loop, not from inside initializeGL) with a
    # user-facing explanation when the context cannot run Bermake's shaders.
    gl_unavailable = Signal(str)

    def __init__(self, model=None, tool_manager=None, parent=None) -> None:
        super().__init__(parent)
        self.camera = Camera()
        self.scene_renderer = SceneRenderer()
        self.model = model
        self.tool_manager = tool_manager
        self.selection = None  # M4b — set by MainWindow (bermake.selection.Selection)
        self.snap_engine = SnapEngine()
        self.inference = InferenceState()
        self._status_bar = None
        self._on_event_finished = None
        self._units_provider = None  # M7d — callable () -> bermake.units.Units (or None)
        self._vcb_active_provider = None  # M7.6b Task 9 -- callable () -> bool (VCB.active)
        self._camera_input_callback = None  # M7e — invoked when the user moves the camera
        self._last_snap = None  # M7.6b: most recent SnapResult, for key handlers with no event
        # M7.6b Task 4: the active gesture's drawing plane, pinned once at anchor
        # time (see _update_gesture_plane_normal / _drawing_plane_normal).
        self._gesture_plane_normal: np.ndarray | None = None
        self._gesture_plane_captured = False
        # M7.6b: previous frame's `has_active_gesture`, for the True -> False
        # edge that releases the lock and the acquisition (see
        # _sync_gesture_lifecycle).
        self._gesture_was_active = False
        # M7.6b Task 7: View > Guides. A filter on which annotation PLANS get
        # PAINTED, not on which get collected -- hiding a guide must never
        # renumber anyone's ids. Guides are visible by default (SketchUp).
        self.show_guides = True

        # M7.6c Task 7: View > Select Vertices. Off by default, so an
        # untouched bermake picks exactly as it did before this milestone.
        self.select_vertices = False

        # M7.6c: Qt has no triple-click event, so the third press is counted
        # here and dispatched as Tool.on_mouse_triple_click. The interval is
        # read from Qt once, at construction, so the gesture honours the
        # user's system setting while staying a plain number underneath.
        from PySide6.QtWidgets import QApplication

        from bermake.viewport.click_runs import ClickRuns

        app = QApplication.instance()
        self._click_runs = ClickRuns(
            interval_ms=float(app.doubleClickInterval()) if app is not None else 400.0
        )
        # Reuses inference's clock rather than adding a second wall-clock call
        # site. M7.6b deliberately left exactly one in the codebase and that
        # is worth keeping true.
        self._now_ms_provider = _default_now_ms
        # M7.6c fix round 1: the (x, y, now_ms) most recently fed into
        # _click_runs, so a press and a dblclick reporting the same physical
        # click (identical position and timestamp) cannot advance the run
        # counter twice. See _feed_click_run.
        self._last_click_feed: tuple[float, float, float] | None = None

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._last_mouse_pos: QPoint | None = None
        self._dragging_button: Qt.MouseButton = Qt.MouseButton.NoButton
        self._dragging_modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier
        # M7.6b Task 9 -- the most recent cursor position, widget-local pixels,
        # for the on-canvas measurement readout. None before the mouse has
        # ever moved over the viewport.
        self._last_cursor_px: tuple[float, float] | None = None

        # M7.9: filled by initializeGL; read by the About dialog.
        self.gl_info: GlInfo | None = None
        self._gl_failed = False

    @property
    def scene(self):
        """The active scene from the model (delegates to model.active_scene)."""
        return self.model.active_scene if self.model is not None else None

    def set_status_bar(self, status_bar) -> None:
        self._status_bar = status_bar

    def set_event_finished_callback(self, fn) -> None:
        self._on_event_finished = fn

    def set_render_style(self, style) -> None:
        """Set the viewport display style and repaint (called from the View menu)."""
        self.scene_renderer.set_render_style(style)
        self.update()

    def set_environment(self, environment) -> None:
        """Set the document environment and repaint (called from the View menu)."""
        self.scene_renderer.set_environment(environment)
        self.update()

    def set_units_provider(self, fn) -> None:
        """M7d: install a callable () -> bermake.units.Units, used by
        _paint_annotations to format dimension text. Set by MainWindow
        (mirrors set_status_bar / set_event_finished_callback)."""
        self._units_provider = fn

    def set_vcb_active_provider(self, fn) -> None:
        """M7.6b Task 9: install a callable () -> bool, mirroring the
        ValueControlBox's `active` flag. _paint_annotations consults this to
        decide whether the cursor readout should draw: while the VCB is
        active the typed buffer already occupies both sinks (the Measurements
        box and, by extension, the user's attention), so the readout stands
        down rather than show a second, stale number beside the cursor."""
        self._vcb_active_provider = fn

    def set_now_ms_provider(self, fn) -> None:
        """M7.6c: install a callable () -> float returning milliseconds, for
        the multi-click run counter. Tests inject a value here so no test of
        the triple-click gesture has to sleep."""
        self._now_ms_provider = fn

    def set_camera_input_callback(self, fn) -> None:
        """M7e: install a zero-arg callable invoked when the user manipulates the
        camera (MMB orbit/pan, wheel zoom). MainWindow wires this to the view
        animator's cancel(), so a manual camera move interrupts a running tween."""
        self._camera_input_callback = fn

    def _notify_camera_input(self) -> None:
        if self._camera_input_callback is not None:
            self._camera_input_callback()

    # --- GL lifecycle -----------------------------------------------------

    def initializeGL(self) -> None:
        fmt = QOpenGLContext.currentContext().format()
        info = GlInfo(
            version=(fmt.majorVersion(), fmt.minorVersion()),
            version_string=_gl_string(GL.GL_VERSION),
            renderer=_gl_string(GL.GL_RENDERER),
        )
        self._apply_gl_verdict(info, self.scene_renderer.initialize_gl)

    def _apply_gl_verdict(self, info: GlInfo, initialize: Callable[[], None]) -> None:
        """Initialise the renderer if the context can run it, or report why not.

        Split from initializeGL so the decision is testable without a context.
        The signal is deferred to the event loop because a modal dialog opened
        from inside initializeGL would re-enter Qt's GL setup.
        """
        self.gl_info = info
        shader_error = None
        if info.version >= MIN_GL_VERSION:
            try:
                initialize()
            except RuntimeError as exc:
                shader_error = str(exc)
        verdict = evaluate_gl(info, shader_error)
        if verdict.ok:
            return
        self._gl_failed = True
        logger.error("OpenGL unavailable: %s", verdict.message)
        QTimer.singleShot(0, self, lambda: self.gl_unavailable.emit(verdict.message))

    def resizeGL(self, w: int, h: int) -> None:
        self.scene_renderer.resize(w, h)
        self.camera.aspect = float(w) / max(float(h), 1.0)

    def paintGL(self) -> None:
        if self._gl_failed:
            # Nothing initialised, so nothing can be drawn; clearing is GL 1.1.
            GL.glClearColor(0.25, 0.25, 0.25, 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            return
        active = self.tool_manager.active if self.tool_manager is not None else None
        overlay = active.overlay() if active is not None else None
        self.scene_renderer.render(self.camera, self.model, overlay, self.selection)
        self._paint_annotations(overlay)
        self._paint_vertex_glyphs()

    # --- Mouse handling ---------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_button = Qt.MouseButton.MiddleButton
            self._dragging_modifiers = event.modifiers()
            self._last_mouse_pos = event.position().toPoint()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            active = self.tool_manager.active if self.tool_manager is not None else None
            if active is not None:
                snap = self._snap_for_event(event)
                active.on_mouse_press(event, snap)
                pos = event.position()
                run = self._feed_click_run(float(pos.x()), float(pos.y()), self._now_ms_provider())
                if run == 3:
                    triple = getattr(active, "on_mouse_triple_click", None)
                    if triple is not None:
                        triple(event, snap)
                if self._status_bar is not None:
                    self._status_bar.set_snap(snap.label if snap.kind != SnapKind.NONE else "")
                if self._on_event_finished is not None:
                    self._on_event_finished()
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self._last_cursor_px = (float(pos.x()), float(pos.y()))

        # Camera drag — unchanged from M1.
        if (
            self._dragging_button == Qt.MouseButton.MiddleButton
            and self._last_mouse_pos is not None
        ):
            self._notify_camera_input()
            current = event.position().toPoint()
            dx = float(current.x() - self._last_mouse_pos.x())
            dy = float(current.y() - self._last_mouse_pos.y())
            self._last_mouse_pos = current
            if self._dragging_modifiers & Qt.KeyboardModifier.ShiftModifier:
                self.camera.pan(dx_pixels=dx, dy_pixels=dy)
            else:
                self.camera.orbit(dx_pixels=dx, dy_pixels=-dy)
            self.update()
            event.accept()
            return

        # Tool delegation
        active = self.tool_manager.active if self.tool_manager is not None else None
        if active is not None:
            snap = self._snap_for_event(event)
            active.on_mouse_move(event, snap)
            if self._status_bar is not None:
                self._status_bar.set_snap(snap.label if snap.kind != SnapKind.NONE else "")
                # Coordinates ride the snap the tools already compute. That
                # ties the readout to "a tool is active" -- pressing Esc
                # blanks it (see MainWindow._on_escape's disarm branch).
                # Computing an inference point on every move regardless would
                # cost work on every event for a state users pass through,
                # not sit in.
                units = self._units_provider() if self._units_provider is not None else Units()
                self._status_bar.set_coordinates(format_coordinates(snap.world_position, units))
            if self._on_event_finished is not None:
                self._on_event_finished()
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            active = self.tool_manager.active if self.tool_manager is not None else None
            if active is not None:
                snap = self._snap_for_event(event)
                active.on_mouse_double_click(event, snap)
                # M7.6c fix round 1: the real triple-click sequence is Press,
                # DblClick, Press -- Qt never sends a second MousePress, so
                # the run counter must be fed from here too, or a genuine
                # triple click can only ever reach run 2. This is additive
                # beside the dispatch above, not a replacement for it: entering
                # a group and editing a label's text still flow through
                # on_mouse_double_click exactly as before.
                pos = event.position()
                run = self._feed_click_run(float(pos.x()), float(pos.y()), self._now_ms_provider())
                if run == 3:
                    triple = getattr(active, "on_mouse_triple_click", None)
                    if triple is not None:
                        triple(event, snap)
                if self._on_event_finished is not None:
                    self._on_event_finished()
                self.update()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        # M7.6b Task 9: the cursor readout is meaningless once the cursor has
        # left the viewport -- drop the last known position so _paint_annotations
        # stops drawing it at a stale spot rather than following the cursor
        # onto other widgets.
        self._last_cursor_px = None
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._dragging_button = Qt.MouseButton.NoButton
            self._last_mouse_pos = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            active = self.tool_manager.active if self.tool_manager is not None else None
            if active is not None:
                snap = self._snap_for_event(event)
                active.on_mouse_release(event, snap)
                if self._on_event_finished is not None:
                    self._on_event_finished()
                self.update()
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        notches = event.angleDelta().y() / 120.0
        if notches == 0:
            super().wheelEvent(event)
            return
        self._notify_camera_input()
        cursor = event.position()
        ndc = self._cursor_to_ndc(cursor.x(), cursor.y())
        self.camera.zoom(scroll_delta=notches, cursor_ndc=ndc)
        self.update()
        event.accept()

    def contextMenuEvent(self, event) -> None:
        """M7.2 Task 14: emit a right-click resolution request for MainWindow.

        Suppressed while a tool is genuinely mid-gesture (a multi-click draw,
        or a click-drag in progress): right-click during one of those means
        "cancel", which Esc already handles, so this defers to that instead
        of popping a menu.

        SelectTool needs a narrower test than the others. Its
        has_active_gesture is also True for a merely non-empty selection --
        that is what lets Esc clear one -- and a right-click on an existing
        selection is the ordinary case this whole feature exists to serve,
        so treating that as "mid gesture" would swallow the most common
        right-click of all. But a live box-select drag *is* a gesture in the
        click sense, so is_box_selecting is what we consult for Select.
        """
        active = self.tool_manager.active if self.tool_manager is not None else None
        if active is not None and active.has_active_gesture:
            mid_gesture = active.is_box_selecting if isinstance(active, SelectTool) else True
            if mid_gesture:
                event.ignore()
                return
        position = event.pos()
        self.context_menu_requested.emit(position.x(), position.y())
        event.accept()

    # --- Helpers ----------------------------------------------------------

    def _feed_click_run(self, x: float, y: float, now_ms: float) -> int | None:
        """Feed one press-equivalent event into the multi-click run counter.

        M7.6c fix round 1: Qt's real click sequence replaces the second
        press of a double-click with MouseButtonDblClick, so both
        mousePressEvent and mouseDoubleClickEvent must feed this counter or a
        genuine triple click (Press, DblClick, Press) never reaches run 3 --
        see SelectTool._suppress_next_release, which exists only because
        that DblClick's own release has no matching press.

        Idempotent for a press and a dblclick reported at the same position
        and timestamp: were some platform ever to deliver both for a single
        physical click, feeding the counter twice would make a real
        double-click read as a triple. The guard costs one false negative in
        exchange -- two genuinely distinct clicks landing at the exact same
        pixel at the exact same millisecond would be under-counted -- which
        real input hardware and Qt's own event loop do not produce.
        """
        key = (x, y, now_ms)
        if key == self._last_click_feed:
            return None
        self._last_click_feed = key
        return self._click_runs.press(x, y, now_ms)

    def _cursor_to_ndc(self, x: float, y: float) -> np.ndarray:
        w = max(self.width(), 1)
        h = max(self.height(), 1)
        nx = (2.0 * x / w) - 1.0
        ny = 1.0 - (2.0 * y / h)
        return np.array([nx, ny], dtype=np.float32)

    def _snap_for_event(self, event: QMouseEvent):
        pos = event.position()
        active = self.tool_manager.active if self.tool_manager is not None else None
        anchor = active.anchor_or_none if active is not None else None
        self._sync_gesture_lifecycle(active)
        self._update_gesture_plane_normal(anchor)
        wt = self.model.active_world_transform if self.model is not None else None
        guides, guide_points = self._gather_guides()
        snap = self.snap_engine.snap(
            (float(pos.x()), float(pos.y())),
            (self.width(), self.height()),
            self.camera,
            self.scene,
            anchor=anchor,
            world_transform=wt,
            acquired=self.inference.acquired,
            plane_normal=self._drawing_plane_normal(),
            guides=guides,
            guide_points=guide_points,
        )
        # Acquisition reads the snap the engine just produced; the scene lives
        # here, not in InferenceState, so the edge direction is resolved here.
        self.inference.observe(snap, (float(pos.x()), float(pos.y())), self._edge_direction(snap))
        result = self.inference.apply_lock(
            snap,
            self.camera,
            (self.width(), self.height()),
            (float(pos.x()), float(pos.y())),
            anchor=anchor,
        )
        self._last_snap = result
        return result

    def settle_gesture_lifecycle(self) -> None:
        """Resolve a pending gesture-end edge NOW, before a lock is armed.

        `_sync_gesture_lifecycle` otherwise runs only from `_snap_for_event`,
        i.e. on the next MOUSE event, and a gesture can end on a key: Enter,
        Escape, a typed VCB value. The user is then free to arm a lock before
        moving the mouse at all, which is exactly D6's "a lock often outlives
        the keypress" and the arm-then-click-the-start-point workflow C1
        exists to serve. Left lazy, the stale edge fires on the first mouse
        move afterwards and destroys the lock armed after it.

        `MainWindow` calls this immediately before every arming site, so by
        the time a lock exists the edge has already been consumed and cannot
        fire later. Settling early is never wrong on its own terms either:
        the gesture really has ended, and this only brings the release
        forward from the next mouse move to now.
        """
        active = self.tool_manager.active if self.tool_manager is not None else None
        self._sync_gesture_lifecycle(active)

    def _sync_gesture_lifecycle(self, active) -> None:
        """Release lock + acquisition on the frame after a gesture finishes.

        Spec 2.3: "a lock also releases when the gesture ends". Watching the
        active tool's `has_active_gesture` for a True -> False edge from here
        covers every way a gesture can end -- a commit click, a double-click,
        Enter, Escape, a typed VCB value, or the tool being swapped out --
        without twenty tools each having to remember to notify. This runs at
        the top of `_snap_for_event`, i.e. before the snap it precedes is
        computed, so no frame is ever snapped against a finished gesture's
        state.

        Detection is by edge, so the edge has to be consumed before anything
        that would be destroyed by it is created: see
        `settle_gesture_lifecycle`.
        """
        live = bool(active is not None and active.has_active_gesture)
        if self._gesture_was_active and not live:
            self.inference.end_gesture()
        self._gesture_was_active = live

    def on_active_context_changed(self) -> None:
        """Drop per-context transient state when the user enters or leaves a
        group.

        `Acquired` holds a world position and a per-context entity id. Both
        are meaningless once the context changes: the position is stale, and
        the id names whatever entity happens to carry it in the new context,
        which may be a different edge or none at all.

        Final review I3: the multi-click run is dropped here for the same
        reason. `ClickRuns` measures each press against the PREVIOUS press
        only, so a click a third of a second after a double-click continues
        the run to three -- and in bermake a double-click ENTERS a group, so
        that follow-on click flood-selected the context the user had only
        just stepped into. A multi-click run is a single gesture against a
        single context; a change of context ends it. This is the seam
        MainWindow already drives on every enter/exit (via
        `ToolContext.request_context_rebuild`), which keeps SelectTool from
        having to know the viewport owns a click counter.
        """
        self.inference.clear_acquisition()
        self._click_runs.reset()
        self._last_click_feed = None

    def _gather_guides(self):
        """World-space (lines, points) from the active context's guides.

        Task 8: a guide is an inference target, so the snap engine needs it
        in the same world space as everything else it snaps to, even though
        `Guide`/`GuidePoint` store context-local coordinates. The origin (a
        point) goes through the full `active_world_transform`; the direction
        (a vector) goes through its linear block only, the same split
        `draw_plan._to_world`/`_vec_to_world` use for painting a guide.

        Returns ([], []) when `show_guides` is False, so a hidden guide never
        reaches the snap engine -- Task 7 made a hidden guide unpickable on
        the same reasoning: a hidden thing the user can still act on is a
        trap. Filters what is gathered rather than mutating
        `active_context.annotations`, so hiding a guide never touches any
        annotation's id.
        """
        if not self.show_guides or self.model is None:
            return [], []
        wt = self.model.active_world_transform
        linear = wt[:3, :3]
        lines = []
        points = []
        for ann in self.model.active_context.annotations:
            kind = getattr(ann, "kind", None)
            if kind == "guide":
                origin = apply_mat(ann.origin, wt)[0]
                direction = linear @ np.asarray(ann.direction, dtype=np.float64)
                lines.append((origin, direction))
            elif kind == "guide_point":
                points.append(apply_mat(ann.position, wt)[0])
        return lines, points

    def _edge_direction(self, snap):
        """World-space unit direction of the snapped edge, or None.

        `self.scene` is the ACTIVE CONTEXT's mesh, so the two vertices come
        back in local coordinates. Everything downstream is world:
        `Acquired.position` is `snap.world_position`, tool anchors are world,
        and `snap_candidates.directional_candidates` adds this direction to
        the world anchor. An ordinary direction transforms by the world
        transform's linear block -- the same split `_gather_guides` applies
        seventeen lines above, and `tape_measure_tool._local_vec_to_world`
        applies to the same quantity. A normal would need the inverse
        transpose instead; an edge direction is not a normal.
        """
        if snap is None or snap.edge_id is None or self.scene is None:
            return None
        try:
            edge = self.scene.edge(snap.edge_id)
            p1 = self.scene.vertex(edge.v1_id).position
            p2 = self.scene.vertex(edge.v2_id).position
        except (KeyError, AttributeError):
            return None
        d = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
        wt = self.model.active_world_transform if self.model is not None else None
        if wt is not None and not is_identity_transform(wt):
            d = np.asarray(wt, dtype=np.float64)[:3, :3] @ d
        n = float(np.linalg.norm(d))
        return None if n < 1e-12 else d / n

    @property
    def last_snap(self):
        """The most recent SnapResult, for key handlers that have no event."""
        return self._last_snap

    def _update_gesture_plane_normal(self, anchor) -> None:
        """Pin the drawing plane's normal once, at the anchor's None -> not-None edge.

        Called at the top of `_snap_for_event`, before `self._last_snap` is
        overwritten with this frame's result, so `self._last_snap` here still
        holds the PREVIOUS frame's snap. On the frame the anchor first appears,
        that previous snap is exactly the click that set it -- the frame order
        in mousePressEvent computes the snap before delegating to the tool,
        so the anchor is still None while that snap is taken, and only becomes
        non-None afterwards, once the tool consumes it. That makes this the
        one moment `_last_snap` reliably names the face the gesture started on.

        Captured exactly once per gesture, not re-attempted on later frames
        even if the first attempt found no face (`face_id is None`, or the
        scene lookup fails): a later frame's snap belongs to wherever the
        cursor is now, not to the anchor, so trying again there would silently
        reintroduce the per-frame chasing this exists to prevent.
        """
        if anchor is None:
            self._gesture_plane_normal = None
            self._gesture_plane_captured = False
            return
        if self._gesture_plane_captured:
            return
        self._gesture_plane_captured = True
        snap = self._last_snap
        if snap is None or snap.face_id is None or self.scene is None:
            return
        try:
            normal_local = np.asarray(self.scene.face_normal(snap.face_id), dtype=np.float64)
        except (KeyError, ValueError):
            self._gesture_plane_normal = None
            return
        self._gesture_plane_normal = self._normal_to_world(normal_local)

    def _normal_to_world(self, normal_local):
        """Take a local face normal into world space, or None if degenerate.

        A normal is not an ordinary vector: under a non-uniform scale the
        plain linear block tilts it off perpendicular, so it goes through the
        INVERSE TRANSPOSE of that block instead. Same rule, same reason, and
        the same M7.4 issue (#92) as `Model.pick_face_local`,
        `paint_tool` and `tape_measure_tool._plane_normal_world`. Contrast
        `_edge_direction` just above, which is an ordinary direction and so
        uses the plain linear block.
        """
        wt = self.model.active_world_transform if self.model is not None else None
        n = np.asarray(normal_local, dtype=np.float64)
        if wt is not None and not is_identity_transform(wt):
            n = mat_invert(np.asarray(wt, dtype=np.float64))[:3, :3].T @ n
        length = float(np.linalg.norm(n))
        return None if length < 1e-12 else n / length

    def _drawing_plane_normal(self):
        """The active gesture's plane normal, for Perpendicular. None if unknown.

        Returns the value pinned by `_update_gesture_plane_normal` at anchor
        time, not a value re-derived from the current frame. Spec 2.2 resolves
        Perpendicular "inside the gesture's drawing plane", which is a
        property of where the gesture started, not of whatever face happens to
        be under the cursor this frame: re-deriving it every frame would make
        the perpendicular direction flip when the cursor crosses onto an
        adjoining wall mid-gesture, and vanish outright over empty space, and
        would also feed the snap engine's own output back into its next call's
        inputs.
        """
        return self._gesture_plane_normal

    def _paint_annotations(self, overlay=None) -> None:
        """M7d: draw every visible context's annotations in screen space, on
        top of the GL render, dimmed outside the active context exactly like
        geometry (#95 -- Task 13). All layout is delegated to the pure
        draw_plan module (collect_annotation_plans/plan_annotation); this
        method only projects + paints via QPainter.

        `overlay`: Task 10b -- the active tool's ToolOverlay for this frame
        (the same object paintGL already asks the tool for and hands to
        scene_renderer.render() to draw edge/face/instance hover). Its
        `hovered_annotation_id` is threaded through to paint_annotation_plans
        so the hovered annotation gets a hover highlight too, without a new
        tool -> painter channel.

        Picking is unaffected by this: pick_annotation and every tool's
        _pick_annotation still only ever see active_context.annotations, so
        dimmed (non-active-context) annotations are visible but not
        selectable/hoverable -- annotation ids are per-context, not globally
        unique, so a dimmed plan is never routed through the same
        paint_annotation_plans call as the selection/hover ids (which belong
        to the active context only); doing so could otherwise collide."""
        from PySide6.QtGui import QColor, QFont, QPainter, QPen

        from bermake.annotations.draw_plan import (
            FONT_PX,
            collect_annotation_plans,
            plan_cursor_readout,
        )
        from bermake.viewport.annotation_painter import paint_annotation_plans
        from bermake.viewport.scene_renderer import _DIM_ALPHA_BLEND

        if self.model is None:
            return
        width, height = self.width(), self.height()
        # Every other units provider in this codebase (wall/opening/roof options
        # bars) always yields a real Units object -- None is not a value
        # format_length expects, so default to Units() rather than None.
        units = self._units_provider() if self._units_provider is not None else Units()
        plans = collect_annotation_plans(self.model, self.camera, width, height, units)

        # M7.6b Task 9: the on-canvas measurement readout, transient chrome
        # beside the cursor -- NOT a collected annotation. It is built here,
        # kept out of `plans` entirely, and painted separately below, so it
        # can never reach pick_annotation or collide with a real annotation's
        # id (it uses the sentinel id -1; see plan_cursor_readout). Shown only
        # while a tool is active, has a live measurement, the cursor is known,
        # and the VCB isn't already showing the typed buffer in the same box.
        readout_plan = None
        active_tool = self.tool_manager.active if self.tool_manager is not None else None
        vcb_active = self._vcb_active_provider() if self._vcb_active_provider is not None else False
        if active_tool is not None and not vcb_active and self._last_cursor_px is not None:
            measurement_text = active_tool.measurement_text
            if measurement_text is not None:
                readout_plan = plan_cursor_readout(
                    measurement_text, self._last_cursor_px, width, height
                )

        if not plans and readout_plan is None:
            return
        # Task 7: show_guides filters what gets PAINTED, not what got
        # COLLECTED above -- hiding a guide must never change any other
        # annotation's id, which collecting fewer plans up front would risk.
        if not self.show_guides:
            plans = [(plan, dimmed) for plan, dimmed in plans if plan.kind not in _GUIDE_KINDS]
        active_plans = [plan for plan, dimmed in plans if not dimmed]
        dimmed_plans = [plan for plan, dimmed in plans if dimmed]

        color = QColor(30, 30, 30)
        dim_color = QColor(color)
        dim_color.setAlphaF(_DIM_ALPHA_BLEND)

        # Task 6: guides and guide points draw dashed and grey regardless of
        # selection/hover -- they are construction geometry, not measured
        # annotations, so they never compete visually with dimensions/labels.
        guide_color = QColor.fromRgbF(0.45, 0.45, 0.52)
        guide_pen = QPen(guide_color)
        guide_pen.setStyle(Qt.PenStyle.DashLine)
        dim_guide_color = QColor(guide_color)
        dim_guide_color.setAlphaF(_DIM_ALPHA_BLEND)
        dim_guide_pen = QPen(dim_guide_color)
        dim_guide_pen.setStyle(Qt.PenStyle.DashLine)

        selected_ids = set(self.selection.annotations) if self.selection is not None else set()
        hovered_id = overlay.hovered_annotation_id if overlay is not None else None
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            font = QFont()
            font.setPixelSize(int(FONT_PX))
            painter.setFont(font)
            # Dimmed plans first (never selectable/hoverable -- picking stays
            # active-context-only), so the active-context plans draw on top.
            paint_annotation_plans(
                painter, dimmed_plans, dim_color, set(), dim_color, None, None, dim_guide_pen
            )
            paint_annotation_plans(
                painter,
                active_plans,
                color,
                selected_ids,
                QColor(51, 140, 242),
                hovered_id,
                QColor(
                    round(_HOVER_EDGE_COLOR[0] * 255),
                    round(_HOVER_EDGE_COLOR[1] * 255),
                    round(_HOVER_EDGE_COLOR[2] * 255),
                ),
                guide_pen,
            )
            if readout_plan is not None:
                # Plain color, no selection/hover/guide styling -- this is
                # chrome, not a selectable/dimmable annotation. Painted last
                # so it sits on top of everything else.
                paint_annotation_plans(painter, [readout_plan], color, set(), color)
        finally:
            painter.end()

    _VERTEX_GLYPH_PX = 7.0

    def _selected_vertex_points(self, width: int, height: int) -> list[tuple[float, float]]:
        """Screen positions of every selected vertex, in the active context.

        Pure: projection only, no painting, so the layout is testable without
        a GL context. Returns [] when the mode is off, matching how View >
        Guides makes a hidden guide unpaintable rather than merely dimmed. A
        vertex that does not project (behind the eye) is skipped rather than
        clamped, because a clamped position is a glyph drawn somewhere the
        vertex is not.
        """
        if not self.select_vertices or self.model is None or self.selection is None:
            return []
        selected = self.selection.vertices
        if not selected:
            return []
        scene = self.model.active_scene
        wt = self.model.active_world_transform
        out: list[tuple[float, float]] = []
        for v_id in sorted(selected):
            try:
                local = scene.vertex(v_id).position
            except KeyError:
                continue
            world = (
                local
                if is_identity_transform(wt)
                else apply_mat(
                    np.asarray(local, dtype=np.float64).reshape(1, 3),
                    np.asarray(wt, dtype=np.float64),
                )[0]
            )
            s = self.camera.world_to_screen(np.asarray(world, dtype=np.float32), width, height)
            if s is None:
                continue
            out.append((float(s[0]), float(s[1])))
        return out

    def _paint_vertex_glyphs(self) -> None:
        """Draw a small filled square at each selected vertex.

        Its own QPainter pass rather than a branch inside _paint_annotations,
        which returns early when there is nothing annotation-shaped to draw
        and would therefore hide the glyph in any document without
        annotations, which is most of them.

        No depth test: the 2D layer has no depth buffer, so a selected vertex
        behind a wall draws on top of it. Accepted (spec D13) and consistent
        with how snap markers already behave.
        """
        from PySide6.QtGui import QColor, QPainter

        points = self._selected_vertex_points(self.width(), self.height())
        if not points:
            return
        half = self._VERTEX_GLYPH_PX * 0.5
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            painter.setPen(QColor(20, 20, 20))
            painter.setBrush(QColor(51, 140, 242))
            for x, y in points:
                painter.drawRect(
                    round(x - half),
                    round(y - half),
                    int(self._VERTEX_GLYPH_PX),
                    int(self._VERTEX_GLYPH_PX),
                )
        finally:
            painter.end()
