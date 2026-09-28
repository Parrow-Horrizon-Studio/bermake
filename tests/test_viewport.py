"""Smoke tests for the main window and the M1 viewport widget.

These use pytest-qt for the QApplication fixture. Rendering is not pixel-
verified (that requires framebuffer capture, out of scope for M1) — but
construction, GL context creation, mouse handling, and one full paint cycle
are exercised via the Qt offscreen platform.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent


def test_main_window_constructs(qtbot):
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "Untitled - Bermake"  # M6a: filename + dirty-state title


def test_viewport_widget_constructs(qtbot):
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    assert widget is not None


def test_viewport_widget_has_camera_and_scene(qtbot):
    from bermake.viewport.camera import Camera
    from bermake.viewport.scene_renderer import SceneRenderer
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    assert isinstance(widget.camera, Camera)
    assert isinstance(widget.scene_renderer, SceneRenderer)


def test_resize_updates_camera_aspect(qtbot):
    """Resizing the widget must update the camera's aspect ratio."""
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    widget.resize(1600, 800)
    # Qt may not deliver resizeGL until the widget is shown; we call it directly.
    widget.resizeGL(1600, 800)
    assert widget.camera.aspect == 2.0


def test_middle_button_drag_orbits_camera(qtbot):
    """MMB drag should change the camera position (orbit)."""
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    pos_before = widget.camera.position.copy()

    # Simulate MMB press at (100, 100) and release at (200, 150).
    qtbot.mousePress(widget, Qt.MouseButton.MiddleButton, pos=QPoint(100, 100))
    qtbot.mouseMove(widget, QPoint(200, 150))
    qtbot.mouseRelease(widget, Qt.MouseButton.MiddleButton, pos=QPoint(200, 150))

    assert not np.allclose(widget.camera.position, pos_before)


def test_wheel_event_zooms_camera(qtbot):
    """Scrolling should move the camera position along the zoom direction.

    Since the viewport always passes cursor_ndc to camera.zoom, both position
    and target move together (pure zoom, no rotation). The camera-to-target
    distance is preserved; it is camera.position that moves.
    """
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    pos_before = widget.camera.position.copy()

    # Drive the wheel event directly (qtbot doesn't have a wheel helper).
    widget.wheelEvent(_make_wheel_event(widget, delta_y=120))

    assert not np.allclose(widget.camera.position, pos_before), (
        "camera position should have moved after a wheel zoom"
    )


def test_wheel_event_with_zero_vertical_delta_does_not_zoom(qtbot):
    """Horizontal-only trackpad scrolls (delta_y=0) must not change the camera.

    The viewport's wheelEvent should also leave the event unaccepted so a
    parent widget could handle horizontal scroll without it being swallowed.
    """
    from bermake.viewport.viewport_widget import ViewportWidget

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    pos_before = widget.camera.position.copy()
    target_before = widget.camera.target.copy()

    event = _make_wheel_event(widget, delta_y=0)
    widget.wheelEvent(event)

    np.testing.assert_array_equal(widget.camera.position, pos_before)
    np.testing.assert_array_equal(widget.camera.target, target_before)
    assert not event.isAccepted()


def _make_wheel_event(widget, delta_y: int) -> QWheelEvent:
    """Construct a QWheelEvent suitable for delivery to a widget's wheelEvent."""
    from PySide6.QtCore import QPointF

    pos = QPointF(widget.width() / 2.0, widget.height() / 2.0)
    return QWheelEvent(
        pos,                              # position (local)
        pos,                              # global position (offscreen: same as local)
        QPoint(0, 0),                     # pixelDelta
        QPoint(0, delta_y),               # angleDelta (120 = one notch on most mice)
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,                            # inverted
    )


def test_keyboard_l_activates_line_tool(qtbot):
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)
    qtbot.keyClick(window, Qt.Key.Key_L)
    assert window._tool_manager.active is not None
    assert window._tool_manager.active.name == "Line"


def test_keyboard_r_activates_rectangle_tool(qtbot):
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)
    qtbot.keyClick(window, Qt.Key.Key_R)
    assert window._tool_manager.active.name == "Rectangle"


def test_esc_two_stage_cancel_then_deactivate(qtbot):
    """First ESC cancels gesture; second ESC deactivates the tool entirely."""
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)

    # Activate the Line tool, start a gesture.
    qtbot.keyClick(window, Qt.Key.Key_L)
    assert window._tool_manager.active is not None
    # Drive a first click via the active tool directly (skipping the mouse
    # plumbing because the QOpenGLWidget doesn't render the offscreen ray).
    from bermake.viewport.snap_engine import SnapKind, SnapResult

    active = window._tool_manager.active
    snap = SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Grid",
    )
    active.on_mouse_press(None, snap)  # type: ignore[arg-type]
    assert active.has_active_gesture is True

    # First ESC: cancels the gesture, tool stays active.
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._tool_manager.active is not None  # tool still active
    assert active.has_active_gesture is False       # gesture cleared

    # Second ESC: deactivates the tool entirely.
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    assert window._tool_manager.active is None      # no active tool


def test_clear_active_context_clears_scene(qtbot):
    # Ctrl+N is now File ▸ New (M6a); the old clear-scene behavior moved to
    # Edit ▸ "Clear Active Context" (_on_clear_scene), still undoable.
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    # Seed the scene with a vertex.
    window.scene.add_vertex(np.array([1.0, 2.0, 0.0], dtype=np.float32))
    assert len(list(window.scene.vertices_iter())) == 1
    window._on_clear_scene()
    assert len(list(window.scene.vertices_iter())) == 0


def test_ctrl_z_undoes_completed_rectangle(qtbot):
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    qtbot.wait(50)

    # Activate Rectangle, simulate two clicks via the tool directly.
    qtbot.keyClick(window, Qt.Key.Key_R)
    from bermake.viewport.snap_engine import SnapKind, SnapResult
    active = window._tool_manager.active

    def snap_at(x, y):
        return SnapResult(
            kind=SnapKind.GRID,
            world_position=np.array([x, y, 0.0], dtype=np.float32),
            axis=None,
            vertex_id=None,
            label="Grid",
        )

    active.on_mouse_press(None, snap_at(0.0, 0.0))  # type: ignore[arg-type]
    active.on_mouse_press(None, snap_at(3.0, 2.0))  # type: ignore[arg-type]
    assert len(list(window.scene.faces_iter())) == 1

    qtbot.keyClick(window, Qt.Key.Key_Z, modifier=Qt.KeyboardModifier.ControlModifier)
    assert len(list(window.scene.faces_iter())) == 0

    qtbot.keyClick(window, Qt.Key.Key_Y, modifier=Qt.KeyboardModifier.ControlModifier)
    assert len(list(window.scene.faces_iter())) == 1


def test_clear_active_context_is_undoable(qtbot):
    # Clear Active Context (formerly Ctrl+N, now on the Edit menu) stays undoable.
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    # Seed the scene with two vertices.
    window.scene.add_vertex(np.array([1.0, 2.0, 0.0], dtype=np.float32))
    window.scene.add_vertex(np.array([3.0, 4.0, 0.0], dtype=np.float32))
    assert len(list(window.scene.vertices_iter())) == 2

    window._on_clear_scene()
    assert len(list(window.scene.vertices_iter())) == 0

    window._on_undo()
    assert len(list(window.scene.vertices_iter())) == 2


class TestStatusBarThirdSlot:
    # M7.3 Task 15 split set_status's target out of the joined prompt string
    # into its own Measurements box, so what was once the "third segment" of
    # bar.text() is now bar.measurements_text(); the prompt keeps only tool
    # and snap.
    def test_set_status_appends_third_segment(self, qtbot):  # noqa: ARG002
        from bermake.ui.status_bar import StatusBar

        bar = StatusBar()
        qtbot.addWidget(bar)
        bar.set_tool("Push/Pull")
        bar.set_snap("")
        bar.set_status("depth: 1.500")
        assert bar.prompt_text() == "Push/Pull · —"
        assert bar.measurements_text() == "depth: 1.500"

    def test_set_status_empty_omits_the_third_segment(self, qtbot):  # noqa: ARG002
        from bermake.ui.status_bar import StatusBar

        bar = StatusBar()
        qtbot.addWidget(bar)
        bar.set_tool("Rectangle")
        bar.set_snap("Endpoint")
        bar.set_status("")  # PushPullTool's status_text returns None outside DRAGGING
        assert bar.prompt_text() == "Rectangle · Endpoint"
        assert bar.measurements_text() == ""


class TestCoordinatesWithNoSnap:
    # M7.3 Task 15 review finding 2: SnapResult.world_position
    # (python/bermake/viewport/snap_engine.py:43) is a non-optional np.ndarray,
    # always populated even when kind == SnapKind.NONE. So with no snap match
    # the coordinates readout should still show the inferred point -- only the
    # snap *label* goes blank (viewport_widget.py's mouseMoveEvent feed site
    # calls set_coordinates unconditionally but set_snap only when kind !=
    # SnapKind.NONE).
    def test_no_snap_still_fills_coordinates_but_blanks_the_snap_label(
        self, qtbot, main_window, monkeypatch
    ):
        from PySide6.QtCore import QPointF
        from bermake.viewport.snap_engine import SnapKind, SnapResult

        main_window._activate("line")

        forced = SnapResult(
            kind=SnapKind.NONE,
            world_position=np.array([5.0, 6.0, 0.0], dtype=np.float32),
            axis=None,
            vertex_id=None,
            label="Should Not Appear",
        )
        monkeypatch.setattr(main_window._viewport, "_snap_for_event", lambda event: forced)

        event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(40.0, 40.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        main_window._viewport.mouseMoveEvent(event)

        coords = main_window._status_bar.coordinates_text()
        assert "5 m" in coords
        assert "6 m" in coords
        # The label from the forced SnapKind.NONE result must not leak into
        # the prompt -- only the inferred point (via coordinates) survives.
        assert "Should Not Appear" not in main_window._status_bar.prompt_text()


class TestPushPullToolIntegration:
    def test_p_keybind_activates_push_pull_tool(self, qtbot):
        from PySide6.QtCore import Qt

        from bermake.ui.main_window import MainWindow

        win = MainWindow()
        qtbot.addWidget(win)
        win.show()
        qtbot.waitExposed(win)
        qtbot.wait(50)
        qtbot.keyClick(win, Qt.Key.Key_P)
        assert win._tool_manager.active is not None
        assert win._tool_manager.active.name == "Push/Pull"

    def test_tool_context_carries_camera_and_widget_size_provider(self, qtbot):  # noqa: ARG002
        """Sanity: MainWindow wires the viewport's camera + size accessor
        into the ToolContext so PushPullTool can compute camera rays."""
        from bermake.ui.main_window import MainWindow

        win = MainWindow()
        ctx = win._tool_manager._ctx  # noqa: SLF001
        assert ctx.camera is win._viewport.camera
        assert callable(ctx.widget_size_provider)
        size = ctx.widget_size_provider()
        assert isinstance(size, tuple)
        assert len(size) == 2


def test_camera_input_callback_fires(qtbot):
    from bermake.viewport.viewport_widget import ViewportWidget
    vp = ViewportWidget()
    qtbot.addWidget(vp)
    fired = []
    vp.set_camera_input_callback(lambda: fired.append(1))
    vp._notify_camera_input()
    assert fired == [1]


def test_drawing_plane_normal_is_pinned_at_anchor_time_not_re_derived(qtbot):
    """M7.6b Task 4 fix round 2: the plane must not chase the cursor mid-gesture.

    The original implementation re-derived the plane from self._last_snap on
    every frame, which names whatever face the cursor is over NOW, not the
    face the gesture started on: crossing onto an adjoining wall flipped the
    perpendicular direction, and moving into empty space made it vanish. This
    drives _update_gesture_plane_normal / _drawing_plane_normal directly (the
    pair _snap_for_event calls) to check the fixed behaviour: the captured
    normal must survive the cursor moving onto a different face while the
    anchor is still set, and must clear once the anchor goes away.
    """
    from bermake.viewport.snap_engine import SnapKind, SnapResult
    from bermake.viewport.viewport_widget import ViewportWidget

    class _FakeScene:
        def __init__(self):
            self._normals = {
                1: np.array([0.0, 0.0, 1.0]),
                2: np.array([1.0, 0.0, 0.0]),
            }

        def face_normal(self, f_id):
            return self._normals[f_id]

    class _FakeModel:
        def __init__(self, scene):
            self.active_scene = scene
            self.active_world_transform = None

    def snap_on_face(f_id):
        return SnapResult(
            kind=SnapKind.ON_FACE,
            world_position=np.zeros(3, dtype=np.float32),
            axis=None,
            vertex_id=None,
            label="On Face",
            face_id=f_id,
        )

    widget = ViewportWidget()
    qtbot.addWidget(widget)
    widget.model = _FakeModel(_FakeScene())

    # No anchor yet: no plane, regardless of what is under the cursor.
    widget._last_snap = snap_on_face(1)
    widget._update_gesture_plane_normal(None)
    assert widget._drawing_plane_normal() is None

    # Anchor appears: _last_snap still holds the click's own snap (face 1),
    # which is the frame ordering _update_gesture_plane_normal relies on.
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    widget._update_gesture_plane_normal(anchor)
    np.testing.assert_allclose(widget._drawing_plane_normal(), [0.0, 0.0, 1.0])

    # Cursor moves onto a different face mid-gesture: the pinned plane must
    # not follow it.
    widget._last_snap = snap_on_face(2)
    widget._update_gesture_plane_normal(anchor)
    np.testing.assert_allclose(widget._drawing_plane_normal(), [0.0, 0.0, 1.0])

    # Gesture ends: the plane clears, ready for the next one.
    widget._update_gesture_plane_normal(None)
    assert widget._drawing_plane_normal() is None
