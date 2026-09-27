"""PushPullTool overlay tests — what polygons are returned per state."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent


def _make_event(pos=(100.0, 100.0)):
    return QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(*pos),
        QPointF(*pos),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _make_tool_with_unit_rect():
    """Same helper as in test_push_pull_tool.py; duplicated here to keep tests independent."""
    from bermake.scene import Scene
    from bermake.tools.push_pull_tool import PushPullTool
    from bermake.tools.tool import ToolContext

    scene = Scene()
    v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    f = scene.add_face_from_loop([v0, v1, v2, v3])

    camera = MagicMock()
    camera.ray_from_screen.return_value = (
        np.array([0.5, 0.5, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
    )
    cmd_stack = MagicMock()
    tool = PushPullTool()
    tool.activate(
        ToolContext(
            scene=scene,
            command_stack=cmd_stack,
            camera=camera,
            widget_size_provider=lambda: (800, 600),
        )
    )
    return tool, scene, f, camera, cmd_stack


def _enter_dragging(tool, camera, depth_target=2.0):
    """Hover, click to arm, then move the camera ray so depth = depth_target."""
    tool.on_mouse_move(_make_event(), snap=None)
    tool.on_mouse_press(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(100.0, 100.0),
            QPointF(100.0, 100.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
        snap=None,
    )
    # Horizontal ray at z=depth_target → CPA gives t=depth_target.
    camera.ray_from_screen.return_value = (
        np.array([-3.0, 0.5, depth_target], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )
    tool.on_mouse_move(_make_event(), snap=None)


class TestPushPullOverlay:
    def test_idle_overlay_has_no_polygons(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        # Move OFF the rectangle so we stay IDLE.
        camera.ray_from_screen.return_value = (
            np.array([5.0, 5.0, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        overlay = tool.overlay()
        assert overlay.face_fill_polygons == []

    def test_hovering_overlay_has_one_polygon(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 1
        # Hover color (light blue, 0.20 alpha) — see PushPullTool constants.
        assert overlay.face_fill_color[3] == 0.20

    def test_dragging_overlay_has_armed_face_plus_ghost_prism(self):
        """A 4-vertex source face produces a 6-polygon overlay during drag:
        1 armed face + 1 ghost top + 4 ghost sides = 6."""
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        _enter_dragging(tool, camera, depth_target=2.0)
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 6

    def test_dragging_ghost_top_is_source_loop_shifted_by_depth_times_normal(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        _enter_dragging(tool, camera, depth_target=2.0)
        overlay = tool.overlay()
        # By convention: index 0 is the armed face, index 1 is the ghost top.
        ghost_top = overlay.face_fill_polygons[1]
        assert ghost_top.shape == (4, 3)
        # All z-coordinates should be exactly depth (2.0).
        np.testing.assert_allclose(ghost_top[:, 2], [2.0, 2.0, 2.0, 2.0], atol=1e-5)
        # X/Y match the source.
        np.testing.assert_allclose(
            sorted(map(tuple, ghost_top[:, :2].tolist())),
            sorted([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]),
            atol=1e-5,
        )

    def test_dragging_side_polygons_each_have_four_vertices(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        _enter_dragging(tool, camera, depth_target=2.0)
        overlay = tool.overlay()
        sides = overlay.face_fill_polygons[2:]  # 4 side polys
        assert len(sides) == 4
        for side in sides:
            assert side.shape == (4, 3)

    def test_dragging_color_is_ghost_color(self):
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        _enter_dragging(tool, camera, depth_target=2.0)
        overlay = tool.overlay()
        # Ghost color (light blue, 0.15 alpha) — see PushPullTool constants.
        assert overlay.face_fill_color[3] == 0.15


class TestPushPullOverlayWorldLift:
    """M4e double-transform fix: face_fill_polygons must be in WORLD coords.

    When push/pull is active inside a translated group (+10 on X), the returned
    polygons must already be offset by +10 so the renderer can draw them at
    identity without an additional transform.  At root (identity) the lift is a
    no-op and existing test assertions still hold.
    """

    def _make_tool_inside_translated_group(self, tx: float = 10.0):
        """Build a Model with one group translated by (tx, 0, 0), enter it,
        then activate PushPullTool against the group's scene."""
        from bermake.model.model import Model
        from bermake.scene import Scene
        from bermake.tools.push_pull_tool import PushPullTool
        from bermake.tools.tool import ToolContext

        model = Model()

        # Build a unit-square face in a new definition (group).
        group_def = model.new_definition("G", is_group=True)
        scene: Scene = group_def.mesh
        v0 = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        v1 = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        v2 = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
        v3 = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        face_id = scene.add_face_from_loop([v0, v1, v2, v3])

        # Place instance at +tx on X.
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = tx
        inst = model.new_instance(group_def, T)
        model.root.children.append(inst)

        # Enter the instance so active_world_transform == T.
        model.enter(inst)

        camera = MagicMock()
        # The camera mock must return a WORLD-space ray.  ray_into_local will
        # subtract tx from x, giving local origin (0.5, 0.5, 5) → hits face.
        camera.ray_from_screen.return_value = (
            np.array([tx + 0.5, 0.5, 5.0], dtype=np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        tool = PushPullTool()
        tool.activate(
            ToolContext(
                scene=scene,
                command_stack=MagicMock(),
                camera=camera,
                widget_size_provider=lambda: (800, 600),
                model=model,
            )
        )
        return tool, scene, face_id, camera, model, tx

    def test_hovering_polygon_is_lifted_to_world(self):
        """HOVERING: the single polygon's x-coords must all be >= tx (world offset)."""
        tool, scene, face_id, camera, model, tx = self._make_tool_inside_translated_group(tx=10.0)
        # Hover over the face — ray_into_local converts the world ray to local,
        # so the mock ray (local 0.5,0.5,5→-Z) hits the local face correctly.
        tool.on_mouse_move(_make_event(), snap=None)
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 1, "HOVERING must produce exactly one polygon"
        poly = overlay.face_fill_polygons[0]
        # Local face has x in [0,1]; world-lifted it must have x in [tx, tx+1].
        np.testing.assert_array_less(
            np.full(poly.shape[0], tx - 1e-4), poly[:, 0],
            err_msg="All x-coords must be >= tx (polygon not lifted to world)",
        )
        np.testing.assert_array_less(
            poly[:, 0], np.full(poly.shape[0], tx + 1.0 + 1e-4),
            err_msg="All x-coords must be <= tx+1 (polygon not doubled-offset)",
        )

    def test_hovering_at_root_is_identity_noop(self):
        """At root (no model / identity transform) coordinates must be unchanged."""
        tool, scene, f, camera, _ = _make_tool_with_unit_rect()
        tool.on_mouse_move(_make_event(), snap=None)
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 1
        poly = overlay.face_fill_polygons[0]
        # Local face: x in [0,1], y in [0,1], z == 0.
        assert poly[:, 0].max() <= 1.0 + 1e-5
        assert poly[:, 0].min() >= 0.0 - 1e-5

    def test_dragging_ghost_top_lifted_to_world(self):
        """DRAGGING: ghost-top z == depth AND x-coords are world-offset by tx."""
        tool, scene, face_id, camera, model, tx = self._make_tool_inside_translated_group(tx=10.0)
        # Hover to arm.
        tool.on_mouse_move(_make_event(), snap=None)
        tool.on_mouse_press(
            QMouseEvent(
                QMouseEvent.Type.MouseButtonPress,
                QPointF(100.0, 100.0),
                QPointF(100.0, 100.0),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
            snap=None,
        )
        depth = 3.0
        # Horizontal world ray that, after ray_into_local (subtract tx from x),
        # gives local origin (-3, 0.5, depth) → CPA yields depth=3.0.
        camera.ray_from_screen.return_value = (
            np.array([tx - 3.0, 0.5, depth], dtype=np.float32),
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        tool.on_mouse_move(_make_event(), snap=None)
        overlay = tool.overlay()
        assert len(overlay.face_fill_polygons) == 6, "4-vert face: 1+1+4 = 6 polygons"
        ghost_top = overlay.face_fill_polygons[1]
        # z must equal depth (unchanged by X translation).
        np.testing.assert_allclose(ghost_top[:, 2], depth, atol=1e-4)
        # x-coords must be in world space: [tx, tx+1].
        assert ghost_top[:, 0].min() >= tx - 1e-4
        assert ghost_top[:, 0].max() <= tx + 1.0 + 1e-4
