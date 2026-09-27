"""Tests for the transform-awareness added to MoveTool and RotateTool.

Validates that WORLD-space gesture quantities are correctly converted to the
active context's LOCAL frame when editing inside a moved group/component.

Key invariants
--------------
* At IDENTITY (root context) every conversion is a no-op — existing behaviour
  is fully preserved.
* Through a PURE TRANSLATION the inverse 3×3 block is identity, so a world
  delta is unchanged as a local delta (translation only shifts points, not
  vectors).
* Through a ROTATE-90-ABOUT-Z (world ←→ local) a world vector [10, 0, 0]
  becomes [0, -10, 0] in local space (the inverse rotation).

The tests below cover the conversion math (world_vec_to_local) and the
full tool-drive path (build Model, enter translated instance, drive Move,
assert the local vertex moved by the correct local delta).
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.geometry.transforms import (
    apply_mat,
    is_identity_transform,
    mat_invert,
    mat_rotate,
    mat_translate,
)
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.move_tool import MoveTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.tool import ToolContext
from bermake.viewport.picking import world_to_local_point
from bermake.viewport.snap_engine import SnapKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _press(x=0.0, y=0.0):
    return QMouseEvent(
        QEvent.Type.MouseButtonPress, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _release(x=0.0, y=0.0):
    return QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(x, y),
        Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )


def _snap(pos, kind=SnapKind.ENDPOINT):
    return types.SimpleNamespace(
        kind=kind,
        world_position=np.asarray(pos, np.float32),
        axis=None, vertex_id=None, edge_id=None, edge_t=None,
    )


def _ctx(scene, stack, selection, model=None):
    return ToolContext(
        scene=scene, command_stack=stack, camera=None,
        widget_size_provider=lambda: (800, 600), selection=selection,
        model=model,
    )


def _world_vec_to_local(world_vec, wt):
    """Mirror of MoveTool._world_vec_to_local — used in assertion math."""
    if is_identity_transform(wt):
        return np.asarray(world_vec, np.float32)
    inv3 = mat_invert(np.asarray(wt, np.float64))[:3, :3]
    return (inv3 @ np.asarray(world_vec, np.float64)).astype(np.float32)


# ---------------------------------------------------------------------------
# Conversion math — no tool machinery needed
# ---------------------------------------------------------------------------

class TestWorldVecToLocal:
    """Unit tests for the inverse-3×3 vector conversion."""

    def test_identity_noop(self):
        wt = np.eye(4, dtype=np.float64)
        v = np.array([10.0, 0.0, 0.0], np.float32)
        result = _world_vec_to_local(v, wt)
        assert np.allclose(result, v, atol=1e-6)

    def test_none_noop(self):
        v = np.array([10.0, 0.0, 0.0], np.float32)
        result = _world_vec_to_local(v, None)
        assert np.allclose(result, v, atol=1e-6)

    def test_pure_translation_noop_on_vector(self):
        # A pure translation shifts points but must NOT change directions.
        wt = mat_translate([5.0, 0.0, 0.0])
        v = np.array([10.0, 0.0, 0.0], np.float32)
        result = _world_vec_to_local(v, wt)
        assert np.allclose(result, v, atol=1e-6), (
            "Translation should not affect a direction vector"
        )

    def test_rotate90_z_inverts_x_to_neg_y(self):
        # World is rotated 90° CCW about Z relative to local.
        # local_x = world_y, local_y = -world_x  →  world [10,0,0] → local [0,-10,0]
        wt = mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        v = np.array([10.0, 0.0, 0.0], np.float32)
        result = _world_vec_to_local(v, wt)
        assert np.allclose(result, [0.0, -10.0, 0.0], atol=1e-5), (
            f"Expected [0, -10, 0], got {result}"
        )

    def test_rotate90_z_y_to_pos_x(self):
        wt = mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        v = np.array([0.0, 10.0, 0.0], np.float32)
        result = _world_vec_to_local(v, wt)
        assert np.allclose(result, [10.0, 0.0, 0.0], atol=1e-5), (
            f"Expected [10, 0, 0], got {result}"
        )


class TestWorldPointToLocal:
    """Unit tests for world_to_local_point."""

    def test_identity_noop(self):
        pt = np.array([3.0, 4.0, 5.0], np.float32)
        result = world_to_local_point(pt, np.eye(4))
        assert np.allclose(result, pt, atol=1e-6)

    def test_translate_shifts_point(self):
        wt = mat_translate([5.0, 0.0, 0.0])
        pt = np.array([10.0, 0.0, 0.0], np.float32)
        result = world_to_local_point(pt, wt)
        # world point 10 is at local 10-5 = 5
        assert np.allclose(result, [5.0, 0.0, 0.0], atol=1e-6)

    def test_rotate90_z_center(self):
        wt = mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        # World point [1, 0, 0]: after inverse rotation (−90°) → [0, 1, 0]? No:
        # The WORLD transform takes LOCAL→WORLD. Inverse takes WORLD→LOCAL.
        # Rotate 90 about Z: L→W: (x,y)→(-y,x). W→L (inverse): (x,y)→(y,-x).
        # So world [1,0,0] → local [0,-1,0].
        pt = np.array([1.0, 0.0, 0.0], np.float32)
        result = world_to_local_point(pt, wt)
        assert np.allclose(result, [0.0, -1.0, 0.0], atol=1e-5), (
            f"Expected [0, -1, 0], got {result}"
        )


# ---------------------------------------------------------------------------
# Full tool-drive: MoveTool inside a translated group
# ---------------------------------------------------------------------------

class TestMoveToolInsideTranslatedGroup:
    """Drive MoveTool while the active context is a group translated by +5 on X.

    The snap engine gives WORLD positions.  The mesh vertices are in LOCAL space.
    A world delta of [10, 0, 0] should move the local vertex by [10, 0, 0]
    (pure translation → inverse 3×3 = identity → no change in direction).
    """

    def _setup(self):
        model = Model()
        # Build a group definition with one vertex at local [1, 0, 0].
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        vid = scene.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))

        # Place the group at +5 on X.
        grp_inst = model.new_instance(grp_def, mat_translate([5.0, 0.0, 0.0]))
        model.root.children.append(grp_inst)

        # Enter the group so active_context == grp_def.
        model.enter(grp_inst)

        return model, scene, vid

    def test_entity_move_world_delta_translates_local_vertex_correctly(self, qtbot):
        model, scene, vid = self._setup()

        sel = Selection()
        # Select the vertex's edges/faces — or just rely on vertex selection via face.
        # Simplest: add a face so selection_vertices works.
        b = scene.add_vertex(np.array([2.0, 0.0, 0.0], np.float32))
        c = scene.add_vertex(np.array([1.0, 1.0, 0.0], np.float32))
        fid = scene.add_face_from_loop([vid, b, c])
        sel.replace(faces=[fid])

        stack = CommandStack()
        tool = MoveTool()
        tool.activate(_ctx(scene, stack, sel, model=model))

        # World grab point. The group is at world +5X so local origin is at world +5X.
        # Snap grab at world [5, 0, 0] (= local [0,0,0]).
        # Snap release at world [15, 0, 0] → world delta = [10, 0, 0].
        tool.on_mouse_press(_press(), _snap([5.0, 0.0, 0.0]))
        tool.on_mouse_release(_release(), _snap([15.0, 0.0, 0.0]))

        # Pure translation context → local delta == world delta = [10, 0, 0].
        assert np.allclose(
            scene.vertex(vid).position,
            [1.0 + 10.0, 0.0, 0.0],
            atol=1e-5,
        ), f"Got {scene.vertex(vid).position}"

    def test_entity_move_inside_rotated_group_uses_local_delta(self, qtbot):
        """World delta [10,0,0] through a 90° rotation context → local delta [0,-10,0]."""
        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        vid = scene.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
        b = scene.add_vertex(np.array([2.0, 0.0, 0.0], np.float32))
        c = scene.add_vertex(np.array([1.0, 1.0, 0.0], np.float32))
        fid = scene.add_face_from_loop([vid, b, c])

        # Group rotated 90° CCW about Z → L→W: (x,y)→(-y,x).
        wt = mat_rotate([0, 0, 0], [0, 0, 1], math.radians(90))
        grp_inst = model.new_instance(grp_def, wt)
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection(); sel.replace(faces=[fid])
        stack = CommandStack()
        tool = MoveTool()
        tool.activate(_ctx(scene, stack, sel, model=model))

        # World delta [10, 0, 0] → local should be [0, -10, 0].
        tool.on_mouse_press(_press(), _snap([0.0, 0.0, 0.0]))
        tool.on_mouse_release(_release(), _snap([10.0, 0.0, 0.0]))

        expected_local_delta = np.array([0.0, -10.0, 0.0], np.float32)
        assert np.allclose(
            scene.vertex(vid).position,
            np.array([1.0, 0.0, 0.0]) + expected_local_delta,
            atol=1e-4,
        ), f"Got {scene.vertex(vid).position}"


# ---------------------------------------------------------------------------
# Full tool-drive: RotateTool inside a translated group
# ---------------------------------------------------------------------------

class TestRotateToolInsideTranslatedGroup:
    """Drive RotateTool with a non-identity context to confirm center/normal
    are correctly localised before rotating mesh vertices."""

    def test_rotate_inside_translated_group(self, qtbot):
        """A vertex at local [1,0,0] rotated 90° about local origin (Z axis)
        should end at local [0,1,0] regardless of whether the group is translated."""
        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        # Single edge: [1,0,0] → [2,0,0] in local space.
        a = scene.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
        b = scene.add_vertex(np.array([2.0, 0.0, 0.0], np.float32))
        eid = scene.add_edge(a, b)

        # Group placed at +5 on X in world.
        grp_inst = model.new_instance(grp_def, mat_translate([5.0, 0.0, 0.0]))
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection(); sel.replace(edges=[eid])
        stack = CommandStack()
        tool = RotateTool()
        tool.activate(_ctx(scene, stack, sel, model=model))

        # The center is at WORLD [5+0, 0, 0] = [5,0,0] (= local origin).
        # Start dir: WORLD [5+1, 0, 0] = [6,0,0].
        # End dir: WORLD [5+0, 1, 0] = [5,1,0].
        # Monkeypatch _pick_plane_normal to return world +Z.
        tool._pick_plane_normal = lambda ev: np.array([0.0, 0.0, 1.0], np.float32)

        tool.on_mouse_press(_press(), _snap([5.0, 0.0, 0.0]))   # center = world [5,0,0]
        tool.on_mouse_press(_press(), _snap([6.0, 0.0, 0.0]))   # start dir = world +X
        tool.on_mouse_press(_press(), _snap([5.0, 1.0, 0.0]))   # end dir = world +Y → +90°

        # Local vertex 'a' was at [1,0,0]; after 90° about Z through origin → [0,1,0].
        assert np.allclose(scene.vertex(a).position, [0.0, 1.0, 0.0], atol=1e-4), (
            f"Got {scene.vertex(a).position}"
        )
        assert np.allclose(scene.vertex(b).position, [0.0, 2.0, 0.0], atol=1e-4), (
            f"Got {scene.vertex(b).position}"
        )
