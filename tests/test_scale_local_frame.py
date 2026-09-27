"""Tests for the transform-awareness fix in ScaleTool (M4e Scale entity-mode).

Validates that the screen-space operations (grip picking, cursor projection,
overlay) correctly use WORLD positions when editing inside a translated
group/component, while the factor math and vertex write still operate in LOCAL.

Key invariants
--------------
* At IDENTITY (root context) every conversion is a no-op — existing behaviour
  is fully preserved.
* _grip_world_pos: a local grip at [2, 1, 0] inside a group translated +10 on X
  should report world position [12, 1, 0].
* _cursor_world: with _grips_are_local and a known world transform, the returned
  cursor is in LOCAL space (so factor math compares apples to apples).
* End-to-end entity scale inside a translated group produces the same LOCAL
  vertex result as scaling at the root with the same numerical cursor/factor.

Screen-fixture notes
--------------------
The actual screen picking (_pick_grip → camera.world_to_screen) and the full
overlay render path require a live camera and a viewport, which are not
available in this headless pytest environment.  Those paths are therefore
tested via monkeypatching / helper inspection:
  - _grip_world_pos is unit-tested directly (pure math, no camera).
  - _cursor_world is verified by driving the tool with a monkeypatched camera
    (ray_from_screen stubbed) and asserting the LOCAL result.
  - The end-to-end commit path is driven via monkeypatched _pick_grip and
    _cursor_world (as in test_scale_tool.py), confirming factor math is
    unaffected by the new frame flag.
The overlay world-position output is implicitly covered: if _grip_world_pos
returns correct world coords (unit-tested below), and overlay() calls it, the
markers will be at the right world positions.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from bermake.commands.command_stack import CommandStack
from bermake.geometry.transforms import apply_mat, mat_compose, mat_rotate, mat_translate
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.scale_tool import ScaleTool
from bermake.tools.tool import ToolContext
from bermake.tools.transform_support import GripSpec
from bermake.viewport.picking import world_to_local_point


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


def _ctx(scene, stack, selection, model=None, camera=None):
    return ToolContext(
        scene=scene, command_stack=stack, camera=camera,
        widget_size_provider=lambda: (800, 600), selection=selection,
        model=model,
    )


def _square(s: Scene):
    a = s.add_vertex(np.array([0, 0, 0], np.float32))
    b = s.add_vertex(np.array([2, 0, 0], np.float32))
    c = s.add_vertex(np.array([2, 2, 0], np.float32))
    d = s.add_vertex(np.array([0, 2, 0], np.float32))
    f = s.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def _make_translated_model(tx: float):
    """Return (model, scene, face_id, vertex_ids) for a group translated by tx on X."""
    model = Model()
    grp_def = model.new_definition("Grp", is_group=True)
    scene = grp_def.mesh
    vids_and_fid = _square(scene)
    a, b, c, d, fid = vids_and_fid
    grp_inst = model.new_instance(grp_def, mat_translate([tx, 0.0, 0.0]))
    model.root.children.append(grp_inst)
    model.enter(grp_inst)
    return model, scene, fid, [a, b, c, d]


# ---------------------------------------------------------------------------
# Unit tests: _grip_world_pos helper math
# ---------------------------------------------------------------------------

class TestGripWorldPos:
    """_grip_world_pos must lift local grip positions to world when _grips_are_local."""

    def _tool_with_local_grip(self, grip_pos, tx):
        """Build a minimal ScaleTool state: entity-mode, one grip at local grip_pos,
        active context translated by tx on X."""
        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        a, b, c, d, fid = _square(scene)
        grp_inst = model.new_instance(grp_def, mat_translate([tx, 0.0, 0.0]))
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection(); sel.replace(faces=[fid])
        stack = CommandStack()
        tool = ScaleTool()
        tool.activate(_ctx(scene, stack, sel, model=model))
        return tool, model

    def test_identity_noop(self):
        """At root (identity wt), local == world."""
        tool = ScaleTool()
        tool._model = None  # identity / root
        tool._grips_are_local = True
        g = GripSpec(position=np.array([2.0, 1.0, 0.0], np.float32),
                     opposite=np.array([0.0, 1.0, 0.0], np.float32), axes=(0,))
        result = tool._grip_world_pos(g)
        assert np.allclose(result, [2.0, 1.0, 0.0], atol=1e-6), f"Got {result}"

    def test_translate_lifts_grip_to_world(self):
        """Local grip at [2,1,0] inside a group translated +10X → world [12,1,0]."""
        tx = 10.0
        tool, model = self._tool_with_local_grip([2.0, 1.0, 0.0], tx)
        # The tool built grips from the local AABB of the unit square [0,2]x[0,2].
        # Find the grip at local [2,1,0] (face-center on +X face).
        local_pos = np.array([2.0, 1.0, 0.0], np.float32)
        g = GripSpec(position=local_pos,
                     opposite=np.array([0.0, 1.0, 0.0], np.float32), axes=(0,))
        # _grips_are_local should be True since we're in entity-mode.
        assert tool._grips_are_local, "Expected entity-mode (local grips)"
        result = tool._grip_world_pos(g)
        expected = np.array([2.0 + tx, 1.0, 0.0], np.float32)
        assert np.allclose(result, expected, atol=1e-5), (
            f"Expected world pos {expected}, got {result}"
        )

    def test_instance_mode_grips_unchanged(self):
        """In instance-mode grips are already world; _grip_world_pos returns them as-is."""
        tool = ScaleTool()
        tool._model = None
        tool._grips_are_local = False  # instance-mode
        g = GripSpec(position=np.array([5.0, 3.0, 0.0], np.float32),
                     opposite=np.array([0.0, 3.0, 0.0], np.float32), axes=(0,))
        result = tool._grip_world_pos(g)
        assert np.allclose(result, [5.0, 3.0, 0.0], atol=1e-6), f"Got {result}"

    def test_apply_mat_reference(self):
        """Direct apply_mat sanity check: translating [2,1,0] by +10X gives [12,1,0]."""
        wt = mat_translate([10.0, 0.0, 0.0])
        local_pt = np.array([[2.0, 1.0, 0.0]], np.float64)
        world_pt = apply_mat(local_pt, wt)[0]
        assert np.allclose(world_pt, [12.0, 1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: _cursor_world returns LOCAL frame in entity-mode
# ---------------------------------------------------------------------------

class TestCursorWorldLocalFrame:
    """_cursor_world must return LOCAL coords in entity-mode so factor math works."""

    def _stub_camera(self, origin, direction):
        """Minimal camera stub: ray_from_screen always returns given origin+dir."""
        cam = types.SimpleNamespace(
            ray_from_screen=lambda x, y, w, h: (
                np.asarray(origin, np.float32),
                np.asarray(direction, np.float32),
            ),
            # Top-down camera: view direction is +Z, so the camera-facing cursor
            # plane normal equals world-Z — the frame these local tests assert in.
            position=np.array([0.0, 0.0, 100.0], np.float32),
            target=np.array([0.0, 0.0, 0.0], np.float32),
        )
        return cam

    def test_cursor_world_at_identity_is_unchanged(self):
        """At identity wt the local←world round-trip is a no-op."""
        # Camera ray from (0,0,100) looking straight down (−Z).
        origin = np.array([5.0, 5.0, 100.0], np.float32)
        direction = np.array([0.0, 0.0, -1.0], np.float32)
        cam = self._stub_camera(origin, direction)

        tool = ScaleTool()
        tool._model = None  # identity / root
        tool._grips_are_local = True
        tool._camera = cam
        tool._size_provider = lambda: (800, 600)
        grip = GripSpec(position=np.array([5.0, 5.0, 0.0], np.float32),
                        opposite=np.array([0.0, 0.0, 0.0], np.float32), axes=(0, 1))
        tool._active = grip
        tool._anchor = np.array([0.0, 0.0, 0.0], np.float32)

        event = _press(400.0, 300.0)
        result = tool._cursor_world(event)
        # Ray hits z=0 plane at [5,5,0]; at identity wt that is also local [5,5,0].
        assert result is not None
        assert np.allclose(result, [5.0, 5.0, 0.0], atol=1e-4), f"Got {result}"

    def test_cursor_world_translated_returns_local(self):
        """Inside a group translated +10X, a world cursor at [15,5,0]
        must be returned as local [5,5,0] (world − translation)."""
        tx = 10.0
        # Ray from high up, straight down, aimed at world x=15,y=5.
        origin = np.array([15.0, 5.0, 100.0], np.float32)
        direction = np.array([0.0, 0.0, -1.0], np.float32)
        cam = self._stub_camera(origin, direction)

        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        a, b, c, d, fid = _square(scene)
        grp_inst = model.new_instance(grp_def, mat_translate([tx, 0.0, 0.0]))
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection(); sel.replace(faces=[fid])
        stack = CommandStack()
        tool = ScaleTool()
        tool.activate(_ctx(scene, stack, sel, model=model, camera=cam))

        # Confirm entity-mode and local grips flag.
        assert tool._grips_are_local, "Expected entity-mode"

        # Place a grip at local [5,5,0]; its world pos is [15,5,0].
        grip = GripSpec(position=np.array([5.0, 5.0, 0.0], np.float32),
                        opposite=np.array([0.0, 0.0, 0.0], np.float32), axes=(0, 1))
        tool._active = grip

        event = _press(400.0, 300.0)
        result = tool._cursor_world(event)
        # The ray hits z=0 at world [15,5,0]; converted to local → [5,5,0].
        assert result is not None, "_cursor_world returned None"
        assert np.allclose(result, [5.0, 5.0, 0.0], atol=1e-4), (
            f"Expected local [5,5,0], got {result}"
        )

    def test_cursor_world_local_single_axis_uses_rotated_world_axis(self):
        """A single-axis (face) grip in LOCAL mode must project the cursor onto
        the transform-ROTATED world axis line (scale_tool.py's
        `if self._grips_are_local: axis_dir = wt[:3, ax]` branch), not the raw
        local basis vector. This exercises #47's new local-mode axis-rotation
        logic, which the plane/2-axis tests above never touch (they only use
        2-axis grips) and which the #47 test in test_scale_tool_cursor_plane.py
        never touches either (it uses `_grips_are_local=False`).

        A PURE TRANSLATION would not discriminate this branch: translation
        doesn't touch the transform's rotation part, so `wt[:3, ax]` would
        equal the plain unit basis vector regardless of whether the code used
        the (correct) rotated column or the (buggy) unrotated basis vector.
        The rotation is therefore essential to this test having teeth.
        """
        # Group transform: rotate +90 deg about world Z (local +X -> world +Y),
        # then translate +5 on world X.
        rot = mat_rotate((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), math.pi / 2.0)
        wt = mat_compose(rot, mat_translate([5.0, 0.0, 0.0]))

        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        _, _, _, _, fid = _square(scene)
        grp_inst = model.new_instance(grp_def, wt)
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection()
        sel.replace(faces=[fid])
        stack = CommandStack()

        # Ray straight down (-Z), offset from the axis line in world X (origin
        # x=1, while the axis line sits at world x=4 -- see below). The offset
        # matters: it's what makes the WRONG axis (unrotated local +X, i.e.
        # world +X) actually pull the projected point's x away from 4. With an
        # origin already at x=4, the wrong axis's projection would coincide
        # with p0 by symmetry and the test would pass even when broken.
        origin = np.array([1.0, 3.0, 100.0], np.float32)
        direction = np.array([0.0, 0.0, -1.0], np.float32)
        cam = self._stub_camera(origin, direction)

        tool = ScaleTool()
        tool.activate(_ctx(scene, stack, sel, model=model, camera=cam))
        assert tool._grips_are_local, "Expected entity-mode (local grips)"

        # Local single-axis (X) grip at [2, 1, 0]; its world position is
        # rot @ [2,1,0] + [5,0,0] = [-1, 2, 0] + [5,0,0] = [4, 2, 0]. Since
        # local +X maps to world +Y, the world axis line is x=4, z=0 (y free).
        grip = GripSpec(
            position=np.array([2.0, 1.0, 0.0], np.float32),
            opposite=np.array([0.0, 1.0, 0.0], np.float32),
            axes=(0,),
        )
        tool._active = grip

        event = _press(400.0, 300.0)
        local_result = tool._cursor_world(event)
        assert local_result is not None, "_cursor_world returned None"

        # _cursor_world returns LOCAL coords in entity-mode; lift back to
        # world to check against the world-space (rotated) axis line.
        world_result = apply_mat(
            np.asarray(local_result, np.float64).reshape(1, 3), wt
        )[0]
        assert abs(float(world_result[0]) - 4.0) < 1e-4, (
            f"world x should stay pinned to the axis line (4): {world_result}"
        )
        assert abs(float(world_result[2]) - 0.0) < 1e-4, (
            f"world z should stay pinned to the axis line (0): {world_result}"
        )


# ---------------------------------------------------------------------------
# End-to-end: ScaleTool entity-mode inside a translated group
# ---------------------------------------------------------------------------

class TestScaleEntityInsideTranslatedGroup:
    """Drive ScaleTool via monkeypatched _pick_grip + _cursor_world.

    The LOCAL verts should be scaled by the expected local factor,
    identical to what would happen at the root with the same numeric cursor.
    This confirms that the frame fix doesn't break the write path.
    """

    def test_entity_scale_2x_inside_translated_group(self, qtbot):
        """Select a [0,2]x[0,2] face inside a group at +10X.
        Scale the +X face grip from its local position [2,1,0] with anchor [0,1,0].
        Provide cursor at local [4,1,0] → factor x=2.
        Expect corner at local [2,2,0] to move to [4,2,0].
        """
        tx = 10.0
        model, scene, fid, (va, vb, vc, vd) = _make_translated_model(tx)

        sel = Selection(); sel.replace(faces=[fid])
        stack = CommandStack()
        tool = ScaleTool()
        tool.activate(_ctx(scene, stack, sel, model=model))

        # Grip on the +X face of the AABB: position=[2,1,0], anchor=[0,1,0], axis=X.
        grip = GripSpec(
            position=np.array([2.0, 1.0, 0.0], np.float32),
            opposite=np.array([0.0, 1.0, 0.0], np.float32),
            axes=(0,),
        )
        # Cursor at local [4,1,0] → scale factor x = (4-0)/(2-0) = 2.
        local_cursor = np.array([4.0, 1.0, 0.0], np.float32)

        from unittest.mock import patch
        with (
            patch.object(tool, "_pick_grip", return_value=grip),
            patch.object(tool, "_cursor_world", return_value=local_cursor),
        ):
            press = _press(0.0, 0.0)
            rel = _release(0.0, 0.0)
            tool.on_mouse_press(press, None)
            tool.on_mouse_release(rel, None)

        # The local verts: va=[0,0,0], vb=[2,0,0], vc=[2,2,0], vd=[0,2,0].
        # Anchor = grip.opposite = [0,1,0]; scale x by 2 from anchor x=0.
        # vb x: 2 → 4;  vc x: 2 → 4;  va x: 0 → 0;  vd x: 0 → 0.
        assert np.allclose(scene.vertex(vb).position, [4.0, 0.0, 0.0], atol=1e-5), (
            f"vb: expected [4,0,0], got {scene.vertex(vb).position}"
        )
        assert np.allclose(scene.vertex(vc).position, [4.0, 2.0, 0.0], atol=1e-5), (
            f"vc: expected [4,2,0], got {scene.vertex(vc).position}"
        )
        assert np.allclose(scene.vertex(va).position, [0.0, 0.0, 0.0], atol=1e-5), (
            f"va: expected [0,0,0] (anchor side unchanged), got {scene.vertex(va).position}"
        )
        assert stack.can_undo

    def test_entity_scale_at_root_same_result(self, qtbot):
        """Confirm that at root (identity wt) the same numerical cursor produces
        the same local result — regression guard for the identity no-op path."""
        scene = Scene()
        a, b, c, d, fid = _square(scene)
        sel = Selection(); sel.replace(faces=[fid])
        stack = CommandStack()
        tool = ScaleTool()
        tool.activate(_ctx(scene, stack, sel))

        grip = GripSpec(
            position=np.array([2.0, 1.0, 0.0], np.float32),
            opposite=np.array([0.0, 1.0, 0.0], np.float32),
            axes=(0,),
        )
        local_cursor = np.array([4.0, 1.0, 0.0], np.float32)

        from unittest.mock import patch
        with (
            patch.object(tool, "_pick_grip", return_value=grip),
            patch.object(tool, "_cursor_world", return_value=local_cursor),
        ):
            tool.on_mouse_press(_press(), None)
            tool.on_mouse_release(_release(), None)

        assert np.allclose(scene.vertex(b).position, [4.0, 0.0, 0.0], atol=1e-5)
        assert np.allclose(scene.vertex(c).position, [4.0, 2.0, 0.0], atol=1e-5)
        assert np.allclose(scene.vertex(a).position, [0.0, 0.0, 0.0], atol=1e-5)
        assert stack.can_undo
