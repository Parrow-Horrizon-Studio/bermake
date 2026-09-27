"""Tests for anchor_or_none world-frame correctness (M4e cleanup).

Fix 1: LineTool.anchor_or_none must return a WORLD-space position when the
active context has a non-identity world transform (e.g. inside a translated
group).  At identity the value is unchanged.

Fix 2: RotateTool._disk_radius must use a LOCAL center when computing vertex
distances, because the stored vertex positions are LOCAL but self._center is
WORLD.  At identity (root context) this is a no-op.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from bermake.geometry.transforms import apply_mat, mat_translate
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.tools.line_tool import LineTool
from bermake.tools.rotate_tool import RotateTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_snap(pos):
    return types.SimpleNamespace(
        kind=SnapKind.GRID,
        world_position=np.asarray(pos, np.float32),
        axis=None,
        vertex_id=None,
        edge_id=None,
        edge_t=None,
        label="Grid",
    )


def _endpoint_snap(pos, vid):
    return types.SimpleNamespace(
        kind=SnapKind.ENDPOINT,
        world_position=np.asarray(pos, np.float32),
        axis=None,
        vertex_id=vid,
        edge_id=None,
        edge_t=None,
        label="Endpoint",
    )


# ---------------------------------------------------------------------------
# Fix 1 — LineTool.anchor_or_none world frame
# ---------------------------------------------------------------------------

class TestLineToolAnchorWorldFrame:
    """anchor_or_none must return the WORLD-space position of the last placed
    gesture vertex, not the LOCAL mesh position."""

    def test_identity_context_returns_local_position_unchanged(self):
        """At root context (identity), anchor == the vertex local position."""
        scene = Scene()
        tool = LineTool()
        tool.activate(ToolContext(scene=scene))

        # Place a vertex at local (and world) [2, 3, 0].
        tool.on_mouse_press(None, _grid_snap([2.0, 3.0, 0.0]))

        anchor = tool.anchor_or_none
        assert anchor is not None
        assert np.allclose(anchor, [2.0, 3.0, 0.0], atol=1e-6), (
            f"Expected [2, 3, 0], got {anchor}"
        )

    def test_translated_context_lifts_anchor_to_world(self):
        """Inside a group translated by +5 on X, placing a vertex at local
        [1, 0, 0] stores it in LOCAL space.  anchor_or_none must return the
        WORLD position [6, 0, 0] (= local + translation)."""
        translation = [5.0, 0.0, 0.0]
        wt = mat_translate(translation)

        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh

        grp_inst = model.new_instance(grp_def, wt)
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        tool = LineTool()
        # The snap engine gives us a WORLD position; the tool converts it to
        # local before calling AddVertexCommand.  We place the vertex at WORLD
        # [6, 0, 0] which is LOCAL [1, 0, 0] in this translated context.
        tool.activate(ToolContext(scene=scene, model=model))
        tool.on_mouse_press(None, _grid_snap([6.0, 0.0, 0.0]))

        # Confirm the vertex was stored at LOCAL [1, 0, 0].
        v_list = list(scene.vertices_iter())
        assert len(v_list) == 1
        assert np.allclose(v_list[0].position, [1.0, 0.0, 0.0], atol=1e-6), (
            f"Local vertex position wrong: {v_list[0].position}"
        )

        anchor = tool.anchor_or_none
        assert anchor is not None
        # anchor_or_none must return WORLD [6, 0, 0], not LOCAL [1, 0, 0].
        assert np.allclose(anchor, [6.0, 0.0, 0.0], atol=1e-5), (
            f"Expected world anchor [6, 0, 0], got {anchor}"
        )

    def test_anchor_none_when_idle(self):
        """Before the first click anchor_or_none is None."""
        scene = Scene()
        tool = LineTool()
        tool.activate(ToolContext(scene=scene))
        assert tool.anchor_or_none is None

    def test_identity_math_noop(self):
        """Direct math check: applying identity wt to a local position is a no-op."""
        local = np.array([3.0, 4.0, 5.0], np.float32)
        wt = np.eye(4, dtype=np.float64)
        result = apply_mat(local.astype(np.float64).reshape(1, 3), wt)[0]
        assert np.allclose(result, local, atol=1e-7)

    def test_translation_math_lifts_correctly(self):
        """Direct math check: apply_mat with a translation wt shifts the point."""
        local = np.array([1.0, 0.0, 0.0], np.float32)
        wt = mat_translate([5.0, 0.0, 0.0])
        result = apply_mat(local.astype(np.float64).reshape(1, 3), wt)[0]
        assert np.allclose(result, [6.0, 0.0, 0.0], atol=1e-7), f"Got {result}"


# ---------------------------------------------------------------------------
# Fix 2 — RotateTool._disk_radius local center
# ---------------------------------------------------------------------------

class TestRotateToolDiskRadius:
    """_disk_radius must compute distances in LOCAL space so the protractor
    size is correct when editing inside a translated group."""

    def _setup_tool_in_translated_group(self, translation):
        """Build a group with two vertices, enter it, arm the RotateTool with a
        world center equal to (translation + local origin), and return the tool
        plus the expected local radius."""
        from bermake.commands.command_stack import CommandStack
        from bermake.selection import Selection

        wt = mat_translate(translation)
        model = Model()
        grp_def = model.new_definition("Grp", is_group=True)
        scene = grp_def.mesh
        # Vertices at LOCAL [1,0,0] and [0,1,0].
        a = scene.add_vertex(np.array([1.0, 0.0, 0.0], np.float32))
        b = scene.add_vertex(np.array([0.0, 1.0, 0.0], np.float32))
        eid = scene.add_edge(a, b)

        grp_inst = model.new_instance(grp_def, wt)
        model.root.children.append(grp_inst)
        model.enter(grp_inst)

        sel = Selection()
        sel.replace(edges=[eid])
        stack = CommandStack()
        tool = RotateTool()
        tool.activate(ToolContext(
            scene=scene, command_stack=stack, camera=None,
            widget_size_provider=lambda: (800, 600), selection=sel, model=model,
        ))
        return tool, scene, translation

    def test_disk_radius_identity(self):
        """At identity the center is LOCAL == WORLD, so radius is just max(|v-center|)."""
        from bermake.commands.command_stack import CommandStack
        from bermake.selection import Selection

        scene = Scene()
        a = scene.add_vertex(np.array([3.0, 0.0, 0.0], np.float32))
        b = scene.add_vertex(np.array([0.0, 4.0, 0.0], np.float32))
        eid = scene.add_edge(a, b)
        sel = Selection(); sel.replace(edges=[eid])
        stack = CommandStack()
        tool = RotateTool()
        tool.activate(ToolContext(
            scene=scene, command_stack=stack, camera=None,
            widget_size_provider=lambda: (800, 600), selection=sel,
        ))
        # Capture vertices before clicking center.
        tool._orig = {a: np.array([3.0, 0.0, 0.0], np.float32),
                      b: np.array([0.0, 4.0, 0.0], np.float32)}
        tool._center = np.array([0.0, 0.0, 0.0], np.float32)

        r = tool._disk_radius()
        # max distance from origin: max(3, 4) = 4.
        assert abs(r - 4.0) < 1e-5, f"Expected 4.0, got {r}"

    def test_disk_radius_translated_group_uses_local_center(self):
        """Inside a group translated by [5,0,0], the world center [5,0,0]
        is LOCAL [0,0,0].  Vertices at local [3,0,0] and [0,4,0].
        _disk_radius must return 4.0 (max local dist), not some larger value
        caused by mixing LOCAL pts with a WORLD center."""
        tool, scene, translation = self._setup_tool_in_translated_group([5.0, 0.0, 0.0])
        # Manually set the orig dict and world center (simulating what the
        # tool does during HAVE_CENTER stage).
        tool._orig = {
            0: np.array([1.0, 0.0, 0.0], np.float32),
            1: np.array([0.0, 1.0, 0.0], np.float32),
        }
        # World center = local origin = [0,0,0] + translation = [5,0,0].
        tool._center = np.array([5.0, 0.0, 0.0], np.float32)

        r = tool._disk_radius()
        # max local dist from local [0,0,0]: max(|(1,0,0)|, |(0,1,0)|) = 1.0
        assert abs(r - 1.0) < 1e-5, (
            f"Expected local radius 1.0, got {r} "
            "(mixing world center with local pts gives wrong value)"
        )

    def test_disk_radius_fallback_no_orig(self):
        """Returns 1.0 when _orig is empty (e.g. immediately after reset)."""
        from bermake.commands.command_stack import CommandStack
        from bermake.selection import Selection

        scene = Scene()
        sel = Selection()
        stack = CommandStack()
        tool = RotateTool()
        tool.activate(ToolContext(
            scene=scene, command_stack=stack, camera=None,
            widget_size_provider=lambda: (800, 600), selection=sel,
        ))
        assert tool._disk_radius() == 1.0
