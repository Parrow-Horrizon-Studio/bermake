"""Tests for picking.world_to_local_point — identity no-op and translation case.

Also includes an integration test: LineTool inside a +10X-translated group
must place the vertex at LOCAL (0, 2, 3), not world (10, 2, 3).
"""

from __future__ import annotations

import numpy as np
import pytest

from bermake.viewport.picking import world_to_local_point
from bermake.geometry.transforms import mat_translate


# ---------------------------------------------------------------------------
# Helper tests for world_to_local_point
# ---------------------------------------------------------------------------

def test_none_transform_returns_point_unchanged():
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = world_to_local_point(p, None)
    np.testing.assert_allclose(out, p, atol=1e-6)


def test_identity_transform_returns_point_unchanged():
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = world_to_local_point(p, np.eye(4, dtype=np.float64))
    np.testing.assert_allclose(out, p, atol=1e-6)


def test_translate_plus10x_world_10_2_3_becomes_local_0_2_3():
    """world_transform = translate(+10, 0, 0).
    A world-space point at (10, 2, 3) must map to local (0, 2, 3)."""
    wt = mat_translate(np.array([10.0, 0.0, 0.0]))
    world_pt = np.array([10.0, 2.0, 3.0], dtype=np.float32)
    local = world_to_local_point(world_pt, wt)
    np.testing.assert_allclose(local, [0.0, 2.0, 3.0], atol=1e-5)


def test_return_dtype_is_float32_regardless_of_identity_branch():
    """Both the identity no-op path and the apply_mat (float64-internal) path
    must return float32, so callers get a consistent dtype either way."""
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert world_to_local_point(p, None).dtype == np.float32
    assert world_to_local_point(p, np.eye(4, dtype=np.float64)).dtype == np.float32

    wt = mat_translate(np.array([10.0, 0.0, 0.0]))
    assert world_to_local_point(p, wt).dtype == np.float32


# ---------------------------------------------------------------------------
# Integration: LineTool write path in a translated group
# ---------------------------------------------------------------------------

def _grid_snap(world):
    from bermake.viewport.snap_engine import SnapKind, SnapResult
    return SnapResult(
        kind=SnapKind.GRID,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=None,
        label="Grid",
    )


def test_line_tool_writes_local_vertex_inside_translated_group():
    """LineTool inside a group translated by +10 in X:
    Clicking at world (10, 0, 0) should place a vertex at LOCAL (0, 0, 0).
    Without the fix it would write (10, 0, 0) into the local mesh — double-offset.
    """
    from bermake.model.model import Model
    from bermake.tools.line_tool import LineTool
    from bermake.tools.tool import ToolContext

    # Build a model with one group translated +10 on X.
    model = Model()
    d = model.new_definition("G", is_group=True)
    inst = model.new_instance(d, mat_translate([10.0, 0.0, 0.0]))
    model.root.children.append(inst)

    # Enter the group — active_world_transform is now translate(+10, 0, 0).
    model.enter(inst)
    assert not np.allclose(model.active_world_transform, np.eye(4))

    # Activate the LineTool in the group context.
    tool = LineTool()
    tool.activate(ToolContext(scene=model.active_scene, model=model))

    # Snap position is in WORLD space: (10, 0, 0).
    tool.on_mouse_press(None, _grid_snap((10.0, 0.0, 0.0)))

    # The vertex should be at LOCAL (0, 0, 0) in the group mesh.
    verts = list(model.active_scene.vertices_iter())
    assert len(verts) == 1
    np.testing.assert_allclose(verts[0].position, [0.0, 0.0, 0.0], atol=1e-5)
