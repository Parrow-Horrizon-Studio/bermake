"""LineTool splits an edge when a vertex lands on its interior."""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.line_tool import LineTool
from bermake.tools.tool import ToolContext
from bermake.viewport.snap_engine import SnapKind, SnapResult


class _FakeEvent:
    """LineTool.on_mouse_press never touches the event; a stub suffices."""


def _snap(kind, pos, **kw):
    return SnapResult(
        kind=kind, world_position=np.array(pos, dtype=np.float32),
        axis=kw.get("axis"), vertex_id=kw.get("vertex_id"),
        label=kw.get("label", ""), edge_id=kw.get("edge_id"),
        face_id=kw.get("face_id"), edge_t=kw.get("edge_t"),
    )


def _make_tool(scene):
    stack = CommandStack()
    tool = LineTool()
    tool.activate(ToolContext(scene=scene, command_stack=stack,
                              camera=None, widget_size_provider=None))
    return tool, stack


def test_line_onto_edge_interior_splits_it():
    scene = Scene()
    a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    b = scene.add_vertex(np.array([4, 0, 0], dtype=np.float32))
    e = scene.add_edge(a, b)
    tool, _ = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.GRID, [0, 2, 0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ON_EDGE, [2, 0, 0], edge_id=e, edge_t=0.5))

    positions = [tuple(round(float(c), 4) for c in v.position) for v in scene.vertices_iter()]
    assert (2.0, 0.0, 0.0) in positions
    assert not scene._mesh.edge_is_live(e)  # original edge replaced by the split


def test_line_endpoint_snap_reuses_vertex():
    scene = Scene()
    a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    tool, _ = _make_tool(scene)
    before = sum(1 for _ in scene.vertices_iter())
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.GRID, [1, 1, 0]))
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ENDPOINT, [0, 0, 0], vertex_id=a))
    assert sum(1 for _ in scene.vertices_iter()) == before + 1


def test_line_onto_edge_near_endpoint_reuses_endpoint_no_tjunction():
    """A degenerate ON_EDGE snap (edge_t at an endpoint → split is a no-op) must
    reuse the existing endpoint rather than drop a free vertex on the edge.

    The world_position is deliberately set slightly off the endpoint (0.001, 0, 0)
    to expose the bug: the old fallback called AddVertexCommand(world_position) which
    creates a NEW vertex near-but-not-on endpoint 'a', forming a T-junction.
    The fix must reuse 'a' (nearest endpoint) instead.
    """
    scene = Scene()
    a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    b = scene.add_vertex(np.array([4, 0, 0], dtype=np.float32))
    e = scene.add_edge(a, b)
    tool, _ = _make_tool(scene)

    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.GRID, [0, 2, 0]))
    # edge_t=0.0 → C++ split_edge rejects (t must be in (0,1)) → no-op fallback.
    # world_position is slightly off the endpoint to expose the T-junction bug.
    tool.on_mouse_press(_FakeEvent(), _snap(SnapKind.ON_EDGE, [0.001, 0, 0], edge_id=e, edge_t=0.0))

    # The original edge is untouched (no split, no T-junction)...
    assert scene._mesh.edge_is_live(e)
    # ...and NO new vertex was created on the edge — only a (the reused endpoint),
    # b, and the (0,2,0) grid vertex exist.
    assert sum(1 for _ in scene.vertices_iter()) == 3

    # The connecting edge must run from the grid vertex to endpoint `a` (not a
    # free-floating near-duplicate vertex).
    grid_vid = next(v.id for v in scene.vertices_iter()
                    if tuple(round(float(c), 4) for c in v.position) == (0.0, 2.0, 0.0))
    assert scene._mesh.edge_is_live(scene.add_edge(grid_vid, a))
