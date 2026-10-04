"""SelectTool.overlay() hover preview for each pickable kind (#125).

A hovered vertex draws an unfilled 7 px screen-space square in the hover blue,
and every hover arm (vertex, edge, face) must be expressed in WORLD space, so
inside an entered group the preview lands where the geometry is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

_OFFSET = np.array([10.0, 0.0, 0.0])


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = (x, y, z)
    return m


def _build(entered: bool):
    """A model whose root holds a group (translated by (10,0,0)) containing a
    unit square. Returns (tool, ids) with the group entered or not."""
    from bermake.model.model import Model
    from bermake.selection import Selection
    from bermake.tools import ToolContext
    from bermake.tools.select_tool import SelectTool
    from bermake.viewport.camera import Camera

    model = Model()
    group_def = model.new_definition("Group", is_group=True)
    scene = group_def.mesh
    a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    b = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    c = scene.add_vertex(np.array([2, 2, 0], dtype=np.float32))
    d = scene.add_vertex(np.array([0, 2, 0], dtype=np.float32))
    face = scene.add_face_from_loop((a, b, c, d))
    edge = scene.add_edge(a, b)
    inst = model.new_instance(group_def, _translation(*_OFFSET))
    model.root.children.append(inst)
    if entered:
        model.enter(inst)
    cam = Camera()
    cam.aspect = 800.0 / 600.0
    tool = SelectTool()
    tool.activate(
        ToolContext(
            scene=model.active_scene if entered else scene,
            camera=cam,
            widget_size_provider=lambda: (800, 600),
            selection=Selection(),
            model=model,
        )
    )
    return tool, {"vertex": b, "edge": edge, "face": face}


@pytest.mark.parametrize("entered", [False, True])
def test_hovered_vertex_yields_one_unfilled_marker(qtbot, entered):
    from bermake.tools.select_tool import _HOVER_EDGE_COLOR

    tool, ids = _build(entered)
    tool._hovered = ("vertex", ids["vertex"])
    ov = tool.overlay()
    assert len(ov.screen_markers) == 1
    pos, size, color = ov.screen_markers[0]
    local = np.array([2.0, 0.0, 0.0])
    expected = local + (_OFFSET if entered else 0.0)
    np.testing.assert_allclose(np.asarray(pos, dtype=np.float64), expected, atol=1e-5)
    assert size == 7
    assert tuple(color) == _HOVER_EDGE_COLOR
    assert ov.rubber_band_segments.shape[0] == 0
    assert ov.face_fill_polygons == []


def test_hovered_edge_inside_group_is_in_world_space(qtbot):
    tool, ids = _build(entered=True)
    tool._hovered = ("edge", ids["edge"])
    ov = tool.overlay()
    segs = np.asarray(ov.rubber_band_segments, dtype=np.float64)
    assert segs.shape == (2, 3)
    np.testing.assert_allclose(segs[0], [10.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(segs[1], [12.0, 0.0, 0.0], atol=1e-5)
    assert ov.screen_markers == []


def test_hovered_edge_at_root_is_unchanged(qtbot):
    tool, ids = _build(entered=False)
    tool._hovered = ("edge", ids["edge"])
    segs = np.asarray(tool.overlay().rubber_band_segments, dtype=np.float64)
    np.testing.assert_allclose(segs, [[0, 0, 0], [2, 0, 0]], atol=1e-5)


@pytest.mark.parametrize("entered", [False, True])
def test_hovered_face_fill_is_unchanged(qtbot, entered):
    tool, ids = _build(entered)
    tool._hovered = ("face", ids["face"])
    ov = tool.overlay()
    assert len(ov.face_fill_polygons) == 1
    poly = np.asarray(ov.face_fill_polygons[0], dtype=np.float64)
    expected = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float64)
    if entered:
        expected = expected + _OFFSET
    np.testing.assert_allclose(poly, expected, atol=1e-5)
    assert ov.screen_markers == []
    assert ov.rubber_band_segments.shape[0] == 0
