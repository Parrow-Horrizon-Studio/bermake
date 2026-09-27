"""Tests for shape_support: plane resolution + world-points → CompositeCommand."""

from __future__ import annotations

import numpy as np


def _snap(kind, world, *, face_id=None, vertex_id=None):
    from bermake.viewport.snap_engine import SnapResult

    return SnapResult(
        kind=kind,
        world_position=np.array(world, dtype=np.float32),
        axis=None,
        vertex_id=vertex_id,
        label="t",
        face_id=face_id,
    )


def test_resolve_plane_defaults_to_horizontal_through_point():
    from bermake.scene import Scene
    from bermake.tools.shape_support import resolve_drawing_plane
    from bermake.viewport.snap_engine import SnapKind

    plane = resolve_drawing_plane(_snap(SnapKind.ENDPOINT, (1.0, 2.0, 5.0)), Scene())
    assert np.allclose(plane.normal, [0.0, 0.0, 1.0])
    assert np.allclose(plane.origin, [1.0, 2.0, 5.0])


def test_resolve_plane_uses_face_for_on_face_snap():
    from bermake.scene import Scene
    from bermake.tools.shape_support import resolve_drawing_plane
    from bermake.viewport.snap_engine import SnapKind

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.0, 2.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([0.0, 2.0, 2.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 0.0, 2.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))

    plane = resolve_drawing_plane(_snap(SnapKind.ON_FACE, (0.0, 1.0, 1.0), face_id=fid), scene)
    assert abs(abs(float(plane.normal[0])) - 1.0) < 1e-6  # ±X


def test_build_closed_face_creates_ring_face_and_undoes_atomically():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools.shape_support import build_closed_face

    scene = Scene()
    stack = CommandStack()
    pts = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
    composite = build_closed_face(scene, pts, name="X")
    assert composite is not None
    stack.push_executed(composite, scene)

    assert len(list(scene.vertices_iter())) == 4
    assert len(list(scene.edges_iter())) == 4
    assert len(list(scene.faces_iter())) == 1

    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.faces_iter())) == 0

    stack.redo()
    assert len(list(scene.vertices_iter())) == 4
    assert len(list(scene.faces_iter())) == 1


def test_build_closed_face_reuses_coincident_existing_vertex():
    from bermake.scene import Scene
    from bermake.tools.shape_support import build_closed_face

    scene = Scene()
    existing = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    pts = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=np.float32)
    build_closed_face(scene, pts, name="X")
    assert len(list(scene.vertices_iter())) == 4
    assert any(v.id == existing for v in scene.vertices_iter())


def test_build_open_polyline_creates_edges_no_face():
    from bermake.commands import CommandStack
    from bermake.scene import Scene
    from bermake.tools.shape_support import build_open_polyline

    scene = Scene()
    stack = CommandStack()
    pts = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=np.float32)
    result = build_open_polyline(scene, pts, name="A")
    assert result is not None
    composite, vids = result
    assert composite.children
    assert len(vids) == 3
    stack.push_executed(composite, scene)
    assert len(list(scene.vertices_iter())) == 3
    assert len(list(scene.edges_iter())) == 2
    assert len(list(scene.faces_iter())) == 0

    stack.undo()
    assert len(list(scene.vertices_iter())) == 0
    assert len(list(scene.edges_iter())) == 0
    stack.redo()
    assert len(list(scene.edges_iter())) == 2
    assert len(list(scene.faces_iter())) == 0


def test_polyline_segments_closed_and_open():
    from bermake.tools.shape_support import polyline_segments

    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32)
    closed = polyline_segments(pts, closed=True)
    assert closed.shape == (6, 3)
    opened = polyline_segments(pts, closed=False)
    assert opened.shape == (4, 3)
