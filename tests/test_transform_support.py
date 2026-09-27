"""Selection→vertex-set, AABB, and scale-grip geometry helpers."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene
from bermake.selection import Selection
from bermake.tools.transform_support import (
    grip_specs,
    selection_aabb,
    selection_vertices,
)


def _square(scene: Scene):
    a = scene.add_vertex(np.array([0, 0, 0], np.float32))
    b = scene.add_vertex(np.array([2, 0, 0], np.float32))
    c = scene.add_vertex(np.array([2, 2, 0], np.float32))
    d = scene.add_vertex(np.array([0, 2, 0], np.float32))
    f = scene.add_face_from_loop([a, b, c, d])
    return a, b, c, d, f


def test_selection_vertices_from_face_is_loop():
    s = Scene()
    a, b, c, d, f = _square(s)
    sel = Selection()
    sel.replace(faces=[f])
    assert sorted(selection_vertices(s, sel)) == sorted([a, b, c, d])


def test_selection_vertices_from_edge_is_two_endpoints():
    s = Scene()
    a, b, c, d, f = _square(s)
    e = s.face_edges(f)[0]  # an edge of the square
    sel = Selection()
    sel.replace(edges=[e])
    verts = selection_vertices(s, sel)
    assert len(verts) == 2
    assert set(verts) <= {a, b, c, d}


def test_selection_vertices_dedups_shared():
    s = Scene()
    a, b, c, d, f = _square(s)
    edges = s.face_edges(f)
    sel = Selection()
    sel.replace(edges=edges, faces=[f])
    assert sorted(selection_vertices(s, sel)) == sorted([a, b, c, d])


def test_selection_aabb():
    s = Scene()
    a, b, c, d, _f = _square(s)
    lo, hi = selection_aabb(s, [a, b, c, d])
    assert np.allclose(lo, [0, 0, 0])
    assert np.allclose(hi, [2, 2, 0])


def test_selection_aabb_empty_is_none():
    s = Scene()
    assert selection_aabb(s, []) is None


def test_grip_specs_planar_box_has_eight_grips():
    lo = np.array([0, 0, 0], np.float32)
    hi = np.array([2, 2, 0], np.float32)
    grips = grip_specs(lo, hi)
    positions = [tuple(np.round(g.position, 4)) for g in grips]
    assert len(positions) == len(set(positions))   # all distinct
    assert len(grips) == 8


def test_grip_specs_full_box_has_26_grips():
    lo = np.array([0, 0, 0], np.float32)
    hi = np.array([2, 2, 2], np.float32)
    grips = grip_specs(lo, hi)
    assert len(grips) == 26   # 8 corners + 12 edges + 6 faces
    by_pos = {tuple(np.round(g.position, 4)): g for g in grips}
    for g in grips:
        opp = tuple(np.round(g.opposite, 4))
        assert opp in by_pos


def test_grip_specs_corner_axes_are_all_three():
    lo = np.array([0, 0, 0], np.float32)
    hi = np.array([2, 2, 2], np.float32)
    grips = grip_specs(lo, hi)
    corners = [g for g in grips if len(g.axes) == 3]
    assert len(corners) == 8
