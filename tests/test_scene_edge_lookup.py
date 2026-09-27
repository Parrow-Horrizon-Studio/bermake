"""Scene's non-mutating edge accessors (M7.4 Task 2, closes #26)."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene


def _two_vertices(scene):
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    return a, b


def test_edge_between_finds_an_existing_edge_in_either_order():
    scene = Scene()
    a, b = _two_vertices(scene)
    e = scene.add_edge(a, b)
    assert scene.edge_between(a, b) == e
    assert scene.edge_between(b, a) == e


def test_edge_between_returns_none_when_absent():
    scene = Scene()
    a, b = _two_vertices(scene)
    assert scene.edge_between(a, b) is None


def test_edge_between_does_not_create_an_edge():
    # The whole point of #26: the old code called add_halfedge_pair as a
    # lookup, which would have created the edge this test asserts is absent.
    scene = Scene()
    a, b = _two_vertices(scene)
    scene.edge_between(a, b)
    assert scene.edge_between(a, b) is None
    assert list(scene.edges_iter()) == []


def test_edge_is_live_tracks_removal():
    scene = Scene()
    a, b = _two_vertices(scene)
    e = scene.add_edge(a, b)
    assert scene.edge_is_live(e) is True
    scene.remove_edge(e)
    assert scene.edge_is_live(e) is False
