"""Tests for selection picking (pure screen-space)."""

from __future__ import annotations

import numpy as np


def _camera(w, h):
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = float(w) / float(h)
    return cam


def test_pick_returns_edge_near_its_screen_projection():
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)

    mid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(mid, w, h)
    hit = pick_selectable((sx, sy), (w, h), cam, scene)
    assert hit == ("edge", e)


def test_pick_far_from_everything_is_none():
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    scene.add_edge(a, b)
    hit = pick_selectable((2.0, 2.0), (w, h), cam, scene)
    assert hit is None


def test_pick_prefers_edge_over_face_behind_it():
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))

    mid_ab = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(mid_ab, w, h)
    hit = pick_selectable((sx, sy), (w, h), cam, scene)
    assert hit[0] == "edge"

    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, _ = cam.world_to_screen(center, w, h)
    hit2 = pick_selectable((cx, cy), (w, h), cam, scene)
    assert hit2 == ("face", fid)


def _screen(cam, world, w, h):
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), w, h)
    return sx, sy


def test_window_selects_only_fully_enclosed():
    from bermake.scene import Scene
    from bermake.viewport.picking import entities_in_box

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-0.3, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.3, 0.0, 0.0], dtype=np.float32))
    e_in = scene.add_edge(a, b)
    c = scene.add_vertex(np.array([3.0, 0.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([3.5, 0.0, 0.0], dtype=np.float32))
    e_out = scene.add_edge(c, d)

    s1 = _screen(cam, [-0.3, 0.0, 0.0], w, h)
    s2 = _screen(cam, [0.3, 0.0, 0.0], w, h)
    margin = 10.0
    rect = (min(s1[0], s2[0]) - margin, min(s1[1], s2[1]) - margin,
            max(s1[0], s2[0]) + margin, max(s1[1], s2[1]) + margin)
    edges, faces, _vertices = entities_in_box(rect, "window", (w, h), cam, scene)
    assert e_in in edges
    assert e_out not in edges


def test_crossing_selects_straddling_edge_window_does_not():
    from bermake.scene import Scene
    from bermake.viewport.picking import entities_in_box

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-0.5, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.5, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)

    sa = _screen(cam, [-0.5, 0.0, 0.0], w, h)
    sb = _screen(cam, [0.5, 0.0, 0.0], w, h)
    midx = (sa[0] + sb[0]) / 2.0
    top = min(sa[1], sb[1]) - 20.0
    bot = max(sa[1], sb[1]) + 20.0
    left = min(sa[0], sb[0]) - 20.0
    rect = (left, top, midx, bot)

    win_edges, _, _ = entities_in_box(rect, "window", (w, h), cam, scene)
    cross_edges, _, _ = entities_in_box(rect, "crossing", (w, h), cam, scene)
    assert e not in win_edges
    assert e in cross_edges


def test_crossing_face_when_rect_is_inside_the_face():
    from bermake.scene import Scene
    from bermake.viewport.picking import entities_in_box

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    # A large face; corners project far apart on screen.
    a = scene.add_vertex(np.array([-4.0, -4.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([4.0, -4.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([4.0, 4.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([-4.0, 4.0, 0.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))

    # A tiny rect at the face center — inside the face, touching no vertex/edge.
    cx, cy, _ = cam.world_to_screen(np.array([0.0, 0.0, 0.0], dtype=np.float32), w, h)
    rect = (cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)

    _, cross_faces, _ = entities_in_box(rect, "crossing", (w, h), cam, scene)
    assert fid in cross_faces
    # Window mode must NOT select it (its corners are far outside the tiny rect).
    _, win_faces, _ = entities_in_box(rect, "window", (w, h), cam, scene)
    assert fid not in win_faces
