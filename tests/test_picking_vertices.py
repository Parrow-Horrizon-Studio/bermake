"""Vertex picking is opt-in. With the flag off, nothing changes."""

from __future__ import annotations

import numpy as np


def _camera(w, h):
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = float(w) / float(h)
    return cam


def _quad():
    from bermake.scene import Scene

    scene = Scene()
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    f = scene.add_face_from_loop((a, b, c, d))
    return scene, {"a": a, "b": b, "c": c, "d": d, "face": f}


def test_with_the_flag_off_a_corner_still_picks_the_edge():
    """The regression gate for the whole milestone. Every existing tool picks
    through this function with the flag unset."""
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene, ids = _quad()
    corner = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(corner, w, h)
    kind, _ent = pick_selectable((sx, sy), (w, h), cam, scene)
    assert kind == "edge"


def test_with_the_flag_on_a_corner_picks_the_vertex():
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene, ids = _quad()
    corner = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(corner, w, h)
    assert pick_selectable((sx, sy), (w, h), cam, scene, select_vertices=True) == (
        "vertex",
        ids["a"],
    )


def test_with_the_flag_on_a_midspan_click_still_picks_the_edge():
    """Vertex priority must not swallow the edge everywhere, only near a
    corner. Midspan is well outside PICK_PIXEL_TOLERANCE of either end."""
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene, _ids = _quad()
    mid = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(mid, w, h)
    kind, _ent = pick_selectable((sx, sy), (w, h), cam, scene, select_vertices=True)
    assert kind == "edge"


def test_with_the_flag_on_a_face_centre_still_picks_the_face():
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene, ids = _quad()
    centre = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(centre, w, h)
    assert pick_selectable((sx, sy), (w, h), cam, scene, select_vertices=True) == (
        "face",
        ids["face"],
    )


def test_the_nearest_vertex_wins_when_two_are_in_tolerance():
    from bermake.viewport.picking import PICK_PIXEL_TOLERANCE, pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene, ids = _quad()
    a_px = cam.world_to_screen(np.array([0.0, 0.0, 0.0], dtype=np.float32), w, h)
    # Nudge one pixel toward the far corner: still nearest to a.
    kind, ent = pick_selectable((a_px[0] + 1.0, a_px[1]), (w, h), cam, scene, select_vertices=True)
    assert (kind, ent) == ("vertex", ids["a"])
    assert PICK_PIXEL_TOLERANCE == 8.0


def test_entities_in_box_returns_three_sets():
    from bermake.viewport.picking import entities_in_box

    w, h = 800, 600
    cam = _camera(w, h)
    scene, _ids = _quad()
    result = entities_in_box((0, 0, w, h), "crossing", (w, h), cam, scene)
    assert len(result) == 3
    edges, faces, vertices = result
    assert len(edges) == 4
    assert len(faces) == 1
    assert vertices == set()


def test_entities_in_box_collects_vertices_when_asked():
    from bermake.viewport.picking import entities_in_box

    w, h = 800, 600
    cam = _camera(w, h)
    scene, _ids = _quad()
    _edges, _faces, vertices = entities_in_box(
        (0, 0, w, h), "crossing", (w, h), cam, scene, select_vertices=True
    )
    assert len(vertices) == 4
