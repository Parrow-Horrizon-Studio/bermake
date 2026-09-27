"""The drawing plane must come from the face under the cursor, whichever snap wins.

Regression test for the M7.6b section 1 finding: resolve_drawing_plane keyed off
`snap.kind == ON_FACE`, so starting a shape at a wall's own corner resolved a
horizontal plane instead of the wall's.
"""

from __future__ import annotations

import numpy as np
import pytest


def _wall_scene():
    """A single vertical quad in the x = 0 plane, so its normal is +/- X."""
    from bermake.scene import Scene

    scene = Scene()
    vids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (0, 6, 0), (0, 6, 4), (0, 0, 4)]
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        scene.add_edge(vids[a], vids[b])
    scene.add_face_from_loop(vids)
    return scene, vids


def _camera_at_default():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1280.0 / 800.0
    return cam


def _screen_of(cam, world):
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), 1280, 800)
    return (sx, sy)


@pytest.mark.parametrize(
    ("label", "world_point"),
    [
        # Nudged 1e-3 off the exact corner (0, 0, 0): the round trip through
        # world_to_screen -> ray_from_screen is float32 end to end, and at this
        # camera distance the exact corner reprojects to a ray whose x=0 plane
        # hit lands a few ULP outside the quad (observed y ~= -4.8e-7), which
        # makes scene.ray_pick_face miss the mesh entirely. The nudge is far
        # below the endpoint pixel-tolerance radius, so it still snaps to the
        # same corner vertex; it is not testing a different case.
        ("corner", (0.0, 1e-3, 1e-3)),
        ("midpoint", (0.0, 3.0, 0.0)),
    ],
)
def test_drawing_plane_at_a_face_boundary_is_the_face_plane(label, world_point):
    from bermake.tools.shape_support import resolve_drawing_plane
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene, _ = _wall_scene()
    cam = _camera_at_default()
    res = SnapEngine().snap(_screen_of(cam, world_point), (1280, 800), cam, scene)

    # The winning kind is deliberately NOT ON_FACE here; that is the point.
    assert res.kind in (SnapKind.ENDPOINT, SnapKind.MIDPOINT), f"{label}: {res.kind}"
    assert res.face_id is not None, f"{label}: face under the cursor was not reported"

    plane = resolve_drawing_plane(res, scene)
    assert abs(abs(float(plane.normal[0])) - 1.0) < 1e-6, (
        f"{label}: expected the wall's own +/-X normal, got {plane.normal}"
    )
