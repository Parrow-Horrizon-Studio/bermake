"""M3c regression: ray-mesh face picking returns the merged face's id after a
Case 2 seam-merge dissolves the original side faces."""

from __future__ import annotations

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.push_pull_tool import PushPullTool


def _build_unit_box_and_pp_top(scene: Scene) -> None:
    """Build a unit box, then P/P its top by 1.0 (triggers seam merge)."""
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    for a, b in [(v0, v1), (v1, v2), (v2, v3), (v3, v0)]:
        scene.add_edge(a, b)
    f_src = scene.add_face_from_loop([v0, v1, v2, v3])

    tool = PushPullTool()
    tool._scene = scene
    tool._command_stack = CommandStack()
    tool._armed_face_id = f_src
    tool._armed_face_loop = [v0, v1, v2, v3]
    tool._armed_face_normal = scene.face_normal(f_src)
    tool._armed_face_center = scene.face_center(f_src)
    tool._current_depth = 1.0
    tool._commit_extrusion()

    top_id = None
    for f in scene.faces_iter():
        verts = [scene.vertex(vid).position for vid in f.loop_vertex_ids]
        if all(abs(v[2] - 1.0) < 1e-4 for v in verts):
            top_id = f.id
            break
    assert top_id is not None

    tool2 = PushPullTool()
    tool2._scene = scene
    tool2._command_stack = CommandStack()
    tool2._armed_face_id = top_id
    tool2._armed_face_loop = list(scene.face(top_id).loop_vertex_ids)
    tool2._armed_face_normal = scene.face_normal(top_id)
    tool2._armed_face_center = scene.face_center(top_id)
    tool2._current_depth = 1.0
    tool2._commit_extrusion()


def test_picking_returns_merged_face_id_not_stale():
    """After Case 2 P/P, ray-pick a point on a merged side face.
    The picker must return a LIVE face id (not a stale pre-merge id)."""
    scene = Scene()
    _build_unit_box_and_pp_top(scene)

    # Aim a ray at the centre of the front face (y=0 wall), pointing +y.
    origin = np.array([0.5, -5.0, 1.0], dtype=np.float32)
    direction = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    hit = scene.ray_pick_face(origin, direction)
    assert hit is not None
    live_face_ids = {f.id for f in scene.faces_iter()}
    assert hit.face_id in live_face_ids
