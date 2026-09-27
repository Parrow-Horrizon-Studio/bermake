"""M3c regression: every merged-face polygon (e.g., the hexagon resulting from
a two-quad dissolve) must produce a non-empty triangulation. Guards against the
M3b XY-only earcut latent bug recurring on merged loops."""

from __future__ import annotations

import numpy as np
from bermake.scene.scene import Scene


def test_merged_hexagon_face_has_triangles():
    """Dissolve two coplanar quads sharing an edge -> hexagon -> must fan into
    at least N-2 = 4 triangles, all using merged-loop vertices."""
    scene = Scene()
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
    v4 = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
    v5 = scene.add_vertex(np.array([2, 1, 0], dtype=np.float32))
    scene.add_edge(v0, v1)
    e_shared = scene.add_edge(v1, v2)
    scene.add_edge(v2, v3)
    scene.add_edge(v3, v0)
    scene.add_edge(v1, v4)
    scene.add_edge(v4, v5)
    scene.add_edge(v5, v2)
    scene.add_face_from_loop([v0, v1, v2, v3])
    scene.add_face_from_loop([v1, v4, v5, v2])

    merged = scene.dissolve_edge(e_shared)
    assert merged is not None

    merged_face = scene.face(merged)
    # triangles is an (N, 3) array, so len() is the triangle COUNT. A hexagon
    # fan gives N-2 = 4 triangles.
    assert len(merged_face.triangles) >= 4, (
        f"Merged hexagon should have >= 4 triangles; got {len(merged_face.triangles)}."
    )

    # Every triangle vertex id must be in the merged loop.
    loop_set = set(merged_face.loop_vertex_ids)
    for vid in merged_face.triangles.ravel():
        assert int(vid) in loop_set, (
            f"Triangle vertex {vid} not in merged face's loop {loop_set}"
        )
