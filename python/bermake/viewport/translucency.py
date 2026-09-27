"""Back-to-front ordering for the translucent pass (M7.5a).

Pure numpy, no GL and no Qt, so it is fully unit-testable headlessly. Used at
two granularities: triangles within one definition, and definitions within the
scene. `order_back_to_front` serves both.

Transforms are COLUMN-VECTOR (`M @ [p; 1]`), matching sweep_stations and
build_mesh_into_scene.
"""

from __future__ import annotations

import numpy as np


def triangle_centroids(positions: np.ndarray) -> np.ndarray:
    """(3T, 3) vertex positions in triangle order -> (T, 3) centroids."""
    pts = np.asarray(positions, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return pts.reshape(-1, 3, 3).mean(axis=1)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 column-vector transform to (N, 3) points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    homogeneous = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (homogeneous @ m.T)[:, :3]


def order_back_to_front(points_world: np.ndarray, camera_pos: np.ndarray) -> np.ndarray:
    """Indices of `points_world` ordered farthest from the camera first.

    Squared distance is enough: the ordering is identical and the sqrt is not.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    delta = pts - np.asarray(camera_pos, dtype=np.float64).reshape(1, 3)
    d2 = np.einsum("ij,ij->i", delta, delta)
    return np.argsort(-d2, kind="stable").astype(np.int64)


def triangle_order_to_vertex_order(tri_order: np.ndarray) -> np.ndarray:
    """(T,) triangle permutation -> (3T,) vertex permutation."""
    order = np.asarray(tri_order, dtype=np.int64)
    if order.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return (order[:, None] * 3 + np.arange(3)).reshape(-1).astype(np.int64)
