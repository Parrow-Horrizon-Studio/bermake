"""DrawingPlane — an immutable orthonormal frame for 2D-on-3D construction.

A plane carries an `origin` (a world point on the plane), a unit `normal`, and
an in-plane orthonormal basis `u`, `v` with `u x v == normal`. Tools generate
shape vertices in 2D plane coords and lift them to world via `to_world`.
All internal state and outputs are float64; callers feeding scene.add_vertex must cast to float32.
"""

from __future__ import annotations

import numpy as np

_DEGENERATE = 1e-9


class DrawingPlane:
    __slots__ = ("normal", "origin", "u", "v")

    def __init__(
        self,
        origin: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.u = np.asarray(u, dtype=np.float64).reshape(3)
        self.v = np.asarray(v, dtype=np.float64).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float64).reshape(3)

    @classmethod
    def horizontal(cls, origin: np.ndarray) -> DrawingPlane:
        """Ground-parallel plane (normal +Z, u=+X, v=+Y) through `origin`."""
        return cls(
            origin,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )

    @classmethod
    def from_normal(cls, origin: np.ndarray, normal: np.ndarray) -> DrawingPlane:
        """Build a stable orthonormal in-plane basis from an arbitrary normal."""
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        ln = float(np.linalg.norm(n))
        if ln < _DEGENERATE:
            raise ValueError("DrawingPlane.from_normal: degenerate (zero) normal")
        n = n / ln
        ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) <= 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(ref, n)
        u_norm = float(np.linalg.norm(u))
        assert u_norm > _DEGENERATE, "BUG: ref/threshold logic produced a degenerate cross product"
        u = u / u_norm
        v = np.cross(n, u)
        return cls(origin, u, v, n)

    @classmethod
    def from_face(cls, scene, face_id: int, origin: np.ndarray) -> DrawingPlane:
        """Plane coplanar with an existing face (normal from scene.face_normal)."""
        n = np.asarray(scene.face_normal(face_id), dtype=np.float64).reshape(3)
        return cls.from_normal(origin, n)

    def to_world(self, uv: np.ndarray) -> np.ndarray:
        """Map plane coords (..., 2) to world coords (..., 3)."""
        uv = np.asarray(uv, dtype=np.float64)
        return self.origin + uv[..., 0:1] * self.u + uv[..., 1:2] * self.v

    def project(self, world: np.ndarray) -> np.ndarray:
        """Drop world coords (..., 3) onto the plane → coords (..., 2)."""
        d = np.asarray(world, dtype=np.float64) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=-1)
