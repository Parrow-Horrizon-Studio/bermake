from __future__ import annotations

import numpy as np

from bermake.model.instance import Instance
from bermake.scene.scene import Scene


class Definition:
    """Owns geometry (one Scene/HalfEdgeMesh) plus nested child instances."""

    def __init__(
        self, definition_id: int, name: str, is_group: bool, mesh: Scene | None = None
    ) -> None:
        self.id = int(definition_id)
        self.name = str(name)
        self.is_group = bool(is_group)
        self.mesh = mesh if mesh is not None else Scene()
        self.children: list[Instance] = []
        self.instances: list[Instance] = []
        # The annotation rail: per-context Dimension / Label (M7d) and
        # Guide / GuidePoint construction geometry (M7.6b, D7).
        self.annotations: list = []

    def local_aabb(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Axis-aligned bounds over this definition's live vertices, or None if empty."""
        pts = [v.position for v in self.mesh.vertices_iter()]
        if not pts:
            return None
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        return arr.min(axis=0), arr.max(axis=0)
