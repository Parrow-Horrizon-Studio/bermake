"""A single vertex in the Python scene."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class Vertex:
    """A vertex with a stable integer ID and a Z-up world-space position.

    `position` is an (3,) float32 numpy array. Positions are exact — the snap
    engine produces deterministic snapped points, and `Scene.add_vertex` uses
    exact equality (not an epsilon) for idempotent insertion.

    Hashing and equality are by integer `id`. The default frozen-dataclass
    auto-hash hashes every field, which crashes on the ndarray field; we
    override to use the id only since that's the stable identity.
    """

    id: int
    position: np.ndarray

    def __post_init__(self) -> None:
        # Defensive copy + lock writability so v.position[0] = ... is rejected.
        arr = np.asarray(self.position, dtype=np.float32).reshape(3).copy()
        arr.flags.writeable = False
        object.__setattr__(self, "position", arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vertex):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
