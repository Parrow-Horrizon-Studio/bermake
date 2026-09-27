"""An undirected edge between two vertices."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Edge:
    """An undirected edge with a stable integer ID.

    `v1_id < v2_id` is the canonical ordering — `Scene.add_edge` enforces it
    on insertion so unordered de-duplication is a single dict lookup.
    """

    id: int
    v1_id: int
    v2_id: int
