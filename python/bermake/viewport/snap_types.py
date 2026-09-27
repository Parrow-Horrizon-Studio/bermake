"""Shared vocabulary for the snap & inference engine.

`snap_engine.py` (the selection policy) and `snap_candidates.py` (the pure
geometry generators) both need `SnapKind` and `Candidate`, and neither may
import the other without creating a cycle. This module exists so they can
share that vocabulary without importing each other: it deliberately depends
on nothing else in `bermake.viewport`.

`AcquiredKind` and `Acquired` live here too, for the same reason: Task 4's
candidate generators in `snap_candidates.py` need `AcquiredKind` without
importing `inference.py` (which is stateful and depends on this module).
`inference.py` imports both from here and re-exports them. This module must
keep depending on nothing but numpy and the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class SnapKind(IntEnum):
    """The snap kinds the engine can report.

    The member order and numeric values are an arbitrary, stable enumeration
    and carry NO precedence meaning. Precedence is D12's table, held
    separately in `snap_engine._PRECEDENCE`, which is deliberately decoupled
    from these values so a new kind can be added here without renumbering
    (and silently reordering) everything below it.
    """

    NONE = 0
    GRID = 1
    AXIS_LOCK = 2
    MIDPOINT = 3
    ENDPOINT = 4
    ON_FACE = 5
    ON_EDGE = 6
    INTERSECTION = 7
    PARALLEL = 8
    PERPENDICULAR = 9
    FROM_POINT = 10
    ON_GUIDE = 11
    GUIDE_POINT = 12


@dataclass(frozen=True, slots=True)
class SnapResult:
    """The chosen snap for one cursor position."""

    kind: SnapKind
    world_position: np.ndarray
    axis: int | None  # 0=X (red), 1=Y (green), 2=Z (blue); AXIS_LOCK and FROM_POINT
    vertex_id: int | None  # only ENDPOINT
    label: str
    edge_id: int | None = None  # MIDPOINT / ON_EDGE / INTERSECTION
    # D3: populated by whichever kind wins, not only ON_FACE -- that is what
    # makes D2 (ON_FACE dropping below the directional inferences) safe, since
    # `resolve_drawing_plane` keys off this field no matter which kind won.
    face_id: int | None = None
    edge_t: float | None = None  # parameter along edge_id (drives split_edge)
    # Unit direction of the infinite inference LINE this result sits on, in
    # world space: AXIS_LOCK, FROM_POINT, PARALLEL, PERPENDICULAR, ON_GUIDE.
    # None for the point-like kinds, which imply no direction at all.
    direction: np.ndarray | None = None


@dataclass
class Candidate:
    """One in-tolerance snap candidate, before precedence selection."""

    kind: SnapKind
    world_position: np.ndarray
    screen_dist: float
    depth: float
    label: str
    vertex_id: int | None = None
    edge_id: int | None = None
    face_id: int | None = None
    axis: int | None = None
    edge_t: float | None = None
    direction: np.ndarray | None = None  # see SnapResult.direction


class AcquiredKind(IntEnum):
    NONE = 0
    VERTEX = 1
    EDGE = 2


@dataclass(frozen=True, slots=True)
class Acquired:
    """A reference the cursor dwelled on, and what it can infer from."""

    kind: AcquiredKind
    position: np.ndarray
    direction: np.ndarray | None
    entity_id: int
