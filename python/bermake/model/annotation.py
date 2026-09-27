"""Annotation entities: the rail every non-geometric entity rides.

Dimensions and leader text labels (M7d), plus construction geometry added in
M7.6b: infinite `Guide` lines and `GuidePoint`s, which are inference targets
and never contribute faces or edges (D7).

All of them store CONTEXT-LOCAL coordinates as plain 3-tuples and live in
Definition.annotations, so they ride along when their group/component moves,
and so they inherit placement, persistence, picking, selection, erasing and
undo from one rail rather than four. Pure data, with no Model/Scene/Qt/GL
imports. A dimension's measurement text is NOT stored; it is derived at draw
time from the world distance and the document's units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float, float]


def _pt(value) -> Point:
    x, y, z = value
    return (float(x), float(y), float(z))


@dataclass
class Dimension:
    """A linear dimension between two local points.

    `offset` is a local vector from the p1->p2 midpoint to the dimension-line
    midpoint (it positions the dimension line away from the geometry).
    """

    id: int
    p1: Point
    p2: Point
    offset: Point
    kind: str = "dimension"

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.p1 = _pt(self.p1)
        self.p2 = _pt(self.p2)
        self.offset = _pt(self.offset)


@dataclass
class Label:
    """A text note: an anchor point, where the text sits, and the text."""

    id: int
    anchor: Point
    text_pos: Point
    text: str
    kind: str = "label"

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.anchor = _pt(self.anchor)
        self.text_pos = _pt(self.text_pos)
        self.text = str(self.text)


@dataclass
class Guide:
    """An infinite construction line: a point on it and a unit direction.

    Construction geometry, not model geometry: guides are inference targets and
    never contribute faces or edges. They live in Definition.annotations so they
    inherit context-local placement, persistence, picking, selection and undo
    from the rail Dimension and Label already ride.
    """

    id: int
    origin: Point
    direction: Point
    kind: str = "guide"

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.origin = _pt(self.origin)
        d = _pt(self.direction)
        length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if length < 1e-12:
            raise ValueError("Guide.direction must be non-zero")
        self.direction = (d[0] / length, d[1] / length, d[2] / length)


@dataclass
class GuidePoint:
    """A construction point with no geometry attached."""

    id: int
    position: Point
    kind: str = "guide_point"

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.position = _pt(self.position)
