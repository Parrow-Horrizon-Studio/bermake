"""What the current selection is, for the Entity Info panel (M7.3).

Pure: walks the model and returns plain data. Every measurement is a NUMBER,
never a formatted string -- the widget formats with format_length /
format_area, so switching metric<->imperial re-labels the panel with no model
involvement. That is the same rule M7d set for dimension text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bermake.model.model_queries import selection_bounds

_KIND_NOTHING = "Nothing"
_KIND_MIXED = "Mixed"


@dataclass(frozen=True, slots=True)
class EntitySummary:
    """A read-out of the selection. Fields are None when they do not apply.

    `kind == "Mixed"` is ambiguous on its own: `_instance_kind` returns it for
    a same-type selection of instances with heterogeneous definitions (a
    Group and a Component together), while `entity_summary` also returns it
    for a cross-type selection (instances plus faces, say). Those two cases
    need different behaviour -- Tag and Hidden apply to any instance
    selection, homogeneous or not, but not to a cross-type one -- so
    `instances_only` discriminates them without overloading `kind` further.
    """

    kind: str
    count: int
    name: str | None = None
    definition_name: str | None = None
    child_count: int | None = None
    edge_count: int | None = None
    face_count: int | None = None
    size: tuple[float, float, float] | None = None
    length: float | None = None
    area: float | None = None
    tag_id: int | None = None
    material_id: int | None = None
    hidden: bool | None = None
    instances_only: bool = False


def _polygon_area(points: np.ndarray) -> float:
    """Newell's method: half the magnitude of the summed edge cross products.

    Exact for any planar polygon, convex or not, and needs no triangulation.
    """
    total = np.zeros(3, dtype=np.float64)
    count = len(points)
    for i in range(count):
        total += np.cross(points[i], points[(i + 1) % count])
    return float(np.linalg.norm(total) / 2.0)


def _common(values):
    """The single shared value, or None when the set disagrees or is empty."""
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def _instance_kind(instances) -> str:
    is_group = {bool(inst.definition.is_group) for inst in instances}
    if len(is_group) != 1:
        return _KIND_MIXED
    return "Group" if next(iter(is_group)) else "Component"


def _annotation_kind(annotations) -> str:
    kinds = {getattr(a, "kind", None) for a in annotations}
    if len(kinds) != 1:
        return _KIND_MIXED
    return "Label" if next(iter(kinds)) == "label" else "Dimension"


def entity_summary(model, selection) -> EntitySummary:
    """Summarise the current selection for the Entity Info panel."""
    context = model.active_context
    scene = context.mesh

    instances = [inst for inst in context.children if inst.id in selection.instances]
    annotations = [a for a in context.annotations if a.id in selection.annotations]
    edge_ids = sorted(selection.edges)
    face_ids = sorted(selection.faces)

    populated = [bool(instances), bool(edge_ids), bool(face_ids), bool(annotations)]
    count = len(instances) + len(edge_ids) + len(face_ids) + len(annotations)
    if count == 0:
        return EntitySummary(kind=_KIND_NOTHING, count=0)
    if sum(populated) > 1:
        return EntitySummary(kind=_KIND_MIXED, count=count)

    if instances:
        only = instances[0] if len(instances) == 1 else None
        bounds = selection_bounds(model, selection)
        size = None
        if bounds is not None:
            lo, hi = bounds
            size = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2]))
        definition = only.definition if only is not None else None
        return EntitySummary(
            kind=_instance_kind(instances),
            count=len(instances),
            # Single-instance only: a shared name across a multi-selection has
            # no meaning, so report nothing rather than pick one arbitrarily.
            name=only.name if only is not None else None,
            definition_name=definition.name if definition is not None else None,
            child_count=len(definition.children) if definition is not None else None,
            edge_count=(
                sum(1 for _ in definition.mesh.edges_iter()) if definition is not None else None
            ),
            face_count=(
                sum(1 for _ in definition.mesh.faces_iter()) if definition is not None else None
            ),
            size=size,
            tag_id=_common(inst.tag_id for inst in instances),
            hidden=_common(bool(inst.hidden) for inst in instances),
            instances_only=True,
        )

    if face_ids:
        area = None
        if len(face_ids) == 1:
            loop = scene.face_loop(face_ids[0])
            points = np.asarray(
                [scene.vertex(v_id).position for v_id in loop], dtype=np.float64
            ).reshape(-1, 3)
            area = _polygon_area(points)
        return EntitySummary(
            kind="Face",
            count=len(face_ids),
            area=area,
            material_id=_common(scene.face_material(f_id) for f_id in face_ids),
        )

    if edge_ids:
        length = None
        if len(edge_ids) == 1:
            edge = scene.edge(edge_ids[0])
            a = np.asarray(scene.vertex(edge.v1_id).position, dtype=np.float64)
            b = np.asarray(scene.vertex(edge.v2_id).position, dtype=np.float64)
            length = float(np.linalg.norm(b - a))
        return EntitySummary(kind="Edge", count=len(edge_ids), length=length)

    return EntitySummary(kind=_annotation_kind(annotations), count=len(annotations))
