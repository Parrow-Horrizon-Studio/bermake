"""Read-only queries over the model graph (M7.2, extended M7.3).

Every function is pure: they walk the graph and return plain data, mutating
nothing. select_all_ids backs Edit > Select All; model_bounds backs
View > Zoom Extents; outliner_rows and instance_path back the M7.3 Outliner;
selection_bounds backs Entity Info's dimensions read-out.

All honour the active editing context and visibility, so they agree with what
the user can actually see and click.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bermake.geometry.transforms import apply_mat
from bermake.model.instance import Instance


def select_all_ids(model) -> tuple[set[int], set[int], set[int]]:
    """Every selectable entity in the active editing context.

    Returns (edge_ids, face_ids, instance_ids), scoped to the active context
    only -- matching how every other operation behaves -- and skipping
    instances hidden by tag visibility, so Select All never selects something
    the user cannot see. Does not reach into child definitions: an instance
    itself is selectable, but the geometry nested inside it is not part of
    this context's selection.
    """
    context = model.active_context
    scene = context.mesh

    edges = {e.id for e in scene.edges_iter()}
    faces = {f.id for f in scene.faces_iter()}
    instances = {child.id for child in context.children if model.tags.is_visible(child.tag_id)}
    return edges, faces, instances


def model_bounds(model) -> tuple[np.ndarray, np.ndarray] | None:
    """The world-space axis-aligned bounds of everything currently visible.

    Returns (min_xyz, max_xyz) as float32 arrays, or None when nothing is
    visible -- an empty model, or every tag hidden. Callers treat None as
    "nothing to frame" and do nothing.

    Walks every definition reachable via traverse_visible() (so hidden-tag
    subtrees are excluded but the active editing context is always included),
    and transforms each definition's live vertex positions into world space
    with its accumulated world transform before folding them into the box --
    a definition's own geometry is otherwise only known in local coordinates.
    """
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None

    for definition, world in model.traverse_visible():
        positions = [v.position for v in definition.mesh.vertices_iter()]
        if not positions:
            continue
        local = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        world_points = apply_mat(local, world)
        def_lo = world_points.min(axis=0)
        def_hi = world_points.max(axis=0)
        if lo is None:
            lo, hi = def_lo, def_hi
        else:
            lo = np.minimum(lo, def_lo)
            hi = np.maximum(hi, def_hi)

    if lo is None:
        return None
    return lo, hi


@dataclass(frozen=True, slots=True)
class OutlinerRow:
    """One Outliner line. Plain data -- no Qt, no Instance reference.

    The three visibility fields are deliberately separate. `hidden` is the
    instance's own flag and the only one the eye toggle writes;
    `inherited_hidden` says an ancestor is hidden; `tag_hidden` says its tag
    is switched off. A row dimmed by either of the latter two will NOT come
    back when its eye is clicked, so a widget that rendered all three the
    same would misrepresent what the click does.

    Selection is deliberately absent: it changes many times a second during a
    box-select drag, and the widget can apply highlight from Selection itself
    without rebuilding every row.
    """

    instance_id: int
    depth: int
    label: str
    is_component: bool
    hidden: bool
    inherited_hidden: bool
    tag_hidden: bool
    on_active_path: bool


def outliner_rows(model) -> tuple[OutlinerRow, ...]:
    """Every instance in the model, depth-first, parents before children.

    That order is the tree's render order: a consumer rebuilds the hierarchy
    by tracking `depth` with a running stack. The root context is NOT a row --
    it has no Instance, so giving it one would mean an instance_id matching
    nothing, which every consumer would then special-case.
    """
    active_ids = {inst.id for inst in model.active_path}
    rows: list[OutlinerRow] = []

    def walk(definition, depth: int, ancestor_hidden: bool) -> None:
        for inst in definition.children:
            hidden = bool(inst.hidden)
            rows.append(
                OutlinerRow(
                    instance_id=inst.id,
                    depth=depth,
                    label=inst.name or inst.definition.name,
                    is_component=not inst.definition.is_group,
                    hidden=hidden,
                    inherited_hidden=ancestor_hidden,
                    tag_hidden=not model.tags.is_visible(inst.tag_id),
                    on_active_path=inst.id in active_ids,
                )
            )
            walk(inst.definition, depth + 1, ancestor_hidden or hidden)

    walk(model.root, 0, False)
    return tuple(rows)


def _subtree(definition, world):
    """(definition, world_transform) for `definition` and everything under it."""
    yield definition, world
    for inst in definition.children:
        yield from _subtree(inst.definition, world @ inst.transform)


def selection_bounds(model, selection) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space AABB of the selected instances, or None when none are selected.

    Measures actual transformed vertex positions rather than scaling each
    definition's local AABB by its matrix -- the latter is wrong as soon as an
    instance is rotated, which is the common case for anything placed by hand.
    """
    if not selection.instances:
        return None

    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    base = model.active_world_transform

    for inst in model.active_context.children:
        if inst.id not in selection.instances:
            continue
        for definition, world in _subtree(inst.definition, base @ inst.transform):
            positions = [v.position for v in definition.mesh.vertices_iter()]
            if not positions:
                continue
            local = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            world_points = apply_mat(local, world)
            def_lo = world_points.min(axis=0)
            def_hi = world_points.max(axis=0)
            if lo is None:
                lo, hi = def_lo, def_hi
            else:
                lo = np.minimum(lo, def_lo)
                hi = np.maximum(hi, def_hi)

    if lo is None:
        return None
    return lo, hi


def instance_path(model, instance_id: int) -> tuple[Instance, ...] | None:
    """Root-first ancestor chain ending at `instance_id`, or None if unreachable.

    Returns real Instance objects, not ids, because Model.enter() appends one
    instance at a time and revalidate_active_path() checks each against its
    parent's children list.
    """

    def walk(definition, prefix: tuple):
        for inst in definition.children:
            chain = (*prefix, inst)
            if inst.id == instance_id:
                return chain
            found = walk(inst.definition, chain)
            if found is not None:
                return found
        return None

    return walk(model.root, ())
