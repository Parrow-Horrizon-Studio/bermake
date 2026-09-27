"""Pure snap candidate generators.

Each function turns geometry plus a cursor ray into zero or more `Candidate`s.
No engine state, no acquisition, no locking: those live in `snap_engine.py` and
`inference.py` respectively. Split out of `snap_engine.py` in M7.6b so the file
that gained six new generators did not also own the selection policy.

`Candidate` and `SnapKind` come from `snap_types.py`, not `snap_engine.py`: this
module and `snap_engine` share that vocabulary without importing each other.
"""

from __future__ import annotations

import math

import numpy as np

from bermake.geometry.ray import (
    closest_point_on_segment_to_ray as _closest_point_on_segment_to_ray,
)
from bermake.geometry.ray import (
    closest_points_two_lines as _closest_points_two_lines,
)
from bermake.viewport.snap_types import AcquiredKind, Candidate, SnapKind

_AXIS_NAMES = {0: "Red", 1: "Green", 2: "Blue"}

_INTERSECTION_EPS = 1e-3  # world-space closest-approach below this = a real crossing
_AXIS_DIRS = {
    0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
    1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    2: np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


def endpoint_candidates(px, py, width, height, camera, scene, to_world, pixel_tolerance):
    out: list[Candidate] = []
    for v in scene.vertices_iter():
        world_pos = to_world(v.position)
        proj = camera.world_to_screen(world_pos, width, height)
        if proj is None:
            continue
        sx, sy, depth = proj
        d = math.hypot(sx - px, sy - py)
        if d <= pixel_tolerance:
            out.append(
                Candidate(
                    kind=SnapKind.ENDPOINT,
                    world_position=np.asarray(world_pos, dtype=np.float32),
                    screen_dist=d,
                    depth=depth,
                    label="Endpoint",
                    vertex_id=v.id,
                )
            )
    return out


def edge_point_candidates(
    px,
    py,
    width,
    height,
    camera,
    scene,
    ray_origin,
    ray_dir,
    to_world,
    ray_origin_local=None,
    ray_dir_local=None,
    *,
    pixel_tolerance,
):
    """Midpoint AND On-Edge candidates for each live edge.

    The Midpoint block projects the geometric midpoint of each edge.
    The On-Edge block uses `ray_origin`/`ray_dir` (the cursor ray) to find
    the closest point on the 3D segment to the ray, producing an ON_EDGE
    candidate whenever that projected point is within pixel tolerance.

    to_world: callable that maps a local position to world space.
    ray_origin_local / ray_dir_local: the camera ray in local space (for
    finding the closest point on local-space segments).
    """
    if ray_origin_local is None:
        ray_origin_local = ray_origin
    if ray_dir_local is None:
        ray_dir_local = ray_dir

    out: list[Candidate] = []
    for e in scene.edges_iter():
        p1 = scene.vertex(e.v1_id).position
        p2 = scene.vertex(e.v2_id).position
        # Work in local space for segment math; project world positions to screen.
        mid_local = (p1 + p2) * 0.5
        mid_world = to_world(mid_local)
        proj = camera.world_to_screen(mid_world, width, height)
        if proj is not None:
            sx, sy, depth = proj
            d = math.hypot(sx - px, sy - py)
            if d <= pixel_tolerance:
                out.append(
                    Candidate(
                        kind=SnapKind.MIDPOINT,
                        world_position=np.asarray(mid_world, dtype=np.float32),
                        screen_dist=d,
                        depth=depth,
                        label="Midpoint",
                        edge_id=e.id,
                        edge_t=0.5,
                    )
                )
        # On-Edge: closest point on the local 3D segment to the local cursor ray.
        on_pt_local, t = _closest_point_on_segment_to_ray(ray_origin_local, ray_dir_local, p1, p2)
        on_pt_world = to_world(on_pt_local)
        proj_e = camera.world_to_screen(on_pt_world, width, height)
        if proj_e is not None:
            sx, sy, depth = proj_e
            d = math.hypot(sx - px, sy - py)
            if d <= pixel_tolerance:
                out.append(
                    Candidate(
                        kind=SnapKind.ON_EDGE,
                        world_position=np.asarray(on_pt_world, dtype=np.float32),
                        screen_dist=d,
                        depth=depth,
                        label="On Edge",
                        edge_id=e.id,
                        edge_t=t,
                    )
                )
    return out


_COLLINEAR_DEG = 1.0
"""#31: `_closest_points_two_lines` solves via denom = |ray_dir x axis_dir|^2,
proportional to sin^2 of the angle between the two directions. That collapses
toward zero exactly when the cursor ray runs parallel (collinear) to the axis
line, which is where the closest-point solve stops being trustworthy -- a
sub-degree wobble in the cursor ray then swings the reported axis point by
whole world units. Guarded below by comparing the angle's cosine (its
magnitude, since anti-parallel is just as collinear as parallel) against a
tolerance measured from true parallel."""
_COLLINEAR_COS = math.cos(math.radians(_COLLINEAR_DEG))


def axis_candidates(px, py, width, height, camera, anchor, ray_origin, ray_dir, pixel_tolerance):
    out: list[Candidate] = []
    ray_dir_arr = np.asarray(ray_dir, dtype=np.float64)
    ray_norm = float(np.linalg.norm(ray_dir_arr))
    ray_unit = ray_dir_arr / ray_norm if ray_norm > 1e-12 else ray_dir_arr
    for axis_idx, axis_dir in _AXIS_DIRS.items():
        if abs(float(np.dot(ray_unit, axis_dir))) >= _COLLINEAR_COS:
            # Cursor ray within _COLLINEAR_DEG of parallel to this axis: the
            # two-line solve below is ill-conditioned here, so skip the axis
            # entirely rather than emit a wildly displaced candidate.
            continue
        # Point on the infinite axis line (through anchor) nearest the cursor ray.
        _, _, _c_ray, c_axis = _closest_points_two_lines(ray_origin, ray_dir, anchor, axis_dir)
        proj = camera.world_to_screen(c_axis, width, height)
        if proj is None:
            continue
        sx, sy, depth = proj
        d = math.hypot(sx - px, sy - py)
        if d <= pixel_tolerance:
            out.append(
                Candidate(
                    kind=SnapKind.AXIS_LOCK,
                    world_position=c_axis,
                    screen_dist=d,
                    depth=depth,
                    label=f"on {_AXIS_NAMES[axis_idx]} Axis",
                    axis=axis_idx,
                    direction=np.asarray(axis_dir, dtype=np.float64),
                )
            )
    return out


def intersection_candidates(px, py, width, height, camera, scene, anchor, pixel_tolerance):
    out: list[Candidate] = []
    for _axis_idx, axis_dir in _AXIS_DIRS.items():
        for e in scene.edges_iter():
            a = scene.vertex(e.v1_id).position
            b = scene.vertex(e.v2_id).position
            seg_dir = b - a
            _, t, c_axis, c_edge = _closest_points_two_lines(anchor, axis_dir, a, seg_dir)
            if t < 0.0 or t > 1.0:
                continue  # crossing lies outside the edge segment
            if float(np.linalg.norm(c_axis - c_edge)) > _INTERSECTION_EPS:
                continue  # skew -- no genuine 3D crossing
            proj = camera.world_to_screen(c_edge, width, height)
            if proj is None:
                continue
            sx, sy, depth = proj
            d = math.hypot(sx - px, sy - py)
            if d <= pixel_tolerance:
                out.append(
                    Candidate(
                        kind=SnapKind.INTERSECTION,
                        world_position=c_edge,
                        screen_dist=d,
                        depth=depth,
                        label="Intersection",
                        edge_id=e.id,
                        edge_t=float(t),
                    )
                )
    return out


_PLANE_PARALLEL_EPS = 1e-4
"""Minimum length of cross(unit(n), unit(d)) for Perpendicular to be defined.

Below this, the edge runs along (or nearly along) the plane normal, cross
degenerates toward the zero vector, and normalising it would produce NaN
rather than a direction. Measured on the cross product itself, not on the
dot product of the raw inputs, so it stays correct when `d` is not unit
length: dot(n, d) only tracks the angle between them when `d` is a unit
vector, but cross's length does not depend on that assumption once both
inputs are normalised first.
"""


def _line_candidate(
    px, py, width, height, camera, origin, direction, kind, label, pixel_tolerance, axis=None
):
    """One candidate on the infinite line through `origin` along `direction`.

    The same shape axis_candidates already uses: the point on the line nearest
    the cursor ray, projected and tested against the pixel tolerance.

    The candidate carries the line's own unit direction. That is what lets
    Shift-lock pin a Parallel / Perpendicular / On-Guide inference without
    re-deriving a direction from the snapped point, which is impossible for
    a guide (the line passes nowhere near the gesture anchor) and lossy for
    the rest.
    """
    ray_origin, ray_dir = camera.ray_from_screen(px, py, width, height)
    _, _, _c_ray, c_line = _closest_points_two_lines(ray_origin, ray_dir, origin, direction)
    proj = camera.world_to_screen(c_line, width, height)
    if proj is None:
        return None
    sx, sy, depth = proj
    d = math.hypot(sx - px, sy - py)
    if d > pixel_tolerance:
        return None
    dir_arr = np.asarray(direction, dtype=np.float64).reshape(3)
    dir_len = float(np.linalg.norm(dir_arr))
    return Candidate(
        kind=kind,
        world_position=np.asarray(c_line, dtype=np.float32),
        screen_dist=d,
        depth=depth,
        label=label,
        axis=axis,
        direction=dir_arr / dir_len if dir_len > 1e-12 else None,
    )


def directional_candidates(
    px, py, width, height, camera, anchor, acquired, plane_normal, pixel_tolerance
):
    """PARALLEL and PERPENDICULAR against an acquired EDGE, from the anchor.

    Parallel is well defined in 3D: the line through the anchor along the edge.
    Perpendicular is not, so it resolves inside the drawing plane. When the edge
    runs along the plane normal the cross product degenerates and NO candidate is
    offered, rather than a guessed direction.
    """
    out = []
    if acquired is None or acquired.kind != AcquiredKind.EDGE or acquired.direction is None:
        return out
    if anchor is None:
        return out

    origin = np.asarray(anchor, dtype=np.float64).reshape(3)
    d = np.asarray(acquired.direction, dtype=np.float64).reshape(3)

    cand = _line_candidate(
        px,
        py,
        width,
        height,
        camera,
        origin,
        d,
        SnapKind.PARALLEL,
        "Parallel to Edge",
        pixel_tolerance,
    )
    if cand is not None:
        out.append(cand)

    if plane_normal is not None:
        n = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        n_len = float(np.linalg.norm(n))
        d_len = float(np.linalg.norm(d))
        if n_len > 0.0 and d_len > 0.0:
            n_hat = n / n_len
            d_hat = d / d_len
            perp = np.cross(n_hat, d_hat)
            perp_len = float(np.linalg.norm(perp))
            # Guard on cross's own length, the quantity that actually degenerates,
            # not on dot(n, d): dot only tracks the angle between them when d is
            # unit length, and d is not guaranteed to be here.
            if perp_len > _PLANE_PARALLEL_EPS:
                perp = perp / perp_len
                cand = _line_candidate(
                    px,
                    py,
                    width,
                    height,
                    camera,
                    origin,
                    perp,
                    SnapKind.PERPENDICULAR,
                    "Perpendicular to Edge",
                    pixel_tolerance,
                )
                if cand is not None:
                    out.append(cand)
    return out


def from_point_candidates(px, py, width, height, camera, acquired, pixel_tolerance):
    """The three axis lines radiating from an acquired VERTEX.

    Anchored at the acquired point, not at the gesture anchor: the whole purpose
    is aligning to geometry the line never touches.
    """
    out = []
    if acquired is None or acquired.kind != AcquiredKind.VERTEX:
        return out
    origin = np.asarray(acquired.position, dtype=np.float64).reshape(3)
    for axis_idx, axis_dir in _AXIS_DIRS.items():
        cand = _line_candidate(
            px,
            py,
            width,
            height,
            camera,
            origin,
            np.asarray(axis_dir, dtype=np.float64),
            SnapKind.FROM_POINT,
            f"From Point on {_AXIS_NAMES[axis_idx]} Axis",
            pixel_tolerance,
            axis=axis_idx,
        )
        if cand is not None:
            out.append(cand)
    return out


def guide_candidates(px, py, width, height, camera, guides, guide_points, pixel_tolerance):
    """ON_GUIDE for each infinite guide line, GUIDE_POINT for each guide point.

    `guides` is a sequence of world-space (origin, direction) pairs; a guide
    is an infinite line, exactly what `_line_candidate` already handles.
    `guide_points` is a sequence of world-space positions. Both are supplied
    already in world space by the caller: this module never touches a
    world_transform itself.
    """
    out: list[Candidate] = []
    for origin, direction in guides:
        cand = _line_candidate(
            px,
            py,
            width,
            height,
            camera,
            np.asarray(origin, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            SnapKind.ON_GUIDE,
            "On Guide",
            pixel_tolerance,
        )
        if cand is not None:
            out.append(cand)
    for position in guide_points:
        world_pos = np.asarray(position, dtype=np.float32)
        proj = camera.world_to_screen(world_pos, width, height)
        if proj is None:
            continue
        sx, sy, depth = proj
        d = math.hypot(sx - px, sy - py)
        if d <= pixel_tolerance:
            out.append(
                Candidate(
                    kind=SnapKind.GUIDE_POINT,
                    world_position=world_pos,
                    screen_dist=d,
                    depth=depth,
                    label="Guide Point",
                )
            )
    return out


def guide_intersection_candidates(
    px, py, width, height, camera, scene, guides, to_world, pixel_tolerance
):
    """INTERSECTION where a guide crosses a scene edge, or another guide.

    Reuses `closest_points_two_lines` and the same skew-rejection
    `intersection_candidates` already applies. A guide is infinite, so only
    the scene EDGE's parameter is bound to [0, 1]; the guide side of a
    guide-versus-edge crossing, and both sides of a guide-versus-guide
    crossing, carry no such bound.

    `guides` is a sequence of world-space (origin, direction) pairs, already
    converted by the caller. Scene edges are local, so `to_world` (the same
    local-to-world callable `snap_engine.snap` builds from the active world
    transform) is applied to each edge's endpoints before the line math, so
    both lines being intersected live in the same space.
    """
    guides = list(guides)
    out: list[Candidate] = []
    for origin, direction in guides:
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        for e in scene.edges_iter():
            p1 = np.asarray(to_world(scene.vertex(e.v1_id).position), dtype=np.float64)
            p2 = np.asarray(to_world(scene.vertex(e.v2_id).position), dtype=np.float64)
            seg_dir = p2 - p1
            _, t, c_guide, c_edge = _closest_points_two_lines(o, d, p1, seg_dir)
            if t < 0.0 or t > 1.0:
                continue  # the edge is finite; the guide is not
            if float(np.linalg.norm(c_guide - c_edge)) > _INTERSECTION_EPS:
                continue  # skew -- no genuine 3D crossing
            proj = camera.world_to_screen(c_edge, width, height)
            if proj is None:
                continue
            sx, sy, depth = proj
            dist = math.hypot(sx - px, sy - py)
            if dist <= pixel_tolerance:
                out.append(
                    Candidate(
                        kind=SnapKind.INTERSECTION,
                        world_position=np.asarray(c_edge, dtype=np.float32),
                        screen_dist=dist,
                        depth=depth,
                        label="Intersection",
                        edge_id=e.id,
                        edge_t=float(t),
                    )
                )
    for i in range(len(guides)):
        o1 = np.asarray(guides[i][0], dtype=np.float64)
        d1 = np.asarray(guides[i][1], dtype=np.float64)
        for j in range(i + 1, len(guides)):
            o2 = np.asarray(guides[j][0], dtype=np.float64)
            d2 = np.asarray(guides[j][1], dtype=np.float64)
            _, _, c1, c2 = _closest_points_two_lines(o1, d1, o2, d2)
            if float(np.linalg.norm(c1 - c2)) > _INTERSECTION_EPS:
                continue  # skew -- neither guide bounds the other
            proj = camera.world_to_screen(c1, width, height)
            if proj is None:
                continue
            sx, sy, depth = proj
            dist = math.hypot(sx - px, sy - py)
            if dist <= pixel_tolerance:
                out.append(
                    Candidate(
                        kind=SnapKind.INTERSECTION,
                        world_position=np.asarray(c1, dtype=np.float32),
                        screen_dist=dist,
                        depth=depth,
                        label="Intersection",
                    )
                )
    return out


def face_candidate(ray_origin, ray_dir, scene):
    """On-Face via the C++ ray-mesh pick. Screen distance is 0 (under cursor)."""
    hit = scene.ray_pick_face(ray_origin, ray_dir)
    if hit is None:
        return None
    point = np.array([hit.point[0], hit.point[1], hit.point[2]], dtype=np.float32)
    return Candidate(
        kind=SnapKind.ON_FACE,
        world_position=point,
        screen_dist=0.0,
        depth=float(hit.t),
        label="On Face",
        face_id=int(hit.face_id),
    )
