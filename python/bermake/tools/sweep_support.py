"""Loop-to-loop lofting, shared by Push/Pull, Offset and Follow Me.

All three tools perform the same operation with different destinations:
Push/Pull displaces the source loop along the face normal, Offset displaces
it along angle bisectors in plane, and Follow Me transforms a profile to
each station along a path. Only the destination positions differ, so the
stitching lives here.

Qt-free on purpose: this is the arithmetic most likely to be wrong, and it
must be testable without a QApplication.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from bermake.commands.scene_commands import (
    AddEdgeCommand,
    AddFaceCommand,
    AddVertexCommand,
    DissolveEdgeCommand,
)


@dataclass(frozen=True, slots=True)
class LoftResult:
    """Commands already executed against the scene, plus the ids they created.

    `dst_vertex_ids` is parallel to the `src_loop_vids` passed in. Returning
    it is what lets callers avoid re-deriving the ids from the executed
    commands themselves.
    """

    commands: list
    dst_vertex_ids: list


def loft_between_loops(
    scene,
    src_loop_vids: Sequence[int],
    dst_positions: Sequence[np.ndarray],
    *,
    cap_start: bool,
    cap_end: bool,
) -> LoftResult:
    """Stitch a ring of quads between `src_loop_vids` and new vertices at
    `dst_positions`, executing each command as it is built.

    Commands are executed here, not by the caller: every caller pushes the
    finished CompositeCommand with push_executed, so the scene must already
    reflect the change.

    `cap_start` closes the source end with the source loop REVERSED, so its
    normal points opposite the sweep. `cap_end` closes the destination end
    in source winding. Two flags rather than one because the callers need
    three different answers: Push/Pull always caps the far end and caps the
    near end only for a standalone source, Follow Me caps both ends of an
    open path, and a closed path caps neither.
    """
    n = len(src_loop_vids)
    assert n == len(dst_positions), "loop and destination lengths must match"

    commands: list = []

    dst_vert_cmds: list[AddVertexCommand] = []
    for pos in dst_positions:
        c = AddVertexCommand(np.asarray(pos, dtype=np.float32))
        c.do(scene)
        dst_vert_cmds.append(c)
        commands.append(c)
    dst_vids = [c.vertex_id for c in dst_vert_cmds]

    for src_vid, dst_vid in zip(src_loop_vids, dst_vids, strict=True):
        c = AddEdgeCommand(src_vid, dst_vid)
        c.do(scene)
        commands.append(c)

    for i in range(n):
        c = AddEdgeCommand(dst_vids[i], dst_vids[(i + 1) % n])
        c.do(scene)
        commands.append(c)

    for i in range(n):
        a = src_loop_vids[i]
        b = src_loop_vids[(i + 1) % n]
        c = AddFaceCommand((a, b, dst_vids[(i + 1) % n], dst_vids[i]))
        c.do(scene)
        commands.append(c)

    if cap_end:
        c = AddFaceCommand(tuple(dst_vids))
        c.do(scene)
        commands.append(c)

    if cap_start:
        c = AddFaceCommand(tuple(reversed(list(src_loop_vids))))
        c.do(scene)
        commands.append(c)

    return LoftResult(commands=commands, dst_vertex_ids=dst_vids)


def seam_merge(scene, candidate_edges: Sequence[int]) -> list:
    """Dissolve candidate edges whose two incident faces are coplanar.

    Single pass over the candidates only. Callers capture them BEFORE
    removing a source face, because removal invalidates the boundary.
    """
    out: list = []
    for e in candidate_edges:
        if not scene.edge_is_live(e):
            continue
        f_a, f_b = scene.edge_faces(e)
        if f_a is None or f_b is None:
            continue
        if scene.faces_are_coplanar(f_a, f_b):
            cmd = DissolveEdgeCommand(e)
            cmd.do(scene)
            out.append(cmd)
    return out


_EPS = 1e-9

# Relative back-off applied when a requested distance is clamped (Finding 1,
# Task 6 review): the analytic/binary-search limit is the distance AT WHICH
# an edge reaches exactly zero length or the loop starts to self-intersect,
# not a safe distance short of it. Pulling back by a fixed fraction of that
# limit -- rather than a fixed absolute amount -- keeps the back-off
# proportional to the polygon's own scale: a millimetre-scale polygon and a
# kilometre-scale one both end up with an offset edge shorter by the same
# *fraction* of the limit, so neither a tiny polygon gets over-corrected nor
# a huge one gets a back-off too small to matter at its own scale. 1e-4 is
# comfortably above float64 rounding noise (~1e-16 relative) yet visually
# imperceptible on any drag.
_COLLAPSE_BACKOFF = 1e-4


def offset_polygon(
    points: np.ndarray, normal: np.ndarray, distance: float
) -> tuple[np.ndarray, float]:
    """Offset a planar polygon along each edge's inward normal by `distance`.

    Positive `distance` offsets inward (may collapse the loop); negative
    offsets outward (never collapses). Each vertex moves along the bisector
    of its two adjacent edge normals, scaled so every offset edge stays
    parallel to its original -- the classic "miter" polygon offset.

    Returns `(offset_points, clamped_distance)`. `clamped_distance` equals
    the requested `distance` unless offsetting further would collapse the
    polygon (an analytic per-edge limit) or make it self-intersect (concave
    loops, caught by a bounded binary search), in which case it is the
    largest distance, short of that limit by a small relative margin, that
    still yields a non-degenerate polygon: every edge has strictly positive
    length and no two vertices coincide. The margin exists because the limit
    itself is defined by exact collapse/self-intersection, and a distance
    exactly at it is unusable (some callers weld coincident vertices).

    The general concave straight-skeleton collapse problem is not solved
    here -- only a two-stage approximation: an exact analytic limit for the
    common case, backstopped by a 12-iteration binary search against a
    simple-polygon check for loops where that limit is optimistic.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    normal_u = _unit(np.asarray(normal, dtype=np.float64))

    edge_dirs = [_unit(pts[(i + 1) % n] - pts[i]) for i in range(n)]
    edge_lengths = [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]

    # Winding depends on how the source face was built, not on an assumed
    # CCW convention, so derive which normal sign makes the loop CCW from
    # the signed area itself rather than trusting the caller's `normal`.
    cross_sum = np.zeros(3)
    for i in range(n):
        cross_sum += np.cross(pts[i], pts[(i + 1) % n])
    n_eff = normal_u if np.dot(cross_sum, normal_u) >= 0 else -normal_u

    edge_normals = [_unit(np.cross(n_eff, d)) for d in edge_dirs]

    # Per-vertex bisector displacement: K = (n_prev + n_next) / (1 + n_prev.n_next).
    # This is 1 / sin(theta / 2) in disguise (theta the interior angle at the
    # vertex) -- it diverges as theta -> 0 (a needle-thin vertex), so the
    # denominator is floored away from zero to keep the scale finite.
    bisectors = []
    for i in range(n):
        n_prev = edge_normals[i - 1]
        n_next = edge_normals[i]
        denom = max(1.0 + float(np.dot(n_prev, n_next)), _EPS)
        bisectors.append((n_prev + n_next) / denom)

    def offset_at(d: float) -> np.ndarray:
        return np.array([pts[i] + d * bisectors[i] for i in range(n)])

    if distance <= 0.0:
        return offset_at(distance), distance

    # Stage 1: analytic per-edge collapse limit. Each offset edge's length
    # is affine in d (both its endpoints move at a fixed rate), so the
    # distance at which it hits zero solves exactly.
    analytic_limit = float("inf")
    for i in range(n):
        rate = float(np.dot(bisectors[(i + 1) % n] - bisectors[i], edge_dirs[i]))
        if rate < -_EPS:
            analytic_limit = min(analytic_limit, -edge_lengths[i] / rate)

    clamped = min(distance, analytic_limit)
    was_clamped = clamped < distance

    # Stage 2: the analytic limit only guards against an edge collapsing
    # onto itself. A concave loop's reflex corners can self-intersect with a
    # *different* edge first, which the per-edge check cannot see, so
    # validate and binary search downward if needed, bounded at 12 steps.
    e1, e2 = _plane_basis(n_eff)
    if not _is_simple_offset(offset_at(clamped), e1, e2):
        was_clamped = True
        lo, hi = 0.0, clamped
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if _is_simple_offset(offset_at(mid), e1, e2):
                lo = mid
            else:
                hi = mid
        clamped = lo

    if was_clamped:
        # `clamped` above is the distance AT WHICH degeneracy first occurs
        # (an exact analytic root, or a binary-search value pinned
        # arbitrarily close to the self-intersection boundary) -- not a
        # safe distance short of it. Pull back by a fraction of `clamped`
        # itself so every offset edge keeps strictly positive length. See
        # `_COLLAPSE_BACKOFF` for why this is relative, not absolute.
        clamped *= 1.0 - _COLLAPSE_BACKOFF

    return offset_at(clamped), clamped


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < _EPS:
        return v
    return v / norm


def _plane_basis(n_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([1.0, 0.0, 0.0]) if abs(n_eff[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(np.cross(helper, n_eff))
    e2 = np.cross(n_eff, e1)
    return e1, e2


def _is_simple_offset(pts: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> bool:
    """No two non-adjacent edges of the closed, projected polygon cross.

    Plain four-orientation sign-of-cross-product test, with no collinearity
    tolerance: two segments are judged to cross only when their endpoints
    fall on strictly opposite sides of each other (`sign() != sign()`).
    Exact collinearity -- both signs landing on 0 -- is therefore never a
    crossing, which is what lets the rectangle's own stage-1 analytic
    collapse (two opposite edges retracing the same line segment) pass this
    check without tripping the binary search.

    This is an internal implementation choice, not an external contract:
    there is nothing outside this module that defines "simple" for an
    offset polygon, so this check exists only to drive the binary search
    towards a distance that this same check accepts. `tests/test_offset_polygon.py`
    verifies the *result* with its own, independently-derived parametric
    segment-intersection algorithm rather than reusing this one, so a bug
    shared between the two would still surface as a test failure.
    """
    n = len(pts)
    proj = np.stack([pts @ e1, pts @ e2], axis=-1)
    for i in range(n):
        a1, a2 = proj[i], proj[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            b1, b2 = proj[j], proj[(j + 1) % n]
            if _segments_cross(a1, a2, b1, b2):
                return False
    return True


def _segments_cross(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
    def side(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.floating:
        # Scalar 2D cross product computed directly: np.cross on length-2
        # vectors is deprecated in NumPy 2.0.
        ab = b - a
        ac = c - a
        return np.sign(ab[0] * ac[1] - ab[1] * ac[0])

    d1, d2 = side(p3, p4, p1), side(p3, p4, p2)
    d3, d4 = side(p1, p2, p3), side(p1, p2, p4)
    return bool(d1 != d2 and d3 != d4)


# ---------------------------------------------------------------------------
# sweep_stations (M7.4 Task 7): per-station transforms for Follow Me
# ---------------------------------------------------------------------------


class SweepRefused(Exception):
    """A corner is too tight for the profile: the mitered plane inverts it."""


def sweep_stations(
    profile_loop: np.ndarray, path_points: np.ndarray, *, closed: bool
) -> list[np.ndarray]:
    """Build one 4x4 world-from-profile transform per path vertex.

    Convention: column-vector, right-multiplied, matching
    `geometry/transforms.py`'s `apply_mat` (`p_world = M @ [p_local; 1]`).
    `p_local` is `profile_loop` in the caller's own coordinates -- there is
    no separate "local space" to move into first; each returned matrix maps
    the profile's own points directly to their position at that station.

    RELATIVE-FRAME CONTRACT. Station `i` is
    `Station_i @ Station_0_unmitered^-1`, where `Station_i` is the absolute
    frame built at `path_points[i]` below and `Station_0_unmitered` is the
    frame at `path_points[0]` oriented by segment 0 alone (never mitered,
    even when the path closes). Every returned matrix therefore maps
    `path_points[0]` -> `path_points[i]`: the frame's origin is the path's
    OWN start, not the profile's centroid. A profile drawn off to one side
    of the path keeps that offset all the way along, which is exactly what
    makes a lathe rather than a tube (spec 1.6). An earlier revision mapped
    the profile centroid onto every path point, which silently recentred
    every offset profile onto the path axis and made a lathe unreachable.

    Two consequences a caller must handle:

    * Open path: station 0 IS `Station_0_unmitered`, so `stations[0]` is
      the identity. The profile as drawn is already the first
      cross-section, and a caller may skip station 0 and loft the source
      loop straight to station 1.
    * Closed path: station 0 is the seam miter and is NOT the identity. It
      must be applied. The ring sequence is
      `M_0(profile) -> M_1 -> ... -> M_{n-1} -> back to M_0(profile)`;
      terminating the closing segment on the raw profile instead lands it
      on a cross-section that does not lie on the seam's miter plane.

    The relative composition is exact for the canonical Follow Me setup --
    a profile whose plane is perpendicular to segment 0 and contains
    `path_points[0]`. A profile plane tilted relative to segment 0, or one
    that `path_points[0]` does not lie in, additionally shifts every
    cross-section a constant distance along its own station normal; that is
    inherent to expressing placements relative to a single starting frame
    and is not corrected here.

    At an interior vertex (or every vertex of a closed path) the plane
    normal is the unit bisector of the incoming and outgoing directions --
    the classic miter plane, tilted equally into both segments (Task 8
    lofts consecutive stations with straight quads; a tilted-but-shared
    plane is what makes that loft meet without a gap at the corner). At an
    open path's two endpoints there is only one adjoining segment, so the
    normal is just that segment's direction.

    A rigid copy of the profile rotated onto the miter plane would not
    actually close the gap: the incoming and outgoing straight extrusions
    only agree at a corner if each profile vertex is slid along ITS OWN
    segment direction until it lands on the shared miter plane (the same
    construction as a mitered picture-frame corner, generalized to an
    arbitrary profile). This module computes that per-vertex slide from
    both sides and averages them, since a general (non-symmetric) profile
    has no exact common answer.

    Raises `SweepRefused` if the corner is too tight for the profile: see
    `_mitered_station` for the exact criterion and why it is geometric
    rather than a tuned constant.

    Known limitation (Task 7 review): each station's frame is built FRESH
    from the source profile's own normal, not transported sequentially from
    the previous station, so there is no torsion-minimising frame across a
    path with several non-coplanar corners in a row -- the swept profile can
    accumulate more visual twist along such a path than a transported frame
    would. This is out of scope for what Follow Me (Task 8) is specified to
    do; a future caller chasing minimal-twist sweeps needs to know it isn't
    here.
    """
    profile = np.asarray(profile_loop, dtype=np.float64)
    path = np.asarray(path_points, dtype=np.float64)
    n_path = len(path)

    centroid = profile.mean(axis=0)
    normal = _cross_sum_normal(profile)
    u_axis, v_axis = _plane_basis(normal)

    if closed:
        seg_dirs = [_unit(path[(i + 1) % n_path] - path[i]) for i in range(n_path)]
        seg_lens = [float(np.linalg.norm(path[(i + 1) % n_path] - path[i])) for i in range(n_path)]
    else:
        seg_dirs = [_unit(path[i + 1] - path[i]) for i in range(n_path - 1)]
        seg_lens = [float(np.linalg.norm(path[i + 1] - path[i])) for i in range(n_path - 1)]

    stations: list[np.ndarray] = []
    for i in range(n_path):
        if closed:
            has_in = has_out = True
            d_in, len_in = seg_dirs[(i - 1) % n_path], seg_lens[(i - 1) % n_path]
            d_out, len_out = seg_dirs[i % n_path], seg_lens[i % n_path]
        else:
            has_in = i > 0
            has_out = i < n_path - 1
            d_in, len_in = (seg_dirs[i - 1], seg_lens[i - 1]) if has_in else (None, None)
            d_out, len_out = (seg_dirs[i], seg_lens[i]) if has_out else (None, None)

        if has_in and has_out:
            stations.append(
                _mitered_station(
                    profile,
                    centroid,
                    normal,
                    u_axis,
                    v_axis,
                    path[i],
                    d_in,
                    len_in,
                    d_out,
                    len_out,
                    i,
                )
            )
        else:
            d = d_out if has_out else d_in
            stations.append(_straight_station(centroid, normal, u_axis, v_axis, path[i], d))

    # Re-express every absolute station relative to the profile's own
    # starting frame (see the RELATIVE-FRAME CONTRACT above). `base` is
    # deliberately the UNMITERED frame at path[0]: for an open path it is
    # station 0 itself, so stations[0] comes back as the identity; for a
    # closed path station 0 is the seam miter and stays distinct from it.
    base = _straight_station(centroid, normal, u_axis, v_axis, path[0], seg_dirs[0])
    base_inv = _invert_rigid(base)
    return [s @ base_inv for s in stations]


def _cross_sum_normal(pts: np.ndarray) -> np.ndarray:
    """Unit plane normal of a planar loop, via the shoelace cross-sum.

    Same technique as `offset_polygon`'s winding derivation: translation
    invariant, and its sign follows whatever winding `pts` was authored
    with rather than an assumed convention.
    """
    n = len(pts)
    cs = np.zeros(3)
    for i in range(n):
        cs += np.cross(pts[i], pts[(i + 1) % n])
    return _unit(cs)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal (shortest-arc) rotation matrix mapping unit vector `a` to `b`.

    Standard Rodrigues construction. `a` antiparallel to `b` (dot == -1) is
    the one case a rotation axis can't be read off `cross(a, b)` (it is a
    zero vector, not just small); a 180-degree rotation about ANY axis
    perpendicular to `a` maps a to b there, so one is built from `a` alone.
    """
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < _EPS:
        if c > 0.0:
            return np.eye(3)
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, helper))
        k = _skew(axis)
        return np.eye(3) + 2.0 * (k @ k)
    k = _skew(v)
    return np.eye(3) + k + k @ k * ((1.0 - c) / (s * s))


def _invert_rigid(m: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 whose linear part is a rotation.

    Only ever called on a `_straight_station` result, whose linear part is
    an orthonormal-triad-to-orthonormal-triad map, i.e. a rotation. Its
    transpose IS its inverse, so this needs no general solve and carries no
    conditioning worry -- unlike a mitered station's linear part, which
    slides profile vertices along the segments and is not orthonormal.
    """
    r = m[:3, :3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -(r.T @ m[:3, 3])
    return inv


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _compose_frame(
    centroid: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    position: np.ndarray,
    u_image: np.ndarray,
    v_image: np.ndarray,
    n_image: np.ndarray,
) -> np.ndarray:
    """Assemble a 4x4 sending `u_axis -> u_image`, `v_axis -> v_image`,
    `normal -> n_image`, and `centroid -> position`.

    `u_axis`, `v_axis`, `normal` are an orthonormal triad, so this rank-3
    reconstruction (`M_linear = outer(u_image,u_axis) + outer(v_image,v_axis)
    + outer(n_image,normal)`) is exact: it is the unique linear map with
    the three requested images, and every profile point decomposes cleanly
    into that triad since `centroid`-relative offsets are, by construction
    (`normal` comes from the profile's own cross-sum), perpendicular to
    `normal`.
    """
    linear = np.outer(u_image, u_axis) + np.outer(v_image, v_axis) + np.outer(n_image, normal)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = linear
    m[:3, 3] = position - linear @ centroid
    return m


def _straight_station(
    centroid: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    position: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Station at an open path's endpoint: a rigid copy, cross-section
    turned to face straight down the one adjoining segment.
    """
    r = _rotation_between(normal, direction)
    return _compose_frame(
        centroid, normal, u_axis, v_axis, position, r @ u_axis, r @ v_axis, direction
    )


def _mitered_station(
    profile: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    position: np.ndarray,
    d_in: np.ndarray,
    len_in: float,
    d_out: np.ndarray,
    len_out: float,
    vertex_index: int,
) -> np.ndarray:
    """Station at an interior path vertex (or any vertex of a closed path).

    The construction, per vertex offset `o` (relative to `centroid`):

    1. Rotate `o` into each segment's own straight-extrusion frame:
       `o_in = R(normal -> d_in) @ o`, and likewise `o_out` for `d_out`.
       This is what a plain (unmitered) straight sweep would place at this
       station from that side alone -- `o_in`/`o_out` are exactly
       perpendicular to `d_in`/`d_out`.
    2. Slide each along its own segment direction until it lands on the
       shared miter plane (through `position`, normal = the unit bisector
       `t` of `d_in` and `d_out`): `o_in - ((o_in . t) / (d_in . t)) * d_in`,
       and likewise for `o_out`. This is a line/plane intersection, solved
       once per vertex since each vertex's line is fixed but its offset
       from the plane differs.
    3. Average the two. They agree exactly only for a profile symmetric
       about the corner's bisector plane; averaging is the natural
       (rotation-and-reflection-symmetric, order-independent) choice for
       the general case, and both feed into a linear combination that
       reduces to a plain rotation whenever `d_in == d_out` (a "corner"
       that isn't one) or the profile is centred and symmetric.

    Refusal criterion: step 2's slide distance for a profile vertex,
    measured from `position` back along `d_in` (or forward along `d_out`),
    is unbounded as the turn approaches 180 degrees -- there is no tuned
    threshold for "too far". What IS a hard geometric fact is that a slide
    longer than the segment itself reaches past the OTHER end of that
    segment, i.e. past a neighbouring station that already exists; the
    corner has then consumed geometry that belongs to the next joint over,
    which is exactly the profile folding onto itself. So the refusal test
    is: does any profile vertex's slide distance, on either side, exceed
    that side's own segment length. `len_in`/`len_out` come from the path
    itself, not a constant.
    """
    t = d_in + d_out
    t_norm = float(np.linalg.norm(t))
    if t_norm < _EPS:
        raise SweepRefused(
            f"corner at path vertex {vertex_index} reverses back on itself exactly: "
            "no miter plane bisects a 180-degree turn"
        )
    t = t / t_norm

    r_in = _rotation_between(normal, d_in)
    r_out = _rotation_between(normal, d_out)
    offsets = profile - centroid
    o_in = offsets @ r_in.T
    o_out = offsets @ r_out.T

    dot_in_t = float(np.dot(d_in, t))
    dot_out_t = float(np.dot(d_out, t))
    # dot_in_t == dot_out_t == cos(half the turn angle) always (t bisects
    # d_in/d_out by construction), positive for any turn short of a full
    # 180-degree reversal (already excluded above), so this never divides
    # by (near) zero without the t_norm guard already having fired.
    slide_in = (o_in @ t) / dot_in_t
    slide_out = (o_out @ t) / dot_out_t

    reach_in = float(np.max(np.abs(slide_in)))
    reach_out = float(np.max(np.abs(slide_out)))
    if reach_in > len_in or reach_out > len_out:
        raise SweepRefused(
            f"corner at path vertex {vertex_index} is too tight for this profile: "
            f"mitering would slide a profile vertex {max(reach_in, reach_out):.6g} "
            f"units along a segment that is only {min(len_in, len_out):.6g} long"
        )

    # Steps 1-3 are linear in the offset, so the same slide-and-average
    # construction applied directly to `u_axis`/`v_axis` gives exactly the
    # images those two basis vectors need for `_compose_frame` -- no need
    # to redo it per profile vertex (that already happened above, for the
    # refusal check).
    def slide_and_average(axis: np.ndarray) -> np.ndarray:
        a_in = r_in @ axis
        a_out = r_out @ axis
        s_in = float(np.dot(a_in, t)) / dot_in_t
        s_out = float(np.dot(a_out, t)) / dot_out_t
        return 0.5 * ((a_in - s_in * d_in) + (a_out - s_out * d_out))

    u_image = slide_and_average(u_axis)
    v_image = slide_and_average(v_axis)
    return _compose_frame(centroid, normal, u_axis, v_axis, position, u_image, v_image, t)
