"""chain_cuts_face: which face, if any, a drawn chain divides (M7.6a)."""

import numpy as np
from bermake.scene.scene import Scene
from bermake.tools.shape_support import chain_cuts_face


def _quad(s, z=0.0, size=2.0):
    v = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (size, 0, z), (size, size, z), (0, size, z)]
    ]
    return s.add_face_from_loop(v), v


def test_a_chord_between_opposite_corners_cuts_the_face():
    s = Scene()
    fid, v = _quad(s)
    s.add_edge(v[0], v[2])
    assert chain_cuts_face(s, [v[0], v[2]]) == fid


def test_a_chain_with_an_interior_vertex_cuts_the_face():
    s = Scene()
    fid, v = _quad(s)
    mid = s.add_vertex(np.array((1.0, 0.5, 0.0), dtype=np.float32))
    s.add_edge(v[0], mid)
    s.add_edge(mid, v[2])
    assert chain_cuts_face(s, [v[0], mid, v[2]]) == fid


def test_a_closing_loop_is_never_a_split():
    """Snapping back onto the first vertex is the face-creation path."""
    s = Scene()
    _fid, v = _quad(s)
    assert chain_cuts_face(s, [v[0], v[1], v[2], v[0]]) is None


def test_a_chain_running_along_the_boundary_does_not_cut():
    """Two adjacent corners. The segment lies ON the loop rather than across
    the interior, so its midpoint sits on the boundary rather than strictly
    inside it. This is the case the midpoint test exists for."""
    s = Scene()
    _fid, v = _quad(s)
    assert chain_cuts_face(s, [v[0], v[1]]) is None


def test_a_chain_whose_interior_leaves_the_plane_does_not_cut():
    s = Scene()
    _fid, v = _quad(s)
    off = s.add_vertex(np.array((1.0, 1.0, 5.0), dtype=np.float32))
    s.add_edge(v[0], off)
    s.add_edge(off, v[2])
    assert chain_cuts_face(s, [v[0], off, v[2]]) is None


def test_a_chain_whose_interior_leaves_the_polygon_does_not_cut():
    """In plane, but outside the face. A concave parent is where this bites,
    so build one: an L shape whose notch a straight chord would cross."""
    s = Scene()
    pts = [(0, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    v = [s.add_vertex(np.array(p, dtype=np.float32)) for p in pts]
    s.add_face_from_loop(v)
    outside = s.add_vertex(np.array((2.5, 2.5, 0.0), dtype=np.float32))
    s.add_edge(v[1], outside)
    s.add_edge(outside, v[4])
    assert chain_cuts_face(s, [v[1], outside, v[4]]) is None


def test_an_interior_chain_vertex_that_is_on_the_loop_does_not_cut():
    s = Scene()
    _fid, v = _quad(s)
    assert chain_cuts_face(s, [v[0], v[1], v[2]]) is None


def test_a_chain_segment_that_crosses_out_of_a_concave_face_does_not_cut():
    """Both the interior vertex and both segment midpoints can land strictly
    inside a concave polygon while the straight segment between two of them
    still exits and re-enters through a boundary edge. Sampling those points
    alone cannot see that; only a real segment-vs-boundary-edge crossing test
    can. Reproduces the exact case found on the brief's own L-shape: chain
    (1,3) -> (0.6,1.8) -> (3,0) crosses out near (1, 1.5)."""
    s = Scene()
    pts = [(0, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0), (1, 3, 0), (0, 3, 0)]
    v = [s.add_vertex(np.array(p, dtype=np.float32)) for p in pts]
    s.add_face_from_loop(v)
    interior = s.add_vertex(np.array((0.6, 1.8, 0.0), dtype=np.float32))
    s.add_edge(v[4], interior)
    s.add_edge(interior, v[1])
    assert chain_cuts_face(s, [v[4], interior, v[1]]) is None


def test_a_chain_that_exits_and_reenters_exactly_at_loop_vertices_does_not_cut():
    """A chain can leave the polygon and come back without a single proper
    (transversal) edge crossing anywhere, if the exit and the re-entry both
    happen exactly AT a loop vertex rather than through an edge's interior.
    Each such touch alone is a tangent (an orientation of exactly zero
    against at least one of the two edges meeting there), which a
    crossing-only test cannot tell apart from a vertex the chain safely
    grazes without ever leaving -- only a real sub-interval/containment
    test built on the actual intersection parameters can.

    Built as a small zigzag-boundary loop: it goes up through y=2 at x=2
    (P1, a genuine vertex), peaks, back down through y=2 at x=4.5 (P2),
    dips into a narrow valley, back up through y=2 at x=5.5 (P3), then
    climbs the rest of the way up through y=2 at x=13 (P4) on a long final
    rise. P1 and P4 are the chain's own two endpoints (legitimately on the
    loop). P2 and P3 are two OTHER vertices the straight chain passes
    through in between, exactly at the polygon's boundary, framing a narrow
    exterior sliver (x in roughly (4.5, 5.5)) that the chain's own single
    segment midpoint (at x=7.5, deep in the following wide interior stretch)
    never samples. Mirrors an axis-aligned T-slot floor-plan shape reachable
    by ordinary endpoint snapping -- the case that matters most, per the
    finding this pins."""
    s = Scene()
    pts = [
        (0.0, -2.0, 0.0),
        (20.0, -2.0, 0.0),
        (20.0, 1.0, 0.0),  # E, valley
        (13.0, 2.0, 0.0),  # P4, chain end
        (6.0, 3.0, 0.0),  # D, peak
        (5.5, 2.0, 0.0),  # P3
        (5.0, 1.0, 0.0),  # C, narrow valley
        (4.5, 2.0, 0.0),  # P2
        (4.0, 3.0, 0.0),  # B, peak
        (2.0, 2.0, 0.0),  # P1, chain start
        (0.0, 1.0, 0.0),  # A, valley
    ]
    v = [s.add_vertex(np.array(p, dtype=np.float32)) for p in pts]
    s.add_face_from_loop(v)
    p1, p4 = v[9], v[3]
    s.add_edge(p1, p4)
    assert chain_cuts_face(s, [p1, p4]) is None


def test_a_chain_between_two_different_faces_does_not_cut():
    """Two disjoint quads. Each end lies on a boundary loop, but no single
    face's loop contains both, so there is no candidate at all -- distinct
    from the zero-survivors-after-filtering cases above."""
    s = Scene()
    fid1, v1 = _quad(s, z=0.0)
    fid2, v2 = _quad(s, z=10.0)
    assert fid1 != fid2
    assert chain_cuts_face(s, [v1[0], v2[2]]) is None
