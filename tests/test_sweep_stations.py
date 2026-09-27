"""Path stations, miters, and the refusal test (M7.4 Task 7)."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.tools.sweep_support import SweepRefused, sweep_stations

UNIT_SQUARE = np.array(
    [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
    dtype=np.float64,
)


def test_a_straight_path_gives_one_station_per_point():
    path = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    stations = sweep_stations(UNIT_SQUARE, path, closed=False)
    assert len(stations) == 3
    assert stations[0].shape == (4, 4)


def test_a_closed_path_wraps_without_duplicating_a_station():
    square_path = np.array(
        [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=np.float64
    )
    stations = sweep_stations(UNIT_SQUARE, square_path, closed=True)
    assert len(stations) == 4


def test_a_right_angle_miters_rather_than_gapping():
    # The cross-section at the corner must bisect, so its plane normal is
    # 45 degrees from both segments rather than aligned with either.
    #
    # Asserted on the transformed POSITIONS of an asymmetric profile held
    # off the path axis (_OFFSET_PROFILE), not on a dot product of a matrix
    # column against a symmetric unit square: a dot product cannot see
    # displacement at all, and a symmetric profile centred on the path
    # hides a wrong orientation. Both were what let the recentring defect
    # through.
    path = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 0]], dtype=np.float64)
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    ring = _ring(stations[1], _OFFSET_PROFILE)

    seg_in = np.array([1.0, 0.0, 0.0])
    seg_out = np.array([0.0, 1.0, 0.0])
    corner_normal = _ring_normal(ring)
    assert abs(np.dot(corner_normal, seg_in)) == pytest.approx(
        abs(np.dot(corner_normal, seg_out)), abs=1e-9
    )
    # The bisecting plane through the corner (5, 0, 0) with normal
    # (1, 1, 0)/sqrt(2) is x + y == 5. Every ring vertex sits on it, which
    # is what makes the incoming and outgoing quads meet without a gap.
    assert (ring[:, 0] + ring[:, 1]) == pytest.approx(5.0, abs=1e-9)
    # ... and it sits there at the profile's OWN distance from the path,
    # rather than recentred onto it: the source profile spans y 3..4 and so
    # does the mitered ring. A station that put the profile centroid on
    # (5, 0, 0) would place this ring at y in [-1, 1].
    assert ring[:, 1].min() == pytest.approx(3.0, abs=1e-9)
    assert ring[:, 1].max() == pytest.approx(4.0, abs=1e-9)
    assert ring == pytest.approx(
        np.array([[2.0, 3.0, 0.0], [1.0, 4.0, 0.0], [1.0, 4.0, 1.0], [2.0, 3.0, 2.0]]),
        abs=1e-9,
    )


def test_a_corner_too_tight_for_the_profile_is_refused():
    # A hairpin: the profile is 1 wide, the turn doubles back, so the
    # mitered plane inverts the profile.
    big_profile = UNIT_SQUARE * 10.0
    path = np.array([[0, 0, 0], [1, 0, 0], [0.02, 0.01, 0]], dtype=np.float64)
    with pytest.raises(SweepRefused):
        sweep_stations(big_profile, path, closed=False)


def test_refusal_names_the_offending_corner():
    big_profile = UNIT_SQUARE * 10.0
    path = np.array([[0, 0, 0], [1, 0, 0], [0.02, 0.01, 0]], dtype=np.float64)
    with pytest.raises(SweepRefused) as exc:
        sweep_stations(big_profile, path, closed=False)
    assert "corner" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Additional discrimination tests (not in the task brief).
#
# The brief's own tests use a symmetric square profile and, mostly,
# symmetric turns -- exactly the trap the task calls out: a wrong rotation
# is easy to hide behind that symmetry. These use an off-centre, asymmetric
# profile and check actual transformed positions/orientations, not just
# station counts.
# ---------------------------------------------------------------------------

# An asymmetric, off-centre right triangle: no symmetry to hide a wrong
# rotation or a "translate only" implementation behind.
_TRIANGLE = np.array(
    [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float64
)

# The lathe fixture: an asymmetric trapezoid (no mirror symmetry in either
# in-plane axis) held 3 units off the path in +Y, in the x = 0 plane. That
# plane is perpendicular to a +X first segment and contains the path start,
# which is the canonical Follow Me setup -- and holding the profile OFF the
# path axis is precisely what makes a lathe rather than a tube (spec 1.6).
# A symmetric profile centred on the path, as the tests above use, cannot
# see either a wrong orientation or a recentred position.
_OFFSET_PROFILE = np.array(
    [[0.0, 3.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 1.0], [0.0, 3.0, 2.0]],
    dtype=np.float64,
)


def _apply(station: np.ndarray, point: np.ndarray) -> np.ndarray:
    h = np.array([point[0], point[1], point[2], 1.0])
    return (station @ h)[:3]


def _ring(station: np.ndarray, profile: np.ndarray) -> np.ndarray:
    return np.array([_apply(station, p) for p in profile])


def _ring_normal(ring: np.ndarray) -> np.ndarray:
    """Unit plane normal of a planar loop, by shoelace cross-sum.

    Derived from the ring's own transformed POSITIONS rather than read off
    a column of the station matrix: under the relative-frame contract the
    linear part is `L_i @ L_0^-1`, so no column of it is the station's
    plane normal any more. Reading the geometry back is also the stronger
    check -- it fails if the profile lands on the right plane in the wrong
    place, which a matrix column cannot see.
    """
    n = len(ring)
    cs = np.zeros(3)
    for i in range(n):
        cs += np.cross(ring[i], ring[(i + 1) % n])
    return cs / np.linalg.norm(cs)


def test_an_open_paths_first_station_is_the_identity():
    # The relative-frame contract, pinned rather than left incidental:
    # station 0 of an open path IS the frame the whole sequence is
    # expressed relative to, so it must be exactly the identity. Follow Me
    # relies on this when it skips station 0 and lofts the source loop as
    # drawn straight to station 1.
    path = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float64)
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    assert stations[0] == pytest.approx(np.eye(4), abs=1e-12)


def test_an_offset_profile_keeps_its_offset_along_a_straight_path():
    # The lathe case, at its simplest. A profile held 3 units off the path
    # must arrive at the far station STILL 3 units off it -- the sweep
    # translates it, it does not recentre it. An implementation that maps
    # the profile centroid onto each path point (as sweep_stations once
    # did) puts this ring at y in [-1, 1] around the path axis instead,
    # and the whole first tube segment is then a gross lateral skew.
    path = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    ring = _ring(stations[1], _OFFSET_PROFILE)
    # Straight run, so the station is a pure translation by the segment.
    assert ring == pytest.approx(_OFFSET_PROFILE + np.array([5.0, 0.0, 0.0]), abs=1e-9)


def test_a_turn_actually_rotates_an_off_axis_profile():
    # Catches "returning positions dressed as transforms": an
    # implementation that only ever translates would leave the profile's
    # own y/z spread intact at every station. Under the relative-frame
    # contract a straight run legitimately IS a pure translation (the test
    # above), so the probe must be a station whose tangent differs from
    # segment 0's -- here the far end of an L, where the cross-section has
    # been turned through 90 degrees to face +Y.
    #
    # Expected positions, derived rather than observed: the end station
    # faces straight down the +Y segment, so the whole cross-section is
    # planar at y = 5. Travelling +X the profile sat 3..4 units to the
    # LEFT (+Y); travelling +Y, left is -X, so that same 3..4 becomes
    # x = 5 - 3 .. 5 - 4. Height on the profile (z) is untouched by either
    # turn.
    path = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]], dtype=np.float64)
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    ring = _ring(stations[-1], _OFFSET_PROFILE)
    expected = np.array(
        [[2.0, 5.0, 0.0], [1.0, 5.0, 0.0], [1.0, 5.0, 1.0], [2.0, 5.0, 2.0]]
    )
    assert ring == pytest.approx(expected, abs=1e-9)
    # A translate-only implementation could not have flattened the y
    # spread: the source profile spans a full unit of y.
    assert float(np.ptp(_OFFSET_PROFILE[:, 1])) == pytest.approx(1.0)
    assert float(np.ptp(ring[:, 1])) == pytest.approx(0.0, abs=1e-9)


def test_every_station_maps_the_path_start_onto_its_own_path_point():
    # The relative-frame contract's placement half: the frame's origin is
    # the PATH's own start, not the profile's centroid, so station i sends
    # path[0] -> path[i]. True at every station regardless of corner
    # tightness or winding, since the miter plane always passes through the
    # path point.
    path = np.array(
        [[0, 0, 0], [3, 0, 0], [3, 3, 0], [3, 3, 5]], dtype=np.float64
    )
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    for v, station in zip(path, stations, strict=True):
        assert _apply(station, path[0]) == pytest.approx(v, abs=1e-9)


def test_no_station_snaps_the_profile_centroid_onto_the_path():
    # The regression pin for the recentring defect, stated as the negative
    # of what the old contract asserted: with the profile held 3.5 units
    # off the path axis, a station that lands its centroid ON the path has
    # thrown that offset away.
    path = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0]], dtype=np.float64)
    stations = sweep_stations(_OFFSET_PROFILE, path, closed=False)
    for v, station in zip(path, stations, strict=True):
        centroid_image = _ring(station, _OFFSET_PROFILE).mean(axis=0)
        assert float(np.linalg.norm(centroid_image - v)) > 1.0


def test_a_straight_path_never_twists_the_profile_between_stations():
    # A wrong implementation that recomputes an arbitrary in-plane basis
    # per station (rather than transporting the profile's own orientation)
    # can introduce spurious rotation even along a dead-straight run.
    path = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    stations = sweep_stations(UNIT_SQUARE, path, closed=False)
    linear_parts = [s[:3, :3] for s in stations]
    for m in linear_parts[1:]:
        assert m == pytest.approx(linear_parts[0], abs=1e-9)


def test_a_sharp_but_survivable_corner_is_not_refused():
    # 170 degrees off straight (only 10 degrees short of a full reversal)
    # is about as sharp as a corner gets, but the profile is small (half
    # width 0.5) relative to the 10-unit segments either side of it, so
    # the miter comfortably fits. A blanket "reflex corners always refuse"
    # implementation would wrongly reject this.
    angle = np.radians(170.0)
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0 + 10.0 * np.cos(angle), 10.0 * np.sin(angle), 0.0],
        ],
        dtype=np.float64,
    )
    stations = sweep_stations(UNIT_SQUARE, path, closed=False)
    assert len(stations) == 3


def test_the_same_sharp_corner_is_refused_for_a_large_enough_profile():
    # Same 170-degree turn as above, same segment lengths -- only the
    # profile grows. This is the pair the brief asks for: a criterion
    # tight at both ends must accept the small profile and reject the
    # large one on the identical corner, not just react to angle alone.
    angle = np.radians(170.0)
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0 + 10.0 * np.cos(angle), 10.0 * np.sin(angle), 0.0],
        ],
        dtype=np.float64,
    )
    huge_profile = UNIT_SQUARE * 20.0
    with pytest.raises(SweepRefused):
        sweep_stations(huge_profile, path, closed=False)


def test_a_closed_path_first_and_last_station_are_both_mitered_at_the_seam():
    # Every vertex of a closed path has both a predecessor and a
    # successor, including vertex 0 (whose "incoming" segment is the
    # closing one back from the last point) -- so station 0's cross-section
    # must bisect the closing segment and the first segment, not just the
    # first and second like an open path's interior corners do.
    #
    # Checked on transformed positions of an asymmetric off-axis profile
    # rather than a dot product against a symmetric square: this is the
    # station Follow Me used to skip entirely, and neither a face count nor
    # a normal-direction dot product could see that.
    square_path = np.array(
        [[0, 0, 0], [12, 0, 0], [12, 12, 0], [0, 12, 0]], dtype=np.float64
    )
    stations = sweep_stations(_OFFSET_PROFILE, square_path, closed=True)
    seam = _ring(stations[0], _OFFSET_PROFILE)

    seam_normal = _ring_normal(seam)
    closing_seg = np.array([0.0, -1.0, 0.0])  # from vertex 3 back to vertex 0
    first_seg = np.array([1.0, 0.0, 0.0])  # from vertex 0 to vertex 1
    assert abs(np.dot(seam_normal, closing_seg)) == pytest.approx(
        abs(np.dot(seam_normal, first_seg)), abs=1e-9
    )
    # The seam's bisecting plane through (0, 0, 0) has normal
    # (1, -1, 0)/sqrt(2), i.e. x == y. Unlike an open path's station 0,
    # this one is emphatically NOT the identity -- the profile was drawn in
    # the x = 0 plane, and the seam miter shears it onto the diagonal.
    assert (seam[:, 0] - seam[:, 1]) == pytest.approx(0.0, abs=1e-9)
    assert np.abs(stations[0] - np.eye(4)).max() > 0.5
    assert seam == pytest.approx(
        np.array([[3.0, 3.0, 0.0], [4.0, 4.0, 0.0], [4.0, 4.0, 1.0], [3.0, 3.0, 2.0]]),
        abs=1e-9,
    )
