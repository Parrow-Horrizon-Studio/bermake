"""Parallel, Perpendicular and From-Point inference against an acquired reference."""

from __future__ import annotations

import math

import numpy as np


def _camera_at_default():
    from bermake.viewport.camera import Camera

    cam = Camera()
    cam.aspect = 1280.0 / 800.0
    return cam


def _screen_of(cam, world):
    sx, sy, _ = cam.world_to_screen(np.asarray(world, dtype=np.float32), 1280, 800)
    return (sx, sy)


def _acquired_edge(position, direction):
    from bermake.viewport.inference import Acquired, AcquiredKind

    d = np.asarray(direction, dtype=np.float64)
    return Acquired(
        kind=AcquiredKind.EDGE,
        position=np.asarray(position, dtype=np.float64),
        direction=d / float(np.linalg.norm(d)),
        entity_id=0,
    )


def _acquired_vertex(position):
    from bermake.viewport.inference import Acquired, AcquiredKind

    return Acquired(
        kind=AcquiredKind.VERTEX,
        position=np.asarray(position, dtype=np.float64),
        direction=None,
        entity_id=0,
    )


def test_parallel_to_an_acquired_edge():
    """An edge running 1,1,0 gives a parallel inference along 1,1,0 from the anchor."""
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((5.0, 5.0, 0.0), (1.0, 1.0, 0.0))

    # A point on the parallel line through the anchor.
    probe = np.array([2.0, 2.0, 0.0], dtype=np.float32)
    res = SnapEngine().snap(
        _screen_of(cam, probe), (1280, 800), cam, scene, anchor=anchor, acquired=acquired
    )
    assert res.kind == SnapKind.PARALLEL, f"got {res.kind}"
    np.testing.assert_allclose(res.world_position, probe, atol=1e-3)


def test_perpendicular_resolves_in_the_drawing_plane():
    """Perpendicular to a 1,0,0 edge in the Z-normal plane runs along 0,1,0."""
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((3.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    probe = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    res = SnapEngine().snap(
        _screen_of(cam, probe),
        (1280, 800),
        cam,
        scene,
        anchor=anchor,
        acquired=acquired,
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert res.kind == SnapKind.PERPENDICULAR, f"got {res.kind}"
    np.testing.assert_allclose(res.world_position, probe, atol=1e-3)


def test_perpendicular_is_skipped_when_the_edge_runs_along_the_plane_normal():
    """cross(n, d) degenerates, so no candidate is offered rather than a guess."""
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))

    probe = np.array([0.0, 3.0, 0.0], dtype=np.float32)
    res = SnapEngine().snap(
        _screen_of(cam, probe),
        (1280, 800),
        cam,
        scene,
        anchor=anchor,
        acquired=acquired,
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert res.kind != SnapKind.PERPENDICULAR


def test_from_point_radiates_axes_from_an_acquired_vertex():
    """A point on the red axis THROUGH THE ACQUIRED VERTEX, not through the anchor."""
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_vertex((0.0, 4.0, 0.0))

    probe = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # on X through (0,4,0)
    res = SnapEngine().snap(
        _screen_of(cam, probe), (1280, 800), cam, scene, anchor=anchor, acquired=acquired
    )
    assert res.kind == SnapKind.FROM_POINT, f"got {res.kind}"
    assert res.axis == 0
    np.testing.assert_allclose(res.world_position, probe, atol=1e-3)


def test_no_acquisition_yields_no_directional_candidates():
    """The pre-M7.6b path must be unchanged when nothing is acquired."""
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    scene = Scene()
    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    probe = np.array([2.0, 2.0, 0.0], dtype=np.float32)

    res = SnapEngine().snap(_screen_of(cam, probe), (1280, 800), cam, scene, anchor=anchor)
    assert res.kind not in (SnapKind.PARALLEL, SnapKind.PERPENDICULAR, SnapKind.FROM_POINT)


# --- Direct generator tests -------------------------------------------------
#
# The three tests below call directional_candidates / from_point_candidates
# directly instead of going through SnapEngine.snap(). This is deliberate,
# not redundant with the end-to-end tests above: snap() has filters of its
# own that mask defects in these generators before they ever reach a
# SnapResult. Its tolerance filter drops a NaN screen_dist (nan <= tolerance
# is False, so a broken candidate is silently excluded from "within" as if it
# had never been offered), and its own "if acquired is not None" guard means
# a broken inner guard in directional_candidates is never even reached when
# acquired is None. Testing a pure generator through the engine that wraps it
# hid both defects; these tests exercise the generators at their own level so
# a later edit cannot reintroduce either bug unnoticed. Do not delete these as
# duplicates of the end-to-end tests above: they check a different thing.


def test_directional_candidates_emits_no_perpendicular_when_the_edge_is_the_plane_normal():
    """Direct call: the equivalent check through snap() passes for the wrong reason.

    cross(n, d) is the zero vector here, so normalising it produces NaN rather
    than raising. That NaN direction survives _line_candidate (nan > tolerance
    is False, so its guard does not reject it) and would reach SnapEngine.snap()
    as a PERPENDICULAR candidate with screen_dist=nan -- but snap()'s own
    "within tolerance" filter (nan <= tolerance is also False) drops it before
    it can be selected, so an end-to-end assertion on snap()'s result cannot
    tell a correct degeneracy guard from a missing one. This test inspects the
    generator's own return value, where the NaN candidate has nowhere to hide.
    """
    from bermake.viewport.snap_candidates import directional_candidates
    from bermake.viewport.snap_engine import SnapKind

    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((0.0, 0.0, 3.0), (0.0, 0.0, 1.0))

    out = directional_candidates(
        640.0,
        400.0,
        1280,
        800,
        cam,
        anchor,
        acquired,
        np.array([0.0, 0.0, 1.0]),
        8.0,
    )
    assert not any(c.kind == SnapKind.PERPENDICULAR for c in out), f"got {out}"


def test_directional_candidates_emits_no_perpendicular_for_a_non_unit_edge_direction():
    """The degeneracy guard must key off cross(n, d), not dot(n, d).

    dot(n, d) only tracks the angle between n and d when d is unit length.
    Here d=(0, 0, 0.5) is parallel to n=(0, 0, 1) but half its length, so
    dot(n, d) == 0.5, well inside the "not degenerate" range the old guard
    checked -- yet cross(n, d) is still the zero vector, so normalising it
    still produces NaN. A guard keyed on dot alone lets this case through
    and hands back a "guessed" PERPENDICULAR candidate at [nan, nan, nan],
    exactly the outcome the requirement forbids. This constructs the Acquired
    directly (not via _acquired_edge, which normalises) so the non-unit
    direction survives intact.
    """
    from bermake.viewport.inference import Acquired, AcquiredKind
    from bermake.viewport.snap_candidates import directional_candidates
    from bermake.viewport.snap_engine import SnapKind

    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = Acquired(
        kind=AcquiredKind.EDGE,
        position=np.array([0.0, 0.0, 3.0], dtype=np.float64),
        direction=np.array([0.0, 0.0, 0.5], dtype=np.float64),
        entity_id=0,
    )

    out = directional_candidates(
        640.0,
        400.0,
        1280,
        800,
        cam,
        anchor,
        acquired,
        np.array([0.0, 0.0, 1.0]),
        8.0,
    )
    assert not any(c.kind == SnapKind.PERPENDICULAR for c in out), f"got {out}"


def test_directional_candidates_returns_nothing_with_no_acquisition():
    """Direct call: snap()'s own "if acquired is not None" guard hides this.

    snap() never calls directional_candidates at all when acquired is None,
    so mutating this function's own None-guard is invisible to any assertion
    made through SnapEngine.snap(). This test calls the generator directly so
    a broken guard here cannot hide behind the engine's outer one.
    """
    from bermake.viewport.snap_candidates import directional_candidates

    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    out = directional_candidates(
        640.0, 400.0, 1280, 800, cam, anchor, None, np.array([0.0, 0.0, 1.0]), 8.0
    )
    assert out == []


def test_from_point_candidates_returns_nothing_with_no_acquisition():
    """Direct call, for the same reason as the directional_candidates case above.

    from_point_candidates has no anchor-vs-acquired ambiguity to hide behind,
    but it is still only reachable through snap() when acquired is not None,
    so its own None-guard is likewise untested by any end-to-end assertion.
    """
    from bermake.viewport.snap_candidates import from_point_candidates

    cam = _camera_at_default()

    out = from_point_candidates(640.0, 400.0, 1280, 800, cam, None, 8.0)
    assert out == []


def test_marker_colors_reflect_the_d14_intersection_move():
    """Guards the deliberate colour change: no test previously asserted these.

    D14 moves INTERSECTION off magenta (SketchUp reserves magenta for Parallel
    and Perpendicular) to near-black, and gives PARALLEL and PERPENDICULAR the
    freed-up magenta. Without an assertion on the actual values, a revert of
    this change -- accidental or "helpful" -- would pass every other test.
    """
    from bermake.viewport.snap_engine import MARKER_COLOR_BY_KIND, SnapKind

    assert MARKER_COLOR_BY_KIND[SnapKind.INTERSECTION] == (0.10, 0.10, 0.12)
    assert MARKER_COLOR_BY_KIND[SnapKind.PARALLEL] == (0.82, 0.23, 0.82)
    assert MARKER_COLOR_BY_KIND[SnapKind.PERPENDICULAR] == (0.82, 0.23, 0.82)


def _snap_at(cam, probe, **kwargs):
    from bermake.scene import Scene
    from bermake.viewport.snap_engine import SnapEngine

    return SnapEngine().snap(_screen_of(cam, probe), (1280, 800), cam, Scene(), **kwargs)


def test_shift_locks_a_parallel_inference():
    """Spec 2.3: Shift holds "whatever inference is currently showing".

    Magenta Parallel is one of the inferences a SketchUp user reaches for
    Shift with most. Before the fix `_direction_of` only understood a snap
    that carried an `axis`, which PARALLEL never does, so holding Shift over
    one silently did nothing at all.
    """
    from bermake.viewport.inference import InferenceState
    from bermake.viewport.snap_engine import SnapKind

    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((5.0, 5.0, 0.0), (1.0, 1.0, 0.0))
    snap = _snap_at(cam, [2.0, 2.0, 0.0], anchor=anchor, acquired=acquired)
    assert snap.kind == SnapKind.PARALLEL

    state = InferenceState()
    state.set_shift_lock(True, snap)

    assert state.lock is not None
    unit = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    np.testing.assert_allclose(state.lock.direction, unit, atol=1e-6)


def test_shift_locks_a_perpendicular_inference():
    """Perpendicular carries no axis either, and its direction depends on the
    drawing plane, which InferenceState never sees. It rides on the snap."""
    from bermake.viewport.inference import InferenceState
    from bermake.viewport.snap_engine import SnapKind

    cam = _camera_at_default()
    anchor = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    acquired = _acquired_edge((5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    snap = _snap_at(
        cam,
        [0.0, 3.0, 0.0],
        anchor=anchor,
        acquired=acquired,
        plane_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert snap.kind == SnapKind.PERPENDICULAR

    state = InferenceState()
    state.set_shift_lock(True, snap)

    assert state.lock is not None
    np.testing.assert_allclose(np.abs(state.lock.direction), [0.0, 1.0, 0.0], atol=1e-6)


def test_shift_locks_an_on_guide_inference():
    """A guide's direction is nowhere in the snapped point's own geometry and
    nowhere near the gesture anchor, so it has to be carried on the snap."""
    from bermake.viewport.inference import InferenceState
    from bermake.viewport.snap_engine import SnapKind

    cam = _camera_at_default()
    guide_origin = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    guide_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    snap = _snap_at(cam, [1.0, 0.0, 4.0], guides=[(guide_origin, guide_direction)])
    assert snap.kind == SnapKind.ON_GUIDE

    state = InferenceState()
    state.set_shift_lock(True, snap)

    assert state.lock is not None
    np.testing.assert_allclose(np.abs(state.lock.direction), [0.0, 0.0, 1.0], atol=1e-6)


def test_shift_over_a_point_inference_locks_nothing():
    """An endpoint answers "where", not "which way". There is no line to
    hold, so Shift declines rather than inventing a direction."""
    from bermake.scene import Scene
    from bermake.viewport.inference import InferenceState
    from bermake.viewport.snap_engine import SnapEngine, SnapKind

    cam = _camera_at_default()
    scene = Scene()
    scene.add_vertex(np.array([2.0, 1.0, 0.0], dtype=np.float32))
    snap = SnapEngine().snap(_screen_of(cam, [2.0, 1.0, 0.0]), (1280, 800), cam, scene)
    assert snap.kind == SnapKind.ENDPOINT

    state = InferenceState()
    state.set_shift_lock(True, snap)

    assert state.lock is None
