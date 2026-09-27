"""M7.5b Task 2: projecting face corners to UVs, and placing the result."""

from __future__ import annotations

import numpy as np
import pytest
from bermake.viewport.uv_projection import (
    apply_placement,
    apply_placements,
    plane_bases,
    plane_basis,
    project_corners,
    project_onto_bases,
)

_SQ = np.array(
    [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64
)
_UP = np.array([0.0, 0.0, 1.0])
_CENTRE = np.array([1.0, 1.0, 0.0])


def test_the_basis_is_orthonormal_and_perpendicular_to_the_normal():
    for n in ([0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 1], [0.3, -0.7, 0.2]):
        u, v = plane_basis(np.asarray(n, dtype=np.float64))
        nn = np.asarray(n, dtype=np.float64) / np.linalg.norm(n)
        assert np.linalg.norm(u) == pytest.approx(1.0)
        assert np.linalg.norm(v) == pytest.approx(1.0)
        assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(u, nn)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(v, nn)) == pytest.approx(0.0, abs=1e-9)


def test_the_basis_is_deterministic_for_a_given_normal():
    # plane_basis takes no vertex argument, so this only pins purity/
    # determinism: same normal in, same basis out. The actual guarantee that
    # the basis does not depend on vertex order comes from the function's
    # signature (no vertex list to leak through) and is exercised end to end,
    # at the project_corners level, by
    # test_rotating_the_corner_list_does_not_move_the_uvs below.
    u1, v1 = plane_basis(_UP)
    u2, v2 = plane_basis(_UP.copy())
    assert np.allclose(u1, u2)
    assert np.allclose(v1, v2)


def test_rotating_the_corner_list_does_not_move_the_uvs():
    # The invariant test_the_basis_is_deterministic_for_a_given_normal cannot
    # reach: an edge-derived basis would give each physical corner a different
    # UV depending on where the loop starts. Rolling the corner list must not
    # move any physical corner's UV, because the basis comes from the normal
    # alone, never from positions[0] or positions[1].
    rolled = np.roll(_SQ, 1, axis=0)
    original_uvs = project_corners(_SQ, _UP, _CENTRE, (1.0, 1.0))
    rolled_uvs = project_corners(rolled, _UP, _CENTRE, (1.0, 1.0))
    np.testing.assert_allclose(rolled_uvs, np.roll(original_uvs, 1, axis=0), atol=1e-6)


def test_a_unit_texture_on_a_two_unit_square_tiles_twice():
    # texture_size is the image's extent in MODEL UNITS, so a 1x1 texture on a
    # 2x2 face spans 2 tiles across. A projection that normalised the face to
    # 0..1 instead would give a span of 1 and never tile.
    uvs = project_corners(_SQ, _UP, _CENTRE, (1.0, 1.0))
    span_u = float(uvs[:, 0].max() - uvs[:, 0].min())
    span_v = float(uvs[:, 1].max() - uvs[:, 1].min())
    assert span_u == pytest.approx(2.0)
    assert span_v == pytest.approx(2.0)


def test_a_wider_texture_tiles_fewer_times_across():
    # Kills a projection that ignores texture_size, and one that multiplies
    # where it should divide.
    uvs = project_corners(_SQ, _UP, _CENTRE, (4.0, 1.0))
    span_u = float(uvs[:, 0].max() - uvs[:, 0].min())
    span_v = float(uvs[:, 1].max() - uvs[:, 1].min())
    assert span_u == pytest.approx(0.5)
    assert span_v == pytest.approx(2.0)


def test_the_face_centroid_maps_to_the_uv_origin():
    uvs = project_corners(_SQ, _UP, _CENTRE, (1.0, 1.0))
    assert float(uvs[:, 0].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(uvs[:, 1].mean()) == pytest.approx(0.0, abs=1e-6)


def test_a_vertical_face_projects_without_collapsing():
    # The XY-only trap: a face whose normal is +X has zero extent in X, so any
    # projection that drops the Z axis collapses it to a line and every UV
    # lands on one axis.
    wall = np.array(
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 2.0], [0.0, 0.0, 2.0]], dtype=np.float64
    )
    uvs = project_corners(wall, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0]), (1.0, 1.0))
    assert float(uvs[:, 0].max() - uvs[:, 0].min()) == pytest.approx(2.0)
    assert float(uvs[:, 1].max() - uvs[:, 1].min()) == pytest.approx(2.0)


def test_projection_returns_float32_two_columns():
    uvs = project_corners(_SQ, _UP, _CENTRE, (1.0, 1.0))
    assert uvs.shape == (4, 2)
    assert uvs.dtype == np.float32


def test_an_empty_corner_set_projects_to_nothing():
    uvs = project_corners(np.zeros((0, 3)), _UP, _CENTRE, (1.0, 1.0))
    assert uvs.shape == (0, 2)
    assert uvs.dtype == np.float32


def test_the_identity_placement_changes_nothing():
    uvs = project_corners(_SQ, _UP, _CENTRE, (1.0, 1.0))
    assert np.allclose(apply_placement(uvs, 0.0, 0.0, 1.0, 0.0), uvs)


def test_offset_shifts_the_uvs():
    uvs = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    moved = apply_placement(uvs, 0.25, -0.5, 1.0, 0.0)
    assert np.allclose(moved, [[0.25, -0.5], [1.25, -0.5]])


def test_a_larger_scale_makes_the_texture_appear_larger():
    # scale 2 means the image covers twice as much surface, so the UV span
    # HALVES. Kills an implementation that multiplies instead of divides.
    uvs = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    scaled = apply_placement(uvs, 0.0, 0.0, 2.0, 0.0)
    assert float(scaled[1, 0] - scaled[0, 0]) == pytest.approx(1.0)


def test_rotation_turns_the_uvs_about_the_origin():
    # A stored `rotation` of +90 degrees means the IMAGE turns 90 degrees
    # counter-clockwise (v-up frame), and this function returns the sample
    # coordinate a fixed face point reads the texture at -- which moves the
    # opposite way from the image, exactly as paint_tool._pixel_delta_to_uv
    # documents for offset_u/offset_v. So a +90 degree rotation must turn the
    # raw (1, 0) UV to (0, -1): mathematically, rotating a vector by -90
    # degrees (cos(-90)=0, sin(-90)=-1) sends (x, y) -> (y, -x), i.e.
    # (1, 0) -> (0, -1).
    uvs = np.array([[1.0, 0.0]], dtype=np.float32)
    turned = apply_placement(uvs, 0.0, 0.0, 1.0, np.pi / 2.0)
    assert turned[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert turned[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_a_plus_90_rotation_turns_the_image_counter_clockwise_not_clockwise():
    # This is the eye-checkable pin for the sign convention itself, written so
    # a reader who does not want to trust the matrix algebra can still verify
    # it against a mental picture of a rotating photograph.
    #
    # `raw_right` is the raw UV of a face corner that, at rotation=0, samples
    # the RIGHT edge of the texture (texture coordinates run u=right, v=up,
    # per the v-up convention this milestone settled on and the module
    # docstring's contract). apply_placement returns the coordinate that same
    # FIXED face corner samples the texture at once a placement is applied --
    # not where a piece of the image moves to. paint_tool._pixel_delta_to_uv
    # spells out why those two are opposite for offset_u/offset_v ("increasing
    # offset_u samples the image further along +u for a FIXED surface point --
    # which looks like the image sliding toward -u on the surface"); the same
    # inversion applies to rotation.
    #
    # So: turn the image +90 degrees counter-clockwise, and ask what a viewer
    # now sees at the face corner that used to show the image's right edge.
    # Picture a photo with a distinct top, bottom, left and right (like the
    # scratchpad's asymmetric test texture). Rotate that photo 90 degrees
    # counter-clockwise: its BOTTOM edge swings around to where its RIGHT edge
    # used to be (bottom sits at -90 degrees on the clock face; +90 degrees of
    # counter-clockwise turn brings it to 0 degrees, i.e. the right). So the
    # face corner that used to show the image's right edge must now sample the
    # image's BOTTOM edge instead: raw (1, 0) must land at (0, -1), not (0, 1).
    # (0, 1) is the clockwise answer this test exists to rule out -- it is
    # what a viewer would see if the image had turned clockwise instead, and
    # it was this module's actual behaviour before the M7.5b rotation-sign fix.
    raw_right = np.array([[1.0, 0.0]], dtype=np.float32)
    sampled_after_ccw_90 = apply_placement(raw_right, 0.0, 0.0, 1.0, np.pi / 2.0)
    assert sampled_after_ccw_90[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert sampled_after_ccw_90[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_scale_is_applied_before_rotation_and_offset_last():
    # Pins the documented order. A different order passes every test above,
    # because each of those exercises one parameter at a time, and would put
    # Task 10's fields and Task 12's drag out of step with the renderer.
    uvs = np.array([[2.0, 0.0]], dtype=np.float32)
    out = apply_placement(uvs, 10.0, 0.0, 2.0, np.pi / 2.0)
    # scale: (2,0) -> (1,0);  a +90 degree rotation is applied to the UV as
    # -90 degrees (see test_rotation_turns_the_uvs_about_the_origin), so
    # (1,0) -> (0,-1);  offset: -> (10,-1)
    assert out[0, 0] == pytest.approx(10.0, abs=1e-6)
    assert out[0, 1] == pytest.approx(-1.0, abs=1e-6)


def test_placement_returns_float32():
    out = apply_placement(np.zeros((3, 2), dtype=np.float32), 0.0, 0.0, 1.0, 0.0)
    assert out.dtype == np.float32


# --- The batched siblings must agree with the scalar reference --------------
#
# The scalar trio above is the specification; the batched trio exists only
# because the renderer bakes every corner of every face on every upload and the
# per-face call overhead dominated it. These tests are what make that safe: if
# the two paths ever disagree, the fast one is wrong by definition.


def _varied_faces(seed=11):
    """(normal, origin, texture_size, placement) cases spanning the awkward ones.

    Deliberately includes both sides of the _AXIS_ALIGNED seed switch and its
    boundary, unnormalised and negative normals, a zero-length normal, zero
    texture extents (the scalar `or 1.0` fallback) and non-identity placements.
    """
    rng = np.random.default_rng(seed)
    normals = [
        [0.0, 0.0, 1.0],  # +Z, seed switch takes the X branch
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],  # in-plane, takes the Z branch
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.89],  # just below _AXIS_ALIGNED, unnormalised
        [0.0, 0.0, 0.91],  # just above
        [0.0, 0.6, 0.8],
        [0.0, -0.6, -0.8],
        [5.0, 0.0, 0.0],  # unnormalised
        [-1.0, -2.0, -3.0],
        [0.0, 0.0, 0.0],  # degenerate: no plane at all
    ]
    normals += [list(rng.normal(size=3)) for _ in range(40)]
    sizes = [(1.0, 1.0), (0.5, 4.0), (3.0, 0.25), (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)]
    placements = [
        (0.0, 0.0, 1.0, 0.0),  # identity
        (0.25, -0.5, 1.0, 0.0),
        (0.0, 0.0, 3.0, 0.0),
        (0.0, 0.0, 1.0, 0.7),
        (1.5, 2.5, 0.4, -2.1),
        (0.0, 0.0, 0.0, 0.0),  # zero scale, the `or 1.0` fallback
    ]
    out = []
    for i, n in enumerate(normals):
        out.append(
            (
                np.asarray(n, dtype=np.float64),
                rng.normal(scale=3.0, size=3),
                sizes[i % len(sizes)],
                placements[i % len(placements)],
            )
        )
    return out


def test_the_batched_basis_matches_the_scalar_basis_for_every_normal():
    cases = _varied_faces()
    normals = np.array([c[0] for c in cases])
    bu, bv = plane_bases(normals)
    for i, (n, _, _, _) in enumerate(cases):
        su, sv = plane_basis(n)
        np.testing.assert_allclose(bu[i], su, atol=1e-12, err_msg=f"u differs for {n}")
        np.testing.assert_allclose(bv[i], sv, atol=1e-12, err_msg=f"v differs for {n}")


def test_the_batched_projection_matches_the_scalar_one_corner_for_corner():
    # Built exactly as the renderer builds it: each face contributes several
    # corners, and the face's basis, origin and texture size are gathered onto
    # them, so a mis-gather shows up here rather than on screen.
    cases = _varied_faces()
    rng = np.random.default_rng(5)

    positions, origins, sizes, expected = [], [], [], []
    for normal, origin, size, _ in cases:
        corners = rng.normal(scale=2.0, size=(rng.integers(3, 9), 3))
        positions.append(corners)
        origins.append(np.repeat(origin[None, :], corners.shape[0], axis=0))
        sizes.append(np.repeat(np.asarray(size)[None, :], corners.shape[0], axis=0))
        expected.append(project_corners(corners, normal, origin, size))

    normals_per_corner = np.concatenate(
        [np.repeat(c[0][None, :], o.shape[0], axis=0) for c, o in zip(cases, origins, strict=True)]
    )
    u_axes, v_axes = plane_bases(normals_per_corner)
    batched = project_onto_bases(
        np.concatenate(positions),
        u_axes,
        v_axes,
        np.concatenate(origins),
        np.concatenate(sizes),
    )

    np.testing.assert_allclose(batched, np.concatenate(expected), rtol=0, atol=1e-6)
    assert batched.dtype == np.float32


def test_the_batched_placement_matches_the_scalar_one_row_for_row():
    cases = _varied_faces(seed=3)
    rng = np.random.default_rng(9)
    uvs = rng.normal(scale=4.0, size=(len(cases), 2))

    offsets = np.array([[p[0], p[1]] for _, _, _, p in cases])
    scales = np.array([p[2] for _, _, _, p in cases])
    rotations = np.array([p[3] for _, _, _, p in cases])

    batched = apply_placements(uvs, offsets, scales, rotations)
    for i, (_, _, _, p) in enumerate(cases):
        one = apply_placement(uvs[i : i + 1], p[0], p[1], p[2], p[3])
        np.testing.assert_allclose(batched[i], one[0], rtol=0, atol=1e-6)
    assert batched.dtype == np.float32


def test_the_batched_functions_answer_empty_input_like_the_scalar_ones():
    assert project_onto_bases(
        np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2))
    ).shape == (0, 2)
    assert apply_placements(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), np.zeros(0)).shape == (
        0,
        2,
    )
    u, v = plane_bases(np.zeros((0, 3)))
    assert u.shape == (0, 3) and v.shape == (0, 3)


def test_the_module_imports_nothing_from_model_scene_qt_or_gl():
    import subprocess
    import sys

    code = (
        "import bermake.viewport.uv_projection, sys; "
        "bad = [m for m in sys.modules if m.startswith(('PySide6','OpenGL','bermake.model','bermake.scene'))]; "
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
