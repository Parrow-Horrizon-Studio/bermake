"""Test that overlay geometry (selection segments, face polygons) is correctly
transformed by the active world transform when editing inside a moved instance.

GL draws are not testable headlessly, but the math path — building local-space
coords then applying apply_mat() with a translation — is pure NumPy and is
tested here directly.

Focus: confirm that apply_mat(segs, translate(+10x)) shifts x-coords by +10
so the code path added in the T19 fix produces world-space positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bermake.geometry.transforms import apply_mat, is_identity_transform
from bermake.selection import Selection
from bermake.viewport.scene_renderer import _selection_edge_segments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _translate_x(dx: float) -> np.ndarray:
    """4x4 float64 translation matrix that shifts +dx along X."""
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = dx
    return m


def _make_scene_with_edge():
    """Return (scene, edge_id) for a single edge from (1,2,3) to (4,5,6)."""
    from bermake.scene import Scene  # noqa: PLC0415

    scene = Scene()
    v0 = scene.add_vertex(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    e_id = scene.add_edge(v0, v1)
    return scene, e_id


# ---------------------------------------------------------------------------
# is_identity_transform short-circuit
# ---------------------------------------------------------------------------

class TestIdentityShortCircuit:
    def test_identity_matrix_is_identity(self):
        assert is_identity_transform(np.eye(4, dtype=np.float64))

    def test_none_is_identity(self):
        assert is_identity_transform(None)

    def test_translation_is_not_identity(self):
        assert not is_identity_transform(_translate_x(10.0))

    def test_tiny_numerical_noise_still_identity(self):
        m = np.eye(4, dtype=np.float64)
        m[0, 0] += 1e-16  # well within allclose tolerance
        assert is_identity_transform(m)


# ---------------------------------------------------------------------------
# _selection_edge_segments local coords
# ---------------------------------------------------------------------------

class TestSelectionEdgeSegmentsLocal:
    def test_segments_are_local_before_transform(self):
        """_selection_edge_segments returns the raw local vertex positions."""
        scene, e_id = _make_scene_with_edge()
        sel = Selection()
        sel.replace(edges=[e_id])

        segs = _selection_edge_segments(scene, sel)

        assert segs.shape == (2, 3)
        np.testing.assert_allclose(segs[0], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(segs[1], [4.0, 5.0, 6.0], atol=1e-5)

    def test_empty_selection_returns_empty(self):
        scene, _ = _make_scene_with_edge()
        sel = Selection()  # nothing selected

        segs = _selection_edge_segments(scene, sel)

        assert segs.shape == (0, 3)


# ---------------------------------------------------------------------------
# apply_mat: the transform path added in T19
# ---------------------------------------------------------------------------

class TestApplyMatTransformPath:
    def test_translate_x_shifts_segments(self):
        """Core math: apply_mat with +10x translation shifts x by 10."""
        scene, e_id = _make_scene_with_edge()
        sel = Selection()
        sel.replace(edges=[e_id])

        segs = _selection_edge_segments(scene, sel)
        world = _translate_x(10.0)

        transformed = apply_mat(segs, world)

        # x must be shifted by +10; y and z unchanged
        np.testing.assert_allclose(transformed[0], [11.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(transformed[1], [14.0, 5.0, 6.0], atol=1e-5)

    def test_identity_leaves_segments_unchanged(self):
        """Identity world transform must leave segment coords identical."""
        scene, e_id = _make_scene_with_edge()
        sel = Selection()
        sel.replace(edges=[e_id])

        segs = _selection_edge_segments(scene, sel)
        transformed = apply_mat(segs, np.eye(4, dtype=np.float64))

        np.testing.assert_allclose(transformed, segs, atol=1e-5)

    def test_polygon_translate_x(self):
        """apply_mat also works on (N,3) face-polygon arrays."""
        poly = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)
        world = _translate_x(5.0)

        transformed = apply_mat(poly, world)

        expected_x = np.array([5.0, 6.0, 6.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(transformed[:, 0], expected_x, atol=1e-5)
        # y and z unchanged
        np.testing.assert_allclose(transformed[:, 1], poly[:, 1], atol=1e-5)
        np.testing.assert_allclose(transformed[:, 2], poly[:, 2], atol=1e-5)


# ---------------------------------------------------------------------------
# Confirm the no-op guard: when is_identity → skip transform
# ---------------------------------------------------------------------------

class TestIdentityNoOp:
    def test_identity_guard_means_no_transform_applied(self):
        """Simulates the renderer guard: if is_identity, skip apply_mat."""
        scene, e_id = _make_scene_with_edge()
        sel = Selection()
        sel.replace(edges=[e_id])

        segs = _selection_edge_segments(scene, sel)
        world = np.eye(4, dtype=np.float64)  # root context

        # The guard in _draw_selection:
        need_transform = not is_identity_transform(world)
        if need_transform:
            segs = apply_mat(segs, world)

        # Must be unchanged — no transform was applied
        np.testing.assert_allclose(segs[0], [1.0, 2.0, 3.0], atol=1e-5)
        np.testing.assert_allclose(segs[1], [4.0, 5.0, 6.0], atol=1e-5)
