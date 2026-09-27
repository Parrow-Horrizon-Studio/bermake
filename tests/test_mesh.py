"""Tests for the C++ Mesh class and make_cube primitive, accessed via nanobind."""

from __future__ import annotations

import numpy as np
import pytest

import bermake


def test_make_cube_default_returns_mesh():
    m = bermake.make_cube()
    assert isinstance(m, bermake.Mesh)


def test_make_cube_vertex_and_triangle_counts():
    m = bermake.make_cube()
    assert m.vertex_count == 24
    assert m.triangle_count == 12


def test_make_cube_array_shapes_and_dtypes():
    m = bermake.make_cube()
    assert m.positions.shape == (24, 3)
    assert m.normals.shape == (24, 3)
    assert m.indices.shape == (36,)
    assert m.positions.dtype == np.float32
    assert m.normals.dtype == np.float32
    assert m.indices.dtype == np.uint32


def test_make_cube_arrays_are_read_only():
    m = bermake.make_cube()
    with pytest.raises((ValueError, RuntimeError)):
        m.positions[0, 0] = 999.0
    with pytest.raises((ValueError, RuntimeError)):
        m.normals[0, 0] = 999.0
    with pytest.raises((ValueError, RuntimeError)):
        m.indices[0] = 999


def test_make_cube_bottom_on_ground():
    """The cube sits on z=0 with x,y centered around the origin."""
    m = bermake.make_cube(size=2.0)
    positions = np.asarray(m.positions)
    # x,y in [-1, 1]; z in [0, 2]
    assert positions[:, 0].min() == pytest.approx(-1.0)
    assert positions[:, 0].max() == pytest.approx(+1.0)
    assert positions[:, 1].min() == pytest.approx(-1.0)
    assert positions[:, 1].max() == pytest.approx(+1.0)
    assert positions[:, 2].min() == pytest.approx(0.0)
    assert positions[:, 2].max() == pytest.approx(2.0)


def test_make_cube_normals_are_unit_length():
    m = bermake.make_cube()
    normals = np.asarray(m.normals)
    lengths = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_make_cube_indices_in_range():
    m = bermake.make_cube()
    indices = np.asarray(m.indices)
    assert indices.min() >= 0
    assert indices.max() < m.vertex_count


def test_default_constructed_mesh_is_empty():
    m = bermake.Mesh()
    assert m.vertex_count == 0
    assert m.triangle_count == 0
    assert m.positions.shape == (0, 3)
    assert m.normals.shape == (0, 3)
    assert m.indices.shape == (0,)


def test_mesh_array_is_a_view_not_a_copy():
    """Accessing positions repeatedly should yield views that share memory."""
    m = bermake.make_cube()
    a = np.asarray(m.positions)
    b = np.asarray(m.positions)
    # Pointer equality is strictly stronger than np.may_share_memory (which
    # is conservative and can false-positive on overlapping-but-distinct
    # buffers). For our zero-copy `reference_internal` binding, both views
    # must point at the exact same underlying buffer.
    assert a.ctypes.data == b.ctypes.data
