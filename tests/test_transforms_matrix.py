# tests/test_transforms_matrix.py
import numpy as np
from bermake.geometry.transforms import (
    apply_mat, mat_compose, mat_invert, mat_rotate, mat_scale, mat_translate,
)


def test_translate_moves_points():
    M = mat_translate([1, 2, 3])
    out = apply_mat(np.array([[0, 0, 0], [1, 1, 1]], np.float32), M)
    assert np.allclose(out, [[1, 2, 3], [2, 3, 4]])


def test_scale_about_anchor():
    M = mat_scale([1, 0, 0], [2, 2, 2])
    out = apply_mat(np.array([[2, 0, 0]], np.float32), M)
    assert np.allclose(out, [[3, 0, 0]])  # (2-1)*2 + 1 = 3


def test_rotate_90_about_z_through_origin():
    M = mat_rotate([0, 0, 0], [0, 0, 1], np.pi / 2)
    out = apply_mat(np.array([[1, 0, 0]], np.float32), M)
    assert np.allclose(out, [[0, 1, 0]], atol=1e-6)


def test_invert_roundtrip():
    M = mat_compose(mat_translate([3, -1, 2]), mat_rotate([0, 0, 0], [0, 1, 0], 0.7))
    Minv = mat_invert(M)
    p = np.array([[4, 5, 6]], np.float32)
    back = apply_mat(apply_mat(p, M), Minv)
    assert np.allclose(back, p, atol=1e-5)


def test_compose_order_is_left_then_right():
    # translate by (1,0,0) THEN scale x2 about origin → (1,0,0) maps to (2,0,0)+... check a point at origin
    M = mat_compose(mat_translate([1, 0, 0]), mat_scale([0, 0, 0], [2, 2, 2]))
    out = apply_mat(np.array([[0, 0, 0]], np.float32), M)
    assert np.allclose(out, [[2, 0, 0]])
