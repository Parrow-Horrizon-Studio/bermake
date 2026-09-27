"""Unit tests for pure transform math (no Qt/GL/Scene)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from bermake.geometry.transforms import rotate, scale, translate


def test_translate_known():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    out = translate(pts, [10, 0, -1])
    assert np.allclose(out, [[11, 2, 2], [14, 5, 5]])
    assert out.dtype == np.float32


def test_translate_zero_is_identity():
    pts = np.array([[1, 2, 3]], dtype=np.float32)
    assert np.allclose(translate(pts, [0, 0, 0]), pts)


def test_rotate_90_about_z_origin():
    pts = np.array([[1, 0, 0]], dtype=np.float32)
    out = rotate(pts, center=[0, 0, 0], axis=[0, 0, 1], angle_rad=math.pi / 2)
    assert np.allclose(out, [[0, 1, 0]], atol=1e-5)


def test_rotate_about_offset_center_keeps_center_fixed():
    c = np.array([5, 5, 0], dtype=np.float32)
    out = rotate(c.reshape(1, 3), center=c, axis=[0, 0, 1], angle_rad=1.2345)
    assert np.allclose(out, c.reshape(1, 3), atol=1e-5)


def test_rotate_zero_angle_is_identity():
    pts = np.array([[2, -3, 4]], dtype=np.float32)
    out = rotate(pts, center=[0, 0, 0], axis=[0, 1, 0], angle_rad=0.0)
    assert np.allclose(out, pts, atol=1e-6)


def test_rotate_degenerate_axis_raises():
    with pytest.raises(ValueError):
        rotate(np.zeros((1, 3), np.float32), center=[0, 0, 0], axis=[0, 0, 0], angle_rad=1.0)


def test_scale_anisotropic_about_anchor():
    pts = np.array([[2, 2, 2]], dtype=np.float32)
    out = scale(pts, anchor=[0, 0, 0], factors=[2, 1, 0.5])
    assert np.allclose(out, [[4, 2, 1]])


def test_scale_keeps_anchor_fixed():
    anchor = np.array([1, 1, 1], dtype=np.float32)
    out = scale(anchor.reshape(1, 3), anchor=anchor, factors=[3, 3, 3])
    assert np.allclose(out, anchor.reshape(1, 3))


def test_scale_factor_one_is_identity():
    pts = np.array([[7, 8, 9]], dtype=np.float32)
    assert np.allclose(scale(pts, anchor=[1, 1, 1], factors=[1, 1, 1]), pts)
