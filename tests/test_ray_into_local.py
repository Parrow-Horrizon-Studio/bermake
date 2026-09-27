"""Tests for picking.ray_into_local — identity no-op and translation case."""
from __future__ import annotations

import numpy as np
import pytest

from bermake.viewport.picking import ray_into_local
from bermake.geometry.transforms import mat_translate


_ORIGIN = np.array([1.0, 2.0, 3.0], dtype=np.float32)
_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def test_none_transform_is_noop():
    o, d = ray_into_local(_ORIGIN, _DIRECTION, None)
    np.testing.assert_allclose(o, _ORIGIN, atol=1e-6)
    np.testing.assert_allclose(d, _DIRECTION, atol=1e-6)


def test_identity_transform_is_noop():
    eye = np.eye(4, dtype=np.float64)
    o, d = ray_into_local(_ORIGIN, _DIRECTION, eye)
    np.testing.assert_allclose(o, _ORIGIN, atol=1e-6)
    np.testing.assert_allclose(d, _DIRECTION, atol=1e-6)


def test_translation_shifts_origin_not_direction():
    """world_transform = translate(+10, 0, 0)
    A world-space origin at (10, 0, 5) should map to local (0, 0, 5).
    A direction (0, 0, -1) should be unchanged (pure translation → identity 3x3)."""
    wt = mat_translate(np.array([10.0, 0.0, 0.0]))
    world_origin = np.array([10.0, 0.0, 5.0], dtype=np.float32)
    world_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    local_o, local_d = ray_into_local(world_origin, world_dir, wt)

    np.testing.assert_allclose(local_o, [0.0, 0.0, 5.0], atol=1e-6)
    np.testing.assert_allclose(local_d, [0.0, 0.0, -1.0], atol=1e-6)
