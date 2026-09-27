"""ToolOverlay generic-primitive fields + renderer screen projection helper."""

from __future__ import annotations

import numpy as np
from bermake.tools.tool import ToolOverlay
from bermake.viewport.scene_renderer import _screen_marker_ndc_quad


def _empty_overlay(**kw) -> ToolOverlay:
    base = {
        "rubber_band_segments": np.zeros((0, 3), np.float32),
        "rubber_band_color": (1, 1, 1),
        "snap_marker_position": None,
        "snap_marker_color": (1, 1, 1),
    }
    base.update(kw)
    return ToolOverlay(**base)


def test_overlay_defaults_empty_primitives():
    ov = _empty_overlay()
    assert ov.world_polylines == []
    assert ov.screen_markers == []


def test_overlay_carries_primitives():
    seg = np.zeros((2, 3), np.float32)
    ov = _empty_overlay(
        world_polylines=[(seg, (1, 0, 0), 2.0)],
        screen_markers=[(np.zeros(3, np.float32), 8.0, (0, 1, 0))],
    )
    assert len(ov.world_polylines) == 1
    assert len(ov.screen_markers) == 1


def test_screen_marker_ndc_quad_centers_on_pixel():
    # A 10px square centred at pixel (50, 50) in a 100x100 viewport →
    # NDC centre (0, 0), corners at ±0.1 in x and y (so the full x-span is 0.2:
    # 10px over a 100px viewport that maps to NDC [-1, 1] = 0.1 NDC/px * ... ;
    # 10px / 100px * 2 = 0.2 NDC of total width).
    quad = _screen_marker_ndc_quad(50.0, 50.0, 10.0, 100, 100)
    assert quad.shape == (4, 2)
    cx = float(quad[:, 0].mean())
    cy = float(quad[:, 1].mean())
    assert abs(cx) < 1e-6 and abs(cy) < 1e-6
    assert np.isclose(quad[:, 0].max() - quad[:, 0].min(), 0.2, atol=1e-6)
