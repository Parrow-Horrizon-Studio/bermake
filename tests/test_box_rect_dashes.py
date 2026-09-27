"""#39: the crossing box-select rectangle draws dashed, not just green."""

from __future__ import annotations

import numpy as np


def test_a_solid_rect_is_still_four_segments():
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    segs = _box_rect_ndc_segments((10, 10, 110, 60), 800, 600)
    assert segs.shape == (8, 3)


def test_a_dashed_rect_has_more_segments_than_a_solid_one():
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    solid = _box_rect_ndc_segments((10, 10, 410, 310), 800, 600)
    dashed = _box_rect_ndc_segments((10, 10, 410, 310), 800, 600, dashed=True)
    assert dashed.shape[0] > solid.shape[0]


def test_dash_segments_come_in_pairs():
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    dashed = _box_rect_ndc_segments((10, 10, 410, 310), 800, 600, dashed=True)
    assert dashed.shape[0] % 2 == 0
    assert dashed.shape[1] == 3


def test_dashes_stay_inside_the_rect_in_ndc():
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    w, h = 800, 600
    rect = (100, 100, 500, 400)
    dashed = _box_rect_ndc_segments(rect, w, h, dashed=True)
    xs, ys = dashed[:, 0], dashed[:, 1]
    x0_ndc = (2.0 * rect[0] / w) - 1.0
    x1_ndc = (2.0 * rect[2] / w) - 1.0
    y0_ndc = 1.0 - (2.0 * rect[1] / h)
    y1_ndc = 1.0 - (2.0 * rect[3] / h)
    eps = 1e-5
    assert xs.min() >= min(x0_ndc, x1_ndc) - eps
    assert xs.max() <= max(x0_ndc, x1_ndc) + eps
    assert ys.min() >= min(y0_ndc, y1_ndc) - eps
    assert ys.max() <= max(y0_ndc, y1_ndc) + eps


def test_the_dash_period_is_stable_in_pixels_not_in_rect_fractions():
    """A dash period defined as a fraction of the rect would make dashes grow
    with the drag, which reads as a different line style at each size."""
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    small = _box_rect_ndc_segments((0, 0, 100, 100), 800, 600, dashed=True)
    large = _box_rect_ndc_segments((0, 0, 400, 400), 800, 600, dashed=True)
    # Four times the perimeter should give roughly four times the dashes.
    ratio = large.shape[0] / max(small.shape[0], 1)
    assert 3.0 <= ratio <= 5.0


def test_a_degenerate_rect_does_not_hang_or_divide_by_zero():
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    segs = _box_rect_ndc_segments((50, 50, 50, 50), 800, 600, dashed=True)
    assert np.all(np.isfinite(segs))


def test_the_overlay_carries_the_dashed_flag_for_crossing_mode():
    from bermake.scene import Scene
    from bermake.selection import Selection
    from bermake.tools.select_tool import SelectTool
    from bermake.tools.tool import ToolContext
    from bermake.viewport.camera import Camera

    cam = Camera()
    tool = SelectTool()
    tool.activate(
        ToolContext(
            scene=Scene(),
            camera=cam,
            widget_size_provider=lambda: (800, 600),
            selection=Selection(),
        )
    )
    tool._is_box = True
    tool._box_rect = (10.0, 10.0, 110.0, 60.0)
    tool._box_window = False
    assert tool.overlay().box_rect_dashed is True
    tool._box_window = True
    assert tool.overlay().box_rect_dashed is False
