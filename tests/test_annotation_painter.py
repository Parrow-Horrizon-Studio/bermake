"""Tests for the M7d annotation painter (Task 4).

paint_annotation_plans is deliberately painter-agnostic: it works against any
object exposing setPen/drawLine/drawText, so these tests use a recording stub
instead of a real QPainter (keeping the suite headless).
"""

from __future__ import annotations

import numpy as np
from bermake.annotations.draw_plan import AnnotationDraw, TextDraw
from bermake.viewport.annotation_painter import paint_annotation_plans

# M7.6b Task 9 fix round 3: this file used to hand-roll its own
# `_FakeViewport`, which is exactly how it drifted from the real
# ViewportWidget when Task 7 added a `show_guides` read (fix round 1 patched
# that one attribute locally). Now shared with test_annotation_render_scope.py
# (and any future file that needs to drive _paint_annotations headlessly) via
# tests/_annotation_paint_fakes.py, so a future read added to
# _paint_annotations only needs updating in one place. See that module's
# docstring for the full attribute list and why it must stay in sync.
from tests._annotation_paint_fakes import FakeViewport as _FakeViewport


class _RecordingPainter:
    """Stands in for a QPainter. Exposes Qt's camelCase method names (setPen /
    drawLine / drawText) as instance attributes bound to snake_case methods,
    since ruff's N802 forbids camelCase `def` names but the painter contract
    paint_annotation_plans calls against is Qt's, not ours."""

    def __init__(self):
        self.lines = []
        self.texts = []
        self.pens = []
        self.setPen = self._set_pen
        self.drawLine = self._draw_line
        self.drawText = self._draw_text

    def _set_pen(self, pen):
        self.pens.append(pen)

    def _draw_line(self, x1, y1, x2, y2):
        self.lines.append((x1, y1, x2, y2))

    def _draw_text(self, x, y, s):
        self.texts.append((x, y, s))


def _plan(ann_id=1):
    return AnnotationDraw(
        annotation_id=ann_id,
        segments_px=[(0.0, 0.0, 10.0, 0.0), (10.0, 0.0, 10.0, 10.0)],
        texts=[TextDraw("3600", 5.0, -2.0, "center")],
        hit_boxes=[],
    )


def test_paints_every_segment_and_text():
    p = _RecordingPainter()
    paint_annotation_plans(p, [_plan()], (0.1, 0.1, 0.1), set(), (0.2, 0.5, 0.9))
    assert len(p.lines) == 2
    assert len(p.texts) == 1
    assert p.texts[0][2] == "3600"


def test_selected_annotation_uses_the_selection_pen():
    """The selected colour must be the one actually passed to setPen — not
    merely that *a* pen was set (which would pass even if the selection
    branch were inverted)."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    paint_annotation_plans(p, [_plan(7)], color, {7}, selected_color)
    assert p.pens == [selected_color]


def test_unselected_annotation_uses_the_default_pen():
    """Complementary case: an annotation NOT in selected_ids must get the
    default colour, not the selection colour."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    paint_annotation_plans(p, [_plan(7)], color, set(), selected_color)
    assert p.pens == [color]


def test_empty_plan_list_draws_nothing():
    p = _RecordingPainter()
    paint_annotation_plans(p, [], (0.1, 0.1, 0.1), set(), (0.2, 0.5, 0.9))
    assert p.lines == [] and p.texts == []


def test_hovered_annotation_uses_the_hover_pen():
    """Task 10b: a hovered (but not selected) annotation must be painted with
    the hover colour, distinct from both the default and the selected pen."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    hover_color = (0.45, 0.70, 1.00)
    paint_annotation_plans(
        p, [_plan(7)], color, set(), selected_color, hovered_id=7, hover_color=hover_color
    )
    assert p.pens == [hover_color]


def test_unhovered_annotation_uses_the_default_pen_when_hover_param_set():
    """A hovered_id that doesn't match this plan's id must not leak the hover
    colour onto an unrelated annotation."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    hover_color = (0.45, 0.70, 1.00)
    paint_annotation_plans(
        p, [_plan(7)], color, set(), selected_color, hovered_id=99, hover_color=hover_color
    )
    assert p.pens == [color]


def test_hovered_and_selected_annotation_uses_the_selected_pen_not_hover():
    """Precedence: selected beats hover, exactly as it does for edges/faces/
    instances (Task 10b design decision)."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    hover_color = (0.45, 0.70, 1.00)
    paint_annotation_plans(
        p, [_plan(7)], color, {7}, selected_color, hovered_id=7, hover_color=hover_color
    )
    assert p.pens == [selected_color]


def test_no_hovered_id_uses_the_default_pen():
    """Backward-compatible default: omitting hovered_id/hover_color entirely
    (as the pre-Task-10b callers do) must behave exactly as before."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    paint_annotation_plans(p, [_plan(7)], color, set(), selected_color)
    assert p.pens == [color]


def _guide_plan(ann_id=1, kind="guide"):
    return AnnotationDraw(
        annotation_id=ann_id,
        segments_px=[(0.0, 0.0, 10.0, 0.0)],
        texts=[],
        hit_boxes=[],
        kind=kind,
    )


def test_guide_plan_uses_the_guide_pen_when_one_is_supplied():
    """Task 6: a guide's plan is painted with `guide_pen`, not the ordinary
    default colour, when the caller supplies one."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    guide_pen = (0.45, 0.45, 0.52)
    paint_annotation_plans(p, [_guide_plan()], color, set(), selected_color, guide_pen=guide_pen)
    assert p.pens == [guide_pen]


def test_guide_point_plan_also_uses_the_guide_pen():
    """The same treatment applies to guide_point plans, not only guide plans."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    guide_pen = (0.45, 0.45, 0.52)
    paint_annotation_plans(
        p, [_guide_plan(kind="guide_point")], color, set(), selected_color, guide_pen=guide_pen
    )
    assert p.pens == [guide_pen]


def test_guide_plan_ignores_selection_and_hover_when_a_guide_pen_is_supplied():
    """Guides are construction geometry, not measured annotations: selection
    and hover state must not override the guide pen once one is supplied."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    hover_color = (0.45, 0.70, 1.00)
    guide_pen = (0.45, 0.45, 0.52)
    paint_annotation_plans(
        p,
        [_guide_plan(7)],
        color,
        {7},
        selected_color,
        hovered_id=7,
        hover_color=hover_color,
        guide_pen=guide_pen,
    )
    assert p.pens == [guide_pen]


def test_guide_plan_falls_back_to_the_default_pen_when_no_guide_pen_is_given():
    """Backward-compatible default: omitting `guide_pen` entirely must leave a
    guide plan on the ordinary colour rules, matching pre-Task-6 behaviour."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    paint_annotation_plans(p, [_guide_plan()], color, set(), selected_color)
    assert p.pens == [color]


def test_a_non_guide_plan_is_unaffected_by_a_supplied_guide_pen():
    """A dimension/label plan (kind not in the guide set) keeps the pen it
    already had even when the caller supplies a guide_pen for other plans in
    the same call."""
    p = _RecordingPainter()
    color = (0.1, 0.1, 0.1)
    selected_color = (0.2, 0.5, 0.9)
    guide_pen = (0.45, 0.45, 0.52)
    paint_annotation_plans(p, [_plan(7)], color, set(), selected_color, guide_pen=guide_pen)
    assert p.pens == [color]


class _RecordingQPainter:
    """Stands in for QPainter inside ViewportWidget._paint_annotations.

    `self` in that method is a duck-typed stand-in below, not a real
    QPaintDevice, so the real QPainter can't be constructed against it. This
    fake is swapped in via monkeypatch (PySide6.QtGui.QPainter is looked up
    fresh on each call because the method does a local `from PySide6.QtGui
    import QPainter`), so the method under test runs unmodified.

    Exposes Qt's camelCase method names as instance attributes bound to
    snake_case methods, matching the _RecordingPainter convention above
    (ruff's N802 forbids camelCase `def` names, but the contract here is
    Qt's, not ours).
    """

    class RenderHint:
        Antialiasing = 1  # only value the method under test references

    last = None  # set to the most recently constructed instance

    def __init__(self, _device):
        self.pens = []
        self.texts = []
        self.setRenderHint = self._set_render_hint
        self.setFont = self._set_font
        self.setPen = self._set_pen
        self.drawLine = self._draw_line
        self.drawText = self._draw_text
        type(self).last = self

    def _set_render_hint(self, *args, **kwargs):
        pass

    def _set_font(self, *args, **kwargs):
        pass

    def _set_pen(self, pen):
        self.pens.append(pen)

    def _draw_line(self, x1, y1, x2, y2):
        pass

    def _draw_text(self, x, y, s):
        self.texts.append((x, y, s))

    def end(self):
        pass


class _FakeContext:
    def __init__(self, annotations):
        self.annotations = annotations


class _FakeModel:
    """Stands in for bermake.model.model.Model: exposes exactly what
    collect_annotation_plans (called by the real _paint_annotations) reads --
    traverse_visible() and definition_is_dimmed() -- for a single, root-only
    context with no groups. active_path is empty, so definition_is_dimmed
    always returns False here, matching a real root Model (#95, Task 13)."""

    def __init__(self, annotations):
        self.active_context = _FakeContext(annotations)
        self.active_world_transform = np.eye(4)
        self.active_path = []

    def traverse_visible(self):
        yield self.active_context, self.active_world_transform

    def definition_is_dimmed(self, definition):
        return False


def test_paint_annotations_defaults_to_real_units_when_no_provider_set(monkeypatch):
    """Regression test for the M7d Task 4 review (Finding 1).

    MainWindow does not currently call set_units_provider, so `units` being
    None is the default configuration today, not an edge case. Before the
    fix, _paint_annotations passed units=None straight through to
    plan_annotation -> _plan_dimension -> format_length(measured, None),
    which crashes: AttributeError: 'NoneType' object has no attribute
    'system'. It must default to a real bermake.units.Units() instead,
    matching every other units provider in the codebase (wall/opening/roof
    options bars).

    This calls the real _paint_annotations method (not a reimplementation of
    its units logic), faking only QPainter -- since `self` here isn't an
    actual QPaintDevice -- to keep the test headless with no QApplication or
    real window.
    """
    from bermake.model.annotation import Dimension
    from bermake.viewport.camera import Camera
    from bermake.viewport.viewport_widget import ViewportWidget
    from PySide6 import QtGui

    monkeypatch.setattr(QtGui, "QPainter", _RecordingQPainter)

    dim = Dimension(id=1, p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0), offset=(0.0, 0.0, 0.2))
    fake_viewport = _FakeViewport(_FakeModel([dim]), Camera())

    ViewportWidget._paint_annotations(fake_viewport)  # must not raise

    drawn_strings = [text for _x, _y, text in _RecordingQPainter.last.texts]
    assert drawn_strings == ["1 m"]


def test_paint_annotations_paints_the_hovered_annotation_with_the_hover_pen(monkeypatch):
    """Task 10b wiring test: _paint_annotations must thread the active tool's
    hovered annotation id (surfaced via ToolOverlay.hovered_annotation_id --
    the same per-frame channel paintGL already uses for edge/face/instance
    hover) through to paint_annotation_plans.

    Reachable headlessly: paintGL passes the tool's freshly-computed
    ToolOverlay into _paint_annotations as an explicit argument, so this test
    can hand in a plain ToolOverlay instance directly -- no real GL widget,
    QApplication, or ToolManager needed.
    """
    import numpy as np
    from bermake.model.annotation import Dimension
    from bermake.tools.select_tool import _HOVER_EDGE_COLOR
    from bermake.tools.tool import ToolOverlay
    from bermake.viewport.camera import Camera
    from bermake.viewport.viewport_widget import ViewportWidget
    from PySide6 import QtGui
    from PySide6.QtGui import QColor

    monkeypatch.setattr(QtGui, "QPainter", _RecordingQPainter)

    dim = Dimension(id=1, p1=(0.0, 0.0, 0.0), p2=(1.0, 0.0, 0.0), offset=(0.0, 0.0, 0.2))
    fake_viewport = _FakeViewport(_FakeModel([dim]), Camera())
    overlay = ToolOverlay(
        rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
        rubber_band_color=(0.0, 0.0, 0.0),
        snap_marker_position=None,
        snap_marker_color=(0.0, 0.0, 0.0),
        hovered_annotation_id=1,
    )

    ViewportWidget._paint_annotations(fake_viewport, overlay)

    expected_hover_color = QColor(
        round(_HOVER_EDGE_COLOR[0] * 255),
        round(_HOVER_EDGE_COLOR[1] * 255),
        round(_HOVER_EDGE_COLOR[2] * 255),
    )
    hover_pen = _RecordingQPainter.last.pens[0]
    assert isinstance(hover_pen, QColor)
    assert hover_pen == expected_hover_color
