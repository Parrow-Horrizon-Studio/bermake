import numpy as np
from bermake.annotations.draw_plan import collect_annotation_plans, plan_annotation
from bermake.model.annotation import Dimension
from bermake.model.model import Model
from bermake.units import Units
from bermake.viewport.camera import Camera

# M7.6b Task 9 fix round 3: this file used to hand-roll its own
# `_FakeViewport`, missing every attribute Task 9 added a read for
# (tool_manager, _vcb_active_provider, _last_cursor_px) -- the same class of
# drift that broke test_annotation_painter.py's own copy over `show_guides`
# in fix round 1. Now shared via tests/_annotation_paint_fakes.py, so a
# future read added to _paint_annotations only needs updating in one place.
# See that module's docstring for the full attribute list and why it must
# stay in sync.
from tests._annotation_paint_fakes import FakeViewport as _FakeViewport


def _model_with_annotation_inside_a_group():
    model = Model()
    inner = model.new_definition("Group", is_group=True)
    inner.annotations.append(Dimension(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.2, 0.0)))
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = 5.0  # offset so world != local
    inst = model.new_instance(inner, t)
    model.root.children.append(inst)
    return model, inner, inst


def test_annotation_in_a_non_active_context_is_still_planned():
    """From the root, a dimension living inside a group must still produce a
    draw plan -- it should render (dimmed), not vanish."""
    model, inner, _inst = _model_with_annotation_inside_a_group()
    assert model.active_context is model.root  # not inside the group
    cam = Camera()
    cam.aspect = 800.0 / 600.0

    planned = []
    for defn, world in model.traverse_visible():
        for ann in defn.annotations:
            p = plan_annotation(ann, world, cam, 800, 600, Units())
            if p is not None:
                planned.append((defn, p))

    assert planned, "the group's annotation must be planned from the root context"
    assert planned[0][0] is inner


def test_active_context_annotations_are_distinguishable_from_the_rest():
    model, inner, _inst = _model_with_annotation_inside_a_group()
    active = model.active_context
    others = [d for d, _ in model.traverse_visible() if d is not active and d.annotations]
    assert inner in others, "the group is not the active context, so it must dim"


def test_collect_annotation_plans_dims_by_the_geometry_dim_rule_not_active_context():
    """collect_annotation_plans must dim EXACTLY when geometry dims -- i.e.
    Model.definition_is_dimmed(defn) -- NOT "defn is not active_context".
    At the root (active_path empty) nothing is dimmed, matching geometry,
    even though the group is not the active context."""
    model, _inner, _inst = _model_with_annotation_inside_a_group()
    cam = Camera()
    cam.aspect = 800.0 / 600.0

    plans = collect_annotation_plans(model, cam, 800, 600, Units())
    assert len(plans) == 1
    plan, dimmed = plans[0]
    assert plan.annotation_id == 0
    assert dimmed is False, "at root, nothing is dimmed -- matches geometry (spec D4)"


def test_collect_annotation_plans_dims_the_previously_active_context_after_entering_group():
    """After entering the group, the group's own annotation renders full
    strength (it is now the active context) while the root's annotation --
    no longer active -- dims. Mirrors definition_is_dimmed exactly."""
    model, inner, inst = _model_with_annotation_inside_a_group()
    root_ann = Dimension(1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.2, 0.0))
    model.root.annotations.append(root_ann)
    cam = Camera()
    cam.aspect = 800.0 / 600.0

    model.enter(inst)
    assert model.active_context is inner

    plans = collect_annotation_plans(model, cam, 800, 600, Units())
    dimmed_by_id = {plan.annotation_id: dimmed for plan, dimmed in plans}
    assert dimmed_by_id[0] is False, "the group is the active context: full colour"
    assert dimmed_by_id[1] is True, "the root is no longer active: dims"


# ---------------------------------------------------------------------------
# Widget-level: this is the test the mandatory revert-and-observe step
# targets. Unlike the pure model-level tests above (which exercise
# traverse_visible / plan_annotation / collect_annotation_plans directly and
# would stay green even if _paint_annotations itself regressed), this test
# calls ViewportWidget._paint_annotations, so it actually catches the
# original defect: reverting _paint_annotations to the single-context read
# makes this go RED.
# ---------------------------------------------------------------------------


class _RecordingQPainter:
    """Minimal QPainter stand-in (mirrors test_annotation_painter.py's), so
    _paint_annotations can run headlessly against a duck-typed viewport."""

    class RenderHint:
        Antialiasing = 1

    last = None

    def __init__(self, _device):
        self.pens = []
        self.texts = []
        self.setRenderHint = lambda *a, **k: None
        self.setFont = lambda *a, **k: None
        self.setPen = self._set_pen
        self.drawLine = lambda *a, **k: None
        self.drawText = self._draw_text
        type(self).last = self

    def _set_pen(self, pen):
        self.pens.append(pen)

    def _draw_text(self, x, y, s):
        self.texts.append((x, y, s))

    def end(self):
        pass


def test_paint_annotations_still_draws_the_groups_dimension_from_the_root(monkeypatch):
    """The actual regression under test (#95): ViewportWidget._paint_annotations
    must not vanish a dimension living inside a group just because the root
    (not the group) is the active context."""
    from bermake.viewport.viewport_widget import ViewportWidget
    from PySide6 import QtGui

    monkeypatch.setattr(QtGui, "QPainter", _RecordingQPainter)
    _RecordingQPainter.last = None

    model, _inner, _inst = _model_with_annotation_inside_a_group()
    assert model.active_context is model.root
    fake_viewport = _FakeViewport(model, Camera())

    ViewportWidget._paint_annotations(fake_viewport)

    assert _RecordingQPainter.last is not None, "must still paint something"
    assert _RecordingQPainter.last.texts, "the group's dimension text must still be drawn"
