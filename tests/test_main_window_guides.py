"""MainWindow integration for M7.6b Task 7: Guide creation and UI.

Modelled on tests/test_main_window_annotations.py: builds a real MainWindow
and drives it through the public `_on_*` handlers / actions rather than
poking tool internals.
"""

from __future__ import annotations

from bermake.model.annotation import Dimension, Guide, GuidePoint, Label
from bermake.ui.main_window import MainWindow


def _seed_annotations(w):
    ctx = w._model.active_context
    dim = Dimension(
        w._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    label = Label(w._model.new_annotation_id(), (1.0, 1.0, 0.0), (1.0, 2.0, 0.0), "note")
    guide = Guide(w._model.new_annotation_id(), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    point = GuidePoint(w._model.new_annotation_id(), (2.0, 2.0, 2.0))
    ctx.annotations.extend([dim, label, guide, point])
    return dim, label, guide, point


def _kinds(ctx):
    return {a.kind for a in ctx.annotations}


# ---------------------------------------------------------------------------
# view_guides / edit_delete_guides actions are declared and wired.
# ---------------------------------------------------------------------------


def test_view_guides_and_delete_guides_actions_registered(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert "view_guides" in w._actions
    assert "edit_delete_guides" in w._actions
    assert w._actions["view_guides"].isCheckable()


def test_view_guides_starts_checked(qtbot):
    """Guides are visible by default, matching ViewportWidget.show_guides."""
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._actions["view_guides"].isChecked()
    assert w._viewport.show_guides is True


# ---------------------------------------------------------------------------
# Delete Guides: removes every guide/guide_point in the active context,
# leaves dimensions and labels alone, as one undoable step.
# ---------------------------------------------------------------------------


def test_delete_guides_removes_guides_leaves_dimensions_and_labels(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    dim, label, guide, point = _seed_annotations(w)
    ctx = w._model.active_context

    w._on_delete_guides()

    remaining = ctx.annotations
    assert dim in remaining
    assert label in remaining
    assert guide not in remaining
    assert point not in remaining
    assert _kinds(ctx) == {"dimension", "label"}


def test_delete_guides_is_a_no_op_with_no_guides(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    dim = Dimension(
        w._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    w._model.active_context.annotations.append(dim)

    w._on_delete_guides()  # must not raise, must not touch the undo stack

    assert w._model.active_context.annotations == [dim]
    assert not w._command_stack.can_undo


def test_delete_guides_undo_restores_all_of_them_in_one_step(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    dim, label, guide, point = _seed_annotations(w)
    ctx = w._model.active_context

    w._on_delete_guides()
    assert _kinds(ctx) == {"dimension", "label"}

    assert w._command_stack.can_undo
    assert w._command_stack.undo()

    remaining = ctx.annotations
    assert dim in remaining
    assert label in remaining
    assert guide in remaining
    assert point in remaining
    assert len(remaining) == 4
    assert not w._command_stack.can_undo


# ---------------------------------------------------------------------------
# View > Guides: toggling paints/hides guides without touching picking or ids.
# ---------------------------------------------------------------------------


def test_toggle_guides_action_flips_viewport_flag(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._viewport.show_guides is True

    w._actions["view_guides"].setChecked(False)
    assert w._viewport.show_guides is False

    w._actions["view_guides"].setChecked(True)
    assert w._viewport.show_guides is True


class _RecordingQPainter:
    """Minimal QPainter stand-in (mirrors test_annotation_render_scope.py's),
    so _paint_annotations can run headlessly."""

    class RenderHint:
        Antialiasing = 1

    last = None

    def __init__(self, _device):
        self.pens = []
        self.lines = []
        self.texts = []
        self.setRenderHint = lambda *a, **k: None
        self.setFont = lambda *a, **k: None
        self.setPen = self._set_pen
        self.drawLine = self._draw_line
        self.drawText = self._draw_text
        type(self).last = self

    def _set_pen(self, pen):
        self.pens.append(pen)

    def _draw_line(self, x1, y1, x2, y2):
        self.lines.append((x1, y1, x2, y2))

    def _draw_text(self, x, y, s):
        self.texts.append((x, y, s))

    def end(self):
        pass


def test_hiding_guides_stops_them_painting_while_dimensions_still_paint(qtbot, monkeypatch):
    from PySide6 import QtGui

    w = MainWindow()
    qtbot.addWidget(w)
    dim, _label, _guide, _point = _seed_annotations(w)

    monkeypatch.setattr(QtGui, "QPainter", _RecordingQPainter)

    w._viewport.show_guides = True
    _RecordingQPainter.last = None
    w._viewport._paint_annotations()
    shown_lines = len(_RecordingQPainter.last.lines)
    shown_texts = list(_RecordingQPainter.last.texts)
    assert shown_texts, "the dimension's measurement text must be drawn"
    assert dim.kind == "dimension"

    w._viewport.show_guides = False
    _RecordingQPainter.last = None
    w._viewport._paint_annotations()
    hidden_lines = len(_RecordingQPainter.last.lines)
    hidden_texts = list(_RecordingQPainter.last.texts)

    assert hidden_texts == shown_texts, "the dimension must keep painting"
    assert hidden_lines < shown_lines, "the guide's own lines must stop painting"


def test_hiding_guides_deselects_them_so_delete_cannot_destroy_the_invisible(qtbot):
    """Task 7 made a hidden guide unpickable and unsnappable, on the grounds
    that a hidden thing the user can still act on is a trap. The selection
    half of that path was left open: select a guide, turn guides off, press
    Delete, and an annotation vanishes with nothing on screen to say which.
    A dimension selected at the same time is untouched -- View > Guides
    hides guides, not everything on the annotation rail.
    """
    w = MainWindow()
    qtbot.addWidget(w)
    dim, _label, guide, point = _seed_annotations(w)
    w._selection.replace(annotations=[dim.id, guide.id, point.id])

    w._actions["view_guides"].setChecked(False)
    w._on_toggle_guides(False)

    assert guide.id not in w._selection.annotations
    assert point.id not in w._selection.annotations
    assert dim.id in w._selection.annotations
    # And nothing was deleted: hiding is not erasing.
    assert {a.id for a in w._model.active_context.annotations} >= {guide.id, point.id}


def test_showing_guides_again_does_not_reselect_them(qtbot):
    """The deselect is not a stash-and-restore. Turning guides back on leaves
    the selection where the user's last action left it."""
    w = MainWindow()
    qtbot.addWidget(w)
    _dim, _label, guide, _point = _seed_annotations(w)
    w._selection.replace(annotations=[guide.id])

    w._on_toggle_guides(False)
    w._on_toggle_guides(True)

    assert guide.id not in w._selection.annotations
