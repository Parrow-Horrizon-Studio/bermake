"""MainWindow integration for M7d annotations (Task 13):

Part A -- DimensionTool ("I") and TextTool ("N") registered like every other
drawing tool, with their own QShortcut and a Tools-menu entry.

Part B -- the viewport's units provider (Task 4 carry-over) is wired to the
document's live units, so `_paint_annotations` formats dimension text in the
document's actual unit setting instead of silently falling back to a default
`Units()`.

Part C -- an annotation-only (or mixed) selection now shows a non-empty
status string naming the annotation count (Task 7 carry-over); previously
`_refresh_selection_status` only knew about edges/faces/instances and went
blank for an annotation-only selection.
"""

from __future__ import annotations

from bermake.tools.dimension_tool import DimensionTool
from bermake.tools.text_tool import TextTool
from bermake.ui.main_window import MainWindow


def test_dimension_tool_registered_with_i(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._tool_manager.activate_by_shortcut("I")
    assert isinstance(w._tool_manager.active, DimensionTool)


def test_text_tool_registered_with_n(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._tool_manager.activate_by_shortcut("N")
    assert isinstance(w._tool_manager.active, TextTool)


def test_i_and_n_key_shortcuts_registered(qtbot):
    # M7.2 Task 10: I/N are no longer raw QShortcuts -- they are the
    # tool_dimension/tool_text actions' registry shortcuts, wired to fire at
    # window scope by build_shortcuts() (added to w.actions()).
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._actions["tool_dimension"].shortcut().toString().upper() == "I"
    assert w._actions["tool_text"].shortcut().toString().upper() == "N"
    assert w._actions["tool_dimension"] in w.actions()
    assert w._actions["tool_text"] in w.actions()


# ---------------------------------------------------------------------------
# Part B (Task 4 carry-over): viewport units provider must be wired to the
# document's live units, not left unset (which would silently default every
# dimension label to `Units()` regardless of the Units menu selection).
# ---------------------------------------------------------------------------


def test_viewport_units_provider_wired_to_doc_units(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._viewport._units_provider is not None
    assert w._viewport._units_provider() == w._doc.units


def test_viewport_units_provider_reflects_metric_change(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._doc.set_metric("mm")
    assert w._viewport._units_provider() == w._doc.units
    assert w._viewport._units_provider().metric_unit == "mm"


def test_viewport_units_provider_reflects_imperial_change(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._doc.set_imperial()
    got = w._viewport._units_provider()
    assert got == w._doc.units
    assert got.system == w._doc.units.system


# ---------------------------------------------------------------------------
# Part C (Task 7 carry-over): Selection.counts() widened to a 4-tuple and
# _refresh_selection_status now reports the annotation count.
# ---------------------------------------------------------------------------


def test_selection_status_blank_when_nothing_selected(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    w._refresh_selection_status()
    assert w._status_bar._selection == ""


def test_selection_status_annotation_only(qtbot):
    from bermake.model.annotation import Dimension

    w = MainWindow()
    qtbot.addWidget(w)
    ann = Dimension(
        w._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    w._model.active_context.annotations.append(ann)
    w._selection.replace(annotations=[ann.id])

    w._refresh_selection_status()

    assert w._status_bar._selection == "1 annotation selected"


def test_selection_status_mixed_lists_all_kinds(qtbot):
    from bermake.model.annotation import Dimension

    w = MainWindow()
    qtbot.addWidget(w)
    ann = Dimension(
        w._model.new_annotation_id(), (0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (0.0, -2.0, 0.0)
    )
    w._model.active_context.annotations.append(ann)
    w._selection.replace(edges=[1], faces=[2], instances=[3], annotations=[ann.id])

    w._refresh_selection_status()

    assert w._status_bar._selection == "1 edge, 1 face, 1 instance, 1 annotation selected"
