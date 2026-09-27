"""The bottom bar splits into prompt / coordinates / Measurements (M7.3 Task 15)."""

from __future__ import annotations

import numpy as np
from bermake.ui.status_bar import StatusBar


def _bar(qtbot):
    bar = StatusBar()
    qtbot.addWidget(bar)
    return bar


def test_the_prompt_still_joins_tool_and_snap(qtbot):
    bar = _bar(qtbot)
    bar.set_tool("Line")
    bar.set_snap("Endpoint")
    assert "Line" in bar.prompt_text()
    assert "Endpoint" in bar.prompt_text()


def test_the_prompt_shows_breadcrumb_and_selection_with_no_tool(qtbot):
    bar = _bar(qtbot)
    bar.set_breadcrumb("Model ▸ North Wing")
    bar.set_selection("2 faces selected")
    text = bar.prompt_text()
    assert "North Wing" in text
    assert "2 faces selected" in text


def test_coordinates_are_their_own_field(qtbot):
    bar = _bar(qtbot)
    bar.set_coordinates("X 2 m  Y 1 m  Z 0 m")
    assert bar.coordinates_text() == "X 2 m  Y 1 m  Z 0 m"
    # ...and must not leak into the prompt, which is the whole point of splitting.
    assert "X 2 m" not in bar.prompt_text()


def test_coordinates_clear(qtbot):
    bar = _bar(qtbot)
    bar.set_coordinates("X 1 m")
    bar.set_coordinates("")
    assert bar.coordinates_text() == ""


def test_the_measurements_field_carries_the_status_text(qtbot):
    bar = _bar(qtbot)
    bar.set_status("3600")
    assert "3600" in bar.measurements_text()
    assert "3600" not in bar.prompt_text()


def test_the_prompt_carries_the_message_text(qtbot):
    # Mirrors test_the_measurements_field_carries_the_status_text above, but
    # for set_message -- the setter this whole split exists to make safe.
    # Before the split, a sentence like this one rendered inside the
    # Measurements box because set_status carried both measurements and
    # messages; set_message must land only in the prompt.
    bar = _bar(qtbot)
    bar.set_message("Nothing is hidden here.")
    assert "Nothing is hidden here." in bar.prompt_text()
    assert "Nothing is hidden here." not in bar.measurements_text()


def test_set_status_and_set_message_are_independent(qtbot):
    # Calling both must route each to its own field without either clobbering
    # the other -- the two setters share no state.
    bar = _bar(qtbot)
    bar.set_status("3600")
    bar.set_message("Nothing is hidden here.")
    assert "3600" in bar.measurements_text()
    assert "3600" not in bar.prompt_text()
    assert "Nothing is hidden here." in bar.prompt_text()
    assert "Nothing is hidden here." not in bar.measurements_text()


def test_the_measurements_field_is_labelled(qtbot):
    # It stops being an unlabelled run of text among four others -- that is
    # the discoverability problem this task exists to fix.
    bar = _bar(qtbot)
    assert "Measurements" in bar.measurements_label_text()


def test_the_measurements_field_is_not_focusable(qtbot):
    from PySide6.QtCore import Qt

    # Making it a real editable field risks capturing keystrokes the viewport
    # needs; the visual treatment gives the discoverability without the risk.
    bar = _bar(qtbot)
    assert bar.measurements_widget().focusPolicy() == Qt.FocusPolicy.NoFocus


def test_moving_the_mouse_with_a_tool_armed_fills_the_coordinates(qtbot, main_window):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    main_window._activate("line")
    event = QMouseEvent(
        QMouseEvent.Type.MouseMove,
        QPointF(40.0, 40.0),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    main_window._viewport.mouseMoveEvent(event)

    assert main_window._status_bar.coordinates_text() != ""


def test_disarming_blanks_the_coordinates(qtbot, main_window):
    # Documented limitation: the snap only runs while a tool is active.
    # _on_escape's disarm branch only runs when a tool is active (it returns
    # immediately otherwise), so arm one first -- matching how the coordinate
    # readout gets filled in the first place.
    main_window._activate("line")
    main_window._status_bar.set_coordinates("X 1 m")

    main_window._on_escape()

    assert main_window._status_bar.coordinates_text() == ""


def test_format_coordinates_is_unit_aware(qtbot):
    from bermake.units import Units, UnitSystem, format_coordinates

    text = format_coordinates(
        np.array([2.0, 1.0, 0.0]), Units(system=UnitSystem.METRIC, metric_unit="m")
    )
    assert text.startswith("X ")
    assert "2 m" in text and "1 m" in text and "0 m" in text
