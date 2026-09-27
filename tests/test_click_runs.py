"""Tests for ClickRuns, the multi-click run counter.

Qt sends no triple-click event, so the third click is counted here. Every
test feeds explicit timestamps: nothing in this file sleeps or reads a clock.
"""

from __future__ import annotations


def _runs(interval_ms=400.0, move_tolerance_px=4.0):
    from bermake.viewport.click_runs import ClickRuns

    return ClickRuns(interval_ms=interval_ms, move_tolerance_px=move_tolerance_px)


def test_a_single_press_is_run_one():
    runs = _runs()
    assert runs.press(100.0, 100.0, 0.0) == 1


def test_two_quick_presses_in_place_are_run_two():
    runs = _runs()
    runs.press(100.0, 100.0, 0.0)
    assert runs.press(100.0, 100.0, 120.0) == 2


def test_three_quick_presses_in_place_are_run_three():
    runs = _runs()
    runs.press(100.0, 100.0, 0.0)
    runs.press(100.0, 100.0, 120.0)
    assert runs.press(100.0, 100.0, 240.0) == 3


def test_a_fourth_quick_press_restarts_at_one():
    """Saturating rather than continuing means a fourth rapid click starts a
    fresh gesture instead of producing a run nothing handles."""
    runs = _runs()
    runs.press(100.0, 100.0, 0.0)
    runs.press(100.0, 100.0, 120.0)
    runs.press(100.0, 100.0, 240.0)
    assert runs.press(100.0, 100.0, 360.0) == 1


def test_a_slow_second_press_starts_a_new_run():
    runs = _runs(interval_ms=400.0)
    runs.press(100.0, 100.0, 0.0)
    assert runs.press(100.0, 100.0, 401.0) == 1


def test_a_press_exactly_on_the_interval_still_continues_the_run():
    runs = _runs(interval_ms=400.0)
    runs.press(100.0, 100.0, 0.0)
    assert runs.press(100.0, 100.0, 400.0) == 2


def test_a_quick_press_far_away_starts_a_new_run():
    """Without the distance test, two deliberate clicks on opposite corners
    inside the interval would read as a double-click on the second corner."""
    runs = _runs(move_tolerance_px=4.0)
    runs.press(100.0, 100.0, 0.0)
    assert runs.press(400.0, 400.0, 120.0) == 1


def test_a_quick_press_within_the_move_tolerance_continues_the_run():
    runs = _runs(move_tolerance_px=4.0)
    runs.press(100.0, 100.0, 0.0)
    assert runs.press(102.0, 102.0, 120.0) == 2


def test_the_interval_is_measured_from_the_previous_press_not_the_run_start():
    """A slow but steady triple-click, each press inside the interval of the
    one before it, is still a triple-click."""
    runs = _runs(interval_ms=400.0)
    runs.press(100.0, 100.0, 0.0)
    runs.press(100.0, 100.0, 390.0)
    assert runs.press(100.0, 100.0, 780.0) == 3


def test_reset_drops_the_run_so_the_next_press_is_run_one():
    runs = _runs()
    runs.press(100.0, 100.0, 0.0)
    runs.press(100.0, 100.0, 120.0)
    runs.reset()
    assert runs.press(100.0, 100.0, 240.0) == 1


def test_the_module_imports_no_qt():
    """The counter is a value, not a widget. Keeping Qt out is what lets the
    whole triple-click gesture be tested without a QApplication."""
    import bermake.viewport.click_runs as mod

    source = open(mod.__file__, encoding="utf-8").read()
    assert "PySide6" not in source
    assert "QtCore" not in source
