"""AutosaveScheduler: due, then idle, then forced (M7.12, spec 4.4)."""

import pytest
from bermake.recovery.scheduler import (
    DEFAULT_INTERVAL,
    FORCE_AFTER_SECONDS,
    IDLE_SECONDS,
    INTERVAL_CHOICES,
    AutosaveScheduler,
)

MINUTE = 60.0
SAFE = {"button_held": False, "tool_busy": False, "modal_open": False}


class FakeClock:
    """A list holding the current time, advanced by the test."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = [start]

    def __call__(self) -> float:
        return self.now[0]

    def advance(self, seconds: float) -> None:
        self.now[0] += seconds


def _make(minutes: int = 5):
    clock = FakeClock()
    return AutosaveScheduler(minutes, clock=clock), clock


def _idle(sched, idle_for=IDLE_SECONDS, **overrides):
    return sched.should_save(idle_for=idle_for, **{**SAFE, **overrides})


def test_constants():
    assert INTERVAL_CHOICES == (0, 1, 5, 10, 30)
    assert DEFAULT_INTERVAL == 5
    assert IDLE_SECONDS == 2.0
    assert FORCE_AFTER_SECONDS == 60.0


def test_nothing_while_unchanged():
    sched, clock = _make(5)
    clock.advance(10 * MINUTE)
    assert sched.changed_since_autosave is False
    assert _idle(sched, idle_for=100) is False


def test_nothing_while_off_even_with_changes():
    sched, clock = _make(0)
    sched.note_change()
    clock.advance(10 * 60 * MINUTE)
    assert sched.interval_minutes == 0
    assert _idle(sched, idle_for=100) is False


def test_nothing_before_the_interval():
    sched, clock = _make(5)
    sched.note_change()
    clock.advance(5 * MINUTE - 1)
    assert _idle(sched, idle_for=100) is False
    clock.advance(1)
    assert _idle(sched, idle_for=100) is True


def test_due_but_not_idle_then_idle():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE)
    assert _idle(sched, idle_for=IDLE_SECONDS - 0.1) is False
    assert _idle(sched, idle_for=IDLE_SECONDS) is True


def test_forced_at_exactly_sixty_seconds_overdue():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE)  # becomes due now
    assert _idle(sched, idle_for=0) is False
    clock.advance(FORCE_AFTER_SECONDS - 0.5)
    assert _idle(sched, idle_for=0) is False
    clock.advance(0.5)
    assert _idle(sched, idle_for=0) is True


@pytest.mark.parametrize("blocker", ["button_held", "tool_busy", "modal_open"])
def test_forced_save_still_waits_for_a_safe_moment(blocker):
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE + FORCE_AFTER_SECONDS + 1)
    assert _idle(sched, idle_for=0, **{blocker: True}) is False
    # Idle does not override a blocker either.
    assert _idle(sched, idle_for=100, **{blocker: True}) is False
    assert _idle(sched, idle_for=0) is True


def test_note_saved_clears_flag_and_restarts_countdown():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE)
    assert _idle(sched) is True
    sched.note_saved()
    assert sched.changed_since_autosave is False
    sched.note_change()
    clock.advance(MINUTE - 1)
    assert _idle(sched) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_note_failed_keeps_flag_and_retries_one_interval_later():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE)
    assert _idle(sched) is True
    sched.note_failed()
    assert sched.changed_since_autosave is True
    assert _idle(sched, idle_for=100) is False
    clock.advance(MINUTE - 1)
    assert _idle(sched, idle_for=100) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_failed_save_force_window_restarts_too():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE + 100)
    sched.note_failed()
    clock.advance(MINUTE)  # due again, just now
    assert _idle(sched, idle_for=0) is False  # not yet forced
    clock.advance(FORCE_AFTER_SECONDS)
    assert _idle(sched, idle_for=0) is True


def test_set_interval_restarts_countdown():
    sched, clock = _make(5)
    sched.note_change()
    clock.advance(4 * MINUTE)
    sched.set_interval(1)
    assert sched.interval_minutes == 1
    clock.advance(MINUTE - 1)
    assert _idle(sched) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_set_interval_off_then_on_restarts_countdown():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(10 * MINUTE)
    sched.set_interval(0)
    assert _idle(sched, idle_for=100) is False
    sched.set_interval(1)
    assert _idle(sched, idle_for=100) is False
    clock.advance(MINUTE)
    assert _idle(sched) is True


def test_unknown_interval_is_stored_as_given():
    sched, clock = _make(5)
    sched.set_interval(7)
    assert sched.interval_minutes == 7
    sched.note_change()
    clock.advance(7 * MINUTE - 1)
    assert _idle(sched) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_change_after_a_long_quiet_spell_is_not_instantly_forced():
    # Idle for an hour, then the first edit: due at once, but the force clock
    # starts at the edit, so the user is not interrupted mid-stroke.
    sched, clock = _make(5)
    clock.advance(60 * MINUTE)
    sched.note_change()
    assert _idle(sched, idle_for=0) is False
    assert _idle(sched, idle_for=IDLE_SECONDS) is True
    clock.advance(FORCE_AFTER_SECONDS)
    assert _idle(sched, idle_for=0) is True


def test_wake_from_three_hours_of_sleep_gives_one_save_not_a_burst():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(3 * 60 * MINUTE)  # one step: the machine slept
    saves = 0
    # First idle tick after wake saves once.
    if _idle(sched):
        saves += 1
        sched.note_saved()
    assert saves == 1
    # Ticks keep arriving, but with no change nothing is ever due again.
    for _ in range(200):
        clock.advance(1)
        assert _idle(sched, idle_for=100) is False


def test_after_a_wake_save_the_next_save_needs_a_change_and_a_full_interval():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(3 * 60 * MINUTE)
    assert _idle(sched) is True
    sched.note_saved()
    sched.note_change()
    for _ in range(int(MINUTE) - 1):
        clock.advance(1)
        assert _idle(sched, idle_for=100) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_wake_with_failed_save_retries_one_interval_later_not_in_a_burst():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(3 * 60 * MINUTE)
    assert _idle(sched) is True
    sched.note_failed()
    for _ in range(int(MINUTE) - 1):
        clock.advance(1)
        assert _idle(sched, idle_for=100) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_clock_going_backwards_never_saves_early_or_raises():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(-500)
    assert _idle(sched, idle_for=100) is False
    clock.advance(MINUTE - 1)
    assert _idle(sched, idle_for=100) is False
    clock.advance(1)
    assert _idle(sched) is True


def test_note_change_is_idempotent_for_the_flag():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE - 1)
    sched.note_change()  # more edits do not push the due point back
    clock.advance(1)
    assert _idle(sched) is True


def test_continuous_editing_does_not_postpone_the_forced_save():
    sched, clock = _make(1)
    sched.note_change()
    clock.advance(MINUTE)  # due now
    for _ in range(5):
        clock.advance(10)
        sched.note_change()
        assert _idle(sched, idle_for=0) is False
    clock.advance(10)  # 60 s after becoming due
    sched.note_change()
    assert _idle(sched, idle_for=0) is True
