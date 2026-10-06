"""When to autosave: due at the interval, run at a pause, forced after 60 s.

`AutosaveScheduler` is pure policy. It owns no timer and touches no Qt: the
window ticks it once a second and passes in what it observes (how long since
the last input, whether a mouse button is held, whether a tool holds
uncommitted changes, whether a modal is open). The scheduler says yes or no
(spec 4.4).

Time comes from an injected monotonic clock, so a wall-clock change cannot
trigger or suppress a save. A machine that sleeps for hours advances the clock
in one step; because the countdown restarts at every save or failure, that
produces exactly one due autosave, not one per missed interval.
"""

from __future__ import annotations

import time
from collections.abc import Callable

INTERVAL_CHOICES = (0, 1, 5, 10, 30)  # minutes; 0 = Off
DEFAULT_INTERVAL = 5
IDLE_SECONDS = 2.0
FORCE_AFTER_SECONDS = 60.0


class AutosaveScheduler:
    def __init__(
        self,
        interval_minutes: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._interval_minutes = interval_minutes
        self._changed = False
        self._changed_at = 0.0
        self._countdown_start = clock()

    @property
    def interval_minutes(self) -> int:
        return self._interval_minutes

    @property
    def changed_since_autosave(self) -> bool:
        return self._changed

    def set_interval(self, minutes: int) -> None:
        """Store the interval as given (the preference layer validates it)."""
        self._interval_minutes = minutes
        self._restart_countdown()

    def note_change(self) -> None:
        """The document changed. Further changes do not move the due point."""
        if not self._changed:
            self._changed = True
            self._changed_at = self._clock()

    def note_saved(self) -> None:
        """An autosave or a user Save succeeded: nothing is pending."""
        self._changed = False
        self._restart_countdown()

    def note_failed(self) -> None:
        """An autosave failed: keep the flag, retry one interval from now."""
        self._restart_countdown()

    def should_save(
        self,
        *,
        idle_for: float,
        button_held: bool,
        tool_busy: bool,
        modal_open: bool,
    ) -> bool:
        if self._interval_minutes <= 0 or not self._changed:
            return False
        now = self._clock()
        if now < self._countdown_start:
            # A clock that went backwards must neither save early nor stall
            # for the whole gap: start the countdown again from here.
            self._countdown_start = now
        if now < self._changed_at:
            self._changed_at = now
        elapsed_due = self._countdown_start + self._interval_minutes * 60.0
        if now < elapsed_due:
            return False
        if button_held or tool_busy or modal_open:
            return False
        # Due since the later of the interval elapsing and the first change,
        # so a first edit after a long quiet spell is not instantly forced.
        due_since = max(elapsed_due, self._changed_at)
        return idle_for >= IDLE_SECONDS or now - due_since >= FORCE_AFTER_SECONDS

    def _restart_countdown(self) -> None:
        self._countdown_start = self._clock()
