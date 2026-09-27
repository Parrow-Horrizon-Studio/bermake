"""Multi-click run counting.

Qt delivers `MouseButtonPress` and `MouseButtonDblClick` but has no
triple-click event: a third press arrives as an ordinary press. This counts
runs so ViewportWidget can synthesize one.

The clock is a parameter, never read here. That follows M7.6b's inference
dwell, where the only wall-clock call site in the codebase sits at the widget
boundary and the state machine underneath takes `now_ms` as a value, which is
what makes the gesture testable without sleeping.

Note that this deliberately does NOT re-implement double-click detection.
Qt's own `mouseDoubleClickEvent` keeps driving `on_mouse_double_click`; the
run count is consulted only to recognise the third press. Rebuilding the
double-click path on top of this would put entering a group, editing a
label's text and every other existing double-click behaviour at risk for no
gain.
"""

from __future__ import annotations

_MAX_RUN = 3


class ClickRuns:
    """Counts how many presses form the current rapid, stationary run.

    `press` returns 1 for a fresh run, 2 for a double, 3 for a triple, and
    then restarts at 1. A press continues the run when it lands both within
    `interval_ms` of the previous press and within `move_tolerance_px` of it.
    """

    __slots__ = ("_count", "_interval_ms", "_last_ms", "_last_xy", "_move_tolerance_px")

    def __init__(self, interval_ms: float, move_tolerance_px: float = 4.0) -> None:
        self._interval_ms = float(interval_ms)
        self._move_tolerance_px = float(move_tolerance_px)
        self._count = 0
        self._last_ms: float | None = None
        self._last_xy: tuple[float, float] | None = None

    def press(self, x: float, y: float, now_ms: float) -> int:
        x = float(x)
        y = float(y)
        now_ms = float(now_ms)
        if self._continues_run(x, y, now_ms):
            self._count += 1
        else:
            self._count = 1
        if self._count > _MAX_RUN:
            self._count = 1
        self._last_ms = now_ms
        self._last_xy = (x, y)
        return self._count

    def reset(self) -> None:
        """Drop the run. The next press reads as a fresh one."""
        self._count = 0
        self._last_ms = None
        self._last_xy = None

    def _continues_run(self, x: float, y: float, now_ms: float) -> bool:
        if self._last_ms is None or self._last_xy is None or self._count == 0:
            return False
        if now_ms - self._last_ms > self._interval_ms:
            return False
        last_x, last_y = self._last_xy
        # Chebyshev distance rather than Euclidean: the tolerance exists to
        # absorb hand tremor over a few pixels, and the cheaper metric is
        # indistinguishable at that scale.
        return (
            abs(x - last_x) <= self._move_tolerance_px
            and abs(y - last_y) <= self._move_tolerance_px
        )
