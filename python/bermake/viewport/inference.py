"""Acquisition and inference locking: the stateful half of the snap path.

`SnapEngine` is stateless by design. Hover-dwell acquisition needs a clock and
locking needs memory across frames, so both live here, behind an injected
`now_ms` so dwell is a value in tests and never a wall-clock timer.

`ViewportWidget._snap_for_event` is the single choke point every tool already
flows through, so wiring this in there leaves all twenty tools untouched: they
keep receiving exactly one `SnapResult`.

`AcquiredKind` and `Acquired` are defined in `snap_types.py`, not here, and
re-exported below: Task 4's candidate generators in `snap_candidates.py` need
`AcquiredKind` and must never import this (stateful) module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from bermake.viewport.snap_types import Acquired, AcquiredKind, SnapKind, SnapResult

__all__ = [
    "DWELL_MS",
    "MOVE_TOLERANCE_PX",
    "Acquired",
    "AcquiredKind",
    "InferenceState",
    "Lock",
]

DWELL_MS = 250.0
"""Milliseconds the cursor must rest on a candidate before it is acquired."""

MOVE_TOLERANCE_PX = 8.0
"""Cursor travel that restarts the dwell. Matches SnapEngine.PIXEL_TOLERANCE."""

_AXIS_DIRS = {
    0: np.array([1.0, 0.0, 0.0], dtype=np.float64),
    1: np.array([0.0, 1.0, 0.0], dtype=np.float64),
    2: np.array([0.0, 0.0, 1.0], dtype=np.float64),
}
_AXIS_NAMES = {0: "Red", 1: "Green", 2: "Blue"}

_ACQUIRABLE_VERTEX = (SnapKind.ENDPOINT,)
_ACQUIRABLE_EDGE = (SnapKind.MIDPOINT, SnapKind.ON_EDGE)


@dataclass(frozen=True, slots=True)
class Lock:
    """A pinned inference line. Wins outright, bypassing precedence.

    `origin` None means "resolve against the gesture anchor at apply time"
    (see `InferenceState.apply_lock`). The arrow-key locks use it: spec 2.3
    describes them as direction constraints on the gesture being drawn, so
    the line they pin has to run through the point the gesture started from,
    and that point is not known when the arrow is pressed. A concrete origin
    is stored only where the line is already fixed in space, which today
    means the Shift lock: the inference it pins is on screen at that instant,
    and the snapped point under the cursor lies on it.
    """

    origin: np.ndarray | None
    direction: np.ndarray
    axis: int | None
    label: str
    # "axis" | "edge" | "shift": which control armed this lock. A lock's origin
    # decides who is allowed to release it, so toggling one control never gets
    # mistaken for releasing a lock a different control armed.
    source: str


class InferenceState:
    def __init__(self, now_ms=None) -> None:
        self._now = now_ms if now_ms is not None else _default_now_ms
        self._acquired: Acquired | None = None
        self._lock: Lock | None = None
        self._pending_key: tuple[int, int] | None = None  # (kind marker, entity id)
        self._pending_since: float = 0.0
        self._pending_px: tuple[float, float] = (0.0, 0.0)
        self._pending_snap = None

    @property
    def acquired(self) -> Acquired | None:
        return self._acquired

    @property
    def lock(self) -> Lock | None:
        return self._lock

    def observe(self, snap, cursor_px, edge_direction=None) -> None:
        """Feed one frame's snap and cursor position into the dwell timer.

        A frame whose snap is not acquirable drops the PENDING dwell but
        keeps any reference already acquired. Spec 2.3 reads "moving off it
        by more than PIXEL_TOLERANCE releases it", but taken literally that
        cancels the feature the acquisition exists for: the cursor is off the
        reference by definition while the user draws away from it, so
        Parallel, Perpendicular and From-Point would only ever appear while
        hovering the very entity they infer from. Spec 2.3's own next
        sentence, and `from_point_candidates`' "aligning to geometry the line
        never touches", both describe the opposite. An acquisition is instead
        released when it stops being meaningful: when the gesture that used
        it ends, and when the active context changes underneath it (both
        driven from `ViewportWidget`; see `end_gesture` / `clear_acquisition`).
        """
        key = _acquirable_key(snap)
        if key is None:
            self._pending_key = None
            self._pending_snap = None
            return

        moved = (
            self._pending_key != key
            or float(
                np.hypot(cursor_px[0] - self._pending_px[0], cursor_px[1] - self._pending_px[1])
            )
            > MOVE_TOLERANCE_PX
        )
        if moved:
            self._pending_key = key
            self._pending_since = float(self._now())
            self._pending_px = (float(cursor_px[0]), float(cursor_px[1]))
            self._pending_snap = snap
            return

        self._pending_snap = snap
        if float(self._now()) - self._pending_since >= DWELL_MS:
            self._acquired = _acquire(snap, edge_direction)

    def clear_acquisition(self) -> None:
        """Forget the acquired reference.

        Called by `ViewportWidget.on_active_context_changed` when the user
        enters or leaves a group: `Acquired.position` is a world point and
        `Acquired.entity_id` a per-context id, so both go stale the moment
        the context under them changes, and the id can name a different
        entity entirely in the new one.
        """
        self._acquired = None
        self._pending_key = None
        self._pending_snap = None

    def toggle_axis_lock(self, axis: int) -> None:
        """Pin the inference to one world axis through the gesture anchor.

        `origin=None` defers that anchor to `apply_lock`, which is the only
        place it is known. Storing the acquired reference's position here
        instead (the pre-fix behaviour) produced a From-Point line rather
        than an axis lock, and storing the world origin produced a line that
        was axis-aligned with respect to nothing the user had drawn.
        """
        if self._lock is not None and self._lock.source == "axis" and self._lock.axis == axis:
            self._lock = None
            return
        self._lock = Lock(
            origin=None,
            direction=_AXIS_DIRS[axis].copy(),
            axis=axis,
            label=f"on {_AXIS_NAMES[axis]} Axis",
            source="axis",
        )

    def toggle_edge_lock(self) -> None:
        """Lock to the acquired edge's direction. A no-op with no acquired edge.

        Anchor-relative for the same reason `toggle_axis_lock` is, and with
        one more: this lock is the sticky form of the PARALLEL inference and
        wears its label, and `snap_candidates.directional_candidates` draws
        PARALLEL through the gesture anchor. A lock that said "Parallel to
        Edge" while resolving to a different line than the inference of that
        name would be a straightforward lie.
        """
        if self._lock is not None and self._lock.source == "edge":
            self._lock = None
            return
        if self._acquired is None or self._acquired.direction is None:
            return
        self._lock = Lock(
            origin=None,
            direction=self._acquired.direction.astype(np.float64),
            axis=None,
            label="Parallel to Edge",
            source="edge",
        )

    def set_shift_lock(self, held: bool, snap) -> None:
        """Hold the inference currently showing. Releasing Shift clears it.

        Only a lock this same method armed (`source == "shift"`) is released
        here: MainWindow.eventFilter is installed application-wide, so any
        Shift release anywhere -- a panel, a dialog, an unrelated Shift-click --
        must not discard a lock some other control (an arrow key) armed.
        """
        if not held:
            if self._lock is not None and self._lock.source == "shift":
                self._lock = None
            return
        if self._lock is not None or snap is None:
            return
        direction = _direction_of(snap)
        if direction is None:
            return
        self._lock = Lock(
            origin=np.asarray(snap.world_position, dtype=np.float64).reshape(3),
            direction=direction,
            axis=snap.axis,
            label=snap.label,
            source="shift",
        )

    def release_lock(self) -> None:
        self._lock = None

    def end_gesture(self) -> None:
        """Release both the lock and the acquisition. Spec 2.3.

        Driven from `ViewportWidget._snap_for_event`, which watches the
        active tool's `has_active_gesture` for a True -> False edge. That is
        the one hook that catches every way a gesture can finish -- a commit
        click, a double-click, Enter, Escape, a typed VCB value -- without
        each of them having to remember to call this.
        """
        self._lock = None
        self.clear_acquisition()

    def lock_origin(self, anchor=None):
        """Where the active lock's line passes through, in world space.

        A lock with a stored origin uses it. A lock with `origin=None` (the
        arrow-key locks) runs through the gesture anchor, falling back to the
        acquired reference and then to the world origin when no gesture is
        live -- with nothing drawn yet there is no anchor to constrain, and
        an armed-but-anchorless lock should still preview something the user
        can recognise rather than refuse to resolve.
        """
        if self._lock is None:
            return None
        if self._lock.origin is not None:
            return self._lock.origin
        if anchor is not None:
            return np.asarray(anchor, dtype=np.float64).reshape(3)
        if self._acquired is not None:
            return np.asarray(self._acquired.position, dtype=np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    def apply_lock(self, snap, camera, viewport_size, cursor_px, anchor=None):
        """Override `snap` with the locked line's nearest point to the cursor ray.

        `anchor` is the active tool's gesture anchor (world space), which is
        what an arrow-key lock's line runs through; see `lock_origin`.

        Returns `snap` unchanged when no lock is active, so the no-lock path
        stays byte-identical to the pre-M7.6b behaviour.
        """
        if self._lock is None or camera is None or snap is None:
            return snap
        from bermake.geometry.ray import closest_points_two_lines

        width, height = int(viewport_size[0]), int(viewport_size[1])
        ray_origin, ray_dir = camera.ray_from_screen(
            float(cursor_px[0]), float(cursor_px[1]), width, height
        )
        _, _, _c_ray, c_line = closest_points_two_lines(
            ray_origin, ray_dir, self.lock_origin(anchor), self._lock.direction
        )
        return SnapResult(
            kind=SnapKind.AXIS_LOCK,
            world_position=np.asarray(c_line, dtype=np.float32),
            axis=self._lock.axis,
            vertex_id=None,
            label=self._lock.label,
            edge_id=None,
            face_id=snap.face_id,
            edge_t=None,
            direction=np.asarray(self._lock.direction, dtype=np.float64).reshape(3),
        )


def _default_now_ms() -> float:
    return time.monotonic() * 1000.0


def _acquirable_key(snap):
    if snap is None:
        return None
    if snap.kind in _ACQUIRABLE_VERTEX and snap.vertex_id is not None:
        return (int(AcquiredKind.VERTEX), int(snap.vertex_id))
    if snap.kind in _ACQUIRABLE_EDGE and snap.edge_id is not None:
        return (int(AcquiredKind.EDGE), int(snap.edge_id))
    return None


def _acquire(snap, edge_direction=None) -> Acquired:
    if snap.kind in _ACQUIRABLE_VERTEX:
        return Acquired(
            kind=AcquiredKind.VERTEX,
            position=np.asarray(snap.world_position, dtype=np.float64).reshape(3),
            direction=None,
            entity_id=int(snap.vertex_id),
        )
    direction = None
    if edge_direction is not None:
        d = np.asarray(edge_direction, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        direction = d / n if n >= 1e-12 else None
    return Acquired(
        kind=AcquiredKind.EDGE,
        position=np.asarray(snap.world_position, dtype=np.float64).reshape(3),
        direction=direction,
        entity_id=int(snap.edge_id),
    )


_POINT_LIKE_KINDS = (
    SnapKind.NONE,
    SnapKind.GRID,
    SnapKind.MIDPOINT,
    SnapKind.ENDPOINT,
    SnapKind.ON_FACE,
    SnapKind.ON_EDGE,
    SnapKind.INTERSECTION,
    SnapKind.GUIDE_POINT,
)
"""Kinds that name a POINT, not a direction, so Shift cannot lock them.

Spec 2.3 says Shift holds "whatever inference is currently showing", but an
endpoint, a midpoint, an intersection, a guide point, a point on a face or
on an edge, and the grid fallback are all answers to "where", not to "which
way". There is no line to hold. Listed explicitly so a kind added later has
to be classified rather than silently defaulting to unlockable, which is how
PARALLEL, PERPENDICULAR and ON_GUIDE went unlockable for a whole milestone.

The complement -- AXIS_LOCK, FROM_POINT, PARALLEL, PERPENDICULAR, ON_GUIDE --
are the line kinds, and each carries its own `SnapResult.direction`.
"""


def _direction_of(snap):
    """The world direction a snap result implies, or None if it implies none.

    Read straight off the snap. Every candidate that sits on an inference
    LINE records that line's unit direction when it is generated
    (`snap_candidates._line_candidate` / `axis_candidates`), which is the
    only place the direction is known for all five line kinds at once: a
    guide's direction is nowhere in the snap's geometry, and Perpendicular's
    depends on the drawing plane, which `InferenceState` never sees.
    """
    if snap.direction is not None:
        d = np.asarray(snap.direction, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else None
    if snap.kind in _POINT_LIKE_KINDS:
        return None
    # A line kind that reached here was generated without recording its
    # direction. Fall back to the axis it names rather than dropping the lock.
    if snap.axis is not None:
        return _AXIS_DIRS[snap.axis].copy()
    return None
