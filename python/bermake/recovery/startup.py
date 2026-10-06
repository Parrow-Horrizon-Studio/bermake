"""The startup query: which recovery sessions were left by a Bermake that is gone."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from bermake.recovery.liveness import process_is_running
from bermake.recovery.store import RecoverySession, RecoveryStore


def recoverable_sessions(
    store: RecoveryStore,
    is_running: Callable[[int, str], bool] = process_is_running,
) -> list[RecoverySession]:
    """Valid sessions whose writer is not running, newest `saved_at` first.

    Damaged or orphaned files are quarantined by the store and never returned."""
    sessions, _quarantined = store.list_sessions()
    left_behind = [s for s in sessions if not is_running(s.meta.pid, s.meta.process_started_at)]
    # saved_at is ISO 8601 with an offset; compare as instants, not as text.
    return sorted(
        left_behind, key=lambda s: datetime.fromisoformat(s.meta.saved_at).timestamp(), reverse=True
    )
