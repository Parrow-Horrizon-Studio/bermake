"""recoverable_sessions: the startup query (M7.12, spec 4.5)."""

import os
import time
from pathlib import Path

from bermake.recovery.liveness import current_process_started_at
from bermake.recovery.startup import recoverable_sessions
from bermake.recovery.store import RecoveryStore, SessionMeta


def _write(
    store: RecoveryStore,
    sid: str,
    saved_at: str,
    pid: int = 1,
    started_at: str = "2026-10-04T13:00:00+08:00",
) -> SessionMeta:
    meta = SessionMeta(
        session_id=sid,
        original_path=None,
        display_name="Untitled",
        saved_at=saved_at,
        app_version="0.17.0",
        pid=pid,
        process_started_at=started_at,
    )

    def write_berm(path: Path) -> None:
        path.write_bytes(b"x")

    store.write(meta, write_berm)
    return meta


def _dead(_pid: int, _started_at: str) -> bool:
    return False


def test_no_sessions(tmp_path):
    assert recoverable_sessions(RecoveryStore(tmp_path), _dead) == []


def test_one_session(tmp_path):
    store = RecoveryStore(tmp_path)
    meta = _write(store, "a" * 32, "2026-10-04T14:00:00+08:00")
    assert [s.meta for s in recoverable_sessions(store, _dead)] == [meta]


def test_three_sessions_newest_first(tmp_path):
    store = RecoveryStore(tmp_path)
    _write(store, "a" * 32, "2026-10-04T14:00:00+08:00")
    _write(store, "b" * 32, "2026-10-04T16:00:00+08:00")
    _write(store, "c" * 32, "2026-10-04T15:00:00+08:00")
    ids = [s.meta.session_id for s in recoverable_sessions(store, _dead)]
    assert ids == ["b" * 32, "c" * 32, "a" * 32]


def test_newest_is_by_instant_not_by_text(tmp_path):
    store = RecoveryStore(tmp_path)
    # Later wall-clock text, but the earlier instant (UTC+8 versus UTC).
    _write(store, "a" * 32, "2026-10-04T20:00:00+08:00")  # 12:00 UTC
    _write(store, "b" * 32, "2026-10-04T13:00:00+00:00")  # 13:00 UTC
    ids = [s.meta.session_id for s in recoverable_sessions(store, _dead)]
    assert ids == ["b" * 32, "a" * 32]


def test_session_whose_writer_is_running_is_skipped(tmp_path):
    store = RecoveryStore(tmp_path)
    _write(store, "a" * 32, "2026-10-04T14:00:00+08:00", pid=111)
    _write(
        store,
        "b" * 32,
        "2026-10-04T15:00:00+08:00",
        pid=222,
        started_at="2026-10-04T13:01:00+08:00",
    )
    asked: list[tuple[int, str]] = []

    def is_running(pid: int, started_at: str) -> bool:
        asked.append((pid, started_at))
        return pid == 222

    result = recoverable_sessions(store, is_running)
    assert [s.meta.session_id for s in result] == ["a" * 32]
    assert (222, "2026-10-04T13:01:00+08:00") in asked


def test_damaged_pair_is_skipped_and_quarantined(tmp_path):
    store = RecoveryStore(tmp_path, clock=lambda: time.time() + 3600)  # lone files are old
    _write(store, "a" * 32, "2026-10-04T14:00:00+08:00")
    (tmp_path / ("b" * 32 + ".json")).write_text("garbage", "utf-8")
    (tmp_path / ("b" * 32 + ".berm")).write_bytes(b"x")
    (tmp_path / ("c" * 32 + ".berm")).write_bytes(b"orphan")

    result = recoverable_sessions(store, _dead)

    assert [s.meta.session_id for s in result] == ["a" * 32]
    assert (tmp_path / ("b" * 32 + ".broken.json")).exists()
    assert (tmp_path / ("c" * 32 + ".broken.berm")).exists()


def test_default_liveness_treats_this_process_as_running(tmp_path):
    store = RecoveryStore(tmp_path)
    _write(
        store,
        "d" * 32,
        "2026-10-04T14:00:00+08:00",
        pid=os.getpid(),
        started_at=current_process_started_at(),
    )
    assert recoverable_sessions(store) == []


def test_an_unconvertible_saved_at_is_quarantined_and_others_are_still_offered(tmp_path):
    """One hand-edited file whose date has no timestamp (year 1) used to make
    the sort raise, so recoverable_sessions offered nothing at all."""
    store = RecoveryStore(tmp_path)
    _write(store, "a" * 32, "2026-10-04T14:00:00+08:00")
    _write(store, "b" * 32, "0001-01-01T00:00:00")
    _write(store, "c" * 32, "2026-10-04T15:00:00+08:00")

    result = recoverable_sessions(store, _dead)

    assert [s.meta.session_id for s in result] == ["c" * 32, "a" * 32]
    assert (tmp_path / ("b" * 32 + ".broken.berm")).exists()
    assert (tmp_path / ("b" * 32 + ".broken.json")).exists()
