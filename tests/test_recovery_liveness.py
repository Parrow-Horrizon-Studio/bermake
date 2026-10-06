"""process_is_running: PID plus creation time (M7.12, spec 4.5)."""

import os
import subprocess
import sys
from datetime import datetime, timedelta

import pytest
from bermake.recovery import liveness
from bermake.recovery.liveness import current_process_started_at, process_is_running


def _boom(_pid):
    raise OSError("cannot query")


def test_this_process_is_running():
    assert process_is_running(os.getpid(), current_process_started_at()) is True


def test_started_at_is_iso_with_offset():
    assert datetime.fromisoformat(current_process_started_at()).utcoffset() is not None


def _unused_pid() -> int:
    for pid in range(4_000_000, 4_000_000 + 5000, 4):
        try:
            if liveness._query(pid) is None:
                return pid
        except OSError:
            continue
    pytest.skip("no unused PID found")


def test_nonexistent_pid_is_not_running():
    assert process_is_running(_unused_pid(), current_process_started_at()) is False


def test_reused_pid_with_a_different_start_time_is_not_running():
    started = datetime.fromisoformat(current_process_started_at())
    three_hours_earlier = (started - timedelta(hours=3)).isoformat(timespec="seconds")
    assert process_is_running(os.getpid(), three_hours_earlier) is False


def test_start_time_tolerance_is_two_seconds():
    started = datetime.fromisoformat(current_process_started_at())
    assert process_is_running(os.getpid(), (started + timedelta(seconds=1)).isoformat()) is True
    assert process_is_running(os.getpid(), (started - timedelta(seconds=1)).isoformat()) is True
    assert process_is_running(os.getpid(), (started + timedelta(seconds=5)).isoformat()) is False
    assert process_is_running(os.getpid(), (started - timedelta(seconds=5)).isoformat()) is False


def test_unknown_liveness_counts_as_running(monkeypatch):
    monkeypatch.setattr(liveness, "_query", _boom)
    assert process_is_running(os.getpid(), "2026-10-04T13:58:41+08:00") is True


def test_unparseable_started_at_counts_as_running():
    assert process_is_running(os.getpid(), "not a timestamp") is True


def test_current_process_started_at_falls_back_to_import_time(monkeypatch):
    monkeypatch.setattr(liveness, "_query", _boom)
    assert current_process_started_at() == liveness._IMPORTED_AT


def test_exited_process_is_not_running():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        started = None
        for _ in range(100):
            started = liveness._query(proc.pid)
            if started is not None:
                break
        assert started is not None
        recorded = started.astimezone().isoformat(timespec="seconds")
        assert process_is_running(proc.pid, recorded) is True
    finally:
        proc.kill()
        proc.wait()
    assert process_is_running(proc.pid, recorded) is False
