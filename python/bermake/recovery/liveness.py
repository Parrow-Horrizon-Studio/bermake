"""Is the Bermake that wrote a recovery session still running?

A PID alone is not enough, because the OS reuses PIDs. A session therefore
records the writer's process creation time, and `process_is_running` is True
only when a process with that PID exists *and* was created within
`START_TOLERANCE_SECONDS` of the recorded time. When liveness cannot be
determined the answer is True: an unknown writer counts as running, so a live
session is never offered for recovery (spec 4.5).

Windows uses `ctypes` (no extra dependency); elsewhere `os.kill(pid, 0)` plus
`/proc/<pid>/stat` when it exists.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime, timedelta

_log = logging.getLogger(__name__)

START_TOLERANCE_SECONDS = 2.0

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=UTC)

_IMPORTED_AT = datetime.now().astimezone().isoformat(timespec="seconds")


def _iso(moment: datetime) -> str:
    return moment.astimezone().isoformat(timespec="seconds")


def _windows_query(pid: int) -> datetime | None:
    """Creation time of a running process, or None if no such running process.

    Raises OSError if the process cannot be queried for another reason."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 5:  # ERROR_ACCESS_DENIED: it exists, we may not look at it
            raise OSError(error, "access denied querying process")
        return None
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if exit_code.value != _STILL_ACTIVE:
            return None
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        ticks = (created.dwHighDateTime << 32) | created.dwLowDateTime
        return _FILETIME_EPOCH + timedelta(microseconds=ticks // 10)
    finally:
        kernel32.CloseHandle(handle)


def _proc_start_time(pid: int) -> datetime | None:
    """Start time from /proc/<pid>/stat field 22, or None when /proc is absent."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
        with open("/proc/stat", encoding="utf-8") as f:
            btime = next(int(line.split()[1]) for line in f if line.startswith("btime"))
    except (OSError, StopIteration, ValueError):
        return None
    # The command name (field 2) may contain spaces and parentheses.
    fields_after_comm = stat[stat.rindex(")") + 2 :].split()
    start_ticks = int(fields_after_comm[19])  # field 22 overall
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    return datetime.fromtimestamp(btime + start_ticks / ticks_per_second, tz=UTC)


def _posix_query(pid: int) -> datetime | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass  # exists, owned by someone else
    return _proc_start_time(pid) or _UNKNOWN_START


# Sentinel: the process exists but its start time cannot be read.
_UNKNOWN_START = datetime.min.replace(tzinfo=UTC)


def _query(pid: int) -> datetime | None:
    """Creation time of the running process `pid`, `_UNKNOWN_START` if it runs
    but the time is unreadable, or None if it is not running."""
    if sys.platform == "win32":
        return _windows_query(pid)
    return _posix_query(pid)


def current_process_started_at() -> str:
    """ISO 8601 creation time of this process (import time if it cannot be read)."""
    try:
        created = _query(os.getpid())
        if created is not None and created != _UNKNOWN_START:
            return _iso(created)
    except Exception:
        _log.warning("Could not read this process's creation time", exc_info=True)
    return _IMPORTED_AT


def process_is_running(pid: int, started_at: str) -> bool:
    """True only if `pid` exists and was created within 2 s of `started_at`.

    Returns True whenever that cannot be determined (spec 4.5)."""
    try:
        created = _query(pid)
        if created is None:
            return False
        if created == _UNKNOWN_START:
            return True
        recorded = datetime.fromisoformat(started_at)
        if recorded.tzinfo is None:
            recorded = recorded.astimezone()
        return abs((created - recorded).total_seconds()) <= START_TOLERANCE_SECONDS
    except Exception:
        _log.warning("Could not determine whether process %s is running", pid, exc_info=True)
        return True
