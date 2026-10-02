"""Log files and crash capture (M7.9, spec 2.4). No test touches the real log folder."""

import logging
import subprocess
import sys
import threading

import pytest
from bermake.diagnostics import logs
from PySide6.QtCore import QtMsgType


@pytest.fixture
def scratch_logger():
    """A private logger, so no handler is ever added to the real root logger."""
    logger = logging.getLogger("bermake-test-scratch")
    logger.propagate = False
    yield logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _captured_exception(message="boom"):
    try:
        raise ValueError(message)
    except ValueError:
        return sys.exc_info()


def _records_for(caplog, message):
    return [r for r in caplog.records if r.exc_info and str(r.exc_info[1]) == message]


def test_file_logging_writes_to_the_log_file(tmp_path, scratch_logger):
    handler = logs.configure_file_logging(tmp_path / "logs", scratch_logger)
    scratch_logger.warning("hello from the test")
    handler.flush()

    text = (tmp_path / "logs" / logs.LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "hello from the test" in text


def test_file_logging_rotates_at_one_megabyte_keeping_three(tmp_path, scratch_logger):
    handler = logs.configure_file_logging(tmp_path, scratch_logger)
    assert (handler.maxBytes, handler.backupCount) == (1_000_000, 3)


def test_first_error_notifies_once_and_every_error_is_logged(tmp_path, caplog):
    notified = []
    reporter = logs.ErrorReporter(notified.append, tmp_path)

    with caplog.at_level(logging.CRITICAL, logger="bermake"):
        reporter.excepthook(*_captured_exception())
        reporter.excepthook(*_captured_exception())

    assert notified == [tmp_path]
    assert len(_records_for(caplog, "boom")) == 2


def test_a_failing_dialog_is_logged_not_raised(tmp_path, caplog):
    def broken_dialog(_log_dir):
        raise RuntimeError("the dialog itself failed")

    reporter = logs.ErrorReporter(broken_dialog, tmp_path)
    with caplog.at_level(logging.ERROR, logger="bermake"):
        reporter.excepthook(*_captured_exception())

    assert any("could not show the error dialog" in r.getMessage() for r in caplog.records)


def test_keyboard_interrupt_goes_to_the_default_hook(tmp_path, monkeypatch):
    notified, forwarded = [], []
    monkeypatch.setattr(sys, "__excepthook__", lambda *args: forwarded.append(args[0]))
    reporter = logs.ErrorReporter(notified.append, tmp_path)

    reporter.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert notified == []
    assert forwarded == [KeyboardInterrupt]


def test_thread_errors_are_logged_without_a_dialog(tmp_path, caplog, monkeypatch):
    notified = []
    reporter = logs.ErrorReporter(notified.append, tmp_path)
    monkeypatch.setattr(threading, "excepthook", reporter.threading_excepthook)

    def worker():
        raise ValueError("in a thread")

    with caplog.at_level(logging.CRITICAL, logger="bermake"):
        thread = threading.Thread(target=worker, name="worker")
        thread.start()
        thread.join()

    assert notified == []
    assert len(_records_for(caplog, "in a thread")) == 1


def test_a_startup_failure_is_shown_now_and_counts_as_the_one_dialog(tmp_path, caplog):
    deferred, immediate = [], []
    reporter = logs.ErrorReporter(deferred.append, tmp_path)

    with caplog.at_level(logging.CRITICAL, logger="bermake"):
        reporter.report_startup_failure(*_captured_exception("startup"), immediate.append)
        reporter.excepthook(*_captured_exception())

    assert immediate == [tmp_path]
    assert deferred == []
    assert len(_records_for(caplog, "startup")) == 1


def test_install_routes_both_hooks_to_the_reporter(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    reporter = logs.ErrorReporter(lambda _log_dir: None, tmp_path)

    logs.install_exception_hooks(reporter)

    assert sys.excepthook == reporter.excepthook
    assert threading.excepthook == reporter.threading_excepthook


@pytest.mark.parametrize(
    ("mode", "level"),
    [
        (QtMsgType.QtDebugMsg, logging.DEBUG),
        (QtMsgType.QtInfoMsg, logging.INFO),
        (QtMsgType.QtWarningMsg, logging.WARNING),
        (QtMsgType.QtCriticalMsg, logging.ERROR),
        (QtMsgType.QtFatalMsg, logging.CRITICAL),
    ],
)
def test_qt_messages_land_in_the_log_at_the_matching_level(caplog, mode, level):
    with caplog.at_level(logging.DEBUG, logger="qt"):
        logs.qt_message_handler(mode, None, "shader said no")

    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("qt", level, "shader said no")
    ]


def test_the_session_header_names_every_version():
    header = logs.session_header("9.9.9")
    for part in ("Bermake 9.9.9", "Qt ", "PySide6 ", "Python "):
        assert part in header


def test_a_native_crash_leaves_a_stack_trace(tmp_path):
    """The only way to prove the hard-crash path is to crash a real process."""
    code = (
        "import faulthandler\n"
        "from pathlib import Path\n"
        "from bermake.diagnostics import logs\n"
        f"logs.enable_native_crash_log(Path({str(tmp_path)!r}))\n"
        "faulthandler._sigsegv()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)

    assert result.returncode != 0
    text = (tmp_path / logs.NATIVE_CRASH_FILE_NAME).read_text(encoding="utf-8", errors="replace")
    assert "Fatal Python error" in text or "Windows fatal exception" in text
    assert "<string>" in text


def test_the_native_crash_session_line_names_version_time_and_pid():
    from datetime import UTC, datetime

    line = logs.native_crash_session_line(
        "9.9.9", now=datetime(2026, 9, 28, 12, 30, 5, tzinfo=UTC), pid=4242
    )
    assert line == "--- Bermake 9.9.9 session 2026-09-28T12:30:05+00:00 pid 4242 ---"


def test_a_native_crash_trace_follows_its_session_line(tmp_path):
    """The line is flushed before faulthandler is enabled, so it is on disk
    ahead of the trace even though the process dies without cleanup (M3)."""
    code = (
        "import faulthandler, os\n"
        "from pathlib import Path\n"
        "from bermake.diagnostics import logs\n"
        f"logs.enable_native_crash_log(Path({str(tmp_path)!r}), '9.9.9')\n"
        "print(os.getpid(), flush=True)\n"
        "faulthandler._sigsegv()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)

    assert result.returncode != 0
    pid = result.stdout.decode().strip()
    text = (tmp_path / logs.NATIVE_CRASH_FILE_NAME).read_text(encoding="utf-8", errors="replace")
    first_line = text.splitlines()[0]
    assert first_line.startswith("--- Bermake 9.9.9 session ")
    assert first_line.endswith(f" pid {pid} ---")
    trace = max(text.find("Fatal Python error"), text.find("Windows fatal exception"))
    assert trace > len(first_line)


def test_closing_the_native_crash_log_disables_faulthandler_first(monkeypatch):
    events = []

    class FakeStream:
        def close(self):
            events.append("close")

    monkeypatch.setattr(logs.faulthandler, "disable", lambda: events.append("disable"))

    logs.close_native_crash_log(FakeStream())

    assert events == ["disable", "close"]


def test_the_qt_level_table_is_built_once():
    assert logs._qt_levels() is logs._qt_levels()


def _numbered_lines(count):
    return "".join(f"line {n:05d} of the native crash log\n" for n in range(count))


def test_a_small_native_crash_log_is_left_alone(tmp_path):
    path = tmp_path / "native-crash.log"
    text = _numbered_lines(100)
    path.write_text(text, encoding="utf-8")

    logs.trim_native_crash_log(path)

    assert path.read_text(encoding="utf-8") == text


def test_a_native_crash_log_past_256_kib_keeps_its_last_64_kib_from_a_line_start(tmp_path):
    path = tmp_path / "native-crash.log"
    text = _numbered_lines(8000)  # about 300 KB
    assert len(text) > 256 * 1024
    path.write_text(text, encoding="utf-8")

    logs.trim_native_crash_log(path)

    kept = path.read_text(encoding="utf-8")
    assert 60 * 1024 < len(kept) <= 64 * 1024
    assert text.endswith(kept)
    assert kept.startswith("line ")
    assert kept.endswith("\n")


def test_a_native_crash_log_exactly_at_the_bound_is_not_trimmed(tmp_path):
    path = tmp_path / "native-crash.log"
    path.write_bytes(b"x" * logs.NATIVE_CRASH_MAX_BYTES)

    logs.trim_native_crash_log(path)

    assert path.stat().st_size == logs.NATIVE_CRASH_MAX_BYTES


def test_starting_a_session_trims_before_the_header_is_appended(tmp_path):
    path = tmp_path / logs.NATIVE_CRASH_FILE_NAME
    path.write_text(_numbered_lines(8000), encoding="utf-8")

    stream = logs.enable_native_crash_log(tmp_path, "9.9.9")
    logs.close_native_crash_log(stream)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("--- Bermake 9.9.9 session ")
    assert path.stat().st_size < 70 * 1024
    assert lines[0].startswith("line ")


def test_a_trim_that_fails_does_not_stop_the_log_being_enabled(tmp_path, monkeypatch):
    path = tmp_path / logs.NATIVE_CRASH_FILE_NAME
    path.write_text(_numbered_lines(8000), encoding="utf-8")

    def refuse(self, data):
        raise PermissionError("held open")

    with monkeypatch.context() as patch:
        patch.setattr(type(path), "write_bytes", refuse)
        stream = logs.enable_native_crash_log(tmp_path, "9.9.9")
        logs.close_native_crash_log(stream)

    assert path.read_text(encoding="utf-8").splitlines()[-1].startswith("--- Bermake 9.9.9")


def test_a_missing_native_crash_log_is_nothing_to_trim_and_says_nothing(tmp_path, caplog):
    """The first launch has no file yet; that must not put a warning and a
    traceback at the top of the first log a tester sends."""
    path = tmp_path / "native-crash.log"

    with caplog.at_level(logging.DEBUG):
        logs.trim_native_crash_log(path)

    assert caplog.records == []
    assert not path.exists()


def test_a_native_crash_log_cut_exactly_on_a_line_start_keeps_that_line(tmp_path):
    path = tmp_path / "native-crash.log"
    line = "x" * 63 + "\n"  # 64 bytes, so the 64 KiB kept is exactly 1024 whole lines
    lines = [f"{n:04d}" + line[4:] for n in range(4200)]
    path.write_bytes("".join(lines).encode("ascii"))

    logs.trim_native_crash_log(path)

    assert path.read_bytes() == "".join(lines[-1024:]).encode("ascii")
