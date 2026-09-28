"""Log files and crash capture for testers (M7.9, spec 2.4).

A windowed executable has no console, so without this every print and every
uncaught exception disappears, and a crash looks like the window vanishing.
Everything here writes under the per-user app-data folder and nothing is
uploaded anywhere (D9).

Three channels, because three different things can go wrong:

* Python exceptions reach sys.excepthook (PySide6 routes exceptions raised in
  Qt slots there too) and threading.excepthook.
* Qt's own warnings, including shader compile errors, reach the Qt message
  handler.
* A crash in C++ or the GPU driver kills the process before Python can react,
  so faulthandler writes a stack trace to a separate, permanently open file.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import sys
import threading
from collections.abc import Callable
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType
from typing import TextIO

LOG_FILE_NAME = "bermake.log"
NATIVE_CRASH_FILE_NAME = "native-crash.log"
MAX_LOG_BYTES = 1_000_000
LOG_BACKUP_COUNT = 3
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger("bermake")


def default_log_directory() -> Path:
    """`<AppLocalDataLocation>/logs`.

    With the organisation and application names app.py sets, this resolves to
    `%LOCALAPPDATA%/Parrow Horrizon Studio/Bermake/logs`. The names must be set
    before this is called, or Qt returns a folder without them.
    """
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base) / "logs"


def configure_file_logging(
    log_dir: Path, target: logging.Logger | None = None
) -> RotatingFileHandler:
    """Attach a rotating file handler to `target` (the root logger by default)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / LOG_FILE_NAME,
        maxBytes=MAX_LOG_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FORMAT))
    destination = target if target is not None else logging.getLogger()
    destination.addHandler(handler)
    destination.setLevel(logging.INFO)
    return handler


def native_crash_session_line(version: str, *, now: datetime | None = None, pid: int) -> str:
    """The line that opens each session in `native-crash.log`, so a trace in a
    tester's attachment can be matched to the session that wrote it."""
    stamp = (now or datetime.now().astimezone()).isoformat(timespec="seconds")
    return f"--- Bermake {version or 'unknown version'} session {stamp} pid {pid} ---"


def enable_native_crash_log(log_dir: Path, version: str = "") -> TextIO:
    """Point faulthandler at `native-crash.log`, after a session line.

    The stream is returned so the caller can keep it alive: faulthandler holds
    only the file descriptor, and a closed stream would make the handler write
    nowhere at the one moment it matters. Close it with close_native_crash_log.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stream = (log_dir / NATIVE_CRASH_FILE_NAME).open("a", encoding="utf-8")
    stream.write(native_crash_session_line(version, pid=os.getpid()) + "\n")
    # faulthandler writes to the descriptor directly, past Python's buffer, so
    # the session line must be on disk before any trace can follow it.
    # CPython's faulthandler.enable() also flushes the file it is given; this
    # flush keeps the order from depending on that detail.
    stream.flush()
    faulthandler.enable(file=stream, all_threads=True)
    return stream


def close_native_crash_log(stream: TextIO) -> None:
    """Stop faulthandler before closing its file, never the other way round:
    a crash in between would otherwise write to a closed descriptor, or to
    whatever file reused its number."""
    faulthandler.disable()
    stream.close()


class ErrorReporter:
    """Logs every uncaught exception and tells the user about the first one.

    Only the first per session gets a dialog: an exception inside paintGL
    would otherwise raise one per frame (D8). The application keeps running
    after the dialog, as it does today.
    """

    def __init__(self, notify: Callable[[Path], None], log_dir: Path) -> None:
        self._notify = notify
        self.log_dir = log_dir
        self._notified = False

    def _tell_user(self, show: Callable[[Path], None]) -> None:
        if self._notified:
            return
        self._notified = True
        try:
            show(self.log_dir)
        except Exception:
            logger.exception("could not show the error dialog")

    def excepthook(
        self, exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        self._tell_user(self._notify)

    def threading_excepthook(self, args: threading.ExceptHookArgs) -> None:
        """Log only: a dialog must not be raised from a non-GUI thread."""
        name = args.thread.name if args.thread is not None else "unknown"
        logger.critical(
            "uncaught exception in thread %s",
            name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    def report_startup_failure(
        self,
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
        show_now: Callable[[Path], None],
    ) -> None:
        """For failures before the event loop runs, where a deferred dialog
        would never be shown and the application would silently exit."""
        logger.critical("Bermake failed to start", exc_info=(exc_type, exc, tb))
        self._tell_user(show_now)


def install_exception_hooks(reporter: ErrorReporter) -> None:
    sys.excepthook = reporter.excepthook
    threading.excepthook = reporter.threading_excepthook


def _qt_levels() -> dict[object, int]:
    from PySide6.QtCore import QtMsgType

    return {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }


def qt_message_handler(mode, context, message: str) -> None:
    logging.getLogger("qt").log(_qt_levels().get(mode, logging.WARNING), "%s", message)


def install_qt_message_handler() -> None:
    from PySide6.QtCore import qInstallMessageHandler

    qInstallMessageHandler(qt_message_handler)


def session_header(version: str) -> str:
    """The first line of every session's log: what a bug report needs first."""
    import PySide6
    from PySide6.QtCore import qVersion

    return (
        f"Bermake {version} | Qt {qVersion()} | PySide6 {PySide6.__version__} | "
        f"Python {platform.python_version()} | {platform.platform()}"
    )
