"""Shared helpers for the startup recovery tests (M7.12, #77; Task 6).

`answer_recovery_dialog` and `answer_message_box` follow the timer-click
pattern of tests/test_error_dialog.py `_answer_the_box`: a QTimer clicks from
inside the dialog's own event loop, retrying every 10 ms, and gives up after a
bounded number of attempts so a dialog that never appears fails the test
instead of hanging it or clicking a later test's dialog.

`write_session` writes a recovery session through the real RecoveryStore, the
way a crashed Bermake would have left it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import bermake
from bermake.document import DocumentSettings
from bermake.io import save_document
from bermake.model import Model
from bermake.recovery.store import RecoveryStore, SessionMeta
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import RenderStyle
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QListWidget, QMessageBox, QPushButton

DEAD_PID = 999_999
DEAD_STARTED_AT = "2026-10-04T13:58:41+08:00"


def write_session(
    folder: Path,
    model: Model | None = None,
    *,
    session_id: str | None = None,
    original_path: str | None = "C:/drawings/House.berm",
    display_name: str = "House.berm",
    saved_at: str | None = None,
    pid: int = DEAD_PID,
    process_started_at: str = DEAD_STARTED_AT,
):
    """Write one session (`.berm` then `.json`) and return its RecoverySession
    fields as (session_id, berm_path, meta_path)."""
    session_id = session_id or uuid.uuid4().hex
    meta = SessionMeta(
        session_id=session_id,
        original_path=original_path,
        display_name=display_name,
        saved_at=saved_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        app_version=bermake.__version__,
        pid=pid,
        process_started_at=process_started_at,
    )
    store = RecoveryStore(folder)
    store.write(
        meta,
        lambda target: save_document(
            target, model or Model(), Camera(), DocumentSettings(), RenderStyle()
        ),
    )
    return session_id, folder / f"{session_id}.berm", folder / f"{session_id}.json"


def _visible(kind) -> list:
    return [w for w in QApplication.topLevelWidgets() if isinstance(w, kind) and w.isVisible()]


def _bounded(find, act, attempts: int):
    """Call act(widget) once find() returns one; retry every 10 ms up to
    `attempts` times, then mark the returned record spent."""
    state = SimpleNamespace(spent=False)
    remaining = [attempts]

    def attempt():
        found = find()
        if not found:
            remaining[0] -= 1
            if remaining[0] > 0:
                QTimer.singleShot(10, attempt)
            else:
                state.spent = True
            return
        act(found[0])

    QTimer.singleShot(0, attempt)
    return state


def answer_recovery_dialog(button_text, seen, *, select_row=None, attempts=500):
    """Click `button_text` on the open RecoveryDialog, first selecting
    `select_row` in its list if given. Records the dialog's text, buttons,
    rows, selected row and parent in `seen`. A missing button is recorded and
    the dialog rejected, so the test fails on an assertion, not in a hang."""
    from bermake.ui.recovery_dialog import RecoveryDialog

    def act(dialog):
        message = dialog.findChild(QLabel, "RecoveryMessage")
        rows = dialog.findChild(QListWidget, "RecoverySessions")
        # Left to right, as the tester sees them.
        buttons = sorted(
            (b for b in dialog.findChildren(QPushButton) if b.isVisible()),
            key=lambda b: b.mapTo(dialog, b.rect().topLeft()).x(),
        )
        labels = [b.text().replace("&", "") for b in buttons]
        record = {
            "text": message.text() if message is not None else None,
            "buttons": labels,
            "rows": [rows.item(i).text() for i in range(rows.count())] if rows else None,
            "selected": rows.currentRow() if rows else None,
            "parent": dialog.parent(),
            "modal": dialog.isModal(),
        }
        seen.append(record)
        if select_row is not None and rows is not None:
            rows.setCurrentRow(select_row)
        if button_text not in labels:
            record["missing_button"] = button_text
            dialog.reject()
            return
        next(b for b in buttons if b.text().replace("&", "") == button_text).click()

    return _bounded(lambda: _visible(RecoveryDialog), act, attempts)


def answer_message_box(button_text, seen, *, attempts=500):
    """Click `button_text` on the open QMessageBox, recording its title, text
    and buttons; rejects it if the button is missing."""

    def act(box):
        labels = [b.text().replace("&", "") for b in box.buttons()]
        seen.append({"title": box.windowTitle(), "text": box.text(), "buttons": labels})
        if button_text not in labels:
            seen[-1]["missing_button"] = button_text
            box.reject()
            return
        next(b for b in box.buttons() if b.text().replace("&", "") == button_text).click()

    return _bounded(lambda: _visible(QMessageBox), act, attempts)
