"""The deferred error dialog (M7.9). The dialog itself is a modal QMessageBox and
is checked by the manual clean-machine gate; this pins the scheduling around it.

The OpenGL dialog shared by MainWindow and the startup preflight (final review
I1) is driven for real: a timer clicks its buttons from inside its event loop."""

from pathlib import Path

from bermake.diagnostics.error_dialog import deferred_error_dialog, show_gl_fallback_dialog
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox


def test_no_application_means_no_dialog(tmp_path):
    scheduled = []
    deferred_error_dialog(
        tmp_path,
        instance=lambda: None,
        schedule=lambda delay, fn: scheduled.append((delay, fn)),
        show=lambda log_dir: None,
    )
    assert scheduled == []


def test_the_dialog_is_scheduled_on_the_event_loop_not_shown_inline(tmp_path):
    scheduled, shown = [], []
    deferred_error_dialog(
        tmp_path,
        instance=lambda: object(),
        schedule=lambda delay, fn: scheduled.append((delay, fn)),
        show=shown.append,
    )
    assert shown == []
    assert [delay for delay, _fn in scheduled] == [0]

    scheduled[0][1]()
    assert shown == [Path(tmp_path)]


def _answer_the_box(button_text, seen, *, attempts=500):
    """Click `button_text` on the open QMessageBox once it is showing,
    recording its buttons and text; retries every 10 ms until the box exists.

    Bounded so it cannot outlive its test (about 5 s at the default): a helper
    that re-armed forever could click a later test's box. If the box has no
    such button, that is recorded in `seen` and the box is rejected, so the
    test fails on its assertion instead of hanging in the modal loop.
    """
    remaining = [attempts]

    def attempt():
        boxes = [
            w
            for w in QApplication.topLevelWidgets()
            if isinstance(w, QMessageBox) and w.isVisible()
        ]
        if not boxes:
            remaining[0] -= 1
            if remaining[0] > 0:
                QTimer.singleShot(10, attempt)
            return
        box = boxes[0]
        labels = [b.text().replace("&", "") for b in box.buttons()]
        seen.append(
            {
                "buttons": labels,
                "text": box.informativeText(),
                "parent": box.parent(),
            }
        )
        if button_text not in labels:
            seen[-1]["missing_button"] = button_text
            box.reject()
            return
        next(b for b in box.buttons() if b.text().replace("&", "") == button_text).click()

    QTimer.singleShot(0, attempt)


def test_the_helper_closes_a_box_that_lacks_the_button_instead_of_hanging(qapp):
    seen = []
    _answer_the_box("No such button", seen)
    assert show_gl_fallback_dialog("no driver", False) is False
    assert seen[0]["missing_button"] == "No such button"


def test_the_helper_stops_retrying_when_no_box_ever_appears(qapp):
    """Once spent, it must not click a box that a later test opens."""
    seen = []
    _answer_the_box("Close", seen, attempts=3)
    loop = QEventLoop()
    QTimer.singleShot(300, loop.quit)
    loop.exec()

    def close_it():
        for w in QApplication.topLevelWidgets():
            if isinstance(w, QMessageBox) and w.isVisible():
                w.reject()

    QTimer.singleShot(300, close_it)
    show_gl_fallback_dialog("no driver", False)
    assert seen == []


def test_the_gl_dialog_offers_the_restart_and_reports_the_choice(qapp):
    seen = []
    _answer_the_box("Restart using compatibility rendering", seen)
    assert show_gl_fallback_dialog("OpenGL 1.1 only", True) is True
    assert "Restart using compatibility rendering" in seen[0]["buttons"]
    assert seen[0]["text"].startswith("OpenGL 1.1 only")
    assert seen[0]["parent"] is None

    _answer_the_box("Close", seen)
    assert show_gl_fallback_dialog("OpenGL 1.1 only", True) is False


def test_the_gl_dialog_without_an_offer_only_explains(qapp):
    seen = []
    _answer_the_box("Close", seen)
    assert show_gl_fallback_dialog("no driver", False) is False
    assert seen[0]["buttons"] == ["Close"]
    assert "Details are in the log" in seen[0]["text"]
