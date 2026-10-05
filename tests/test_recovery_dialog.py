"""The startup recovery dialog (M7.12, #77; spec 4.5).

Driven for real: a timer clicks its buttons from inside its own event loop
(tests/_recovery_helpers.py, the tests/test_error_dialog.py pattern)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pytest
from bermake.recovery.store import RecoverySession, SessionMeta
from bermake.ui.recovery_dialog import RecoveryChoice, RecoveryDialog, describe_saved_at
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from tests._recovery_helpers import answer_recovery_dialog

NOW = datetime(2026, 10, 5, 16, 0).astimezone()


@pytest.fixture
def app(qapp):
    return qapp


def _session(name: str, saved_at: datetime, session_id: str | None = None) -> RecoverySession:
    session_id = session_id or name.replace(".", "_")
    meta = SessionMeta(
        session_id=session_id,
        original_path=f"C:/drawings/{name}",
        display_name=name,
        saved_at=saved_at.isoformat(timespec="seconds"),
        app_version="0.17.0",
        pid=999999,
        process_started_at="2026-10-05T09:00:00+08:00",
    )
    return RecoverySession(meta, Path(f"{session_id}.berm"), Path(f"{session_id}.json"))


def _at(hour: int, minute: int, days_ago: int = 0) -> datetime:
    """A local wall-clock time `days_ago` calendar days before 5 Oct 2026.

    Built naive and then made aware, so each day gets its own UTC offset:
    a DST change between the two dates (Sydney, 4 Oct 2026) cannot move the
    clock reading the test expects."""
    day = date(2026, 10, 5) - timedelta(days=days_ago)
    return datetime.combine(day, time(hour, minute, 5)).astimezone()


# --- <when> ------------------------------------------------------------------


def test_when_today():
    assert describe_saved_at(_at(14, 32).isoformat(), NOW) == "today at 14:32"


def test_when_yesterday():
    assert describe_saved_at(_at(14, 32, days_ago=1).isoformat(), NOW) == "yesterday at 14:32"


def test_when_older_date():
    assert describe_saved_at(_at(9, 5, days_ago=1).isoformat(), NOW) == "yesterday at 09:05"
    assert describe_saved_at(_at(14, 32, days_ago=2).isoformat(), NOW) == "3 Oct 2026 at 14:32"
    assert describe_saved_at(_at(14, 32, days_ago=31).isoformat(), NOW) == "4 Sep 2026 at 14:32"


def test_when_is_in_local_time():
    """saved_at carries the writer's offset; the dialog shows the local clock."""
    moment = _at(14, 32)
    elsewhere = moment.astimezone(timezone(moment.utcoffset() + timedelta(hours=3)))
    assert elsewhere.hour != moment.hour
    assert describe_saved_at(elsewhere.isoformat(), NOW) == "today at 14:32"


def test_when_a_future_date_is_shown_as_a_date():
    """A clock set back since the autosave must not say "today" for tomorrow."""
    tomorrow = _at(8, 0, days_ago=-1)
    assert describe_saved_at(tomorrow.isoformat(), NOW) == "6 Oct 2026 at 08:00"


def test_when_across_midnight():
    """Calendar days, not 24-hour spans: 23:59 read at 00:01 is yesterday."""
    just_after_midnight = datetime(2026, 10, 5, 0, 1).astimezone()
    late = datetime(2026, 10, 4, 23, 59).astimezone()
    early = datetime(2026, 10, 5, 0, 0).astimezone()
    assert describe_saved_at(late.isoformat(), just_after_midnight) == "yesterday at 23:59"
    assert describe_saved_at(early.isoformat(), just_after_midnight) == "today at 00:00"


def test_when_never_raises_on_an_unusable_timestamp():
    assert describe_saved_at("not a time", NOW) == "not a time"


# --- One session -------------------------------------------------------------


def test_one_session_text_and_buttons(app):
    seen = []
    dialog = RecoveryDialog([_session("House.berm", _at(14, 32))], now=NOW)
    answer_recovery_dialog("Decide later", seen)
    dialog.exec()

    assert seen[0]["text"] == (
        "Bermake closed unexpectedly. Recover House.berm, last autosaved today at 14:32?"
    )
    assert seen[0]["buttons"] == ["Recover", "Discard", "Decide later"]
    assert seen[0]["rows"] is None
    assert seen[0]["modal"] is True


def test_the_dialog_is_modal_over_its_parent(app, qtbot):
    parent = QWidget()
    qtbot.addWidget(parent)
    seen = []
    dialog = RecoveryDialog([_session("House.berm", _at(14, 32))], parent, now=NOW)
    answer_recovery_dialog("Decide later", seen)
    dialog.exec()
    assert seen[0]["parent"] is parent
    assert seen[0]["modal"] is True


@pytest.mark.parametrize(
    ("button", "action"),
    [("Recover", "recover"), ("Discard", "discard"), ("Decide later", "later")],
)
def test_each_button_maps_to_its_choice_for_one_session(app, button, action):
    session = _session("House.berm", _at(14, 32))
    seen = []
    answer_recovery_dialog(button, seen)
    choice = RecoveryDialog([session], now=NOW).exec()

    assert isinstance(choice, RecoveryChoice)
    assert choice.action == action
    assert choice.session is (None if action == "later" else session)


@pytest.mark.parametrize(
    "dismiss",
    [
        pytest.param(lambda d: QTest.keyClick(d, Qt.Key.Key_Escape), id="escape"),
        pytest.param(lambda d: d.close(), id="close-box"),
    ],
)
def test_escape_or_closing_the_dialog_is_decide_later(app, dismiss):
    """Closing it any other way must keep the work, never discard it."""
    session = _session("House.berm", _at(14, 32))
    dialog = RecoveryDialog([session], now=NOW)

    def recover_if_still_open():
        # If the dismissal did nothing, end the modal loop with a choice the
        # assertion rejects, so the test fails instead of hanging.
        if dialog.isVisible():
            next(b for b in dialog.findChildren(QPushButton) if b.text() == "Recover").click()

    QTimer.singleShot(0, lambda: dismiss(dialog))
    QTimer.singleShot(2000, recover_if_still_open)
    assert dialog.exec() == RecoveryChoice("later", None)


def test_an_untitled_session_is_named_untitled(app):
    seen = []
    session = _session("Untitled", _at(14, 32))
    answer_recovery_dialog("Decide later", seen)
    RecoveryDialog([session], now=NOW).exec()
    assert seen[0]["text"] == (
        "Bermake closed unexpectedly. Recover Untitled, last autosaved today at 14:32?"
    )


def test_a_dialog_without_sessions_is_refused(app):
    with pytest.raises(ValueError):
        RecoveryDialog([], now=NOW)


# --- Several sessions --------------------------------------------------------


def _three():
    # recoverable_sessions already returns newest first; the dialog keeps it.
    return [
        _session("House.berm", _at(14, 32)),
        _session("Shed.berm", _at(10, 15, days_ago=1)),
        _session("Untitled", _at(9, 0, days_ago=3)),
    ]


def test_several_sessions_are_listed_newest_first_with_the_first_selected(app):
    seen = []
    answer_recovery_dialog("Decide later", seen)
    RecoveryDialog(_three(), now=NOW).exec()

    assert seen[0]["rows"] == [
        "House.berm, last autosaved today at 14:32",
        "Shed.berm, last autosaved yesterday at 10:15",
        "Untitled, last autosaved 2 Oct 2026 at 09:00",
    ]
    assert seen[0]["selected"] == 0
    assert seen[0]["buttons"] == ["Recover", "Discard", "Decide later"]
    assert seen[0]["text"] == (
        "Bermake closed unexpectedly. Choose the autosaved work to recover. "
        "The rest is kept for next time."
    )


@pytest.mark.parametrize(("button", "action"), [("Recover", "recover"), ("Discard", "discard")])
@pytest.mark.parametrize("row", [0, 2])
def test_recover_and_discard_act_on_the_selected_row(app, button, action, row):
    sessions = _three()
    seen = []
    answer_recovery_dialog(button, seen, select_row=row)
    choice = RecoveryDialog(sessions, now=NOW).exec()
    assert choice.action == action
    assert choice.session is sessions[row]


def test_decide_later_keeps_all_of_them_whatever_is_selected(app):
    seen = []
    answer_recovery_dialog("Decide later", seen, select_row=1)
    assert RecoveryDialog(_three(), now=NOW).exec() == RecoveryChoice("later", None)


def test_with_no_row_selected_recover_and_discard_are_disabled(app):
    dialog = RecoveryDialog(_three(), now=NOW)
    from PySide6.QtWidgets import QListWidget, QPushButton

    rows = dialog.findChild(QListWidget, "RecoverySessions")
    buttons = {b.text().replace("&", ""): b for b in dialog.findChildren(QPushButton)}
    assert buttons["Recover"].isEnabled() and buttons["Discard"].isEnabled()

    rows.clearSelection()
    assert not buttons["Recover"].isEnabled()
    assert not buttons["Discard"].isEnabled()
    assert buttons["Decide later"].isEnabled()

    rows.setCurrentRow(1)
    assert buttons["Recover"].isEnabled() and buttons["Discard"].isEnabled()
    dialog.deleteLater()


def test_recover_is_the_default_button(app):
    from PySide6.QtWidgets import QPushButton

    dialog = RecoveryDialog([_session("House.berm", _at(14, 32))], now=NOW)
    buttons = {b.text().replace("&", ""): b for b in dialog.findChildren(QPushButton)}
    assert buttons["Recover"].isDefault()
    assert not buttons["Discard"].isDefault()
    dialog.deleteLater()


def test_the_helper_leaves_no_dialog_open(app):
    """Sanity: every dialog above closed, so later tests start clean."""
    QApplication.processEvents()
    assert not [
        w for w in QApplication.topLevelWidgets() if isinstance(w, RecoveryDialog) and w.isVisible()
    ]
