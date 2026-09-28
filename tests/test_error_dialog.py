"""The deferred error dialog (M7.9). The dialog itself is a modal QMessageBox and
is checked by the manual clean-machine gate; this pins the scheduling around it."""

from pathlib import Path

from bermake.diagnostics.error_dialog import deferred_error_dialog


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
