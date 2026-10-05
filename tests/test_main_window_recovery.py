"""Recovery at startup in the window (M7.12, #77; spec 4.5 and 5).

Stale sessions are written through the real RecoveryStore with a fake PID and
start time, and liveness is patched to say their writer is gone. The recovery
folder is the per-test one the conftest autouse fixture installs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import bermake.io.bermake_file as bermake_file
import bermake.recovery.liveness as liveness
import bermake.recovery.startup as recovery_startup
import bermake.ui.main_window as main_window_module
import numpy as np
import pytest
from bermake.io import SCHEMA_VERSION, load_document
from bermake.model import Model
from bermake.recovery.liveness import current_process_started_at
from bermake.recovery.startup import recoverable_sessions
from bermake.recovery.store import RecoveryStore
from bermake.ui.recovery_dialog import RecoveryChoice
from PySide6.QtWidgets import QFileDialog

from tests._recovery_helpers import answer_message_box, answer_recovery_dialog, write_session


@pytest.fixture(autouse=True)
def _writers_are_gone(monkeypatch):
    """Every session written here was left by a Bermake that is not running."""
    monkeypatch.setattr(liveness, "process_is_running", lambda pid, started_at: False)


@pytest.fixture
def win(main_window):
    main_window._autosave_timer.stop()
    return main_window


@pytest.fixture
def folder(win) -> Path:
    return win._recovery_store.folder


@pytest.fixture
def failures(win, monkeypatch):
    """The "could not open" messages the window showed."""
    shown = []
    monkeypatch.setattr(win, "_show_recovery_failed", shown.append)
    return shown


def _square() -> Model:
    model = Model()
    scene = model.root.mesh
    vids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0))
    ]
    scene.add_face_from_loop(vids)
    return model


def _counts(model) -> tuple[int, int]:
    mesh = model.root.mesh
    return len(list(mesh.vertices_iter())), len(list(mesh.faces_iter()))


def _choose(win, monkeypatch, action, pick=0):
    """Answer the recovery prompt with `action` on the `pick`th session."""
    asked = []

    def ask(sessions):
        asked.append(list(sessions))
        return RecoveryChoice(action, None if action == "later" else sessions[pick])

    monkeypatch.setattr(win, "_ask_recovery", ask)
    return asked


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def _failure_text(win, name: str) -> str:
    return (
        f"Bermake could not open the autosaved work for {name}. "
        f"It was kept in {win._recovery_store.folder} for a bug report."
    )


# --- No sessions -------------------------------------------------------------


def test_no_sessions_shows_nothing_and_returns_false(win, monkeypatch):
    asked = _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert asked == []


def test_a_live_writers_session_is_not_offered(win, folder, monkeypatch):
    monkeypatch.setattr(liveness, "process_is_running", lambda pid, started_at: True)
    _sid, berm, meta = write_session(folder, _square())
    asked = _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert asked == []
    assert berm.exists() and meta.exists()


# --- Recover -----------------------------------------------------------------


@pytest.fixture
def recovered(win, folder, monkeypatch, tmp_path):
    """A House.berm session recovered into the window."""
    original = str(tmp_path / "drawings" / "House.berm")
    old_id, berm, meta = write_session(
        folder, _square(), original_path=original, display_name="House.berm"
    )
    _choose(win, monkeypatch, "recover")
    result = win.run_startup_recovery()
    return {"result": result, "old_id": old_id, "berm": berm, "meta": meta, "original": original}


def test_recover_returns_true(recovered):
    assert recovered["result"] is True


def test_recover_loads_the_saved_model(win, recovered):
    assert _counts(win._model) == (4, 1)


def test_recover_leaves_the_document_dirty_with_no_path(win, recovered):
    assert win._doc_controller.dirty is True
    assert win._doc_controller.current_path is None


def test_recover_titles_the_window_as_recovered(win, recovered):
    assert "House.berm (recovered)" in win.windowTitle()
    assert win.windowTitle() == "House.berm (recovered)* - Bermake"


def test_save_after_recover_opens_save_as_at_the_original_path(win, recovered, monkeypatch):
    offered = []
    monkeypatch.setattr(
        win, "_prompt_save_path", lambda *a, suggested=None, **k: offered.append(suggested)
    )
    assert win._on_file_save() is False
    assert offered == [recovered["original"]]


def test_save_after_recover_writes_where_the_user_chose_and_drops_the_tag(
    win, recovered, monkeypatch, tmp_path
):
    target = tmp_path / "Saved.berm"
    monkeypatch.setattr(win, "_prompt_save_path", lambda *a, **k: str(target))
    assert win._on_file_save() is True
    assert win.windowTitle() == "Saved.berm - Bermake"
    assert _counts(load_document(target).model) == (4, 1)


def test_save_as_after_saving_a_recovered_document_no_longer_suggests_the_original(
    win, recovered, monkeypatch, tmp_path
):
    monkeypatch.setattr(win, "_prompt_save_path", lambda *a, **k: str(tmp_path / "Saved.berm"))
    assert win._on_file_save() is True

    offered = []
    monkeypatch.setattr(
        win, "_prompt_save_path", lambda *a, suggested=None, **k: offered.append(suggested)
    )
    win._on_file_save_as()
    assert offered == [None]


def test_recover_hands_the_files_to_a_new_live_session(win, folder, recovered):
    new_id = win._session_id
    assert new_id != recovered["old_id"]
    assert not recovered["berm"].exists()
    assert not recovered["meta"].exists()
    assert _names(folder) == sorted([f"{new_id}.berm", f"{new_id}.json"])

    meta = json.loads((folder / f"{new_id}.json").read_text(encoding="utf-8"))
    assert meta["session_id"] == new_id
    assert meta["pid"] == os.getpid()
    assert meta["process_started_at"] == current_process_started_at()
    assert meta["original_path"] == recovered["original"]
    assert meta["display_name"] == "House.berm"
    assert _counts(load_document(folder / f"{new_id}.berm").model) == (4, 1)


def test_recover_starts_with_nothing_new_to_autosave(win, recovered):
    """The disk copy already matches the window (spec 4.5)."""
    assert win._autosave.changed_since_autosave is False


def test_recover_records_the_current_time(win, folder, recovered):
    from datetime import datetime

    meta = json.loads((folder / f"{win._session_id}.json").read_text(encoding="utf-8"))
    age = datetime.now().astimezone() - datetime.fromisoformat(meta["saved_at"])
    assert abs(age.total_seconds()) < 60


def test_a_crash_after_recover_still_leaves_the_work_recoverable(win, folder, recovered):
    """Without closing the window: treat this process as dead and look again."""
    [session] = recoverable_sessions(RecoveryStore(folder), is_running=lambda pid, started: False)
    assert session.meta.session_id == win._session_id
    assert session.meta.display_name == "House.berm"
    assert _counts(load_document(session.berm_path).model) == (4, 1)


def test_new_after_recover_drops_the_recovered_title(win, recovered):
    win._on_file_new()
    assert win.windowTitle() == "Untitled - Bermake"


def test_the_close_prompt_names_the_recovered_drawing(win, recovered):
    assert win._unsaved_changes_name() == "House.berm"


def test_recover_an_untitled_session(win, folder, monkeypatch):
    write_session(folder, _square(), original_path=None, display_name="Untitled")
    _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is True
    assert win.windowTitle() == "Untitled (recovered)* - Bermake"

    offered = []
    monkeypatch.setattr(
        win, "_prompt_save_path", lambda *a, suggested=None, **k: offered.append(suggested)
    )
    win._on_file_save()
    assert offered == [None]


def test_recover_one_of_several_keeps_the_rest(win, folder, monkeypatch):
    first = write_session(folder, _square(), display_name="A.berm", original_path="C:/d/A.berm")
    second = write_session(folder, Model(), display_name="B.berm", original_path="C:/d/B.berm")
    asked = _choose(win, monkeypatch, "recover", pick=1)
    assert win.run_startup_recovery() is True
    assert len(asked[0]) == 2
    picked = asked[0][1]
    kept = first if picked.meta.session_id == second[0] else second
    assert kept[1].exists() and kept[2].exists()
    assert win.windowTitle() == f"{picked.meta.display_name} (recovered)* - Bermake"


def test_the_real_dialog_is_shown_modal_over_the_window(win, folder):
    write_session(folder, _square())
    seen = []
    answer_recovery_dialog("Recover", seen)
    assert win.run_startup_recovery() is True
    assert seen[0]["parent"] is win
    assert seen[0]["text"].startswith("Bermake closed unexpectedly. Recover House.berm,")


# --- Discard and Decide later ------------------------------------------------


def test_discard_deletes_the_sessions_files(win, folder, monkeypatch):
    _sid, berm, meta = write_session(folder, _square())
    _choose(win, monkeypatch, "discard")
    assert win.run_startup_recovery() is False
    assert not berm.exists() and not meta.exists()
    assert _names(folder) == []
    assert win.windowTitle() == "Untitled - Bermake"


def test_discard_deletes_only_the_selected_session(win, folder, monkeypatch):
    first = write_session(folder, _square(), display_name="A.berm")
    second = write_session(folder, _square(), display_name="B.berm")
    asked = _choose(win, monkeypatch, "discard", pick=0)
    win.run_startup_recovery()
    gone = asked[0][0].meta.session_id
    kept = second if gone == first[0] else first
    assert kept[1].exists() and kept[2].exists()
    assert not (folder / f"{gone}.berm").exists()


def test_decide_later_keeps_the_files_and_the_window(win, folder, monkeypatch):
    _sid, berm, meta = write_session(folder, _square())
    _choose(win, monkeypatch, "later")
    assert win.run_startup_recovery() is False
    assert berm.exists() and meta.exists()
    assert _counts(win._model) == (0, 0)
    assert win.windowTitle() == "Untitled - Bermake"


# --- Nothing escapes ---------------------------------------------------------


def test_an_error_while_asking_is_logged_and_returns_false(win, folder, monkeypatch, caplog):
    write_session(folder, _square())

    def broken(sessions):
        raise RuntimeError("dialog exploded")

    monkeypatch.setattr(win, "_ask_recovery", broken)
    with caplog.at_level(logging.ERROR):
        assert win.run_startup_recovery() is False
    assert "dialog exploded" in caplog.text


def test_an_error_while_listing_is_logged_and_returns_false(win, monkeypatch, caplog):
    def broken(store, is_running=None):
        raise OSError("folder is gone")

    monkeypatch.setattr(recovery_startup, "recoverable_sessions", broken)
    with caplog.at_level(logging.ERROR):
        assert win.run_startup_recovery() is False
    assert "folder is gone" in caplog.text


# --- Review I1: a hand-over that fails part way -------------------------------


@pytest.fixture(params=[5, 0], ids=["interval-5", "autosave-off"])
def interval(request, win):
    """Run each hand-over test with the default interval and with autosave Off."""
    win._autosave.set_interval(request.param)
    return request.param


def _rename_fails(win, monkeypatch):
    """hand_over's first step, the .berm rename, fails (the file is locked)."""

    def locked(old_id, new_meta):
        raise PermissionError("locked by antivirus")

    monkeypatch.setattr(win._recovery_store, "hand_over", locked)


def _metadata_write_fails_once(win, monkeypatch):
    """hand_over renames the .berm, then its metadata write fails (disk full);
    later metadata writes work."""
    store = win._recovery_store
    real = store._write_meta
    calls = []

    def flaky(meta):
        calls.append(meta.session_id)
        if len(calls) == 1:
            raise OSError(28, "No space left on device")
        real(meta)

    monkeypatch.setattr(store, "_write_meta", flaky)
    return calls


def _autosave_fails(win, monkeypatch):
    def full(meta, write_berm):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(win._recovery_store, "write", full)


def _after_a_second_crash(folder):
    """What the next launch offers if this process dies now."""
    return recoverable_sessions(RecoveryStore(folder), is_running=lambda pid, started: False)


@pytest.mark.parametrize("failure", [_rename_fails, _metadata_write_fails_once])
def test_a_failed_hand_over_autosaves_at_once_and_drops_the_old_session(
    win, folder, monkeypatch, caplog, interval, failure
):
    _old_id, _berm, _meta = write_session(folder, _square())
    failure(win, monkeypatch)
    _choose(win, monkeypatch, "recover")

    with caplog.at_level(logging.WARNING):
        assert win.run_startup_recovery() is True

    assert _counts(win._model) == (4, 1)
    assert "Could not hand recovery session" in caplog.text
    # The window's own session is on disk now, not one interval from now.
    assert _names(folder) == sorted([f"{win._session_id}.berm", f"{win._session_id}.json"])
    assert win._autosave.changed_since_autosave is False
    [session] = _after_a_second_crash(folder)
    assert session.meta.session_id == win._session_id
    assert session.meta.pid == os.getpid()
    assert session.meta.display_name == "House.berm"
    assert _counts(load_document(session.berm_path).model) == (4, 1)


def test_after_a_failed_rename_a_save_leaves_nothing_to_offer(
    win, folder, monkeypatch, tmp_path, interval
):
    """Case A of I1: the old session must not outlive the user's Save and
    come back next launch with older content."""
    write_session(folder, _square())
    _rename_fails(win, monkeypatch)
    _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is True

    monkeypatch.setattr(win, "_prompt_save_path", lambda *a, **k: str(tmp_path / "Saved.berm"))
    assert win._on_file_save() is True
    assert _after_a_second_crash(folder) == []


def test_a_locked_old_session_is_logged_not_raised(win, folder, monkeypatch, caplog):
    write_session(folder, _square())
    _rename_fails(win, monkeypatch)
    real_delete = win._recovery_store.delete

    def locked_delete(session_id):
        if session_id != win._session_id:
            raise PermissionError("old session locked")
        real_delete(session_id)

    monkeypatch.setattr(win._recovery_store, "delete", locked_delete)
    _choose(win, monkeypatch, "recover")
    with caplog.at_level(logging.WARNING):
        assert win.run_startup_recovery() is True
    assert "old session locked" in caplog.text
    assert (folder / f"{win._session_id}.json").exists()


@pytest.mark.parametrize("failure", [_rename_fails, _metadata_write_fails_once])
def test_when_the_immediate_autosave_fails_too_the_old_session_stays_recoverable(
    win, folder, monkeypatch, caplog, interval, failure
):
    """Nothing written under the new session: the old one, put back together
    if the rename had already happened, is what the next launch offers."""
    old_id, berm, meta = write_session(folder, _square())
    failure(win, monkeypatch)
    _autosave_fails(win, monkeypatch)
    _choose(win, monkeypatch, "recover")

    with caplog.at_level(logging.WARNING):
        assert win.run_startup_recovery() is True

    assert _counts(win._model) == (4, 1)
    assert _names(folder) == sorted([berm.name, meta.name])
    [session] = _after_a_second_crash(folder)
    assert session.meta.session_id == old_id
    assert _counts(load_document(session.berm_path).model) == (4, 1)
    # Still unsaved work, retried at the next interval.
    assert win._autosave.changed_since_autosave is True
    assert f"recovery session {old_id} is kept" in caplog.text


# --- Review Focus 2: vanished or locked between listing and loading ----------


def test_a_session_deleted_before_loading_is_skipped_with_the_message(
    win, folder, monkeypatch, failures
):
    sid, berm, meta = write_session(folder, _square(), display_name="House.berm")

    def ask(sessions):
        berm.unlink()  # antivirus, another Bermake, or the user, mid-startup
        return RecoveryChoice("recover", sessions[0])

    monkeypatch.setattr(win, "_ask_recovery", ask)
    assert win.run_startup_recovery() is False
    assert failures == [_failure_text(win, "House.berm")]
    assert not meta.exists()
    assert (folder / f"{sid}.broken.json").exists()
    assert win.windowTitle() == "Untitled - Bermake"
    assert _counts(win._model) == (0, 0)


def test_a_locked_session_is_quarantined_with_the_message(win, folder, monkeypatch, failures):
    sid, berm, meta = write_session(folder, _square(), display_name="House.berm")

    def locked(path):
        raise PermissionError(13, "The process cannot access the file", str(path))

    monkeypatch.setattr(main_window_module, "load_document", locked)
    _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert failures == [_failure_text(win, "House.berm")]
    assert (folder / f"{sid}.broken.berm").exists()
    assert (folder / f"{sid}.broken.json").exists()
    assert not berm.exists() and not meta.exists()
    assert win.windowTitle() == "Untitled - Bermake"


def test_any_load_error_is_quarantined_with_the_message(win, folder, monkeypatch, failures):
    sid, _berm, _meta = write_session(folder, _square(), display_name="House.berm")

    def bug(path):
        raise ValueError("a codec bug")

    monkeypatch.setattr(main_window_module, "load_document", bug)
    _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert failures == [_failure_text(win, "House.berm")]
    assert (folder / f"{sid}.broken.berm").exists()


def test_the_failure_message_is_a_real_warning_box(win, folder, monkeypatch):
    _sid, berm, _meta = write_session(folder, _square(), display_name="House.berm")
    berm.unlink()
    berm.write_bytes(b"not a zip")
    _choose(win, monkeypatch, "recover")
    seen = []
    answer_message_box("OK", seen)
    assert win.run_startup_recovery() is False
    assert seen[0]["text"] == _failure_text(win, "House.berm")
    assert seen[0]["title"] == "Recovery failed"


def test_a_failing_quarantine_still_shows_the_message(win, folder, monkeypatch, failures):
    write_session(folder, _square(), display_name="House.berm")

    def bug(path):
        raise OSError("gone")

    def broken_quarantine(session_id, reason):
        raise OSError("cannot rename")

    monkeypatch.setattr(main_window_module, "load_document", bug)
    monkeypatch.setattr(win._recovery_store, "quarantine", broken_quarantine)
    _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert failures == [_failure_text(win, "House.berm")]


# --- Review Focus 3: an autosave from a newer Bermake ------------------------


def test_a_newer_schema_session_is_quarantined_never_deleted(win, folder, monkeypatch, failures):
    with monkeypatch.context() as newer_build:
        newer_build.setattr(bermake_file, "SCHEMA_VERSION", SCHEMA_VERSION + 1)
        sid, berm, meta = write_session(folder, _square(), display_name="House.berm")
    _choose(win, monkeypatch, "recover")

    assert win.run_startup_recovery() is False

    assert failures == [_failure_text(win, "House.berm")]
    assert not berm.exists() and not meta.exists()
    broken = folder / f"{sid}.broken.berm"
    assert broken.exists()
    assert (folder / f"{sid}.broken.json").exists()
    # The newer build's file is kept intact for the bug report.
    import zipfile

    with zipfile.ZipFile(broken) as zf:
        assert json.loads(zf.read("manifest.json"))["schema_version"] == SCHEMA_VERSION + 1
    # Never offered again.
    asked = _choose(win, monkeypatch, "recover")
    assert win.run_startup_recovery() is False
    assert asked == []


# --- Save As suggestion plumbing ---------------------------------------------


def test_prompt_save_path_passes_the_suggestion_to_the_file_dialog(win, monkeypatch):
    calls = []

    def fake(parent, title, directory, file_filter):
        calls.append(directory)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(fake))
    win._prompt_save_path(suggested="C:/drawings/House.berm")
    win._prompt_save_path()
    assert calls == ["C:/drawings/House.berm", ""]
