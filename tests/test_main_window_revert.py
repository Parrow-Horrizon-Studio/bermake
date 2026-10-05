"""File > Revert to Saved (M7.12, #77; spec 4.6).

The prompt and the "Open failed" box are real Qt dialogs, so each test patches
the seam it needs: `_prompt_revert` on the window, and `QMessageBox.critical`
and `QMessageBox.exec` where the real text is under test.
"""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.scene_commands import AddVertexCommand
from PySide6.QtWidgets import QMessageBox


def _face(window, offset: float) -> None:
    mesh = window._model.active_context.mesh
    shift = np.array([offset, 0, 0], dtype=np.float32)
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32) + shift)
        for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    ]
    mesh.add_face_from_loop(ids)


def _face_count(window) -> int:
    return len(list(window._model.active_context.mesh.faces_iter()))


def _change(window) -> None:
    window._command_stack.execute(
        AddVertexCommand(np.array([5.0, 6.0, 7.0])), window._model.active_scene
    )


@pytest.fixture
def saved(main_window, tmp_path):
    """A window holding House.berm with one face, saved and clean."""
    _face(main_window, 0.0)
    target = tmp_path / "House.berm"
    assert main_window._save_to(target) is True
    return main_window, target


def _revert_action(window):
    return window._actions["file_revert"]


def _record_critical(monkeypatch) -> list[tuple[str, str]]:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox, "critical", lambda parent, title, text: shown.append((title, text))
    )
    return shown


def test_action_is_disabled_for_an_untitled_document(main_window):
    assert _revert_action(main_window).isEnabled() is False
    _change(main_window)
    assert main_window._doc_controller.dirty is True
    assert _revert_action(main_window).isEnabled() is False


def test_action_is_disabled_for_a_saved_clean_document(saved):
    window, _ = saved
    assert window._doc_controller.dirty is False
    assert _revert_action(window).isEnabled() is False


def test_action_is_enabled_after_a_change_to_a_saved_document(saved):
    window, _ = saved
    _change(window)
    assert _revert_action(window).isEnabled() is True


def test_action_is_disabled_again_after_saving_the_change(saved):
    window, target = saved
    _change(window)
    assert _revert_action(window).isEnabled() is True
    assert window._save_to(target) is True
    assert _revert_action(window).isEnabled() is False


def test_action_follows_a_path_change_on_a_dirty_document(main_window, tmp_path):
    _change(main_window)
    assert _revert_action(main_window).isEnabled() is False
    main_window._doc_controller.set_path(tmp_path / "x.berm")
    main_window._update_window_title()
    assert _revert_action(main_window).isEnabled() is True


def test_action_sits_in_the_file_menu_after_the_autosave_submenu(main_window):
    top = main_window._file_menu.actions()
    revert = top.index(_revert_action(main_window))
    autosave = next(i for i, a in enumerate(top) if a.menu() is not None and a.text() == "Autosave")
    assert revert > autosave
    assert _revert_action(main_window).text() == "Revert to Saved"


def test_triggering_the_action_runs_the_handler(saved):
    window, _ = saved
    _face(window, 3.0)
    _change(window)
    window._prompt_revert = lambda name: True
    _revert_action(window).trigger()
    assert _face_count(window) == 1


def test_cancel_changes_nothing(saved):
    window, target = saved
    _face(window, 3.0)
    _change(window)
    session = window._session_id
    window._prompt_revert = lambda name: False
    window._on_file_revert()
    assert _face_count(window) == 2
    assert window._doc_controller.dirty is True
    assert window._doc_controller.current_path == target
    assert window._session_id == session
    assert window._command_stack.can_undo is True


def test_revert_reloads_the_file_and_clears_undo(saved):
    window, target = saved
    _face(window, 3.0)
    _change(window)
    assert _face_count(window) == 2
    assert window._command_stack.can_undo is True
    window._prompt_revert = lambda name: True
    window._on_file_revert()
    assert _face_count(window) == 1
    assert window._command_stack.can_undo is False
    assert window._doc_controller.dirty is False
    assert window._doc_controller.current_path == target
    assert window.windowTitle() == "House.berm - Bermake"
    assert _revert_action(window).isEnabled() is False


def test_the_prompt_is_given_the_file_name(saved):
    window, _ = saved
    _change(window)
    seen: list[str] = []
    window._prompt_revert = lambda name: seen.append(name) or False
    window._on_file_revert()
    assert seen == ["House.berm"]


def test_revert_deletes_the_session_files_and_starts_a_new_session(saved):
    window, _ = saved
    _change(window)
    assert window._autosave_now() is True
    old = window._session_id
    folder = window._recovery_store.folder
    assert (folder / f"{old}.berm").exists()
    assert (folder / f"{old}.json").exists()
    window._prompt_revert = lambda name: True
    window._on_file_revert()
    assert window._session_id != old
    assert not (folder / f"{old}.berm").exists()
    assert not (folder / f"{old}.json").exists()
    # Task 2 review minor 4: the countdown restarts at the revert.
    assert window._autosave.changed_since_autosave is False


def test_an_unreadable_file_shows_open_failed_and_changes_nothing(saved, monkeypatch):
    window, target = saved
    _face(window, 3.0)
    _change(window)
    session = window._session_id
    target.unlink()
    shown = _record_critical(monkeypatch)
    window._prompt_revert = lambda name: True
    window._on_file_revert()
    assert [title for title, _ in shown] == ["Open failed"]
    assert _face_count(window) == 2
    assert window._doc_controller.dirty is True
    assert window._doc_controller.current_path == target
    assert window._session_id == session
    assert window._command_stack.can_undo is True
    assert _revert_action(window).isEnabled() is True


def test_a_damaged_file_shows_open_failed_and_changes_nothing(saved, monkeypatch):
    window, target = saved
    _face(window, 3.0)
    _change(window)
    target.write_bytes(b"this is not a berm file")
    shown = _record_critical(monkeypatch)
    window._prompt_revert = lambda name: True
    window._on_file_revert()
    assert [title for title, _ in shown] == ["Open failed"]
    assert _face_count(window) == 2
    assert window._doc_controller.dirty is True


def test_the_real_prompt_text_and_buttons(saved, monkeypatch):
    window, _ = saved
    captured: dict = {}

    def fake_exec(box):
        captured["text"] = box.text()
        captured["buttons"] = sorted(b.text() for b in box.buttons())
        default = box.defaultButton()
        captured["default"] = default.text() if default else None
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    assert window._prompt_revert("House.berm") is False
    assert captured["text"] == (
        "Discard all changes since you last saved House.berm? This cannot be undone."
    )
    assert captured["buttons"] == ["Cancel", "Revert"]
    # The destructive choice is never the default.
    assert captured["default"] == "Cancel"


@pytest.mark.parametrize(("label", "expected"), [("Revert", True), ("Cancel", False)])
def test_the_real_prompt_returns_true_only_for_revert(saved, monkeypatch, label, expected):
    window, _ = saved

    def fake_exec(box):
        next(b for b in box.buttons() if b.text() == label).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    assert window._prompt_revert("House.berm") is expected
