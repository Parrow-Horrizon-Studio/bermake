"""The load path actually adopts the saved environment, not just accepts it.

A required keyword-only `environment` on _reset_document guarantees a caller
cannot omit the argument. It does not guarantee the callee applies it: a
_reset_document that took `environment` and silently dropped it would satisfy
every signature check and every existing test (none of which reads
win._doc.environment after a reopen), while a Plain White document reopened as
Sky and Ground with no error -- exactly the "file failed to save its
appearance" failure the brief's own rationale for the required keyword warns
about.

Mirrors the save/new/reopen shape of
test_main_window_scenes.test_render_style_persists_through_save_new_open, but
follows self._doc.environment instead of self._render_style.
"""

from bermake.viewport.environment import DEFAULT_ENVIRONMENT, PLAIN_WHITE, STUDIO


def _make_window(qtbot):
    from bermake.ui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_environment_persists_through_save_new_open(qtbot, tmp_path):
    win = _make_window(qtbot)
    win._doc.set_environment(PLAIN_WHITE)
    path = str(tmp_path / "environment.berm")
    assert win._save_to(path) is True

    win._on_file_new()
    assert win._doc.environment == DEFAULT_ENVIRONMENT  # New pins the default

    # Move to a third state so the reopen assertion below discriminates on its
    # own rather than free-riding on the assertion above: if it were still
    # PLAIN_WHITE from before New, a _reset_document that dropped `environment`
    # would leave it PLAIN_WHITE and the assertion would pass for the wrong
    # reason.
    win._doc.set_environment(STUDIO)

    # Re-open and confirm the saved environment is adopted, not just accepted:
    from bermake.io.bermake_file import load_document

    loaded = load_document(path)
    win._reset_document(
        loaded.model,
        loaded.camera_state,
        loaded.units,
        loaded.style,
        path,
        environment=loaded.environment,
    )
    assert win._doc.environment == PLAIN_WHITE
