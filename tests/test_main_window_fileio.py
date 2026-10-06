import numpy as np
import pytest
from bermake.commands.scene_commands import ClearSceneCommand
from bermake.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    yield a


def _draw_something(win):
    scene = win._model.active_scene
    vids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    ]
    scene.add_face_from_loop(vids)


def test_command_execution_marks_dirty_and_titles(app):
    win = MainWindow()
    assert win._doc_controller.dirty is False
    assert win.windowTitle() == "Untitled - Bermake"
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    assert win._doc_controller.dirty is True
    assert win.windowTitle() == "Untitled* - Bermake"


def test_save_as_writes_file_and_marks_clean(app, tmp_path, monkeypatch):
    win = MainWindow()
    _draw_something(win)
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    assert win._doc_controller.dirty is True

    target = tmp_path / "out.berm"
    monkeypatch.setattr(win, "_prompt_save_path", lambda *a, **k: str(target))
    assert win._on_file_save_as() is True
    assert target.exists()
    assert win._doc_controller.dirty is False
    assert win.windowTitle() == "out.berm - Bermake"


def test_guard_cancel_aborts(app):
    win = MainWindow()
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    win._prompt_discard = lambda: "cancel"
    assert win._confirm_discard_if_dirty() is False


def test_guard_discard_proceeds(app):
    win = MainWindow()
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    win._prompt_discard = lambda: "discard"
    assert win._confirm_discard_if_dirty() is True


def test_file_menu_has_actions(app):
    win = MainWindow()
    labels = [a.text() for a in win._file_menu.actions() if a.text()]
    assert any("New" in t for t in labels)
    assert any("Open" in t for t in labels)
    assert any("Save" in t for t in labels)


def test_close_event_ignored_when_guard_cancels(app):
    from PySide6.QtGui import QCloseEvent

    win = MainWindow()
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)  # dirty
    win._prompt_discard = lambda: "cancel"
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()  # guard refused the close, window stays open


def test_close_event_accepted_when_clean(app):
    from PySide6.QtGui import QCloseEvent

    win = MainWindow()
    assert win._doc_controller.dirty is False
    event = QCloseEvent()
    win.closeEvent(event)
    assert event.isAccepted()  # nothing to lose, closes without prompting


def test_new_resets_to_clean_untitled(app, tmp_path):
    win = MainWindow()
    _draw_something(win)
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    win._prompt_discard = lambda: "discard"
    win._on_file_new()
    assert win._doc_controller.current_path is None
    assert win._doc_controller.dirty is False
    assert win.windowTitle() == "Untitled - Bermake"
    assert not win._command_stack.can_undo  # history cleared


def test_new_rebuilds_the_outliner(app):
    # CommandStack.clear() (called by _reset_document) fires no listeners --
    # so without an explicit rebuild, the Outliner keeps showing the PREVIOUS
    # document's hierarchy until the user happens to run any command.
    win = MainWindow()
    _draw_something(win)
    win._on_select_all()
    win._on_make_group()
    instance_id = next(iter(win._selection.instances))
    assert win._outliner.item_for(instance_id) is not None  # sanity: row exists pre-New

    win._prompt_discard = lambda: "discard"
    win._on_file_new()

    assert win._outliner.item_for(instance_id) is None
    assert win._outliner.topLevelItem(0).childCount() == 0


def test_open_success_swaps_model_and_clears_history(app, tmp_path, monkeypatch):
    # First, save a file with a known box.
    saver = MainWindow()
    _draw_something(saver)
    target = tmp_path / "doc.berm"
    saver._prompt_save_path = lambda *a, **k: str(target)
    assert saver._on_file_save_as() is True

    # Now open it in a fresh window with a dirty scratch doc.
    win = MainWindow()
    win._command_stack.execute(ClearSceneCommand(), win._model.active_scene)
    win._prompt_discard = lambda: "discard"
    win._prompt_open_path = lambda: str(target)
    win._on_file_open()
    assert win._doc_controller.current_path == target
    assert win._doc_controller.dirty is False
    assert len(list(win._model.root.mesh.faces_iter())) == 1
    assert not win._command_stack.can_undo


def test_open_failure_keeps_current_model(app, monkeypatch):
    from bermake.io import BermakeFormatError

    win = MainWindow()
    _draw_something(win)
    before_root = win._model.root
    win._prompt_open_path = lambda: "/whatever.berm"

    import bermake.ui.main_window as mw

    monkeypatch.setattr(
        mw, "load_document", lambda p: (_ for _ in ()).throw(BermakeFormatError("bad"))
    )
    # Suppress + record the error dialog.
    shown = {}
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))
    win._on_file_open()
    assert win._model.root is before_root  # unchanged
    assert shown.get("called") is True  # error surfaced


def test_open_dialog_default_filter_includes_all_files():
    """The Open dialog's default filter must offer an all-files entry.

    A document written by the old (pre-rename) application does not match
    "*.berm". Without an all-files entry alongside it, that document would
    simply not appear in the dialog: the user would see an empty folder,
    read it as lost work, and never reach the error message that names the
    old format and explains why it will not open.
    """
    import inspect

    parameters = inspect.signature(MainWindow._prompt_open_path).parameters
    default_filter = parameters["file_filter"].default
    filters = [f.strip() for f in default_filter.split(";;")]
    assert any(f.endswith("(*.berm)") for f in filters)
    assert any(f.endswith("(*)") for f in filters)
