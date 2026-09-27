"""Shared pytest fixtures and configuration."""

import os

import numpy as np
import pytest
from bermake.commands.group_commands import MakeGroupCommand
from bermake.model.model import Model

# Ensure Qt uses the offscreen platform in CI / headless environments.
# This must run BEFORE QApplication is created (i.e., before any pytest-qt fixture).
if os.environ.get("CI") == "true" or os.environ.get("QT_QPA_PLATFORM"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _scratch_window_settings(tmp_path, monkeypatch):
    """Keep MainWindow's window-state persistence (M7.2, Task 11) off the
    real user registry during tests.

    MainWindow.__init__ builds `self._settings` from a default-constructed
    QSettings(), which is correct for the real app (app.py sets the
    organization/application name on QApplication before creating the
    window) but would write to the real HKCU registry on Windows for every
    MainWindow() built anywhere in this suite -- including the many test
    files that construct one directly rather than through the `main_window`
    fixture below. Patch the name main_window.py resolves at call time so
    every QSettings() it creates is actually an ini-backed scratch store
    unique to this test.
    """
    try:
        import bermake.ui.main_window as main_window_module
    except Exception:
        return

    from PySide6.QtCore import QSettings

    ini_path = str(tmp_path / "window_state.ini")

    def _scratch_settings(*_args, **_kwargs):
        return QSettings(ini_path, QSettings.Format.IniFormat)

    monkeypatch.setattr(main_window_module, "QSettings", _scratch_settings)


@pytest.fixture(autouse=True)
def _no_blocking_close_dialog(monkeypatch):
    """Keep MainWindow's unsaved-changes modal from hanging test teardown.

    pytest-qt closes every qtbot-tracked widget at teardown. MainWindow.closeEvent
    (M6a) prompts via a modal QMessageBox when the document is dirty; in a headless
    run nothing dismisses it, so `.exec()` would block the whole suite forever.
    Default the prompt to "discard" for every test so a teardown-close never blocks.
    Tests that actually exercise the guard override `_prompt_discard` on the window
    instance, which shadows this class-level patch.
    """
    try:
        from bermake.ui.main_window import MainWindow
    except Exception:
        return
    monkeypatch.setattr(MainWindow, "_prompt_discard", lambda self: "discard", raising=False)


@pytest.fixture
def model_factory():
    """Build a fresh, empty Model."""

    def make() -> Model:
        return Model()

    return make


@pytest.fixture
def group_factory():
    """Wrap every live entity in the model's active context into one new group.

    Mirrors the MakeGroupCommand construction used by tests/test_group_commands.py
    and tests/test_make_group_tag_inherit.py: the command takes the parent
    Definition plus explicit vertex/edge/face id lists, and do(model) is called
    directly (no CommandStack -- these are test fixtures building state, not
    undoable user actions).
    """

    def make(model: Model):
        context = model.active_context
        scene = context.mesh
        vertex_ids = [v.id for v in scene.vertices_iter()]
        edge_ids = [e.id for e in scene.edges_iter()]
        face_ids = [f.id for f in scene.faces_iter()]
        command = MakeGroupCommand(context, vertex_ids, edge_ids, face_ids)
        command.do(model)
        return command.created_instance

    return make


@pytest.fixture
def main_window(qtbot):
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    return window


@pytest.fixture
def main_window_with_square(main_window):
    scene = main_window._model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)
    return main_window


@pytest.fixture
def main_window_with_group(main_window_with_square):
    window = main_window_with_square
    window._on_select_all()
    window._on_make_group()
    window._selection.clear()
    return window
