"""Task 10: MainWindow owns a Model; ToolContext routes to the active scene."""
import numpy as np
import pytest
from bermake.ui.main_window import MainWindow


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_mainwindow_has_model_with_root_scene(win):
    assert win._model is not None
    assert win.scene is win._model.active_scene
    assert win.scene is win._model.root.mesh


def test_active_scene_follows_entered_context(win):
    d = win._model.new_definition("G", is_group=True)
    inst = win._model.new_instance(d)
    win._model.root.children.append(inst)
    win._model.enter(inst)
    win._rebuild_tool_context()
    assert win.scene is d.mesh
