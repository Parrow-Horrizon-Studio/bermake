import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from bermake.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _draw_quad(win):
    scene = win._model.active_scene
    vids = [scene.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    scene.add_face_from_loop(vids)


def test_export_obj_writes_file_and_leaves_doc_clean(app, tmp_path):
    win = MainWindow()
    _draw_quad(win)
    target = tmp_path / "out.obj"
    win._prompt_save_path = lambda *a, **k: str(target)
    win._on_export_obj()
    assert target.exists()
    assert win._doc_controller.dirty is False           # export never dirties the .berm doc
    assert win._doc_controller.current_path is None      # nor sets a current path


def test_import_obj_adds_geometry_dirties_and_is_undoable(app, tmp_path):
    # a flat OBJ (no o/g) -> merges into the active scene
    obj = tmp_path / "in.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    win = MainWindow()
    win._prompt_open_path = lambda *a, **k: str(obj)
    win._on_import_obj()
    assert len(list(win._model.active_scene.faces_iter())) == 1
    assert win._doc_controller.dirty is True             # command dirties the doc
    assert win._command_stack.can_undo
    win._command_stack.undo()
    assert len(list(win._model.active_scene.faces_iter())) == 0


def test_import_obj_bad_file_shows_dialog_no_change(app, tmp_path, monkeypatch):
    obj = tmp_path / "bad.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nf 1 2 9\n")     # out-of-range index
    win = MainWindow()
    win._prompt_open_path = lambda *a, **k: str(obj)
    shown = {}
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))
    win._on_import_obj()
    assert shown.get("called") is True
    assert win._doc_controller.dirty is False


def test_file_menu_has_obj_actions(app):
    win = MainWindow()
    labels = [a.text() for a in win._file_menu.actions() if a.text()]
    assert any("Import OBJ" in t for t in labels)
    assert any("Export OBJ" in t for t in labels)
