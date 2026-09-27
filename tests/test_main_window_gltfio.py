from __future__ import annotations

from pathlib import Path

import bermake.ui.main_window as mw_mod
from bermake.scene.scene import Side
from bermake.ui.main_window import MainWindow

_GLTF_DATA = Path(__file__).parent / "data" / "gltf"


def _win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_menu_has_gltf_actions(qtbot):
    w = _win(qtbot)
    labels = [a.text() for a in w._file_menu.actions()]
    assert any("glTF" in t and "Import" in t for t in labels)
    assert any("glTF" in t and "Export" in t for t in labels)


def test_export_gltf_calls_export(qtbot, monkeypatch, tmp_path):
    w = _win(qtbot)
    w._prompt_save_path = lambda *a, **k: str(tmp_path / "m.glb")
    called = {}
    monkeypatch.setattr(mw_mod, "export_gltf",
                        lambda model, path: called.setdefault("path", path))
    w._on_export_gltf()
    assert called["path"].endswith(".glb")


def test_import_gltf_cancelled_is_noop(qtbot):
    w = _win(qtbot)
    w._prompt_open_path = lambda *a, **k: None
    w._on_import_gltf()   # must not raise


def test_import_gltf_runs_command(qtbot, monkeypatch, tmp_path):
    from bermake.io.gltf_scene import GltfMesh, GltfNode, GltfSceneData
    tri = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    ident = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    scene = GltfSceneData(
        nodes=(GltfNode(name="A", parent=-1, transform=ident, mesh_indices=(0,)),),
        meshes=(GltfMesh(positions=tri, triangles=((0, 1, 2),), material_index=-1),),
        materials=(),
    )
    w = _win(qtbot)
    w._prompt_open_path = lambda *a, **k: str(tmp_path / "m.glb")
    monkeypatch.setattr(mw_mod, "read_gltf_scene", lambda path: scene)
    before = len(w._model.active_context.children)
    w._on_import_gltf()
    assert len(w._model.active_context.children) == before + 1


def test_export_gltf_appends_glb_extension(qtbot, monkeypatch, tmp_path):
    w = _win(qtbot)
    w._prompt_save_path = lambda *a, **k: str(tmp_path / "model")
    called = {}
    monkeypatch.setattr(mw_mod, "export_gltf",
                        lambda model, path: called.setdefault("path", path))
    w._on_export_gltf()
    assert called["path"].endswith(".glb")


def test_export_gltf_cancelled_is_noop(qtbot, monkeypatch):
    w = _win(qtbot)
    w._prompt_save_path = lambda *a, **k: None
    called = {}
    monkeypatch.setattr(mw_mod, "export_gltf",
                        lambda model, path: called.setdefault("path", path))
    w._on_export_gltf()   # must not raise
    assert "path" not in called


def test_import_gltf_stores_uvs_and_textures_and_repaints(qtbot, monkeypatch):
    """Stage 2's reviewer found six command-level tests and nothing asserting
    the viewport actually repaints (the M7.5b regression shape) -- this uses
    the real MainWindow fixture, not a mock window, and a counting stub on
    viewport.update rather than trusting that the call happened."""
    w = _win(qtbot)
    w._prompt_open_path = lambda *a, **k: str(_GLTF_DATA / "textured_box.glb")

    update_calls = {"n": 0}

    def _counting_update():
        update_calls["n"] += 1

    monkeypatch.setattr(w._viewport, "update", _counting_update)

    w._on_import_gltf()

    assert update_calls["n"] >= 1, "a completed import must repaint the viewport"

    textures = list(w._model.textures.textures())
    assert len(textures) == 1

    found_uvs = False
    for defn, _world in w._model.traverse():
        for f in defn.mesh.faces_iter():
            if defn.mesh.face_uvs(f.id, Side.FRONT) is not None:
                found_uvs = True
                break
        if found_uvs:
            break
    assert found_uvs, "at least one imported face must carry stored UVs"


def test_import_gltf_bad_file_shows_dialog(qtbot, monkeypatch, tmp_path):
    from bermake.io.errors import BermakeFormatError

    def _raise(path):
        raise BermakeFormatError("bad")

    w = _win(qtbot)
    w._prompt_open_path = lambda *a, **k: str(tmp_path / "bad.glb")
    monkeypatch.setattr(mw_mod, "read_gltf_scene", _raise)
    shown = {}
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: shown.setdefault("called", True))
    before = len(w._model.active_context.children)
    w._on_import_gltf()   # must not raise
    assert shown.get("called") is True
    assert len(w._model.active_context.children) == before
