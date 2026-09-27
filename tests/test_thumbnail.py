"""M7.5b Task 11: an optional preview image in the container."""

from __future__ import annotations

import zipfile

from PySide6.QtGui import QImage

from bermake.document import DocumentSettings
from bermake.io.bermake_file import load_document, save_document
from bermake.model.model import Model
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import RenderStyle

_PNG = b"\x89PNG\r\n\x1a\npretend-thumbnail"


def _camera():
    return Camera()


def _doc():
    return DocumentSettings()


def test_a_save_without_a_thumbnail_writes_no_entry_and_still_succeeds(tmp_path):
    # The headless path. Every existing test saves this way, so a save that
    # required a thumbnail would break the whole suite.
    path = tmp_path / "a.berm"
    save_document(path, Model(), _camera(), _doc(), RenderStyle())
    with zipfile.ZipFile(path) as zf:
        assert "thumbnail.png" not in zf.namelist()
    assert load_document(path) is not None


def test_a_supplied_thumbnail_is_written_verbatim(tmp_path):
    path = tmp_path / "b.berm"
    save_document(path, Model(), _camera(), _doc(), RenderStyle(), thumbnail=_PNG)
    with zipfile.ZipFile(path) as zf:
        assert zf.read("thumbnail.png") == _PNG


def test_a_thumbnail_does_not_disturb_the_document(tmp_path):
    # The entry is a sibling; nothing about document.json changes.
    plain, withthumb = tmp_path / "c.berm", tmp_path / "d.berm"
    model = Model()
    save_document(plain, model, _camera(), _doc(), RenderStyle())
    save_document(withthumb, model, _camera(), _doc(), RenderStyle(), thumbnail=_PNG)
    with zipfile.ZipFile(plain) as a, zipfile.ZipFile(withthumb) as b:
        assert a.read("document.json") == b.read("document.json")


def test_a_file_with_a_thumbnail_loads_normally(tmp_path):
    # An unknown-to-the-loader sibling entry must be ignored, not rejected.
    path = tmp_path / "e.berm"
    save_document(path, Model(), _camera(), _doc(), RenderStyle(), thumbnail=_PNG)
    assert load_document(path) is not None


# --- MainWindow._capture_thumbnail ------------------------------------------
#
# grabFramebuffer() is monkeypatched directly on the viewport instance, so
# these run headlessly with no real GL context: `self._viewport` is a plain
# attribute, and a QImage of an arbitrary size can be constructed without GL.


def test_a_landscape_capture_is_capped_to_512_on_its_long_edge(main_window, monkeypatch):
    monkeypatch.setattr(
        main_window._viewport,
        "grabFramebuffer",
        lambda: QImage(2048, 1024, QImage.Format.Format_RGB32),
    )
    data = main_window._capture_thumbnail()
    decoded = QImage()
    assert decoded.loadFromData(data, "PNG")
    assert decoded.width() == 512  # long edge capped
    assert decoded.height() == 256  # short edge scaled proportionally, not capped itself


def test_a_portrait_capture_is_capped_to_512_on_its_long_edge(main_window, monkeypatch):
    monkeypatch.setattr(
        main_window._viewport,
        "grabFramebuffer",
        lambda: QImage(1024, 2048, QImage.Format.Format_RGB32),
    )
    data = main_window._capture_thumbnail()
    decoded = QImage()
    assert decoded.loadFromData(data, "PNG")
    assert decoded.width() == 256  # short edge scaled proportionally, not capped itself
    assert decoded.height() == 512  # long edge capped


def test_a_capture_under_the_cap_is_encoded_at_its_own_size(main_window, monkeypatch):
    # Exercises the encode path independent of the scaling branch: a broken
    # encoder (or a cap applied unconditionally) would show up here too.
    monkeypatch.setattr(
        main_window._viewport,
        "grabFramebuffer",
        lambda: QImage(64, 48, QImage.Format.Format_RGB32),
    )
    data = main_window._capture_thumbnail()
    decoded = QImage()
    assert decoded.loadFromData(data, "PNG")
    assert (decoded.width(), decoded.height()) == (64, 48)


def test_a_null_framebuffer_capture_degrades_to_no_thumbnail(main_window, monkeypatch, tmp_path):
    # The context-less case: an unshown/uninitialized QOpenGLWidget returns a
    # null QImage from grabFramebuffer() rather than raising (verified against
    # a real ViewportWidget under QT_QPA_PLATFORM=offscreen). The save must
    # still succeed, with no thumbnail.png entry.
    monkeypatch.setattr(main_window._viewport, "grabFramebuffer", lambda: QImage())
    assert main_window._capture_thumbnail() is None

    path = tmp_path / "f.berm"
    assert main_window._save_to(path) is True
    with zipfile.ZipFile(path) as zf:
        assert "thumbnail.png" not in zf.namelist()
