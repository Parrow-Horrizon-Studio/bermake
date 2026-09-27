from __future__ import annotations

import struct
import zlib

import pytest
from bermake.ui.main_window import MainWindow
from bermake.ui.materials_page import MaterialsPage


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


class _RecordingGL:
    """Stands in for the GL module so TextureCache calls never touch real GL."""

    def glDeleteTextures(self, ids):
        pass

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            pass

        return _noop


def _png(w, h, rgba):
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


_PNG = _png(2, 2, [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 9, 9, 9, 255])


def test_main_window_has_materials_page(win):
    assert isinstance(win._materials_page, MaterialsPage)


def test_paint_tool_registered_under_b(win):
    assert win._tool_manager.activate_by_shortcut("B")
    assert win._tool_manager.active.name == "Paint"


def test_tool_context_exposes_material_hooks(win):
    ctx = win._tool_manager._ctx  # installed ToolContext (ToolManager stores it as _ctx)
    assert ctx.active_material_provider is not None
    assert ctx.set_active_material is not None
    # provider returns the model's active material (Default at startup)
    assert ctx.active_material_provider().id == win._model.materials.DEFAULT_ID


def test_page_selection_updates_active_material_id(win):
    brick = next(m for m in win._model.materials.materials() if m.name == "Brick Red")
    win._materials_page._on_pick(brick.id)
    assert win._active_material_id == brick.id


def test_undoing_a_texture_import_queues_it_for_renderer_eviction(win, tmp_path):
    # M7.5b Task 9 fix round 1: AddTextureCommand.undo() removes the Texture
    # record, but _on_after_undo_redo is an ordinary Qt slot with no current
    # GL context (ViewportWidget never calls makeCurrent() outside
    # initializeGL/resizeGL/paintGL) -- so it can only QUEUE the id for
    # eviction; the actual glDeleteTextures is deferred to the next render()
    # (see tests/test_renderer_texture_eviction.py for that half). Calling
    # TextureCache.invalidate() straight from this slot would be a silent
    # no-op on WGL that also forgets the handle, permanently, which is worse
    # than doing nothing.
    from bermake.viewport.texture_cache import TextureCache

    win._viewport.scene_renderer._texture_cache = TextureCache(gl=_RecordingGL())
    page = win._materials_page
    mat = win._model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    page.set_library(win._model.materials)
    page.set_active(mat.id)
    path = tmp_path / "brick.png"
    path.write_bytes(_PNG)
    page._choose_texture(chooser=lambda: str(path))
    tid = win._model.materials.get(mat.id).texture_id
    win._viewport.scene_renderer._texture_cache._ids[tid] = 55  # simulate an upload

    win._command_stack.undo()

    assert tid in win._viewport.scene_renderer._pending_stale_textures
    # Not deleted yet -- the slot must not touch GL itself.
    assert tid in win._viewport.scene_renderer._texture_cache.cached_ids()


def test_file_new_releases_the_renderer_texture_cache(win):
    # M7.5b Task 9: a new document's TextureLibrary restarts id numbering
    # from 1, so a stale GL upload from the closed document must not survive
    # to be handed back for the new document's id 1.
    calls = []
    win._viewport.scene_renderer.release_all_textures = lambda: calls.append(1)
    win._prompt_discard = lambda: "discard"

    win._on_file_new()

    assert calls == [1]


def test_paint_tool_status_text_refreshes_without_error(win):
    # Regression (M5b): PaintTool.status_text was missing its @property, so
    # `active.status_text or ""` stored the *bound method* in the status bar,
    # raising "sequence item …: expected str instance, method found" from the
    # status-bar join on every mouse move. It must resolve to a plain string.
    import inspect

    from bermake.tools.paint_tool import PaintTool

    assert isinstance(inspect.getattr_static(PaintTool, "status_text"), property)
    win._activate("paint")  # activates Paint AND sets the status-bar tool name
    win._refresh_status_text()  # would raise TypeError before the fix
    text = win._status_bar.prompt_text()
    assert isinstance(text, str)
    assert "Paint" in text
