"""M7.5b Task 9: wiring TextureCache.invalidate/release_all to real callers.

Task 6 built TextureCache.invalidate() and release_all() but left them with no
caller anywhere (task-6-report.md). Task 9 owns wiring both: evict_stale_textures()
reconciles the cache against the model's TextureLibrary after an undo/redo (an
undone import removes a Texture record), and release_all_textures() is the
document-close path (New / Open), where the incoming Model's TextureLibrary
restarts id numbering from 1 -- the same ints a stale GL upload could still be
cached under.

Fix round 1: both of those are called from ordinary Qt slots
(MainWindow._on_after_undo_redo / _reset_document), never from the render
path, and ViewportWidget never calls makeCurrent() outside
initializeGL/resizeGL/paintGL -- so calling TextureCache.invalidate() /
release_all() (which issue glDeleteTextures) directly from those slots runs
with no current GL context. On WGL that is a silent no-op AND the handle is
now forgotten, i.e. worse than doing nothing: unfreeable for the process
lifetime. The fix mirrors evict_unreachable(), which is documented as "can
only run here in the render path" and is in fact only ever called from
render(): the slot methods now only QUEUE which ids are stale; render()
flushes the queue (_flush_pending_texture_evictions) before drawing anything,
where a GL context is guaranteed current.

Mirrors test_renderer_buffer_eviction.py's approach: construct SceneRenderer()
directly (no GL context needed) and inject a recording GL stand-in so no test
here ever touches real GL.
"""

from __future__ import annotations

from bermake.model.texture import TextureLibrary
from bermake.viewport import scene_renderer as sr
from bermake.viewport.camera import Camera
from bermake.viewport.scene_renderer import _ENVIRONMENT_UNIFORMS, _LINE_UNIFORMS, SceneRenderer
from bermake.viewport.texture_cache import TextureCache


class _RecordingGL:
    """Stands in for the GL module. `__getattr__` no-ops everything not
    listed, including every GL_* constant lookup render() needs to OR
    together (glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) etc.) --
    a real int is not needed, only a value, so a fresh no-op callable per
    unknown name is fine EXCEPT for GL_* names, which must be stable ints
    reused across lookups within one recorder instance.
    """

    def __init__(self):
        self.calls: list[tuple] = []
        self._consts: dict[str, int] = {}

    def glDeleteTextures(self, ids):
        self.calls.append(("delete", tuple(ids) if hasattr(ids, "__iter__") else (ids,)))

    def __getattr__(self, name):
        if name.startswith("GL_"):
            return self._consts.setdefault(name, len(self._consts) + 1)

        def _noop(*args, **kwargs):
            self.calls.append((name,))
            return 0

        return _noop


def _renderer_with_fake_gl() -> tuple[SceneRenderer, _RecordingGL]:
    r = SceneRenderer()
    assert r._initialized is False  # no GL context needed to construct
    gl = _RecordingGL()
    r._texture_cache = TextureCache(gl=gl)
    return r, gl


def _make_renderable(r: SceneRenderer, gl: _RecordingGL, monkeypatch) -> None:
    """Enough state for render(camera, model=None) to run to completion
    without a real GL context. model=None skips the whole per-definition
    drawing block (materials/textures included), leaving only the environment
    pass (M7.7, on by default) and grid + axes line drawing, which needs a
    program + the two view/projection locations.

    render() calls the module-level `GL` (`from OpenGL import GL`), not
    `self._texture_cache._gl` -- the recorder has to replace that name too,
    or every GL call goes to the real (context-less) OpenGL binding and
    raises GLError(1282, "invalid operation") on the very first glClearColor.
    """
    monkeypatch.setattr(sr, "GL", gl)
    r._initialized = True
    r._line_program = 2
    r._line_locs = dict.fromkeys(_LINE_UNIFORMS, 0)
    r._environment_program = 3
    r._environment_locs = dict.fromkeys(_ENVIRONMENT_UNIFORMS, 0)


def test_evict_stale_textures_only_queues_the_slot_issues_no_gl_call():
    # The slot (MainWindow._on_after_undo_redo) runs with no current GL
    # context -- it must not call glDeleteTextures itself.
    r, gl = _renderer_with_fake_gl()
    textures = TextureLibrary()
    kept = textures.add("kept.png", b"x", "png", 1, 1, False)
    # Simulate an upload having happened for a texture that has since been
    # removed from the library (e.g. AddTextureCommand.undo()).
    r._texture_cache._ids[999] = 42
    r._texture_cache._ids[kept.id] = 7

    r.evict_stale_textures(textures)

    assert not [c for c in gl.calls if c[0] == "delete"]
    # Not yet applied -- still cached until the next render() flushes it.
    assert 999 in r._texture_cache.cached_ids()
    assert kept.id in r._texture_cache.cached_ids()


def test_render_flushes_a_queued_stale_texture_and_only_that_one(monkeypatch):
    r, gl = _renderer_with_fake_gl()
    _make_renderable(r, gl, monkeypatch)
    textures = TextureLibrary()
    kept = textures.add("kept.png", b"x", "png", 1, 1, False)
    r._texture_cache._ids[999] = 42
    r._texture_cache._ids[kept.id] = 7
    r.evict_stale_textures(textures)

    r.render(Camera(), model=None)

    assert ("delete", (42,)) in gl.calls
    assert 999 not in r._texture_cache.cached_ids()
    assert kept.id in r._texture_cache.cached_ids()  # not queued, must survive


def test_evict_stale_textures_also_queues_a_remembered_decode_failure(monkeypatch):
    r, gl = _renderer_with_fake_gl()
    _make_renderable(r, gl, monkeypatch)
    textures = TextureLibrary()
    r._texture_cache._failed.add(999)

    r.evict_stale_textures(textures)
    assert 999 in r._texture_cache.cached_ids()  # queued, not yet dropped

    r.render(Camera(), model=None)
    assert 999 not in r._texture_cache.cached_ids()


def test_release_all_textures_only_queues_the_slot_issues_no_gl_call():
    # The slot (MainWindow._reset_document) runs with no current GL context.
    r, gl = _renderer_with_fake_gl()
    r._texture_cache._ids[1] = 10
    r._texture_cache._ids[2] = 11

    r.release_all_textures()

    assert not [c for c in gl.calls if c[0] == "delete"]
    assert r._texture_cache.cached_ids() == frozenset({1, 2})


def test_render_flushes_a_queued_release_all(monkeypatch):
    r, gl = _renderer_with_fake_gl()
    _make_renderable(r, gl, monkeypatch)
    r._texture_cache._ids[1] = 10
    r._texture_cache._ids[2] = 11
    r.release_all_textures()

    r.render(Camera(), model=None)

    assert r._texture_cache.cached_ids() == frozenset()
    deleted = {i for c in gl.calls if c[0] == "delete" for i in c[1]}
    assert {10, 11} <= deleted


def test_a_queued_release_all_supersedes_an_earlier_queued_eviction(monkeypatch):
    r, gl = _renderer_with_fake_gl()
    _make_renderable(r, gl, monkeypatch)
    textures = TextureLibrary()
    r._texture_cache._ids[999] = 42
    r.evict_stale_textures(textures)  # queues 999 individually
    r._texture_cache._ids[5] = 50
    r.release_all_textures()  # then a document close supersedes it

    r.render(Camera(), model=None)

    assert r._texture_cache.cached_ids() == frozenset()
