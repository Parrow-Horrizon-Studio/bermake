"""M7.5b Task 6: the shader samples the material's texture.

Both sides, not one. The controller's Task-4 ruling bakes a front UV at
attribute location 2 and a back UV at location 3, because two-sided shading
draws both sides in one pass selected by gl_FrontFacing — with a single UV
attribute an independent back placement would be physically unrepresentable.
The fragment shader therefore picks the UV SET with gl_FrontFacing exactly as
it already picks between the two uniform sets.
"""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import numpy as np
from bermake.model.material import MaterialLibrary
from bermake.model.texture import TextureLibrary
from bermake.viewport import scene_renderer as sr
from bermake.viewport.environment import STUDIO
from bermake.viewport.face_batches import FaceBatch
from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.viewport.scene_renderer import _PHONG_UNIFORMS, resolve_batch_sides

_SHADERS = Path(__file__).resolve().parents[1] / "python" / "bermake" / "viewport" / "shaders"
_VERT = (_SHADERS / "phong.vert").read_text(encoding="utf-8")
_FRAG = (_SHADERS / "phong.frag").read_text(encoding="utf-8")


def _declared(src: str) -> set[str]:
    return set(re.findall(r"^uniform\s+\w+\s+(\w+)\s*;", src, re.M))


def _frag_main() -> str:
    return _FRAG[_FRAG.index("void main()") :]


def _png(w: int, h: int, rgba: list[int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgba[y * w * 4 : (y + 1) * w * 4]) for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


_FIRST_GL_TEXTURE = 101  # the first id _GLRecorder.glGenTextures hands out

_OPAQUE_PNG = _png(1, 1, [255, 255, 255, 255])
_CUTOUT_PNG = _png(1, 1, [255, 255, 255, 0])


# --- Shader source ----------------------------------------------------------


def test_every_declared_uniform_has_a_cached_location():
    # Cross-artifact invariant: a uniform the shader declares but the Python
    # tuple omits is never located and never set, with no error anywhere.
    assert _declared(_VERT) | _declared(_FRAG) == set(_PHONG_UNIFORMS)


def test_the_fragment_shader_declares_both_samplers():
    assert "u_texture" in _declared(_FRAG)
    assert "u_texture_back" in _declared(_FRAG)


def test_the_fragment_shader_declares_a_has_texture_flag_per_side():
    # A sampler cannot be "unset" in GLSL, so an untextured material needs an
    # explicit flag rather than relying on a bound-or-not test.
    assert "u_has_texture" in _declared(_FRAG)
    assert "u_has_texture_back" in _declared(_FRAG)


def test_the_vertex_shader_takes_both_uvs_and_passes_them_on():
    # Locations 2 and 3 are the offsets _alloc_def_buffers already enables and
    # _upload_definition already bakes. A vertex shader that declared only one
    # would leave the back's placement with no way to reach the fragment stage.
    assert re.search(r"layout\s*\(\s*location\s*=\s*2\s*\)\s*in\s+vec2\s+in_uv\s*;", _VERT)
    assert re.search(r"layout\s*\(\s*location\s*=\s*3\s*\)\s*in\s+vec2\s+in_uv_back\s*;", _VERT)
    assert re.search(r"\bout\s+vec2\s+v_uv\s*;", _VERT)
    assert re.search(r"\bout\s+vec2\s+v_uv_back\s*;", _VERT)
    assert re.search(r"\bv_uv\s*=\s*in_uv\s*;", _VERT)
    assert re.search(r"\bv_uv_back\s*=\s*in_uv_back\s*;", _VERT)


def test_the_fragment_shader_receives_both_uvs():
    assert re.search(r"\bin\s+vec2\s+v_uv\s*;", _FRAG)
    assert re.search(r"\bin\s+vec2\s+v_uv_back\s*;", _FRAG)


def test_the_texture_multiplies_the_diffuse_rather_than_replacing_it():
    # Spec D5: the image TINTS base_color. A shader that assigns the sampled
    # colour over m_diffuse loses tinting and makes one texture unusable in two
    # colourways.
    body = _frag_main()
    assert re.search(r"m_diffuse\s*\*=", body) or re.search(r"m_diffuse\s*=\s*m_diffuse\s*\*", body)


def test_the_ambient_is_tinted_too():
    # Otherwise an unlit or dimly lit part of a textured face shows the flat
    # material colour while the lit part shows the image.
    body = _frag_main()
    assert re.search(r"m_ambient\s*\*=", body) or re.search(r"m_ambient\s*=\s*m_ambient\s*\*", body)


def test_the_sampled_alpha_multiplies_the_material_alpha():
    body = _frag_main()
    assert re.search(r"m_alpha\s*\*=", body) or re.search(r"m_alpha\s*=\s*m_alpha\s*\*", body)


def test_the_back_side_samples_the_back_sampler_with_the_back_uv():
    # A front-only implementation, or one that reads u_texture for both sides,
    # is the plausible bug here and it looks correct from the front. Sampling
    # the back texture with the FRONT's uv is the subtler variant of the same
    # bug, so the pairing is asserted rather than mere presence.
    body = _frag_main()
    assert re.search(r"texture\s*\(\s*u_texture_back\s*,\s*v_uv_back\s*\)", body)
    assert re.search(r"texture\s*\(\s*u_texture\s*,\s*v_uv\s*\)", body)


def test_the_texel_ternary_is_not_swapped():
    # The two assertions above only prove each sampler/uv pairing appears
    # SOMEWHERE. Swapping the whole ternary keeps both pairings intact and
    # shows the back's image on front faces:
    #
    #   front ? texture(u_texture_back, v_uv_back) : texture(u_texture, v_uv)
    #
    # That is the load-bearing invariant of this task, so the ternary is pinned
    # end to end rather than by its halves.
    body = _frag_main()
    assert re.search(
        r"front\s*\?\s*texture\s*\(\s*u_texture\s*,\s*v_uv\s*\)"
        r"\s*:\s*texture\s*\(\s*u_texture_back\s*,\s*v_uv_back\s*\)",
        body,
    )


def test_the_side_is_chosen_by_gl_front_facing_like_the_uniform_sets():
    body = _frag_main()
    assert re.search(r"\bfront\s*\?\s*u_has_texture\s*:\s*u_has_texture_back\b", body)


# --- Texture units ----------------------------------------------------------


class _GLRecorder:
    """Stand-in for the OpenGL module. Records the calls the texture path makes
    and lets every other GL entry point through as a no-op."""

    def __init__(self) -> None:
        self.uniforms: dict[int, object] = {}
        self.active_unit: object = None
        self.bound: list[tuple[object, int]] = []
        # Blending is enabled and then disabled again around each blended
        # draw, so the interesting fact is whether it was EVER enabled, not
        # what the flag reads at the end of the frame.
        self.blend_enables = 0
        self._consts: dict[str, int] = {}
        self._next_texture = _FIRST_GL_TEXTURE - 1

    def glGenTextures(self, _n):
        # A real, non-zero id. GL never allocates texture name 0, and 0 is
        # exactly what "untextured" means downstream, so a recorder handing
        # back 0 would let "resolved to nothing" pass as "resolved to a
        # texture" in every test that only checks `is not None`.
        self._next_texture += 1
        return self._next_texture

    def glEnable(self, cap):
        if cap == self.GL_BLEND:
            self.blend_enables += 1

    def glUniform1i(self, loc, value):
        self.uniforms[int(loc)] = int(value)

    def glUniform1f(self, loc, value):
        self.uniforms[int(loc)] = float(value)

    def glActiveTexture(self, unit):
        self.active_unit = unit

    def glBindTexture(self, _target, tid):
        self.bound.append((self.active_unit, int(tid)))

    def __getattr__(self, name):
        # Names beginning GL_ resolve to stable ints rather than callables:
        # render() ORs GL_COLOR_BUFFER_BIT with GL_DEPTH_BUFFER_BIT.
        if name.startswith("GL_"):
            return self._consts.setdefault(name, len(self._consts) + 1)

        def _call(*args, **kwargs):
            if name.startswith("glUniform") and args:
                self.uniforms.setdefault(int(args[0]), None)
            return 0

        return _call


def _renderer(recorder) -> sr.SceneRenderer:
    renderer = sr.SceneRenderer.__new__(sr.SceneRenderer)
    renderer._phong_program = 1
    renderer._phong_locs = {name: i for i, name in enumerate(_PHONG_UNIFORMS)}
    renderer._texture_cache = sr.TextureCache(gl=recorder)
    renderer._render_style = RenderStyle()
    return renderer


def test_the_samplers_are_bound_to_distinct_texture_units():
    # Both samplers default to unit 0, so leaving them unset makes the back
    # side silently show the front's image.
    assert sr._TEXTURE_UNIT_FRONT != sr._TEXTURE_UNIT_BACK


def test_linking_points_each_sampler_at_its_own_unit(monkeypatch):
    # Sampler values are program state, set once after linking rather than per
    # draw, so nothing in the per-draw path can guard them. This is that guard.
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)

    renderer._bind_sampler_units()

    locs = renderer._phong_locs
    assert recorder.uniforms[locs["u_texture"]] == sr._TEXTURE_UNIT_FRONT
    assert recorder.uniforms[locs["u_texture_back"]] == sr._TEXTURE_UNIT_BACK


def _draw(renderer, *, front_texture=None, back_texture=None):
    lib = MaterialLibrary()
    red = lib.add_custom("Red", (0.8, 0.1, 0.1))
    batch = FaceBatch(front_material_id=red.id, back_material_id=0, first=0, count=3)
    front, back = resolve_batch_sides(
        batch, lib, RenderStyle(), bg=STUDIO.background, dimmed=False, translucent_ids=frozenset()
    )
    renderer._draw_definition_faces(
        sr._DefBuffers(face_vao=1, face_count=3),
        np.eye(4),
        np.eye(4),
        np.zeros(3),
        np.eye(4),
        front=front,
        back=back,
        first=0,
        count=3,
        front_texture=front_texture,
        back_texture=back_texture,
    )


def test_a_textured_draw_binds_each_side_to_its_own_unit(monkeypatch):
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)

    _draw(renderer, front_texture=11, back_texture=22)

    assert (sr._TEXTURE_UNIT_ENUM[sr._TEXTURE_UNIT_FRONT], 11) in recorder.bound
    assert (sr._TEXTURE_UNIT_ENUM[sr._TEXTURE_UNIT_BACK], 22) in recorder.bound
    locs = renderer._phong_locs
    assert recorder.uniforms[locs["u_has_texture"]] == 1.0
    assert recorder.uniforms[locs["u_has_texture_back"]] == 1.0


def test_a_draw_leaves_unit_zero_active(monkeypatch):
    # TextureCache's own upload calls glBindTexture without choosing a unit
    # first, so a draw that returns with unit 1 current would upload the next
    # new texture into unit 1 and leave unit 0 holding a stale object.
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)

    _draw(renderer, front_texture=11, back_texture=22)

    assert recorder.active_unit == sr._TEXTURE_UNIT_ENUM[sr._TEXTURE_UNIT_FRONT]


def test_an_untextured_draw_flags_both_sides_off_and_unbinds(monkeypatch):
    # The flag is what makes an untextured material untextured: the sampler
    # still reads whatever object is in its unit, so a stale flag would show
    # the previous batch's image on an unpainted face.
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)

    _draw(renderer)

    locs = renderer._phong_locs
    assert recorder.uniforms[locs["u_has_texture"]] == 0.0
    assert recorder.uniforms[locs["u_has_texture_back"]] == 0.0
    assert {tid for _, tid in recorder.bound} == {0}


# --- Resolving a material to a GL texture -----------------------------------


def _painted(*, transparent: bool = False):
    materials = MaterialLibrary()
    textures = TextureLibrary()
    data = _CUTOUT_PNG if transparent else _OPAQUE_PNG
    tex = textures.add("t.png", data, "png", 1, 1, transparent)
    mat = materials.add_custom("Brick", (1.0, 1.0, 1.0))
    materials.edit(mat.id, texture_id=tex.id)
    return materials, textures, mat, tex


def test_a_painted_textured_material_resolves_to_the_cached_gl_id(monkeypatch):
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)
    materials, textures, mat, tex = _painted()

    gl_id = renderer._texture_for_material(materials, textures, mat.id)
    # The actual id, not merely "not None": 0 is a legal `is not None` value
    # and is precisely what every caller downstream reads as untextured.
    assert gl_id == _FIRST_GL_TEXTURE
    # Cached: the second resolve is the same object, not a second upload.
    assert renderer._texture_for_material(materials, textures, mat.id) == gl_id


def test_the_default_material_is_never_textured_even_when_it_carries_one(monkeypatch):
    # MaterialLibrary.edit permits editing id 0, so a texture CAN be put on
    # Default. The renderer must ignore it on both halves and agree with
    # itself. Id 0 means "unpainted", and every unpainted back face carries it,
    # so a cutout on Default entering translucent_ids would blend and stop
    # depth writes for essentially every batch in the model while nothing
    # rendered textured at all -- incoherent in both directions at once.
    #
    # A Default with NO texture would pass whether or not the guards exist, so
    # the fixture deliberately gives it a cutout.
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)
    materials = MaterialLibrary()
    textures = TextureLibrary()
    tex = textures.add("cut.png", _CUTOUT_PNG, "png", 1, 1, True)
    materials.edit(0, texture_id=tex.id)

    assert renderer._texture_for_material(materials, textures, 0) is None
    assert 0 not in sr._translucent_ids(materials, textures)


def test_an_untextured_material_resolves_to_no_texture(monkeypatch):
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)
    materials = MaterialLibrary()
    plain = materials.add_custom("Plain", (0.5, 0.5, 0.5))

    assert renderer._texture_for_material(materials, TextureLibrary(), plain.id) is None


def test_a_texture_id_pointing_at_nothing_renders_untextured(monkeypatch):
    # TextureLibrary.get returns None for an unknown id, deliberately unlike
    # MaterialLibrary.get. Spec 1.8: a document referencing a missing container
    # entry opens with that material untextured rather than being refused, so
    # the renderer must not substitute a placeholder or raise.
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)
    renderer = _renderer(recorder)
    materials = MaterialLibrary()
    ghost = materials.add_custom("Ghost", (1.0, 1.0, 1.0))
    materials.edit(ghost.id, texture_id=999)

    assert renderer._texture_for_material(materials, TextureLibrary(), ghost.id) is None


# --- Which styles show textures ---------------------------------------------


def test_shaded_shows_textures():
    assert sr.textures_visible(RenderStyle(face_style=FaceStyle.SHADED), tag_color=None) is True


def test_hidden_line_and_monochrome_show_no_textures():
    # Hidden Line fills with the background colour and Monochrome with
    # MONO_COLOR; both deliberately discard the painted colour. Multiplying
    # either by a texel puts the image back into a style whose whole point is
    # that it has none — and Hidden Line's flat fill comes out modulated by the
    # image rather than flat.
    for style in (FaceStyle.HIDDEN_LINE, FaceStyle.MONOCHROME):
        assert sr.textures_visible(RenderStyle(face_style=style), tag_color=None) is False


def test_color_by_tag_shows_no_textures():
    assert (
        sr.textures_visible(RenderStyle(face_style=FaceStyle.SHADED), tag_color=(1.0, 0.0, 0.0))
        is False
    )


# --- Translucency ------------------------------------------------------------


def test_a_cutout_texture_puts_its_opaque_material_in_the_translucent_pass():
    # Without this a cutout draws in the opaque pass, writes depth through its
    # own holes, and shows background instead of what is behind it.
    materials, textures, mat, _ = _painted(transparent=True)
    assert materials.get(mat.id).is_translucent is False
    assert mat.id in sr._translucent_ids(materials, textures)


def test_an_opaque_texture_leaves_its_material_in_the_opaque_pass():
    # The other half: treating every textured material as translucent would
    # pay the sort and lose depth writes for every brick wall in the model.
    materials, textures, mat, _ = _painted(transparent=False)
    assert mat.id not in sr._translucent_ids(materials, textures)


def test_a_translucent_alpha_still_counts_without_any_texture():
    materials = MaterialLibrary()
    glass = materials.add_custom("Glass", (0.8, 0.9, 1.0))
    materials.edit(glass.id, alpha=0.3)
    assert glass.id in sr._translucent_ids(materials, TextureLibrary())


def test_a_cutout_batch_blends_even_though_its_material_alpha_is_one():
    # Landing in the sorted pass is only half of it. GL blending is decided
    # from the material's own alpha, which a cutout leaves at 1.0, so without
    # this the shader's sampled alpha is written to a buffer that ignores it
    # and the holes come out solid.
    materials, textures, mat, _ = _painted(transparent=True)
    batch = FaceBatch(front_material_id=mat.id, back_material_id=0, first=0, count=3)
    ids = sr._translucent_ids(materials, textures)

    front, back = resolve_batch_sides(
        batch, materials, RenderStyle(), bg=STUDIO.background, dimmed=False, translucent_ids=ids
    )
    assert front.blend is True
    assert back.blend is True
    assert front.depth_write is False
    # The alpha uniform itself is untouched — the texel's alpha does the work.
    assert front.alpha == 1.0


def test_an_opaque_textured_batch_still_draws_unblended():
    materials, textures, mat, _ = _painted(transparent=False)
    batch = FaceBatch(front_material_id=mat.id, back_material_id=0, first=0, count=3)
    ids = sr._translucent_ids(materials, textures)

    front, _ = resolve_batch_sides(
        batch, materials, RenderStyle(), bg=STUDIO.background, dimmed=False, translucent_ids=ids
    )
    assert front.blend is False
    assert front.depth_write is True


# --- End to end through render() --------------------------------------------
#
# Everything above tests a seam. None of it would notice render() forgetting to
# pass `textures` or `translucent_ids` through to the two functions that need
# them — a wiring bug with no grep-visible symptom, in the one path that is
# only exercised on a GPU.


def _square(scene, z=0.0):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, z], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def _rendered(monkeypatch, *, transparent: bool, style: RenderStyle | None = None):
    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.scene_renderer import _ENVIRONMENT_UNIFORMS, _LINE_UNIFORMS, SceneRenderer

    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)

    model = Model()
    face = _square(model.root.mesh)
    tex = model.textures.add(
        "t.png", _CUTOUT_PNG if transparent else _OPAQUE_PNG, "png", 1, 1, transparent
    )
    mat = model.materials.add_custom("Painted", (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_id=tex.id)
    model.root.mesh.set_face_material(face, mat.id)

    renderer = SceneRenderer()
    renderer._initialized = True
    renderer._phong_program = 1
    renderer._line_program = 2
    renderer._phong_locs = {n: i for i, n in enumerate(_PHONG_UNIFORMS)}
    renderer._line_locs = {n: i for i, n in enumerate(_LINE_UNIFORMS)}
    renderer._environment_program = 3
    renderer._environment_locs = {n: i for i, n in enumerate(_ENVIRONMENT_UNIFORMS)}
    renderer._texture_cache = sr.TextureCache(gl=recorder)
    if style is not None:
        renderer.set_render_style(style)
    renderer.render(Camera(), model)
    return renderer, recorder


def test_render_carries_the_model_texture_all_the_way_to_the_draw(monkeypatch):
    renderer, recorder = _rendered(monkeypatch, transparent=False)

    locs = renderer._phong_locs
    assert recorder.uniforms[locs["u_has_texture"]] == 1.0
    assert (sr._TEXTURE_UNIT_ENUM[sr._TEXTURE_UNIT_FRONT], _FIRST_GL_TEXTURE) in recorder.bound


def test_render_does_not_texture_under_a_style_that_has_no_colour(monkeypatch):
    renderer, recorder = _rendered(
        monkeypatch, transparent=False, style=RenderStyle(face_style=FaceStyle.MONOCHROME)
    )

    assert recorder.uniforms[renderer._phong_locs["u_has_texture"]] == 0.0
    assert _FIRST_GL_TEXTURE not in {tid for _, tid in recorder.bound}


def test_render_blends_a_cutout_and_leaves_an_opaque_texture_unblended(monkeypatch):
    # The end of the chain _translucent_ids starts: render() has to reach
    # resolve_batch_sides with the set, or the cutout draws solid.
    _, cutout_gl = _rendered(monkeypatch, transparent=True)
    assert cutout_gl.blend_enables == 1

    _, opaque_gl = _rendered(monkeypatch, transparent=False)
    assert opaque_gl.blend_enables == 0
