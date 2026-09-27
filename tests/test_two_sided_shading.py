"""M7.5a Task 6: per-side material resolution feeding the shader."""

from __future__ import annotations

import re

import numpy as np
import pytest
from bermake.model.material import MaterialLibrary
from bermake.viewport import scene_renderer as sr
from bermake.viewport.environment import STUDIO
from bermake.viewport.face_batches import FaceBatch
from bermake.viewport.render_style import BACK_DEFAULT_COLOR, FaceStyle, RenderStyle
from bermake.viewport.scene_renderer import _PHONG_UNIFORMS, _load_shader_source, resolve_batch_sides

_BACK_UNIFORMS = (
    "u_material_ambient_back",
    "u_material_diffuse_back",
    "u_material_specular_back",
    "u_material_shininess_back",
    "u_alpha_back",
)

# `uniform <type> <name>;` — trailing comments are allowed after the semicolon.
_UNIFORM_DECL = re.compile(r"^\s*uniform\s+\w+\s+(\w+)\s*;", re.MULTILINE)


def _lib():
    lib = MaterialLibrary()
    red = lib.add_custom("Red", (0.8, 0.1, 0.1))
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))
    return lib, red.id, blue.id


def _ids(lib) -> frozenset[int]:
    """The library's translucent set, computed the way render() computes it.

    resolve_batch_sides requires it (M7.5b Task 6): a cutout texture is
    translucent in a way the material's own alpha cannot show, so the set is
    the only thing that can tell the resolver to blend. None of the fixtures
    here is textured, so this is just the alpha-driven set these tests always
    had implicitly.
    """
    return sr._translucent_ids(lib, None)


def _declared_uniforms(shader: str) -> set[str]:
    return set(_UNIFORM_DECL.findall(_load_shader_source(shader)))


def _frag_main_body() -> str:
    return _load_shader_source("phong.frag").split("void main()", 1)[1]


def test_the_two_sides_resolve_to_different_diffuse():
    lib, red, blue = _lib()
    batch = FaceBatch(front_material_id=red, back_material_id=blue, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    assert front.diffuse != back.diffuse
    assert front.diffuse[0] > front.diffuse[2]  # red-dominant
    assert back.diffuse[2] > back.diffuse[0]  # blue-dominant


def test_an_unpainted_back_uses_the_back_default_not_the_front_default():
    lib, red, _ = _lib()
    batch = FaceBatch(front_material_id=red, back_material_id=0, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    # The back must be the distinct blue-grey, not the front's red and not the
    # front default. A resolver that ignored side entirely would fail here.
    assert back.diffuse == pytest.approx(BACK_DEFAULT_COLOR, abs=1e-3)
    assert back.diffuse != pytest.approx(front.diffuse)


def test_both_sides_unpainted_still_differ():
    lib, _, _ = _lib()
    batch = FaceBatch(front_material_id=0, back_material_id=0, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    assert front.diffuse != pytest.approx(back.diffuse)


def test_a_translucent_side_makes_the_whole_batch_blend():
    lib, red, blue = _lib()
    lib.edit(blue, alpha=0.3)
    batch = FaceBatch(front_material_id=red, back_material_id=blue, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    # blend and depth_write are per-BATCH, taken from whichever side is more
    # translucent (correction 4), so the opaque front reports them too.
    assert front.blend is True
    assert back.blend is True
    assert front.depth_write is False
    assert back.depth_write is False
    # The opaque front must keep its own alpha; only blend/depth_write are
    # shared batch-wide (correction 4). A resolver that smeared the
    # batch-combined alpha into both sides would fail this.
    assert front.alpha == 1.0


def test_an_all_opaque_batch_writes_depth():
    lib, red, blue = _lib()
    batch = FaceBatch(front_material_id=red, back_material_id=blue, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    assert front.blend is False
    assert front.depth_write is True


def test_metallic_and_roughness_reach_the_resolved_uniforms():
    lib, red, _ = _lib()
    lib.edit(red, metallic=1.0, roughness=0.05)
    batch = FaceBatch(front_material_id=red, back_material_id=0, first=0, count=3)
    front, _ = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.SHADED),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    # a smooth metal: no diffuse, tight highlight
    assert front.diffuse == pytest.approx((0.0, 0.0, 0.0))
    assert front.shininess > 100.0


def test_wireframe_style_skips_both_sides():
    lib, red, blue = _lib()
    batch = FaceBatch(front_material_id=red, back_material_id=blue, first=0, count=3)
    front, back = resolve_batch_sides(
        batch,
        lib,
        RenderStyle(face_style=FaceStyle.WIREFRAME),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    assert front.draw_faces is False
    assert back.draw_faces is False


# --- Uniform wiring ---------------------------------------------------------
#
# The shader itself cannot run in this suite, so the three tests below close
# the loop on its plumbing by source inspection instead: every uniform the
# shader declares has a cached location, is actually read in main(), and is
# assigned a value during a face draw. A back uniform that is declared but
# never set (or set but never declared) would otherwise pass every test above
# and simply render wrong.


def test_every_phong_shader_uniform_has_a_cached_location():
    declared = _declared_uniforms("phong.vert") | _declared_uniforms("phong.frag")
    assert declared == set(_PHONG_UNIFORMS)


def test_every_fragment_uniform_is_read_in_main():
    body = _frag_main_body()
    for name in _declared_uniforms("phong.frag"):
        assert re.search(rf"\b{re.escape(name)}\b", body), (
            f"{name} is declared but never read; GL will optimize it out"
        )


def test_the_fragment_shader_declares_a_back_uniform_set():
    declared = _declared_uniforms("phong.frag")
    for name in _BACK_UNIFORMS:
        assert name in declared


def test_the_fragment_shader_flips_the_normal_for_back_facing_fragments():
    """Guards the back-face normal flip against silent removal.

    Culling is disabled, so without the flip a back-facing fragment shades
    against a normal pointing away from the viewer. There is no GL context in
    this suite, so a source assertion is the only available guard.
    """
    body = _frag_main_body()
    assert "gl_FrontFacing" in body
    assert re.search(r"if\s*\(\s*!\s*front\s*\)\s*N\s*=\s*-\s*N\s*;", body), (
        "back-facing normal is never flipped"
    )


class _GLRecorder:
    """Stand-in for the OpenGL module that records which uniform locations
    were written. Every attribute resolves to a no-op callable, so it also
    satisfies the GL constants the draw path passes straight through."""

    def __init__(self) -> None:
        self.written: set[int] = set()

    def __getattr__(self, name: str):
        def _call(*args, **kwargs):
            if name.startswith("glUniform") and args:
                self.written.add(int(args[0]))
            return 0

        return _call


def test_a_face_draw_sets_every_cached_phong_uniform(monkeypatch):
    """Kills a uniform that is cached but never assigned in the draw path.

    A missing `_set_vec3(locs["u_material_diffuse_back"], ...)` leaves the
    uniform at its GL default (zero) and every other test in this file still
    passes.
    """
    recorder = _GLRecorder()
    monkeypatch.setattr(sr, "GL", recorder)

    renderer = sr.SceneRenderer.__new__(sr.SceneRenderer)
    renderer._phong_program = 1
    renderer._phong_locs = {name: i for i, name in enumerate(_PHONG_UNIFORMS)}
    buf = sr._DefBuffers(face_vao=1, face_count=3)

    lib, red, blue = _lib()
    batch = FaceBatch(front_material_id=red, back_material_id=blue, first=0, count=3)
    front, back = resolve_batch_sides(
        batch, lib, RenderStyle(), bg=STUDIO.background, dimmed=False, translucent_ids=_ids(lib)
    )

    renderer._draw_definition_faces(
        buf,
        np.eye(4),
        np.eye(4),
        np.zeros(3),
        np.eye(4),
        front=front,
        back=back,
        first=0,
        count=3,
    )

    # M7.5b Task 6 exemption: the two sampler uniforms carry a texture-unit
    # index, which is program state rather than per-draw state, so they are set
    # once after linking instead. They are not left unguarded — the link-time
    # setter has its own test in tests/test_textured_shading.py
    # (test_linking_points_each_sampler_at_its_own_unit). Nothing else may join
    # this exemption without the same treatment.
    exempt = {renderer._phong_locs[n] for n in sr._PHONG_LINK_TIME_UNIFORMS}
    missing = {
        name
        for name, loc in renderer._phong_locs.items()
        if loc not in recorder.written and loc not in exempt
    }
    assert not missing, f"uniforms never set during a face draw: {sorted(missing)}"
