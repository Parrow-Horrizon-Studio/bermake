"""M7.5a Task 11: per-tag colour and the Color-by-Tag mode."""

from __future__ import annotations

import pytest
from bermake.commands.tag_commands import SetTagColorCommand
from bermake.model.model import Model
from bermake.model.tag import TagLibrary
from bermake.viewport.environment import STUDIO
from bermake.viewport.render_style import RenderStyle
from bermake.viewport.scene_renderer import (
    _translucent_ids,
    resolve_batch_sides,
    resolve_tag_color,
    traverse_visible_tagged,
)


def _ids(lib) -> frozenset[int]:
    """The library's translucent set, computed the way render() computes it.

    resolve_batch_sides requires it (M7.5b Task 6): a cutout texture is
    translucent in a way the material's own alpha cannot show, so the set is
    the only thing that can tell the resolver to blend. Nothing here is
    textured, so this is the alpha-driven set these tests always had
    implicitly.
    """
    return _translucent_ids(lib, None)


def test_a_new_tag_gets_a_colour():
    lib = TagLibrary()
    t = lib.add("Walls")
    assert len(t.color) == 3
    assert all(0.0 <= c <= 1.0 for c in t.color)


def test_consecutive_tags_do_not_share_a_colour():
    lib = TagLibrary()
    a = lib.add("Walls")
    b = lib.add("Roof")
    assert a.color != b.color


def test_set_colour_replaces_it():
    lib = TagLibrary()
    t = lib.add("Walls")
    lib.set_color(t.id, (0.1, 0.2, 0.3))
    assert lib.get(t.id).color == (0.1, 0.2, 0.3)


def test_colour_round_trips_through_records():
    lib = TagLibrary()
    t = lib.add("Walls")
    lib.set_color(t.id, (0.1, 0.2, 0.3))
    rebuilt = TagLibrary.from_records(lib.to_records(), lib.next_id)
    assert rebuilt.get(t.id).color == (0.1, 0.2, 0.3)


def test_records_without_a_colour_still_load():
    # Schema <= 4 wrote no tag colour. Task 12's migration relies on this.
    # Asserting WHICH colour, not merely that there are three floats: keying
    # the fallback on record position alone hands Untagged (always record 0)
    # _PALETTE[0] -- bright red -- so every untagged face in a pre-Task-11
    # file reads as a real red tag under Color-by-Tag.
    rebuilt = TagLibrary.from_records(
        [
            {"id": 0, "name": "Untagged", "visible": True},
            {"id": 1, "name": "Walls", "visible": True},
        ],
        2,
    )
    assert rebuilt.get(0).color == TagLibrary._UNTAGGED_COLOR
    assert rebuilt.get(1).color == TagLibrary._PALETTE[0]


def test_untagged_consumes_no_palette_hue_after_a_load():
    # Untagged occupies a record slot but no hue, so the first tag added
    # after a load must continue the cycle at the next UNUSED entry --
    # seeding _next_palette_index from len(records) skips one.
    lib = TagLibrary()
    lib.add("Walls")
    rebuilt = TagLibrary.from_records(lib.to_records(), lib.next_id)
    assert rebuilt.add("Roof").color == TagLibrary._PALETTE[1]


def test_color_by_tag_defaults_off():
    assert RenderStyle().color_by_tag is False


def test_color_by_tag_is_independent_of_face_style():
    from bermake.viewport.render_style import FaceStyle

    s = RenderStyle(face_style=FaceStyle.MONOCHROME, color_by_tag=True)
    assert s.face_style is FaceStyle.MONOCHROME
    assert s.color_by_tag is True


# --- The palette: cycling and its wrap ---------------------------------


def test_the_palette_wraps_back_to_the_first_colour():
    # Kills an implementation that indexes _PALETTE without a modulo (crashes
    # or clamps at the last entry once the cycle runs out) rather than
    # deliberately wrapping. Also kills a clamp-at-the-end implementation:
    # a clamp would make the wrapped tag equal its IMMEDIATE predecessor too,
    # which the second assertion catches.
    lib = TagLibrary()
    n = len(TagLibrary._PALETTE)
    tags = [lib.add(f"Tag {i}") for i in range(n + 1)]
    assert tags[0].color == tags[n].color
    assert tags[n - 1].color != tags[n].color


# --- SetTagColorCommand: undo restores the right tag, not just A colour -


def test_set_tag_color_command_is_undoable_and_tag_specific():
    # A test that only checks "the colour came back" cannot tell a command
    # that recolors every tag, or one whose undo restores the wrong tag's
    # colour, from a correct one. Two tags make both bugs visible.
    lib = TagLibrary()
    a = lib.add("Walls")
    b = lib.add("Roof")
    a_before, b_before = a.color, b.color

    cmd = SetTagColorCommand(lib, a.id, (0.9, 0.1, 0.1))
    cmd.do(None)
    assert lib.get(a.id).color == (0.9, 0.1, 0.1)
    assert lib.get(b.id).color == b_before  # kills a command that recolors every tag

    cmd.undo(None)
    assert lib.get(a.id).color == a_before  # kills undo restoring the wrong tag
    assert lib.get(b.id).color == b_before


# --- Color-by-Tag reaching the draw: the decision seams --------------------
#
# resolve_tag_color decides WHICH colour applies; resolve_batch_sides applies
# it, by substituting it for the batch's materials BEFORE resolution (the fix
# for the final-review finding: a post-resolution diffuse patch left the
# material's own ambient and specular reaching the shader, so three faces on
# one tag painted three ways rendered three different colours).
# traverse_visible_tagged is the walk that supplies the tag id in the first
# place, since Model.traverse_visible()'s (definition, world) pairs do not
# carry the owning Instance. Together these three cover "does Color-by-Tag
# actually change what an occurrence draws with" without a GL context.


def _batch(front_material_id=0, back_material_id=0):
    from bermake.viewport.face_batches import FaceBatch

    return FaceBatch(
        front_material_id=front_material_id, back_material_id=back_material_id, first=0, count=3
    )


def test_resolve_tag_color_is_none_when_the_mode_is_off():
    # Kills an implementation that always returns a colour regardless of the
    # style flag -- a materials-bypassing override that never turns off.
    lib = TagLibrary()
    t = lib.add("Walls")
    assert resolve_tag_color(t.id, lib, RenderStyle(color_by_tag=False)) is None


def test_resolve_tag_color_returns_the_tags_own_colour():
    # Kills an implementation that ignores tag_id and returns some fixed or
    # default colour instead of the specific tag's.
    lib = TagLibrary()
    t = lib.add("Walls")
    lib.set_color(t.id, (0.3, 0.4, 0.5))
    color = resolve_tag_color(t.id, lib, RenderStyle(color_by_tag=True))
    assert color == (0.3, 0.4, 0.5)


def test_a_tag_colour_replaces_both_sides_diffuse():
    # The brief calls out a front-only override as a plausible, easy-to-miss
    # bug -- asserting only the front side would not catch it.
    from bermake.model.material import MaterialLibrary

    lib = MaterialLibrary()
    red = lib.add_custom("Red", (0.8, 0.1, 0.1))
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))
    color = (0.2, 0.55, 0.9)

    front, back = resolve_batch_sides(
        _batch(red.id, blue.id),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
        tag_color=color,
    )

    assert front.diffuse == pytest.approx(color, abs=1e-6)
    assert back.diffuse == pytest.approx(color, abs=1e-6)


def test_the_tag_colour_also_drives_ambient_and_specular_on_both_sides():
    # The final-review finding, inverted. The defect patched `diffuse` AFTER
    # resolve_face_pass, so ambient (and, for a metal, specular) still came
    # from the painted material -- the old test asserted ambient was UNCHANGED
    # and so pinned exactly that leak. The tag colour must reach every colour
    # term, identically on both sides, or two faces on one tag painted two
    # ways still render two different colours.
    from bermake.model.material import MaterialLibrary
    from bermake.viewport.render_style import phong_material_for

    lib = MaterialLibrary()
    gold = lib.add_custom("Gold", (0.9, 0.75, 0.2))
    lib.edit(gold.id, metallic=1.0)  # a metal: base_color reaches specular too
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))
    color = (0.2, 0.55, 0.9)
    expected = phong_material_for(color)

    front, back = resolve_batch_sides(
        _batch(gold.id, blue.id),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
        tag_color=color,
    )

    assert front.ambient == pytest.approx(expected.ambient, abs=1e-6)
    assert front.specular == pytest.approx(expected.specular, abs=1e-6)
    assert front.ambient == pytest.approx(back.ambient, abs=1e-6)
    assert front.specular == pytest.approx(back.specular, abs=1e-6)


def test_a_tag_colour_does_not_bypass_a_materials_alpha():
    # Alpha is not a colour. A translucent material under Color-by-Tag must
    # still blend, and each side keeps its OWN alpha -- kills a fix that
    # substitutes a whole opaque material (or one shared alpha) for both sides.
    from bermake.model.material import MaterialLibrary

    lib = MaterialLibrary()
    glass = lib.add_custom("Glass", (0.8, 0.1, 0.1))
    lib.edit(glass.id, alpha=0.5)

    front, back = resolve_batch_sides(
        _batch(glass.id, 0),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
        tag_color=(0.2, 0.55, 0.9),
    )

    assert front.alpha == pytest.approx(0.5)
    assert back.alpha == pytest.approx(1.0)
    assert front.blend and back.blend  # blend is draw-call state: both sides
    assert not front.depth_write


def test_no_tag_colour_leaves_the_materials_alone():
    # Kills an implementation that always overrides, ignoring the None sentinel
    # that means "Color-by-Tag is off".
    from bermake.model.material import MaterialLibrary

    lib = MaterialLibrary()
    red = lib.add_custom("Red", (0.8, 0.1, 0.1))
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))

    plain = resolve_batch_sides(
        _batch(red.id, blue.id),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
    )
    explicit = resolve_batch_sides(
        _batch(red.id, blue.id),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(lib),
        tag_color=None,
    )

    assert plain == explicit
    assert plain[0].diffuse[0] > plain[0].diffuse[2]  # still red-dominant
    assert plain[1].diffuse[2] > plain[1].diffuse[0]  # still blue-dominant


def _tagged_occurrences(model, definition):
    """The tag id of every visible occurrence of `definition`, in draw order."""
    return [tid for d, _, tid in traverse_visible_tagged(model) if d is definition]


def test_traverse_visible_tagged_carries_the_placing_instances_tag():
    m = Model()
    d = m.new_definition("Wall", is_group=True)
    inst = m.new_instance(d)
    inst.tag_id = 5
    m.root.children.append(inst)

    assert _tagged_occurrences(m, d) == [5]


def test_the_root_occurrence_is_reported_untagged():
    # The root definition is placed by no Instance, so the walk has no tag to
    # carry for it and must report UNTAGGED_ID -- the render loop draws the
    # root's own geometry and needs a real tag id for it, which is why this
    # is a documented value rather than a missing entry.
    m = Model()
    root_entry = next(iter(traverse_visible_tagged(m)))
    assert root_entry[0] is m.root
    assert root_entry[2] == TagLibrary.UNTAGGED_ID


def test_two_instances_of_one_definition_on_two_tags_resolve_to_two_colours():
    # THE per-definition bug: a component placed twice, on two tags, is the
    # ordinary case Color-by-Tag exists to serve. A map keyed by
    # id(definition) collapses both placements onto whichever instance was
    # visited last, so both draw in one colour.
    m = Model()
    chair = m.new_definition("Chair", is_group=False)
    walls = m.tags.add("Walls")
    roof = m.tags.add("Roof")
    m.tags.set_color(walls.id, (0.90, 0.25, 0.25))
    m.tags.set_color(roof.id, (0.95, 0.60, 0.15))
    for tag in (walls, roof):
        inst = m.new_instance(chair)
        inst.tag_id = tag.id
        m.root.children.append(inst)

    style = RenderStyle(color_by_tag=True)
    colors = [resolve_tag_color(tid, m.tags, style) for tid in _tagged_occurrences(m, chair)]

    assert colors == [(0.90, 0.25, 0.25), (0.95, 0.60, 0.15)]


def test_a_hidden_instance_does_not_supply_the_colour_a_visible_one_draws_with():
    # The aggravating half of the same bug: a walk of the whole tree sees
    # instances the draw never does, so a hidden placement visited later
    # could hand its colour to the visible one. The hidden instance is
    # appended SECOND on purpose -- last-write-wins is what it would win.
    m = Model()
    chair = m.new_definition("Chair", is_group=False)
    walls = m.tags.add("Walls")
    roof = m.tags.add("Roof")
    m.tags.set_color(walls.id, (0.90, 0.25, 0.25))
    m.tags.set_color(roof.id, (0.95, 0.60, 0.15))

    shown = m.new_instance(chair)
    shown.tag_id = walls.id
    hidden = m.new_instance(chair)
    hidden.tag_id = roof.id
    hidden.hidden = True
    m.root.children.extend([shown, hidden])

    style = RenderStyle(color_by_tag=True)
    colors = [resolve_tag_color(tid, m.tags, style) for tid in _tagged_occurrences(m, chair)]

    assert colors == [(0.90, 0.25, 0.25)]


def test_an_instance_on_a_hidden_tag_does_not_supply_a_colour_either():
    # Same aggravation via the other pruning rule traverse_visible applies.
    m = Model()
    chair = m.new_definition("Chair", is_group=False)
    walls = m.tags.add("Walls")
    roof = m.tags.add("Roof")
    m.tags.set_color(walls.id, (0.90, 0.25, 0.25))
    m.tags.set_color(roof.id, (0.95, 0.60, 0.15))
    m.tags.set_visible(roof.id, False)

    shown = m.new_instance(chair)
    shown.tag_id = walls.id
    off_tag = m.new_instance(chair)
    off_tag.tag_id = roof.id
    m.root.children.extend([shown, off_tag])

    style = RenderStyle(color_by_tag=True)
    colors = [resolve_tag_color(tid, m.tags, style) for tid in _tagged_occurrences(m, chair)]

    assert colors == [(0.90, 0.25, 0.25)]


def test_traverse_visible_tagged_visits_exactly_what_traverse_visible_does():
    # traverse_visible_tagged repeats traverse_visible's pruning rule inside
    # scene_renderer (so Model.traverse_visible keeps its two-tuple shape).
    # This pins the two walks together -- same entries, same order -- so the
    # copy cannot drift from the walk the rest of the renderer uses.
    m = Model()
    outer = m.new_definition("Outer", is_group=True)
    inner = m.new_definition("Inner", is_group=True)
    hidden_tag = m.tags.add("Hidden")
    m.tags.set_visible(hidden_tag.id, False)

    child = m.new_instance(inner)
    outer.children.append(child)
    a = m.new_instance(outer)
    b = m.new_instance(outer)
    b.hidden = True
    c = m.new_instance(inner)
    c.tag_id = hidden_tag.id
    m.root.children.extend([a, b, c])

    plain = [(id(d), w.tobytes()) for d, w in m.traverse_visible()]
    tagged = [(id(d), w.tobytes()) for d, w, _ in traverse_visible_tagged(m)]
    assert tagged == plain


def test_the_full_chain_from_instance_tag_to_resolved_diffuse():
    # An end-to-end composition of the three seams above: an instance's tag
    # colour reaches BOTH sides' diffuse when Color-by-Tag is on, and neither
    # side is touched when it is off -- the actual behaviour a GL draw call
    # would show, minus the GL context.
    m = Model()
    d = m.new_definition("Wall", is_group=True)
    inst = m.new_instance(d)
    m.root.children.append(inst)
    walls = m.tags.add("Walls")
    m.tags.set_color(walls.id, (0.9, 0.2, 0.2))
    inst.tag_id = walls.id

    painted = m.materials.add_custom("Brick", (0.7, 0.27, 0.22))
    batch = _batch(painted.id, 0)

    (tag_id,) = _tagged_occurrences(m, d)

    off_color = resolve_tag_color(tag_id, m.tags, RenderStyle(color_by_tag=False))
    off_front, off_back = resolve_batch_sides(
        batch,
        m.materials,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(m.materials),
        tag_color=off_color,
    )
    assert off_front.diffuse == pytest.approx((0.7, 0.27, 0.22), abs=1e-6)

    on_color = resolve_tag_color(tag_id, m.tags, RenderStyle(color_by_tag=True))
    on_front, on_back = resolve_batch_sides(
        batch,
        m.materials,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=False,
        translucent_ids=_ids(m.materials),
        tag_color=on_color,
    )
    assert on_front.diffuse == pytest.approx((0.9, 0.2, 0.2), abs=1e-6)
    assert on_back.diffuse == pytest.approx((0.9, 0.2, 0.2), abs=1e-6)


# --- Color-by-Tag through the real render() path ---------------------------
#
# The seam tests above prove resolve_batch_sides applies the colour; these
# prove render() actually routes the tag colour into it, for every batch, in
# both passes. GL is replaced by a recorder (the pattern from
# test_translucent_pass.py) rather than a real context, so these run in CI
# where the offscreen platform is forced and hardware GL may be absent.


class _RecordingGL:
    """Stand-in for the OpenGL module. GL_* names resolve to stable ints
    (render() ORs GL_COLOR_BUFFER_BIT with GL_DEPTH_BUFFER_BIT); everything
    else is a no-op returning 0, so glGenVertexArrays hands back zero handles
    that _DefBuffers.release() already guards against. Every call is recorded
    so blend / depth-mask state is assertable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self._consts: dict[str, int] = {}

    def __getattr__(self, name: str):
        if name.startswith("GL_"):
            return self._consts.setdefault(name, len(self._consts) + 1)

        def _call(*args, **kwargs):
            self.calls.append((name, args))
            return 0

        return _call


class _TagRenderHarness:
    """One group holding three coplanar-free triangles on ONE tag: unpainted,
    Brick Red, Forest Green. Driven through the real SceneRenderer.render().

    Three faces, one tag, three different paints is exactly the configuration
    the final-review finding measured: under Color-by-Tag they must resolve to
    one colour, and a fixture with only painted faces (or only one paint) would
    not show the leak.
    """

    TAG_COLOR = (0.20, 0.55, 0.90)

    def __init__(self, monkeypatch, *, translucent_brick=False) -> None:
        import numpy as np
        from bermake.viewport import scene_renderer
        from bermake.viewport.camera import Camera
        from bermake.viewport.scene_renderer import (
            _ENVIRONMENT_UNIFORMS,
            _LINE_UNIFORMS,
            _PHONG_UNIFORMS,
            SceneRenderer,
        )

        self.gl = _RecordingGL()
        monkeypatch.setattr(scene_renderer, "GL", self.gl)

        self.model = Model()
        lib = self.model.materials
        self.brick = next(m for m in lib.materials() if m.name == "Brick Red").id
        self.forest = next(m for m in lib.materials() if m.name == "Forest Green").id
        if translucent_brick:
            lib.edit(self.brick, alpha=0.5)

        def triangle(scene, x):
            ids = [
                scene.add_vertex(np.array([x, 0.0, 0.0])),
                scene.add_vertex(np.array([x, 1.0, 0.0])),
                scene.add_vertex(np.array([x, 0.0, 1.0])),
            ]
            return scene.add_face_from_loop(ids)

        self.wall = self.model.new_definition("Wall", is_group=True)
        self.unpainted_face = triangle(self.wall.mesh, 0.0)
        self.wall.mesh.set_face_material(triangle(self.wall.mesh, 1.0), self.brick)
        self.wall.mesh.set_face_material(triangle(self.wall.mesh, 2.0), self.forest)

        tag = self.model.tags.add("Blue")
        self.model.tags.set_color(tag.id, self.TAG_COLOR)
        inst = self.model.new_instance(self.wall)
        inst.tag_id = tag.id
        self.model.root.children.append(inst)

        self.renderer = SceneRenderer()
        self.renderer._initialized = True
        self.renderer._phong_program = 1
        self.renderer._line_program = 2
        self.renderer._phong_locs = {n: i for i, n in enumerate(_PHONG_UNIFORMS)}
        self.renderer._line_locs = {n: i for i, n in enumerate(_LINE_UNIFORMS)}
        self.renderer._environment_program = 3
        self.renderer._environment_locs = {n: i for i, n in enumerate(_ENVIRONMENT_UNIFORMS)}

        self._passes: list[tuple] = []
        real_faces = self.renderer._draw_definition_faces

        def faces(buf, *a, front, back, first=0, count=None, **kw):
            self._passes.append((front, back))
            real_faces(buf, *a, front=front, back=back, first=first, count=count, **kw)

        monkeypatch.setattr(self.renderer, "_draw_definition_faces", faces)

        self.camera = Camera(
            position=np.array([-10.0, 0.5, 0.5], dtype=np.float32),
            target=np.array([1.0, 0.5, 0.5], dtype=np.float32),
        )

    def render(self, style):
        """Every (front, back) pass render() drew this frame, one per batch."""
        self._passes.clear()
        self.gl.calls.clear()
        self.renderer.set_render_style(style)
        self.renderer.render(self.camera, self.model)
        return list(self._passes)


def _colour_terms(pass_):
    return (pass_.diffuse, pass_.ambient, pass_.specular)


def test_three_differently_painted_faces_on_one_tag_draw_in_one_colour(monkeypatch):
    # THE final-review finding. Unpainted, Brick Red and Forest Green on one
    # blue tag: diffuse, ambient AND specular must agree across all three, on
    # both sides. The pre-fix renderer agreed on diffuse only, so asserting
    # diffuse alone would still pass against the defect.
    h = _TagRenderHarness(monkeypatch)

    off = h.render(RenderStyle(color_by_tag=False))
    assert len({_colour_terms(f) for f, _ in off}) == 3, "fixture must paint three ways"

    on = h.render(RenderStyle(color_by_tag=True))
    assert len(on) == 3
    assert len({_colour_terms(f) for f, _ in on}) == 1
    assert len({_colour_terms(b) for _, b in on}) == 1
    front, back = on[0]
    assert _colour_terms(front) == _colour_terms(back)
    assert front.diffuse == pytest.approx(_TagRenderHarness.TAG_COLOR, abs=1e-6)


def test_hidden_line_keeps_its_flat_background_fill_under_color_by_tag(monkeypatch):
    # The defect's second half: patching diffuse after resolution clobbered
    # Hidden Line's deliberate (0, 0, 0) unlit fill with the tag colour, so
    # faces rendered lit instead of filled. Hidden Line must look exactly the
    # same whether the mode is on or off.
    from bermake.viewport.render_style import FaceStyle

    h = _TagRenderHarness(monkeypatch)

    off = h.render(RenderStyle(face_style=FaceStyle.HIDDEN_LINE, color_by_tag=False))
    on = h.render(RenderStyle(face_style=FaceStyle.HIDDEN_LINE, color_by_tag=True))

    assert [_colour_terms(f) for f, _ in on] == [_colour_terms(f) for f, _ in off]
    for f, b in on:
        assert f.diffuse == (0.0, 0.0, 0.0)
        assert b.diffuse == (0.0, 0.0, 0.0)
        assert f.specular == (0.0, 0.0, 0.0)


def test_monochrome_keeps_mono_color_under_color_by_tag(monkeypatch):
    # Monochrome's whole job is one uniform grey. Color-by-Tag must not turn it
    # into a tag-coloured shade -- but ambient still follows the tag, which is
    # what distinguishes this from Hidden Line's full bypass.
    from bermake.viewport.render_style import MONO_COLOR, FaceStyle, phong_material_for

    h = _TagRenderHarness(monkeypatch)
    on = h.render(RenderStyle(face_style=FaceStyle.MONOCHROME, color_by_tag=True))

    expected = phong_material_for(_TagRenderHarness.TAG_COLOR)
    for f, b in on:
        assert f.diffuse == MONO_COLOR
        assert b.diffuse == MONO_COLOR
        assert f.ambient == pytest.approx(expected.ambient, abs=1e-6)


def test_a_translucent_face_still_blends_under_color_by_tag(monkeypatch):
    # Alpha is deliberately NOT bypassed. Through the real render path: the
    # translucent batch keeps alpha 0.5, turns GL_BLEND on, masks depth writes,
    # and stays in the second (translucent) pass.
    h = _TagRenderHarness(monkeypatch, translucent_brick=True)
    on = h.render(RenderStyle(color_by_tag=True))

    blended = [f for f, _ in on if f.blend]
    assert len(blended) == 1
    assert blended[0].alpha == pytest.approx(0.5)
    assert not blended[0].depth_write
    assert blended[0].diffuse == pytest.approx(_TagRenderHarness.TAG_COLOR, abs=1e-6)
    # The translucent batch is drawn last: pass 2 follows every opaque batch.
    assert on[-1][0].blend

    names = [n for n, _ in h.gl.calls]
    assert "glEnable" in names and "glDepthMask" in names
    enabled = {args[0] for n, args in h.gl.calls if n == "glEnable" and args}
    assert h.gl.GL_BLEND in enabled
    assert (h.gl.GL_FALSE,) in [args for n, args in h.gl.calls if n == "glDepthMask"]


# --- TagsPage: recoloring goes through the command stack, and is undoable --
#
# Mirrors tests/test_materials_editor.py's pattern for MaterialsPage: patch
# QColorDialog.getColor (never let the modal dialog actually open) and drive
# the page's own handler.


def test_editing_tag_color_goes_through_the_command_stack(main_window, monkeypatch):
    from bermake.ui import tags_page as tags_page_module
    from PySide6.QtGui import QColor

    win = main_window
    lib = win._model.tags
    walls = lib.add("Walls")
    win._tags_page.refresh()
    win._tags_page.set_active(walls.id)
    depth = len(win._command_stack._undo)

    monkeypatch.setattr(
        tags_page_module.QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(10, 20, 30)),
    )

    win._tags_page._on_edit_color()

    assert len(win._command_stack._undo) == depth + 1, "recoloring must be undoable"
    assert lib.get(walls.id).color == pytest.approx((10 / 255, 20 / 255, 30 / 255), abs=1e-6)

    win._command_stack.undo()
    assert lib.get(walls.id).color != pytest.approx((10 / 255, 20 / 255, 30 / 255), abs=1e-6)


def test_picking_the_same_tag_colour_pushes_no_undo_entry(main_window, monkeypatch):
    from bermake.ui import tags_page as tags_page_module
    from PySide6.QtGui import QColor

    win = main_window
    lib = win._model.tags
    walls = lib.add("Walls")
    win._tags_page.refresh()
    win._tags_page.set_active(walls.id)
    depth = len(win._command_stack._undo)

    r, g, b = (round(c * 255) for c in lib.get(walls.id).color)
    monkeypatch.setattr(
        tags_page_module.QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(r, g, b)),
    )

    win._tags_page._on_edit_color()

    assert len(win._command_stack._undo) == depth, "an unchanged colour is not an edit"


def test_a_library_recolor_repaints_the_viewport(main_window):
    # library_changed marks the document dirty and retitles the window, which
    # is not enough: a recolored tag under Color-by-Tag (and a recolored
    # material under any face style) changes what is on screen right now, so
    # the viewport must repaint instead of waiting for an incidental one.
    # PySide's disconnect() only warns ("Failed to disconnect ...") when the
    # connection is absent, so the warning is promoted to an error -- that
    # promotion IS the assertion. Each page is reconnected immediately.
    import warnings

    win = main_window
    for page in (win._tags_page, win._materials_page):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            page.library_changed.disconnect(win._viewport.update)
        page.library_changed.connect(win._viewport.update)


def test_edit_color_is_a_noop_without_an_injected_command_stack(qtbot, monkeypatch):
    # Kills a version that crashes (or silently mutates the library without
    # undo support) when used standalone -- the shape every pre-Task-11
    # TagsPage(lib) test in test_tags_page.py already constructs.
    #
    # QColorDialog is patched even though the None guard should return first:
    # unpatched, a regressed guard opens a real modal and HANGS the suite
    # instead of failing it. Patched, the same regression fails this
    # assertion in milliseconds.
    from bermake.ui import tags_page as tags_page_module
    from bermake.ui.tags_page import TagsPage
    from PySide6.QtGui import QColor

    monkeypatch.setattr(
        tags_page_module.QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(10, 20, 30)),
    )

    lib = TagLibrary()
    walls = lib.add("Walls")
    page = TagsPage(lib)
    qtbot.addWidget(page)
    page.set_active(walls.id)
    before = lib.get(walls.id).color

    page._on_edit_color()  # no command_stack injected -> must not raise or mutate

    assert lib.get(walls.id).color == before


def test_the_dim_pass_still_dims_under_a_tag_colour():
    # M7.5b (#107 item 4). scene_renderer's resolve_tag_color docstring claims
    # "the dim pass still dims", but nothing asserted it: every other seam test
    # in this file passes dimmed=False. The behaviour genuinely changed during
    # M7.5a's final fix wave -- the old post-resolution patch let the tag colour
    # win over the dim colours, and resolving FROM the tag colour instead means
    # _DIM_DIFFUSE/_DIM_AMBIENT correctly take precedence again.
    #
    # Red-probed against two opposite mutations, both of which it kills:
    #   - render_style dropping the dim override (`ambient, diffuse =
    #     fu.ambient, fu.diffuse`) lets the tag colour win, and the
    #     _DIM_AMBIENT/_DIM_DIFFUSE assertions fail.
    #   - resolve_batch_sides ignoring its tag_color argument resolves from the
    #     painted materials instead, and the specular assertion fails.
    # Note the second mutation has to be in resolve_batch_sides itself, not in
    # resolve_tag_color: this test passes tag_color directly, so stubbing the
    # resolver never reaches it.
    from bermake.model.material import MaterialLibrary
    from bermake.viewport.render_style import phong_material_for
    from bermake.viewport.scene_renderer import _DIM_AMBIENT, _DIM_DIFFUSE

    lib = MaterialLibrary()
    gold = lib.add_custom("Gold", (0.9, 0.75, 0.2))
    lib.edit(gold.id, metallic=1.0)
    blue = lib.add_custom("Blue", (0.1, 0.1, 0.8))
    color = (0.2, 0.55, 0.9)

    front, back = resolve_batch_sides(
        _batch(gold.id, blue.id),
        lib,
        RenderStyle(),
        bg=STUDIO.background,
        dimmed=True,
        translucent_ids=_ids(lib),
        tag_color=color,
    )

    for side in (front, back):
        # Dim owns the colour terms it overrides.
        assert side.ambient == pytest.approx(_DIM_AMBIENT)
        assert side.diffuse == pytest.approx(_DIM_DIFFUSE)
        # But the tag still drives what dim does not override, so the batch was
        # resolved FROM the tag colour rather than from either material.
        assert side.specular == pytest.approx(phong_material_for(color).specular)
