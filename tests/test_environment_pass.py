"""The sky/ground pass: shader plumbing by source inspection, plus the skip rule.

The fragment shader's horizon arithmetic runs on the GPU and there is nothing
here to call, so most of these tests pin the seams instead: the Python uniform
tuple against the GLSL declarations, and the predicate that decides whether the
pass runs at all.

Fix round 1 adds a render()-level check of the one invariant no source
inspection can see: that the pass actually restores GL_DEPTH_TEST and the
depth mask before returning, and that it runs before the grid. That needs a
recording GL stand-in rather than source inspection, since it is a sequence of
calls, not a piece of text.
"""

import re

import numpy as np

from bermake.tools.tool import ToolOverlay
from bermake.viewport import scene_renderer as sr
from bermake.viewport.camera import Camera
from bermake.viewport.environment import PLAIN_WHITE, SKY_AND_GROUND, STUDIO, environment_pass_needed
from bermake.viewport.scene_renderer import (
    _ENVIRONMENT_UNIFORMS,
    _LINE_UNIFORMS,
    SceneRenderer,
    _load_shader_source,
    _snap_marker_halo,
    _snap_marker_vertices,
)
from bermake.viewport.snap_engine import SnapKind

_UNIFORM_DECL = re.compile(r"^\s*uniform\s+\w+\s+(\w+)\s*;", re.MULTILINE)
_LINE_COMMENT = re.compile(r"//.*")


def _declared(shader: str) -> set[str]:
    return set(_UNIFORM_DECL.findall(_load_shader_source(shader)))


def _frag_main_body() -> str:
    return _load_shader_source("environment.frag").split("void main()", 1)[1]


def _strip_line_comments(text: str) -> str:
    """Drop `//` comments so a uniform named only in prose does not count
    as read.
    """
    return _LINE_COMMENT.sub("", text)


def test_every_environment_uniform_has_a_cached_location():
    """The tuple the renderer caches locations from must match the GLSL exactly.

    Discriminates: drop "u_ground_opacity" from _ENVIRONMENT_UNIFORMS and this
    fails; the ground would then draw at whatever opacity the uninitialised
    uniform held.
    """
    declared = _declared("environment.vert") | _declared("environment.frag")
    assert declared == set(_ENVIRONMENT_UNIFORMS)


def test_every_fragment_uniform_is_read_in_main():
    """A declared-and-set uniform that main() ignores renders wrong silently.

    Comments are stripped first: a uniform named only in a `//` comment inside
    main() would otherwise count as read while never actually being touched.
    """
    body = _strip_line_comments(_frag_main_body())
    for name in _declared("environment.frag"):
        assert name in body, f"{name} is declared but never read in main()"


def test_the_fragment_shader_splits_on_the_ray_z_component():
    """Z-up: sky is ray.z > 0, ground is ray.z < 0, and the horizon is z == 0.

    Pinned by source because getting this wrong (using .y, the Y-up convention)
    produces a horizon perpendicular to the real one, which no other test here
    can see.
    """
    body = _frag_main_body()
    assert "ray.z" in body
    assert "ray.y" not in body

    # Fix round 1: pin the pairing between each half of the sky and its own
    # enable flag, not merely that ray.z appears somewhere in the file. Each
    # branch is captured from its "if"/"else if" condition through its own
    # closing brace, so a flag swapped onto the wrong condition is caught even
    # though the file would still mention both flags overall.
    sky_branch = re.search(r"if\s*\(\s*ray\.z\s*>\s*0\.0.*?\{.*?\}", body, re.DOTALL)
    ground_branch = re.search(r"else\s+if\s*\(\s*ray\.z\s*<\s*0\.0.*?\{.*?\}", body, re.DOTALL)
    assert sky_branch is not None, "no ray.z > 0.0 branch found"
    assert ground_branch is not None, "no ray.z < 0.0 branch found"
    assert "u_sky_enabled" in sky_branch.group(0)
    assert "u_ground_enabled" in ground_branch.group(0)


def test_the_pass_is_skipped_when_both_halves_are_disabled():
    assert environment_pass_needed(SKY_AND_GROUND) is True
    assert environment_pass_needed(PLAIN_WHITE) is False
    assert environment_pass_needed(STUDIO) is False


# --- render()-level check: depth state restore and draw order ---------------
#
# Reuses the (name, args)-recording _RecordingGL shape from
# tests/test_tag_color.py: GL_* names resolve to stable ints per instance and
# every other call is appended as (name, args), so both call order and call
# arguments are assertable. That is what distinguishes the environment quad's
# glDrawArrays(GL_TRIANGLES, ...) from the grid's glDrawArrays(GL_LINES, ...) --
# both are the same call name, so ordering by name alone could not tell them
# apart.


class _RecordingGL:
    """Stand-in for the OpenGL module. GL_* names resolve to stable ints;
    every other call is recorded as (name, args) so both order and arguments
    are assertable.
    """

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


def _render_with_recording_gl(monkeypatch, environment) -> _RecordingGL:
    """Drive SceneRenderer.render() through a recording GL, no real context.

    model=None skips every per-definition draw, leaving only the environment
    pass and the grid/axes lines -- enough to see the environment pass's own
    call sequence and where it sits relative to the grid's draw.
    """
    recorder = _RecordingGL()
    monkeypatch.setattr(sr, "GL", recorder)

    renderer = SceneRenderer()
    renderer._initialized = True
    renderer._environment = environment
    renderer._line_program = 2
    renderer._line_locs = dict.fromkeys(_LINE_UNIFORMS, 0)
    renderer._environment_program = 3
    renderer._environment_locs = dict.fromkeys(_ENVIRONMENT_UNIFORMS, 0)

    renderer.render(Camera(), model=None)
    return recorder


def _index_from(calls: list[tuple[str, tuple]], start: int, name: str, arg) -> int | None:
    """The first index at or after `start` where `name` was called with `arg`
    as its first positional argument, or None.
    """
    for i in range(start, len(calls)):
        call_name, args = calls[i]
        if call_name == name and args and args[0] == arg:
            return i
    return None


def test_the_environment_pass_restores_depth_state_and_draws_before_the_grid(monkeypatch):
    """The pass runs with depth test off and depth writes masked, then must
    restore both before returning: glDisable(GL_DEPTH_TEST), glDepthMask(FALSE),
    the quad's glDrawArrays, glDepthMask(TRUE), glEnable(GL_DEPTH_TEST), in that
    order, all before the grid's own glDrawArrays.

    If the pass ever left depth state behind, every pass after it in the frame
    -- starting with the grid drawn right after it -- would draw wrong, and no
    other test here would notice.
    """
    recorder = _render_with_recording_gl(monkeypatch, SKY_AND_GROUND)
    calls = recorder.calls

    depth_test = recorder.GL_DEPTH_TEST
    gl_false = recorder.GL_FALSE
    gl_true = recorder.GL_TRUE
    triangles = recorder.GL_TRIANGLES
    lines = recorder.GL_LINES

    i_disable = _index_from(calls, 0, "glDisable", depth_test)
    assert i_disable is not None, "the pass never disables GL_DEPTH_TEST"

    i_mask_false = _index_from(calls, i_disable, "glDepthMask", gl_false)
    assert i_mask_false is not None, "the pass never masks depth writes off"

    i_draw_tri = _index_from(calls, i_mask_false, "glDrawArrays", triangles)
    assert i_draw_tri is not None, "the environment quad is never drawn"

    i_mask_true = _index_from(calls, i_draw_tri, "glDepthMask", gl_true)
    assert i_mask_true is not None, "depth writes are never re-enabled"

    i_enable = _index_from(calls, i_mask_true, "glEnable", depth_test)
    assert i_enable is not None, "GL_DEPTH_TEST is never re-enabled"

    i_draw_lines = _index_from(calls, 0, "glDrawArrays", lines)
    assert i_draw_lines is not None, "the grid is never drawn"

    assert i_disable < i_mask_false < i_draw_tri < i_mask_true < i_enable < i_draw_lines


def test_the_pass_is_skipped_at_render_time_when_both_halves_are_disabled(monkeypatch):
    """Pins the skip rule where it actually matters: at render(), not only at
    the environment_pass_needed predicate tested above. With PLAIN_WHITE,
    render() must never draw the environment quad at all.
    """
    recorder = _render_with_recording_gl(monkeypatch, PLAIN_WHITE)
    triangles = recorder.GL_TRIANGLES
    environment_draws = [
        (name, args)
        for name, args in recorder.calls
        if name == "glDrawArrays" and args and args[0] == triangles
    ]
    assert environment_draws == []


# --- Snap-marker halo (Task 6b) ---------------------------------------------
#
# The visual pass reported the snap marker is hard to see on light
# backgrounds; measured luminance deltas as low as 0.017 confirmed it is
# invisible, not merely faint. The fix draws the marker geometry twice at the
# single site it is already drawn: a slightly larger halo in the
# environment's own edge colour first, then the marker unchanged on top, so
# every marker kind reads against every background without the marker
# colours themselves changing meaning.


def _draw_overlay_with_recording_gl(monkeypatch, environment, overlay) -> _RecordingGL:
    """Drive SceneRenderer._draw_tool_overlay() through a recording GL, no
    real GL context, mirroring _render_with_recording_gl above but scoped to
    just the tool-overlay draw so the marker block's calls aren't mixed in
    with a full frame's worth of grid/axis calls.
    """
    recorder = _RecordingGL()
    monkeypatch.setattr(sr, "GL", recorder)

    renderer = SceneRenderer()
    renderer._line_program = 2
    renderer._line_locs = dict.fromkeys(_LINE_UNIFORMS, 0)
    renderer._environment = environment

    view = np.eye(4, dtype=np.float32)
    projection = np.eye(4, dtype=np.float32)
    renderer._draw_tool_overlay(overlay, view, projection)
    return recorder


def _snap_marker_overlay() -> ToolOverlay:
    """A bare overlay carrying only a snap marker -- no rubber band, so the
    marker block's calls are the only glDrawArrays(GL_LINES, ...) calls made.
    """
    return ToolOverlay(
        rubber_band_segments=np.zeros((0, 3), dtype=np.float32),
        rubber_band_color=(1.0, 1.0, 1.0),
        snap_marker_position=np.array([1.0, 2.0, 0.0], dtype=np.float32),
        snap_marker_color=(0.70, 0.70, 0.70),
        snap_marker_kind=int(SnapKind.GRID),
    )


def _line_draws(recorder: _RecordingGL) -> list[tuple]:
    lines = recorder.GL_LINES
    return [
        (name, args)
        for name, args in recorder.calls
        if name == "glDrawArrays" and args and args[0] == lines
    ]


def test_the_snap_marker_draws_a_halo_behind_itself(monkeypatch):
    """The marker block must record two GL_LINES draws, the halo first.

    Discriminates: with the halo draw removed, only one GL_LINES draw remains
    for the marker block and this fails.
    """
    recorder = _draw_overlay_with_recording_gl(monkeypatch, SKY_AND_GROUND, _snap_marker_overlay())
    assert len(_line_draws(recorder)) == 2, "expected a halo draw and a marker draw"


def test_the_halo_uses_the_environments_edge_colour(monkeypatch):
    """The halo's uploaded vertex colours must equal self._environment.edge_color.

    That colour, not a computed one, is the point of the design: the contrast
    floor (tests/test_environment.py) already guarantees edge_color differs
    from every enabled backdrop of its own preset by at least 0.35 luminance.

    Discriminates: with the halo draw removed, the first buffer upload is the
    marker's own snap_marker_color (0.70, 0.70, 0.70), not STUDIO's edge_color
    (0.85, 0.85, 0.85), and this fails.
    """
    recorder = _draw_overlay_with_recording_gl(monkeypatch, STUDIO, _snap_marker_overlay())
    buffer_uploads = [args for name, args in recorder.calls if name == "glBufferData"]
    assert len(buffer_uploads) >= 1, "no buffer was uploaded for the marker block"
    halo_data = buffer_uploads[0][2]  # (target, size, data, usage)
    halo_colors = {tuple(round(float(c), 4) for c in row[3:]) for row in halo_data}
    assert halo_colors == {STUDIO.edge_color}


def test_the_halo_is_a_scaled_copy_about_the_snap_point():
    """Pure test of _snap_marker_halo, no GL and no renderer.

    Discriminates: scale about the origin instead of about p and the halo's
    centroid moves off the marker's centroid whenever p is not the origin.
    """
    p = np.array([3.0, -2.0, 5.0], dtype=np.float32)
    marker = _snap_marker_vertices(int(SnapKind.GRID), p)
    scale = 1.8
    halo = _snap_marker_halo(marker, p, scale)

    assert halo.shape == marker.shape
    assert np.allclose(halo.mean(axis=0), marker.mean(axis=0), atol=1e-5)

    centre = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
    for orig, scaled in zip(marker, halo, strict=True):
        assert np.allclose(scaled - centre, (orig - centre) * scale, atol=1e-5)
