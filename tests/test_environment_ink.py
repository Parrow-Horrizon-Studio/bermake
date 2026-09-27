"""Grid and edge ink following the environment, without a GL context."""

from bermake.viewport.environment import PLAIN_WHITE, STUDIO
from bermake.viewport.scene_renderer import _GRID_HALF_EXTENT, _build_grid_vertex_array


def _row_is_centerline(row) -> bool:
    """True when `row` (x, y, z, r, g, b) belongs to one of the grid's two
    centre lines.

    A row's two coordinates are x and y; exactly one of them is pinned to
    +/-_GRID_HALF_EXTENT (the endpoint of the line segment this row is one end
    of), and the other is the value that varies from one grid line to the next
    across the whole family (named `v` in _build_grid_vertex_array). The row
    is a centre-line row when that second, varying value is within 1e-5 of
    zero.
    """
    x, y = float(row[0]), float(row[1])
    varying = y if abs(abs(x) - _GRID_HALF_EXTENT) < 1e-5 else x
    return abs(varying) < 1e-5


def _assert_rows_carry_the_right_colour(verts, grid_color, centerline_color) -> None:
    """Every row must carry grid_color or centerline_color, never the other,
    according to which line it belongs to -- not merely have both colours
    somewhere in the array.
    """
    for row in verts:
        expected = centerline_color if _row_is_centerline(row) else grid_color
        actual = tuple(round(float(c), 4) for c in row[3:])
        assert actual == expected, f"row {row!r} carries {actual}, expected {expected}"


def test_the_grid_array_carries_the_environments_grid_colours():
    """Every row carries the right one of the two colours: the two centre
    lines get grid_centerline_color, the other twenty get grid_color.

    This calls _build_grid_vertex_array directly with its own arguments in the
    same order every call, so it cannot see whether the real _init_grid_buffers
    call site passes grid_color and centerline_color in that same order --
    swapping them there produces identical rows-carry-both-colours-somewhere
    results. That call site is pinned separately, by
    test_init_grid_buffers_passes_the_colours_in_order below, which drives
    _init_grid_buffers itself rather than calling the function directly.
    """
    verts = _build_grid_vertex_array(PLAIN_WHITE.grid_color, PLAIN_WHITE.grid_centerline_color)
    _assert_rows_carry_the_right_colour(
        verts, PLAIN_WHITE.grid_color, PLAIN_WHITE.grid_centerline_color
    )


def test_the_grid_array_shape_is_unchanged():
    """44 vertices of 6 floats: parameterising the colours must not move geometry."""
    verts = _build_grid_vertex_array(STUDIO.grid_color, STUDIO.grid_centerline_color)
    assert verts.shape == (44, 6)


def test_the_studio_grid_matches_the_pre_m77_array():
    """The dark environment's grid carries the same two colours in the same
    rows as v0.12.0 drew, checked row by row rather than as a colour set --
    a set comparison cannot tell a row-for-row match from the two colours
    merely being swapped between the grid and centre lines.
    """
    verts = _build_grid_vertex_array(STUDIO.grid_color, STUDIO.grid_centerline_color)
    _assert_rows_carry_the_right_colour(verts, STUDIO.grid_color, STUDIO.grid_centerline_color)


def test_init_grid_buffers_passes_the_colours_in_order(monkeypatch):
    """Pins the _init_grid_buffers call site, which the two tests above cannot
    see: they call _build_grid_vertex_array directly, with its arguments in
    the same order the call site is supposed to use, so a swap made only at
    the call site is invisible to them.

    _upload_interleaved_lines is monkeypatched to capture the array it is
    handed instead of uploading it to a real GL context. SceneRenderer's own
    __init__ leaves _grid_vao and _grid_vbo at 0, so the delete guards at the
    top of _init_grid_buffers no-op and no GL stand-in is needed for them.

    Discriminates: swap the two arguments in the _build_grid_vertex_array(...)
    call inside _init_grid_buffers and this fails, because the array handed to
    _upload_interleaved_lines then carries grid_color on the centre-line rows
    and grid_centerline_color everywhere else.
    """
    from bermake.viewport.scene_renderer import SceneRenderer

    renderer = SceneRenderer()
    assert renderer._grid_vao == 0
    assert renderer._grid_vbo == 0

    captured: dict[str, object] = {}

    def _capture_upload(verts):
        captured["verts"] = verts
        return 1, 2

    monkeypatch.setattr(renderer, "_upload_interleaved_lines", _capture_upload)
    renderer._init_grid_buffers()

    assert "verts" in captured, "_upload_interleaved_lines was never called"
    env = renderer._environment
    _assert_rows_carry_the_right_colour(
        captured["verts"], env.grid_color, env.grid_centerline_color
    )


def test_switching_environment_marks_the_grid_for_rebuild():
    """set_environment can be called with no current GL context.

    MainWindow's menu handler runs outside paintGL, exactly like the texture
    evictions _flush_pending_texture_evictions exists for, so the rebuild is
    deferred to the next frame rather than done in the setter.
    """
    from bermake.viewport.scene_renderer import SceneRenderer

    renderer = SceneRenderer()
    renderer._grid_dirty = False
    renderer.set_environment(PLAIN_WHITE)
    assert renderer._grid_dirty is True


def test_setting_the_same_environment_does_not_mark_the_grid_dirty():
    """A redundant menu click must not queue a re-upload every time."""
    from bermake.viewport.scene_renderer import SceneRenderer

    renderer = SceneRenderer()
    renderer.set_environment(PLAIN_WHITE)
    renderer._grid_dirty = False
    renderer.set_environment(PLAIN_WHITE)
    assert renderer._grid_dirty is False


def test_a_definition_with_no_edges_still_records_the_colour_it_was_uploaded_under():
    """The predicate alone, not the production line that has to call it right.

    This builds a `_DefBuffers` by hand and asserts on `_edge_buffer_is_stale`
    directly, so it never drives `_upload_definition` and cannot see whether
    the real upload actually performs the `buf.edge_color = edge_color`
    assignment on the zero-edge path. That assignment sits after the
    if/else, not inside a branch, precisely so a third branch could not skip
    it again; the test that pins the real upload behaviour, including the
    forever-stale bug this recording exists to prevent, is
    test_a_zero_edge_definition_does_not_re_upload_every_frame below.
    """
    from bermake.viewport.scene_renderer import _DefBuffers, _edge_buffer_is_stale

    buf = _DefBuffers()
    buf.edge_count = 0
    buf.edge_color = STUDIO.edge_color
    assert _edge_buffer_is_stale(buf, STUDIO.edge_color) is False
    assert _edge_buffer_is_stale(buf, PLAIN_WHITE.edge_color) is True


def test_a_never_uploaded_buffer_is_stale():
    """The default of None must read as stale, so the first frame uploads."""
    from bermake.viewport.scene_renderer import _DefBuffers, _edge_buffer_is_stale

    assert _edge_buffer_is_stale(_DefBuffers(), STUDIO.edge_color) is True


def test_the_real_upload_records_the_environments_edge_colour(monkeypatch):
    """Drives _upload_definition itself, not a hand-built buffer.

    Uses the GL-stub recipe from
    test_texture_vertex_format.py::test_uploading_a_definition_with_a_split_edge_face_does_not_raise:
    glBindBuffer/glBufferData are no-ops and _alloc_def_buffers is stubbed, so
    the real upload runs with no GL context.

    Discriminates: comment out `buf.edge_color = edge_color` in
    _upload_definition and this fails, because buf.edge_color stays None.
    """
    from bermake._core import make_box
    from bermake.model.model import Model
    from bermake.scene.mesh_builder import build_mesh_into_scene
    from bermake.viewport import scene_renderer
    from bermake.viewport.scene_renderer import SceneRenderer, _DefBuffers

    model = Model()
    build_mesh_into_scene(make_box(2.0, 2.0, 2.0), model.root.mesh)
    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(scene_renderer.GL, "glBufferData", lambda *a: None)
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))

    buf = r._upload_definition(model.root, frozenset(), model)
    assert buf.edge_count > 0
    assert buf.edge_color == r._environment.edge_color


def test_a_zero_edge_definition_does_not_re_upload_every_frame(monkeypatch):
    """The expensive half of the bug: nothing on screen changes, so no visual
    check and no fresh-load test can see a definition re-uploading on every
    single frame forever.

    Calls _ensure_buffers twice with nothing changed in between and counts
    calls that reach the real _upload_definition. A definition with zero
    edges that skipped recording edge_color on that branch would read as
    stale forever, so the second call would re-upload and this would fail.

    Discriminates: comment out `buf.edge_color = edge_color` in
    _upload_definition and this fails (the second call re-uploads).
    """
    from bermake.model.model import Model
    from bermake.viewport import scene_renderer
    from bermake.viewport.scene_renderer import SceneRenderer, _DefBuffers

    model = Model()  # empty scene: no faces, no edges
    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(scene_renderer.GL, "glBufferData", lambda *a: None)
    r = SceneRenderer()
    monkeypatch.setattr(r, "_alloc_def_buffers", lambda: _DefBuffers(face_vbo=1, edge_vbo=2))

    upload_calls: list[int] = []
    real_upload = r._upload_definition

    def counted_upload(definition, translucent_ids, model):
        upload_calls.append(1)
        return real_upload(definition, translucent_ids, model)

    monkeypatch.setattr(r, "_upload_definition", counted_upload)

    first = r._ensure_buffers(model.root, frozenset(), model)
    assert first.edge_count == 0
    second = r._ensure_buffers(model.root, frozenset(), model)
    assert second is first
    assert len(upload_calls) == 1
