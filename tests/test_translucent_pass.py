"""M7.5a Task 7: what draws after what.

The GL calls in the draw loop are not testable headlessly, but every *decision*
the translucent pass makes is: which definitions draw in which order, whether a
buffer needs re-planning, how the depth-sorted suffix is re-cut into batches,
and whether a sort is cached. Those are the module-level seams tested here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from bermake.model.model import Model
from bermake.viewport import scene_renderer
from bermake.viewport.camera import Camera
from bermake.viewport.environment import STUDIO
from bermake.viewport.face_batches import FaceBatch, plan_face_batches
from bermake.viewport.render_style import RenderStyle
from bermake.viewport.scene_renderer import (
    _ENVIRONMENT_UNIFORMS,
    _FACE_VERTEX_BYTES,
    _FACE_VERTEX_FLOATS,
    _LINE_UNIFORMS,
    _PHONG_UNIFORMS,
    SceneRenderer,
    _DefBuffers,
    _reset_translucent_state,
    order_definitions_for_translucent_pass,
    rebuild_translucent_batches,
    resolve_batch_sides,
)


class _Fake:
    """Stands in for a Definition; the orderer only needs a centroid source."""

    def __init__(self, centroid):
        self.centroid = np.asarray(centroid, dtype=np.float64)


def _identity():
    return np.eye(4, dtype=np.float64)


def _translate(x):
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    return m


# --- Step 4: ordering definitions among themselves --------------------------


def test_definitions_are_ordered_farthest_first():
    near = (_Fake((0.0, 0.0, 0.0)), _identity())
    far = (_Fake((0.0, 0.0, 0.0)), _translate(50.0))
    order = order_definitions_for_translucent_pass(
        [near, far], camera_pos=np.array([0.0, 0.0, 0.0]), centroid_of=lambda d: d.centroid
    )
    assert order == [1, 0]


def test_the_world_transform_is_applied_before_sorting():
    # Both definitions have the SAME local centroid; only their world
    # transforms differ. A sort that ignored the transform would tie.
    a = (_Fake((0.0, 0.0, 0.0)), _translate(1.0))
    b = (_Fake((0.0, 0.0, 0.0)), _translate(30.0))
    order = order_definitions_for_translucent_pass(
        [a, b], camera_pos=np.array([0.0, 0.0, 0.0]), centroid_of=lambda d: d.centroid
    )
    assert order == [1, 0]


def test_moving_the_camera_reverses_the_order():
    a = (_Fake((0.0, 0.0, 0.0)), _translate(0.0))
    b = (_Fake((0.0, 0.0, 0.0)), _translate(20.0))
    near_a = order_definitions_for_translucent_pass(
        [a, b], camera_pos=np.array([-10.0, 0.0, 0.0]), centroid_of=lambda d: d.centroid
    )
    near_b = order_definitions_for_translucent_pass(
        [a, b], camera_pos=np.array([30.0, 0.0, 0.0]), centroid_of=lambda d: d.centroid
    )
    assert near_a == list(reversed(near_b))


def test_an_empty_list_orders_to_nothing():
    assert (
        order_definitions_for_translucent_pass(
            [], camera_pos=np.array([0.0, 0.0, 0.0]), centroid_of=lambda d: d.centroid
        )
        == []
    )


# --- Step 6b: re-planning when a material crosses the boundary --------------


def test_editing_alpha_moves_a_material_between_the_partitions():
    opaque_plan = plan_face_batches([1, 2], [0, 0], frozenset())
    assert opaque_plan.translucent == []

    # the same geometry, once material 2 has been edited translucent
    translucent_plan = plan_face_batches([1, 2], [0, 0], frozenset({2}))
    assert len(translucent_plan.translucent) == 1
    assert len(translucent_plan.opaque) == 1


def test_a_colour_edit_does_not_change_the_partition():
    # A colour edit leaves the translucent id set alone, so the plan is
    # identical. A renderer that re-uploaded on every edit would still pass
    # this, so it is paired with the buffer-level tests below.
    before = plan_face_batches([1, 2], [0, 0], frozenset({2}))
    after = plan_face_batches([1, 2], [0, 0], frozenset({2}))
    assert before.translucent_first == after.translucent_first
    assert [b.first for b in before.opaque] == [b.first for b in after.opaque]


class _FakeMesh:
    def __init__(self):
        self.dirty = False
        self.clean_calls = 0

    def mark_clean(self):
        self.dirty = False
        self.clean_calls += 1


class _FakeDefinition:
    def __init__(self):
        self.mesh = _FakeMesh()


@pytest.fixture
def stubbed_renderer(monkeypatch):
    """A SceneRenderer whose _upload_definition is a counted stub.

    Lets _ensure_buffers be exercised without a GL context, so the
    reuse-vs-rebuild decision can be observed directly rather than inferred.
    """
    r = SceneRenderer()
    calls = []

    def fake_upload(definition, translucent_ids, model):
        calls.append(translucent_ids)
        return _DefBuffers(translucent_ids=translucent_ids, edge_color=r._environment.edge_color)

    monkeypatch.setattr(r, "_upload_definition", fake_upload)
    return r, calls


def test_an_unchanged_translucent_set_reuses_the_buffer(stubbed_renderer):
    # Kills a renderer that re-uploads unconditionally every frame: it would
    # return a fresh buffer and record a second upload call.
    r, calls = stubbed_renderer
    defn = _FakeDefinition()
    first = r._ensure_buffers(defn, frozenset({2}), Model())
    again = r._ensure_buffers(defn, frozenset({2}), Model())
    assert again is first
    assert len(calls) == 1


def test_a_changed_translucent_set_rebuilds_the_buffer(stubbed_renderer):
    # Kills the bug Step 6b exists for: editing a material's alpha dirties no
    # mesh, so a renderer keyed only on mesh.dirty keeps the stale partition
    # and goes on drawing the now-translucent material in the opaque pass.
    r, calls = stubbed_renderer
    defn = _FakeDefinition()
    first = r._ensure_buffers(defn, frozenset({2}), Model())
    changed = r._ensure_buffers(defn, frozenset({2, 3}), Model())
    assert changed is not first
    assert calls == [frozenset({2}), frozenset({2, 3})]
    assert changed.translucent_ids == frozenset({2, 3})


def test_a_dirty_mesh_still_rebuilds_the_buffer(stubbed_renderer):
    # The pre-existing invalidation must survive the new one being added.
    r, calls = stubbed_renderer
    defn = _FakeDefinition()
    r._ensure_buffers(defn, frozenset({2}), Model())
    defn.mesh.dirty = True
    r._ensure_buffers(defn, frozenset({2}), Model())
    assert len(calls) == 2
    assert defn.mesh.clean_calls == 2  # re-upload marks the mesh clean again


def test_an_unseen_definition_uploads_once(stubbed_renderer):
    r, calls = stubbed_renderer
    r._ensure_buffers(_FakeDefinition(), frozenset(), Model())
    assert len(calls) == 1


# --- Step 6: re-cutting the sorted suffix into batches ----------------------


def test_rebuilt_batches_split_at_every_material_change():
    # Depth order interleaves two translucent materials, so the four triangles
    # must become four single-triangle batches. A rebuild that kept the
    # upload-order batches (two batches of two) would draw half the triangles
    # with the other material's uniforms.
    sorted_pairs = np.array([[9, 0], [7, 0], [9, 0], [7, 0]], dtype=np.int64)
    batches = rebuild_translucent_batches(sorted_pairs, first_vertex=6)
    assert batches == [
        FaceBatch(front_material_id=9, back_material_id=0, first=6, count=3),
        FaceBatch(front_material_id=7, back_material_id=0, first=9, count=3),
        FaceBatch(front_material_id=9, back_material_id=0, first=12, count=3),
        FaceBatch(front_material_id=7, back_material_id=0, first=15, count=3),
    ]


def test_rebuilt_batches_merge_a_run_of_one_material():
    # The common case — one translucent material — must not fragment. Kills a
    # rebuild that emitted one batch per triangle unconditionally.
    sorted_pairs = np.array([[7, 0], [7, 0], [7, 0]], dtype=np.int64)
    batches = rebuild_translucent_batches(sorted_pairs, first_vertex=0)
    assert batches == [FaceBatch(front_material_id=7, back_material_id=0, first=0, count=9)]


def test_the_back_material_alone_splits_a_batch():
    # Batches are keyed on the (front, back) PAIR. A rebuild that compared
    # only the front id would merge these two into one batch and shade the
    # second triangle's back side wrong.
    sorted_pairs = np.array([[7, 0], [7, 5]], dtype=np.int64)
    batches = rebuild_translucent_batches(sorted_pairs, first_vertex=0)
    assert [(b.front_material_id, b.back_material_id) for b in batches] == [(7, 0), (7, 5)]


def test_rebuilt_batches_cover_the_suffix_exactly():
    sorted_pairs = np.array([[1, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int64)
    batches = rebuild_translucent_batches(sorted_pairs, first_vertex=12)
    assert batches[0].first == 12
    assert sum(b.count for b in batches) == 12
    for a, b in zip(batches[:-1], batches[1:], strict=True):
        assert a.first + a.count == b.first


def test_worst_case_fragmentation_is_one_batch_per_triangle():
    # Recorded deliberately rather than capped: alternating materials in depth
    # order degenerate to one draw call per triangle. This pins the bound so
    # it is a known property of option B rather than a surprise in profiling.
    n = 64
    sorted_pairs = np.array([[1 + (i % 2), 0] for i in range(n)], dtype=np.int64)
    batches = rebuild_translucent_batches(sorted_pairs, first_vertex=0)
    assert len(batches) == n


def test_an_empty_suffix_rebuilds_to_no_batches():
    assert rebuild_translucent_batches(np.zeros((0, 2), dtype=np.int64), first_vertex=0) == []


# --- Step 6: sorting the suffix and re-uploading it -------------------------


def _tri_rows(centroid_x: float) -> np.ndarray:
    """Three interleaved face-vertex rows for a triangle centred at x.

    Position and normal are all the translucent sort reads. The remaining
    columns — the UV blocks — are padded out to _FACE_VERTEX_FLOATS so this
    fixture follows the vertex format rather than breaking each time it grows.
    """
    core = np.array(
        [
            [centroid_x, -0.5, 0.0, 0.0, 0.0, 1.0],
            [centroid_x, 0.5, 0.0, 0.0, 0.0, 1.0],
            [centroid_x, 0.0, 0.5, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pad = np.zeros((3, _FACE_VERTEX_FLOATS - core.shape[1]), dtype=np.float32)
    return np.concatenate([core, pad], axis=1)


# Upload order groups by material pair, exactly as plan_face_batches leaves it:
# triangles 0,1 carry (7, 0); triangles 2,3 carry (9, 0). Their x positions
# interleave the two materials in depth, so a correct sort MUST re-cut the
# batches. The resulting triangle order is [3, 1, 2, 0] — neither the input
# order nor its reverse, so a no-op sort and a blind reversal both fail.
_SUFFIX_X = {0: 0.0, 1: 2.0, 2: 1.0, 3: 3.0}
_PAIR_OF_X = {0.0: (7, 0), 2.0: (7, 0), 1.0: (9, 0), 3.0: (9, 0)}
_OPAQUE_TRIS = 2
_TRANSLUCENT_FIRST = _OPAQUE_TRIS * 3


def _suffix_buffer() -> _DefBuffers:
    opaque = np.zeros((_TRANSLUCENT_FIRST, _FACE_VERTEX_FLOATS), dtype=np.float32)
    opaque[:, 0] = -99.0  # sentinel: the prefix must never be re-uploaded
    suffix = np.concatenate([_tri_rows(_SUFFIX_X[i]) for i in range(4)], axis=0)
    interleaved = np.concatenate([opaque, suffix], axis=0)

    plan = plan_face_batches([1, 1, 7, 7, 9, 9], [0, 0, 0, 0, 0, 0], frozenset({7, 9}))
    assert plan.translucent_first == _TRANSLUCENT_FIRST
    buf = _DefBuffers(face_vbo=1, face_count=interleaved.shape[0], plan=plan)
    _reset_translucent_state(buf, plan, interleaved)
    return buf


@pytest.fixture
def captured_gl(monkeypatch):
    """Capture glBufferSubData without a GL context."""
    uploads = []

    def fake_sub_data(target, offset, size, data):
        uploads.append((int(offset), int(size), np.asarray(data, dtype=np.float32).copy()))

    monkeypatch.setattr(scene_renderer.GL, "glBindBuffer", lambda *a: None)
    monkeypatch.setattr(scene_renderer.GL, "glBufferSubData", fake_sub_data)
    return uploads


def test_sorting_uploads_only_the_suffix_back_to_front(captured_gl):
    r = SceneRenderer()
    buf = _suffix_buffer()
    r._sort_translucent_slice(buf, _identity(), np.array([-10.0, 0.0, 0.0]))

    assert len(captured_gl) == 1
    offset, size, data = captured_gl[0]
    # Offset and size are the suffix alone: the opaque prefix is untouched.
    assert offset == _TRANSLUCENT_FIRST * _FACE_VERTEX_BYTES
    assert size == 12 * _FACE_VERTEX_BYTES
    rows = data.reshape(-1, _FACE_VERTEX_FLOATS)
    assert rows.shape[0] == 12
    assert not np.any(rows[:, 0] == -99.0)
    # Farthest first from a camera at x = -10 means descending x.
    tri_x = [float(rows[3 * i, 0]) for i in range(4)]
    assert tri_x == [3.0, 2.0, 1.0, 0.0]


def test_the_rebuilt_batches_match_the_materials_of_what_was_uploaded(captured_gl):
    # This is the reason option B was chosen over keeping plan.translucent's
    # batch ranges: after the suffix is permuted across material boundaries,
    # the original ranges name the wrong materials. Asserting each rebuilt
    # batch's range against the materials of the triangles actually uploaded
    # into it fails for a renderer that reuses plan.translucent unchanged.
    r = SceneRenderer()
    buf = _suffix_buffer()
    r._sort_translucent_slice(buf, _identity(), np.array([-10.0, 0.0, 0.0]))

    rows = captured_gl[0][2].reshape(-1, _FACE_VERTEX_FLOATS)
    assert len(buf.translucent_draw_batches) == 4  # fully interleaved
    for batch in buf.translucent_draw_batches:
        lo = batch.first - _TRANSLUCENT_FIRST
        for v in range(lo, lo + batch.count):
            expected = _PAIR_OF_X[float(rows[v, 0])]
            assert (batch.front_material_id, batch.back_material_id) == expected


def test_a_static_camera_does_not_re_upload(captured_gl):
    # Kills a sort that runs every frame regardless.
    r = SceneRenderer()
    buf = _suffix_buffer()
    camera = np.array([-10.0, 0.0, 0.0])
    r._sort_translucent_slice(buf, _identity(), camera)
    batches = buf.translucent_draw_batches
    r._sort_translucent_slice(buf, _identity(), camera)
    assert len(captured_gl) == 1
    assert buf.translucent_draw_batches is batches  # cached, not rebuilt


def test_moving_the_camera_re_sorts_and_re_uploads(captured_gl):
    # Kills a sort that caches on first use and never invalidates.
    r = SceneRenderer()
    buf = _suffix_buffer()
    r._sort_translucent_slice(buf, _identity(), np.array([-10.0, 0.0, 0.0]))
    r._sort_translucent_slice(buf, _identity(), np.array([+10.0, 0.0, 0.0]))
    assert len(captured_gl) == 2
    near_side = captured_gl[1][2].reshape(-1, _FACE_VERTEX_FLOATS)
    assert [float(near_side[3 * i, 0]) for i in range(4)] == [0.0, 1.0, 2.0, 3.0]


def test_moving_the_definition_re_sorts_and_re_uploads(captured_gl):
    # The world transform is part of the cache key: a definition dragged past
    # the camera must re-sort even though the camera never moved.
    r = SceneRenderer()
    buf = _suffix_buffer()
    camera = np.array([0.0, 0.0, 0.0])
    r._sort_translucent_slice(buf, _translate(-100.0), camera)
    r._sort_translucent_slice(buf, _translate(+100.0), camera)
    assert len(captured_gl) == 2
    # Mirrored across the camera, so the depth order flips.
    first = [float(captured_gl[0][2].reshape(-1, _FACE_VERTEX_FLOATS)[3 * i, 0]) for i in range(4)]
    second = [float(captured_gl[1][2].reshape(-1, _FACE_VERTEX_FLOATS)[3 * i, 0]) for i in range(4)]
    assert first == list(reversed(second))


def test_a_definition_with_no_translucent_faces_uploads_nothing(captured_gl):
    r = SceneRenderer()
    plan = plan_face_batches([1, 1], [0, 0], frozenset())
    buf = _DefBuffers(face_vbo=1, plan=plan)
    _reset_translucent_state(buf, plan, np.zeros((6, _FACE_VERTEX_FLOATS), dtype=np.float32))
    r._sort_translucent_slice(buf, _identity(), np.array([1.0, 2.0, 3.0]))
    assert captured_gl == []


def test_re_upload_clears_a_stale_sort_key(captured_gl):
    # A mesh edit or an alpha edit rebuilds the buffer in place. If the sort
    # key survived, the next frame would keep the previous frame's permutation
    # and batch ranges over completely different vertex data.
    r = SceneRenderer()
    buf = _suffix_buffer()
    camera = np.array([-10.0, 0.0, 0.0])
    r._sort_translucent_slice(buf, _identity(), camera)
    assert buf.translucent_sort_key is not None

    _reset_translucent_state(buf, buf.plan, buf.face_interleaved)
    assert buf.translucent_sort_key is None
    assert buf.translucent_draw_batches == buf.plan.translucent  # pre-sort state

    r._sort_translucent_slice(buf, _identity(), camera)
    assert len(captured_gl) == 2  # sorted again rather than trusting the stale key


def test_an_all_opaque_definition_keeps_no_cpu_copy_of_its_faces():
    # Only the translucent pass reads face_interleaved. Holding a CPU copy for
    # every opaque definition would roughly double scene face memory for a pass
    # those definitions never enter. Kills a _reset_translucent_state that
    # stores the array unconditionally.
    plan = plan_face_batches([1, 2], [0, 0], frozenset())
    buf = _DefBuffers()
    _reset_translucent_state(buf, plan, np.ones((6, _FACE_VERTEX_FLOATS), dtype=np.float32))
    assert buf.face_interleaved.shape[0] == 0


def test_a_translucent_definition_keeps_its_faces_for_re_sorting():
    # The paired half: the array the sort reads must actually be retained.
    buf = _suffix_buffer()
    assert buf.face_interleaved.shape[0] == _TRANSLUCENT_FIRST + 12


def test_the_local_centroid_is_the_mean_of_the_translucent_triangles():
    # The definition's own sort key for pass 2. Kills a key taken from the
    # whole mesh (which would include the opaque prefix's x = -99 sentinel).
    buf = _suffix_buffer()
    assert buf.translucent_local_centroid[0] == pytest.approx(1.5)


# --- Task 7: the two-pass draw order inside render() ------------------------
#
# Everything above tests a seam. render() is the loop those seams hang off, and
# it had no automated coverage at all — four plausible regressions in it pass
# every test above:
#
#   1. drawing `buf.plan.translucent` instead of `buf.translucent_draw_batches`
#      (the wrong-material bug, at the one line where it reaches the screen);
#   2. collapsing pass 2 back into pass 1, so translucent faces no longer
#      follow ALL opaque geometry across ALL definitions;
#   3. dropping the `_sort_translucent_slice(...)` call, so nothing sorts;
#   4. moving edge drawing out of pass 1 and into pass 2.
#
# The recorder pattern from
# test_two_sided_shading.py::test_a_face_draw_sets_every_cached_phong_uniform
# is extended here to record the *sequence* of draws rather than a set of
# uniform writes, so ordering is assertable with no GL context — which is what
# makes these tests runnable in CI, where the offscreen platform is forced and
# hardware GL may be unavailable.


class _SequenceGL:
    """Stand-in for the OpenGL module for a whole `render()` call.

    Names beginning `GL_` resolve to stable ints rather than callables, because
    render() ORs GL_COLOR_BUFFER_BIT with GL_DEPTH_BUFFER_BIT. Everything else
    is a no-op returning 0, so glGenVertexArrays et al. hand back zero handles
    that _DefBuffers.release() is already guarded against.

    glBufferSubData is captured: it is the only GL call the translucent sort
    makes that the recorded draw order does not already reveal, so it is how
    "the suffix was actually re-uploaded back to front" gets asserted.
    """

    def __init__(self) -> None:
        self.sub_data: list[tuple[int, np.ndarray]] = []
        self._consts: dict[str, int] = {}

    def __getattr__(self, name: str):
        if name.startswith("GL_"):
            return self._consts.setdefault(name, len(self._consts) + 1)
        if name == "glBufferSubData":

            def _sub(target, offset, size, data):
                self.sub_data.append((int(offset), np.asarray(data, dtype=np.float32).copy()))
                return 0

            return _sub

        def _call(*args, **kwargs):
            return 0

        return _call


@dataclass(frozen=True)
class _Draw:
    """One recorded draw call: what was drawn, for which definition, where in
    that definition's VBO, and which material it was shaded with."""

    kind: str  # "face" | "edge"
    definition: str
    first: int
    count: int
    front_diffuse: tuple | None


def _triangle(scene, x: float) -> int:
    """A standalone triangle in the plane X = x, with its own vertices so it
    contributes its own three edges."""
    ids = [
        scene.add_vertex(np.array([x, 0.0, 0.0])),
        scene.add_vertex(np.array([x, 1.0, 0.0])),
        scene.add_vertex(np.array([x, 0.0, 1.0])),
    ]
    return scene.add_face_from_loop(ids)


class _Harness:
    """A real SceneRenderer driven through its real render() over a real Model,
    with GL replaced by a recorder.

    The scene is built so a collapsed single-traversal loop produces a
    DIFFERENT recorded sequence from the correct two-pass one: the translucent
    definition is traversed FIRST, so collapsing pass 2 into pass 1 would put
    its faces ahead of the opaque definition's face and edges. A fixture where
    both orderings coincide would prove nothing.

    Inside the translucent definition two translucent materials alternate in
    depth (x = 0 red, 1 blue, 2 red, 3 blue; camera on the -X side). Upload
    order groups them into two batches of two triangles; the depth sort re-cuts
    them into four batches of one. plan.translucent and translucent_draw_batches
    are therefore observably different lists, which is what lets the
    wrong-material regression be seen at all.
    """

    OPAQUE_X = 50.0

    def __init__(self, monkeypatch) -> None:
        self.gl = _SequenceGL()
        monkeypatch.setattr(scene_renderer, "GL", self.gl)

        self.model = Model()
        lib = self.model.materials
        self.red = lib.add_custom("GlassRed", (0.9, 0.1, 0.1)).id
        self.blue = lib.add_custom("GlassBlue", (0.1, 0.1, 0.9)).id
        self.green = lib.add_custom("Solid", (0.1, 0.9, 0.1)).id
        lib.edit(self.red, alpha=0.4)
        lib.edit(self.blue, alpha=0.5)

        self.glass = self.model.new_definition("Glass", is_group=True)
        for x, mid in ((0.0, self.red), (1.0, self.blue), (2.0, self.red), (3.0, self.blue)):
            self.glass.mesh.set_face_material(_triangle(self.glass.mesh, x), mid)

        self.solid = self.model.new_definition("Solid", is_group=True)
        self.solid.mesh.set_face_material(_triangle(self.solid.mesh, self.OPAQUE_X), self.green)

        # Glass first: a collapsed loop would draw it before Solid.
        self.model.root.children.append(self.model.new_instance(self.glass))
        self.model.root.children.append(self.model.new_instance(self.solid))

        self.renderer = SceneRenderer()
        self.renderer._initialized = True
        self.renderer._phong_program = 1
        self.renderer._line_program = 2
        self.renderer._phong_locs = {n: i for i, n in enumerate(_PHONG_UNIFORMS)}
        self.renderer._line_locs = {n: i for i, n in enumerate(_LINE_UNIFORMS)}
        self.renderer._environment_program = 3
        self.renderer._environment_locs = {n: i for i, n in enumerate(_ENVIRONMENT_UNIFORMS)}

        self._records: list[tuple] = []
        real_faces = self.renderer._draw_definition_faces
        real_edges = self.renderer._draw_definition_edges

        def faces(buf, *a, front, back, first=0, count=None, **kw):
            self._records.append(("face", buf, int(first), int(count), front.diffuse))
            real_faces(buf, *a, front=front, back=back, first=first, count=count, **kw)

        def edges(buf, *a, **kw):
            self._records.append(("edge", buf, 0, int(buf.edge_count), None))
            real_edges(buf, *a, **kw)

        monkeypatch.setattr(self.renderer, "_draw_definition_faces", faces)
        monkeypatch.setattr(self.renderer, "_draw_definition_edges", edges)

        # Looking along +X from the far side of x = 0, so back to front is
        # descending x.
        self.camera = Camera(
            position=np.array([-10.0, 0.5, 0.5], dtype=np.float32),
            target=np.array([1.5, 0.5, 0.5], dtype=np.float32),
        )

    def render(self) -> list[_Draw]:
        self._records.clear()
        self.gl.sub_data.clear()
        self.renderer.render(self.camera, self.model)
        names = {id(d): d.name for d, _ in self.model.traverse()}
        by_buf = {id(buf): names[key] for key, buf in self.renderer._def_buffers.items()}
        return [
            _Draw(kind, by_buf[id(buf)], first, count, diffuse)
            for kind, buf, first, count, diffuse in self._records
        ]

    def diffuse_of(self, material_id: int) -> tuple:
        """The front diffuse a batch of `material_id` resolves to, computed the
        way render() computes it — so the assertion names a material rather
        than a magic colour triple."""
        front, _ = resolve_batch_sides(
            FaceBatch(front_material_id=material_id, back_material_id=0, first=0, count=3),
            self.model.materials,
            RenderStyle(),
            bg=STUDIO.background,
            dimmed=False,
            translucent_ids=scene_renderer._translucent_ids(
                self.model.materials, self.model.textures
            ),
        )
        return front.diffuse

    def suffix_triangle_x(self) -> list[float]:
        """The X of each triangle in the single suffix upload this frame made."""
        assert len(self.gl.sub_data) == 1, (
            f"expected one translucent suffix upload, got {len(self.gl.sub_data)}"
        )
        rows = self.gl.sub_data[0][1].reshape(-1, _FACE_VERTEX_FLOATS)
        return [float(rows[3 * i, 0]) for i in range(rows.shape[0] // 3)]


@pytest.fixture
def harness(monkeypatch):
    return _Harness(monkeypatch)


def test_render_draws_every_opaque_batch_and_edge_before_any_translucent_face(harness):
    """Kills the collapse of pass 2 back into pass 1.

    Glass is traversed first, so a single-traversal loop draws its translucent
    faces before Solid's opaque face and edges ever run — exactly the artefact
    the split traversal exists to prevent. Pinning the whole sequence also
    catches the missing sort and edges moved into pass 2.
    """
    draws = harness.render()
    assert [(d.kind, d.definition) for d in draws] == [
        ("edge", "Glass"),
        ("face", "Solid"),
        ("edge", "Solid"),
        ("face", "Glass"),
        ("face", "Glass"),
        ("face", "Glass"),
        ("face", "Glass"),
    ]


def test_every_definitions_edges_draw_in_pass_one(harness):
    """Kills edge drawing moved into pass 2.

    Pass 2 visits only definitions that HAVE translucent geometry, so an
    all-opaque definition's edges would silently stop being drawn at all — and
    Glass's own edges would land after its translucent faces.
    """
    draws = harness.render()
    edge_at = [i for i, d in enumerate(draws) if d.kind == "edge"]
    glass_faces_at = [
        i for i, d in enumerate(draws) if d.kind == "face" and d.definition == "Glass"
    ]

    assert [draws[i].definition for i in edge_at] == ["Glass", "Solid"]
    assert glass_faces_at, "no translucent faces were drawn at all"
    assert max(edge_at) < min(glass_faces_at)


def test_the_translucent_pass_draws_the_re_cut_batches_not_the_upload_order_ones(harness):
    """Kills `buf.plan.translucent` at the one line where the wrong-material
    bug reaches the screen.

    Upload order groups the four triangles into two batches of two (red, red |
    blue, blue). Depth order from the -X side is 3, 2, 1, 0 — blue, red, blue,
    red — so the sorted suffix must draw as four one-triangle batches whose
    materials follow the permutation. Drawing the upload-order ranges over the
    sorted buffer shades half the triangles with the other material.
    """
    draws = harness.render()
    glass_faces = [d for d in draws if d.kind == "face" and d.definition == "Glass"]

    buf = harness.renderer._def_buffers[id(harness.glass)]
    assert [(b.first, b.count) for b in buf.plan.translucent] == [(0, 6), (6, 6)]

    assert [(d.first, d.count) for d in glass_faces] == [(0, 3), (3, 3), (6, 3), (9, 3)]
    assert [d.front_diffuse for d in glass_faces] == [
        harness.diffuse_of(harness.blue),
        harness.diffuse_of(harness.red),
        harness.diffuse_of(harness.blue),
        harness.diffuse_of(harness.red),
    ]


def test_render_uploads_the_translucent_suffix_back_to_front(harness):
    """Kills a dropped `_sort_translucent_slice(...)` call.

    Without it no suffix upload happens at all, so the triangles keep their
    upload order in the VBO and near glass composites over far glass.
    """
    assert harness.render()
    assert harness.suffix_triangle_x() == [3.0, 2.0, 1.0, 0.0]


def test_moving_the_camera_re_sorts_through_render(harness):
    """The sort is not a once-per-buffer initialisation: a camera move must
    re-upload the suffix in the new depth order, and the batches drawn must
    follow it."""
    harness.render()
    assert harness.suffix_triangle_x() == [3.0, 2.0, 1.0, 0.0]

    harness.camera.position = np.array([20.0, 0.5, 0.5], dtype=np.float32)
    draws = harness.render()
    assert harness.suffix_triangle_x() == [0.0, 1.0, 2.0, 3.0]

    glass_faces = [d for d in draws if d.kind == "face" and d.definition == "Glass"]
    assert [d.front_diffuse for d in glass_faces] == [
        harness.diffuse_of(harness.red),
        harness.diffuse_of(harness.blue),
        harness.diffuse_of(harness.red),
        harness.diffuse_of(harness.blue),
    ]


def test_pass_two_selects_definitions_by_the_very_list_it_draws(harness):
    """Pass 2's filter and its draw source must be the same list.

    The filter asks whether a definition has translucent batches; the loop then
    draws `buf.translucent_draw_batches`. Asking one list and drawing another
    only works while the two are seeded together in _reset_translucent_state —
    an invariant nothing enforced. Here a cached buffer is handed a translucent
    batch directly with `plan.translucent` left empty: a filter keyed on
    `plan.translucent` skips the definition and the batch is never drawn.
    """
    harness.render()  # first frame builds and caches the buffers
    buf = harness.renderer._def_buffers[id(harness.solid)]
    assert buf.plan.translucent == []  # Solid is entirely opaque
    buf.translucent_draw_batches = [
        FaceBatch(front_material_id=harness.green, back_material_id=0, first=0, count=3)
    ]

    # The mesh is clean and the translucent id set is unchanged, so pass 1
    # reuses this buffer rather than re-uploading over the seeded list.
    draws = harness.render()
    solid_faces = [d for d in draws if d.kind == "face" and d.definition == "Solid"]
    assert len(solid_faces) == 2  # once opaque in pass 1, once translucent in pass 2
    assert draws[-1] == _Draw("face", "Solid", 0, 3, harness.diffuse_of(harness.green))
