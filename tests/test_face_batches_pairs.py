"""M7.5a Task 4: pair-keyed batches and the translucent suffix."""

from __future__ import annotations

import numpy as np
from bermake.viewport.face_batches import plan_face_batches


def test_an_empty_input_gives_an_empty_plan():
    plan = plan_face_batches([], [])
    assert plan.vertex_order.shape == (0,)
    assert plan.opaque == []
    assert plan.translucent == []
    assert plan.translucent_first == 0


def test_faces_sharing_a_pair_land_in_one_batch():
    plan = plan_face_batches([1, 1, 1], [2, 2, 2])
    assert len(plan.opaque) == 1
    b = plan.opaque[0]
    assert (b.front_material_id, b.back_material_id) == (1, 2)
    assert (b.first, b.count) == (0, 9)


def test_the_same_front_with_different_backs_splits_into_two_batches():
    # The whole point of the pair key: a single-id key would merge these.
    plan = plan_face_batches([1, 1], [2, 3])
    assert len(plan.opaque) == 2
    assert {(b.front_material_id, b.back_material_id) for b in plan.opaque} == {(1, 2), (1, 3)}


def test_translucent_triangles_form_a_contiguous_suffix():
    # front ids 1,9,1,9 with 9 translucent: the two 9s must end up adjacent
    # and at the END, whatever order they arrived in.
    plan = plan_face_batches([1, 9, 1, 9], [0, 0, 0, 0], frozenset({9}))
    n_vertices = 4 * 3
    assert plan.translucent_first == 6  # two opaque triangles first
    covered = []
    for b in plan.translucent:
        assert b.first >= plan.translucent_first
        covered.extend(range(b.first, b.first + b.count))
    assert covered == list(range(plan.translucent_first, n_vertices))


def test_a_low_numbered_translucent_id_still_sorts_to_the_suffix():
    # The final-review finding on Task 4: every other fixture in this file (and
    # in test_face_batches.py) gives its translucent materials ids numerically
    # ABOVE every opaque id, so sorting by (front, back) alone already puts
    # them last. Dropping the translucency term from the sort key therefore
    # broke the contiguous-suffix property outright and all 21 tests still
    # passed.
    #
    # Here the translucent id (1) is BELOW the opaque ids (5), so a key of
    # (front, back) alone sorts the translucent triangle FIRST and
    # translucent_first collapses to 0. Only a key that leads with
    # translucency puts it at the end.
    plan = plan_face_batches([5, 1, 5], [0, 0, 0], frozenset({1}))

    assert plan.translucent_first == 6  # the two opaque triangles come first
    assert len(plan.translucent) == 1
    assert plan.translucent[0].first == 6
    assert plan.translucent[0].front_material_id == 1
    # And the permutation agrees: the suffix really holds triangle 1.
    assert plan.vertex_order[6:].tolist() == [3, 4, 5]
    assert all(b.first < 6 for b in plan.opaque)


def test_a_translucent_back_makes_the_whole_face_translucent():
    # Spec D5. The front is opaque; the back is not; the face goes translucent.
    plan = plan_face_batches([1], [9], frozenset({9}))
    assert plan.opaque == []
    assert len(plan.translucent) == 1
    assert plan.translucent_first == 0


def test_nothing_translucent_leaves_the_suffix_empty_and_past_the_end():
    plan = plan_face_batches([1, 2], [0, 0])
    assert plan.translucent == []
    assert plan.translucent_first == 6  # == 3T, an empty suffix


def test_the_permutation_actually_reorders_the_vertex_arrays():
    # Triangle 0 is translucent, triangle 1 is not, so they must swap.
    plan = plan_face_batches([9, 1], [0, 0], frozenset({9}))
    verts = np.arange(6 * 3).reshape(6, 3)  # 2 triangles, 3 vertices each
    reordered = verts[plan.vertex_order]
    assert reordered[0].tolist() == verts[3].tolist()
    assert plan.translucent_first == 3


def test_the_permutation_is_a_genuine_permutation():
    plan = plan_face_batches([3, 1, 2, 1], [0, 5, 0, 5], frozenset({2}))
    assert sorted(plan.vertex_order.tolist()) == list(range(12))


def test_batch_counts_cover_every_vertex_exactly_once():
    plan = plan_face_batches([1, 2, 9, 2], [0, 0, 0, 7], frozenset({9, 7}))
    covered = []
    for b in plan.opaque + plan.translucent:
        covered.extend(range(b.first, b.first + b.count))
    assert sorted(covered) == list(range(12))


def test_mismatched_side_lengths_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        plan_face_batches([1, 2], [1])


def test_a_front_id_at_the_2_20_limit_is_rejected():
    # 2**20 is one past the documented safe boundary (2**20 - 1, pinned
    # above): at this value the id's bits collide into the translucency
    # flag's bit during packing. The guard must reject it loudly rather
    # than silently corrupting the sort key.
    import pytest

    limit = 1 << 20
    with pytest.raises(ValueError):
        plan_face_batches([limit], [0])


def test_a_back_id_at_the_2_20_limit_is_rejected():
    # Same as above but in back_ids, so a guard that only checks front_ids
    # (a plausible half-fix) is caught.
    import pytest

    limit = 1 << 20
    with pytest.raises(ValueError):
        plan_face_batches([0], [limit])


def test_a_negative_front_id_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        plan_face_batches([-1], [0])


def test_a_negative_back_id_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        plan_face_batches([0], [-1])


def test_vertex_order_actually_agrees_with_the_batches_and_the_suffix():
    # Regression for the review finding on Task 4: the other tests read
    # plan.translucent_first and plan.translucent/opaque but never check that
    # plan.vertex_order actually produces those ranges. An implementation
    # that computed the batches correctly but returned an unrelated (even
    # identity) vertex_order would pass every other test in this file.
    #
    # 6 triangles, 4 distinct (front, back) pairs, translucent triangles
    # interleaved with opaque ones in the input (not already a suffix):
    #   tri 0: (1, 0) opaque       tri 3: (1, 0) opaque
    #   tri 1: (9, 0) translucent  tri 4: (2, 9) translucent
    #   tri 2: (2, 0) opaque       tri 5: (2, 0) opaque
    front_ids = [1, 9, 2, 1, 2, 2]
    back_ids = [0, 0, 0, 0, 9, 0]
    translucent_mids = frozenset({9})
    pairs = list(zip(front_ids, back_ids, strict=True))
    translucent_tris = {i for i, (f, b) in enumerate(pairs) if f in translucent_mids or b in translucent_mids}
    assert len({(f, b) for f, b in pairs}) >= 3
    assert len(pairs) >= 4
    assert translucent_tris == {1, 4}  # confirms they're interleaved, not a suffix already

    plan = plan_face_batches(front_ids, back_ids, translucent_mids)

    # Real per-vertex data: an affine encoding of the original vertex index,
    # deliberately not equal to the index itself, so a test bug that just
    # compared indices to indices can't accidentally pass.
    T = len(front_ids)
    verts = np.arange(3 * T, dtype=np.int64) * 100 + 7

    def origin_triangle(value: int) -> int:
        return int((value - 7) // 100) // 3

    reordered = verts[plan.vertex_order]

    # (3) the suffix contains exactly the translucent triangles' vertices.
    suffix = reordered[plan.translucent_first :]
    assert suffix.shape[0] == len(translucent_tris) * 3
    assert {origin_triangle(v) for v in suffix.tolist()} == translucent_tris

    # (4) every batch's range, read through vertex_order, agrees with the
    # (front, back) pair of the triangles that actually live there. This is
    # the assertion that ties vertex_order to the batch lists.
    for batch in plan.opaque + plan.translucent:
        segment = reordered[batch.first : batch.first + batch.count]
        for value in segment.tolist():
            assert pairs[origin_triangle(value)] == (batch.front_material_id, batch.back_material_id)


def test_material_ids_up_to_the_2_20_minus_1_boundary_sort_correctly():
    # face_batches.py packs (is_translucent, front, back) into one int64 sort
    # key with _ID_BITS = 20 bits per side. This pins the documented limit:
    # a material id up to 2**20 - 1 must sort and batch correctly. (An id at
    # or above 2**20 is a known latent bug, not covered here — see the Task 4
    # fix report.)
    near_limit = (1 << 20) - 1
    plan = plan_face_batches([near_limit, 5], [0, 0])
    assert len(plan.opaque) == 2
    by_pair = {(b.front_material_id, b.back_material_id): b for b in plan.opaque}
    assert set(by_pair) == {(near_limit, 0), (5, 0)}
    assert by_pair[(5, 0)].count == 3
    assert by_pair[(near_limit, 0)].count == 3
    # the smaller id must sort first
    assert by_pair[(5, 0)].first < by_pair[(near_limit, 0)].first
