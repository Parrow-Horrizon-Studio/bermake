"""Loop-and-UV arithmetic for carrying stored UVs across an edge split."""

from bermake.scene.uv_transfer import transfer_uvs_across_split


def test_inserted_uv_is_lerped_at_the_split_parameter():
    # Loop 10, 11, 12, 13 with w=99 inserted between va=10 and vb=11.
    old_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    result = transfer_uvs_across_split(
        old_uvs,
        old_loop=[10, 11, 12, 13],
        new_loop=[10, 99, 11, 12, 13],
        w=99,
        va=10,
        vb=11,
        t=0.25,
    )
    assert result is not None
    assert len(result) == 5
    # 0.25 of the way from (0,0) toward (1,0).
    assert result[1] == (0.25, 0.0)
    # Every original corner keeps its own UV, in the new loop's order.
    assert result[0] == (0.0, 0.0)
    assert result[2] == (1.0, 0.0)
    assert result[3] == (1.0, 1.0)
    assert result[4] == (0.0, 1.0)


def test_a_loop_traversing_the_edge_backwards_uses_one_minus_t():
    # Same split (t measured from va=10 toward vb=11), but this face's loop
    # runs 11 -> 10, so w sits 0.75 of the way along the loop's direction.
    old_uvs = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0)]
    result = transfer_uvs_across_split(
        old_uvs,
        old_loop=[11, 10, 12],
        new_loop=[11, 99, 10, 12],
        w=99,
        va=10,
        vb=11,
        t=0.25,
    )
    assert result is not None
    # From (1,0) toward (0,0) at fraction 0.75.
    assert result[1] == (0.25, 0.0)


def test_insertion_at_the_loop_wrap_point():
    # va is the LAST loop entry and vb the first, so w lands at the very end.
    old_uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    result = transfer_uvs_across_split(
        old_uvs,
        old_loop=[10, 11, 12],
        new_loop=[10, 11, 12, 99],
        w=99,
        va=12,
        vb=10,
        t=0.5,
    )
    assert result is not None
    assert result[3] == (0.5, 0.5)


def test_returns_none_when_the_new_loop_is_not_one_insertion():
    assert (
        transfer_uvs_across_split(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            old_loop=[10, 11, 12],
            new_loop=[10, 11, 12, 99, 98],
            w=99,
            va=10,
            vb=11,
            t=0.5,
        )
        is None
    )


def test_returns_none_when_the_uv_count_does_not_match_the_old_loop():
    assert (
        transfer_uvs_across_split(
            [(0.0, 0.0), (1.0, 0.0)],
            old_loop=[10, 11, 12],
            new_loop=[10, 99, 11, 12],
            w=99,
            va=10,
            vb=11,
            t=0.5,
        )
        is None
    )


def test_returns_none_when_w_is_absent_from_the_new_loop():
    assert (
        transfer_uvs_across_split(
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
            old_loop=[10, 11, 12],
            new_loop=[10, 11, 12, 77],
            w=99,
            va=10,
            vb=11,
            t=0.5,
        )
        is None
    )
