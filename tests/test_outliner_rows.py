"""Pure Outliner row derivation (M7.3 Task 5)."""

from __future__ import annotations

import numpy as np
from bermake.model.model_queries import instance_path, outliner_rows


def _square(model):
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def test_an_empty_model_has_no_rows(model_factory):
    assert outliner_rows(model_factory()) == ()


def test_one_group_is_one_row_at_depth_zero(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)

    (row,) = outliner_rows(model)

    assert row.instance_id == instance.id
    assert row.depth == 0


def test_nesting_increases_depth_and_keeps_parents_first(model_factory, group_factory):
    # group_factory wraps whatever live geometry sits in the active context;
    # calling it twice back-to-back at the same context produces siblings
    # (the second call finds the mesh already emptied by the first), not
    # nesting. Genuine nesting requires entering the outer group before
    # adding the geometry that becomes the inner one.
    model = model_factory()
    _square(model)
    outer = group_factory(model)
    model.enter(outer)
    _square(model)
    inner = group_factory(model)
    model.exit_one()

    rows = outliner_rows(model)

    assert [r.instance_id for r in rows] == [outer.id, inner.id]
    assert [r.depth for r in rows] == [0, 1]


def test_the_label_falls_back_to_the_definition_name(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    instance.definition.name = "Wall"

    (row,) = outliner_rows(model)
    assert row.label == "Wall"

    instance.name = "North Wing"
    (row,) = outliner_rows(model)
    assert row.label == "North Wing"


def test_a_blank_instance_name_still_falls_back(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    instance.definition.name = "Roof"
    instance.name = ""

    (row,) = outliner_rows(model)

    assert row.label == "Roof"


def test_hidden_is_own_flag_and_inherited_hidden_propagates(model_factory, group_factory):
    # Real nesting is required here: inherited_hidden only propagates down an
    # ancestor chain, so siblings (see the comment in the nesting test above)
    # would never exercise it.
    model = model_factory()
    _square(model)
    outer = group_factory(model)
    model.enter(outer)
    _square(model)
    inner = group_factory(model)
    model.exit_one()
    outer.hidden = True

    by_id = {r.instance_id: r for r in outliner_rows(model)}

    assert by_id[outer.id].hidden is True
    assert by_id[outer.id].inherited_hidden is False
    # The child's own flag is untouched; only the inherited one moved. This is
    # what lets the child's eye toggle keep meaning something.
    assert by_id[inner.id].hidden is False
    assert by_id[inner.id].inherited_hidden is True


def test_inherited_hidden_reaches_a_grandchild_through_a_parent(model_factory, group_factory):
    # Three genuine levels: outer > middle > inner. The recursive walk ORs
    # ancestor_hidden down through each call (`ancestor_hidden or hidden`), so
    # a bug that only propagated one level (e.g. passing `hidden` instead of
    # the accumulated `ancestor_hidden or hidden`) would show up only here --
    # the two-level test above can't distinguish "propagates one level" from
    # "propagates all the way down". Depth is checked at all three levels too,
    # since Task 5 never had a test past depth 1.
    model = model_factory()
    _square(model)
    outer = group_factory(model)
    model.enter(outer)
    _square(model)
    middle = group_factory(model)
    model.enter(middle)
    _square(model)
    inner = group_factory(model)
    model.exit_one()
    model.exit_one()
    outer.hidden = True

    by_id = {r.instance_id: r for r in outliner_rows(model)}

    assert by_id[outer.id].depth == 0
    assert by_id[middle.id].depth == 1
    assert by_id[inner.id].depth == 2

    assert by_id[outer.id].hidden is True
    assert by_id[outer.id].inherited_hidden is False

    # Neither descendant's own flag moved; only inherited_hidden did, and it
    # reaches the grandchild exactly like it reaches the immediate child.
    assert by_id[middle.id].hidden is False
    assert by_id[middle.id].inherited_hidden is True

    assert by_id[inner.id].hidden is False
    assert by_id[inner.id].inherited_hidden is True


def test_tag_hidden_is_independent_of_hidden(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    tag = model.tags.add("Walls")
    instance.tag_id = tag.id
    model.tags.set_visible(tag.id, False)

    (row,) = outliner_rows(model)

    assert row.tag_hidden is True
    assert row.hidden is False
    assert row.inherited_hidden is False


def test_a_group_is_not_a_component(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)

    (row,) = outliner_rows(model)

    assert row.is_component is (not instance.definition.is_group)


def test_on_active_path_marks_the_entered_chain(model_factory, group_factory):
    model = model_factory()
    _square(model)
    inner = group_factory(model)
    outer = group_factory(model)
    model.enter(outer)

    by_id = {r.instance_id: r for r in outliner_rows(model)}

    assert by_id[outer.id].on_active_path is True
    assert by_id[inner.id].on_active_path is False


def test_instance_path_returns_the_root_first_ancestor_chain(model_factory, group_factory):
    # Requires real nesting: the ancestor chain for `inner` is only
    # (outer, inner) when inner actually lives inside outer's definition.
    model = model_factory()
    _square(model)
    outer = group_factory(model)
    model.enter(outer)
    _square(model)
    inner = group_factory(model)
    model.exit_one()

    assert instance_path(model, outer.id) == (outer,)
    assert instance_path(model, inner.id) == (outer, inner)


def test_instance_path_is_none_for_an_unknown_id(model_factory):
    assert instance_path(model_factory(), 4242) is None
