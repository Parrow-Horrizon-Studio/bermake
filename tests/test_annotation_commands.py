from __future__ import annotations

import pytest
from bermake.commands.annotation_commands import (
    CreateAnnotationCommand,
    DeleteAnnotationsCommand,
    EditLabelTextCommand,
    MoveAnnotationsCommand,
)
from bermake.model.annotation import Dimension, Label
from bermake.model.model import Model


def _dim(model):
    return Dimension(model.new_annotation_id(), (0, 0, 0), (4, 0, 0), (0, -1, 0))


def _label(model, text="note"):
    return Label(model.new_annotation_id(), (0, 0, 0), (2, 2, 0), text)


def test_create_adds_and_undo_removes():
    model = Model()
    ctx = model.active_context
    cmd = CreateAnnotationCommand(_dim(model), ctx)
    cmd.do(model)
    assert len(ctx.annotations) == 1
    cmd.undo(model)
    assert len(ctx.annotations) == 0
    cmd.do(model)   # redo re-adds the same object
    assert len(ctx.annotations) == 1


def test_delete_removes_and_undo_restores():
    model = Model()
    ctx = model.active_context
    ann = _dim(model)
    ctx.annotations.append(ann)
    cmd = DeleteAnnotationsCommand([ann.id], ctx)
    cmd.do(model)
    assert ctx.annotations == []
    cmd.undo(model)
    assert len(ctx.annotations) == 1
    assert ctx.annotations[0].id == ann.id


def test_edit_label_text_and_undo_restores_old_text():
    model = Model()
    ctx = model.active_context
    lab = _label(model, "before")
    ctx.annotations.append(lab)
    cmd = EditLabelTextCommand(lab.id, "after", ctx)
    cmd.do(model)
    assert ctx.annotations[0].text == "after"
    cmd.undo(model)
    assert ctx.annotations[0].text == "before"


def test_move_shifts_dimension_offset_and_label_text_pos():
    model = Model()
    ctx = model.active_context
    dim = _dim(model)
    lab = _label(model)
    ctx.annotations.extend([dim, lab])
    cmd = MoveAnnotationsCommand([dim.id, lab.id], (0.0, 0.0, 1.0), ctx)
    cmd.do(model)
    assert ctx.annotations[0].offset == (0.0, -1.0, 1.0)
    assert ctx.annotations[1].text_pos == (2.0, 2.0, 1.0)
    assert ctx.annotations[1].anchor == (0.0, 0.0, 0.0)   # anchor stays put
    cmd.undo(model)
    assert ctx.annotations[0].offset == (0.0, -1.0, 0.0)
    assert ctx.annotations[1].text_pos == (2.0, 2.0, 0.0)


def test_delete_middle_element_restores_at_original_index():
    """Verify deletion of non-last element restores at correct index."""
    model = Model()
    ctx = model.active_context
    ann1 = _dim(model)
    ann2 = _dim(model)
    ann3 = _dim(model)
    ctx.annotations.extend([ann1, ann2, ann3])
    cmd = DeleteAnnotationsCommand([ann2.id], ctx)
    cmd.do(model)
    assert len(ctx.annotations) == 2
    assert ctx.annotations[0].id == ann1.id
    assert ctx.annotations[1].id == ann3.id
    cmd.undo(model)
    assert len(ctx.annotations) == 3
    assert ctx.annotations[0].id == ann1.id
    assert ctx.annotations[1].id == ann2.id
    assert ctx.annotations[2].id == ann3.id
    cmd.do(model)
    assert len(ctx.annotations) == 2
    cmd.undo(model)
    assert len(ctx.annotations) == 3
    assert ctx.annotations[1].id == ann2.id  # restored at index 1


def test_move_undo_restores_true_original_despite_external_mutation():
    """Regression for review Finding 1.

    undo() must restore the absolute original value it captured on do(), not
    derive "original" by negating the delta from whatever the field currently
    holds. If anything else mutates the annotation's offset/text_pos between
    do() and undo(), a delta-negate undo silently produces a bogus value.
    """
    model = Model()
    ctx = model.active_context
    dim = _dim(model)  # offset starts at (0.0, -1.0, 0.0)
    ctx.annotations.append(dim)
    cmd = MoveAnnotationsCommand([dim.id], (5.0, 5.0, 4.0), ctx)
    cmd.do(model)
    assert ctx.annotations[0].offset == pytest.approx((5.0, 4.0, 4.0))

    # Something else mutates the annotation's offset between do() and undo()
    # (a drag preview, an unrelated command, whatever) -- undo must not care.
    ctx.annotations[0].offset = (99.0, 99.0, 99.0)

    cmd.undo(model)
    assert ctx.annotations[0].offset == pytest.approx((0.0, -1.0, 0.0))


def test_create_annotation_membership_is_identity_based():
    """Regression for review Finding 2.

    Two field-for-field-equal (but distinct-object) annotations must be
    told apart by identity, not by dataclass value equality. Before the
    fix, `in`/`list.remove()` resolved through the auto-generated `__eq__`
    and could add/remove the wrong object.
    """
    model = Model()
    ctx = model.active_context
    shared_id = model.new_annotation_id()
    twin_a = Dimension(shared_id, (0, 0, 0), (4, 0, 0), (0, -1, 0))
    twin_b = Dimension(shared_id, (0, 0, 0), (4, 0, 0), (0, -1, 0))
    assert twin_a == twin_b        # value-equal ...
    assert twin_a is not twin_b    # ... but distinct objects

    ctx.annotations.append(twin_b)  # a value-equal annotation already present
    cmd = CreateAnnotationCommand(twin_a, ctx)
    cmd.do(model)
    assert len(ctx.annotations) == 2
    assert ctx.annotations[0] is twin_b
    assert ctx.annotations[1] is twin_a

    cmd.undo(model)
    # Must remove the exact object it added (twin_a), not the first
    # value-equal match (twin_b).
    assert len(ctx.annotations) == 1
    assert ctx.annotations[0] is twin_b


def test_create_annotation_in_nested_definition_leaves_root_untouched():
    """Per-context storage: target_context can be a nested Definition."""
    model = Model()
    root = model.root
    nested = model.new_definition("Group", is_group=True)
    ann = _dim(model)
    cmd = CreateAnnotationCommand(ann, nested)
    cmd.do(model)
    assert nested.annotations == [ann]
    assert root.annotations == []
    assert model.active_context is root   # nested is not the active context

    cmd.undo(model)
    assert nested.annotations == []
    assert root.annotations == []


def test_delete_multiple_noncontiguous_restores_at_original_indices():
    """Multi-id delete: non-contiguous, not-last targets, each restored at
    its original index on undo."""
    model = Model()
    ctx = model.active_context
    anns = [_dim(model) for _ in range(5)]
    ctx.annotations.extend(anns)
    # Delete indices 0, 2, 3 -- non-contiguous, and index 4 (last) survives.
    ids_to_delete = [anns[0].id, anns[2].id, anns[3].id]
    cmd = DeleteAnnotationsCommand(ids_to_delete, ctx)
    cmd.do(model)
    assert [a.id for a in ctx.annotations] == [anns[1].id, anns[4].id]

    cmd.undo(model)
    assert len(ctx.annotations) == 5
    for i, ann in enumerate(anns):
        assert ctx.annotations[i] is ann   # every one back at its own index


def test_edit_label_text_do_undo_do_undo_cycle():
    """Genuine redo cycle, and the second undo() must still restore the
    TRUE original text -- not something derived from the redo."""
    model = Model()
    ctx = model.active_context
    lab = _label(model, "original")
    ctx.annotations.append(lab)
    cmd = EditLabelTextCommand(lab.id, "edited", ctx)

    cmd.do(model)
    assert ctx.annotations[0].text == "edited"
    cmd.undo(model)
    assert ctx.annotations[0].text == "original"
    cmd.do(model)   # redo
    assert ctx.annotations[0].text == "edited"
    cmd.undo(model)   # second undo
    assert ctx.annotations[0].text == "original"


def test_edit_label_text_undo_restores_true_original_despite_external_mutation():
    """Regression for review Finding 3.

    undo() must restore the absolute original value it captured on do(), not
    derive "original" by rereading the current text on each do(). If anything
    else mutates the annotation's text between undo() and a following do(),
    recomputing the "old" text on the second do() silently captures the
    externally-mutated value, and the subsequent undo() produces a bogus result.
    """
    model = Model()
    ctx = model.active_context
    lab = _label(model, "original")
    ctx.annotations.append(lab)
    cmd = EditLabelTextCommand(lab.id, "edited", ctx)

    cmd.do(model)
    assert ctx.annotations[0].text == "edited"

    cmd.undo(model)
    assert ctx.annotations[0].text == "original"

    # Something else mutates the annotation's text between undo() and do()
    # (a UI edit, an unrelated command, whatever) -- undo must not care.
    ctx.annotations[0].text = "externally-mutated"

    cmd.do(model)   # redo
    assert ctx.annotations[0].text == "edited"

    cmd.undo(model)   # second undo
    assert ctx.annotations[0].text == "original"


def test_move_do_undo_do_undo_cycle():
    """Genuine redo cycle for MoveAnnotationsCommand, multiple ids at once,
    pinning per-kind semantics (dimension offset vs. label text_pos/anchor)
    across the full cycle."""
    model = Model()
    ctx = model.active_context
    dim = _dim(model)
    lab = _label(model)
    ctx.annotations.extend([dim, lab])
    cmd = MoveAnnotationsCommand([dim.id, lab.id], (1.0, 2.0, 0.0), ctx)

    cmd.do(model)
    assert ctx.annotations[0].offset == pytest.approx((1.0, 1.0, 0.0))
    assert ctx.annotations[1].text_pos == pytest.approx((3.0, 4.0, 0.0))
    assert ctx.annotations[1].anchor == pytest.approx((0.0, 0.0, 0.0))

    cmd.undo(model)
    assert ctx.annotations[0].offset == pytest.approx((0.0, -1.0, 0.0))
    assert ctx.annotations[1].text_pos == pytest.approx((2.0, 2.0, 0.0))

    cmd.do(model)   # redo
    assert ctx.annotations[0].offset == pytest.approx((1.0, 1.0, 0.0))
    assert ctx.annotations[1].text_pos == pytest.approx((3.0, 4.0, 0.0))

    cmd.undo(model)   # second undo
    assert ctx.annotations[0].offset == pytest.approx((0.0, -1.0, 0.0))
    assert ctx.annotations[1].text_pos == pytest.approx((2.0, 2.0, 0.0))
    assert ctx.annotations[1].anchor == pytest.approx((0.0, 0.0, 0.0))
