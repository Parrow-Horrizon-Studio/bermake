"""Undoable annotation commands (M7d)."""

from __future__ import annotations

from bermake.commands.command import Command


def _find(context, annotation_id):
    for ann in context.annotations:
        if ann.id == annotation_id:
            return ann
    return None


def _shift(point, delta):
    return (point[0] + delta[0], point[1] + delta[1], point[2] + delta[2])


class CreateAnnotationCommand(Command):
    """Append one annotation to a context; undo detaches the same object.

    Membership is checked by identity (`is`), not `==`. Annotation entities
    are dataclasses, so `in`/`list.remove()` would resolve through the
    auto-generated value-equality `__eq__` and could add/remove the wrong
    object if two annotations ever happen to be field-for-field equal.
    """

    name = "Create Annotation"

    def __init__(self, annotation, target_context) -> None:
        self._annotation = annotation
        self._target = target_context

    def do(self, model) -> None:
        annotations = self._target.annotations
        if not any(a is self._annotation for a in annotations):
            annotations.append(self._annotation)

    def undo(self, model) -> None:
        annotations = self._target.annotations
        for index, a in enumerate(annotations):
            if a is self._annotation:
                del annotations[index]
                break


class DeleteAnnotationsCommand(Command):
    """Remove annotations by id; undo restores them at their original indices."""

    name = "Delete Annotations"

    def __init__(self, annotation_ids, target_context) -> None:
        self._ids = list(annotation_ids)
        self._target = target_context
        self._removed = []  # (index, annotation), ascending by index

    def do(self, model) -> None:
        self._removed = []
        wanted = set(self._ids)
        for index, ann in enumerate(list(self._target.annotations)):
            if ann.id in wanted:
                self._removed.append((index, ann))
        for _index, ann in self._removed:
            self._target.annotations.remove(ann)

    def undo(self, model) -> None:
        for index, ann in self._removed:
            self._target.annotations.insert(index, ann)
        self._removed = []


class EditLabelTextCommand(Command):
    """Replace a Label's text; undo restores the previous string."""

    name = "Edit Label Text"

    def __init__(self, annotation_id, new_text, target_context) -> None:
        self._id = int(annotation_id)
        self._new_text = str(new_text)
        self._target = target_context
        self._old_text = None

    def do(self, model) -> None:
        ann = _find(self._target, self._id)
        if ann is None or getattr(ann, "kind", None) != "label":
            return
        if self._old_text is None:
            self._old_text = ann.text
        ann.text = self._new_text

    def undo(self, model) -> None:
        if self._old_text is None:
            return
        ann = _find(self._target, self._id)
        if ann is not None:
            ann.text = self._old_text


_MOVE_FIELD_BY_KIND = {"dimension": "offset", "label": "text_pos"}


class MoveAnnotationsCommand(Command):
    """Translate annotations by a local delta.

    A Dimension's `offset` shifts (moving the dimension line); a Label's
    `text_pos` shifts while its `anchor` stays put so the leader re-aims.

    On the first `do()`, captures each affected annotation's absolute
    original field value and computes the absolute new value from the
    delta. do()/undo() thereafter are plain absolute writes of the
    captured (field, old, new) tuples -- re-entrant like
    TransformVerticesCommand, and immune to anything else mutating the
    field between do() and undo() (a delta-negate undo is not).
    """

    name = "Move Annotations"

    def __init__(self, annotation_ids, delta, target_context) -> None:
        self._ids = list(annotation_ids)
        self._delta = (float(delta[0]), float(delta[1]), float(delta[2]))
        self._target = target_context
        self._moves: dict[int, tuple[str, tuple, tuple]] = {}

    def do(self, model) -> None:
        if not self._moves:
            wanted = set(self._ids)
            for ann in self._target.annotations:
                if ann.id not in wanted:
                    continue
                field = _MOVE_FIELD_BY_KIND.get(getattr(ann, "kind", None))
                if field is None:
                    continue
                old = getattr(ann, field)
                new = _shift(old, self._delta)
                self._moves[ann.id] = (field, old, new)
        for ann in self._target.annotations:
            move = self._moves.get(ann.id)
            if move is None:
                continue
            field, _old, new = move
            setattr(ann, field, new)

    def undo(self, model) -> None:
        for ann in self._target.annotations:
            move = self._moves.get(ann.id)
            if move is None:
                continue
            field, old, _new = move
            setattr(ann, field, old)
