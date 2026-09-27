"""The selection controller is Qt-free (M7.3 Task 1)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from bermake.commands.command_stack import CommandStack
from bermake.selection import Selection
from bermake.ui import selection_controller as sc


def _square(model):
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def test_importing_the_controller_loads_no_qt(model_factory):
    # The whole point of the extraction: this logic must be testable without
    # a QApplication, so it may not drag PySide6 in transitively.
    code = (
        "import sys; import bermake.ui.selection_controller; "
        "print(any(m.startswith('PySide6') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_select_all_then_none(model_factory):
    model = model_factory()
    _square(model)
    selection = Selection()

    sc.select_all(model, selection)
    assert len(selection.faces) == 1
    assert len(selection.edges) == 4

    sc.select_none(selection)
    assert selection.is_empty()


def test_status_text_counts_each_kind(model_factory):
    model = model_factory()
    _square(model)
    selection = Selection()
    sc.select_all(model, selection)
    text = sc.selection_status_text(selection)
    assert "4 edges" in text
    assert "1 face" in text
    assert text.endswith(" selected")


def test_status_text_is_empty_for_an_empty_selection():
    assert sc.selection_status_text(Selection()) == ""


def test_prune_drops_ids_that_are_no_longer_live(model_factory):
    model = model_factory()
    _square(model)
    selection = Selection()
    sc.select_all(model, selection)
    ghost = max(selection.faces) + 99
    selection.add(faces=[ghost])

    sc.prune_to_live(model, selection)

    assert ghost not in selection.faces
    assert len(selection.faces) == 1


def test_assign_tag_reports_refusal_when_nothing_is_selected(model_factory):
    model = model_factory()
    did, message = sc.assign_tag(model, Selection(), CommandStack(), 0)
    assert did is False
    assert "Select objects" in message


def test_assign_tag_applies_to_selected_instances(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    selection = Selection()
    selection.replace(instances=[instance.id])
    stack = CommandStack()
    tag = model.tags.add("Walls")

    did, message = sc.assign_tag(model, selection, stack, tag.id)

    assert did is True
    assert "Walls" in message
    assert instance.tag_id == tag.id
    assert stack.can_undo


def test_selection_tag_label_reports_multiple(model_factory, group_factory):
    model = model_factory()
    _square(model)
    first = group_factory(model)
    _square(model)
    second = group_factory(model)
    tag = model.tags.add("Walls")
    second.tag_id = tag.id
    selection = Selection()
    selection.replace(instances=[first.id, second.id])

    assert sc.selection_tag_label(model, selection) == "(multiple)"


def test_selection_tag_label_is_none_without_instances(model_factory):
    assert sc.selection_tag_label(model_factory(), Selection()) is None
