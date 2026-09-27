from __future__ import annotations

import numpy as np
import pytest
from bermake.ui.main_window import MainWindow
from bermake.ui.tags_page import TagsPage


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_has_tags_page(win):
    assert isinstance(win._tags_page, TagsPage)


def test_active_tag_tracks_page(win):
    walls = win._model.tags.add("Walls")
    win._tags_page._rebuild()
    win._tags_page.set_active(walls.id)
    assert win._active_tag_id == walls.id


def test_assign_tags_selected_instances(win):
    s = win._model.active_scene
    v = [s.add_vertex(np.array([0.0, 0.0, 0.0])),
         s.add_vertex(np.array([1.0, 0.0, 0.0])),
         s.add_vertex(np.array([0.0, 1.0, 0.0]))]
    f = s.add_face_from_loop(v)
    win._selection.replace(faces=[f])
    win._on_make_group()                          # groups the face; selects the new instance
    walls = win._model.tags.add("Walls")
    win._active_tag_id = walls.id
    win._on_assign_tag()
    inst_id = next(iter(win._selection.instances))
    inst = next(i for i in win._model.active_context.children if i.id == inst_id)
    assert inst.tag_id == walls.id


def test_new_group_inherits_active_tag(win):
    walls = win._model.tags.add("Walls")
    win._active_tag_id = walls.id
    s = win._model.active_scene
    v = [s.add_vertex(np.array([0.0, 0.0, 0.0])),
         s.add_vertex(np.array([1.0, 0.0, 0.0])),
         s.add_vertex(np.array([0.0, 1.0, 0.0]))]
    f = s.add_face_from_loop(v)
    win._selection.replace(faces=[f])
    win._on_make_group()
    inst_id = next(iter(win._selection.instances))
    inst = next(i for i in win._model.active_context.children if i.id == inst_id)
    assert inst.tag_id == walls.id


def test_selection_indicator_reflects_selected_object(win):
    s = win._model.active_scene
    v = [s.add_vertex(np.array([0.0, 0.0, 0.0])),
         s.add_vertex(np.array([1.0, 0.0, 0.0])),
         s.add_vertex(np.array([0.0, 1.0, 0.0]))]
    f = s.add_face_from_loop(v)
    walls = win._model.tags.add("Walls")
    win._active_tag_id = walls.id
    win._selection.replace(faces=[f])
    win._on_make_group()                      # new group inherits Walls; selects the instance
    win._update_selection_tag_indicator()
    assert win._tags_page._selection_label.text() == "Selection: Walls"


def test_selection_indicator_empty_when_no_instance(win):
    win._selection.replace()                  # clear selection
    win._update_selection_tag_indicator()
    assert win._tags_page._selection_label.text() == "Selection: —"


def test_assign_updates_selection_indicator(win):
    s = win._model.active_scene
    v = [s.add_vertex(np.array([0.0, 0.0, 0.0])),
         s.add_vertex(np.array([1.0, 0.0, 0.0])),
         s.add_vertex(np.array([0.0, 1.0, 0.0]))]
    f = s.add_face_from_loop(v)
    win._selection.replace(faces=[f])
    win._on_make_group()                      # untagged group, selected
    walls = win._model.tags.add("Walls")
    win._active_tag_id = walls.id
    win._on_assign_tag()
    assert win._tags_page._selection_label.text() == "Selection: Walls"
