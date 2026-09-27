from __future__ import annotations

import numpy as np
from bermake.model.model import Model


def _child(m, parent, tag_id=0):
    d = m.new_definition("D", is_group=True)
    inst = m.new_instance(d)
    inst.tag_id = tag_id
    parent.children.append(inst)
    return inst, d


def test_no_hidden_tags_matches_traverse():
    m = Model()
    _child(m, m.root)
    _child(m, m.root)
    plain = [id(d) for d, _ in m.traverse()]
    visible = [id(d) for d, _ in m.traverse_visible()]
    assert plain == visible


def test_hidden_tag_prunes_instance():
    m = Model()
    walls = m.tags.add("Walls")
    _a, da = _child(m, m.root, tag_id=walls.id)
    m.tags.set_visible(walls.id, False)
    defs = [d for d, _ in m.traverse_visible()]
    assert da not in defs
    assert m.root in defs


def test_hidden_tag_prunes_subtree():
    m = Model()
    walls = m.tags.add("Walls")
    _a, da = _child(m, m.root, tag_id=walls.id)
    _c, dc = _child(m, da, tag_id=0)            # visible child inside hidden parent
    m.tags.set_visible(walls.id, False)
    defs = [d for d, _ in m.traverse_visible()]
    assert da not in defs
    assert dc not in defs                       # subtree pruned even though child is Untagged


def test_active_path_instance_bypasses_hidden():
    m = Model()
    walls = m.tags.add("Walls")
    a, da = _child(m, m.root, tag_id=walls.id)
    m.tags.set_visible(walls.id, False)
    m.enter(a)                                  # editing inside a
    defs = [d for d, _ in m.traverse_visible()]
    assert da in defs


def test_pick_instance_skips_hidden():
    m = Model()
    walls = m.tags.add("Walls")
    a, da = _child(m, m.root, tag_id=walls.id)
    v = [da.mesh.add_vertex(np.array([0.0, 0.0, 0.0])),
         da.mesh.add_vertex(np.array([1.0, 0.0, 0.0])),
         da.mesh.add_vertex(np.array([0.0, 1.0, 0.0]))]
    da.mesh.add_face_from_loop(v)
    origin = np.array([0.25, 0.25, 1.0])
    direction = np.array([0.0, 0.0, -1.0])
    assert m.pick_instance(origin, direction) is a      # visible → hit
    m.tags.set_visible(walls.id, False)
    assert m.pick_instance(origin, direction) is None    # hidden → skipped
