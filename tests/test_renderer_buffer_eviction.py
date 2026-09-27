"""Headless test for M7.1 Task 16: evict GL buffers for definitions no longer
reachable from the model (#59).

The per-definition buffer cache (_def_buffers) is keyed by id(definition) and
is append-only unless something reconciles it against the live model. This
test exercises evict_unreachable() without a GL context: the stand-in
_DefBuffers() has all-zero handles, so the guarded release path is a no-op.
"""

from __future__ import annotations

from bermake.model.model import Model
from bermake.viewport.scene_renderer import SceneRenderer, _DefBuffers


def test_cache_drops_definitions_no_longer_in_the_model():
    """The buffer cache must not retain definitions that left the model."""
    model = Model()
    defn = model.new_definition("Temp", is_group=True)
    inst = model.new_instance(defn, None)
    model.root.children.append(inst)

    r = SceneRenderer()
    assert r._initialized is False  # no GL context needed to construct
    r._def_buffers[id(defn)] = _DefBuffers()  # stand-in; all handles are 0

    model.root.children.remove(inst)  # defn is now unreachable (no instance)
    r.evict_unreachable(model)

    assert id(defn) not in r._def_buffers


def test_reachable_definitions_keep_their_buffers():
    """A definition still instantiated anywhere must not be evicted, even if
    every instance is on a hidden tag (traverse(), not traverse_visible())."""
    model = Model()
    defn = model.new_definition("Temp", is_group=True)
    inst = model.new_instance(defn, None)
    model.root.children.append(inst)

    r = SceneRenderer()
    r._def_buffers[id(defn)] = _DefBuffers()
    r._def_buffers[id(model.root)] = _DefBuffers()

    r.evict_unreachable(model)

    assert id(defn) in r._def_buffers
    assert id(model.root) in r._def_buffers
