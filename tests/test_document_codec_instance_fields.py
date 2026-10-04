"""Instance name + hidden round-trip, and v3 files still open (M7.3 Task 4)."""

from __future__ import annotations

import numpy as np
from bermake.io.document_codec import model_from_dict, model_to_dict
from bermake.io.bermake_file import SCHEMA_VERSION


def _square(model):
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)


def test_schema_version_is_at_least_four():
    # Pinned to 4 by M7.3; M7.5a (Task 12) bumped it again to 5. This file's
    # concern is instance name/hidden (schema 3 -> 4), not the exact current
    # value, so assert the floor rather than re-pinning a number that isn't
    # this test's story. The exact current value is pinned in
    # tests/test_schema_compatibility.py::test_the_schema_version_and_floor_are_pinned.
    assert SCHEMA_VERSION >= 4


def test_name_and_hidden_survive_a_round_trip(model_factory, group_factory):
    model = model_factory()
    _square(model)
    instance = group_factory(model)
    instance.name = "North Wing"
    instance.hidden = True

    restored = model_from_dict(model_to_dict(model))

    child = restored.root.children[0]
    assert child.name == "North Wing"
    assert child.hidden is True


def test_defaults_survive_a_round_trip(model_factory, group_factory):
    model = model_factory()
    _square(model)
    group_factory(model)

    restored = model_from_dict(model_to_dict(model))

    child = restored.root.children[0]
    assert child.name == ""
    assert child.hidden is False


def test_a_v3_document_without_the_new_keys_still_opens(model_factory, group_factory):
    # Forward compatibility is not the concern here -- BACKWARD is. A file
    # written by v0.4.0 has no "name"/"hidden" on its children and must load
    # with both defaulted rather than raising KeyError.
    model = model_factory()
    _square(model)
    group_factory(model)
    data = model_to_dict(model)
    for definition in data["definitions"]:
        for child in definition["children"]:
            child.pop("name", None)
            child.pop("hidden", None)

    restored = model_from_dict(data)

    child = restored.root.children[0]
    assert child.name == ""
    assert child.hidden is False
