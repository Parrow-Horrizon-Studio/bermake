import numpy as np
import pytest

from bermake.io.document_codec import geometry_from_dict, geometry_to_dict
from bermake.io.errors import BermakeFormatError
from bermake.scene.scene import Scene, Side


def _quad(scene, z=0.0):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    ]
    return scene.add_face_from_loop(ids)


def test_a_document_with_no_stored_uvs_writes_empty_dicts():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    assert data["face_uvs"] == {}
    assert data["face_uvs_back"] == {}


def test_stored_uvs_round_trip():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], Side.FRONT)
    data = geometry_to_dict(s)
    assert data["face_uvs"] == {"0": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]}

    loaded = Scene()
    geometry_from_dict(loaded, data)
    new_face = next(iter(loaded.faces_iter())).id
    np.testing.assert_allclose(
        loaded.face_uvs(new_face, Side.FRONT), [[0, 0], [1, 0], [1, 1], [0, 1]]
    )


def test_the_two_sides_round_trip_independently():
    s = Scene()
    f = _quad(s)
    s.set_face_uvs(f, [(0.0, 0.0)] * 4, Side.FRONT)
    s.set_face_uvs(f, [(0.5, 0.5)] * 4, Side.BACK)
    loaded = Scene()
    geometry_from_dict(loaded, geometry_to_dict(s))
    new_face = next(iter(loaded.faces_iter())).id
    np.testing.assert_allclose(loaded.face_uvs(new_face, Side.FRONT), np.zeros((4, 2)))
    np.testing.assert_allclose(loaded.face_uvs(new_face, Side.BACK), np.full((4, 2), 0.5))


def test_a_v0_8_0_document_without_the_keys_still_loads():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    del data["face_uvs"]
    del data["face_uvs_back"]
    loaded = Scene()
    geometry_from_dict(loaded, data)  # must not raise
    assert loaded.faces_with_uvs() == []


def test_a_wrong_length_uv_array_is_corruption_not_a_fallback():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    data["face_uvs"] = {"0": [0.0, 0.0, 1.0, 0.0]}  # 2 corners for a 4-corner face
    with pytest.raises(BermakeFormatError, match="face_uvs"):
        geometry_from_dict(Scene(), data)


def test_an_odd_length_uv_array_is_rejected():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    data["face_uvs"] = {"0": [0.0, 0.0, 1.0]}
    with pytest.raises(BermakeFormatError, match="face_uvs"):
        geometry_from_dict(Scene(), data)


def test_an_out_of_range_face_index_is_rejected():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    data["face_uvs"] = {"9": [0.0, 0.0] * 4}
    with pytest.raises(BermakeFormatError, match="face_uvs"):
        geometry_from_dict(Scene(), data)


def test_a_non_integer_face_index_is_rejected():
    s = Scene()
    _quad(s)
    data = geometry_to_dict(s)
    data["face_uvs"] = {"abc": [0.0, 0.0] * 4}
    with pytest.raises(BermakeFormatError, match="face_uvs"):
        geometry_from_dict(Scene(), data)


def test_stored_uvs_survive_when_face_ids_and_indices_diverge():
    # face_uvs is keyed by INDEX (position in faces_iter()), not by the kernel's
    # face id, same as face_materials. A single-face fixture can't tell an
    # index-keyed implementation from an id-keyed one, because face 0 has index 0
    # either way. Build three quads, remove the first, so the two surviving faces
    # keep their original (non-zero) kernel ids while their positions in
    # faces_iter() compact down to 0 and 1. Give the two survivors clearly
    # different arrays so a swap on the way back fails this test instead of
    # passing by symmetry.
    s = Scene()
    f0 = _quad(s, z=0.0)
    f1 = _quad(s, z=1.0)
    f2 = _quad(s, z=2.0)
    s.remove_face(f0)

    remaining_ids = [f.id for f in s.faces_iter()]
    assert remaining_ids == [f1, f2]
    # The premise this test relies on: after removing the first face, the
    # surviving faces sit at indices 0 and 1 but kept their original,
    # non-matching kernel ids -- id and index have genuinely diverged.
    assert remaining_ids[0] != 0

    # Dyadic (eighths), so each value round-trips through float32 exactly and
    # the dict equality below needs no tolerance.
    uvs_f1 = [(0.125, 0.125), (0.25, 0.25), (0.375, 0.375), (0.5, 0.5)]
    uvs_f2 = [(0.875, 0.875), (0.75, 0.75), (0.625, 0.625), (1.0, 1.0)]
    s.set_face_uvs(f1, uvs_f1, Side.FRONT)
    s.set_face_uvs(f2, uvs_f2, Side.FRONT)

    data = geometry_to_dict(s)
    # Keyed by index (0, 1), NOT by f1's/f2's own ids -- an id-keyed
    # implementation would write under str(f1)/str(f2) instead and this
    # would fail.
    assert data["face_uvs"] == {
        "0": [0.125, 0.125, 0.25, 0.25, 0.375, 0.375, 0.5, 0.5],
        "1": [0.875, 0.875, 0.75, 0.75, 0.625, 0.625, 1.0, 1.0],
    }

    loaded = Scene()
    geometry_from_dict(loaded, data)
    ids_in_order = [f.id for f in loaded.faces_iter()]
    assert len(ids_in_order) == 2
    np.testing.assert_allclose(loaded.face_uvs(ids_in_order[0], Side.FRONT), uvs_f1)
    np.testing.assert_allclose(loaded.face_uvs(ids_in_order[1], Side.FRONT), uvs_f2)
