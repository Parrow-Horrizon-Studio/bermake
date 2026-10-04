"""M7.5b Task 7: schema 6, additive, with blobs beside document.json."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from bermake.document import DocumentSettings
from bermake.io.document_codec import geometry_from_dict, geometry_to_dict
from bermake.io.errors import BermakeFormatError
from bermake.scene.scene import DEFAULT_PLACEMENT, Scene, Side, TexturePlacement
from bermake.viewport.camera import Camera

_PNG = b"\x89PNG\r\n\x1a\nfake-but-stable-bytes"


def _camera():
    return Camera()


def _doc():
    return DocumentSettings()


def _square(scene, z=0.0):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, z], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return scene.add_face_from_loop(v)


def test_placement_round_trips_per_side():
    s = Scene()
    f = _square(s)
    s.set_face_placement(f, TexturePlacement(0.25, -0.5, 2.0, 1.0), Side.FRONT)
    s.set_face_placement(f, TexturePlacement(scale=3.0), Side.BACK)

    out = Scene()
    geometry_from_dict(out, geometry_to_dict(s))
    assert out.face_placement(f, Side.FRONT) == TexturePlacement(0.25, -0.5, 2.0, 1.0)
    assert out.face_placement(f, Side.BACK).scale == 3.0


def test_unadjusted_faces_are_not_written():
    # Task 3 clears identity placements, so the dicts hold only adjusted faces.
    # Writing every face would bloat every document that ever touched a texture.
    s = Scene()
    _square(s)
    d = geometry_to_dict(s)
    assert d["face_placements"] == {}
    assert d["face_placements_back"] == {}


def test_placement_is_keyed_by_face_index_not_face_id():
    # Matches the face_materials convention. A single-face fixture cannot tell
    # the two apart, because face 0 has index 0, so build two and delete one.
    s = Scene()
    a = _square(s)
    b = _square(s, z=1.0)
    s.remove_face(a)
    s.set_face_placement(b, TexturePlacement(scale=2.0))
    d = geometry_to_dict(s)
    assert list(d["face_placements"].keys()) == ["0"]
    assert b != 0


def test_an_out_of_range_placement_face_index_raises():
    # Structural corruption: the sidecar names a face that doesn't exist in
    # this document's faces[] list. Mirrors
    # test_geometry_from_dict_rejects_an_out_of_range_material_id for
    # face_materials (test_document_codec.py) -- silently dropping the entry
    # would erase the user's adjustment on load, and the very next save would
    # then write that loss back out permanently.
    s = Scene()
    _square(s)
    d = geometry_to_dict(s)
    d["face_placements"] = {"99": [0.0, 0.0, 1.0, 0.0]}
    with pytest.raises(BermakeFormatError):
        geometry_from_dict(Scene(), d)


def test_a_placement_with_the_wrong_arity_raises():
    # A truncated record (e.g. [offset_u, offset_v] with scale/rotation
    # missing) must not silently become a full TexturePlacement with invented
    # defaults, and must raise BermakeFormatError (not a raw TypeError) so a
    # direct caller of geometry_from_dict sees the documented exception.
    s = Scene()
    _square(s)
    d = geometry_to_dict(s)
    d["face_placements"] = {"0": [0.5, 0.5]}
    with pytest.raises(BermakeFormatError):
        geometry_from_dict(Scene(), d)


def test_a_payload_without_placement_keys_still_loads():
    # The codec reads the optional placement keys with defaults. Any document that
    # lacks them (hand-written, or from a build that never wrote them) must open.
    s = Scene()
    _square(s)
    d = geometry_to_dict(s)
    del d["face_placements"]
    del d["face_placements_back"]
    out = Scene()
    geometry_from_dict(out, d)  # must not raise
    assert out.faces_with_placement() == []


def test_placement_round_trips_through_a_real_file(tmp_path):
    # The dict-level tests above go through geometry_to_dict/geometry_from_dict
    # directly; this exercises the real zip + json.dumps/loads container path,
    # a front placement, a back placement, a nested Definition's face (not just
    # root), and an unadjusted face staying at identity -- the other half of
    # this task, alongside the texture-blob real-file tests below.
    from bermake.io.bermake_file import load_document, save_document
    from bermake.model.model import Model
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    f_root = _square(model.root.mesh)
    model.root.mesh.set_face_placement(f_root, TexturePlacement(0.1, 0.2, 1.5, 0.3), Side.FRONT)
    model.root.mesh.set_face_placement(f_root, TexturePlacement(scale=2.0), Side.BACK)
    _square(model.root.mesh, z=1.0)  # left at identity

    chair = model.new_definition("Chair", is_group=False)
    f_nested = _square(chair.mesh, z=2.0)
    chair.mesh.set_face_placement(f_nested, TexturePlacement(offset_u=0.5))
    model.root.children.append(model.new_instance(chair))

    path = tmp_path / "t.berm"
    save_document(path, model, _camera(), _doc(), RenderStyle())
    loaded = load_document(path)

    root_faces = list(loaded.model.root.mesh.faces_iter())
    loaded_f_root, loaded_f_plain = root_faces[0].id, root_faces[1].id
    assert loaded.model.root.mesh.face_placement(loaded_f_root, Side.FRONT) == TexturePlacement(
        0.1, 0.2, 1.5, 0.3
    )
    assert loaded.model.root.mesh.face_placement(loaded_f_root, Side.BACK).scale == 2.0
    assert loaded.model.root.mesh.face_placement(loaded_f_plain) == DEFAULT_PLACEMENT

    chair_def = loaded.model.root.children[0].definition
    loaded_f_nested = next(iter(chair_def.mesh.faces_iter())).id
    assert chair_def.mesh.face_placement(loaded_f_nested).offset_u == 0.5


def test_texture_records_and_blobs_round_trip_through_a_real_file(tmp_path):
    from bermake.io.bermake_file import load_document, save_document
    from bermake.model.model import Model
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    tex = model.textures.add("brick.png", _PNG, "png", 8, 8, True)
    mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    model.materials.edit(mat.id, texture_id=tex.id, texture_size=(2.0, 3.0))

    path = tmp_path / "t.berm"
    save_document(path, model, _camera(), _doc(), RenderStyle())
    loaded = load_document(path)

    back = loaded.model.textures.get(tex.id)
    assert back is not None
    assert back.data == _PNG
    assert back.has_transparency is True
    assert loaded.model.materials.get(mat.id).texture_size == (2.0, 3.0)


def test_next_id_survives_removing_the_highest_id_texture(tmp_path):
    # Regression flagged for Task 8's upcoming DeleteTextureCommand:
    # from_records used to recompute next_id as max(ids) + 1 whenever the
    # container didn't carry a saved counter, so deleting the highest-id
    # texture and saving regressed next_id. A texture added after reload
    # would then reuse the freed id, and any material whose texture_id
    # pointed at the original image would silently point at whatever gets
    # that id next.
    from bermake.io.bermake_file import load_document, save_document
    from bermake.model.model import Model
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    first = model.textures.add("a.png", _PNG, "png", 4, 4, False)
    second = model.textures.add("b.png", _PNG, "png", 4, 4, False)
    model.textures.remove(second.id)

    path = tmp_path / "t.berm"
    save_document(path, model, _camera(), _doc(), RenderStyle())
    loaded = load_document(path)

    reused = loaded.model.textures.add("c.png", _PNG, "png", 4, 4, False)
    assert reused.id == second.id + 1
    assert loaded.model.textures.get(first.id) is not None


def test_the_blob_is_a_sibling_entry_not_base64_in_the_json(tmp_path):
    from bermake.io.bermake_file import save_document
    from bermake.model.model import Model
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    tex = model.textures.add("brick.png", _PNG, "png", 8, 8, False)
    path = tmp_path / "t.berm"
    save_document(path, model, _camera(), _doc(), RenderStyle())

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert f"textures/{tex.id}.png" in names
        assert zf.read(f"textures/{tex.id}.png") == _PNG
        doc = json.loads(zf.read("document.json"))
    record = doc["textures"]["items"][0]
    # Exact key set, not a substring check: a texture named "metadata.png"
    # would make `"data" not in json.dumps(record)` fire spuriously, and a
    # bare substring check would miss bytes smuggled in under a different
    # key (e.g. "payload").
    assert set(record.keys()) == {
        "id",
        "name",
        "image_format",
        "width",
        "height",
        "has_transparency",
    }


def test_a_document_referencing_a_missing_blob_still_opens(tmp_path):
    # Spec 1.8. Rewriting the zip without the texture entry simulates a file
    # that was truncated or hand-edited; the open must succeed untextured.
    from bermake.io.bermake_file import load_document, save_document
    from bermake.model.model import Model
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    tex = model.textures.add("brick.png", _PNG, "png", 8, 8, False)
    src = tmp_path / "a.berm"
    save_document(src, model, _camera(), _doc(), RenderStyle())

    dst = tmp_path / "b.berm"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            if not item.filename.startswith("textures/"):
                zout.writestr(item, zin.read(item.filename))

    loaded = load_document(dst)
    assert loaded.model.textures.get(tex.id).data == b""
