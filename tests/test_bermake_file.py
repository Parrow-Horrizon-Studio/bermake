import json
import zipfile

import numpy as np
import pytest
from bermake.document import DocumentSettings
from bermake.io import (
    BermakeFormatError,
    BermakeVersionError,
    load_document,
    save_document,
)
from bermake.model.model import Model
from bermake.units import Units, UnitSystem
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import RenderStyle


def _model_with_box():
    model = Model()
    vids = [model.root.mesh.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    model.root.mesh.add_face_from_loop(vids)
    return model


def _add_box(scene):
    vids = [scene.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    scene.add_face_from_loop(vids)


def test_save_then_load_roundtrip(tmp_path):
    path = tmp_path / "house.berm"
    save_document(path, _model_with_box(), Camera(), DocumentSettings(), RenderStyle())
    loaded = load_document(path)
    assert len(list(loaded.model.root.mesh.faces_iter())) == 1


def test_load_rejects_newer_schema(tmp_path):
    path = tmp_path / "future.berm"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "bermake", "schema_version": 999}))
        zf.writestr("document.json", "{}")
    with pytest.raises(BermakeVersionError):
        load_document(path)


def test_load_rejects_non_int_schema_version(tmp_path):
    """schema_version must be an int; a non-int value (e.g. a string) must hit
    the `not isinstance(ver, int)` branch of the version gate, not the
    `ver > SCHEMA_VERSION` numeric-comparison branch (which test_load_rejects_
    newer_schema already covers with an int that's simply too big)."""
    path = tmp_path / "bad_version_type.berm"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "bermake", "schema_version": "not-a-number"}))
        zf.writestr("document.json", "{}")
    with pytest.raises(BermakeVersionError):
        load_document(path)


def test_load_rejects_foreign_format(tmp_path):
    path = tmp_path / "alien.berm"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "sketchup", "schema_version": 1}))
        zf.writestr("document.json", "{}")
    with pytest.raises(BermakeFormatError):
        load_document(path)


def test_a_pluton_file_is_rejected_with_a_message_that_names_pluton(tmp_path):
    """Spec D2: the break is deliberate, so it has to explain itself.

    A generic wrong-format message would leave a tester with a file that will
    not open and no idea why. The assertion checks the message, not merely that
    an exception was raised, because the message is the whole deliverable here.
    """
    path = tmp_path / "old.berm"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"format": "pluton", "schema_version": 9}))
        zf.writestr("document.json", "{}")

    with pytest.raises(BermakeFormatError) as excinfo:
        load_document(path)
    assert "Pluton" in str(excinfo.value)


def test_load_rejects_non_zip(tmp_path):
    path = tmp_path / "garbage.berm"
    path.write_text("not a zip at all")
    with pytest.raises(BermakeFormatError):
        load_document(path)


def test_load_rejects_missing_document_entry(tmp_path):
    path = tmp_path / "incomplete.berm"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json",
                    json.dumps({"format": "bermake", "schema_version": 1}))
    with pytest.raises(BermakeFormatError):
        load_document(path)


def test_save_is_atomic_old_file_survives_failure(tmp_path, monkeypatch):
    path = tmp_path / "keep.berm"
    save_document(path, _model_with_box(), Camera(), DocumentSettings(), RenderStyle())
    original = path.read_bytes()

    import bermake.io.bermake_file as pf
    monkeypatch.setattr(pf, "document_to_dict",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        save_document(path, _model_with_box(), Camera(), DocumentSettings(), RenderStyle())
    assert path.read_bytes() == original  # untouched
    assert not (tmp_path / "keep.berm.tmp").exists()  # temp cleaned up


def test_rich_document_roundtrip_through_real_zip_and_json(tmp_path):
    """End-to-end save/load through the REAL zip + json.dumps/loads path (not just
    the codec-dict level), exercising the format's headline guarantees together:
    shared-component identity, nested groups with non-identity transforms, tag
    visibility, custom materials on painted faces, camera state, and imperial
    units. The narrower codec-level version of this scenario lives in
    test_document_codec.test_document_roundtrip_camera_units_materials_tags."""
    model = Model()

    # Root: geometry + a painted face using a custom material.
    _add_box(model.root.mesh)
    root_fid = next(iter(model.root.mesh.faces_iter())).id
    teal = model.materials.add_custom("Teal", (0.1, 0.6, 0.6))
    model.root.mesh.set_face_material(root_fid, teal.id)

    # Hidden tag, applied to one instance.
    hidden_tag = model.tags.add("Hidden Stuff")
    model.tags.set_visible(hidden_tag.id, False)

    # Shared component definition, placed as two instances under root.
    chair = model.new_definition("Chair", is_group=False)
    _add_box(chair.mesh)
    chair_inst_1 = model.new_instance(chair)
    chair_inst_2 = model.new_instance(chair, transform=np.array([
        [1, 0, 0, 5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64))
    chair_inst_2.tag_id = hidden_tag.id
    model.root.children.append(chair_inst_1)
    model.root.children.append(chair_inst_2)

    # Nested group: a group definition with its own geometry, placed with a
    # non-identity transform (translation + non-uniform-looking scale-free rotation
    # substitute — keep it a simple affine transform, exactness is what's checked).
    group_def = model.new_definition("Group 1", is_group=True)
    _add_box(group_def.mesh)
    group_transform = np.array([
        [2, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 3, 3],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    group_inst = model.new_instance(group_def, transform=group_transform)
    model.root.children.append(group_inst)

    # Non-default camera.
    cam = Camera()
    cam.position = np.array([10, -20, 30], dtype=np.float32)
    cam.fov_y_deg = 60.0

    # Imperial units.
    doc = DocumentSettings()
    doc.set_units(Units(system=UnitSystem.IMPERIAL, imperial_denominator=8))

    path = tmp_path / "rich.berm"
    save_document(path, model, cam, doc, RenderStyle())
    loaded = load_document(path)

    loaded_root = loaded.model.root
    assert len(loaded_root.children) == 3
    loaded_chair_1, loaded_chair_2, loaded_group_inst = loaded_root.children

    # Shared component: both instances point at the SAME Definition object.
    assert loaded_chair_1.definition is loaded_chair_2.definition
    assert loaded_chair_1.definition.name == "Chair"

    # Per-child transforms and tag_ids survive.
    assert np.allclose(loaded_chair_1.transform, np.eye(4))
    assert loaded_chair_1.tag_id == 0
    assert np.allclose(loaded_chair_2.transform, chair_inst_2.transform)
    assert loaded_chair_2.tag_id == hidden_tag.id

    # Nested group: its own definition + geometry + non-identity transform.
    assert loaded_group_inst.definition.is_group is True
    assert loaded_group_inst.definition.name == "Group 1"
    assert len(list(loaded_group_inst.definition.mesh.faces_iter())) == 1
    assert np.allclose(loaded_group_inst.transform, group_transform)

    # Painted face's material id + the custom material itself survive.
    loaded_root_fid = next(iter(loaded_root.mesh.faces_iter())).id
    assert loaded_root.mesh.face_material(loaded_root_fid) == teal.id
    assert loaded.model.materials.get(teal.id).name == "Teal"

    # Hidden tag's visibility survives.
    assert loaded.model.tags.is_visible(hidden_tag.id) is False

    # Counters restored so post-load edits won't collide with loaded ids.
    assert loaded.model._next_def_id == model._next_def_id
    assert loaded.model._next_inst_id == model._next_inst_id

    # Camera state survives.
    assert tuple(round(float(x), 3) for x in loaded.camera_state.position) == (10.0, -20.0, 30.0)
    assert round(loaded.camera_state.fov_y_deg, 3) == 60.0

    # Imperial units survive.
    assert loaded.units.system is UnitSystem.IMPERIAL
    assert loaded.units.imperial_denominator == 8


def test_load_schema_version_1_document_without_annotations_key(tmp_path):
    """Back-compat test for v1 documents that predate the per-Definition
    'annotations' array. The real bermake_file load path must handle a v1
    document whose definitions lack the 'annotations' key entirely.

    This exercises the full save/load pipeline (zip + manifest + json), not
    just the dict codec, ensuring the version gate accepts v1 files and
    document_from_dict correctly defaults missing annotations to empty lists."""
    path = tmp_path / "old_v1.berm"

    # Construct a minimal v1 document by hand: the archive layout must match
    # exactly what save_document produces, but with schema_version=1 and no
    # "annotations" keys in the definitions.
    doc_data = {
        "units": {
            "system": "metric",
            "metric_unit": "m",
            "metric_precision": 3,
            "imperial_denominator": 16,
        },
        "camera": {
            "position": [0.0, 0.0, 10.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "fov_y_deg": 45.0,
        },
        "materials": {
            "next_id": 1,
            "items": [],
        },
        "tags": {
            "next_id": 1,
            "items": [],
        },
        "model": {
            "next_def_id": 2,
            "next_inst_id": 1,
            "root_id": 1,
            "definitions": [
                {
                    "id": 1,
                    "name": "Root",
                    "is_group": True,
                    "geometry": {
                        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        "edges": [],
                        "faces": [[0, 1, 2, 3]],
                        "face_materials": {},
                    },
                    "children": [],
                    # NOTE: no "annotations" key, as in pre-M7d files
                },
            ],
        },
    }

    manifest = {
        "format": "bermake",
        "schema_version": 1,
        "app_version": "0.0.0-test",
    }

    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        zf.writestr("document.json", json.dumps(doc_data, separators=(",", ":")))

    # Load the v1 file through the real bermake_file pipeline.
    loaded = load_document(path)

    # The v1 document must load without error.
    assert loaded.model.root.id == 1
    assert loaded.model.root.name == "Root"

    # Every definition must end up with an empty annotations list, not None
    # or missing. This is the back-compat guarantee.
    assert loaded.model.root.annotations == []

    # The annotation id counter must be initialized correctly: since no
    # annotations were loaded, the next id should be 0.
    new_id = loaded.model.new_annotation_id()
    assert new_id == 0

    # Sanity: the newly allocated id is not 0 again on the next call.
    next_new_id = loaded.model.new_annotation_id()
    assert next_new_id == 1


def test_save_load_round_trips_scenes_and_style(tmp_path):
    from bermake.io.document_codec import CameraState
    from bermake.io.bermake_file import load_document, save_document
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import FaceStyle, RenderStyle
    from bermake.views.saved_view import SavedView

    model = _model_with_box()
    cam_state = CameraState(position=(3.0, 3.0, 3.0), target=(0.0, 0.0, 0.0),
                            up=(0.0, 0.0, 1.0), fov_y_deg=55.0)
    model.views.add(SavedView(0, "Iso", cam_state, {}, "HIDDEN_LINE", False))
    style = RenderStyle(face_style=FaceStyle.WIREFRAME, xray=True)

    path = tmp_path / "scenes.berm"
    save_document(path, model, Camera(), DocumentSettings(), style)
    loaded = load_document(path)

    assert [v.name for v in loaded.model.views.views()] == ["Iso"]
    assert loaded.style.face_style is FaceStyle.WIREFRAME
    assert loaded.style.xray is True


def test_v2_file_without_scenes_still_opens(tmp_path):
    # Hand-craft a v2 .berm (schema_version 2, document.json without
    # scenes/style) and confirm the version gate accepts it and load yields
    # an empty ViewLibrary + default RenderStyle.
    import json
    import zipfile

    from bermake.io.document_codec import document_to_dict
    from bermake.io.bermake_file import load_document
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import RenderStyle

    data = document_to_dict(_model_with_box(), Camera(), DocumentSettings(), RenderStyle())
    del data["scenes"]
    del data["style"]
    manifest = {"format": "bermake", "schema_version": 2, "app_version": "0.2.4"}

    path = tmp_path / "legacy.berm"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("document.json", json.dumps(data))

    loaded = load_document(path)
    assert loaded.model.views.views() == []
    assert loaded.style == RenderStyle()
