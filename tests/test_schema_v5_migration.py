"""M7.5a Task 12: schema 4 to 5, additive."""

from __future__ import annotations

import numpy as np
from bermake.io.document_codec import geometry_from_dict, geometry_to_dict
from bermake.scene.scene import Scene, Side


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


def test_both_sides_round_trip():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    s.set_face_material(f, 5, Side.BACK)

    rebuilt = Scene()
    geometry_from_dict(rebuilt, geometry_to_dict(s))
    g = next(iter(rebuilt.faces_iter())).id
    assert rebuilt.face_material(g, Side.FRONT) == 3
    assert rebuilt.face_material(g, Side.BACK) == 5


def test_a_schema_4_payload_loads_with_default_backs():
    # No "face_materials_back" key at all, as v0.6.0 wrote.
    s = Scene()
    _square(s)
    data = geometry_to_dict(s)
    data["face_materials"] = {"0": 3}
    data.pop("face_materials_back", None)

    rebuilt = Scene()
    geometry_from_dict(rebuilt, data)
    g = next(iter(rebuilt.faces_iter())).id
    assert rebuilt.face_material(g, Side.FRONT) == 3
    assert rebuilt.face_material(g, Side.BACK) == 0


def test_unpainted_backs_are_not_written():
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 3, Side.FRONT)
    data = geometry_to_dict(s)
    # an all-Default back side must not bloat every file ever saved
    assert data.get("face_materials_back", {}) == {}


def test_schema_version_is_five():
    # M7.7 bumped SCHEMA_VERSION to 9 (per-document viewport environment); this
    # pin necessarily tracks whatever is current, same as every prior bump.
    from bermake.io.bermake_file import SCHEMA_VERSION

    assert SCHEMA_VERSION == 9


def test_a_v4_geometry_payload_is_unchanged_except_for_the_new_key():
    # Correction 5: a v0.7.0 file with no back paint must differ from a
    # v0.6.0 file ONLY by gaining the (empty) "face_materials_back" key --
    # not by any change to "vertices"/"edges"/"faces"/"face_materials". Pin
    # the literal dict shape so a codec that renames/reorders/reshapes any
    # of those fields is caught here, not just by a same-code round trip.
    #
    # M7.5b (Task 7) added "face_placements" / "face_placements_back" the
    # same way schema 5 added "face_materials_back": additively, empty by
    # default. The pin below gained those two keys for the same reason it
    # gained "face_materials_back" at the previous bump.
    #
    # M7.5c (Task 5) added "face_uvs" / "face_uvs_back" the same way, so
    # the pin gains those two keys too.
    s = Scene()
    f = _square(s)
    s.set_face_material(f, 7, Side.FRONT)

    data = geometry_to_dict(s)
    assert data == {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "faces": [[0, 1, 2, 3]],
        "face_materials": {"0": 7},
        "face_materials_back": {},
        "face_placements": {},
        "face_placements_back": {},
        "face_uvs": {},
        "face_uvs_back": {},
    }


def test_face_materials_survive_when_face_ids_and_indices_diverge():
    # face_materials is keyed by INDEX (position in faces_iter()), not by the
    # kernel's face id. A single-face fixture can't tell an index-keyed
    # implementation from an id-keyed one, because face 0 has index 0 either
    # way. Build three faces, remove the first one, so the two surviving
    # faces keep their original (non-zero) kernel ids while their positions
    # in faces_iter() compact down to 0 and 1. Paint the back of the face
    # that now sits at index 1 but whose id is not 1 -- an id-keyed
    # implementation would look up the wrong face (or none) on reload.
    s = Scene()
    f0 = _square(s, z=0.0)
    _f1 = _square(s, z=1.0)
    f2 = _square(s, z=2.0)
    s.remove_face(f0)

    remaining_ids = [f.id for f in s.faces_iter()]
    assert remaining_ids[1] == f2
    # The premise this test relies on: after removing the first face, the
    # second surviving face sits at index 1 but kept its original (non-1)
    # kernel id -- id and index have genuinely diverged.
    assert f2 != 1

    s.set_face_material(f2, 9, Side.FRONT)
    s.set_face_material(f2, 11, Side.BACK)

    data = geometry_to_dict(s)
    # Keyed by index (1), NOT by f2's own id -- an id-keyed implementation
    # would write under str(f2) instead and this would fail.
    assert data["face_materials"] == {"1": 9}
    assert data["face_materials_back"] == {"1": 11}

    rebuilt = Scene()
    geometry_from_dict(rebuilt, data)
    ids_in_order = [f.id for f in rebuilt.faces_iter()]
    assert len(ids_in_order) == 2
    repainted = ids_in_order[1]
    assert rebuilt.face_material(repainted, Side.FRONT) == 9
    assert rebuilt.face_material(repainted, Side.BACK) == 11
    untouched = ids_in_order[0]
    assert rebuilt.face_material(untouched, Side.FRONT) == 0
    assert rebuilt.face_material(untouched, Side.BACK) == 0


def test_real_bermake_file_round_trips_two_sided_materials(tmp_path):
    # The container path (zip + json.dumps/loads), not just the in-memory
    # codec dict -- a codec bug that only shows up after a JSON string
    # round trip (e.g. int keys silently becoming strings twice) would slip
    # past the dict-level tests above.
    from bermake.document import DocumentSettings
    from bermake.io import load_document, save_document
    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import RenderStyle

    model = Model()
    fid = _square(model.root.mesh)
    model.root.mesh.set_face_material(fid, 3, Side.FRONT)
    model.root.mesh.set_face_material(fid, 5, Side.BACK)

    path = tmp_path / "two_sided.berm"
    save_document(path, model, Camera(), DocumentSettings(), RenderStyle())
    loaded = load_document(path)

    loaded_fid = next(iter(loaded.model.root.mesh.faces_iter())).id
    assert loaded.model.root.mesh.face_material(loaded_fid, Side.FRONT) == 3
    assert loaded.model.root.mesh.face_material(loaded_fid, Side.BACK) == 5


def test_a_hand_crafted_v6_0_shaped_document_opens_through_the_real_container(tmp_path):
    """Stands in for an actual v0.6.0 .berm file (see Task 12 report for
    why a real one wasn't produced from the tag). document_codec.py and
    bermake_file.py are byte-identical between the v0.6.0 tag and this
    milestone's starting point -- Tasks 1/3/11 touched material.py, scene.py
    and tag.py only -- so a hand-built document matching exactly what
    v0.6.0's to_records()/geometry_to_dict()/ViewLibrary.to_records()/
    render_style_to_dict() wrote, pushed through the REAL zip+manifest+json
    container and the CURRENT load_document, exercises the same code path a
    genuine old file would.

    Includes real top-level "scenes" and "style" keys, shaped exactly as
    v0.6.0's document_to_dict() unconditionally wrote them (confirmed via
    `git show v0.6.0:python/bermake/io/document_codec.py`): one saved Scene
    from ViewLibrary.to_records() and a non-default RenderStyle from
    render_style_to_dict(). A fixture omitting those keys is not a faithful
    v0.6.0 shape -- it only happens to load because document_from_dict()'s
    `.get("scenes", {})` / `.get("style")` tolerance for their absence
    predates this milestone and is unrelated to what Task 12 changed; that
    tolerance is not what this test is exercising or asserting on.

    Checks every field the migration is supposed to populate, including the
    ones whose correct value is a default: a loader that silently drops
    back-side materials and one that correctly writes none are
    indistinguishable unless a back side is asserted explicitly (it is,
    below: Default on both faces). The scenes/style values are likewise
    asserted below, not just carried as inert payload."""
    import json
    import zipfile

    from bermake.io.bermake_file import load_document

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
            "next_id": 2,
            # Schema <= 4 shape: "color" only, no base_color/alpha/metallic/
            # roughness -- exactly what MaterialLibrary.to_records() wrote
            # before Task 1.
            "items": [
                {"id": 0, "name": "Default", "color": [0.65, 0.65, 0.70]},
                {"id": 1, "name": "Brick Red", "color": [0.8, 0.1, 0.1]},
            ],
        },
        "tags": {
            "next_id": 2,
            # Schema <= 4 shape: no "color" key at all -- exactly what
            # TagLibrary.to_records() wrote before Task 11.
            "items": [
                {"id": 0, "name": "Untagged", "visible": True},
                {"id": 1, "name": "Interior", "visible": True},
            ],
        },
        # document_to_dict() at v0.6.0 wrote "scenes" and "style"
        # unconditionally -- a genuine v0.6.0 document.json always has both.
        # Shapes below match ViewLibrary.to_records() / render_style_to_dict()
        # at that tag exactly (both unchanged by this milestone).
        "scenes": {
            "next_id": 1,
            "items": [
                {
                    "id": 0,
                    "name": "Front View",
                    "camera": {
                        "position": [0.0, 0.0, 5.0],
                        "target": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0],
                        "fov_y_deg": 35.0,
                    },
                    "tag_visibility": {"1": False},
                    "face_style": "WIREFRAME",
                    "xray": True,
                },
            ],
        },
        "style": {"face_style": "HIDDEN_LINE", "xray": True},
        "model": {
            "next_def_id": 1,
            "next_inst_id": 0,
            "root_id": 0,
            "definitions": [
                {
                    "id": 0,
                    "name": "Root",
                    "is_group": True,
                    "geometry": {
                        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
                        "faces": [[0, 1, 2, 3]],
                        # Front-only, index-keyed, non-Default only -- the
                        # exact convention Correction 5 says must not change.
                        "face_materials": {"0": 1},
                        # NOTE: deliberately no "face_materials_back" key,
                        # as v0.6.0 never wrote one.
                    },
                    "annotations": [],
                    "children": [],
                },
            ],
        },
    }
    manifest = {"format": "bermake", "schema_version": 4, "app_version": "0.6.0"}

    path = tmp_path / "v0_6_0_shaped.berm"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
        zf.writestr("document.json", json.dumps(doc_data, separators=(",", ":")))

    loaded = load_document(path)

    # Materials: "color" migrated to base_color, PBR fields defaulted.
    brick = loaded.model.materials.get(1)
    assert brick.base_color == (0.8, 0.1, 0.1)
    assert brick.alpha == 1.0
    assert brick.metallic == 0.0
    assert brick.roughness == 0.5

    # Tags: palette-derived colour for the real tag, neutral gray (not a
    # palette hue) for Untagged.
    interior = loaded.model.tags.get(1)
    assert interior.color == (0.90, 0.25, 0.25)  # first palette hue
    untagged = loaded.model.tags.get(0)
    assert untagged.color == (0.55, 0.55, 0.55)

    # Face materials: front survives, back defaults to Default (0) -- the
    # face was painted on the front only in this v0.6.0-shaped file, which
    # carries no "face_materials_back" key at all.
    fid = next(iter(loaded.model.root.mesh.faces_iter())).id
    assert loaded.model.root.mesh.face_material(fid, Side.FRONT) == 1
    assert loaded.model.root.mesh.face_material(fid, Side.BACK) == 0

    # Scenes: the one saved Scene from the v0.6.0-shaped "scenes" key
    # survives the load intact -- not merely tolerated as absent.
    view = loaded.model.views.get(0)
    assert view is not None
    assert view.name == "Front View"
    assert view.camera.position == (0.0, 0.0, 5.0)
    assert view.tag_visibility == {1: False}
    assert view.face_style == "WIREFRAME"
    assert view.xray is True

    # Style: the document-level render style from the v0.6.0-shaped "style"
    # key survives the load intact, and is not the RenderStyle() default
    # (SHADED/xray=False) that render_style_from_dict() falls back to when
    # "style" is absent -- so this genuinely checks the value was read, not
    # just that the default happens to match.
    assert loaded.style.face_style.name == "HIDDEN_LINE"
    assert loaded.style.xray is True
