import numpy as np
import pytest
from bermake.document import DocumentSettings
from bermake.io.document_codec import (
    CameraState,
    document_from_dict,
    document_to_dict,
    environment_from_dict,
    environment_to_dict,
    geometry_from_dict,
    geometry_to_dict,
    model_from_dict,
    model_to_dict,
)
from bermake.io.errors import BermakeFormatError
from bermake.io.bermake_file import SCHEMA_VERSION
from bermake.model.model import Model
from bermake.scene.scene import Scene
from bermake.units import Units, UnitSystem
from bermake.viewport.camera import Camera
from bermake.viewport.environment import (
    DEFAULT_ENVIRONMENT,
    LEGACY_ENVIRONMENT,
    PLAIN_WHITE,
    SKY_AND_GROUND,
    STUDIO,
)
from bermake.viewport.render_style import RenderStyle


def _square(scene: Scene) -> list[int]:
    vids = [scene.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    scene.add_face_from_loop(vids)
    return vids


def test_geometry_roundtrip_painted_face():
    src = Scene()
    _square(src)
    fid = next(iter(src.faces_iter())).id
    src.set_face_material(fid, 5)

    data = geometry_to_dict(src)
    assert len(data["vertices"]) == 4
    assert len(data["faces"]) == 1
    assert data["face_materials"] == {"0": 5}

    dst = Scene()
    geometry_from_dict(dst, data)
    assert len(list(dst.vertices_iter())) == 4
    assert len(list(dst.faces_iter())) == 1
    new_fid = next(iter(dst.faces_iter())).id
    assert dst.face_material(new_fid) == 5


def test_geometry_roundtrip_compacts_id_gaps():
    src = Scene()
    vids = _square(src)
    # Add a loose vertex, then delete it -> leaves an id gap in the kernel.
    loose = src.add_vertex(np.array((9, 9, 9), dtype=np.float32))
    src.remove_vertex(loose)

    data = geometry_to_dict(src)
    assert len(data["vertices"]) == 4  # gap compacted away

    dst = Scene()
    geometry_from_dict(dst, data)
    got = sorted(tuple(round(float(c), 3) for c in v.position) for v in dst.vertices_iter())
    want = sorted(tuple(round(float(c), 3) for c in src.vertex(v).position) for v in vids)
    assert got == want


def test_geometry_from_dict_rejects_bad_index():
    dst = Scene()
    bad = {"vertices": [[0, 0, 0]], "edges": [[0, 7]], "faces": [], "face_materials": {}}
    with pytest.raises(BermakeFormatError):
        geometry_from_dict(dst, bad)


def _minimal_document_dict():
    """The smallest dict document_from_dict accepts, for mutating in one place."""
    return document_to_dict(Model(), Camera(), DocumentSettings(), RenderStyle())


def _one_face_doc(face_materials, key="face_materials"):
    return {
        "vertices": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        "edges": [],
        "faces": [[0, 1, 2, 3]],
        key: face_materials,
    }


@pytest.mark.parametrize("bad_id", [-3, -1, 1 << 20, 1 << 40])
@pytest.mark.parametrize("key", ["face_materials", "face_materials_back"])
def test_geometry_from_dict_rejects_an_out_of_range_material_id(bad_id, key):
    # M7.5a regression: the face INDEX was validated but the material id was
    # not, while plan_face_batches (new this milestone) rejects ids outside
    # [0, 2**20). So a document holding {"0": -3} loaded clean and then raised
    # ValueError from inside render() -- every frame, where nothing can report
    # it. Before M7.5a the same file rendered as Default.
    #
    # Both sides and both ends of the range: a guard that only checks
    # face_materials, or only the negative end, is a plausible half-fix.
    with pytest.raises(BermakeFormatError):
        geometry_from_dict(Scene(), _one_face_doc({"0": bad_id}, key))


def test_an_in_range_material_id_naming_no_material_still_loads():
    # Only the RANGE is validated. An id the library does not contain is a
    # different case and stays forgiving: MaterialLibrary.get falls back to
    # Default, so a file that lost a material opens rather than refusing.
    dst = Scene()
    geometry_from_dict(dst, _one_face_doc({"0": 999}))
    fid = next(iter(dst.faces_iter())).id
    assert dst.face_material(fid) == 999


def test_a_document_with_a_malformed_material_id_fails_at_load_not_at_render():
    # The whole point of the fix, through the real document path: the error
    # arrives once, from the loader, as the BermakeFormatError the UI already
    # knows how to show.
    model = Model()
    _square(model.root.mesh)
    data = document_to_dict(model, Camera(), DocumentSettings(), RenderStyle())
    data["model"]["definitions"][0]["geometry"]["face_materials"] = {"0": -3}

    with pytest.raises(BermakeFormatError):
        document_from_dict(data)


def _add_box(scene):
    vids = [scene.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    scene.add_face_from_loop(vids)


def test_model_roundtrip_shared_component_is_one_definition_and_shared():
    model = Model()
    comp = model.new_definition("Chair", is_group=False)
    _add_box(comp.mesh)
    i1 = model.new_instance(comp)
    i2 = model.new_instance(comp)
    model.root.children.extend([i1, i2])

    data = model_to_dict(model)
    chair_records = [d for d in data["definitions"] if d["name"] == "Chair"]
    assert len(chair_records) == 1  # emitted once despite two instances

    loaded = model_from_dict(data)
    kids = loaded.root.children
    assert len(kids) == 2
    assert kids[0].definition is kids[1].definition  # sharing preserved (identity)


def test_model_roundtrip_restores_counters_and_tag_ids():
    model = Model()
    g = model.new_definition("Grp", is_group=True)
    inst = model.new_instance(g)
    inst.tag_id = 7
    model.root.children.append(inst)

    loaded = model_from_dict(model_to_dict(model))
    assert loaded._next_def_id == model._next_def_id
    assert loaded._next_inst_id == model._next_inst_id
    assert loaded.root.children[0].tag_id == 7


def test_model_from_dict_rejects_dangling_definition_ref():
    data = {
        "next_def_id": 2, "next_inst_id": 1, "root_id": 0,
        "definitions": [{
            "id": 0, "name": "Model", "is_group": False,
            "geometry": {"vertices": [], "edges": [], "faces": [], "face_materials": {}},
            "children": [{"id": 0, "definition_id": 99,
                          "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                          "tag_id": 0}],
        }],
    }
    with pytest.raises(BermakeFormatError):
        model_from_dict(data)


def test_document_roundtrip_camera_units_materials_tags():
    model = Model()
    _add_box(model.root.mesh)
    fid = next(iter(model.root.mesh.faces_iter())).id
    teal = model.materials.add_custom("Teal", (0.1, 0.6, 0.6))
    model.root.mesh.set_face_material(fid, teal.id)
    walls = model.tags.add("Walls")
    model.tags.set_visible(walls.id, False)

    cam = Camera()
    cam.position = np.array([3, -4, 5], dtype=np.float32)
    doc = DocumentSettings()
    doc.set_units(Units(system=UnitSystem.IMPERIAL, imperial_denominator=8))

    data = document_to_dict(model, cam, doc, RenderStyle())
    loaded = document_from_dict(data)

    assert loaded.units.system is UnitSystem.IMPERIAL
    assert loaded.units.imperial_denominator == 8
    assert tuple(round(x, 3) for x in loaded.camera_state.position) == (3.0, -4.0, 5.0)
    assert loaded.model.materials.get(teal.id).name == "Teal"
    assert loaded.model.tags.is_visible(walls.id) is False
    new_fid = next(iter(loaded.model.root.mesh.faces_iter())).id
    assert loaded.model.root.mesh.face_material(new_fid) == teal.id


def test_camera_state_apply_to_roundtrip():
    cam = Camera()
    cam.position = np.array([1, 2, 3], dtype=np.float32)
    cam.fov_y_deg = 33.0
    state = CameraState.from_dict(CameraState.from_camera(cam).to_dict())
    target = Camera()
    state.apply_to(target)
    assert tuple(round(float(x), 3) for x in target.position) == (1.0, 2.0, 3.0)
    assert round(target.fov_y_deg, 3) == 33.0


def test_document_from_dict_wraps_structural_errors():
    with pytest.raises(BermakeFormatError):
        document_from_dict({"model": {}})  # missing keys everywhere


def test_document_dict_round_trips_scenes_and_style():
    from bermake.document import DocumentSettings
    from bermake.io.document_codec import (
        CameraState,
        document_from_dict,
        document_to_dict,
    )
    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import FaceStyle, RenderStyle
    from bermake.views.saved_view import SavedView

    model = Model()
    cam_state = CameraState(position=(2.0, 2.0, 2.0), target=(0.0, 0.0, 0.0),
                            up=(0.0, 0.0, 1.0), fov_y_deg=50.0)
    model.views.add(SavedView(0, "Front", cam_state, {1: False}, "WIREFRAME", True))
    style = RenderStyle(face_style=FaceStyle.MONOCHROME, xray=True)

    data = document_to_dict(model, Camera(), DocumentSettings(), style)
    assert data["scenes"]["items"][0]["name"] == "Front"
    assert data["scenes"]["items"][0]["tag_visibility"] == {"1": False}
    assert data["style"] == {"face_style": "MONOCHROME", "xray": True, "color_by_tag": False}

    loaded = document_from_dict(data)
    assert [v.name for v in loaded.model.views.views()] == ["Front"]
    assert loaded.model.views.get(0).tag_visibility == {1: False}
    assert loaded.style.face_style is FaceStyle.MONOCHROME
    assert loaded.style.xray is True


def test_color_by_tag_survives_a_document_round_trip():
    # The final-review finding: render_style_to_dict wrote only face_style and
    # xray, so a document saved with Color-by-Tag on reopened with it off. The
    # assertion is on the LOADED style, not on the dict, so it fails for a
    # writer that emits the key and a reader that ignores it too.
    from bermake.document import DocumentSettings
    from bermake.io.document_codec import document_from_dict, document_to_dict
    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import FaceStyle, RenderStyle

    style = RenderStyle(face_style=FaceStyle.HIDDEN_LINE, xray=True, color_by_tag=True)
    data = document_to_dict(Model(), Camera(), DocumentSettings(), style)

    loaded = document_from_dict(data)
    assert loaded.style == style


def test_a_style_block_without_color_by_tag_still_loads():
    # Schema 5 already carries the key's absence: every .berm written before
    # this fix has a "style" block with only face_style and xray. Those files
    # must open, with the mode off rather than a KeyError.
    from bermake.io.document_codec import render_style_from_dict
    from bermake.viewport.render_style import FaceStyle

    style = render_style_from_dict({"face_style": "WIREFRAME", "xray": True})
    assert style.face_style is FaceStyle.WIREFRAME
    assert style.xray is True
    assert style.color_by_tag is False


def test_document_from_dict_without_scenes_or_style_uses_defaults():
    # A v2-shaped document (no "scenes"/"style" keys) still loads.
    from bermake.document import DocumentSettings
    from bermake.io.document_codec import document_from_dict, document_to_dict
    from bermake.model.model import Model
    from bermake.viewport.camera import Camera
    from bermake.viewport.render_style import RenderStyle

    data = document_to_dict(Model(), Camera(), DocumentSettings(), RenderStyle())
    del data["scenes"]
    del data["style"]
    loaded = document_from_dict(data)
    assert loaded.model.views.views() == []
    assert loaded.style == RenderStyle()   # RenderStyle default (SHADED, xray False)


def test_guide_round_trips_through_the_codec():
    from bermake.io.document_codec import annotation_from_dict, annotation_to_dict
    from bermake.model.annotation import Guide

    g = Guide(3, (1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
    back = annotation_from_dict(annotation_to_dict(g))
    assert isinstance(back, Guide)
    assert back.id == 3
    assert back.origin == (1.0, 2.0, 3.0)
    assert back.direction == (0.0, 1.0, 0.0)


def test_guide_point_round_trips_through_the_codec():
    from bermake.io.document_codec import annotation_from_dict, annotation_to_dict
    from bermake.model.annotation import GuidePoint

    gp = GuidePoint(4, (5.0, 6.0, 7.0))
    back = annotation_from_dict(annotation_to_dict(gp))
    assert isinstance(back, GuidePoint)
    assert back.id == 4
    assert back.position == (5.0, 6.0, 7.0)


def test_an_unknown_annotation_kind_still_raises():
    import pytest

    from bermake.io.document_codec import annotation_from_dict
    from bermake.io.errors import BermakeFormatError

    with pytest.raises(BermakeFormatError):
        annotation_from_dict({"kind": "sprocket", "id": 1})


def test_a_degenerate_guide_direction_is_reported_as_a_format_error():
    """`Guide.__post_init__` rejects a zero-length direction with ValueError.

    A malformed record is this module's own business to report, and
    BermakeFormatError is the one exception `load_document`'s callers are
    told to catch. Today the outer `document_from_dict` also catches
    ValueError, so the whole-document path was already covered; translating
    here means `annotation_from_dict` keeps its own contract whoever calls
    it, and names the offending record the way every other malformed record
    in this module is named.
    """
    from bermake.io.document_codec import annotation_from_dict

    record = {"kind": "guide", "id": 4, "origin": [0, 0, 0], "direction": [0.0, 0.0, 0.0]}
    with pytest.raises(BermakeFormatError) as excinfo:
        annotation_from_dict(record)
    assert "guide" in str(excinfo.value)


def test_a_degenerate_guide_direction_never_escapes_the_whole_document_load():
    """The path MainWindow._on_file_open actually catches on."""
    from bermake.model.annotation import Guide

    model = Model()
    model.active_context.annotations.append(Guide(0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    data = document_to_dict(model, Camera(), DocumentSettings(), RenderStyle())
    for rec in data["model"]["definitions"]:
        for ann in rec.get("annotations", []):
            if ann.get("kind") == "guide":
                ann["direction"] = [0.0, 0.0, 0.0]

    with pytest.raises(BermakeFormatError):
        document_from_dict(data)


@pytest.mark.parametrize("preset", [PLAIN_WHITE, SKY_AND_GROUND])
def test_environment_round_trips_through_the_codec(preset):
    """All nine fields, not just the colours.

    PLAIN_WHITE alone has sky_enabled=False and ground_enabled=False, which are
    also STUDIO's values for those two fields -- a codec bug that always wrote
    or read back False for them would still pass. SKY_AND_GROUND has both
    flags True, so the pair together actually discriminates.
    """
    restored = environment_from_dict(environment_to_dict(preset))
    assert restored == preset


def test_an_absent_environment_key_yields_the_legacy_environment():
    """A schema-8 file has no "environment" key at all.

    It must reopen dark, because that is what its author saw. Note this is
    LEGACY_ENVIRONMENT (Studio), deliberately NOT DEFAULT_ENVIRONMENT, which is
    what a brand-new document gets. Spec D5.
    """
    assert environment_from_dict(None) == LEGACY_ENVIRONMENT
    assert environment_from_dict({}) == LEGACY_ENVIRONMENT


def test_a_partial_environment_dict_fills_the_rest_from_legacy():
    """Per-key defaulting, matching render_style_from_dict.

    A document written by a build that had fewer fields still loads.
    """
    restored = environment_from_dict({"background": [1.0, 1.0, 1.0]})
    assert restored.background == (1.0, 1.0, 1.0)
    assert restored.edge_color == STUDIO.edge_color
    assert restored.grid_color == STUDIO.grid_color


def test_ground_opacity_is_clamped_to_the_unit_interval():
    """A hand-edited file can carry anything, and GLSL mix() extrapolates.

    Without the clamp, ground_opacity 5.0 drives the ground colour far past the
    sky and reads as a rendering bug rather than a bad file.
    """
    assert environment_from_dict({"ground_opacity": 5.0}).ground_opacity == 1.0
    assert environment_from_dict({"ground_opacity": -2.0}).ground_opacity == 0.0
    assert environment_from_dict({"ground_opacity": 0.4}).ground_opacity == pytest.approx(0.4)


def test_an_explicit_null_ground_opacity_falls_back_like_a_missing_one():
    """Review finding: {"ground_opacity": null} reached float(None) and raised
    TypeError, which document_from_dict surfaces as BermakeFormatError, unlike
    every colour field, where an explicit null already falls back cleanly via
    _environment_color. Per-key defaulting should be uniform across the record.
    """
    restored = environment_from_dict({"ground_opacity": None})
    assert restored.ground_opacity == LEGACY_ENVIRONMENT.ground_opacity


def test_a_colour_triple_is_clamped_to_the_unit_interval_per_channel():
    """The same argument _environment_opacity's docstring makes -- an
    out-of-range value from a hand-edited file would look like a renderer
    defect instead of a bad file -- applies identically to a colour channel:
    GL clamps [5, 5, 5] or [-1, 0, 0] silently and at a different stage.
    """
    restored = environment_from_dict({"background": [5.0, -1.0, 0.4]})
    assert restored.background == pytest.approx((1.0, 0.0, 0.4))


@pytest.mark.parametrize("value", ["", 0, False, [], "blue"])
def test_a_non_object_environment_value_is_a_format_error_not_an_attribute_error(value):
    """Review Focus 1, and open issue #116's exact shape.

    document_from_dict catches KeyError/TypeError/ValueError/IndexError. A JSON
    string here would reach `.get` and raise AttributeError, which that handler
    does not catch and no caller expects, so a truncated file would crash out of
    the open path instead of reporting a bad document.

    The falsy shapes ("", 0, False, []) are the point of fix round 1: the type
    check must run before the "absent" check, or these would silently come back
    as LEGACY_ENVIRONMENT instead of raising. None and {} are the genuinely
    absent/empty cases and stay covered by
    test_an_absent_environment_key_yields_the_legacy_environment, not here.
    """
    with pytest.raises(TypeError):
        environment_from_dict(value)


def test_a_document_with_a_non_object_environment_raises_bermake_format_error():
    """The same input one level up, through the real entry point."""
    data = _minimal_document_dict()
    data["environment"] = "blue"
    with pytest.raises(BermakeFormatError):
        document_from_dict(data)


def test_document_to_dict_emits_the_environment():
    doc = DocumentSettings()
    doc.set_environment(PLAIN_WHITE)
    data = document_to_dict(Model(), Camera(), doc, RenderStyle())
    assert environment_from_dict(data["environment"]) == PLAIN_WHITE


def test_the_schema_version_is_nine():
    """M7.7 is the environment bump. Nothing else in this milestone changes it."""
    assert SCHEMA_VERSION == 9


def test_a_fresh_document_opens_in_the_modelling_environment():
    """DEFAULT_ENVIRONMENT, not the codec's LEGACY_ENVIRONMENT. Spec D5."""
    assert DocumentSettings().environment is DEFAULT_ENVIRONMENT


def test_set_environment_replaces_it_wholesale():
    doc = DocumentSettings()
    doc.set_environment(PLAIN_WHITE)
    assert doc.environment == PLAIN_WHITE


def test_set_environment_does_not_disturb_units():
    """Units and environment are independent axes; the welcome dialog relies on it."""
    doc = DocumentSettings()
    doc.set_metric("mm")
    doc.set_environment(PLAIN_WHITE)
    assert doc.units.metric_unit == "mm"
