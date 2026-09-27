import numpy as np
from bermake.io.obj_codec import ObjDocument, ObjFace, ObjObject
from bermake.io.obj_io import build_obj_into_model, model_to_objdoc
from bermake.model.model import Model


def _add_quad(scene):
    vids = [scene.add_vertex(np.array(p, dtype=np.float32))
            for p in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))]
    scene.add_face_from_loop(vids)
    return next(iter(scene.faces_iter())).id


def test_model_to_objdoc_paints_and_world_transforms_and_dedups_names():
    model = Model()
    # root geometry, one face painted
    fid = _add_quad(model.root.mesh)
    teal = model.materials.add_custom("My Teal", (0.1, 0.6, 0.6))
    model.root.mesh.set_face_material(fid, teal.id)

    # a component placed twice at different transforms
    comp = model.new_definition("Chair", is_group=False)
    _add_quad(comp.mesh)
    for tx in (5.0, 9.0):
        t = np.eye(4, dtype=np.float64)
        t[0, 3] = tx
        model.root.children.append(model.new_instance(comp, t))

    doc = model_to_objdoc(model)

    # one object per traversed node with geometry: root + 2 chairs
    names = [o.name for o in doc.objects]
    assert names[0] == "Model"
    assert "Chair" in names and "Chair.001" in names          # de-duplicated
    # world transform applied: the two chairs' vertices are offset by tx
    xs = sorted({round(v[0], 3) for v in doc.vertices})
    assert 5.0 in xs and 9.0 in xs
    # painted material captured (name sanitized)
    assert "My_Teal" in doc.materials
    assert doc.materials["My_Teal"] == (0.1, 0.6, 0.6)
    assert doc.has_object_tags is True

    # find objects by name (robust to traversal order), not by list position
    by_name = {o.name: o for o in doc.objects}
    root_obj = by_name["Model"]
    chair_obj = by_name["Chair"]

    # the root's single face is preserved as a 4-vertex n-gon (not triangulated),
    # its remapped indices are all valid into the shared vertex pool, and the
    # painted material is attached to that face.
    (root_face,) = root_obj.faces
    assert len(root_face.vertex_indices) == 4
    assert all(0 <= i < len(doc.vertices) for i in root_face.vertex_indices)
    assert root_face.material == "My_Teal"

    # an unpainted Chair face carries no material
    (chair_face,) = chair_obj.faces
    assert chair_face.material is None
    assert len(chair_face.vertex_indices) == 4
    assert all(0 <= i < len(doc.vertices) for i in chair_face.vertex_indices)


def _tri(a, b, c, mat=None):
    return ObjFace((a, b, c), mat)


def test_build_grouped_creates_one_group_per_object():
    model = Model()
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 0, 0), (3, 0, 0), (2, 1, 0)),
        objects=(ObjObject("A", (_tri(0, 1, 2),)), ObjObject("B", (_tri(3, 4, 5),))),
        materials={},
        has_object_tags=True,
    )
    result = build_obj_into_model(doc, model, model.active_context)
    assert result.summary.objects == 2
    assert result.summary.faces_imported == 2
    assert len(model.active_context.children) == 2      # two groups
    assert len(result.created_instances) == 2
    assert result.created_geometry == ([], [], [])      # group case records instances, not geom


def test_build_merged_adds_to_active_scene_no_group():
    model = Model()
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("default", (_tri(0, 1, 2),)),),
        materials={},
        has_object_tags=False,
    )
    result = build_obj_into_model(doc, model, model.active_context)
    assert result.summary.objects == 0
    assert len(model.active_context.children) == 0                      # no group
    assert len(list(model.active_context.mesh.faces_iter())) == 1       # merged in place
    # single triangle: created_geometry records exactly its 3 vertices + 1 face for undo
    assert len(result.created_geometry[0]) == 3
    assert len(result.created_geometry[2]) == 1


def test_build_best_effort_skips_out_of_range_index():
    model = Model()
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("default", (_tri(0, 1, 2), _tri(0, 1, 99))),),  # 99 out of range
        materials={},
        has_object_tags=False,
    )
    result = build_obj_into_model(doc, model, model.active_context)
    assert result.summary.faces_imported == 1
    assert result.summary.faces_skipped == 1


def test_build_grouped_best_effort_skips_out_of_range_index():
    model = Model()
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("A", (_tri(0, 1, 2), _tri(0, 1, 99))),),  # 99 out of range
        materials={},
        has_object_tags=True,
    )
    result = build_obj_into_model(doc, model, model.active_context)   # no exception
    assert result.summary.faces_imported == 1
    assert result.summary.faces_skipped == 1


def test_build_best_effort_skips_degenerate_face():
    model = Model()
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("default", (_tri(0, 1, 2), _tri(0, 0, 1))),),   # 2nd is degenerate
        materials={},
        has_object_tags=False,
    )
    result = build_obj_into_model(doc, model, model.active_context)
    assert result.summary.faces_imported == 1
    assert result.summary.faces_skipped == 1


def test_build_dedups_materials_by_name_and_color():
    model = Model()
    before = len(model.materials.materials())
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("default", (_tri(0, 1, 2, "Red"),)),),
        materials={"Red": (0.7, 0.2, 0.2)},
        has_object_tags=False,
    )
    build_obj_into_model(doc, model, model.active_context)
    build_obj_into_model(doc, model, model.active_context)   # again
    reds = [m for m in model.materials.materials() if m.name == "Red"]
    assert len(reds) == 1                                    # no duplicate
    assert len(model.materials.materials()) == before + 1
