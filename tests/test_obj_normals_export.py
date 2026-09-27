"""#113, export half: an exported .obj carries explicit `vn` normals.

Bermake's kernel stores one normal per face, recomputed from the boundary
loop by Newell's method, so what goes out is flat per-face shading. Honouring
IMPORTED per-corner normals is the smooth-shading half and stays open: there
is nowhere for a per-corner normal to live and no consumer for one.
"""

import numpy as np
from bermake.io.obj_codec import parse_obj
from bermake.io.obj_io import export_obj, model_to_objdoc
from bermake.model.definition import Definition
from bermake.model.instance import Instance
from bermake.model.model import Model


def _quad(mesh, z=0.0):
    ids = [
        mesh.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, z), (1, 0, z), (1, 1, z), (0, 1, z)]
    ]
    return mesh.add_face_from_loop(ids)


def test_a_flat_quad_exports_its_own_normal():
    model = Model()
    _quad(model.root.mesh)
    doc = model_to_objdoc(model)
    face = doc.objects[0].faces[0]
    assert face.normal_index is not None
    np.testing.assert_allclose(doc.normals[face.normal_index], (0.0, 0.0, 1.0), atol=1e-6)


def test_two_coplanar_faces_share_one_normal_entry():
    """Normals intern into a shared pool the way UVs already do. A model of
    n coplanar faces must not write n identical `vn` lines."""
    model = Model()
    _quad(model.root.mesh)
    _quad(model.root.mesh, z=1.0)
    doc = model_to_objdoc(model)
    assert len(doc.normals) == 1
    assert {f.normal_index for f in doc.objects[0].faces} == {0}


def test_a_face_in_a_rotated_group_exports_its_world_normal():
    """Export flattens the scene graph to world space, so the normal has to
    travel with it. It transforms by the inverse-transpose of the world
    matrix's linear block (#92), which stays perpendicular under a
    non-uniform scale where the linear block alone would tilt it.
    """
    model = Model()
    child = Definition(1, "Tilted", is_group=True)
    _quad(child.mesh)  # local normal +Z
    # 90 degrees about X, then a non-uniform scale: +Z goes to -Y, and the
    # scale is what separates the inverse-transpose from the naive route.
    rot = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    scale = np.diag([1.0, 1.0, 4.0, 1.0])
    model.root.children.append(Instance(1, child, rot @ scale))

    doc = model_to_objdoc(model)
    tilted = next(o for o in doc.objects if o.name.startswith("Tilted"))
    got = np.asarray(doc.normals[tilted.faces[0].normal_index], dtype=np.float64)
    np.testing.assert_allclose(got, (0.0, -1.0, 0.0), atol=1e-6)


def test_the_written_obj_carries_vn_lines_and_faces_that_reference_them(tmp_path):
    model = Model()
    _quad(model.root.mesh)
    out = tmp_path / "n.obj"
    export_obj(out, model)
    text = out.read_text(encoding="utf-8")
    assert "vn 0.000000 0.000000 1.000000" in text
    assert "f 1//1 2//1 3//1 4//1" in text
    # and the writer's own output still parses
    back = parse_obj(text, None)
    assert len(back.objects[0].faces[0].vertex_indices) == 4


def test_a_degenerate_face_exports_without_a_normal_rather_than_failing():
    """Newell's sum is zero-length on a zero-area face, so there is no
    defensible normal to write. Export is best-effort throughout (a rejected
    material, a missing .mtl), and an exporter that wrote no normals at all
    is exactly what this face falls back to."""
    model = Model()
    mesh = model.root.mesh
    _quad(mesh)
    flat = [
        mesh.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 5), (1, 0, 5), (2, 0, 5)]
    ]
    mesh.add_face_from_loop(flat)  # collinear: zero area

    doc = model_to_objdoc(model)
    faces = doc.objects[0].faces
    assert len(faces) == 2, "both faces still export"
    good_face = next(f for f in faces if len(f.vertex_indices) == 4)
    bad_face = next(f for f in faces if len(f.vertex_indices) == 3)
    assert good_face.normal_index is not None
    assert bad_face.normal_index is None
    assert len(doc.normals) == 1, "the degenerate face contributes no vn line"


def test_a_collapsed_instance_transform_does_not_break_the_export():
    """A zero scale makes the world matrix singular, so there is no
    inverse-transpose to send the normal through. The geometry still exports;
    only the normals for that definition drop."""
    model = Model()
    child = Definition(1, "Squashed", is_group=True)
    _quad(child.mesh)
    model.root.children.append(Instance(1, child, np.diag([1.0, 1.0, 0.0, 1.0])))

    doc = model_to_objdoc(model)
    squashed = next(o for o in doc.objects if o.name.startswith("Squashed"))
    assert len(squashed.faces) == 1
    assert squashed.faces[0].normal_index is None
