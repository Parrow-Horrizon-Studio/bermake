import pytest
from bermake.io.errors import BermakeFormatError
from bermake.io.obj_codec import (
    ObjDocument,
    ObjFace,
    ObjObject,
    parse_obj,
    sanitize_material_name,
    write_obj,
)


def test_sanitize_material_name():
    assert sanitize_material_name("Brick Red") == "Brick_Red"
    assert sanitize_material_name("  a  b ") == "a_b"
    assert sanitize_material_name("") == "material"


def test_write_obj_quad_with_material():
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)),
        objects=(ObjObject("Wall", (ObjFace((0, 1, 2, 3), "Brick_Red"),)),),
        materials={"Brick_Red": (0.70, 0.27, 0.22)},
        has_object_tags=True,
    )
    obj_text, mtl_text = write_obj(doc, "house.mtl")
    lines = obj_text.splitlines()
    assert "mtllib house.mtl" in lines
    assert "v 0.000000 0.000000 0.000000" in lines
    assert "o Wall" in lines
    assert "usemtl Brick_Red" in lines
    assert "f 1 2 3 4" in lines  # 1-based, n-gon preserved
    assert "newmtl Brick_Red" in mtl_text
    assert "Kd 0.700000 0.270000 0.220000" in mtl_text


def test_write_obj_no_materials_returns_none_mtl():
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (0, 1, 0)),
        objects=(ObjObject("Tri", (ObjFace((0, 1, 2), None),)),),
        materials={},
        has_object_tags=True,
    )
    obj_text, mtl_text = write_obj(doc)
    assert mtl_text is None
    assert "mtllib" not in obj_text
    assert "usemtl" not in obj_text  # unpainted face emits no usemtl
    assert "f 1 2 3" in obj_text


def test_write_obj_groups_multiple_faces_per_material_into_one_usemtl_run():
    """write_obj sorts faces (unpainted first, then by material name) so each
    material gets exactly ONE `usemtl` line covering a contiguous run of
    faces, not a `usemtl` re-emitted before every single face. Only a
    single-face case was previously covered — this exercises >1 face per
    material plus an interleaved unpainted face and multiple materials."""
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)),
        objects=(
            ObjObject(
                "Room",
                (
                    ObjFace((0, 1, 2), "Blue"),
                    ObjFace((1, 4, 5), None),  # unpainted, sorts first
                    ObjFace((2, 3, 0), "Blue"),
                    ObjFace((0, 2, 4), "Red"),
                ),
            ),
        ),
        materials={"Blue": (0.2, 0.3, 0.8), "Red": (0.8, 0.2, 0.2)},
        has_object_tags=True,
    )
    obj_text, _mtl_text = write_obj(doc, "room.mtl")
    lines = obj_text.splitlines()
    usemtl_lines = [i for i, ln in enumerate(lines) if ln.startswith("usemtl")]

    # Exactly one usemtl per material (Blue's two faces share a single run).
    assert [lines[i] for i in usemtl_lines] == ["usemtl Blue", "usemtl Red"]

    # The unpainted face's `f` line precedes the first usemtl (sort-first, no
    # usemtl emitted for it), and both Blue faces' `f` lines fall between the
    # Blue usemtl and the Red usemtl (a contiguous run, not re-split).
    f_lines = [i for i, ln in enumerate(lines) if ln.startswith("f ")]
    assert f_lines[0] < usemtl_lines[0]  # unpainted face first
    assert usemtl_lines[0] < f_lines[1] < f_lines[2] < usemtl_lines[1]  # both Blue faces grouped


def test_parse_multi_object_with_materials():
    obj = "\n".join([
        "mtllib x.mtl",
        "v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0",
        "o A", "usemtl Red", "f 1 2 3 4",
        "o B", "f 1 2 3",           # triangle, no material
    ])
    mtl = "newmtl Red\nKd 0.7 0.2 0.2\n"
    doc = parse_obj(obj, mtl)
    assert doc.has_object_tags is True
    assert len(doc.vertices) == 4
    assert [o.name for o in doc.objects] == ["A", "B"]
    assert doc.objects[0].faces[0].vertex_indices == (0, 1, 2, 3)  # 1-based -> 0-based, n-gon
    assert doc.objects[0].faces[0].material == "Red"
    assert doc.objects[1].faces[0].material is None
    assert doc.materials["Red"] == (0.7, 0.2, 0.2)


def test_parse_flat_file_has_no_object_tags():
    doc = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", None)
    assert doc.has_object_tags is False
    assert len(doc.objects) == 1
    assert doc.objects[0].faces[0].vertex_indices == (0, 1, 2)


def test_parse_face_triplets_and_negative_indices():
    # a/vt/vn triplets -> take the v index; negative = relative to verts so far
    doc = parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 -1//3\n", None)
    assert doc.objects[0].faces[0].vertex_indices == (0, 1, 2)  # -1 -> last (index 2)


def test_parse_bad_face_index_raises():
    with pytest.raises(BermakeFormatError):
        parse_obj("v 0 0 0\nv 1 0 0\nf 1 2 9\n", None)   # 9 out of range
    with pytest.raises(BermakeFormatError):
        parse_obj("v 0 0 0\nf 1 x 2\n", None)            # non-numeric


def test_round_trip_write_then_parse():
    doc = ObjDocument(
        vertices=((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (1, 1, 1)),
        objects=(
            ObjObject("Quad", (ObjFace((0, 1, 2, 3), "Teal"),)),
            ObjObject("Tri", (ObjFace((0, 1, 4), None),)),
        ),
        materials={"Teal": (0.1, 0.6, 0.6)},
        has_object_tags=True,
    )
    obj_text, mtl_text = write_obj(doc, "x.mtl")
    back = parse_obj(obj_text, mtl_text)
    assert [o.name for o in back.objects] == ["Quad", "Tri"]
    assert back.objects[0].faces[0].vertex_indices == (0, 1, 2, 3)
    assert back.objects[0].faces[0].material == "Teal"
    assert back.objects[1].faces[0].material is None
    assert back.materials["Teal"] == (0.1, 0.6, 0.6)
    assert back.has_object_tags is True


def test_write_obj_emits_a_vn_line_per_normal():
    """#113: an exported .obj carried no explicit normals at all, so a viewer
    that expects them had to re-derive or flat-shade. Bermake already computes
    one normal per face; writing it costs a line."""
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        objects=(ObjObject("o", (ObjFace((0, 1, 2), normal_index=0),)),),
        normals=((0.0, 0.0, 1.0),),
    )
    text, _ = write_obj(doc)
    assert "vn 0.000000 0.000000 1.000000" in text


def test_a_face_with_a_normal_and_no_uvs_writes_the_double_slash_form():
    """`f v//n` is how OBJ says "normal, no texture coordinate". Writing
    `f v/n` instead would make the normal index read as a vt index."""
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        objects=(ObjObject("o", (ObjFace((0, 1, 2), normal_index=0),)),),
        normals=((0.0, 0.0, 1.0),),
    )
    text, _ = write_obj(doc)
    assert "f 1//1 2//1 3//1" in text


def test_a_face_with_both_uvs_and_a_normal_writes_the_full_triple():
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        objects=(
            ObjObject("o", (ObjFace((0, 1, 2), uv_indices=(0, 1, 2), normal_index=0),)),
        ),
        uvs=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        normals=((0.0, 0.0, 1.0),),
    )
    text, _ = write_obj(doc)
    assert "f 1/1/1 2/2/1 3/3/1" in text


def test_a_face_with_no_normal_writes_no_normal_field():
    doc = ObjDocument(
        vertices=((0, 0, 0), (1, 0, 0), (1, 1, 0)),
        objects=(ObjObject("o", (ObjFace((0, 1, 2)),)),),
    )
    text, _ = write_obj(doc)
    assert "f 1 2 3" in text
    assert "vn " not in text
