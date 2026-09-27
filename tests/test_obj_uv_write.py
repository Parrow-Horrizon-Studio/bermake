from bermake.io.obj_codec import ObjDocument, ObjFace, ObjObject, parse_obj, write_obj


def _doc_with_uvs():
    return ObjDocument(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        uvs=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        objects=(
            ObjObject(
                name="tri",
                faces=(ObjFace(vertex_indices=(0, 1, 2), material="brick", uv_indices=(0, 1, 2)),),
            ),
        ),
        materials={"brick": (1.0, 1.0, 1.0)},
        material_textures={"brick": "brick.png"},
        has_object_tags=True,
    )


def test_vt_lines_are_written():
    obj_text, _ = write_obj(_doc_with_uvs())
    assert "vt 0.000000 0.000000" in obj_text
    assert "vt 1.000000 1.000000" in obj_text


def test_face_tokens_pair_vertex_and_uv():
    obj_text, _ = write_obj(_doc_with_uvs())
    assert "f 1/1 2/2 3/3" in obj_text


def test_map_kd_is_written_to_the_mtl():
    _, mtl_text = write_obj(_doc_with_uvs())
    assert "map_Kd brick.png" in mtl_text


def test_a_material_without_a_texture_gets_no_map_kd():
    doc = ObjDocument(
        vertices=((0.0, 0.0, 0.0),),
        objects=(ObjObject(name="o", faces=()),),
        materials={"plain": (0.5, 0.5, 0.5)},
    )
    _, mtl_text = write_obj(doc)
    assert "map_Kd" not in mtl_text


def test_a_face_without_uvs_writes_bare_vertex_tokens():
    doc = ObjDocument(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
        objects=(ObjObject(name="t", faces=(ObjFace(vertex_indices=(0, 1, 2)),)),),
    )
    obj_text, _ = write_obj(doc)
    assert "f 1 2 3" in obj_text
    assert "/" not in obj_text.split("f 1")[1].split("\n")[0]


def test_uvs_survive_a_write_then_parse_round_trip():
    obj_text, mtl_text = write_obj(_doc_with_uvs())
    back = parse_obj(obj_text, mtl_text)
    assert back.uvs == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    assert back.objects[0].faces[0].uv_indices == (0, 1, 2)
    assert back.material_textures == {"brick": "brick.png"}


def test_a_document_with_no_uvs_writes_no_vt_lines():
    doc = ObjDocument(
        vertices=((0.0, 0.0, 0.0),),
        objects=(ObjObject(name="o", faces=()),),
    )
    obj_text, _ = write_obj(doc)
    assert "vt " not in obj_text
