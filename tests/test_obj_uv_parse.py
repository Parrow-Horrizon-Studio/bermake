from bermake.io.obj_codec import parse_obj


def test_map_kd_is_captured():
    mtl = "newmtl brick\nKd 1.0 1.0 1.0\nmap_Kd brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {"brick": "brick.png"}


def test_map_kd_keeps_a_subdirectory_path():
    mtl = "newmtl brick\nmap_Kd textures/brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {"brick": "textures/brick.png"}


def test_map_kd_options_are_skipped():
    mtl = "newmtl brick\nmap_Kd -s 1 1 1 brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {"brick": "brick.png"}


def test_map_kd_filename_with_spaces_survives():
    mtl = "newmtl brick\nmap_Kd my brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {"brick": "my brick.png"}


def test_a_material_without_map_kd_gets_no_entry():
    mtl = "newmtl plain\nKd 0.5 0.5 0.5\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {}


def test_colors_still_parse_alongside_textures():
    mtl = "newmtl brick\nKd 0.25 0.5 0.75\nmap_Kd brick.png\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.materials == {"brick": (0.25, 0.5, 0.75)}
    assert doc.material_textures == {"brick": "brick.png"}


def test_a_document_with_no_mtl_has_no_textures():
    doc = parse_obj("v 0 0 0\n", None)
    assert doc.material_textures == {}


def test_a_malformed_map_kd_with_no_filename_is_ignored():
    mtl = "newmtl brick\nmap_Kd\n"
    doc = parse_obj("v 0 0 0\n", mtl)
    assert doc.material_textures == {}


def test_vt_pool_is_captured_in_order():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 3/3\n"
    doc = parse_obj(obj, None)
    assert doc.uvs == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_face_uv_indices_are_zero_based_and_parallel():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/1 2/2 3/3\n"
    doc = parse_obj(obj, None)
    face = doc.objects[0].faces[0]
    assert face.vertex_indices == (0, 1, 2)
    assert face.uv_indices == (0, 1, 2)


def test_uv_indices_need_not_match_vertex_indices():
    # The seam case: two corners share a vertex but not a UV.
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 0.5 0.9\nf 1/3 2/1 3/2\n"
    doc = parse_obj(obj, None)
    face = doc.objects[0].faces[0]
    assert face.vertex_indices == (0, 1, 2)
    assert face.uv_indices == (2, 0, 1)


def test_a_face_with_no_vt_has_none():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices is None


def test_the_v_slash_slash_vn_form_has_no_uvs():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices is None


def test_the_v_slash_vt_slash_vn_form_keeps_the_uv():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices == (0, 1, 2)


def test_a_face_with_uvs_on_only_some_corners_gets_none():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nf 1/1 2/2 3\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices is None


def test_negative_vt_indices_are_relative_to_the_pool():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\nf 1/-3 2/-2 3/-1\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices == (0, 1, 2)


def test_an_out_of_range_vt_index_drops_the_face_uvs_rather_than_raising():
    # A bad VERTEX index is a structural error and raises; a bad UV index is
    # recoverable by projecting that face instead.
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nf 1/1 2/9 3/1\n"
    doc = parse_obj(obj, None)
    assert doc.objects[0].faces[0].uv_indices is None
    assert doc.objects[0].faces[0].vertex_indices == (0, 1, 2)


def test_a_three_dimensional_vt_keeps_only_u_and_v():
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0 0\nvt 1 0 0\nvt 1 1 0\nf 1/1 2/2 3/3\n"
    doc = parse_obj(obj, None)
    assert doc.uvs == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))


def test_negative_vt_indices_resolve_against_the_uv_pool_not_the_vertex_pool():
    # Deliberately mismatch pool lengths (4 vertices, 2 vt entries) so resolving
    # a negative vt index against the wrong pool produces an out-of-range index
    # and silently drops the face's UVs. This catches accidental swaps.
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0.25 0.75\nvt 0.5 0.5\nf 1/-2 2/-1 3/-2 4/-1\n"
    doc = parse_obj(obj, None)
    face = doc.objects[0].faces[0]
    # -2 against len(uvs)==2 -> index 0; -1 against len(uvs)==2 -> index 1
    assert face.uv_indices == (0, 1, 0, 1)
    # If resolved against len(vertices)==4 instead, -2 would give index 2,
    # which is out of range and would produce uv_indices is None.
    assert face.uv_indices is not None
