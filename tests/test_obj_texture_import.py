"""Texture import for OBJ. No QApplication: the decoder is injected (spec D7)."""

from bermake.io.obj_codec import parse_obj
from bermake.io.obj_io import build_obj_into_model, read_obj_texture_bytes
from bermake.model.model import Model

FAKE_PNG = b"\x89PNG\r\n\x1a\n-fake-bytes"


def _stub_decoder(data):
    """Stands in for texture_cache.decode_image without importing Qt."""
    if not data.startswith(b"\x89PNG"):
        return None
    return ("png", 4, 8, False)


def _write_obj_tree(tmp_path, map_kd="brick.png", write_image=True):
    obj = tmp_path / "m.obj"
    obj.write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "usemtl brick\nf 1/1 2/2 3/3\n",
        encoding="utf-8",
    )
    (tmp_path / "m.mtl").write_text(
        f"newmtl brick\nKd 1 1 1\nmap_Kd {map_kd}\n", encoding="utf-8"
    )
    if write_image:
        target = tmp_path / map_kd
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(FAKE_PNG)
    return obj


def test_texture_bytes_are_read_from_beside_the_obj(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    assert read_obj_texture_bytes(obj, doc) == {"brick": FAKE_PNG}


def test_a_subdirectory_texture_resolves(tmp_path):
    obj = _write_obj_tree(tmp_path, map_kd="tex/brick.png")
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    assert read_obj_texture_bytes(obj, doc) == {"brick": FAKE_PNG}


def test_a_missing_image_file_is_skipped_not_fatal(tmp_path):
    obj = _write_obj_tree(tmp_path, write_image=False)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    assert read_obj_texture_bytes(obj, doc) == {}


def test_a_dotdot_texture_outside_the_document_directory_is_skipped(tmp_path):
    doc_dir = tmp_path / "proj"
    doc_dir.mkdir()
    obj = doc_dir / "m.obj"
    obj.write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "usemtl brick\nf 1/1 2/2 3/3\n",
        encoding="utf-8",
    )
    (doc_dir / "m.mtl").write_text(
        "newmtl brick\nKd 1 1 1\nmap_Kd ../secret.png\n", encoding="utf-8"
    )
    # The real file exists, one level above the document directory, so this
    # proves containment rather than merely proving the file was absent.
    (tmp_path / "secret.png").write_bytes(FAKE_PNG)
    doc = parse_obj(obj.read_text(), (doc_dir / "m.mtl").read_text())
    assert read_obj_texture_bytes(obj, doc) == {}


def test_an_absolute_path_texture_outside_the_document_directory_is_skipped(tmp_path):
    doc_dir = tmp_path / "proj"
    doc_dir.mkdir()
    obj = doc_dir / "m.obj"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    abs_target = outside_dir / "secret.png"
    abs_target.write_bytes(FAKE_PNG)
    obj.write_text(
        "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 1 1 0\nvt 0 0\nvt 1 0\nvt 1 1\n"
        "usemtl brick\nf 1/1 2/2 3/3\n",
        encoding="utf-8",
    )
    (doc_dir / "m.mtl").write_text(
        f"newmtl brick\nKd 1 1 1\nmap_Kd {abs_target}\n", encoding="utf-8"
    )
    doc = parse_obj(obj.read_text(), (doc_dir / "m.mtl").read_text())
    assert read_obj_texture_bytes(obj, doc) == {}


def test_an_imported_texture_lands_in_the_library(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()
    build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
    )
    textures = model.textures.textures()
    assert len(textures) == 1
    assert textures[0].data == FAKE_PNG
    assert (textures[0].width, textures[0].height) == (4, 8)
    assert textures[0].image_format == "png"


def test_the_material_points_at_the_imported_texture(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()
    build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
    )
    brick = next(m for m in model.materials.materials() if m.name == "brick")
    assert brick.texture_id == model.textures.textures()[0].id


def test_undecodable_bytes_are_skipped_and_the_material_stays_untextured(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()
    build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": b"not-an-image"}, decoder=_stub_decoder
    )
    assert model.textures.textures() == []
    brick = next(m for m in model.materials.materials() if m.name == "brick")
    assert brick.texture_id is None


def test_import_without_a_decoder_imports_geometry_and_no_textures(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()
    result = build_obj_into_model(doc, model, model.root)
    assert model.textures.textures() == []
    assert result.summary.faces_imported == 1


def test_reimporting_the_same_document_does_not_duplicate_the_texture(tmp_path):
    obj = _write_obj_tree(tmp_path)
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()
    build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
    )
    build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
    )
    textures = model.textures.textures()
    assert len(textures) == 1
    brick = next(m for m in model.materials.materials() if m.name == "brick")
    assert brick.texture_id == textures[0].id


def test_reusing_a_material_with_a_different_texture_does_not_retexture_it(tmp_path):
    """Finding 1 (M7.5c-2 review): a material reused by name+color that
    already carries a texture must never be repointed at the import's image.
    That library edit is not undone by ImportObjCommand, so it would
    silently retexture every pre-existing face already painted with it. The
    import should get its own material instead."""
    obj = _write_obj_tree(tmp_path, map_kd="brick_v2.png")
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()

    # A pre-existing "brick" material with the color the import will match,
    # already textured with a different image than the one being imported.
    mat = model.materials.add_custom("brick", (1.0, 1.0, 1.0))
    old_tex = model.textures.add("brick", b"\x89PNG\r\n\x1a\n-old-bytes", "png", 2, 2, False)
    model.materials.edit(mat.id, texture_id=old_tex.id)

    result = build_obj_into_model(
        doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
    )

    # The pre-existing material is untouched.
    unchanged = model.materials.get(mat.id)
    assert unchanged.texture_id == old_tex.id

    # The imported face got its own material, textured with the new image.
    fid = result.created_geometry[2][0]
    new_mid = model.root.mesh.face_material(fid)
    assert new_mid != mat.id
    new_mat = model.materials.get(new_mid)
    assert new_mat.texture_id is not None
    new_tex = model.textures.get(new_mat.texture_id)
    assert new_tex.data == FAKE_PNG


def test_reimporting_a_colliding_document_three_times_does_not_pile_up_materials_or_textures(
    tmp_path,
):
    """Re-review fix: the first cut of finding 1 minted a fresh `_unique_name`
    every time the OBJ's "brick" collided with the pre-existing one, because
    the name lookup only ever matched the pre-existing material, never the
    "brick.001" a previous colliding import had already created with these
    exact bytes. Three imports of the same colliding document must land on
    the same second material+texture pair, not one new pair per import."""
    obj = _write_obj_tree(tmp_path, map_kd="brick_v2.png")
    doc = parse_obj(obj.read_text(), (tmp_path / "m.mtl").read_text())
    model = Model()

    mat = model.materials.add_custom("brick", (1.0, 1.0, 1.0))
    old_tex = model.textures.add("brick", b"\x89PNG\r\n\x1a\n-old-bytes", "png", 2, 2, False)
    model.materials.edit(mat.id, texture_id=old_tex.id)

    for _ in range(3):
        build_obj_into_model(
            doc, model, model.root, texture_bytes={"brick": FAKE_PNG}, decoder=_stub_decoder
        )

    textured = [m for m in model.materials.materials() if m.texture_id is not None]
    assert len(textured) == 2  # the original "brick" plus one for the collision
    assert len(model.textures.textures()) == 2

    # The pre-existing material must still be exactly as it was.
    unchanged = model.materials.get(mat.id)
    assert unchanged.texture_id == old_tex.id


def test_the_io_package_imports_no_qt():
    """Spec D7: bermake/io stays Qt-free so its tests need no QApplication.

    The scanned root is derived from this test file's own location, not the
    cwd: a relative "python/bermake/io" glob silently returns nothing (so the
    assertion passes vacuously) when pytest is run from anywhere other than
    the repo root (a reviewer confirmed this running from F:/tmp).
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "python" / "bermake" / "io"
    assert root.is_dir()  # guard: a wrong derived path must fail loudly, not glob to nothing
    offenders = [
        p.name
        for p in root.glob("*.py")
        if "PySide6" in p.read_text(encoding="utf-8")
        or "texture_cache" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []
