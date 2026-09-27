"""Imported glTF UVs land on the right corners, in Bermake's convention."""

import numpy as np

from bermake.io.gltf_import import build_gltf_into_model
from bermake.io.gltf_scene import GltfMaterial, GltfMesh, GltfNode, GltfSceneData
from bermake.model.model import Model
from bermake.scene.scene import Side

# One quad as two triangles. UVs chosen so that u, v, and 1 - v are all
# different from each other at every corner: a missing flip, a doubled flip
# and a u/v swap each produce a distinct wrong answer.
_POS = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
_UVS = ((0.1, 0.2), (0.7, 0.3), (0.8, 0.9), (0.15, 0.75))
_TRIS = ((0, 1, 2), (0, 2, 3))


def _scene(uvs=_UVS, materials=(), material_index=-1):
    mesh = GltfMesh(positions=_POS, triangles=_TRIS, material_index=material_index, uvs=uvs)
    node = GltfNode(name="N", parent=-1, transform=tuple(np.eye(4).flatten()), mesh_indices=(0,))
    return GltfSceneData(nodes=(node,), meshes=(mesh,), materials=tuple(materials))


def _imported_faces(model):
    """(scene, [face ids]) for the single component definition the import built."""
    # Model has no definitions() accessor; traverse() yields every reachable
    # (definition, world_transform) depth-first from the root.
    for defn, _world in model.traverse():
        faces = [f.id for f in defn.mesh.faces_iter()]
        if faces:
            return defn.mesh, faces
    raise AssertionError("import produced no faces")


def test_bridge_uvs_are_stored_without_a_second_flip():
    """The IR already carries Bermake's convention, so import must NOT flip.

    Assimp's glTF2 importer applies 1 - v before the bridge sees a coordinate
    (pinned by test_assimp_already_flips_v_CI_GATE), so a flip here would be
    the second one and would render every imported texture upside down.

    Pinned as explicit literals, so a flip introduced anywhere in this path
    turns 0.2 into 0.8 and fails loudly. The values are deliberately NOT
    symmetric about 0.5, which a flip would leave unchanged.
    """
    model = Model()
    build_gltf_into_model(_scene(), model, model.root)
    mesh, faces = _imported_faces(model)

    # Triangle 0 is glTF vertices (0, 1, 2), so its loop corners carry
    # _UVS[0], _UVS[1], _UVS[2] unchanged.
    got = mesh.face_uvs(faces[0], Side.FRONT)
    np.testing.assert_allclose(got, [(0.1, 0.2), (0.7, 0.3), (0.8, 0.9)], atol=1e-6)

    got1 = mesh.face_uvs(faces[1], Side.FRONT)
    np.testing.assert_allclose(got1, [(0.1, 0.2), (0.8, 0.9), (0.15, 0.75)], atol=1e-6)


def test_u_is_not_touched_either():
    """Discriminates against flipping the wrong axis, or transposing u and v.

    `_UVS` holds no value twice and no corner where u equals v, so a swap
    produces different numbers rather than coinciding with the expectation.
    """
    model = Model()
    build_gltf_into_model(_scene(), model, model.root)
    mesh, faces = _imported_faces(model)
    us = [float(u) for u, _ in mesh.face_uvs(faces[0], Side.FRONT)]
    np.testing.assert_allclose(us, [0.1, 0.7, 0.8], atol=1e-6)


def test_both_sides_receive_the_imported_array():
    """Spec 1.5: a source format has one UV set and Bermake has two."""
    model = Model()
    build_gltf_into_model(_scene(), model, model.root)
    mesh, faces = _imported_faces(model)
    for fid in faces:
        np.testing.assert_allclose(
            mesh.face_uvs(fid, Side.FRONT), mesh.face_uvs(fid, Side.BACK), atol=1e-6
        )
        assert mesh.face_uvs(fid, Side.BACK) is not None


def test_a_mesh_without_uvs_stores_none():
    """Discriminates against storing a zero array, which would permanently
    freeze every untextured import at the origin instead of letting it
    project."""
    model = Model()
    build_gltf_into_model(_scene(uvs=()), model, model.root)
    mesh, faces = _imported_faces(model)
    for fid in faces:
        assert mesh.face_uvs(fid, Side.FRONT) is None
        assert mesh.face_uvs(fid, Side.BACK) is None


def test_a_short_uv_array_degrades_that_face_alone():
    """Import stays best-effort: a malformed face drops to projection and is
    counted, never failing the import (spec 1.5)."""
    model = Model()
    result = build_gltf_into_model(_scene(uvs=((0.1, 0.2), (0.7, 0.3))), model, model.root)
    mesh, faces = _imported_faces(model)
    assert len(faces) == 2, "geometry must still import"
    assert all(mesh.face_uvs(fid, Side.FRONT) is None for fid in faces)
    assert result.summary.faces_without_uvs == 2


def _decoder(data):
    """The Qt-free decoder shape bermake/io expects, for a 4x4 opaque PNG."""
    return ("png", 4, 4, False)


def _textured_scene():
    from bermake.io.gltf_scene import GltfImage

    img = GltfImage(name="uvgrid", data=b"\x89PNG\r\n\x1a\npretend", format_hint="png")
    mat = GltfMaterial("Brick", (1.0, 1.0, 1.0), texture_index=0, texture_uri="")
    scene = _scene(materials=(mat,), material_index=0)
    return GltfSceneData(
        nodes=scene.nodes, meshes=scene.meshes, materials=scene.materials, images=(img,)
    )


def test_a_textured_material_gains_a_texture():
    model = Model()
    scene = _textured_scene()
    build_gltf_into_model(scene, model, model.root, texture_bytes={0: scene.images[0].data},
                          decoder=_decoder)
    mats = [m for m in model.materials.materials() if m.name == "Brick"]
    assert len(mats) == 1
    tex = model.textures.get(mats[0].texture_id)
    assert tex is not None
    assert tex.data == scene.images[0].data, "the ORIGINAL bytes, not a re-encode"
    assert (tex.width, tex.height) == (4, 4)


def test_an_undecodable_image_leaves_the_material_untextured_and_counts_it():
    model = Model()
    scene = _textured_scene()
    result = build_gltf_into_model(
        scene, model, model.root, texture_bytes={0: b"not an image"}, decoder=lambda d: None
    )
    mats = [m for m in model.materials.materials() if m.name == "Brick"]
    assert mats[0].texture_id is None
    assert result.summary.images_skipped == 1


def test_an_image_dropped_upstream_is_counted_even_with_empty_texture_bytes():
    """read_gltf_texture_bytes DROPS an entry entirely (never a placeholder)
    for an image that is missing, unreadable, outside the containment
    directory, or raw texels with no encoded form. So a document that lost
    every one of its images hands back an EMPTY texture_bytes dict, and
    `not {}` is True -- the trap this test pins is treating that as "nothing
    was attempted" instead of "everything referenced was lost".
    """
    model = Model()
    mat = GltfMaterial("Brick", (1.0, 1.0, 1.0), texture_index=-1, texture_uri="missing.png")
    scene = _scene(materials=(mat,), material_index=0)
    result = build_gltf_into_model(scene, model, model.root, texture_bytes={}, decoder=_decoder)
    mats = [m for m in model.materials.materials() if m.name == "Brick"]
    assert mats[0].texture_id is None
    assert result.summary.images_skipped == 1


def test_an_upstream_drop_and_a_decode_failure_both_reach_the_counter():
    """Two different causes of 'this material stayed untextured' must both
    land in images_skipped, so one cause discovered first in testing does
    not mask a regression in the other."""
    model = Model()
    dropped = GltfMaterial("Dropped", (1.0, 0.0, 0.0), texture_index=-1, texture_uri="missing.png")
    undecodable = GltfMaterial("Undecodable", (0.0, 1.0, 0.0), texture_index=1, texture_uri="")
    mesh = GltfMesh(positions=_POS, triangles=_TRIS, material_index=0)
    node = GltfNode(name="N", parent=-1, transform=tuple(np.eye(4).flatten()), mesh_indices=(0,))
    scene = GltfSceneData(nodes=(node,), meshes=(mesh,), materials=(dropped, undecodable))
    result = build_gltf_into_model(
        scene,
        model,
        model.root,
        texture_bytes={1: b"not an image"},
        decoder=lambda d: None,
    )
    assert result.summary.images_skipped == 2


def test_a_material_with_no_texture_reference_is_never_counted():
    """Discriminates against the count degenerating into 'number of
    materials': a plain, never-textured material must contribute 0."""
    model = Model()
    mat = GltfMaterial("Plain", (0.5, 0.5, 0.5))  # texture_index=-1, texture_uri="" (defaults)
    scene = _scene(materials=(mat,), material_index=0)
    result = build_gltf_into_model(scene, model, model.root, texture_bytes={}, decoder=_decoder)
    assert result.summary.images_skipped == 0


def test_reimporting_the_same_document_does_not_pile_up_textures():
    """The accumulation regression stage 2 shipped and then fixed in OBJ."""
    model = Model()
    scene = _textured_scene()
    for _ in range(5):
        build_gltf_into_model(scene, model, model.root, texture_bytes={0: scene.images[0].data},
                              decoder=_decoder)
    assert len(list(model.textures.textures())) == 1
    # Exact "Brick" or a numbered pileup variant ("Brick.001", ...); NOT a
    # plain startswith, which would also catch the builtin "Brick Red" swatch
    # every Model() seeds (model/material.py's _BUILTIN_PALETTE).
    brick_named = [
        m for m in model.materials.materials() if m.name == "Brick" or m.name.startswith("Brick.")
    ]
    assert len(brick_named) == 1


def test_a_material_colliding_with_a_differently_textured_one_mints_a_new_material():
    """Finding 7 (M7.5c-3 whole-branch review): the ledger left this branch
    untested, reasoning it was "structurally identical to OBJ's proven
    code" (test_obj_texture_import.py's
    test_reusing_a_material_with_a_different_texture_does_not_retexture_it
    and its three-import sequel). It is not identical: `_ensure_gltf_materials`
    searches for a reusable minted material by `(texture_id, base_color)`, a
    key OBJ's equivalent code does not use, and it is roughly 25 lines of new
    branching logic the OBJ test cannot exercise. This pins the same
    accumulation bug stage 2 shipped and then fixed for OBJ, on the glTF
    side: a name+color collision with a material that already carries a
    DIFFERENT texture must mint a new material rather than repoint the
    existing one (which would silently retexture every pre-existing face
    painted with it), and a repeat import of the same colliding document
    must reuse that minted material rather than piling up a third one."""
    model = Model()
    scene = _textured_scene()  # GltfMaterial("Brick", (1.0, 1.0, 1.0), texture_index=0)

    # A pre-existing "Brick" material with the name+color the import will
    # collide on, already textured with a DIFFERENT image than the one this
    # import carries.
    old_mat = model.materials.add_custom("Brick", (1.0, 1.0, 1.0))
    old_tex = model.textures.add("Brick", b"\x89PNG\r\n\x1a\n-old-bytes", "png", 2, 2, False)
    model.materials.edit(old_mat.id, texture_id=old_tex.id)

    build_gltf_into_model(
        scene, model, model.root, texture_bytes={0: scene.images[0].data}, decoder=_decoder
    )

    # The pre-existing material is untouched: never repointed at the new image.
    unchanged = model.materials.get(old_mat.id)
    assert unchanged.texture_id == old_tex.id

    # The import got its own new material, textured with the NEW image.
    textured = [m for m in model.materials.materials() if m.texture_id is not None]
    assert len(textured) == 2, "the original Brick plus one minted for the collision"
    new_mat = next(m for m in textured if m.id != old_mat.id)
    new_tex = model.textures.get(new_mat.texture_id)
    assert new_tex.data == scene.images[0].data

    # A second, identical import reuses the minted material+texture rather
    # than piling up a third material and a duplicate byte-identical image.
    build_gltf_into_model(
        scene, model, model.root, texture_bytes={0: scene.images[0].data}, decoder=_decoder
    )
    textured_after = [m for m in model.materials.materials() if m.texture_id is not None]
    assert len(textured_after) == 2
    assert len(list(model.textures.textures())) == 2
    still_unchanged = model.materials.get(old_mat.id)
    assert still_unchanged.texture_id == old_tex.id
