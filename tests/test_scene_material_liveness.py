"""#115: the face queries must not report faces that no longer exist.

Scene.remove_face and split_face deliberately leave the sidecar entries of the
retired face behind (restore_face relies on that for undo), so the queries
filter by liveness instead.
"""

from __future__ import annotations

import numpy as np
import pytest
from bermake.commands.command_stack import CommandStack
from bermake.commands.material_commands import DeleteMaterialCommand
from bermake.model.model import Model
from bermake.scene.scene import DEFAULT_PLACEMENT, Scene, Side, TexturePlacement

_QUAD = ([0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0])


def _quad(scene):
    v = [scene.add_vertex(np.array(p, dtype=np.float32)) for p in _QUAD]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    return v, scene.add_face_from_loop(v)


def _split(scene, v, f):
    scene.add_edge(v[0], v[2])
    assert scene.split_face(f, [v[0], v[2]]) is not None


def _live(scene, f_id):
    return scene._mesh.face_is_live(f_id)


def test_a_split_painted_face_counts_only_live_faces():
    scene = Scene()
    v, f = _quad(scene)
    scene.set_face_material(f, 7, Side.FRONT)
    _split(scene, v, f)
    found = scene.faces_with_material(7)
    assert not _live(scene, f)
    assert all(_live(scene, fid) for fid, _side in found)
    assert len(found) == 2


def test_the_back_side_is_filtered_too():
    scene = Scene()
    v, f = _quad(scene)
    scene.set_face_material(f, 7, Side.BACK)
    _split(scene, v, f)
    found = scene.faces_with_material(7)
    assert len(found) == 2
    assert {side for _fid, side in found} == {Side.BACK}


def test_a_removed_painted_face_is_not_counted_but_returns_on_restore():
    scene = Scene()
    v, f = _quad(scene)
    scene.set_face_material(f, 7)
    assert scene.faces_with_material(7) == [(f, Side.FRONT)]
    scene.remove_face(f)
    assert scene.faces_with_material(7) == []
    scene.restore_face(f, v)
    assert scene.faces_with_material(7) == [(f, Side.FRONT)]


@pytest.mark.parametrize("side", [Side.FRONT, Side.BACK])
def test_a_split_face_with_placement_counts_only_live_faces(side):
    scene = Scene()
    v, f = _quad(scene)
    placement = TexturePlacement(scale=2.0)
    assert placement != DEFAULT_PLACEMENT
    scene.set_face_placement(f, placement, side)
    _split(scene, v, f)
    found = scene.faces_with_placement()
    assert all(_live(scene, fid) for fid, _side in found)
    assert len(found) == 2
    assert {s for _fid, s in found} == {side}


@pytest.mark.parametrize("side", [Side.FRONT, Side.BACK])
def test_a_split_face_with_stored_uvs_counts_only_live_faces(side):
    scene = Scene()
    v, f = _quad(scene)
    scene.set_face_uvs(f, [[0, 0], [1, 0], [1, 1], [0, 1]], side)
    _split(scene, v, f)
    found = scene.faces_with_uvs()
    assert all(_live(scene, fid) for fid, _side in found)
    assert len(found) == 2
    assert {s for _fid, s in found} == {side}


def test_delete_material_quotes_only_live_faces_after_a_split():
    model = Model()
    scene = model.root.mesh
    v, f = _quad(scene)
    mid = model.materials.add_custom("Brick", (0.7, 0.3, 0.2)).id
    scene.set_face_material(f, mid)
    _split(scene, v, f)
    assert DeleteMaterialCommand(model.materials, mid, model).affected_count == 2


def test_delete_material_after_a_split_undoes_and_redoes_on_the_live_faces():
    model = Model()
    scene = model.root.mesh
    v, f = _quad(scene)
    mid = model.materials.add_custom("Brick", (0.7, 0.3, 0.2)).id
    scene.set_face_material(f, mid)
    _split(scene, v, f)
    stack = CommandStack()

    stack.execute(DeleteMaterialCommand(model.materials, mid, model), model)
    assert scene.faces_with_material(mid) == []

    assert stack.undo()
    painted = scene.faces_with_material(mid)
    assert len(painted) == 2
    assert all(_live(scene, fid) for fid, _side in painted)

    assert stack.redo()
    assert scene.faces_with_material(mid) == []
