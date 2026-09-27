import numpy as np

from bermake.commands.material_commands import ResetFaceUvsCommand
from bermake.scene.scene import Scene, Side, TexturePlacement


def _quad_with_uvs(scene, side=Side.FRONT):
    ids = [
        scene.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    f = scene.add_face_from_loop(ids)
    scene.set_face_uvs(f, [(0.0, 0.0), (0.25, 0.0), (0.5, 0.5), (0.75, 1.0)], side)
    return f


def test_do_clears_the_stored_uvs():
    s = Scene()
    f = _quad_with_uvs(s)
    ResetFaceUvsCommand(f, Side.FRONT).do(s)
    assert s.face_uvs(f, Side.FRONT) is None


def test_undo_restores_them_exactly():
    s = Scene()
    f = _quad_with_uvs(s)
    before = s.face_uvs(f, Side.FRONT)
    cmd = ResetFaceUvsCommand(f, Side.FRONT)
    cmd.do(s)
    cmd.undo(s)
    np.testing.assert_allclose(s.face_uvs(f, Side.FRONT), before, atol=1e-6)


def test_redo_clears_again():
    s = Scene()
    f = _quad_with_uvs(s)
    cmd = ResetFaceUvsCommand(f, Side.FRONT)
    cmd.do(s)
    cmd.undo(s)
    cmd.do(s)
    assert s.face_uvs(f, Side.FRONT) is None


def test_it_leaves_the_placement_alone():
    """Spec D11: reset clears stored UVs only, never the placement."""
    s = Scene()
    f = _quad_with_uvs(s)
    s.set_face_placement(f, TexturePlacement(0.3, 0.4, 2.0, 0.5), Side.FRONT)
    ResetFaceUvsCommand(f, Side.FRONT).do(s)
    p = s.face_placement(f, Side.FRONT)
    assert (p.offset_u, p.offset_v, p.scale, p.rotation) == (0.3, 0.4, 2.0, 0.5)


def test_it_touches_only_the_named_side():
    s = Scene()
    f = _quad_with_uvs(s, Side.FRONT)
    s.set_face_uvs(f, [(0.9, 0.9)] * 4, Side.BACK)
    ResetFaceUvsCommand(f, Side.FRONT).do(s)
    assert s.face_uvs(f, Side.FRONT) is None
    assert s.face_uvs(f, Side.BACK) is not None


def test_resetting_a_face_with_no_stored_uvs_is_a_clean_no_op():
    s = Scene()
    ids = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    ]
    f = s.add_face_from_loop(ids)
    cmd = ResetFaceUvsCommand(f, Side.FRONT)
    cmd.do(s)
    cmd.undo(s)
    assert s.face_uvs(f, Side.FRONT) is None
