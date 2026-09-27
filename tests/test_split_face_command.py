"""SplitFaceCommand: undo, redo, and id preservation (M7.6a)."""

import numpy as np
from bermake.commands.scene_commands import SplitFaceCommand
from bermake.scene.scene import Scene


def _quad_with_chord():
    s = Scene()
    v = [
        s.add_vertex(np.array(p, dtype=np.float32))
        for p in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    ]
    fid = s.add_face_from_loop(v)
    s.add_edge(v[0], v[2])
    return s, fid, v


def test_do_splits_and_records_the_new_ids():
    s, fid, v = _quad_with_chord()
    cmd = SplitFaceCommand(fid, [v[0], v[2]])
    cmd.do(s)
    assert cmd.new_face_ids is not None
    assert len(list(s.faces_iter())) == 2


def test_undo_restores_the_original_face_and_its_id():
    s, fid, v = _quad_with_chord()
    original_loop = tuple(s.face_loop(fid))
    cmd = SplitFaceCommand(fid, [v[0], v[2]])
    cmd.do(s)
    cmd.undo(s)
    assert len(list(s.faces_iter())) == 1
    assert tuple(s.face_loop(fid)) == original_loop


def test_redo_reuses_the_first_runs_face_ids():
    """A sibling command in the same gesture composite may have cached a new
    face id. Minting fresh ids on redo would strand it."""
    s, fid, v = _quad_with_chord()
    cmd = SplitFaceCommand(fid, [v[0], v[2]])
    cmd.do(s)
    first_ids = cmd.new_face_ids
    cmd.undo(s)
    cmd.do(s)
    assert cmd.new_face_ids == first_ids
    live = {f.id for f in s.faces_iter()}
    assert set(first_ids) == live, "Scene has no face_is_live; faces_iter yields live ones"


def test_a_refused_split_is_a_clean_noop_through_undo():
    s, fid, v = _quad_with_chord()
    cmd = SplitFaceCommand(fid, [v[0], v[1]])  # adjacent corners: refused
    cmd.do(s)
    assert cmd.new_face_ids is None
    assert len(list(s.faces_iter())) == 1
    cmd.undo(s)
    assert len(list(s.faces_iter())) == 1
