"""The shared sweep layer is Qt-free and command-based (M7.4 Task 3)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
from bermake.scene.scene import Scene
from bermake.tools.sweep_support import loft_between_loops, offset_polygon, seam_merge


def _square(scene, z=0.0):
    v = [
        scene.add_vertex(np.array([0.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, z], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, z], dtype=np.float32)),
    ]
    scene.add_face_from_loop(v)
    return v


def test_importing_sweep_support_loads_no_qt():
    # Same guarantee selection_controller carries: this maths must be
    # testable with no QApplication.
    code = (
        "import sys; import bermake.tools.sweep_support; "
        "print(any(m.startswith('PySide6') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_loft_creates_one_destination_vertex_per_source_vertex():
    scene = Scene()
    loop = _square(scene)
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    result = loft_between_loops(scene, loop, dst, cap_start=False, cap_end=False)
    assert len(result.dst_vertex_ids) == len(loop)
    assert len(set(result.dst_vertex_ids)) == len(loop)


def test_loft_creates_one_side_face_per_edge():
    scene = Scene()
    loop = _square(scene)
    before = len(list(scene.faces_iter()))
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=False, cap_end=False)
    # Four sides, no caps.
    assert len(list(scene.faces_iter())) == before + 4


def test_caps_are_added_only_when_asked():
    scene = Scene()
    loop = _square(scene)
    before = len(list(scene.faces_iter()))
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=True, cap_end=True)
    # Four sides plus two caps.
    assert len(list(scene.faces_iter())) == before + 6


def test_loft_commands_are_already_executed():
    # The caller uses push_executed, not execute. If the helper returned
    # unexecuted commands the geometry would be missing until undo/redo.
    scene = Scene()
    loop = _square(scene)
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    result = loft_between_loops(scene, loop, dst, cap_start=False, cap_end=False)
    assert result.dst_vertex_ids
    for vid in result.dst_vertex_ids:
        assert scene.vertex(vid) is not None
    assert result.commands


def test_seam_merge_dissolves_only_coplanar_pairs():
    scene = Scene()
    loop = _square(scene)
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=False, cap_end=False)
    # The four side faces meet at right angles: nothing is coplanar, so a
    # seam merge over the source boundary must be a no-op.
    cmds = seam_merge(scene, list(scene.face_edges(next(scene.faces_iter()).id)))
    assert cmds == []


def test_loft_does_not_execute_commands_a_second_time():
    # If loft_between_loops built commands but relied on the caller to
    # execute them (i.e. it forgot the eager .do(scene) call), the scene
    # would have no new vertices/faces immediately after the call returns.
    # This is the same guarantee as test_loft_commands_are_already_executed
    # but pinned to face count too, since a vertex-only bug wouldn't be
    # caught by the vertex check alone.
    scene = Scene()
    loop = _square(scene)
    before_faces = len(list(scene.faces_iter()))
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=True, cap_end=True)
    # If commands were built but never executed, faces_iter would still
    # report `before_faces` here.
    assert len(list(scene.faces_iter())) == before_faces + 6


def test_start_cap_winds_opposite_the_sweep_direction():
    # The start cap must be the SOURCE loop reversed, so its normal points
    # opposite the sweep (down, when sweeping up +Z). A cap built from the
    # source loop un-reversed would still close the prism but with the
    # normal flipped — this must be caught even though face count alone
    # cannot distinguish the two.
    #
    # Note: unlike Push/Pull's real usage, the original source face is still
    # present here (loft_between_loops never removes it), so the scene ends
    # up with two faces spanning the same vertex set: the original face and
    # the new start cap. The original face id is captured before the loft
    # call so the cap can be picked out unambiguously.
    scene = Scene()
    loop = _square(scene)
    original_face_id = next(scene.faces_iter()).id
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=True, cap_end=False)

    start_cap = None
    for face in scene.faces_iter():
        if face.id == original_face_id:
            continue
        loop_ids = scene.face_loop(face.id)
        if set(loop_ids) == set(loop):
            start_cap = face
            break
    assert start_cap is not None, "expected a start-cap face over the source loop"

    normal = scene.face_normal(start_cap.id)
    # The source square lies in the z=0 plane; sweeping toward +Z means the
    # start cap's normal must point toward -Z.
    assert normal[2] < 0, f"start cap normal should point away from sweep, got {normal}"


def test_end_cap_winds_in_source_winding_not_reversed():
    # The end cap must be the DESTINATION loop in SOURCE winding, so its
    # normal points along the sweep direction (up, when sweeping up +Z).
    scene = Scene()
    loop = _square(scene)
    dst = [scene.vertex(v).position + np.array([0, 0, 1], np.float32) for v in loop]
    loft_between_loops(scene, loop, dst, cap_start=False, cap_end=True)

    end_cap = None
    for face in scene.faces_iter():
        loop_ids = set(scene.face_loop(face.id))
        if loop_ids.isdisjoint(loop):
            end_cap = face
            break
    assert end_cap is not None, "expected an end-cap face over the destination loop"

    normal = scene.face_normal(end_cap.id)
    assert normal[2] > 0, f"end cap normal should point along sweep, got {normal}"


def test_offset_polygon_never_returns_a_degenerate_polygon_past_the_limit():
    # Fix round (post Task-6 review, Finding 1): offset_polygon's clamp must
    # land STRICTLY short of collapse, not exactly at it. A 4x4 square
    # collapses at distance 2.0; asking for something far beyond that (not a
    # knife-edge request) must still produce a polygon with every edge of
    # strictly positive length and no two coincident vertices -- otherwise
    # Scene.add_vertex welds the coincident points and the caller's loft
    # request becomes an illegal self-loop edge.
    square = np.array(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]],
        dtype=np.float64,
    )
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pts, clamped = offset_polygon(square, z, 1000.0)

    assert 0.0 < clamped < 2.0
    n = len(pts)
    for i in range(n):
        edge_len = float(np.linalg.norm(pts[(i + 1) % n] - pts[i]))
        assert edge_len > 0.0, f"edge {i} has zero length at clamped={clamped}"
    for i in range(n):
        for j in range(i + 1, n):
            assert float(np.linalg.norm(pts[i] - pts[j])) > 0.0, (
                f"vertices {i} and {j} coincide at clamped={clamped}"
            )
