"""M3c PushPullTool: end-to-end closed-manifold extrusion (Case 1 + Case 2)."""

from __future__ import annotations

import numpy as np

from bermake.commands.command_stack import CommandStack
from bermake.scene.scene import Scene
from bermake.tools.push_pull_tool import PushPullTool


def _draw_rectangle(scene: Scene, w: float = 1.0, h: float = 1.0) -> int:
    """Draw a w×h rectangle on the ground plane (z=0), return the face id."""
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([w, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([w, h, 0], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, h, 0], dtype=np.float32))
    scene.add_edge(v0, v1)
    scene.add_edge(v1, v2)
    scene.add_edge(v2, v3)
    scene.add_edge(v3, v0)
    return scene.add_face_from_loop([v0, v1, v2, v3])


def _draw_pentagon(scene: Scene) -> int:
    """Draw a regular pentagon centred at origin on z=0, return the face id."""
    import math

    verts = []
    for i in range(5):
        a = 2 * math.pi * i / 5
        v = scene.add_vertex(np.array([math.cos(a), math.sin(a), 0], dtype=np.float32))
        verts.append(v)
    for i in range(5):
        scene.add_edge(verts[i], verts[(i + 1) % 5])
    return scene.add_face_from_loop(verts)


def _live_vertex_positions(scene: Scene) -> list:
    """Sorted multiset of live vertex positions, rounded for float-stable compare."""
    return sorted(
        tuple(round(float(c), 5) for c in scene.vertex(v.id).position)
        for v in scene.vertices_iter()
    )


def _commit_pp_directly(scene: Scene, face_id: int, depth: float) -> CommandStack:
    """Skip the click-move-click state machine: directly invoke _commit_extrusion
    with armed-face state populated, simulating a user gesture."""
    tool = PushPullTool()
    stack = CommandStack()
    tool._scene = scene
    tool._command_stack = stack
    tool._armed_face_id = face_id
    tool._armed_face_loop = list(scene.face(face_id).loop_vertex_ids)
    tool._armed_face_normal = scene.face_normal(face_id)
    tool._armed_face_center = scene.face_center(face_id)
    tool._current_depth = depth
    tool._commit_extrusion()
    return stack


# ---- Case 1 — standalone source produces closed manifold ------------------

def test_case1_standalone_rect_produces_closed_prism():
    """P/P on a standalone rectangle → 6 faces total (4 sides + top + bottom)."""
    scene = Scene()
    f = _draw_rectangle(scene)
    _commit_pp_directly(scene, f, depth=2.0)

    live_face_count = sum(1 for _ in scene.faces_iter())
    assert live_face_count == 6


def test_case1_bottom_face_normal_points_down():
    """Bottom cap's normal must oppose the extrusion direction. Else
    backface culling, lighting, and pickability all break — exactly the
    regression that surfaced in M3b without a normal-direction check."""
    scene = Scene()
    f_src = _draw_rectangle(scene)
    src_normal = scene.face_normal(f_src).copy()  # before removal
    _commit_pp_directly(scene, f_src, depth=1.5)

    # Find the bottom face — it's the one whose normal opposes src_normal.
    bottoms = [
        f
        for f in scene.faces_iter()
        if float(np.dot(scene.face_normal(f.id), src_normal)) < 0
    ]
    assert len(bottoms) == 1, "Expected exactly one face with normal opposing src"


def test_case1_pentagon_source_produces_7_faces():
    """N-gon source → N+2 faces (N sides + top + bottom)."""
    scene = Scene()
    f = _draw_pentagon(scene)
    _commit_pp_directly(scene, f, depth=0.5)

    assert sum(1 for _ in scene.faces_iter()) == 7


def _build_unit_box(scene: Scene) -> int:
    """Draw a 1x1 rect and P/P it up by 1 -> returns the id of the new TOP face.
    The standard "existing solid" fixture for Case 2 tests."""
    f_src = _draw_rectangle(scene, 1.0, 1.0)
    _commit_pp_directly(scene, f_src, depth=1.0)
    for f in scene.faces_iter():
        verts = [scene.vertex(vid).position for vid in f.loop_vertex_ids]
        if all(abs(v[2] - 1.0) < 1e-4 for v in verts):
            return f.id
    raise RuntimeError("Could not locate top face of unit box")


# ---- Case 2 — P/P on existing solid produces seamless extension ----------

def test_case2_stacked_pp_face_count_correct():
    """Box (6 faces) + P/P top upward -> still 6 faces (taller box)."""
    scene = Scene()
    top = _build_unit_box(scene)
    assert sum(1 for _ in scene.faces_iter()) == 6  # baseline

    _commit_pp_directly(scene, top, depth=1.0)

    # After seam-merge, the 4 old side faces merged with the 4 new side faces:
    # 4 merged sides + 1 new top + 1 original bottom = 6.
    assert sum(1 for _ in scene.faces_iter()) == 6


def test_case2_no_bottom_cap_for_attached_source():
    """No face should exist at the OLD top height (z=1.0) after the second P/P."""
    scene = Scene()
    top = _build_unit_box(scene)
    _commit_pp_directly(scene, top, depth=1.0)

    for f in scene.faces_iter():
        verts = [scene.vertex(vid).position for vid in f.loop_vertex_ids]
        all_at_one = all(abs(v[2] - 1.0) < 1e-4 for v in verts)
        assert not all_at_one, f"Face {f.id} should not be at z=1.0 after seam merge"


def test_case2_old_top_loop_edges_dissolved():
    """All 4 edges of the OLD top loop must be tombstoned after Case 2 P/P."""
    scene = Scene()
    top = _build_unit_box(scene)
    old_top_edge_ids = list(scene.face_edges(top))
    assert len(old_top_edge_ids) == 4
    assert all(scene._mesh.edge_is_live(e) for e in old_top_edge_ids)

    _commit_pp_directly(scene, top, depth=1.0)

    for e in old_top_edge_ids:
        assert not scene._mesh.edge_is_live(e), (
            f"Edge {e} on the OLD top loop should have been dissolved by seam-merge"
        )


def test_case2_composite_undoes_atomically():
    """One Ctrl+Z must restore the pre-P/P state exactly — face + edge counts
    AND vertex geometry."""
    scene = Scene()
    top = _build_unit_box(scene)
    pre_face_count = sum(1 for _ in scene.faces_iter())
    pre_edge_count = sum(1 for _ in scene.edges_iter())
    pre_positions = _live_vertex_positions(scene)

    stack = _commit_pp_directly(scene, top, depth=1.0)
    stack.undo()

    assert sum(1 for _ in scene.faces_iter()) == pre_face_count
    assert sum(1 for _ in scene.edges_iter()) == pre_edge_count
    assert _live_vertex_positions(scene) == pre_positions


def test_tilted_source_seam_merge_works():
    """A tilted source face's normal is geometry-derived and the coplanarity
    test is rotation-agnostic, so seam-merge still fires on a tilted box."""
    import math

    scene = Scene()
    s = math.sin(math.radians(30))
    c = math.cos(math.radians(30))
    v0 = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
    v1 = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
    v2 = scene.add_vertex(np.array([1, c, s], dtype=np.float32))
    v3 = scene.add_vertex(np.array([0, c, s], dtype=np.float32))
    scene.add_edge(v0, v1)
    scene.add_edge(v1, v2)
    scene.add_edge(v2, v3)
    scene.add_edge(v3, v0)
    f_src = scene.add_face_from_loop([v0, v1, v2, v3])

    _commit_pp_directly(scene, f_src, depth=1.0)
    assert sum(1 for _ in scene.faces_iter()) == 6

    # The new top face is the only one with all vertices above z=0.5.
    top = None
    for f in scene.faces_iter():
        verts = [scene.vertex(vid).position for vid in f.loop_vertex_ids]
        if len(verts) == 4 and all(v[2] > 0.5 for v in verts):
            top = f.id
            break
    assert top is not None

    pre_count = sum(1 for _ in scene.faces_iter())
    _commit_pp_directly(scene, top, depth=0.5)
    # The 4 old + 4 new side faces merge -> still 6 faces.
    assert sum(1 for _ in scene.faces_iter()) == pre_count


def test_case2_composite_redo_restores_merged_state():
    """do -> undo -> redo of the WHOLE Case-2 P/P composite must return to the
    merged (taller box) state: still 6 faces, and the OLD-top edges dissolved
    again. Locks in the full-composite redo path (seam-merge re-resolution +
    AddFaceCommand redo together)."""
    scene = Scene()
    top = _build_unit_box(scene)
    old_top_edge_ids = list(scene.face_edges(top))

    stack = _commit_pp_directly(scene, top, depth=1.0)
    merged_face_count = sum(1 for _ in scene.faces_iter())
    assert merged_face_count == 6

    stack.undo()
    stack.redo()

    assert sum(1 for _ in scene.faces_iter()) == 6
    for e in old_top_edge_ids:
        assert not scene._mesh.edge_is_live(e), (
            f"Edge {e} should be dissolved again after redo"
        )
