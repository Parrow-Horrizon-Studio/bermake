"""Tests for pick_selectable and entities_in_box with a world_transform.

These tests verify that:
1. The world_transform keyword argument exists on pick_selectable,
   entities_in_box, and SnapEngine.snap.
2. Passing None (default) or identity gives the same result as no arg.
3. A non-identity translation transform correctly shifts the picking location:
   geometry at local position (0,0,0) inside a group translated +10 in X
   is only hit when the cursor is aimed at the world position (+10,0,0).
"""

from __future__ import annotations

import inspect

import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _camera(w, h):
    from bermake.viewport.camera import Camera
    cam = Camera()
    cam.aspect = float(w) / float(h)
    return cam


def _identity():
    return np.eye(4, dtype=np.float64)


def _translation(dx, dy=0.0, dz=0.0):
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = dx
    m[1, 3] = dy
    m[2, 3] = dz
    return m


# ---------------------------------------------------------------------------
# signature checks
# ---------------------------------------------------------------------------

def test_pick_selectable_accepts_world_transform_kwarg():
    from bermake.viewport.picking import pick_selectable
    sig = inspect.signature(pick_selectable)
    assert "world_transform" in sig.parameters


def test_entities_in_box_accepts_world_transform_kwarg():
    from bermake.viewport.picking import entities_in_box
    sig = inspect.signature(entities_in_box)
    assert "world_transform" in sig.parameters


def test_snap_engine_snap_accepts_world_transform_kwarg():
    from bermake.viewport.snap_engine import SnapEngine
    sig = inspect.signature(SnapEngine.snap)
    assert "world_transform" in sig.parameters


# ---------------------------------------------------------------------------
# identity / None regression: both must match the no-arg behaviour
# ---------------------------------------------------------------------------

def test_pick_selectable_identity_matches_no_arg():
    """Passing world_transform=None and world_transform=I4 both equal no-arg."""
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    scene.add_edge(a, b)

    mid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx, sy, _ = cam.world_to_screen(mid, w, h)
    cursor = (sx, sy)
    vp = (w, h)

    hit_no_arg = pick_selectable(cursor, vp, cam, scene)
    hit_none = pick_selectable(cursor, vp, cam, scene, world_transform=None)
    hit_identity = pick_selectable(cursor, vp, cam, scene, world_transform=_identity())

    assert hit_no_arg == hit_none
    assert hit_no_arg == hit_identity


# ---------------------------------------------------------------------------
# BEHAVIOURAL test: translated world_transform shifts the hit location
# ---------------------------------------------------------------------------

def test_pick_selectable_translated_world_transform():
    """Geometry at local (0,0,0) with world_transform=T(+10,0,0) should be
    hit at the screen position of world point (+10,0,0), NOT at world (0,0,0).

    The scene stores vertices in LOCAL space (around the origin).  When we
    pass world_transform=T(+10x), pick_selectable must transform each vertex
    to world before projecting to screen.  So:
      - cursor aimed at world (+10,0,0) → hit
      - cursor aimed at world (0,0,0)   → miss
    """
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable
    from bermake.geometry.transforms import mat_translate

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()

    # Local geometry: a short edge centred on the origin.
    a = scene.add_vertex(np.array([-0.5, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([0.5, 0.0, 0.0], dtype=np.float32))
    e = scene.add_edge(a, b)

    wt = mat_translate([10.0, 0.0, 0.0])  # group moved +10 in X

    # The edge's local midpoint is (0,0,0).  In world space it lives at (10,0,0).
    world_mid = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    sx_hit, sy_hit, _ = cam.world_to_screen(world_mid, w, h)

    # Screen position of the UNtransformed local midpoint (0,0,0).
    local_mid = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sx_miss, sy_miss, _ = cam.world_to_screen(local_mid, w, h)

    # Cursor at the WORLD position of the edge → should hit.
    hit = pick_selectable((sx_hit, sy_hit), (w, h), cam, scene, world_transform=wt)
    assert hit is not None, (
        "Expected an edge hit at the translated world location but got None"
    )
    assert hit == ("edge", e), f"Expected ('edge', {e}), got {hit}"

    # Cursor at the LOCAL position (ignoring transform) → should miss.
    miss = pick_selectable((sx_miss, sy_miss), (w, h), cam, scene, world_transform=wt)
    assert miss is None or miss[1] != e, (
        "Cursor at un-translated origin should NOT hit the edge when world_transform is applied"
    )


def test_pick_selectable_identity_and_none_equivalent_to_untransformed():
    """Passing identity must give the same result as passing None and as no-arg
    for a scene that lives at the origin (canonical regression guard)."""
    from bermake.scene import Scene
    from bermake.viewport.picking import pick_selectable

    w, h = 800, 600
    cam = _camera(w, h)
    scene = Scene()
    a = scene.add_vertex(np.array([-1.0, -1.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, -1.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([-1.0, 1.0, 0.0], dtype=np.float32))
    fid = scene.add_face_from_loop((a, b, c, d))

    # Face centre
    centre = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cx, cy, _ = cam.world_to_screen(centre, w, h)

    hit_no_arg = pick_selectable((cx, cy), (w, h), cam, scene)
    hit_none = pick_selectable((cx, cy), (w, h), cam, scene, world_transform=None)
    hit_identity = pick_selectable((cx, cy), (w, h), cam, scene, world_transform=_identity())

    assert hit_no_arg == hit_none == hit_identity


# ---------------------------------------------------------------------------
# M4e review-fix: ToolContext.model field + is_identity_transform helper
# ---------------------------------------------------------------------------

def test_tool_context_has_model_field_defaulting_to_none():
    """ToolContext must expose a `model` field that defaults to None."""
    from bermake.tools.tool import ToolContext

    class _FakeScene:
        pass

    ctx = ToolContext(scene=_FakeScene())
    assert hasattr(ctx, "model"), "ToolContext must have a 'model' attribute"
    assert ctx.model is None, "ToolContext.model default must be None"


def test_tool_context_model_can_be_set():
    """ToolContext.model must accept an arbitrary object (the Model)."""
    from bermake.tools.tool import ToolContext

    class _FakeScene:
        pass

    class _FakeModel:
        pass

    ctx = ToolContext(scene=_FakeScene(), model=_FakeModel())
    assert isinstance(ctx.model, _FakeModel)


def test_is_identity_transform_none_is_true():
    from bermake.geometry.transforms import is_identity_transform
    assert is_identity_transform(None) is True


def test_is_identity_transform_eye4_is_true():
    from bermake.geometry.transforms import is_identity_transform
    assert is_identity_transform(np.eye(4)) is True


def test_is_identity_transform_translated_is_false():
    from bermake.geometry.transforms import is_identity_transform
    m = np.eye(4)
    m[0, 3] = 1.0  # translate X by 1
    assert is_identity_transform(m) is False
