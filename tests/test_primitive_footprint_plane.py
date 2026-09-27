"""M7.5a Task 13 (#106): the footprint preview and the commit share a plane.

`PrimitiveTool.overlay` used to draw the stage-one rubber band with a
hardcoded world Z=0, while `_commit_footprint` (correctly) resolves the
footprint in the active context's *local* frame. Inside a group translated
in Z, those disagree: the rubber band floats at literal world Z=0 while the
primitive actually lands on the group's own local ground plane (which sits
at a different world Z). This test drives the tool exactly as a user would
-- corners snapped onto the group's own floor, which after a Z translation
report a world Z equal to that translation -- and checks the *rendered*
(world-space) preview against the *rendered* (world-space) commit.

A test that only inspects `_build_ghost_polygons()` or the committed mesh's
local vertex positions cannot fail here: both are already local-frame and
already agree even in the broken code, since `_commit_footprint` was never
the buggy side. Only comparing world-space output -- what `overlay()`
actually hands the renderer versus where the primitive actually ends up in
world space -- exposes the mismatch.
"""

from __future__ import annotations

import numpy as np

from bermake.tools.primitive_tool import _State


def test_a_primitive_lands_on_the_plane_its_preview_drew(main_window, group_factory):
    win = main_window
    model = win._model
    scene = model.active_context.mesh
    v = [
        scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32)),
        scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32)),
    ]
    for a, b in zip(v, v[1:] + v[:1], strict=True):
        scene.add_edge(a, b)
    scene.add_face_from_loop(v)
    inst = group_factory(model)
    # Translated in Z: the group's own local Z=0 floor now sits at world Z=7.
    inst.transform[:3, 3] = [0.0, 0.0, 7.0]
    model.enter(inst)

    win._activate("box")
    tool = win._tool_manager.active

    # Stage one: the user drags a footprint with both corners snapped onto
    # the group's own floor -- i.e. reported at world Z=7, exactly what a
    # real vertex/edge snap onto that (now-translated) geometry would give.
    tool._state = _State.DRAGGING_FOOTPRINT
    tool._first_corner = np.array([0.0, 0.0, 7.0], dtype=np.float64)
    tool._preview_corner = np.array([2.0, 2.0, 7.0], dtype=np.float64)

    preview_segments = tool.overlay().rubber_band_segments
    assert preview_segments.shape[0] > 0
    preview_world_z = float(preview_segments[0][2])

    tool._commit_footprint(np.array([2.0, 2.0, 7.0], dtype=np.float64))
    tool._commit_primitive(width=2.0, depth_=2.0, height=1.0)

    inner = model.active_context.mesh
    placed_local_z = min(float(inner.vertex(x.id).position[2]) for x in inner.vertices_iter())
    placed_world_z = placed_local_z + 7.0  # translation-only transform: local -> world

    # Before the fix: preview_world_z == 0.0 (hardcoded), placed_world_z ==
    # 7.0 (correctly local-frame). A group translation other than 7.0 would
    # still expose the same 7.0-unit gap; 7.0 is simply the group's offset
    # used here, chosen to be unmistakably nonzero.
    assert round(preview_world_z, 4) == round(placed_world_z, 4)
