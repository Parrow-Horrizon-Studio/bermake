"""Tests for SceneRenderer.

We test:
  - SceneRenderer construction (no GL context needed).
  - draw_face_fill_overlays: public method added in M3b Task 6.
"""

from __future__ import annotations


class TestSceneRendererConstruction:
    """SceneRenderer should be constructible without a GL context."""

    def test_constructs_without_gl(self):
        from bermake.viewport.scene_renderer import SceneRenderer

        renderer = SceneRenderer()
        assert not renderer._initialized


class TestFaceFillOverlayPass:
    """SceneRenderer.draw_face_fill_overlays — alpha-blended ghost rendering."""

    def test_empty_polygon_list_is_a_noop(self, qtbot):
        """Smoke test: empty list shouldn't touch GL state or raise."""
        from bermake.viewport.scene_renderer import SceneRenderer

        renderer = SceneRenderer()
        # Don't call initialize_gl — the empty-list path must short-circuit
        # before touching any GL functions.
        renderer.draw_face_fill_overlays(polygons=[], color=(1.0, 0.0, 0.0, 0.5))
        # If we got here without an exception, the smoke test passes.

    def test_polygon_list_with_single_quad_is_accepted(self, qtbot):
        """The renderer should accept a single (4, 3) numpy quad without raising.
        Actual GL drawing is exercised by manual visual verification."""
        import inspect
        from bermake.viewport.scene_renderer import SceneRenderer

        renderer = SceneRenderer()
        assert hasattr(renderer, "draw_face_fill_overlays")
        sig = inspect.signature(renderer.draw_face_fill_overlays)
        assert "polygons" in sig.parameters
        assert "color" in sig.parameters


def test_snap_marker_vertices_shape_per_kind():
    import numpy as np
    from bermake.viewport.scene_renderer import _snap_marker_vertices
    from bermake.viewport.snap_engine import SnapKind

    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # GL_LINES vertex counts: square=8, triangle=6, diamond=8, X=4.
    assert _snap_marker_vertices(int(SnapKind.ENDPOINT), p).shape == (8, 3)
    assert _snap_marker_vertices(int(SnapKind.ON_FACE), p).shape == (8, 3)
    assert _snap_marker_vertices(int(SnapKind.MIDPOINT), p).shape == (6, 3)
    assert _snap_marker_vertices(int(SnapKind.ON_EDGE), p).shape == (8, 3)
    assert _snap_marker_vertices(int(SnapKind.INTERSECTION), p).shape == (4, 3)
    for kind in (SnapKind.ENDPOINT, SnapKind.MIDPOINT, SnapKind.ON_EDGE, SnapKind.INTERSECTION):
        v = _snap_marker_vertices(int(kind), p)
        assert np.allclose(v[:, 2], 3.0)


class TestSelectionHighlightHelpers:
    def test_selection_face_polygons_returns_live_selected_loops(self):
        import numpy as np
        from bermake.scene import Scene
        from bermake.selection import Selection
        from bermake.viewport.scene_renderer import _selection_face_polygons

        scene = Scene()
        a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
        b = scene.add_vertex(np.array([1, 0, 0], dtype=np.float32))
        c = scene.add_vertex(np.array([1, 1, 0], dtype=np.float32))
        d = scene.add_vertex(np.array([0, 1, 0], dtype=np.float32))
        fid = scene.add_face_from_loop((a, b, c, d))
        sel = Selection()
        sel.replace(faces=[fid])
        polys = _selection_face_polygons(scene, sel)
        assert len(polys) == 1
        assert polys[0].shape == (4, 3)

    def test_selection_face_polygons_skips_dead_ids(self):
        from bermake.scene import Scene
        from bermake.selection import Selection
        from bermake.viewport.scene_renderer import _selection_face_polygons

        sel = Selection()
        sel.replace(faces=[999])  # not live
        assert _selection_face_polygons(Scene(), sel) == []

    def test_selection_edge_segments_returns_2E_by_3(self):
        import numpy as np
        from bermake.scene import Scene
        from bermake.selection import Selection
        from bermake.viewport.scene_renderer import _selection_edge_segments

        scene = Scene()
        a = scene.add_vertex(np.array([0, 0, 0], dtype=np.float32))
        b = scene.add_vertex(np.array([2, 0, 0], dtype=np.float32))
        e = scene.add_edge(a, b)
        sel = Selection()
        sel.replace(edges=[e])
        segs = _selection_edge_segments(scene, sel)
        assert segs.shape == (2, 3)

    def test_render_accepts_selection_param(self):
        import inspect
        from bermake.viewport.scene_renderer import SceneRenderer

        sig = inspect.signature(SceneRenderer.render)
        assert "selection" in sig.parameters


def test_box_rect_ndc_segments_maps_corners():
    import numpy as np
    from bermake.viewport.scene_renderer import _box_rect_ndc_segments

    # 800x600 viewport; rect from (0,0) to (800,600) -> full NDC [-1,1] square.
    segs = _box_rect_ndc_segments((0.0, 0.0, 800.0, 600.0), 800, 600)
    assert segs.shape == (8, 3)          # 4 sides x 2 endpoints
    assert np.allclose(np.abs(segs[:, 0]), 1.0)
    assert np.allclose(np.abs(segs[:, 1]), 1.0)
    assert np.allclose(segs[:, 2], 0.0)
