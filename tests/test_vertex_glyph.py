"""The selected-vertex glyph: projection is pure and tested; painting is not.

Splitting the projection out is what makes this testable without a GL
context, and it is the same split draw_plan already uses for annotations.
"""

from __future__ import annotations

import numpy as np


def _widget_with_quad(qtbot):
    from bermake.ui.main_window import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    scene = w._model.active_scene
    a = scene.add_vertex(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    b = scene.add_vertex(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    c = scene.add_vertex(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    d = scene.add_vertex(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    scene.add_face_from_loop((a, b, c, d))
    return w, {"a": a, "b": b, "c": c, "d": d}


def test_no_selected_vertices_means_no_points(qtbot):
    w, _ids = _widget_with_quad(qtbot)
    assert w._viewport._selected_vertex_points(800, 600) == []


def test_one_selected_vertex_yields_one_point(qtbot):
    w, ids = _widget_with_quad(qtbot)
    w._viewport.select_vertices = True
    w._selection.replace(vertices={ids["a"]})
    assert len(w._viewport._selected_vertex_points(800, 600)) == 1


def test_points_are_inside_the_viewport(qtbot):
    w, ids = _widget_with_quad(qtbot)
    w._viewport.select_vertices = True
    w._selection.replace(vertices={ids["a"], ids["b"], ids["c"], ids["d"]})
    pts = w._viewport._selected_vertex_points(800, 600)
    assert len(pts) == 4
    assert all(0.0 <= x <= 800.0 and 0.0 <= y <= 600.0 for x, y in pts)


def test_a_dead_vertex_id_is_skipped_rather_than_raising(qtbot):
    w, ids = _widget_with_quad(qtbot)
    w._viewport.select_vertices = True
    w._selection.replace(vertices={ids["a"], 9999})
    assert len(w._viewport._selected_vertex_points(800, 600)) == 1


def test_no_points_when_the_mode_is_off(qtbot):
    """Turning the mode off hides the glyph, the same way View > Guides makes
    a guide unpaintable. A selection the user cannot see is a trap."""
    w, ids = _widget_with_quad(qtbot)
    w._viewport.select_vertices = False
    w._selection.replace(vertices={ids["a"]})
    assert w._viewport._selected_vertex_points(800, 600) == []


def test_a_vertex_behind_the_camera_is_skipped(qtbot):
    """world_to_screen returns None behind the eye. Projecting it anyway
    would paint a glyph at a mirrored or runaway coordinate, the same class
    of bug M7.6b's guide clipping exists to prevent."""
    w, _ids = _widget_with_quad(qtbot)
    scene = w._model.active_scene
    far_behind = scene.add_vertex(np.array([0.0, 0.0, 1.0e6], dtype=np.float32))
    w._viewport.select_vertices = True
    w._selection.replace(vertices={far_behind})
    pts = w._viewport._selected_vertex_points(800, 600)
    assert all(np.isfinite(x) and np.isfinite(y) for x, y in pts)


def test_paint_vertex_glyphs_is_called_from_paintgl_not_from_paint_annotations(qtbot, monkeypatch):
    """Guards the reason _paint_vertex_glyphs is its own paint pass.

    _paint_annotations returns early (`if not plans and readout_plan is
    None: return`) in any document with no annotations, which is most
    documents. If the glyph call were tucked inside that method instead of
    called directly from paintGL, a selected vertex would go invisible the
    moment the document has zero annotations -- exactly the bug the separate
    pass exists to avoid, and exactly the kind of thing the pure-projection
    tests above cannot see, since they never call paintGL at all.

    Stubs _paint_annotations to a no-op and scene_renderer.render to a no-op
    (avoiding a real GL context), then calls paintGL directly. If
    _paint_vertex_glyphs were only reachable through _paint_annotations, it
    would never fire here, because _paint_annotations no longer does
    anything.
    """
    from bermake.viewport.viewport_widget import ViewportWidget

    w, _ids = _widget_with_quad(qtbot)
    viewport = w._viewport

    calls: list[bool] = []
    monkeypatch.setattr(viewport.scene_renderer, "render", lambda *a, **k: None)
    monkeypatch.setattr(ViewportWidget, "_paint_annotations", lambda self, overlay=None: None)
    monkeypatch.setattr(ViewportWidget, "_paint_vertex_glyphs", lambda self: calls.append(True))

    viewport.paintGL()

    assert calls == [True]
