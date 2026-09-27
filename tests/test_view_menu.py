from __future__ import annotations

import pytest
from bermake.ui.main_window import MainWindow
from bermake.viewport.render_style import FaceStyle


@pytest.fixture
def win(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_view_menu_defaults_to_shaded(win):
    assert win._render_style.face_style is FaceStyle.SHADED
    assert win._render_style.xray is False
    assert win._face_style_actions[FaceStyle.SHADED].isChecked()
    assert not win._xray_action.isChecked()


def test_face_style_actions_are_mutually_exclusive(win):
    win._on_set_face_style(FaceStyle.WIREFRAME)
    assert win._render_style.face_style is FaceStyle.WIREFRAME
    assert win._face_style_actions[FaceStyle.WIREFRAME].isChecked()
    assert not win._face_style_actions[FaceStyle.SHADED].isChecked()


def test_set_face_style_propagates_to_renderer(win):
    win._on_set_face_style(FaceStyle.HIDDEN_LINE)
    assert win._viewport.scene_renderer._render_style.face_style is FaceStyle.HIDDEN_LINE


def test_xray_toggle_is_independent_of_face_style(win):
    win._on_set_face_style(FaceStyle.MONOCHROME)
    win._on_toggle_xray(True)
    assert win._render_style.xray is True
    assert win._render_style.face_style is FaceStyle.MONOCHROME
    assert win._viewport.scene_renderer._render_style.xray is True
