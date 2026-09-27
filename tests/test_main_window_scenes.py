from bermake.viewport.render_style import FaceStyle


def _make_window(qtbot):
    from bermake.ui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_create_scene_adds_to_library_and_page(qtbot):
    win = _make_window(qtbot)
    win._on_create_view()
    assert len(win._model.views.views()) == 1
    assert win._scenes_page._list.count() == 1


def test_recall_applies_style_and_starts_animation(qtbot):
    win = _make_window(qtbot)
    # Save a scene while in WIREFRAME:
    win._render_style.face_style = FaceStyle.WIREFRAME
    win._on_create_view()
    vid = win._model.views.views()[0].id
    # Switch to SHADED, then recall — style must snap back to WIREFRAME:
    win._render_style.face_style = FaceStyle.SHADED
    win._on_recall_view(vid)
    assert win._render_style.face_style is FaceStyle.WIREFRAME
    assert win._view_animator.is_running        # camera tween started
    win._view_animator.cancel()


def test_delete_scene_is_undoable(qtbot):
    win = _make_window(qtbot)
    win._on_create_view()
    vid = win._model.views.views()[0].id
    win._on_delete_view(vid)
    assert win._model.views.views() == []
    win._command_stack.undo()
    assert len(win._model.views.views()) == 1


def test_render_style_persists_through_save_new_open(qtbot, tmp_path):
    win = _make_window(qtbot)
    win._render_style.face_style = FaceStyle.MONOCHROME
    win._render_style.xray = True
    path = str(tmp_path / "styled.berm")
    assert win._save_to(path) is True
    win._on_file_new()
    assert win._render_style.face_style is FaceStyle.SHADED   # reset by New
    # Re-open and confirm the saved style is adopted:
    from bermake.io.bermake_file import load_document
    loaded = load_document(path)
    win._reset_document(loaded.model, loaded.camera_state, loaded.units,
                        loaded.style, path, environment=loaded.environment)
    assert win._render_style.face_style is FaceStyle.MONOCHROME
    assert win._render_style.xray is True


def test_clamped_reorder_is_a_true_noop(qtbot):
    win = _make_window(qtbot)
    win._on_create_view()
    win._on_create_view()
    first = win._model.views.views()[0].id
    order_before = [v.id for v in win._model.views.views()]

    win._doc_controller.mark_clean()
    undo_before = win._command_stack.can_undo

    win._on_reorder_view(first, -1)          # already at the top -> no-op

    assert [v.id for v in win._model.views.views()] == order_before
    assert win._doc_controller.dirty is False, "a no-op reorder must not dirty the document"
    assert win._command_stack.can_undo == undo_before, "a no-op reorder must not push undo"


def test_real_reorder_still_works(qtbot):
    win = _make_window(qtbot)
    win._on_create_view()
    win._on_create_view()
    first = win._model.views.views()[0].id

    win._on_reorder_view(first, +1)          # genuine move
    assert win._model.views.views()[1].id == first
    assert win._command_stack.can_undo


def test_recall_does_not_dirty_the_document(qtbot):
    win = _make_window(qtbot)
    win._on_create_view()
    vid = win._model.views.views()[0].id
    win._doc_controller.mark_clean()
    undo_before = win._command_stack.can_undo

    win._on_recall_view(vid)

    assert win._doc_controller.dirty is False, "recall is a view change, not an edit"
    assert win._command_stack.can_undo == undo_before, "recall must not touch the undo stack"
    win._view_animator.cancel()
