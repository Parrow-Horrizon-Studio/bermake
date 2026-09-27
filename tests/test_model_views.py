from bermake.model.model import Model
from bermake.views.view_library import ViewLibrary


def test_new_model_has_empty_view_library():
    m = Model()
    assert isinstance(m.views, ViewLibrary)
    assert m.views.views() == []


def test_load_from_copies_views():
    src = Model()
    from bermake.io.document_codec import CameraState
    from bermake.views.saved_view import SavedView
    cam = CameraState(position=(1.0, 0.0, 0.0), target=(0.0, 0.0, 0.0),
                      up=(0.0, 0.0, 1.0), fov_y_deg=45.0)
    src.views.add(SavedView(0, "Front", cam, {}, "SHADED", False))

    dst = Model()
    dst.load_from(src)
    assert [v.name for v in dst.views.views()] == ["Front"]
    assert dst.views is src.views   # adopted by reference, like tags/materials
