import dataclasses

from bermake.io.document_codec import CameraState
from bermake.views.saved_view import SavedView


def _cam():
    return CameraState(position=(1.0, 2.0, 3.0), target=(0.0, 0.0, 0.0),
                       up=(0.0, 0.0, 1.0), fov_y_deg=45.0)


def test_saved_view_holds_its_fields():
    v = SavedView(id=7, name="Front", camera=_cam(),
                  tag_visibility={2: False}, face_style="SHADED", xray=True)
    assert v.id == 7
    assert v.name == "Front"
    assert v.camera.position == (1.0, 2.0, 3.0)
    assert v.tag_visibility == {2: False}
    assert v.face_style == "SHADED"
    assert v.xray is True


def test_saved_view_is_frozen():
    v = SavedView(1, "A", _cam(), {}, "SHADED", False)
    try:
        v.name = "B"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("SavedView must be frozen")
