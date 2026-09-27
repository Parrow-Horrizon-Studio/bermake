from bermake.model.tag import TagLibrary
from bermake.viewport.camera import Camera
from bermake.viewport.render_style import FaceStyle, RenderStyle
from bermake.views.capture import apply_tags_and_style, apply_view, capture_view


def _tags():
    lib = TagLibrary()
    lib.add("Walls")   # id 1
    lib.add("Roof")    # id 2
    return lib


def test_capture_snapshots_camera_tags_and_style():
    cam = Camera()
    cam.position[:] = (5.0, 6.0, 7.0)
    tags = _tags()
    tags.set_visible(2, False)   # hide Roof
    style = RenderStyle(face_style=FaceStyle.WIREFRAME, xray=True)

    v = capture_view(0, "Test", cam, tags, style)

    assert v.id == 0 and v.name == "Test"
    assert v.camera.position == (5.0, 6.0, 7.0)
    assert v.face_style == "WIREFRAME"
    assert v.xray is True
    # Untagged (id 0) is excluded; Walls visible, Roof hidden.
    assert v.tag_visibility == {1: True, 2: False}


def test_apply_view_restores_all_three():
    tags = _tags()
    style = RenderStyle()
    v = capture_view(0, "V", _capture_source_camera(), _hidden_roof_tags(),
                     RenderStyle(face_style=FaceStyle.MONOCHROME, xray=True))
    cam = Camera()
    apply_view(v, cam, tags, style)
    assert tuple(float(x) for x in cam.position) == (9.0, 0.0, 0.0)
    assert tags.is_visible(2) is False
    assert style.face_style is FaceStyle.MONOCHROME
    assert style.xray is True


def test_apply_tolerates_unknown_tag_id():
    tags = TagLibrary()          # only Untagged exists
    style = RenderStyle()
    cam = Camera()
    v = capture_view(0, "V", cam, tags, style)
    # Inject a stale id that no longer exists in this library:
    from bermake.views.saved_view import SavedView
    stale = SavedView(v.id, v.name, v.camera, {999: False}, v.face_style,
                      v.xray)
    apply_view(stale, cam, tags, style)   # must not raise
    assert tags.is_visible(999) is True   # unknown id → treated visible (no-op)


def _capture_source_camera():
    cam = Camera()
    cam.position[:] = (9.0, 0.0, 0.0)
    return cam


def _hidden_roof_tags():
    lib = _tags()
    lib.set_visible(2, False)
    return lib


def test_capture_and_apply_carry_color_by_tag():
    # The final-review finding's second consequence: a Scene recalled via
    # apply_tags_and_style left color_by_tag at whatever it currently was, so
    # a Scene captured in the mode did not reproduce it -- and a Scene
    # captured OUT of it did not turn it back off. Both directions asserted.
    tags = _tags()

    on = capture_view(0, "On", Camera(), tags, RenderStyle(color_by_tag=True))
    assert on.color_by_tag is True
    live = RenderStyle(color_by_tag=False)
    apply_tags_and_style(on, tags, live)
    assert live.color_by_tag is True

    off = capture_view(1, "Off", Camera(), tags, RenderStyle(color_by_tag=False))
    assert off.color_by_tag is False
    apply_tags_and_style(off, tags, live)
    assert live.color_by_tag is False
