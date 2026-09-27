"""Capture / apply helpers (M7e): snapshot live view state into a SavedView
and restore it. Pure — no Qt. Tag-visibility restore is tolerant of ids that
no longer exist (TagLibrary.set_visible is a no-op for unknown ids)."""

from __future__ import annotations

from bermake.io.document_codec import CameraState
from bermake.viewport.render_style import FaceStyle
from bermake.views.saved_view import SavedView


def capture_view(view_id, name, camera, tag_library, render_style) -> SavedView:
    """Snapshot the current camera, tag visibility and render style as a
    SavedView."""
    tag_visibility = {
        t.id: bool(t.visible) for t in tag_library.tags() if t.id != tag_library.UNTAGGED_ID
    }
    return SavedView(
        id=int(view_id),
        name=str(name),
        camera=CameraState.from_camera(camera),
        tag_visibility=tag_visibility,
        face_style=render_style.face_style.name,
        xray=bool(render_style.xray),
        color_by_tag=bool(render_style.color_by_tag),
    )


def apply_tags_and_style(view, tag_library, render_style) -> None:
    """Restore tag visibility + render style from `view` (leaves the camera
    alone)."""
    for tid, visible in view.tag_visibility.items():
        tag_library.set_visible(int(tid), bool(visible))
    render_style.face_style = FaceStyle[view.face_style]
    render_style.xray = bool(view.xray)
    # Every RenderStyle field the SavedView carries is restored, not just the
    # two the Scene shipped with: leaving color_by_tag alone would let a Scene
    # recall reproduce the camera and the tags but not the display mode.
    render_style.color_by_tag = bool(view.color_by_tag)


def apply_view(view, camera, tag_library, render_style) -> None:
    """Restore all three: tags + style (instant) and the camera pose
    (direct)."""
    apply_tags_and_style(view, tag_library, render_style)
    view.camera.apply_to(camera)
