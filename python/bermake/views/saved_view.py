"""SavedView (M7e): an immutable snapshot of a named Scene.

Holds the camera pose plus the tag-visibility and render-style state to
restore when the Scene is recalled. Pure data — no Qt. Named SavedView (not
Scene) to avoid colliding with bermake.scene.scene.Scene (the editable mesh).
"""

from __future__ import annotations

from dataclasses import dataclass

from bermake.io.document_codec import CameraState


@dataclass(frozen=True)
class SavedView:
    """One saved Scene: camera + tag visibility + render style, restored together."""

    id: int
    name: str
    camera: CameraState
    tag_visibility: dict  # dict[int, bool] — {tag_id: visible} at capture time
    face_style: str  # FaceStyle member name, e.g. "SHADED"
    xray: bool
    # M7.5a. Defaulted (and therefore last) so a record written before the
    # Color-by-Tag mode existed rebuilds without it, and so the many positional
    # SavedView(...) constructions elsewhere keep working. A Scene's whole job
    # is reproducing a view, so every RenderStyle field has to be captured here
    # or recalling the Scene silently leaves that one at whatever it happens to
    # be now.
    color_by_tag: bool = False
