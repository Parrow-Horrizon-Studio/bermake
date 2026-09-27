"""Screen-space annotation picking (M7d).

Hit-tests the SAME draw plan the painter renders, so anything visible is
clickable and nothing invisible is.
"""

from __future__ import annotations

from bermake.annotations.draw_plan import plan_annotation

# M7.6b Task 7 fix round 1: kinds View > Guides hides. A hidden guide must be
# unpickable too, or the module's own stated contract above ("nothing
# invisible is [clickable]") is broken -- Select/Erase/the context menu would
# still hover, select and erase something the user cannot see.
_GUIDE_KINDS = frozenset({"guide", "guide_point"})


def _inside(box, x, y):
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _box_distance_sq(box, x, y):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return (cx - x) ** 2 + (cy - y) ** 2


def pick_annotation(
    cursor_px, annotations, world_transform, camera, width, height, units, show_guides=True
):
    """Return the id of the nearest annotation under the cursor, or None.

    `show_guides=False` (View > Guides off) excludes guide/guide_point
    annotations from picking -- filtered here, at the single shared call
    site every picking path (Select, Erase, the right-click context menu)
    goes through, rather than in each caller, and as a plain filter over the
    iterable rather than any mutation of `annotations` itself: hiding a
    guide must never change any other annotation's id.
    """
    x, y = float(cursor_px[0]), float(cursor_px[1])
    best_id = None
    best_d2 = float("inf")
    for ann in annotations:
        if not show_guides and getattr(ann, "kind", None) in _GUIDE_KINDS:
            continue
        plan = plan_annotation(ann, world_transform, camera, width, height, units)
        if plan is None:
            continue
        for box in plan.hit_boxes:
            if not _inside(box, x, y):
                continue
            d2 = _box_distance_sq(box, x, y)
            if d2 < best_d2:
                best_d2 = d2
                best_id = plan.annotation_id
    return best_id
