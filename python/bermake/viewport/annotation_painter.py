"""Execute annotation draw plans (M7d).

Deliberately thin: ALL layout lives in the pure annotations.draw_plan module.
This module only turns a plan into painter calls, so it works against any object
exposing setPen / drawLine / drawText (a QPainter in the app, a recording stub in
tests).
"""

from __future__ import annotations

from bermake.annotations.draw_plan import CHAR_W_PX

# Guides and guide points (Task 6) are construction geometry, not measured
# annotations, so they draw with a distinct pen regardless of selection/hover
# state; dimensions and labels keep the colour rules they already had.
_GUIDE_KINDS = frozenset({"guide", "guide_point"})


def _align_offset(text: str, align: str) -> float:
    if align == "center":
        return -0.5 * CHAR_W_PX * max(len(text), 1)
    if align == "right":
        return -CHAR_W_PX * max(len(text), 1)
    return 0.0


def paint_annotation_plans(
    painter,
    plans,
    color,
    selected_ids,
    selected_color,
    hovered_id=None,
    hover_color=None,
    guide_pen=None,
) -> None:
    """Draw every plan; annotations whose id is in `selected_ids` use the
    selection colour. Task 10b: the single annotation matching `hovered_id`
    (if any) uses `hover_color` instead -- unless it is also selected, in
    which case selection wins (selected beats hover, as it does for
    edges/faces/instances).

    Task 6: when the caller supplies `guide_pen` (a dashed grey pen in the
    real app), every plan whose `kind` is "guide" or "guide_point" is painted
    with it instead, ahead of the selection/hover checks -- construction
    guides are not measured geometry and keep their own look regardless of
    selection state. Omitting `guide_pen` (the default) leaves every plan,
    guides included, on the ordinary colour rules, matching the pre-Task-6
    behaviour exactly."""
    for plan in plans:
        if guide_pen is not None and plan.kind in _GUIDE_KINDS:
            pen_color = guide_pen
        elif plan.annotation_id in selected_ids:
            pen_color = selected_color
        elif hovered_id is not None and plan.annotation_id == hovered_id:
            pen_color = hover_color
        else:
            pen_color = color
        painter.setPen(pen_color)
        for x1, y1, x2, y2 in plan.segments_px:
            painter.drawLine(x1, y1, x2, y2)
        for text in plan.texts:
            painter.drawText(text.x + _align_offset(text.text, text.align), text.y, text.text)
