"""Per-tool cursors composited from the icon set (M7.2 Task 5)."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QPainter

from bermake.ui import actions, cursors


def test_crosshair_hotspot_is_the_crosshair_centre(qtbot):
    cursor = cursors.cursor_for("tool_line")
    assert (cursor.hotSpot().x(), cursor.hotSpot().y()) == cursors.CROSSHAIR_HOTSPOT


def test_arrow_hotspot_is_the_tip(qtbot):
    cursor = cursors.cursor_for("tool_select")
    assert (cursor.hotSpot().x(), cursor.hotSpot().y()) == cursors.ARROW_HOTSPOT


def test_every_tool_action_produces_a_cursor(qtbot):
    for spec in actions.ACTIONS:
        if spec.group != actions.TOOL_GROUP:
            continue
        cursor = cursors.cursor_for(spec.id)
        assert not cursor.pixmap().isNull(), spec.id


def test_cursor_pixmap_is_the_declared_size(qtbot):
    pixmap = cursors.cursor_for("tool_circle").pixmap()
    assert pixmap.width() == cursors.CURSOR_SIZE
    assert pixmap.height() == cursors.CURSOR_SIZE


def test_badge_ink_lands_in_the_lower_right_quadrant(qtbot):
    # The glyph must land in its own corner without disturbing the hotspot's
    # corner. Comparing the composed cursor straight against itself (e.g.
    # lower-right ink vs. upper-left ink) doesn't prove that: the crosshair's
    # own arms put more ink in the hotspot's quadrant than a thin badge ever
    # puts in its own, so that comparison can never distinguish a
    # well-placed badge from a missing or misplaced one. Instead compare
    # against a crosshair-only baseline in the *same* quadrants.
    half = cursors.CURSOR_SIZE // 2

    def ink(image, x0, y0):
        return sum(
            1
            for y in range(y0, y0 + half)
            for x in range(x0, x0 + half)
            if image.pixelColor(x, y).alpha() > 0
        )

    baseline_pixmap = cursors._blank(1.0)
    baseline_painter = QPainter(baseline_pixmap)
    baseline_painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    cursors._paint_crosshair(baseline_painter)
    baseline_painter.end()
    baseline_image = baseline_pixmap.toImage()

    composed_image = cursors.cursor_for("tool_rectangle").pixmap().toImage()

    hotspot_x, hotspot_y = cursors.CROSSHAIR_HOTSPOT
    hotspot_quadrant = (0 if hotspot_x < half else half, 0 if hotspot_y < half else half)
    badge_quadrant = (half, half)

    # The badge landed in the lower-right corner: that quadrant gained ink
    # relative to the crosshair-only baseline (which has none there).
    assert ink(composed_image, *badge_quadrant) > ink(baseline_image, *badge_quadrant)
    # The badge did not drift onto the hotspot: its quadrant is untouched.
    assert ink(composed_image, *hotspot_quadrant) == ink(baseline_image, *hotspot_quadrant)


def test_cursors_are_cached(qtbot):
    first = cursors.cursor_for("tool_move")
    assert first is cursors.cursor_for("tool_move")


def test_non_tool_action_has_no_cursor(qtbot):
    with pytest.raises(ValueError):
        cursors.cursor_for("file_save")


def test_cursor_for_composes_at_the_given_dpr(qtbot):
    # cursor_for is the only entry point MainWindow calls (M7.2 Task 12); if
    # it silently dropped a passed-in ratio, every cursor would ship at 1x
    # and blur on a HiDPI display regardless of what the caller asked for.
    cursor = cursors.cursor_for("tool_line", dpr=2.0)
    pixmap = cursor.pixmap()
    assert pixmap.devicePixelRatio() == pytest.approx(2.0)
    assert pixmap.width() == cursors.CURSOR_SIZE * 2
    assert (cursor.hotSpot().x(), cursor.hotSpot().y()) == cursors.CROSSHAIR_HOTSPOT


# --- dpr and cache-key correctness -------------------------------------
#
# cursor_for() defaults to dpr=1.0 when the caller doesn't pass one, so the
# tests above never exercise the dpr path by default. compose_cursor() is
# exercised directly here instead, since a HiDPI cursor with a mishandled
# ratio or an aliased cache entry would still pass every test above.


def test_compose_cursor_at_2x_dpr_scales_the_pixmap_but_keeps_the_hotspot(qtbot):
    cursor = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=2.0)
    pixmap = cursor.pixmap()

    assert pixmap.devicePixelRatio() == pytest.approx(2.0)
    # The physical buffer is twice the declared logical size...
    assert pixmap.width() == cursors.CURSOR_SIZE * 2
    assert pixmap.height() == cursors.CURSOR_SIZE * 2
    # ...but the hotspot stays in device-independent units, matching the
    # dpr=1.0 case, because Qt scales it against the pixmap's own
    # devicePixelRatio when it binds the platform cursor.
    assert (cursor.hotSpot().x(), cursor.hotSpot().y()) == cursors.CROSSHAIR_HOTSPOT


def test_compose_cursor_at_2x_dpr_still_centres_the_crosshair_ink(qtbot):
    # At dpr=1 the crosshair's own ink is centred on CROSSHAIR_HOTSPOT by
    # construction (see test_crosshair_hotspot_is_the_crosshair_centre); this
    # pins the same claim for the *rendered pixels* once dpr scaling is
    # applied, so a bug that scales the pixmap without scaling the paint
    # coordinates (or vice-versa) shows up as an off-centre cross.
    image = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=2.0).pixmap()
    cx = round(cursors.CROSSHAIR_HOTSPOT[0] * 2.0)
    cy = round(cursors.CROSSHAIR_HOTSPOT[1] * 2.0)
    img = image.toImage()
    assert img.pixelColor(cx, cy).alpha() > 0


def test_compose_cursor_cache_distinguishes_dpr(qtbot):
    cursors.clear_cursor_cache()
    at_1x = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=1.0)
    at_2x = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=2.0)

    assert at_1x is not at_2x
    # Re-requesting either one hits its own cache entry rather than the
    # other ratio's.
    assert cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=1.0) is at_1x
    assert cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line", dpr=2.0) is at_2x


def test_compose_cursor_cache_distinguishes_badge_stem(qtbot):
    cursors.clear_cursor_cache()
    line = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_line")
    rectangle = cursors.compose_cursor(actions.CursorStyle.CROSSHAIR, "tool_rectangle")
    assert line is not rectangle


def test_clear_cursor_cache_drops_previously_built_cursors(qtbot):
    before = cursors.cursor_for("tool_arc")
    cursors.clear_cursor_cache()
    after = cursors.cursor_for("tool_arc")
    assert before is not after
