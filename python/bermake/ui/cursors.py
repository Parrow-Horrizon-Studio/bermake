"""Per-tool cursors (M7.2).

Cursors are *derived from* the icon set rather than drawn as separate art:
the tool's own glyph is composited into the lower-right of a shared base, so
redrawing an icon updates its cursor automatically and the two cannot drift.

Two bases, matching the semantics of the tools:
  CROSSHAIR - the tool places a point; the hotspot is the crosshair centre.
  ARROW     - the tool picks an existing entity, where a crosshair would
              imply an accuracy that is not being used.

The badge glyph is haloed in white before its black ink is painted on top --
the same technique the crosshair and arrow bases already use, and for the
same reason: Bermake's viewport clears to a near-black
`(0.15, 0.15, 0.18)`, so a bare black glyph would disappear into it. The
halo is built from the glyph's own strokes offset by one device-independent
pixel in each of the eight directions, not a filled backdrop panel -- our
icon set is thin single-stroke outlines, not filled shapes, and a panel was
tried and rejected in favour of this lighter-weight look.

`compose_cursor` takes a `dpr` (a QScreen.devicePixelRatio()) so the cursor
stays crisp on a HiDPI display: the pixmap is rendered at `dpr` physical
pixels per logical pixel and carries that ratio as
`QPixmap.devicePixelRatio()`. Painting is done in the fixed, device-
independent CURSOR_SIZE/BADGE_SIZE coordinate space -- Qt's paint engine
scales every drawing operation against the target pixmap's devicePixelRatio,
so the same paint code produces a sharp result at any ratio. The hotspot
constants are likewise device-independent and never scaled by `dpr` in this
module: Qt re-scales them against the pixmap's own devicePixelRatio when it
binds the platform cursor, exactly as it does for the pixmap content.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QColor, QCursor, QPainter, QPen, QPixmap, QPolygonF

from bermake.ui.actions import CursorStyle, action_by_id
from bermake.ui.icons import icon_pixmap

CURSOR_SIZE = 32
BADGE_SIZE = 16
_BADGE_ORIGIN = (CURSOR_SIZE - BADGE_SIZE, CURSOR_SIZE - BADGE_SIZE)
CROSSHAIR_HOTSPOT = (11, 11)
ARROW_HOTSPOT = (0, 0)
_CROSSHAIR_ARM = 8
_INK = QColor("#101010")
_OUTLINE = QColor("#ffffff")

# Keyed on everything that changes the rendered pixels -- style, badge stem,
# and dpr -- so two different cursors can never alias to the same cache slot.
_cursor_cache: dict[tuple[CursorStyle, str, float], QCursor] = {}


def _blank(dpr: float) -> QPixmap:
    physical = round(CURSOR_SIZE * dpr)
    pixmap = QPixmap(QSize(physical, physical))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    return pixmap


def _paint_crosshair(painter: QPainter) -> None:
    cx, cy = CROSSHAIR_HOTSPOT
    # White underlay first so the crosshair stays visible over dark geometry.
    for color, width in ((_OUTLINE, 3.0), (_INK, 1.0)):
        painter.setPen(QPen(color, width))
        painter.drawLine(cx - _CROSSHAIR_ARM, cy, cx + _CROSSHAIR_ARM, cy)
        painter.drawLine(cx, cy - _CROSSHAIR_ARM, cx, cy + _CROSSHAIR_ARM)


_HALO_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))


def _paint_badge(painter: QPainter, stem: str, dpr: float) -> None:
    """Composite `stem`'s glyph into the lower-right corner, haloed in white.

    Same technique as `_paint_crosshair`'s white underlay: the glyph is
    painted once per offset in `_HALO_OFFSETS` in white, then once more on
    top in black, so the badge reads clearly over arbitrary (often dark)
    viewport geometry without a filled backdrop panel.

    The offsets are device-independent pixels, matching the coordinate space
    `painter` is already operating in; the glyph pixmaps themselves are
    rendered at `dpr` physical pixels and tagged with `setDevicePixelRatio`
    so both the halo and the ink stay crisp at any ratio.
    """
    size = round(BADGE_SIZE * dpr)
    white = icon_pixmap(stem, size, _OUTLINE)
    white.setDevicePixelRatio(dpr)
    for dx, dy in _HALO_OFFSETS:
        painter.drawPixmap(_BADGE_ORIGIN[0] + dx, _BADGE_ORIGIN[1] + dy, white)

    ink = icon_pixmap(stem, size, _INK)
    ink.setDevicePixelRatio(dpr)
    painter.drawPixmap(*_BADGE_ORIGIN, ink)


def _paint_arrow(painter: QPainter) -> None:
    points = [
        (0.0, 0.0),
        (0.0, 15.0),
        (4.2, 11.2),
        (7.0, 17.0),
        (9.4, 15.8),
        (6.7, 10.2),
        (12.0, 9.8),
    ]
    polygon = QPolygonF([QPointF(x, y) for x, y in points])
    painter.setPen(QPen(_OUTLINE, 2.0))
    painter.setBrush(_INK)
    painter.drawPolygon(polygon)


def compose_cursor(style: CursorStyle, stem: str, dpr: float = 1.0) -> QCursor:
    """Paint `stem`'s glyph into the lower-right of `style`'s base pixmap.

    `dpr` selects the physical resolution the cursor is rendered at; see the
    module docstring for how the pixmap and the hotspot stay in sync.
    """
    key = (style, stem, dpr)
    cached = _cursor_cache.get(key)
    if cached is not None:
        return cached

    pixmap = _blank(dpr)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    if style is CursorStyle.CROSSHAIR:
        _paint_crosshair(painter)
        hotspot = CROSSHAIR_HOTSPOT
    else:
        _paint_arrow(painter)
        hotspot = ARROW_HOTSPOT

    _paint_badge(painter, stem, dpr)
    painter.end()

    built = QCursor(pixmap, hotspot[0], hotspot[1])
    _cursor_cache[key] = built
    return built


def cursor_for(action_id: str, dpr: float = 1.0) -> QCursor:
    """The cursor for a tool action. Raises ValueError for a non-tool action.

    `dpr` is forwarded to `compose_cursor` unchanged; pass the widget's own
    `devicePixelRatioF()` so the cursor stays crisp on a HiDPI display. It
    defaults to 1.0 for callers (and tests) that don't care.
    """
    spec = action_by_id(action_id)
    if spec.cursor is None or spec.icon is None:
        raise ValueError(f"{action_id} declares no cursor style")
    return compose_cursor(spec.cursor, spec.icon, dpr)


def clear_cursor_cache() -> None:
    """Drop every cached QCursor. Used by tests."""
    _cursor_cache.clear()
