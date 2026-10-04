"""Icon loading for the UI shell (M7.2).

Assets are monochrome SVGs stored beside this module and addressed through
importlib.resources -- the same pattern the renderer already uses for shaders
(see scene_renderer._load_shader_source). They are recolored at load time from
the palette's text color, so one asset set serves light and dark themes.

A missing stem raises KeyError. Returning an empty QIcon instead would put a
blank button on a toolbar and pass every test.

This package's own `__init__.py` doubles as the loader module (rather than a
sibling `icons.py`) because Python's import system resolves a package
directory ahead of a same-named module file: a sibling `icons.py` next to
this `icons/` package would be permanently unreachable via `bermake.ui.icons`.
Keeping the loader here also lets it address its neighboring .svg assets
through `importlib.resources.files(ICON_DIR_PACKAGE)` without a second name.
"""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QToolButton

ICON_DIR_PACKAGE = "bermake.ui.icons"
DEFAULT_ICON_SIZE = 24

_icon_cache: dict[tuple[str, str, str], QIcon] = {}
_checked_ink_cache: dict[tuple, QColor] = {}

# Side of the throwaway button checked_ink renders. Big enough that the centre
# pixel is well inside the rounded checked background on every style measured.
_PROBE_BUTTON_SIZE = 32


def available_icon_stems() -> frozenset[str]:
    """Every .svg present in the icon package, without the extension."""
    root = files(ICON_DIR_PACKAGE)
    return frozenset(entry.name[:-4] for entry in root.iterdir() if entry.name.endswith(".svg"))


def _read_svg(stem: str) -> bytes:
    resource = files(ICON_DIR_PACKAGE) / f"{stem}.svg"
    if not resource.is_file():
        raise KeyError(f"no icon asset named {stem!r} in {ICON_DIR_PACKAGE}")
    return resource.read_bytes()


def icon_pixmap(stem: str, size: int, color: QColor) -> QPixmap:
    """Render one icon at `size` px, recolored to `color`.

    The SVG is rasterized as authored, then CompositionMode_SourceIn replaces
    every color channel while preserving alpha -- so anti-aliased edges and
    any fill-opacity in the source survive the recolor.
    """
    renderer = QSvgRenderer(_read_svg(stem))
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(pixmap.rect(), color)
    painter.end()
    return pixmap


def _relative_luminance(color: QColor) -> float:
    """WCAG 2.x relative luminance of an sRGB colour."""

    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * linear(color.redF())
        + 0.7152 * linear(color.greenF())
        + 0.0722 * linear(color.blueF())
    )


def contrast_ratio(a: QColor, b: QColor) -> float:
    """WCAG 2.x contrast ratio between two colours, 1.0 to 21.0."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def _sample_checked_background(palette: QPalette) -> QColor | None:
    """The colour the current style paints behind a checked auto-raise tool button.

    Rendered rather than read from the palette because no palette role holds it
    on every style (#128): under the native windows11 style the checked
    background is #1d6978 in Light but #71d4db in Dark, while palette.highlight()
    is #258292 there, and under Fusion it is highlight() itself. The button is
    never shown; QWidget.grab() renders a hidden widget through the style.
    Returns None when nothing could be rendered, so the caller can fall back.
    """
    if QApplication.instance() is None:
        return None
    button = QToolButton()
    button.setAutoRaise(True)
    button.setCheckable(True)
    button.setChecked(True)
    button.setPalette(palette)
    button.setFixedSize(_PROBE_BUTTON_SIZE, _PROBE_BUTTON_SIZE)
    image = button.grab().toImage()
    button.deleteLater()
    if image.isNull():
        return None
    centre = _PROBE_BUTTON_SIZE // 2
    sampled = image.pixelColor(centre, centre)
    if not sampled.isValid() or sampled.alpha() == 0:
        return None
    return QColor(sampled.rgb())


def checked_ink(palette: QPalette) -> QColor:
    """The glyph colour that reads best on a checked tool button's background.

    Of the palette's windowText and window colours, whichever contrasts more
    with the sampled checked background. Falls back to highlightedText when the
    background cannot be sampled. Cached per palette, style and colour scheme.
    """
    app = QApplication.instance()
    style_name = app.style().objectName() if app is not None else ""
    scheme = app.styleHints().colorScheme().name if app is not None else ""
    key = (palette.cacheKey(), style_name, scheme)
    cached = _checked_ink_cache.get(key)
    if cached is not None:
        return QColor(cached)

    background = _sample_checked_background(palette)
    if background is None:
        ink = palette.color(QPalette.ColorRole.HighlightedText)
    else:
        text = palette.color(QPalette.ColorRole.WindowText)
        surface = palette.color(QPalette.ColorRole.Window)
        ink = (
            text
            if contrast_ratio(text, background) >= contrast_ratio(surface, background)
            else surface
        )
    ink = QColor(ink.rgb())
    _checked_ink_cache[key] = ink
    return QColor(ink)


def icon(stem: str, color: QColor | None = None, checked_color: QColor | None = None) -> QIcon:
    """A QIcon for `stem`, recolored to `color` (default: mid grey).

    Callers that want palette-tracking icons pass
    widget.palette().windowText().color().

    The Off pixmap is `color`. The On pixmaps (Normal for a checked button at
    rest, Active for a checked button under the mouse) are `checked_color`,
    which defaults to checked_ink for the application palette, so a checked tool
    stays readable against its own highlight (#128). Qt picks the state itself,
    so no call site wires toggled signals.
    """
    resolved = color if color is not None else QColor("#3c3c3c")
    resolved_checked = (
        checked_color if checked_color is not None else checked_ink(QApplication.palette())
    )
    key = (stem, resolved.name(), resolved_checked.name())
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached

    built = QIcon()
    off = icon_pixmap(stem, DEFAULT_ICON_SIZE, resolved)
    on = icon_pixmap(stem, DEFAULT_ICON_SIZE, resolved_checked)
    built.addPixmap(off, QIcon.Mode.Normal, QIcon.State.Off)
    built.addPixmap(on, QIcon.Mode.Normal, QIcon.State.On)
    built.addPixmap(on, QIcon.Mode.Active, QIcon.State.On)
    _icon_cache[key] = built
    return built


def clear_icon_cache() -> None:
    """Drop every cached QIcon and checked ink. Used on a palette change and by tests."""
    _icon_cache.clear()
    _checked_ink_cache.clear()
