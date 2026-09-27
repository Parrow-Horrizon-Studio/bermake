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
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICON_DIR_PACKAGE = "bermake.ui.icons"
DEFAULT_ICON_SIZE = 24

_icon_cache: dict[tuple[str, str], QIcon] = {}


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


def icon(stem: str, color: QColor | None = None) -> QIcon:
    """A QIcon for `stem`, recolored to `color` (default: mid grey).

    Callers that want palette-tracking icons pass
    widget.palette().windowText().color().
    """
    resolved = color if color is not None else QColor("#3c3c3c")
    key = (stem, resolved.name())
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached

    built = QIcon(icon_pixmap(stem, DEFAULT_ICON_SIZE, resolved))
    _icon_cache[key] = built
    return built


def clear_icon_cache() -> None:
    """Drop every cached QIcon. Used when the palette changes and by tests."""
    _icon_cache.clear()
