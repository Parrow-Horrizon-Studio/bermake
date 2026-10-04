"""Readable checked toolbar icons (#128).

A checked tool's button is painted in a highlight colour that no palette role
holds on every style, so the glyph on it has to be chosen by sampling the
style. These tests pin the QIcon state mapping, the ink choice, the rebuild on
a runtime Light/Dark switch (Review Focus 4), and the real contrast.
"""

import pytest
from bermake.ui import icons
from bermake.ui.theme import ThemeChoice, apply_theme
from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication, QToolButton

# A pixel well inside the arrow glyph of tool_select (24 px render).
GLYPH_PIXEL = (9, 10)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _fresh_icon_caches(app):
    icons.clear_icon_cache()
    yield
    icons.clear_icon_cache()


@pytest.fixture
def restore_app_look(app):
    """Put the colour scheme and palette back after a test changes them."""
    palette = QApplication.palette()
    scheme = app.styleHints().colorScheme()
    yield
    app.styleHints().setColorScheme(scheme)
    QApplication.setPalette(palette)


def _pixel(icon: QIcon, state: QIcon.State, mode=QIcon.Mode.Normal) -> QColor:
    image = icon.pixmap(QSize(24, 24), mode, state).toImage()
    return image.pixelColor(*GLYPH_PIXEL)


def test_checked_and_unchecked_request_different_pixmaps(app):
    built = icons.icon("tool_select", QColor("black"), QColor("white"))
    assert _pixel(built, QIcon.State.Off).name() == "#000000"
    assert _pixel(built, QIcon.State.On).name() == "#ffffff"
    # A checked button under the mouse asks for (Active, On).
    assert _pixel(built, QIcon.State.On, QIcon.Mode.Active).name() == "#ffffff"
    # A hovered unchecked button asks for (Active, Off): the ordinary ink.
    assert _pixel(built, QIcon.State.Off, QIcon.Mode.Active).name() == "#000000"


def test_the_cache_distinguishes_checked_colours(app):
    one = icons.icon("tool_select", QColor("black"), QColor("white"))
    two = icons.icon("tool_select", QColor("black"), QColor("red"))
    assert one is not two
    assert icons.icon("tool_select", QColor("black"), QColor("white")) is one
    assert _pixel(two, QIcon.State.On).name() == "#ff0000"


def test_checked_ink_picks_the_candidate_that_contrasts_more(app, monkeypatch):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#101010"))
    palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ff00ff"))

    monkeypatch.setattr(icons, "_sample_checked_background", lambda _palette: QColor("#1d6978"))
    assert icons.checked_ink(palette).name() == "#f0f0f0"

    icons.clear_icon_cache()
    monkeypatch.setattr(icons, "_sample_checked_background", lambda _palette: QColor("#71d4db"))
    assert icons.checked_ink(palette).name() == "#101010"


def test_checked_ink_falls_back_to_highlighted_text_when_sampling_fails(app, monkeypatch):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ff00ff"))
    monkeypatch.setattr(icons, "_sample_checked_background", lambda _palette: None)
    assert icons.checked_ink(palette).name() == "#ff00ff"


def test_checked_ink_is_cached_per_palette(app, monkeypatch):
    calls = []

    def sample(_palette):
        calls.append(1)
        return QColor("#1d6978")

    monkeypatch.setattr(icons, "_sample_checked_background", sample)
    palette = QPalette()
    icons.checked_ink(palette)
    icons.checked_ink(palette)
    assert calls == [1]

    other = QPalette()
    other.setColor(QPalette.ColorRole.Window, QColor("#123456"))
    icons.checked_ink(other)
    assert calls == [1, 1]


def _dark_palette() -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1e1e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#f0e68c"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    return palette


def test_a_palette_change_rebuilds_the_checked_ink(app, restore_app_look):
    """Review Focus 4: Light to Dark at runtime must not leave the old ink.

    The Dark palette goes in through apply_theme, which moves the palette on a
    platform that implements colour schemes. The offscreen platform ignores
    the scheme, so a hand-built dark palette is applied as well; the ink must
    follow whichever palette is live.
    """
    from bermake.ui.main_window import MainWindow

    apply_theme(app, ThemeChoice.LIGHT)
    window = MainWindow()
    action = window._actions["tool_select"]
    old_ink = icons.checked_ink(QApplication.palette())
    assert _pixel(action.icon(), QIcon.State.On).name() == old_ink.name()

    apply_theme(app, ThemeChoice.DARK)
    if icons.checked_ink(QApplication.palette()).name() == old_ink.name():
        QApplication.setPalette(_dark_palette())
    new_ink = icons.checked_ink(QApplication.palette())
    assert new_ink.name() != old_ink.name(), "test needs a palette that changes the ink"

    # What MainWindow runs on PaletteChange.
    window._rebuild_all_icons()

    on = _pixel(action.icon(), QIcon.State.On)
    assert on.name() == new_ink.name()
    assert on.name() != old_ink.name()


def _grab_checked_glyph(theme: ThemeChoice) -> tuple[QColor, QColor]:
    """(glyph colour, background colour) of a really rendered checked button."""
    app = QApplication.instance()
    apply_theme(app, theme)
    icons.clear_icon_cache()
    button = QToolButton()
    button.setAutoRaise(True)
    button.setCheckable(True)
    button.setChecked(True)
    button.setIconSize(QSize(24, 24))
    button.setFixedSize(32, 32)
    button.setIcon(icons.icon("tool_select", button.palette().windowText().color()))
    image = button.grab().toImage()
    button.deleteLater()
    background = image.pixelColor(28, 16)
    glyph = image.pixelColor(GLYPH_PIXEL[0] + 4, GLYPH_PIXEL[1] + 4)
    return glyph, background


@pytest.mark.parametrize("theme", [ThemeChoice.LIGHT, ThemeChoice.DARK])
def test_the_checked_glyph_reaches_wcag_contrast_on_the_real_style(app, restore_app_look, theme):
    if app.platformName() != "windows":
        pytest.skip("needs the windows platform; offscreen ignores the colour scheme")
    glyph, background = _grab_checked_glyph(theme)
    assert icons.contrast_ratio(glyph, background) >= 4.5, (
        f"{theme.value}: glyph {glyph.name()} on {background.name()}"
    )
