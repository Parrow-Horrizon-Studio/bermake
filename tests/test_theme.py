"""Theme selection, and the icon refresh that #101 reports missing."""

import pytest
from bermake.ui import icons, preferences
from bermake.ui.theme import ThemeChoice, apply_theme, theme_for_name
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    return QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)


def test_light_is_the_default_theme(settings):
    """SketchUp is light on every OS regardless of system setting (spec D7)."""
    assert theme_for_name(preferences.read_theme(settings)) is ThemeChoice.LIGHT


def test_theme_for_name_maps_both_values():
    assert theme_for_name("light") is ThemeChoice.LIGHT
    assert theme_for_name("dark") is ThemeChoice.DARK


def test_theme_for_name_falls_back_rather_than_raising():
    """Review Focus 4.

    ThemeChoice("mauve") raises ValueError, and this is read during startup
    before the window exists, so it must not be able to stop the app opening.
    """
    assert theme_for_name("mauve") is ThemeChoice.LIGHT
    assert theme_for_name(None) is ThemeChoice.LIGHT
    assert theme_for_name("") is ThemeChoice.LIGHT


def _platform_tracks_colour_scheme(app) -> bool:
    """Whether this platform theme implements colour schemes at all.

    Measured rather than assumed, and probed through Qt directly rather than
    through apply_theme, so the probe cannot mask the thing the test checks.
    The offscreen plugin that conftest selects under CI accepts
    setColorScheme and does nothing at all with it: the readback stays
    Unknown, the palette does not move, and colorSchemeChanged never fires.
    """
    hints = app.styleHints()
    original = hints.colorScheme()
    try:
        hints.setColorScheme(Qt.ColorScheme.Dark)
        if hints.colorScheme() != Qt.ColorScheme.Dark:
            return False
        hints.setColorScheme(Qt.ColorScheme.Light)
        return hints.colorScheme() == Qt.ColorScheme.Light
    finally:
        hints.setColorScheme(original)


def test_apply_theme_maps_each_choice_to_its_qt_scheme():
    """The mapping, asserted on every platform including the headless one.

    This is the whole of apply_theme's own logic: a _SCHEMES lookup and one
    setter call. A swapped pair is therefore the only regression the function
    can carry, and it is the one that would ship, so it needs a test that
    does not depend on whether the platform theme honours the call. Verified
    against the mutation by swapping _SCHEMES and watching this fail.
    """

    class RecordingHints:
        def __init__(self) -> None:
            self.schemes: list[Qt.ColorScheme] = []

        def setColorScheme(self, scheme: Qt.ColorScheme) -> None:
            self.schemes.append(scheme)

    class FakeApp:
        def __init__(self) -> None:
            self.hints = RecordingHints()

        def styleHints(self) -> RecordingHints:
            return self.hints

    fake = FakeApp()
    apply_theme(fake, ThemeChoice.DARK)
    apply_theme(fake, ThemeChoice.LIGHT)
    assert fake.hints.schemes == [Qt.ColorScheme.Dark, Qt.ColorScheme.Light]


def test_apply_theme_sets_the_qt_colour_scheme(app):
    """The end-to-end half: Qt actually adopts what apply_theme sets.

    The apply_theme calls run unconditionally, so a PySide6 release that
    renames or drops setColorScheme fails here on every platform. Only the
    readback is conditional, because a platform theme without colour-scheme
    support leaves it at Unknown no matter what was set. Nothing in
    python/bermake reads colorScheme() back, so this is the only place that
    difference can bite.
    """
    observable = _platform_tracks_colour_scheme(app)

    apply_theme(app, ThemeChoice.DARK)
    if not observable:
        pytest.skip("platform theme does not implement colour schemes; readback is inert")
    assert app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    apply_theme(app, ThemeChoice.LIGHT)
    assert app.styleHints().colorScheme() == Qt.ColorScheme.Light


def test_the_icon_cache_is_keyed_on_colour_so_a_stale_entry_is_reachable(app):
    """The mechanism behind #101, stated as a test so the fix has a target.

    icon() caches on (stem, colour). Rebuilding from a palette Qt has not
    swapped yet re-caches the OLD colour, and because the entry is then present
    the next request returns it too, so the theme switch appears to do nothing
    until restart. That is exactly the symptom #101 reports.
    """
    from PySide6.QtGui import QColor

    icons.clear_icon_cache()
    light = icons.icon("file_new", QColor("#3c3c3c"))
    dark = icons.icon("file_new", QColor("#f0f0f0"))
    assert light is not dark
    assert icons.icon("file_new", QColor("#3c3c3c")) is light


def test_a_palette_change_re_tints_the_properties_tab_strip(app):
    """Review Focus 5, and the half of the fix that gets forgotten.

    The tab strip is built once in PropertiesDock.__init__ and never rebuilt, so
    nothing would re-request its icons and every tab would keep the previous
    theme's ink. That is a partial fix that looks complete from the toolbar.

    Asserted at the CURRENT palette colour rather than merely "is in the cache":
    the cache is keyed on (stem, colour), so a version that warmed the cache at
    the old colour would satisfy a presence check and still show stale icons.

    Discriminates: drop the _properties_dock.refresh_icons call from
    _rebuild_all_icons and this fails.

    Review finding: membership in the cache is satisfied by a loop that merely
    warms the cache at the current colour, touching no widget at all. The
    assertion that actually pins the fix is on the tab button's own icon: its
    cacheKey() must change across a real palette change, verified independently
    by measuring 30064771072 -> 356482285568 on a real theme switch.
    """
    from bermake.ui.main_window import MainWindow
    from bermake.ui.panel_icons import TAB_ICONS
    from PySide6.QtGui import QColor, QPalette

    window = MainWindow()
    icons.clear_icon_cache()
    window._rebuild_all_icons()
    color_name = window.palette().windowText().color().name()
    cached_at_current = {stem for stem, color, _checked in icons._icon_cache if color == color_name}
    missing = TAB_ICONS - cached_at_current
    assert not missing, f"tab icons not re-tinted at the new colour: {sorted(missing)}"

    tab_button = window._properties_dock.tab_button(window._properties_dock.current_tab_id)
    before_key = tab_button.icon().cacheKey()

    palette = window.palette()
    current = palette.color(QPalette.ColorRole.WindowText)
    new_color = QColor("#204060") if current.name() != "#204060" else QColor("#602040")
    palette.setColor(QPalette.ColorRole.WindowText, new_color)
    window.setPalette(palette)
    window._rebuild_all_icons()

    after_key = tab_button.icon().cacheKey()
    assert after_key != before_key, "the tab button's own icon did not change"


def test_a_palette_change_rebuilds_the_outliner(app, monkeypatch):
    """The Outliner's other half of the same problem, by a different mechanism.

    It sets each row's icon inside set_rows, reading the palette at that moment,
    so it needs a rebuild rather than a re-tint method. An empty model has no
    rows and therefore caches no row icons, so this pins the rebuild call itself
    instead of asserting over the cache.

    Discriminates: drop the _rebuild_outliner call and this fails.
    """
    from bermake.ui.main_window import MainWindow

    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "_rebuild_outliner", lambda: calls.append(1))
    window._rebuild_all_icons()
    assert calls == [1]


def test_a_palette_change_re_tints_the_action_icons(app):
    """The registry half, so the tests above cannot pass alone.

    Review finding: as with the tab strip above, cache membership is satisfied
    by warming the cache at the current colour without touching a single
    action. The widget assertion below is the one that actually pins the fix.
    """
    from bermake.ui import actions
    from bermake.ui.main_window import MainWindow
    from PySide6.QtGui import QColor, QPalette

    window = MainWindow()
    icons.clear_icon_cache()
    window._rebuild_all_icons()
    color_name = window.palette().windowText().color().name()
    cached_at_current = {stem for stem, color, _checked in icons._icon_cache if color == color_name}
    expected = {spec.icon for spec in actions.ACTIONS if spec.icon is not None}
    assert expected <= cached_at_current

    action = window._actions["file_new"]
    before_key = action.icon().cacheKey()

    palette = window.palette()
    current = palette.color(QPalette.ColorRole.WindowText)
    new_color = QColor("#204060") if current.name() != "#204060" else QColor("#602040")
    palette.setColor(QPalette.ColorRole.WindowText, new_color)
    window.setPalette(palette)
    window._rebuild_all_icons()

    after_key = action.icon().cacheKey()
    assert after_key != before_key, "the action's own icon did not change"
