"""Per-user application preferences (M7.7).

The second non-document state in the application, after window_state.py, and
deliberately a separate module from it: that one owns geometry and toolbar
layout and carries its own WINDOW_STATE_VERSION which Qt validates and rejects
on mismatch. A theme choice has no business bumping that version.

Every function takes the QSettings object as a parameter rather than
constructing one, for the reason window_state.py gives: on Windows a
default-constructed QSettings writes the real registry, so tests inject a
temporary IniFormat store and the suite touches no real user state.

No read raises. A corrupt or absent value returns the documented default,
because these are read during startup and a bad settings store must not be able
to stop the application from opening.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings

from bermake.templates import DEFAULT_TEMPLATE_KEY

THEME_KEY = "appearance/theme"
SHOW_WELCOME_KEY = "welcome/show_on_startup"
DEFAULT_TEMPLATE_PREF_KEY = "welcome/default_template"
COMPATIBILITY_RENDERING_KEY = "graphics/compatibility_rendering"

THEME_LIGHT = "light"
THEME_DARK = "dark"
_THEMES = (THEME_LIGHT, THEME_DARK)


def read_theme(settings: QSettings) -> str:
    """The saved theme name, defaulting to light.

    Validated here rather than at the call site: app.py reads this before the
    main window exists, so an unrecognised value that reached ThemeChoice()
    unchecked would raise ValueError and take startup down.
    """
    value = settings.value(THEME_KEY, THEME_LIGHT)
    text = str(value) if value is not None else THEME_LIGHT
    return text if text in _THEMES else THEME_LIGHT


def write_theme(settings: QSettings, theme: str) -> None:
    settings.setValue(THEME_KEY, theme)


def _read_bool(settings: QSettings, key: str, default: bool) -> bool:
    """A stored boolean, tolerant of the ini backend's strings.

    The string comparison is not incidental. QSettings' ini backend round-trips
    booleans as the strings "true" and "false", and bool("false") is True, so
    reading with bool() would make a checkbox impossible to turn off across a
    restart.
    """
    value = settings.value(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "")


def read_show_welcome(settings: QSettings) -> bool:
    """Whether to show the welcome dialog at startup. Defaults to True."""
    return _read_bool(settings, SHOW_WELCOME_KEY, True)


def write_show_welcome(settings: QSettings, show: bool) -> None:
    settings.setValue(SHOW_WELCOME_KEY, bool(show))


def read_default_template(settings: QSettings) -> str:
    """The template key File > New applies. Always a string.

    Whether the key names a real template is templates.template_for_key's
    business, which falls back; this only guarantees the caller gets a str.
    """
    value = settings.value(DEFAULT_TEMPLATE_PREF_KEY, DEFAULT_TEMPLATE_KEY)
    return str(value) if value is not None else DEFAULT_TEMPLATE_KEY


def write_default_template(settings: QSettings, key: str) -> None:
    settings.setValue(DEFAULT_TEMPLATE_PREF_KEY, key)


def read_compatibility_rendering(settings: QSettings) -> bool:
    """Whether to render through the bundled Mesa (M7.9). Defaults to False.

    Read by app.py before QApplication exists, because Qt only honours the
    switch when it is made that early.
    """
    return _read_bool(settings, COMPATIBILITY_RENDERING_KEY, False)


def write_compatibility_rendering(settings: QSettings, enabled: bool) -> None:
    settings.setValue(COMPATIBILITY_RENDERING_KEY, bool(enabled))
