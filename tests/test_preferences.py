"""Per-user preferences, against an injected ini store so no real state is touched."""

import pytest
from bermake.templates import DEFAULT_TEMPLATE_KEY
from bermake.ui import preferences
from PySide6.QtCore import QSettings


@pytest.fixture
def settings(tmp_path):
    """An ini-backed store, the discipline window_state.py already uses.

    A default-constructed QSettings writes the real registry on Windows.
    """
    return QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)


def test_theme_defaults_to_light(settings):
    assert preferences.read_theme(settings) == "light"


def test_theme_round_trips(settings):
    preferences.write_theme(settings, "dark")
    assert preferences.read_theme(settings) == "dark"


def test_an_unrecognised_theme_value_reads_as_light(settings):
    """Review Focus 4.

    app.py reads this before the window exists, so a corrupt value that reached
    ThemeChoice() as-is would be a startup crash rather than a cosmetic problem.
    """
    settings.setValue(preferences.THEME_KEY, "mauve")
    assert preferences.read_theme(settings) == "light"


def test_show_welcome_defaults_to_true(settings):
    assert preferences.read_show_welcome(settings) is True


@pytest.mark.parametrize("stored,expected", [("false", False), ("true", True), (False, False)])
def test_show_welcome_reads_qsettings_string_booleans(settings, stored, expected):
    """QSettings ini round-trips booleans as the strings "true" and "false".

    Reading one with bool() would make "false" truthy, so the checkbox would be
    impossible to turn off across a restart.
    """
    settings.setValue(preferences.SHOW_WELCOME_KEY, stored)
    assert preferences.read_show_welcome(settings) is expected


def test_show_welcome_round_trips_false(settings):
    preferences.write_show_welcome(settings, False)
    assert preferences.read_show_welcome(settings) is False


def test_default_template_defaults_to_the_default_key(settings):
    assert preferences.read_default_template(settings) == DEFAULT_TEMPLATE_KEY


def test_default_template_round_trips(settings):
    preferences.write_default_template(settings, "documentation")
    assert preferences.read_default_template(settings) == "documentation"


def test_a_garbage_default_template_value_still_returns_a_string(settings):
    """Validation belongs to template_for_key, which falls back. This just must
    not raise, so a corrupt store cannot stop the application from starting."""
    settings.setValue(preferences.DEFAULT_TEMPLATE_PREF_KEY, 12345)
    assert isinstance(preferences.read_default_template(settings), str)


def test_compatibility_rendering_defaults_to_off(settings):
    assert preferences.read_compatibility_rendering(settings) is False


@pytest.mark.parametrize("enabled", [True, False])
def test_compatibility_rendering_round_trips(settings, enabled):
    preferences.write_compatibility_rendering(settings, enabled)
    settings.sync()
    assert preferences.read_compatibility_rendering(settings) is enabled


@pytest.mark.parametrize(("stored", "expected"), [("true", True), ("false", False), ("0", False)])
def test_compatibility_rendering_reads_qsettings_string_booleans(settings, stored, expected):
    settings.setValue(preferences.COMPATIBILITY_RENDERING_KEY, stored)
    assert preferences.read_compatibility_rendering(settings) is expected


def test_autosave_interval_defaults_to_five_minutes(settings):
    assert preferences.AUTOSAVE_INTERVAL_KEY == "autosave/interval_minutes"
    assert preferences.read_autosave_interval(settings) == 5


@pytest.mark.parametrize("minutes", [0, 1, 5, 10, 30])
def test_autosave_interval_round_trips_every_choice(settings, minutes):
    preferences.write_autosave_interval(settings, minutes)
    settings.sync()
    assert preferences.read_autosave_interval(settings) == minutes


@pytest.mark.parametrize("minutes", [0, 1, 5, 10, 30])
def test_autosave_interval_reads_the_ini_string_from_a_previous_run(tmp_path, minutes):
    """Written by hand, as a previous run leaves it: Qt shares one in-process
    cache per file, so only a file this process never opened comes back as the
    ini backend's string rather than the int that was set."""
    path = tmp_path / "previous-run.ini"
    path.write_text(f"[autosave]\ninterval_minutes={minutes}\n", encoding="utf-8")
    reader = QSettings(str(path), QSettings.Format.IniFormat)
    assert reader.value(preferences.AUTOSAVE_INTERVAL_KEY) == str(minutes)
    assert preferences.read_autosave_interval(reader) == minutes


@pytest.mark.parametrize("stored", ["7", "abc", "", "-5", "5.5", 2, 60, None])
def test_an_unknown_autosave_interval_reads_as_five(settings, stored):
    """Spec 4.4: unknown values read as 5. A corrupt store must not stop startup."""
    settings.setValue(preferences.AUTOSAVE_INTERVAL_KEY, stored)
    assert preferences.read_autosave_interval(settings) == 5
