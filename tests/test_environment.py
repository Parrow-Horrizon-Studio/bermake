"""Environment presets: the pre-M7.7 reproduction and the contrast floor."""

import pytest
from bermake.viewport.environment import (
    DEFAULT_ENVIRONMENT,
    LANDSCAPE_KEY,
    LEGACY_ENVIRONMENT,
    NEUTRAL_GREY_KEY,
    PLAIN_WHITE,
    PLAIN_WHITE_KEY,
    PRESETS,
    SKY_AND_GROUND,
    SKY_AND_GROUND_KEY,
    STUDIO,
    STUDIO_KEY,
    Environment,
    environment_pass_needed,
    preset_key,
)

# Floors from the spec, section 4. Edge ink must be obvious; grid ink is meant
# to be subtle, so it clears a lower bar.
EDGE_FLOOR = 0.35
GRID_FLOOR = 0.12


def _luminance(color):
    r, g, b = color
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _backdrops(env):
    """Every colour a user can actually see behind the ink in this preset.

    `background` alone is not enough: with sky and ground on it is covered, so
    checking only against it would pass a preset whose edges vanish into the sky.
    """
    seen = [env.background]
    if env.sky_enabled:
        seen.append(env.sky_color)
    if env.ground_enabled:
        seen.append(env.ground_color)
    return seen


def test_studio_preset_matches_the_pre_m77_renderer_constants():
    """STUDIO must reproduce the old look byte for byte.

    These four literals are what scene_renderer.py hardcoded before M7.7:
    _BG_COLOR[:3], _USER_EDGE_COLOR, _GRID_COLOR, _GRID_CENTERLINE_COLOR. This
    is the regression gate on the whole ink refactor: if it passes, a user who
    keeps the dark environment sees exactly what they saw in v0.12.0.
    """
    assert STUDIO.background == (0.15, 0.15, 0.18)
    assert STUDIO.edge_color == (0.85, 0.85, 0.85)
    assert STUDIO.grid_color == (0.40, 0.40, 0.40)
    assert STUDIO.grid_centerline_color == (0.60, 0.60, 0.60)


@pytest.mark.parametrize("key", sorted(PRESETS))
def test_every_preset_ink_clears_the_contrast_floor(key):
    """No preset may ship ink that disappears into its own backdrop.

    Discriminates: set PLAIN_WHITE.edge_color to STUDIO's (0.85, 0.85, 0.85)
    and this fails at 0.150 against a white background, well under 0.35.
    """
    env = PRESETS[key]
    backdrops = [_luminance(c) for c in _backdrops(env)]
    for name, ink, floor in (
        ("edge_color", env.edge_color, EDGE_FLOOR),
        ("grid_color", env.grid_color, GRID_FLOOR),
        ("grid_centerline_color", env.grid_centerline_color, GRID_FLOOR),
    ):
        worst = min(abs(_luminance(ink) - b) for b in backdrops)
        assert worst >= floor, f"{key}.{name} is only {worst:.3f} from its nearest backdrop"


def test_environment_is_frozen():
    with pytest.raises(Exception):
        STUDIO.background = (0.0, 0.0, 0.0)


@pytest.mark.parametrize("key", sorted(PRESETS))
def test_preset_key_round_trips_every_preset(key):
    assert preset_key(PRESETS[key]) == key


def test_preset_key_is_none_for_a_value_no_preset_matches():
    """A document from a future build with colour picking (spec D9) is legal.

    preset_key returns None rather than raising so the View menu can leave every
    environment entry unchecked instead of refusing to open the document.
    """
    custom = Environment(
        background=(0.5, 0.2, 0.2),
        sky_enabled=False,
        sky_color=(0.5, 0.2, 0.2),
        ground_enabled=False,
        ground_color=(0.5, 0.2, 0.2),
        ground_opacity=1.0,
        edge_color=(1.0, 1.0, 1.0),
        grid_color=(0.9, 0.9, 0.9),
        grid_centerline_color=(0.8, 0.8, 0.8),
    )
    assert preset_key(custom) is None


def test_the_two_defaults_are_deliberately_different():
    """Spec D5. A new document opens for modelling; a pre-M7.7 file reopens dark.

    Unifying these would either change how every old file looks on load or make
    every new document dark, so the difference is asserted rather than assumed.
    """
    assert DEFAULT_ENVIRONMENT is SKY_AND_GROUND
    assert LEGACY_ENVIRONMENT is STUDIO
    assert DEFAULT_ENVIRONMENT != LEGACY_ENVIRONMENT


def test_the_pass_is_needed_only_when_a_half_is_enabled():
    """Plain White and Studio are already fully painted by glClearColor."""
    assert environment_pass_needed(SKY_AND_GROUND) is True
    assert environment_pass_needed(PLAIN_WHITE) is False
    assert environment_pass_needed(STUDIO) is False


def test_every_preset_has_a_distinct_key():
    """Two presets must never share a key: registering one would silently
    shadow the other in PRESETS and in the View menu's action lookup.
    """
    keys = list(PRESETS.keys())
    assert len(keys) == len(set(keys))


def test_presets_has_exactly_five_entries_and_every_key_is_registered():
    """Task 6b brings the preset count to five. Pinning the count and the exact
    key set means a future preset added to the module without being wired into
    PRESETS is caught here rather than discovered later as a menu entry with
    nothing behind it.
    """
    assert len(PRESETS) == 5
    assert set(PRESETS) == {
        SKY_AND_GROUND_KEY,
        NEUTRAL_GREY_KEY,
        LANDSCAPE_KEY,
        PLAIN_WHITE_KEY,
        STUDIO_KEY,
    }
