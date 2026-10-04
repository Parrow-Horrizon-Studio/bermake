"""The template table: units configuration is what distinguishes them."""

import pytest
from bermake.templates import (
    DEFAULT_TEMPLATE_KEY,
    TEMPLATES,
    template_for_key,
)
from bermake.units import UnitSystem
from bermake.viewport.environment import PRESETS


def test_every_template_environment_is_one_of_the_presets():
    """No template may invent an environment that skipped the contrast floor."""
    presets = set(PRESETS.values())
    for template in TEMPLATES:
        assert template.environment in presets, template.key


def test_template_keys_are_unique():
    keys = [t.key for t in TEMPLATES]
    assert len(keys) == len(set(keys))


def test_templates_differ_in_units_configuration_not_only_environment():
    """The point of the concept.

    If every template were just a preset under another name, the table would be
    decoration. Architectural and Woodworking share SKY_AND_GROUND and differ
    only in units, which is exactly what SketchUp's templates vary.
    """
    arch = template_for_key("architectural")
    wood = template_for_key("woodworking")
    assert arch.environment == wood.environment
    assert arch.units != wood.units


def test_every_preset_is_used_by_at_least_one_template():
    """Every viewport environment must have a first-run route in the welcome dialog.

    This is a design choice: the human partner decided that all five presets are
    equally viable starting points, so none should require a detour through the
    View menu to reach. If you are adding an environment deliberately without a
    template, you are choosing to change this invariant, not working around a
    broken test.
    """
    preset_environments = set(PRESETS.values())
    template_environments = {t.environment for t in TEMPLATES}
    assert template_environments == preset_environments


def test_the_architectural_template_rounds_to_whole_millimetres():
    arch = template_for_key("architectural")
    assert arch.units.system is UnitSystem.METRIC
    assert arch.units.metric_unit == "mm"
    assert arch.units.metric_precision == 0


def test_the_woodworking_template_works_in_sixteenths():
    wood = template_for_key("woodworking")
    assert wood.units.system is UnitSystem.IMPERIAL
    assert wood.units.imperial_denominator == 16


def test_no_template_is_named_plan_view():
    """Templates do not set the camera (spec D6).

    A template called Plan View would promise a top-down view it never applies,
    which is the class of false promise this project has been caught on before.
    """
    assert "plan view" not in {t.name.lower() for t in TEMPLATES}


def test_template_for_key_falls_back_for_an_unknown_key():
    """Review Focus 3.

    welcome/default_template is a stored string. A renamed key or a hand-edited
    ini must not make File > New raise on the user's first click after an
    upgrade.
    """
    assert template_for_key("no_such_template").key == DEFAULT_TEMPLATE_KEY
    assert template_for_key(None).key == DEFAULT_TEMPLATE_KEY
    assert template_for_key("").key == DEFAULT_TEMPLATE_KEY


def test_the_default_template_key_names_a_real_template():
    assert template_for_key(DEFAULT_TEMPLATE_KEY).key == DEFAULT_TEMPLATE_KEY


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.key)
def test_every_template_has_a_name_and_a_description(template):
    assert template.name
    assert template.description
    assert "—" not in template.name
    assert "—" not in template.description


def test_the_studio_description_says_what_it_looks_like_not_its_history():
    # #132: first-run users have never seen the old default, so the text
    # describes what they will see and for whom it suits.
    assert template_for_key("studio").description == (
        "Dark background with no sky or ground, metres. For long sessions and dark rooms."
    )


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.key)
def test_every_template_units_round_trip(template):
    """A template's Units must survive .berm persistence unchanged.

    A template is only worth having if a document started from it reopens with
    the same units. metric_precision 0 is the one most likely to be lost, since
    it is the only template value that differs from the Units dataclass default.
    """
    from bermake.units import units_from_dict, units_to_dict

    assert units_from_dict(units_to_dict(template.units)) == template.units
