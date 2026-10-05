"""Document templates: a units configuration paired with an environment.

Pure data, no Qt, so the table is testable without a QApplication and the
welcome dialog stays a thin view over it.

Templates differ in UNITS CONFIGURATION, not only in environment, which is what
keeps the concept from being a second name for the preset table. Units already
carries metric_precision and imperial_denominator, and those are exactly what
SketchUp's templates vary: its architectural templates round to whole
millimetres, its woodworking templates work in sixteenths of an inch.

No template sets the camera (spec D6). That is why the white one is called
Documentation rather than Plan View, which would promise a top-down view it
never applies.
"""

from __future__ import annotations

from dataclasses import dataclass

from bermake.units import Units, UnitSystem
from bermake.viewport.environment import (
    LANDSCAPE,
    NEUTRAL_GREY,
    PLAIN_WHITE,
    SKY_AND_GROUND,
    STUDIO,
    Environment,
)


@dataclass(frozen=True)
class Template:
    """One entry in the welcome dialog's template grid."""

    key: str
    name: str
    description: str
    units: Units
    environment: Environment


TEMPLATES: tuple[Template, ...] = (
    Template(
        key="architectural",
        name="Architectural",
        description="Sky and ground, millimetres rounded to whole units.",
        units=Units(
            system=UnitSystem.METRIC,
            metric_unit="mm",
            metric_precision=0,
            imperial_denominator=16,
        ),
        environment=SKY_AND_GROUND,
    ),
    Template(
        key="woodworking",
        name="Woodworking",
        description="Sky and ground, inches to the nearest sixteenth.",
        units=Units(
            system=UnitSystem.IMPERIAL,
            metric_unit="m",
            metric_precision=3,
            imperial_denominator=16,
        ),
        environment=SKY_AND_GROUND,
    ),
    Template(
        key="presentation",
        name="Presentation",
        description="Neutral grey sky and ground, metres. For presentations.",
        units=Units(
            system=UnitSystem.METRIC,
            metric_unit="m",
            metric_precision=3,
            imperial_denominator=16,
        ),
        environment=NEUTRAL_GREY,
    ),
    Template(
        key="site",
        name="Site",
        description="Blue sky over green ground, metres. For site modelling.",
        units=Units(
            system=UnitSystem.METRIC,
            metric_unit="m",
            metric_precision=3,
            imperial_denominator=16,
        ),
        environment=LANDSCAPE,
    ),
    Template(
        key="documentation",
        name="Documentation",
        description="Plain white background, metres. For drawings and export.",
        units=Units(
            system=UnitSystem.METRIC,
            metric_unit="m",
            metric_precision=3,
            imperial_denominator=16,
        ),
        environment=PLAIN_WHITE,
    ),
    Template(
        key="studio",
        name="Studio",
        description=(
            "Dark background with no sky or ground, metres. For long sessions and dark rooms."
        ),
        units=Units(
            system=UnitSystem.METRIC,
            metric_unit="m",
            metric_precision=3,
            imperial_denominator=16,
        ),
        environment=STUDIO,
    ),
)

DEFAULT_TEMPLATE_KEY = "architectural"

_BY_KEY: dict[str, Template] = {t.key: t for t in TEMPLATES}


def template_for_key(key: str | None) -> Template:
    """The template named `key`, or the default when it names nothing.

    Falls back rather than raising because the key arrives from a stored
    preference: a renamed template or a hand-edited settings file must not make
    File > New raise on the first click after an upgrade.
    """
    return _BY_KEY.get(key or "", _BY_KEY[DEFAULT_TEMPLATE_KEY])
