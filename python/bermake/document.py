"""DocumentSettings — per-document preferences (in-memory for M4d).

Holds the active Units and viewport Environment. File persistence arrives with
the native format (M6).
"""

from __future__ import annotations

from bermake.units import Units, UnitSystem
from bermake.viewport.environment import DEFAULT_ENVIRONMENT, Environment


class DocumentSettings:
    def __init__(self) -> None:
        self._units = Units()
        # A new document is a modelling document, so it opens in the modelling
        # environment. The CODEC defaults to Studio instead, for a file written
        # before the environment existed. The two differ on purpose (spec D5).
        self._environment = DEFAULT_ENVIRONMENT

    @property
    def units(self) -> Units:
        return self._units

    @property
    def environment(self) -> Environment:
        return self._environment

    def set_environment(self, environment: Environment) -> None:
        """Replace the active environment wholesale (used by the View menu, the
        welcome dialog's template, File > New and file load).

        Wholesale rather than field by field for the same reason set_units is:
        the value is frozen and always arrives as a complete preset.
        """
        self._environment = environment

    def set_metric(self, metric_unit: str = "m") -> None:
        self._units = Units(
            system=UnitSystem.METRIC,
            metric_unit=metric_unit,
            metric_precision=self._units.metric_precision,
            imperial_denominator=self._units.imperial_denominator,
        )

    def set_imperial(self, denominator: int = 16) -> None:
        self._units = Units(
            system=UnitSystem.IMPERIAL,
            imperial_denominator=denominator,
            metric_unit=self._units.metric_unit,
            metric_precision=self._units.metric_precision,
        )

    def set_units(self, units: Units) -> None:
        """Replace the active units wholesale (used by file load / New)."""
        self._units = units
