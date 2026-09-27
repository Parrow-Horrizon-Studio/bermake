"""Architectural imperial parse → meters."""

from __future__ import annotations

import pytest
from bermake.units import INCH_M, Units, UnitSystem, parse_length

IMP = Units(system=UnitSystem.IMPERIAL)


def _in(inches: float) -> float:
    return inches * INCH_M


def test_feet_inches_fraction():
    assert parse_length("3' 6 1/2\"", IMP) == pytest.approx(_in(42.5))
    assert parse_length("3'6\"", IMP) == pytest.approx(_in(42.0))
    assert parse_length("3'", IMP) == pytest.approx(_in(36.0))
    assert parse_length("42\"", IMP) == pytest.approx(_in(42.0))
    assert parse_length("6 3/4\"", IMP) == pytest.approx(_in(6.75))
    assert parse_length("6 1/2\"", IMP) == pytest.approx(_in(6.5))


def test_feet_no_close_quote():
    assert parse_length("3' 6", IMP) == pytest.approx(_in(42.0))


def test_bare_number_is_inches_in_imperial():
    assert parse_length("42", IMP) == pytest.approx(_in(42.0))


def test_imperial_garbage_returns_none():
    assert parse_length("3' x 5\"", IMP) is None
