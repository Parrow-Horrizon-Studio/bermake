"""Architectural imperial format with carry/reduce/omit."""

from __future__ import annotations

from bermake.units import INCH_M, Units, UnitSystem, format_length

IMP = Units(system=UnitSystem.IMPERIAL, imperial_denominator=16)


def _m(inches: float) -> float:
    return inches * INCH_M


def test_basic_forms():
    assert format_length(_m(42.5), IMP) == "3' 6 1/2\""
    assert format_length(_m(36.0), IMP) == "3'"
    assert format_length(_m(42.0), IMP) == "3' 6\""
    assert format_length(_m(6.5), IMP) == "6 1/2\""
    assert format_length(_m(6.0), IMP) == "6\""


def test_fraction_reduces():
    assert format_length(_m(6.0 + 8 / 16), IMP) == "6 1/2\""   # 8/16 → 1/2


def test_fraction_carries_to_inch_and_foot():
    # 11 + 15.9/16" rounds the fraction up to a whole inch → 12" → carries to 1'
    assert format_length(_m(11.0 + 15.97 / 16), IMP) == "1'"
    # 6 + 15.97/16 → 7"
    assert format_length(_m(6.0 + 15.97 / 16), IMP) == "7\""


def test_zero():
    assert format_length(0.0, IMP) == "0\""
