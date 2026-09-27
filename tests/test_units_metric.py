"""Metric parse/format (internal length = meters)."""

from __future__ import annotations

import pytest
from bermake.units import Units, UnitSystem, format_length, parse_length

M = Units(system=UnitSystem.METRIC, metric_unit="m", metric_precision=3)
MM = Units(system=UnitSystem.METRIC, metric_unit="mm", metric_precision=1)


def test_parse_explicit_units():
    assert parse_length("1500mm", M) == pytest.approx(1.5)
    assert parse_length("150 cm", M) == pytest.approx(1.5)
    assert parse_length("1.5m", M) == pytest.approx(1.5)
    assert parse_length("2.5 m", MM) == pytest.approx(2.5)


def test_parse_bare_number_uses_display_unit():
    assert parse_length("1500", MM) == pytest.approx(1.5)   # mm display
    assert parse_length("1.5", M) == pytest.approx(1.5)     # m display


def test_parse_comma_decimal_and_space():
    assert parse_length("1,5 m", M) == pytest.approx(1.5)


def test_parse_rejects_garbage_and_negative():
    assert parse_length("", M) is None
    assert parse_length("abc", M) is None
    assert parse_length("-3m", M) is None


def test_format_metric():
    assert format_length(1.5, M) == "1.5 m"
    assert format_length(1.5, MM) == "1500 mm"
    assert format_length(2.0, M) == "2 m"        # trailing zeros stripped
