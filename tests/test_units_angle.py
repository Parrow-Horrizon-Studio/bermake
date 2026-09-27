from __future__ import annotations

import pytest
from bermake.units import format_angle, parse_angle


def test_parse_angle():
    assert parse_angle("47") == pytest.approx(47.0)
    assert parse_angle("47.5°") == pytest.approx(47.5)
    assert parse_angle("30 deg") == pytest.approx(30.0)
    assert parse_angle("nope") is None


def test_format_angle():
    assert format_angle(47.0) == "47°"
    assert format_angle(47.5) == "47.5°"
