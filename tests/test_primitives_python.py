"""Tests for the nanobind bindings exposing the primitive generators to Python.

These generators' `segments`/`rings` int parameters directly index vectors on
the C++ side. Before the M7.4 Task 9 fix round, an out-of-range value did not
raise a Python exception — it crashed the whole interpreter (segfault, exit
139), since nothing but this binding layer stands between arbitrary caller
input and that indexing. A C++-only test cannot prove the crash is fixed,
because the bug is specifically about what happens at the Python boundary —
so these tests call through `bermake._core` directly and assert a catchable
`ValueError`, the same as the milestone's own repro commands.
"""

from __future__ import annotations

import pytest


def test_make_cylinder_rejects_too_few_segments():
    from bermake._core import make_cylinder

    for segments in (2, 0, -1):
        with pytest.raises(ValueError):
            make_cylinder(1.0, 1.0, segments)


def test_make_cylinder_accepts_minimum_segments():
    from bermake._core import make_cylinder

    mesh = make_cylinder(1.0, 1.0, 3)
    assert mesh.face_slab_size() == 5  # 3 sides + top + bottom


def test_make_cone_rejects_too_few_segments():
    from bermake._core import make_cone

    for segments in (2, 0, -1):
        with pytest.raises(ValueError):
            make_cone(1.0, 1.0, segments)


def test_make_cone_accepts_minimum_segments():
    from bermake._core import make_cone

    mesh = make_cone(1.0, 1.0, 3)
    assert mesh.face_slab_size() == 4  # 3 sides + base


def test_make_sphere_rejects_too_few_rings_or_segments():
    from bermake._core import make_sphere

    with pytest.raises(ValueError):
        make_sphere(1.0, 1, 8)
    with pytest.raises(ValueError):
        make_sphere(1.0, 0, 8)
    with pytest.raises(ValueError):
        make_sphere(1.0, 6, 2)
    with pytest.raises(ValueError):
        make_sphere(1.0, 6, 0)


def test_make_sphere_accepts_minimum_rings_and_segments():
    from bermake._core import make_sphere

    mesh = make_sphere(1.0, 2, 3)
    assert mesh.face_slab_size() == 6  # 2 pole fans * 3 segments, no interior bands


def test_make_sphere_rejects_too_few_by_keyword():
    # The milestone's reported repro used a plain keyword argument — cover
    # that call shape explicitly, not just positional.
    from bermake._core import make_sphere

    with pytest.raises(ValueError):
        make_sphere(radius=1.0, rings=1, segments=8)
