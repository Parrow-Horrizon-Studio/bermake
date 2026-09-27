"""Unit tests for the shared Selection object (pure, no Qt)."""

from __future__ import annotations

from bermake.selection import Selection


def test_starts_empty():
    s = Selection()
    assert s.is_empty()
    assert s.counts() == (0, 0, 0, 0, 0)
    assert s.edges == set()
    assert s.faces == set()


def test_replace_sets_contents_and_bumps_version():
    s = Selection()
    v0 = s.version
    s.replace(edges=[1, 2], faces=[5])
    assert s.edges == {1, 2}
    assert s.faces == {5}
    assert s.counts() == (2, 1, 0, 0, 0)
    assert not s.is_empty()
    assert s.version > v0
    s.replace(edges=[9])
    assert s.edges == {9}
    assert s.faces == set()


def test_add_unions():
    s = Selection()
    s.replace(edges=[1])
    s.add(edges=[2, 3], faces=[7])
    assert s.edges == {1, 2, 3}
    assert s.faces == {7}


def test_toggle_edge_and_face():
    s = Selection()
    s.toggle_edge(4)
    assert s.contains_edge(4)
    s.toggle_edge(4)
    assert not s.contains_edge(4)
    s.toggle_face(8)
    assert s.contains_face(8)


def test_clear_only_bumps_when_nonempty():
    s = Selection()
    v0 = s.version
    s.clear()
    assert s.version == v0
    s.replace(faces=[1])
    v1 = s.version
    s.clear()
    assert s.is_empty()
    assert s.version > v1
