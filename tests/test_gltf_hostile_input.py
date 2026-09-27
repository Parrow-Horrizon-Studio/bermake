"""Hostile-input hardening for glTF import (#84).

glTF import goes through Assimp, which has a real CVE history. Importing a
file someone sent you is a normal thing for an architect to do, so this is an
untrusted-input path. The bar: a malformed file raises BermakeFormatError; it
never crashes the process, hangs, or exhausts memory.
"""

import json

import pytest
from bermake.io.errors import BermakeFormatError
from bermake.io.gltf_import import _MAX_ELEMENT_COUNT, _validate_gltf_element_counts


def _import(path):
    from bermake.io import read_gltf_scene  # the real public entry point

    return read_gltf_scene(str(path))


def test_truncated_file_raises_format_error(tmp_path):
    p = tmp_path / "truncated.gltf"
    p.write_text('{"asset": {"version": "2.0"}, "meshes": [')  # cut off mid-JSON
    with pytest.raises(BermakeFormatError):
        _import(p)


def test_absurd_declared_counts_are_rejected(tmp_path):
    doc = {
        "asset": {"version": "2.0"},
        "accessors": [{"componentType": 5126, "count": 10**12, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    p = tmp_path / "huge.gltf"
    p.write_text(json.dumps(doc))
    with pytest.raises(BermakeFormatError):
        _import(p)


def test_not_gltf_at_all_raises_format_error(tmp_path):
    p = tmp_path / "nonsense.gltf"
    p.write_bytes(b"\x00\x01\x02 this is not a gltf file \xff\xfe")
    with pytest.raises(BermakeFormatError):
        _import(p)


# The three cases above are also caught by Assimp's own rejection (wrapped as
# BermakeFormatError), so they do not, on their own, prove the pre-Assimp count
# guard fires. These unit tests exercise the guard directly — it is what
# protects against the realistic #84 CVE shape (a buffer-backed accessor whose
# inflated count Assimp would try to honor), which the file-level tests cannot
# reproduce without risking a native OOM.
def test_count_guard_rejects_an_accessor_count_over_the_ceiling():
    doc = {
        "accessors": [{"componentType": 5126, "count": _MAX_ELEMENT_COUNT + 1, "type": "VEC3"}]
    }
    with pytest.raises(BermakeFormatError):
        _validate_gltf_element_counts(doc)


def test_count_guard_accepts_a_reasonable_document():
    doc = {
        "accessors": [{"componentType": 5126, "count": 3, "type": "VEC3", "bufferView": 0}],
        "bufferViews": [{"buffer": 0, "byteLength": 36}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    _validate_gltf_element_counts(doc)  # must not raise


def test_count_guard_rejects_an_out_of_range_bufferview_reference():
    doc = {
        "accessors": [{"componentType": 5126, "count": 3, "type": "VEC3", "bufferView": 7}],
        "bufferViews": [],  # index 7 is out of range
    }
    with pytest.raises(BermakeFormatError):
        _validate_gltf_element_counts(doc)


# Wrong-TYPED container fields must also yield BermakeFormatError, not a raw
# AttributeError/TypeError leaking out of the validator.
def test_count_guard_rejects_wrong_typed_attributes():
    doc = {
        "accessors": [{"componentType": 5126, "count": 3, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": ["POSITION"]}]}],  # array, not object
    }
    with pytest.raises(BermakeFormatError):
        _validate_gltf_element_counts(doc)


def test_count_guard_rejects_wrong_typed_bufferviews():
    doc = {
        "accessors": [{"componentType": 5126, "count": 3, "type": "VEC3", "bufferView": 0}],
        "bufferViews": 5,  # scalar, not an array
    }
    with pytest.raises(BermakeFormatError):
        _validate_gltf_element_counts(doc)
