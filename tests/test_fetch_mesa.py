"""tools/fetch_mesa.py (M7.9, spec 2.4.1). No test downloads anything."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_tool():
    spec = importlib.util.spec_from_file_location("fetch_mesa", ROOT / "tools" / "fetch_mesa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_mesa = _load_tool()


def test_the_pin_is_self_consistent():
    assert fetch_mesa.MESA_ARCHIVE == f"mesa3d-{fetch_mesa.MESA_VERSION}-release-msvc.7z"
    assert fetch_mesa.MESA_URL.endswith(f"/{fetch_mesa.MESA_VERSION}/{fetch_mesa.MESA_ARCHIVE}")
    assert len(fetch_mesa.MESA_SHA256) == 64


def test_the_loader_is_bundled_under_the_name_qt_looks_for():
    assert fetch_mesa.MESA_FILES == {
        "x64/opengl32.dll": "opengl32sw.dll",
        "x64/libgallium_wgl.dll": "libgallium_wgl.dll",
    }


def test_a_matching_digest_verifies(tmp_path):
    archive = tmp_path / "a.7z"
    archive.write_bytes(b"pretend archive")
    digest = hashlib.sha256(b"pretend archive").hexdigest()

    fetch_mesa.verify_archive(archive, digest)


def test_a_wrong_digest_is_refused(tmp_path):
    archive = tmp_path / "a.7z"
    archive.write_bytes(b"tampered archive")

    with pytest.raises(fetch_mesa.MesaFetchError, match="SHA-256"):
        fetch_mesa.verify_archive(archive, "0" * 64)
