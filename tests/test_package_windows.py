"""tools/package_windows.py's pure parts (M7.9, spec 2.2). The build itself is
exercised by running the script, in Step 9 and in CI."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pw():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec = importlib.util.spec_from_file_location(
            "package_windows", ROOT / "tools" / "package_windows.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(ROOT / "tools"))


def test_an_existing_toolchain_setting_is_kept(pw, tmp_path):
    env = pw.toolchain_env({"CMAKE_TOOLCHAIN_FILE": "X:/mine.cmake"}, tmp_path / "missing")
    assert env["CMAKE_TOOLCHAIN_FILE"] == "X:/mine.cmake"


def test_the_default_toolchain_is_supplied_when_it_exists(pw, tmp_path):
    default = tmp_path / "vcpkg.cmake"
    default.write_text("")
    assert pw.toolchain_env({}, default)["CMAKE_TOOLCHAIN_FILE"] == default.as_posix()


def test_no_toolchain_anywhere_fails_naming_the_variable(pw, tmp_path):
    with pytest.raises(pw.PackagingError, match="CMAKE_TOOLCHAIN_FILE"):
        pw.toolchain_env({}, tmp_path / "missing")


def test_the_version_resource_carries_the_version(pw):
    text = pw.version_info_text("0.15.0")
    assert "filevers=(0, 15, 0, 0)" in text
    assert "StringStruct('ProductVersion', '0.15.0')" in text
    assert "StringStruct('OriginalFilename', 'Bermake.exe')" in text


def test_a_malformed_version_is_refused(pw):
    with pytest.raises(ValueError):
        pw.version_info_text("0.15")


def test_the_zip_is_named_for_its_version(pw):
    assert pw.zip_name("0.15.0") == "Bermake-0.15.0-windows-x64.zip"


def test_resource_counts_come_from_the_source_tree(pw):
    counts = pw.expected_resource_counts(ROOT)
    assert counts["icons"] == len(list((ROOT / "python/bermake/ui/icons").glob("*.svg")))
    assert counts["icons"] > 0
    assert counts["shaders"] > 0


def _report(icons=44, shaders=8, ok=True):
    return {
        "ok": ok,
        "checks": [
            {
                "name": "resources",
                "ok": True,
                "detail": "",
                "data": {"icons": icons, "shaders": shaders},
            },
            {"name": "rendering", "ok": ok, "detail": "" if ok else "bad", "data": {}},
        ],
    }


def test_a_clean_report_has_no_problems(pw):
    assert pw.smoke_report_problems(_report(), {"icons": 44, "shaders": 8}) == []


def test_a_missing_icon_is_a_problem(pw):
    problems = pw.smoke_report_problems(_report(icons=43), {"icons": 44, "shaders": 8})
    assert any("icons" in p for p in problems)


def test_a_failed_check_is_a_problem_naming_it(pw):
    problems = pw.smoke_report_problems(_report(ok=False), {"icons": 44, "shaders": 8})
    assert any("rendering" in p for p in problems)


def test_the_size_ceiling(pw, tmp_path):
    archive = tmp_path / "a.zip"
    archive.write_bytes(b"x" * 10)
    assert pw.check_zip_size(archive, ceiling=100) == 10
    with pytest.raises(pw.PackagingError, match="ceiling"):
        pw.check_zip_size(archive, ceiling=5)


def test_the_versions_section_lists_every_component(pw):
    versions = {"Python": "3.13.1", "Qt": "6.11.1", "Mesa 3D": "26.2.3", "zlib (static)": "1.3.2"}
    section = pw.versions_section(versions)
    lines = section.splitlines()
    assert lines[0] == "Versions in this build"
    for name, version in versions.items():
        assert any(line.startswith(name) and line.endswith(f"  {version}") for line in lines)


_STATUS = """Package: vcpkg-cmake-get-vars
Version: 2025-05-29
Architecture: x64-windows-static-md
Status: install ok installed

Package: stb
Version: 2024-07-29
Architecture: x64-windows-static-md
Status: purge ok not-installed

Package: zlib
Version: 1.3.2
Architecture: x64-windows-static-md
Status: install ok installed

Package: assimp
Version: 6.0.4
Port-Version: 2
Architecture: x64-windows-static-md
Status: install ok installed

Package: assimp
Feature: draco
Architecture: x64-windows-static-md
Status: install ok installed

Package: gtest
Version: 1.17.0
Architecture: x64-windows-static-md
Status: install ok installed

Package: zlib
Version: 1.2.0
Architecture: x64-windows
Status: install ok installed
"""


def test_vcpkg_ports_are_read_for_the_triplet_without_helpers_or_tests(pw):
    ports = pw.vcpkg_port_versions(_STATUS, "x64-windows-static-md")
    assert ports == {"zlib": "1.3.2", "assimp": "6.0.4#2"}


def test_a_linked_port_missing_from_the_notices_is_reported(pw):
    template = "Statically linked:\n  minizip  zlib License\n  assimp   BSD\n"
    assert pw.unlisted_components(template, ["assimp", "zlib", "draco"]) == ["zlib", "draco"]


def _rendering_report(renderer):
    return {"checks": [{"name": "rendering", "ok": True, "data": {"renderer": renderer}}]}


def test_the_llvm_version_comes_from_the_smoke_reports_renderer(pw):
    report = _rendering_report("llvmpipe (LLVM 23.1.2, 256 bits)")
    assert pw.llvm_version(report) == "23.1.2"
    assert pw.llvm_version_entry(report, "26.2.3") == "23.1.2"


@pytest.mark.parametrize(
    "report",
    [
        _rendering_report("D3D12 (NVIDIA GeForce RTX 4070 Ti)"),
        {"checks": [{"name": "rendering", "ok": False, "detail": "Mesa not found", "data": {}}]},
        {"checks": []},
    ],
)
def test_without_an_llvm_renderer_the_entry_names_the_mesa_it_came_with(pw, report):
    assert pw.llvm_version(report) is None
    assert pw.llvm_version_entry(report, "26.2.3") == "the LLVM bundled in Mesa 26.2.3"


@pytest.mark.parametrize(
    ("component", "licence"),
    [
        ("LLVM", "LLVM/LICENSE.TXT"),
        ("expat", "expat/COPYING.txt"),
        ("mpdecimal", "mpdecimal/COPYRIGHT.txt"),
    ],
)
def test_shipped_libraries_have_an_entry_and_a_checked_in_licence(component, licence):
    """libgallium_wgl.dll carries LLVM, pyexpat.pyd expat and _decimal.pyd
    mpdecimal; Python's own LICENSE.txt covers none of them (final review I2)."""
    template = (ROOT / "packaging" / "THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8")
    assert re.search(rf"^{re.escape(component)} \(", template, re.MULTILINE)
    assert "licenses\\" + licence.replace("/", "\\") in template
    text = (ROOT / "packaging" / "licenses" / licence).read_text(encoding="utf-8")
    assert len(text) > 500
