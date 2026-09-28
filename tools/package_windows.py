"""Build the portable Windows zip (M7.9, spec 2.2).

Usage: .venv/Scripts/python tools/package_windows.py
Result: dist/Bermake-<version>-windows-x64.zip

Six stages, stopping at the first failure. This script is the only code that
knows PyInstaller is the packager, and CI runs it unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from fetch_mesa import fetch_mesa

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
CONSTRAINTS = PACKAGING / "constraints.txt"
SPEC = PACKAGING / "bermake.spec"
VENV = ROOT / "build" / "package-venv"
WORK = ROOT / "build" / "package"
MESA_DIR = ROOT / "build" / "mesa"
MESA_CACHE = ROOT / "build" / "mesa-cache"
DIST = ROOT / "dist"
DEFAULT_TOOLCHAIN = Path("C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
SIZE_CEILING_BYTES = 100 * 1024 * 1024


class PackagingError(RuntimeError):
    pass


def toolchain_env(environ: Mapping[str, str], default: Path) -> dict[str, str]:
    """The environment for the wheel build, with a vcpkg toolchain (#137)."""
    env = dict(environ)
    if env.get("CMAKE_TOOLCHAIN_FILE"):
        return env
    if default.is_file():
        env["CMAKE_TOOLCHAIN_FILE"] = default.as_posix()
        return env
    raise PackagingError(
        f"CMAKE_TOOLCHAIN_FILE is not set and {default} does not exist; "
        "point it at vcpkg's scripts/buildsystems/vcpkg.cmake"
    )


def version_info_text(version: str) -> str:
    """A PyInstaller version resource, so Bermake.exe's Properties show it."""
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    major, minor, patch = (int(part) for part in version.split("."))
    numbers = f"({major}, {minor}, {patch}, 0)"
    strings = {
        "CompanyName": "Parrow Horrizon Studio",
        "FileDescription": "Bermake",
        "FileVersion": version,
        "InternalName": "Bermake",
        "LegalCopyright": "GPL-3.0-or-later",
        "OriginalFilename": "Bermake.exe",
        "ProductName": "Bermake",
        "ProductVersion": version,
    }
    structs = ",\n          ".join(f"StringStruct({k!r}, {v!r})" for k, v in strings.items())
    return (
        "VSVersionInfo(\n"
        f"  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}, mask=0x3f, flags=0x0,\n"
        "    OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
        "  kids=[\n"
        "    StringFileInfo([StringTable('040904B0', [\n"
        f"          {structs}])]),\n"
        "    VarFileInfo([VarStruct('Translation', [1033, 1200])])\n"
        "  ]\n"
        ")\n"
    )


def zip_name(version: str) -> str:
    return f"Bermake-{version}-windows-x64.zip"


def expected_resource_counts(root: Path) -> dict[str, int]:
    icons = root / "python" / "bermake" / "ui" / "icons"
    shaders = root / "python" / "bermake" / "viewport" / "shaders"
    return {
        "icons": len(list(icons.glob("*.svg"))),
        "shaders": len([p for p in shaders.iterdir() if p.is_file()]),
    }


def smoke_report_problems(report: dict, expected: dict[str, int]) -> list[str]:
    problems = [
        f"smoke check {check['name']!r} failed: {check['detail']}"
        for check in report.get("checks", [])
        if not check.get("ok")
    ]
    resources = next((c for c in report.get("checks", []) if c["name"] == "resources"), None)
    if resources is None:
        problems.append("smoke report has no resources check")
    else:
        for kind, count in expected.items():
            got = resources["data"].get(kind)
            if got != count:
                problems.append(f"bundle has {got} {kind}, source tree has {count}")
    return problems


def check_zip_size(path: Path, ceiling: int = SIZE_CEILING_BYTES) -> int:
    size = path.stat().st_size
    if size > ceiling:
        raise PackagingError(f"{path.name} is {size:,} bytes, over the {ceiling:,} byte ceiling")
    return size


def _stage(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def _run(command: list, **kwargs) -> None:
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def main() -> int:
    env = toolchain_env(os.environ, DEFAULT_TOOLCHAIN)

    _stage("1/6 clean build venv")
    if VENV.exists():
        shutil.rmtree(VENV)
    _run([sys.executable, "-m", "venv", VENV])
    python = VENV / "Scripts" / "python.exe"
    _run([python, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-q",
            "-c",
            CONSTRAINTS,
            "pyinstaller",
            "pyinstaller-hooks-contrib",
        ]
    )

    _stage("2/6 real wheel install (#136)")
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "-q",
            "-c",
            CONSTRAINTS,
            "--config-settings=build-dir=build/package/wheel-build",
            ROOT,
        ],
        env=env,
        cwd=ROOT,
    )
    version = subprocess.run(
        [str(python), "-c", "import bermake; print(bermake.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"built Bermake {version}")

    _stage("3/6 Mesa")
    mesa_dir = fetch_mesa(MESA_DIR, MESA_CACHE)

    _stage("4/6 PyInstaller")
    WORK.mkdir(parents=True, exist_ok=True)
    version_file = WORK / "version_info.txt"
    version_file.write_text(version_info_text(version), encoding="utf-8")
    _run(
        [
            python,
            "-m",
            "PyInstaller",
            SPEC,
            "--noconfirm",
            "--distpath",
            WORK / "dist",
            "--workpath",
            WORK / "pyinstaller",
        ],
        env={**env, "BERMAKE_MESA_DIR": str(mesa_dir), "BERMAKE_VERSION_FILE": str(version_file)},
        cwd=ROOT,
    )
    app_dir = WORK / "dist" / "Bermake"

    _stage("5/6 smoke test")
    report_path = WORK / "smoke-report.json"
    report_path.unlink(missing_ok=True)
    smoke_env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    smoke_env["GALLIUM_DRIVER"] = "llvmpipe"
    result = subprocess.run(
        [str(app_dir / "Bermake.exe"), "--smoke-test", str(report_path)],
        env=smoke_env,
        timeout=600,
        check=False,
    )
    if not report_path.is_file():
        raise PackagingError(f"smoke test wrote no report (exit code {result.returncode})")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2))
    problems = smoke_report_problems(report, expected_resource_counts(ROOT))
    if result.returncode != 0 or problems:
        raise PackagingError("smoke test failed:\n  " + "\n  ".join(problems or ["see report"]))

    _stage("6/6 zip")
    shutil.copy2(PACKAGING / "README.txt", app_dir / "README.txt")
    shutil.copy2(PACKAGING / "THIRD-PARTY-NOTICES.txt", app_dir / "THIRD-PARTY-NOTICES.txt")
    shutil.copy2(ROOT / "LICENSE", app_dir / "LICENSE.txt")
    DIST.mkdir(exist_ok=True)
    archive = DIST / zip_name(version)
    archive.unlink(missing_ok=True)
    shutil.make_archive(
        str(archive.with_suffix("")), "zip", root_dir=WORK / "dist", base_dir="Bermake"
    )
    size = check_zip_size(archive)
    print(f"{archive} ({size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PackagingError as error:
        print(f"packaging failed: {error}", file=sys.stderr)
        sys.exit(1)
