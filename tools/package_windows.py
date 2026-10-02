"""Build the portable Windows zip (M7.9, spec 2.2).

Usage: .venv/Scripts/python tools/package_windows.py
Result: dist/Bermake-<version>-windows-x64.zip

Six stages, stopping at the first failure. This script is the only code that
knows PyInstaller is the packager, and CI runs it unchanged.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from fetch_mesa import MESA_VERSION, MesaFetchError, fetch_mesa

ROOT = Path(__file__).resolve().parent.parent
PACKAGING = ROOT / "packaging"
CONSTRAINTS = PACKAGING / "constraints.txt"
SPEC = PACKAGING / "bermake.spec"
VENV = ROOT / "build" / "package-venv"
WORK = ROOT / "build" / "package"
MESA_DIR = ROOT / "build" / "mesa"
MESA_CACHE = ROOT / "build" / "mesa-cache"
DIST = ROOT / "dist"
WHEEL_BUILD = WORK / "wheel-build"
NOTICES_TEMPLATE = PACKAGING / "THIRD-PARTY-NOTICES.txt"
CHECKED_IN_LICENCES = PACKAGING / "licenses"
DEFAULT_TOOLCHAIN = Path("C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
SIZE_CEILING_BYTES = 100 * 1024 * 1024

# Bundled Python distributions the notices list, as display name ->
# distribution name, in the order they are listed.
BUNDLED_DISTRIBUTIONS = {
    "PySide6": "PySide6",
    "shiboken6": "shiboken6",
    "numpy": "numpy",
    "PyOpenGL": "PyOpenGL",
    "mapbox-earcut": "mapbox_earcut",
    "nanobind": "nanobind",
    "PyInstaller bootloader": "pyinstaller",
}
# Distributions whose dist-info licence files are copied into licenses/. The
# others are covered elsewhere: PySide6 and shiboken6 by licenses/LGPL-3.0.txt,
# numpy by its dist-info, which ships in _internal, and PyOpenGL, whose wheel
# carries no licence file, by the checked-in packaging/licenses/PyOpenGL/.
COPY_LICENCES_OF = ("mapbox_earcut", "nanobind", "pyinstaller")
# vcpkg ports installed for the C++ test suite only, never linked into _core.
VCPKG_TEST_ONLY = {"gtest"}

# Run in the build venv with -I, so it reads what is installed there and not
# a source tree on the path.
_PROBE = """
import decimal, json, platform, pyexpat, sys
from importlib.metadata import distribution
from PySide6.QtCore import qVersion

result = {"Python": platform.python_version(), "Qt": qVersion(),
          "expat": pyexpat.EXPAT_VERSION.removeprefix("expat_"),
          "mpdecimal": decimal.__libmpdec_version__,
          "base_prefix": sys.base_prefix, "distributions": {}}
for name in json.loads(sys.argv[1]):
    dist = distribution(name)
    licences = [
        str(dist.locate_file(f)) for f in dist.files or []
        if f.parts[0].endswith(".dist-info")
        and ("licenses" in f.parts or f.name.upper().startswith(("LICENSE", "COPYING")))
    ]
    result["distributions"][name] = {"version": dist.version, "licences": licences}
print(json.dumps(result))
"""


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


def read_copyright(source: Path | None = None) -> str:
    """The notice from bermake.ui.about_dialog, read from source.

    The packaging interpreter has no built bermake._core (and no Qt), so the
    module cannot be imported here; the constant is a plain string literal.
    """
    path = source or ROOT / "python" / "bermake" / "ui" / "about_dialog.py"
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "COPYRIGHT" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise PackagingError(f"no COPYRIGHT constant in {path}")


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
        "LegalCopyright": read_copyright(),
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


def llvm_version(report: dict) -> str | None:
    """The LLVM version Mesa reports in the smoke test's renderer string
    ("llvmpipe (LLVM 23.1.2, 256 bits)"), or None if it names none.

    Read from the build rather than kept by hand: LLVM is statically linked
    into libgallium_wgl.dll, so the Mesa pin decides it.
    """
    rendering = next((c for c in report.get("checks", []) if c.get("name") == "rendering"), None)
    renderer = str((rendering or {}).get("data", {}).get("renderer", ""))
    match = re.search(r"LLVM (\d+(?:\.\d+)+)", renderer)
    return match.group(1) if match else None


def llvm_version_entry(report: dict, mesa_version: str) -> str:
    return llvm_version(report) or f"the LLVM bundled in Mesa {mesa_version}"


def versions_section(versions: Mapping[str, str]) -> str:
    """The generated tail of the shipped THIRD-PARTY-NOTICES.txt."""
    title = "Versions in this build"
    width = max(len(name) for name in versions)
    lines = [title, "-" * len(title)]
    lines += [f"{name.ljust(width)}  {version}" for name, version in versions.items()]
    return "\n".join(lines) + "\n"


def vcpkg_port_versions(status: str, triplet: str) -> dict[str, str]:
    """Installed vcpkg ports for `triplet`, from vcpkg_installed/vcpkg/status.

    Feature paragraphs (which carry no Version), vcpkg's own helper ports and
    test-only ports are left out. A non-zero port version is appended the way
    vcpkg writes it (1.3#2).
    """
    ports = {}
    for paragraph in status.replace("\r\n", "\n").split("\n\n"):
        fields = dict(line.split(": ", 1) for line in paragraph.splitlines() if ": " in line)
        name = fields.get("Package", "")
        if (
            fields.get("Architecture") != triplet
            or "Version" not in fields
            or not fields.get("Status", "").endswith(" installed")
            or name.startswith("vcpkg-")
            or name in VCPKG_TEST_ONLY
        ):
            continue
        version = fields["Version"]
        if fields.get("Port-Version", "0") != "0":
            version += f"#{fields['Port-Version']}"
        ports[name] = version
    return ports


def unlisted_components(template: str, names: list[str]) -> list[str]:
    """Names with no entry line (indented, name first) in the notices template."""
    return [
        name for name in names if not re.search(rf"^\s+{re.escape(name)}\s", template, re.MULTILINE)
    ]


def check_zip_size(path: Path, ceiling: int = SIZE_CEILING_BYTES) -> int:
    size = path.stat().st_size
    if size > ceiling:
        raise PackagingError(f"{path.name} is {size:,} bytes, over the {ceiling:,} byte ceiling")
    return size


def _stage(label: str) -> None:
    print(f"\n=== {label} ===", flush=True)


def _run(command: list, **kwargs) -> None:
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def _write_notices(app_dir: Path, probe: dict, report: dict) -> None:
    """THIRD-PARTY-NOTICES.txt with this build's versions, and licenses/."""
    cache = (WHEEL_BUILD / "CMakeCache.txt").read_text(encoding="utf-8")
    match = re.search(r"^VCPKG_TARGET_TRIPLET:STRING=(.+)$", cache, re.MULTILINE)
    if match is None:
        raise PackagingError("no VCPKG_TARGET_TRIPLET in the wheel build's CMakeCache.txt")
    triplet = match.group(1).strip()
    vcpkg_installed = WHEEL_BUILD / "vcpkg_installed"
    status = (vcpkg_installed / "vcpkg" / "status").read_text(encoding="utf-8")
    ports = vcpkg_port_versions(status, triplet)

    template = NOTICES_TEMPLATE.read_text(encoding="utf-8")
    unlisted = unlisted_components(template, sorted(ports))
    if unlisted:
        raise PackagingError(
            f"linked into _core but missing from {NOTICES_TEMPLATE.name}: {unlisted}"
        )
    distributions = probe["distributions"]
    versions = {"Python": probe["Python"], "Qt": probe["Qt"]}
    for label, name in BUNDLED_DISTRIBUTIONS.items():
        versions[label] = distributions[name]["version"]
    versions["Mesa 3D"] = MESA_VERSION
    versions["LLVM (in Mesa)"] = llvm_version_entry(report, MESA_VERSION)
    versions["expat"] = probe["expat"]
    versions["mpdecimal"] = probe["mpdecimal"]
    versions.update({f"{name} (static)": version for name, version in ports.items()})
    notices = template + "\n" + versions_section(versions)
    (app_dir / "THIRD-PARTY-NOTICES.txt").write_text(notices, encoding="utf-8")

    licences = app_dir / "licenses"
    shutil.copytree(CHECKED_IN_LICENCES, licences, dirs_exist_ok=True)
    python_licence = Path(probe["base_prefix"]) / "LICENSE.txt"
    if not python_licence.is_file():
        raise PackagingError(f"Python's licence file is missing: {python_licence}")
    (licences / "Python").mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_licence, licences / "Python" / "LICENSE.txt")
    for name in COPY_LICENCES_OF:
        files = [Path(f) for f in distributions[name]["licences"]]
        if not files:
            raise PackagingError(f"{name} carries no licence file to ship")
        (licences / name).mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, licences / name / file.name)
    for name in ports:
        copyright_file = vcpkg_installed / triplet / "share" / name / "copyright"
        if not copyright_file.is_file():
            raise PackagingError(f"vcpkg port {name} has no copyright file: {copyright_file}")
        (licences / name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(copyright_file, licences / name / "copyright.txt")


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
            # Linked into _core; installed here only for its version and licence.
            "nanobind",
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
            # pip 26.2 applies only build constraints inside the isolated
            # build environment; PIP_CONSTRAINT and -c do not reach it.
            "--build-constraint",
            CONSTRAINTS,
            "--config-settings=build-dir=build/package/wheel-build",
            ROOT,
        ],
        env=env,
        cwd=ROOT,
    )
    version = subprocess.run(
        [str(python), "-I", "-c", "import bermake; print(bermake.__version__)"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    print(f"built Bermake {version}")
    probe = json.loads(
        subprocess.run(
            [str(python), "-I", "-c", _PROBE, json.dumps(list(BUNDLED_DISTRIBUTIONS.values()))],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout
    )

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
    _write_notices(app_dir, probe, report)
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
    except (
        PackagingError,
        MesaFetchError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"packaging failed: {error}", file=sys.stderr)
        sys.exit(1)
