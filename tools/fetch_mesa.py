"""Fetch the pinned Mesa build Bermake bundles for compatibility rendering.

PySide6's own opengl32sw.dll is Mesa 11.2.2 and provides OpenGL 3.0, below the
3.3 every Bermake shader needs, so it is replaced by this pinned build
(M7.9 spec 2.4.1, D15). The SHA-256 pin makes the download reproducible and
tamper-evident; upgrading it is a deliberate edit to the four constants below.

The archive is LZMA-compressed 7z. 7-Zip is preferred for extraction because
the tar.exe in older Windows (Server 2022 and so the GitHub windows-2022
runner) is an old libarchive without an LZMA codec and cannot read it; newer
Windows 11 tar.exe can. Windows tar is the fallback for machines without
7-Zip, such as a plain developer box.

Usage: .venv/Scripts/python tools/fetch_mesa.py
Result: build/mesa/opengl32sw.dll and build/mesa/libgallium_wgl.dll
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

MESA_VERSION = "26.2.3"
MESA_ARCHIVE = f"mesa3d-{MESA_VERSION}-release-msvc.7z"
MESA_URL = (
    f"https://github.com/pal1000/mesa-dist-win/releases/download/{MESA_VERSION}/{MESA_ARCHIVE}"
)
MESA_SHA256 = "3f3613adb43cfd0f2e665ce2400b130c275f0b3317cb3a05566320a3a67589ed"

# Archive member -> bundled file name. The loader is renamed because Qt loads
# its software OpenGL from a file called opengl32sw.dll.
MESA_FILES = {
    "x64/opengl32.dll": "opengl32sw.dll",
    "x64/libgallium_wgl.dll": "libgallium_wgl.dll",
}

# Fallback extractor. Windows' own tar reads 7z (newer builds only, see the
# module docstring); Git's GNU tar does not, so the path is explicit.
WINDOWS_TAR = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "tar.exe"

ROOT = Path(__file__).resolve().parent.parent
STAMP_NAME = ".mesa-sha256"


class MesaFetchError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, expected: str = MESA_SHA256) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise MesaFetchError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}. "
            "Delete the file and fetch again; if it persists, the pinned release changed."
        )


def _download(url: str, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.replace(target)


def find_extractor(
    which: Callable[[str], str | None] = shutil.which,
    program_files: str | None = None,
    tar: Path | None = None,
) -> Path:
    """Pick the archive extractor: 7-Zip if present, else Windows tar.

    7-Zip is looked up on PATH, then under Program Files. The arguments exist
    so tests can supply candidates instead of depending on the machine.
    """
    on_path = which("7z")
    if on_path:
        return Path(on_path)
    if program_files is None:
        program_files = os.environ.get("ProgramFiles")
    installed = Path(program_files) / "7-Zip" / "7z.exe" if program_files else None
    if installed is not None and installed.is_file():
        return installed
    fallback = WINDOWS_TAR if tar is None else tar
    if fallback.is_file():
        return fallback
    raise MesaFetchError(
        f"cannot extract 7z: neither 7-Zip (7z on PATH or {installed or 'Program Files'}) "
        f"nor Windows tar ({fallback}) was found"
    )


def extract_command(extractor: Path, archive: Path, into: Path) -> list[str]:
    """Arguments extracting the two Mesa members into `into`, keeping `x64/`."""
    if extractor.name.lower().startswith("7z"):
        # `x` (not `e`) preserves the x64/ folder the copy step reads.
        return [str(extractor), "x", str(archive), f"-o{into}", "-y", *MESA_FILES]
    return [str(extractor), "-xf", str(archive), "-C", str(into), *MESA_FILES]


def fetch_mesa(dest: Path, cache: Path) -> Path:
    """Populate `dest` with the two Mesa DLLs, reusing a verified cache."""
    stamp = dest / STAMP_NAME
    wanted = [dest / name for name in MESA_FILES.values()]
    if stamp.is_file() and stamp.read_text().strip() == MESA_SHA256:
        if all(path.is_file() for path in wanted):
            return dest

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / MESA_ARCHIVE
    if not archive.is_file() or sha256_of(archive) != MESA_SHA256:
        print(f"downloading {MESA_URL}")
        _download(MESA_URL, archive)
    verify_archive(archive)

    extractor = find_extractor()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(extract_command(extractor, archive, Path(scratch)), check=True)
        for member, bundled in MESA_FILES.items():
            shutil.copy2(Path(scratch) / member, dest / bundled)
    stamp.write_text(MESA_SHA256)
    return dest


def main() -> int:
    dest = fetch_mesa(ROOT / "build" / "mesa", ROOT / "build" / "mesa-cache")
    print(f"Mesa {MESA_VERSION} ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
