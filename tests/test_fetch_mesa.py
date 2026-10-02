"""tools/fetch_mesa.py (M7.9, spec 2.4.1). No test downloads anything."""

import hashlib
import importlib.util
import subprocess
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


MEMBERS = ["x64/opengl32.dll", "x64/libgallium_wgl.dll"]


def test_the_7zip_command_extracts_with_folders_preserved(tmp_path):
    extractor = tmp_path / "7z.exe"
    archive = tmp_path / "mesa.7z"
    into = tmp_path / "scratch"

    command = fetch_mesa.extract_command(extractor, archive, into)

    assert command == [str(extractor), "x", str(archive), f"-o{into}", "-y", *MEMBERS]


def test_the_tar_command_keeps_the_original_arguments(tmp_path):
    extractor = tmp_path / "tar.exe"
    archive = tmp_path / "mesa.7z"
    into = tmp_path / "scratch"

    command = fetch_mesa.extract_command(extractor, archive, into)

    assert command == [str(extractor), "-xf", str(archive), "-C", str(into), *MEMBERS]


def _fake_program_files(tmp_path, *, with_7zip):
    program_files = tmp_path / "Program Files"
    if with_7zip:
        (program_files / "7-Zip").mkdir(parents=True)
        (program_files / "7-Zip" / "7z.exe").write_bytes(b"")
    else:
        program_files.mkdir()
    return program_files


def _fake_tar(tmp_path, *, present):
    tar = tmp_path / "tar.exe"
    if present:
        tar.write_bytes(b"")
    return tar


def test_7zip_on_path_is_preferred_over_tar(tmp_path):
    tar = _fake_tar(tmp_path, present=True)
    on_path = str(tmp_path / "tools" / "7z.EXE")

    chosen = fetch_mesa.find_extractor(
        which=lambda name: on_path if name == "7z" else None,
        program_files=str(_fake_program_files(tmp_path, with_7zip=False)),
        tar=tar,
    )

    assert chosen == Path(on_path)


def test_7zip_on_path_wins_over_7zip_in_program_files(tmp_path):
    program_files = _fake_program_files(tmp_path, with_7zip=True)
    on_path = str(tmp_path / "tools" / "7z.exe")

    chosen = fetch_mesa.find_extractor(
        which=lambda name: on_path,
        program_files=str(program_files),
        tar=_fake_tar(tmp_path, present=True),
    )

    assert chosen == Path(on_path)


def test_which_defaults_to_shutil_which_at_call_time(tmp_path, monkeypatch):
    on_path = str(tmp_path / "7z.exe")
    monkeypatch.setattr(fetch_mesa.shutil, "which", lambda name: on_path)

    assert fetch_mesa.find_extractor() == Path(on_path)


def test_7zip_under_program_files_is_preferred_over_tar(tmp_path):
    program_files = _fake_program_files(tmp_path, with_7zip=True)
    tar = _fake_tar(tmp_path, present=True)

    chosen = fetch_mesa.find_extractor(
        which=lambda name: None, program_files=str(program_files), tar=tar
    )

    assert chosen == program_files / "7-Zip" / "7z.exe"


def test_tar_is_the_fallback_without_7zip(tmp_path):
    tar = _fake_tar(tmp_path, present=True)

    chosen = fetch_mesa.find_extractor(
        which=lambda name: None,
        program_files=str(_fake_program_files(tmp_path, with_7zip=False)),
        tar=tar,
    )

    assert chosen == tar


def test_no_extractor_at_all_names_both_options(tmp_path):
    tar = _fake_tar(tmp_path, present=False)

    with pytest.raises(fetch_mesa.MesaFetchError) as raised:
        fetch_mesa.find_extractor(
            which=lambda name: None,
            program_files=str(_fake_program_files(tmp_path, with_7zip=False)),
            tar=tar,
        )

    message = str(raised.value)
    assert "7-Zip" in message
    assert str(tar) in message


def _failing_runner(code):
    def run(command, **kwargs):
        raise subprocess.CalledProcessError(code, command)

    return run


def test_an_extractor_failure_names_the_extractor_and_its_exit_code(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    archive = cache / fetch_mesa.MESA_ARCHIVE
    archive.write_bytes(b"pretend archive")
    digest = hashlib.sha256(b"pretend archive").hexdigest()
    monkeypatch.setattr(fetch_mesa, "MESA_SHA256", digest)
    extractor = tmp_path / "tar.exe"
    monkeypatch.setattr(fetch_mesa, "find_extractor", lambda: extractor)

    with pytest.raises(fetch_mesa.MesaFetchError) as raised:
        fetch_mesa.fetch_mesa(tmp_path / "dest", cache, run=_failing_runner(3))

    message = str(raised.value)
    assert str(extractor) in message
    assert "exit code 3" in message
    assert fetch_mesa.MESA_ARCHIVE in message
    assert isinstance(raised.value.__cause__, subprocess.CalledProcessError)
    assert not (tmp_path / "dest" / fetch_mesa.STAMP_NAME).exists()


def test_the_packaging_script_reports_a_mesa_fetch_error_instead_of_a_traceback():
    text = (ROOT / "tools" / "package_windows.py").read_text(encoding="utf-8")
    caught = text.split("except (", 1)[1].split(") as error", 1)[0]
    assert "MesaFetchError" in caught


def test_running_the_tool_directly_reports_a_fetch_error_without_a_traceback(monkeypatch, capsys):
    def failing(dest, cache):
        raise fetch_mesa.MesaFetchError("extracting x.7z with 7z failed (exit code 2)")

    monkeypatch.setattr(fetch_mesa, "fetch_mesa", failing)

    assert fetch_mesa.main() == 1
    assert "exit code 2" in capsys.readouterr().err
