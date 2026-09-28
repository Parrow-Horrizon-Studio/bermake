"""Switching to the bundled Mesa, and relaunching (M7.9, spec 2.4.1).

Every Qt and PyOpenGL side effect is injected: the suite's own OpenGL state
must not be redirected by a test.
"""

import sys

import pytest
from bermake.diagnostics import compat_rendering as cr


def _mesa(folder, *, driver=True):
    folder.mkdir(parents=True)
    (folder / cr.MESA_LOADER).write_bytes(b"")
    if driver:
        (folder / cr.MESA_DRIVER).write_bytes(b"")
    return folder


def test_frozen_builds_look_in_the_bundle(tmp_path):
    mesa = _mesa(tmp_path / "bundle" / "mesa")
    found = cr.locate_mesa_dir(frozen=True, bundle_dir=tmp_path / "bundle", repo_root=tmp_path)
    assert found == mesa


def test_checkouts_look_in_the_fetched_copy(tmp_path):
    mesa = _mesa(tmp_path / "build" / "mesa")
    assert cr.locate_mesa_dir(frozen=False, bundle_dir=None, repo_root=tmp_path) == mesa


def test_a_missing_driver_means_no_mesa(tmp_path):
    _mesa(tmp_path / "build" / "mesa", driver=False)
    assert cr.locate_mesa_dir(frozen=False, bundle_dir=None, repo_root=tmp_path) is None


def test_enabling_redirects_pyopengl_and_sets_the_qt_attribute(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "OpenGL.GL", raising=False)
    calls = []

    cr.enable_compatibility_rendering(
        tmp_path,
        set_gl_library=lambda path: calls.append(("pyopengl", path)),
        set_attribute=lambda: calls.append(("qt",)),
    )

    assert sorted(calls) == [("pyopengl", tmp_path / cr.MESA_LOADER), ("qt",)]


def test_enabling_after_opengl_gl_is_imported_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "OpenGL.GL", object())
    with pytest.raises(RuntimeError, match=r"before OpenGL\.GL"):
        cr.enable_compatibility_rendering(
            tmp_path, set_gl_library=lambda path: None, set_attribute=lambda: None
        )


def test_a_frozen_relaunch_reuses_argv_without_the_flag():
    command = cr.relaunch_command(
        frozen=True,
        executable="C:/B/Bermake.exe",
        argv=["C:/B/Bermake.exe", "--compatibility-rendering", "C:/m/a b.berm"],
        orig_argv=["unused"],
    )
    assert command == ["C:/B/Bermake.exe", "C:/m/a b.berm"]


def test_a_checkout_relaunch_reuses_the_interpreter_arguments():
    command = cr.relaunch_command(
        frozen=False,
        executable="C:/v/python.exe",
        argv=["C:/r/python/bermake/app.py"],
        orig_argv=["C:/v/python.exe", "-m", "bermake.app", "--compatibility-rendering"],
    )
    assert command == ["C:/v/python.exe", "-m", "bermake.app"]


def test_relaunch_starts_the_command_detached():
    started = []

    def fake_start(program, arguments):
        started.append((program, arguments))
        return True

    assert cr.relaunch(command=["C:/B/Bermake.exe", "x.berm"], start_detached=fake_start)
    assert started == [("C:/B/Bermake.exe", ["x.berm"])]
