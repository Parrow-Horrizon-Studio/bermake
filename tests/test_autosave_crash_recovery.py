"""End-to-end crash test for autosave and recovery (M7.12, #77; spec section 6).

Three real child processes, one after the other, share one recovery folder:

1. The first builds a MainWindow, draws one face, lets an autosave run and then
   dies with `os._exit(1)`: no `closeEvent`, no cleanup, exactly what a crash
   or End Task leaves behind.
2. The second finds the session through the real liveness check (the first
   child's PID is gone by now) and reads the face back from the recovery file.
3. The third recovers it through `MainWindow.recover_session`, with the dialog
   bypassed, and checks the title and that the files now belong to it.

The children run with `QT_QPA_PLATFORM=offscreen`, so no window appears, and
redirect the recovery folder by patching the one module attribute
`bermake.recovery.paths.default_recovery_directory`, so the real folder is
never touched. Nothing here depends on Windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from bermake.recovery.store import RecoveryStore

# What every child does first: an offscreen app, the recovery folder and the
# QSettings store redirected into the test's temporary directory.
_PRELUDE = """
import json, os, sys
import numpy as np
folder, ini = sys.argv[1], sys.argv[2]

import bermake.recovery.paths as recovery_paths
from pathlib import Path
recovery_paths.default_recovery_directory = lambda: Path(folder)

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import bermake.ui.main_window as main_window_module
main_window_module.QSettings = lambda *a, **k: QSettings(ini, QSettings.Format.IniFormat)

app = QApplication([])
"""

_FIRST_CHILD = (
    _PRELUDE
    + """
from bermake.recovery.scheduler import AutosaveScheduler
from bermake.ui.main_window import MainWindow

class Clock:
    now = 1000.0
    def __call__(self):
        return self.now

clock = Clock()
window = MainWindow()
window._autosave_timer.stop()
window._autosave = AutosaveScheduler(5, clock)

scene = window._model.root.mesh
vids = [
    scene.add_vertex(np.array(p, dtype=np.float32))
    for p in ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0))
]
scene.add_face_from_loop(vids)
window._on_document_changed()

clock.now += 5 * 60 + 1
window._last_input -= 3600.0
window._autosave_tick()

files = sorted(p.name for p in Path(folder).iterdir())
assert any(n.endswith(".berm") for n in files), files
assert any(n.endswith(".json") for n in files), files
print(json.dumps({"pid": os.getpid(), "files": files}), flush=True)
os._exit(1)
"""
)

_SECOND_CHILD = (
    _PRELUDE
    + """
from bermake.io import load_document
from bermake.recovery.startup import recoverable_sessions
from bermake.recovery.store import RecoveryStore

sessions = recoverable_sessions(RecoveryStore(Path(folder)))
faces = None
if sessions:
    mesh = load_document(sessions[0].berm_path).model.root.mesh
    faces = len(list(mesh.faces_iter()))
print(json.dumps({"sessions": len(sessions), "faces": faces}), flush=True)
"""
)

_THIRD_CHILD = (
    _PRELUDE
    + """
from bermake.recovery.startup import recoverable_sessions
from bermake.recovery.store import RecoveryStore
from bermake.ui.main_window import MainWindow

window = MainWindow()
window._autosave_timer.stop()
sessions = recoverable_sessions(RecoveryStore(Path(folder)))
assert len(sessions) == 1, sessions
recovered = window.recover_session(sessions[0])
mesh = window._model.root.mesh
print(
    json.dumps(
        {
            "pid": os.getpid(),
            "recovered": recovered,
            "title": window.windowTitle(),
            "faces": len(list(mesh.faces_iter())),
            "session_id": window._session_id,
            "old_session_id": sessions[0].meta.session_id,
        }
    ),
    flush=True,
)
"""
)


def _run(script: str, folder: Path, ini: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(folder), str(ini)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _last_json(result: subprocess.CompletedProcess) -> dict:
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"no result line\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return json.loads(lines[-1])


@pytest.fixture
def crashed(tmp_path):
    """Run the first child to its `os._exit(1)`; return what it reported."""
    folder = tmp_path / "recovery"
    ini = tmp_path / "child_settings.ini"
    result = _run(_FIRST_CHILD, folder, ini)
    # os._exit(1) is the point: a clean exit would mean closeEvent-style cleanup.
    assert result.returncode == 1, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return folder, ini, _last_json(result)


def test_autosave_survives_a_hard_exit_and_a_new_process_finds_it(crashed):
    folder, ini, first = crashed

    # The crash left both files behind (nothing deleted them).
    assert sorted(p.name for p in folder.iterdir()) == first["files"]

    result = _run(_SECOND_CHILD, folder, ini)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    found = _last_json(result)
    assert found == {"sessions": 1, "faces": 1}

    # The metadata names the process that died, which is why liveness cleared it.
    sessions, quarantined = RecoveryStore(folder).list_sessions()
    assert quarantined == []
    assert [s.meta.pid for s in sessions] == [first["pid"]]


def test_recovering_in_a_new_process_titles_it_and_takes_over_the_files(crashed):
    folder, ini, first = crashed

    result = _run(_THIRD_CHILD, folder, ini)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    out = _last_json(result)

    assert out["recovered"] is True
    assert out["faces"] == 1
    assert "(recovered)" in out["title"]
    assert out["pid"] != first["pid"]
    assert out["session_id"] != out["old_session_id"]

    # The work is on disk under the recovering process, not the dead one.
    sessions, quarantined = RecoveryStore(folder).list_sessions()
    assert quarantined == []
    assert [s.meta.session_id for s in sessions] == [out["session_id"]]
    assert [s.meta.pid for s in sessions] == [out["pid"]]
