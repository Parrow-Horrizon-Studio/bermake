"""The OpenGL check before the main window opens (M7.9, final review I1).

This machine has a real GPU, so the OpenGL 1.x and no-context cases are
simulated: the decision and the sequence around it take the probe's result
as input. probe_gl itself runs once against the suite's own platform.
"""

import logging
import subprocess
import sys

import pytest
from bermake.diagnostics.gl_check import GlInfo
from bermake.diagnostics.gl_preflight import (
    NO_CONTEXT_MESSAGE,
    PreflightProblem,
    preflight_problem,
    run_gl_preflight,
)

GL_1_1 = GlInfo((1, 1), "1.1.0", "GDI Generic")
GL_4_6 = GlInfo((4, 6), "4.6.0 NVIDIA 617.14", "NVIDIA GeForce RTX 4070 Ti")


# --- The decision ------------------------------------------------------------


def test_no_context_at_all_has_its_own_message_and_offers_the_switch():
    problem = preflight_problem(None, mesa_available=True, compat_active=False)
    assert problem == PreflightProblem(NO_CONTEXT_MESSAGE, True)
    assert "no OpenGL context could be created" in problem.message


def test_an_old_context_names_what_it_got_and_offers_the_switch():
    problem = preflight_problem(GL_1_1, mesa_available=True, compat_active=False)
    assert problem.offer_restart is True
    assert "OpenGL 1.1" in problem.message
    assert "GDI Generic" in problem.message


def test_an_old_context_without_mesa_explains_without_offering():
    problem = preflight_problem(GL_1_1, mesa_available=False, compat_active=False)
    assert problem.offer_restart is False
    assert "OpenGL 1.1" in problem.message


def test_no_offer_when_compatibility_rendering_is_already_active():
    assert preflight_problem(None, mesa_available=True, compat_active=True).offer_restart is False
    assert preflight_problem(GL_1_1, mesa_available=True, compat_active=True).offer_restart is False


@pytest.mark.parametrize("version", [(3, 3), (4, 6)])
def test_a_good_context_is_no_problem(version):
    info = GlInfo(version, "x", "y")
    assert preflight_problem(info, mesa_available=True, compat_active=False) is None


# --- The sequence around it --------------------------------------------------


class _Recorder:
    def __init__(self, *, answer=True, relaunched=True):
        self.answer = answer
        self.relaunched = relaunched
        self.events = []

    def prompt(self, message, offer):
        self.events.append(("prompt", message, offer))
        return self.answer

    def store(self):
        self.events.append(("store",))

    def relaunch(self):
        self.events.append(("relaunch",))
        return self.relaunched

    def run(self, info, *, mesa_available=True, compat_active=False):
        return run_gl_preflight(
            mesa_available=mesa_available,
            compat_active=compat_active,
            prompt=self.prompt,
            store_preference=self.store,
            relaunch=self.relaunch,
            probe=lambda: info,
        )


def test_a_good_context_shows_nothing_and_continues():
    recorder = _Recorder()
    assert recorder.run(GL_4_6) is False
    assert recorder.events == []


def test_accepting_the_offer_stores_the_preference_then_relaunches():
    recorder = _Recorder(answer=True)
    assert recorder.run(None) is True
    assert recorder.events == [("prompt", NO_CONTEXT_MESSAGE, True), ("store",), ("relaunch",)]


def test_declining_the_offer_continues_to_the_window():
    recorder = _Recorder(answer=False)
    assert recorder.run(GL_1_1) is False
    assert [event[0] for event in recorder.events] == ["prompt"]
    assert recorder.events[0][2] is True


def test_without_mesa_the_message_is_shown_and_startup_continues():
    recorder = _Recorder(answer=True)
    assert recorder.run(GL_1_1, mesa_available=False) is False
    assert [event[0] for event in recorder.events] == ["prompt"]
    assert recorder.events[0][2] is False


def test_already_active_compatibility_rendering_explains_and_continues():
    recorder = _Recorder(answer=True)
    assert recorder.run(None, compat_active=True) is False
    assert recorder.events == [("prompt", NO_CONTEXT_MESSAGE, False)]


def test_a_failed_relaunch_is_logged_and_startup_continues(caplog):
    recorder = _Recorder(answer=True, relaunched=False)
    with caplog.at_level(logging.ERROR, logger="bermake.diagnostics.gl_preflight"):
        assert recorder.run(GL_1_1) is False
    assert [event[0] for event in recorder.events] == ["prompt", "store", "relaunch"]
    assert "failed to relaunch" in caplog.text


def test_a_probe_that_raises_is_logged_and_skipped(caplog):
    def broken_probe():
        raise RuntimeError("probe blew up")

    prompts = []
    with caplog.at_level(logging.ERROR, logger="bermake.diagnostics.gl_preflight"):
        result = run_gl_preflight(
            mesa_available=True,
            compat_active=False,
            prompt=lambda message, offer: prompts.append(message) or True,
            store_preference=lambda: None,
            relaunch=lambda: True,
            probe=broken_probe,
        )
    assert result is False
    assert prompts == []
    assert "probe blew up" in caplog.text


# --- The probe ---------------------------------------------------------------


def test_the_probe_reports_a_context_or_none_without_importing_opengl():
    """Under the suite's offscreen platform a context may not exist at all
    (tests/conftest.py); either way the probe must answer, not raise, and
    must not have imported PyOpenGL's GL module to do it."""
    code = (
        "import sys\n"
        "from PySide6.QtWidgets import QApplication\n"
        "from bermake.diagnostics.gl_preflight import probe_gl\n"
        "app = QApplication([sys.argv[0]])\n"
        "info = probe_gl()\n"
        "print('OpenGL.GL' in sys.modules)\n"
        "print(info is None or (info.version >= (1, 0) and bool(info.renderer)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.stdout.split() == ["False", "True"], result.stderr
