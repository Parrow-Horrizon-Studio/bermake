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
    PreflightOutcome,
    PreflightProblem,
    _guarded_draw_test,
    draw_test_problem,
    preflight_problem,
    run_gl_preflight,
)

GL_1_1 = GlInfo((1, 1), "1.1.0", "GDI Generic")
GL_4_6 = GlInfo((4, 6), "4.6.0 NVIDIA 617.14", "NVIDIA GeForce RTX 4070 Ti")
BROKEN_DRIVER = GlInfo(
    (4, 6),
    "4.6.0 Compatibility Profile Context",
    "AMD Radeon(TM) Graphics",
    draw_error="a test image came back wrong",
)
BLUE = (0, 0, 255)
RED = (255, 0, 0)


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


# --- The draw test's decision ------------------------------------------------


def test_the_exact_colours_pass():
    assert draw_test_problem(BLUE, RED) is None


def test_colours_within_the_tolerance_pass():
    assert draw_test_problem((8, 8, 247), (247, 8, 8)) is None


def test_colours_just_outside_the_tolerance_fail():
    assert draw_test_problem((9, 0, 255), RED) is not None
    assert draw_test_problem(BLUE, (255, 0, 9)) is not None
    assert draw_test_problem((0, 0, 246), RED) is not None


@pytest.mark.parametrize(
    ("corner", "centre"),
    [
        ((255, 255, 255), (255, 255, 255)),  # draws nothing and clears to white
        ((0, 0, 0), (0, 0, 0)),  # all black
        (BLUE, BLUE),  # the triangle is missing
        (RED, RED),  # everything is red
        (RED, BLUE),  # swapped
    ],
)
def test_wrong_images_fail(corner, centre):
    assert draw_test_problem(corner, centre) is not None


def test_the_problem_names_what_was_read_back():
    problem = draw_test_problem((1, 2, 3), (4, 5, 6))
    assert "rgb(4, 5, 6) at the centre" in problem
    assert "rgb(1, 2, 3) at the corner" in problem
    assert "red on blue" in problem


# --- A driver that reports a good version and draws nothing ------------------


def test_a_good_version_that_fails_the_draw_test_is_a_problem_with_the_offer():
    problem = preflight_problem(BROKEN_DRIVER, mesa_available=True, compat_active=False)
    assert problem.offer_restart is True
    assert "AMD Radeon(TM) Graphics" in problem.message
    assert "could not set up Bermake's viewport" in problem.message
    assert "a test image came back wrong" in problem.message


def test_the_draw_failure_offers_nothing_when_compatibility_rendering_is_active():
    problem = preflight_problem(BROKEN_DRIVER, mesa_available=True, compat_active=True)
    assert problem is not None
    assert problem.offer_restart is False


def test_the_draw_failure_offers_nothing_without_mesa():
    problem = preflight_problem(BROKEN_DRIVER, mesa_available=False, compat_active=False)
    assert problem is not None
    assert problem.offer_restart is False


def test_a_passed_draw_test_is_no_problem():
    info = GlInfo((4, 6), "x", "y", draw_error=None)
    assert preflight_problem(info, mesa_available=True, compat_active=False) is None


def test_a_low_version_wins_over_the_draw_error():
    info = GlInfo((2, 1), "2.1", "old", draw_error="a test image came back wrong")
    problem = preflight_problem(info, mesa_available=True, compat_active=False)
    assert "needs OpenGL 3.3" in problem.message


def test_an_unexpected_exception_in_the_draw_step_is_logged_and_not_the_drivers_fault(caplog):
    def broken():
        raise RuntimeError("our own bug")

    with caplog.at_level(logging.ERROR, logger="bermake.diagnostics.gl_preflight"):
        assert _guarded_draw_test(broken) is None
    assert "our own bug" in caplog.text
    assert "Traceback" in caplog.text


def test_the_guard_says_the_error_is_bermakes_and_counts_as_passed(caplog):
    def broken():
        raise RuntimeError("our own bug")

    with caplog.at_level(logging.ERROR, logger="bermake.diagnostics.gl_preflight"):
        _guarded_draw_test(broken)
    assert "error in Bermake" in caplog.text
    assert "treated as passed" in caplog.text


def test_a_strict_guard_lets_an_unexpected_exception_through():
    def broken():
        raise RuntimeError("our own bug")

    with pytest.raises(RuntimeError, match="our own bug"):
        _guarded_draw_test(broken, strict=True)


def test_a_strict_guard_still_returns_what_the_driver_reports():
    assert _guarded_draw_test(lambda: "bad", strict=True) == "bad"
    assert _guarded_draw_test(lambda: None, strict=True) is None


def test_a_failure_the_driver_reports_passes_through_the_guard():
    message = "could not create an offscreen image"
    assert _guarded_draw_test(lambda: message) == message


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
    assert recorder.run(GL_4_6) is PreflightOutcome.OK
    assert recorder.events == []


def test_accepting_the_offer_stores_the_preference_then_relaunches():
    recorder = _Recorder(answer=True)
    assert recorder.run(None) is PreflightOutcome.RELAUNCHED
    assert recorder.events == [("prompt", NO_CONTEXT_MESSAGE, True), ("store",), ("relaunch",)]


def test_declining_the_offer_continues_to_the_window():
    recorder = _Recorder(answer=False)
    assert recorder.run(GL_1_1) is PreflightOutcome.REPORTED
    assert [event[0] for event in recorder.events] == ["prompt"]
    assert recorder.events[0][2] is True


def test_a_low_version_declined_is_reported_so_the_window_stays_quiet():
    """GL 2.x to 3.2: Qt made a context, the tester was told and said no."""
    recorder = _Recorder(answer=False)
    assert recorder.run(GlInfo((2, 1), "2.1 Mesa", "llvmpipe")) is PreflightOutcome.REPORTED


def test_without_mesa_the_message_is_shown_and_startup_continues():
    recorder = _Recorder(answer=True)
    assert recorder.run(GL_1_1, mesa_available=False) is PreflightOutcome.REPORTED
    assert [event[0] for event in recorder.events] == ["prompt"]
    assert recorder.events[0][2] is False


def test_already_active_compatibility_rendering_explains_and_continues():
    recorder = _Recorder(answer=True)
    assert recorder.run(None, compat_active=True) is PreflightOutcome.REPORTED
    assert recorder.events == [("prompt", NO_CONTEXT_MESSAGE, False)]


def test_a_failed_relaunch_is_logged_and_startup_continues(caplog):
    recorder = _Recorder(answer=True, relaunched=False)
    with caplog.at_level(logging.ERROR, logger="bermake.diagnostics.gl_preflight"):
        assert recorder.run(GL_1_1) is PreflightOutcome.REPORTED
    assert [event[0] for event in recorder.events] == ["prompt", "store", "relaunch"]
    assert "failed to relaunch" in caplog.text


def test_a_broken_driver_is_offered_the_restart_in_the_sequence():
    recorder = _Recorder(answer=True)
    assert recorder.run(BROKEN_DRIVER) is PreflightOutcome.RELAUNCHED
    assert [event[0] for event in recorder.events] == ["prompt", "store", "relaunch"]
    assert "a test image came back wrong" in recorder.events[0][1]


def test_the_log_says_the_draw_test_passed(caplog):
    with caplog.at_level(logging.INFO, logger="bermake.diagnostics.gl_preflight"):
        _Recorder().run(GL_4_6)
    assert "draw test: passed" in caplog.text


def test_the_log_says_the_draw_test_failed_and_why(caplog):
    with caplog.at_level(logging.INFO, logger="bermake.diagnostics.gl_preflight"):
        _Recorder(answer=False).run(BROKEN_DRIVER)
    assert "draw test: failed, a test image came back wrong" in caplog.text


def test_the_log_says_the_draw_test_did_not_run_on_an_old_version(caplog):
    with caplog.at_level(logging.INFO, logger="bermake.diagnostics.gl_preflight"):
        _Recorder(answer=False).run(GL_1_1)
    assert "draw test: not run" in caplog.text


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
    assert result is PreflightOutcome.OK
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
        "print(info is None or info.version < (3, 3) or info.draw_error is None)\n"
        # Falsification: with an expectation the triangle cannot meet, a 3.3+
        # context must report a problem. This fails if the draw stops running
        # (gate disabled, early return, exception swallowed).
        "import bermake.diagnostics.gl_preflight as g\n"
        "g._RED = (0, 255, 0)\n"
        "wrong = g.probe_gl()\n"
        "print(wrong is None or wrong.version < (3, 3) or wrong.draw_error is not None)\n"
        # Strict mode: an unexpected error in the draw step propagates, while
        # the default swallows it.
        "g._RED = (255, 0, 0)\n"
        "def boom(context):\n"
        "    raise RuntimeError('boom')\n"
        "g._draw_test = boom\n"
        "quiet = g.probe_gl()\n"
        "print(quiet is None or quiet.draw_error is None)\n"
        "try:\n"
        "    g.probe_gl(strict=True)\n"
        "    print(quiet is None or quiet.version < (3, 3))\n"
        "except RuntimeError:\n"
        "    print(True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    # Lines: no OpenGL module, a context or None, a 3.3+ context passes the draw
    # test, a wrong expectation is caught, an error in the draw step is
    # swallowed by default and raised when strict.
    assert result.stdout.split() == ["False"] + ["True"] * 5, result.stdout + result.stderr
