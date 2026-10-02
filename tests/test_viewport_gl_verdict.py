"""The viewport's reaction to an unusable context (M7.9). No GL context is
created: _apply_gl_verdict takes the facts initializeGL would read."""

from bermake.diagnostics.gl_check import GlInfo
from bermake.viewport.viewport_widget import ViewportWidget
from OpenGL.error import NullFunctionError


def _widget(qtbot):
    widget = ViewportWidget()
    qtbot.addWidget(widget)
    return widget


def test_an_old_context_skips_initialisation_and_reports(qtbot):
    widget = _widget(qtbot)
    initialised = []
    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000) as blocker:
        widget._apply_gl_verdict(
            GlInfo((1, 1), "1.1.0", "GDI Generic"), lambda: initialised.append(True)
        )
    assert initialised == []
    assert widget._gl_failed
    assert "OpenGL 1.1" in blocker.args[0]


def test_a_version_failure_already_reported_at_startup_is_not_shown_again(qtbot):
    """M7.9 pre-tag fix: the startup check showed this very dialog."""
    widget = _widget(qtbot)
    widget.version_failure_reported = True
    initialised = []
    with qtbot.assertNotEmitted(widget.gl_unavailable, wait=300):
        widget._apply_gl_verdict(
            GlInfo((2, 1), "2.1 Mesa", "llvmpipe"), lambda: initialised.append(True)
        )
    assert initialised == []
    assert widget._gl_failed


def test_the_flag_defaults_off_so_an_unreported_version_failure_still_shows(qtbot):
    assert _widget(qtbot).version_failure_reported is False


def test_a_setup_failure_on_a_new_context_reports_even_when_a_version_was_reported(qtbot):
    widget = _widget(qtbot)
    widget.version_failure_reported = True

    def failing_initialise():
        raise RuntimeError("fragment shader compile failed")

    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000) as blocker:
        widget._apply_gl_verdict(GlInfo((4, 6), "4.6", "Odd Driver"), failing_initialise)
    assert "fragment shader compile failed" in blocker.args[0]
    assert widget._gl_failed


def test_a_shader_failure_reports_the_renderer_error(qtbot):
    widget = _widget(qtbot)

    def failing_initialise():
        raise RuntimeError("fragment shader compile failed:\n0:1 error")

    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000) as blocker:
        widget._apply_gl_verdict(GlInfo((4, 6), "4.6", "Odd Driver"), failing_initialise)
    assert "fragment shader compile failed" in blocker.args[0]
    assert widget._gl_failed


def test_a_pyopengl_error_during_setup_still_reports(qtbot):
    """PyOpenGL's errors derive from Exception, not RuntimeError (final
    review I4); they must reach the offer, not the generic error dialog."""
    assert not issubclass(NullFunctionError, RuntimeError)
    widget = _widget(qtbot)

    def failing_initialise():
        raise NullFunctionError("Attempt to call an undefined function glGenVertexArrays")

    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000) as blocker:
        widget._apply_gl_verdict(GlInfo((4, 6), "4.6", "Odd Driver"), failing_initialise)
    assert "could not set up Bermake's viewport" in blocker.args[0]
    assert "glGenVertexArrays" in blocker.args[0]
    assert "Odd Driver" in blocker.args[0]
    assert widget._gl_failed


def test_a_good_context_initialises_quietly_and_records_its_info(qtbot):
    widget = _widget(qtbot)
    initialised = []
    info = GlInfo((4, 6), "4.6.0 NVIDIA", "GeForce")
    with qtbot.assertNotEmitted(widget.gl_unavailable, wait=200):
        widget._apply_gl_verdict(info, lambda: initialised.append(True))
    assert initialised == [True]
    assert not widget._gl_failed
    assert widget.gl_info == info


def test_a_good_context_after_a_bad_one_renders_again(qtbot):
    """initializeGL can run again on a new context (reparenting); the verdict
    on the old one must not stick."""
    widget = _widget(qtbot)
    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000):
        widget._apply_gl_verdict(GlInfo((1, 1), "1.1.0", "GDI Generic"), lambda: None)
    assert widget._gl_failed

    initialised = []
    with qtbot.assertNotEmitted(widget.gl_unavailable, wait=200):
        widget._apply_gl_verdict(
            GlInfo((4, 6), "4.6.0", "GeForce"), lambda: initialised.append(True)
        )
    assert initialised == [True]
    assert not widget._gl_failed


def test_a_failure_on_a_second_context_is_still_a_failure(qtbot):
    widget = _widget(qtbot)
    widget._apply_gl_verdict(GlInfo((4, 6), "4.6.0", "GeForce"), lambda: None)
    assert not widget._gl_failed
    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000):
        widget._apply_gl_verdict(GlInfo((1, 1), "1.1.0", "GDI Generic"), lambda: None)
    assert widget._gl_failed
