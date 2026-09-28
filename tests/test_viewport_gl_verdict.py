"""The viewport's reaction to an unusable context (M7.9). No GL context is
created: _apply_gl_verdict takes the facts initializeGL would read."""

from bermake.diagnostics.gl_check import GlInfo
from bermake.viewport.viewport_widget import ViewportWidget


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


def test_a_shader_failure_reports_the_renderer_error(qtbot):
    widget = _widget(qtbot)

    def failing_initialise():
        raise RuntimeError("fragment shader compile failed:\n0:1 error")

    with qtbot.waitSignal(widget.gl_unavailable, timeout=2000) as blocker:
        widget._apply_gl_verdict(GlInfo((4, 6), "4.6", "Odd Driver"), failing_initialise)
    assert "fragment shader compile failed" in blocker.args[0]
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
