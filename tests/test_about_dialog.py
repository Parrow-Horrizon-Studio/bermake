"""Help > About Bermake (M7.9, spec 2.4)."""

from pathlib import Path
from types import SimpleNamespace

import bermake.ui.about_dialog as about_module
from bermake.diagnostics.gl_check import GlInfo
from bermake.ui.about_dialog import (
    NOT_INITIALISED,
    AboutDialog,
    AboutInfo,
    about_text,
    gather_about_info,
)
from PySide6.QtWidgets import QLabel

INFO = AboutInfo(
    bermake="9.8.7",
    qt="6.66.6",
    pyside="6.55.5",
    python="3.44.4",
    gl_version="4.6 Mesa 26.2.3",
    gl_renderer="llvmpipe (LLVM 23.1.2)",
    compatibility_rendering=True,
    log_dir="C:/logs/here",
)


def test_the_copied_text_carries_every_field():
    text = about_text(INFO)
    for value in (
        "Bermake 9.8.7",
        "Qt 6.66.6",
        "PySide6 6.55.5",
        "Python 3.44.4",
        "4.6 Mesa 26.2.3",
        "llvmpipe (LLVM 23.1.2)",
        "Compatibility rendering: on",
        "C:/logs/here",
    ):
        assert value in text


def test_compatibility_off_says_off():
    from dataclasses import replace

    assert "Compatibility rendering: off" in about_text(
        replace(INFO, compatibility_rendering=False)
    )


def test_gathering_before_the_viewport_initialised_says_so(tmp_path):
    info = gather_about_info(None, tmp_path, False)
    assert (info.gl_version, info.gl_renderer) == (NOT_INITIALISED, NOT_INITIALISED)
    assert info.log_dir == str(tmp_path)


def test_gathering_reads_the_viewport_facts(tmp_path):
    gl = GlInfo((4, 6), "4.6.0 NVIDIA 617.14", "GeForce RTX")
    info = gather_about_info(gl, tmp_path, False)
    assert (info.gl_version, info.gl_renderer) == ("4.6.0 NVIDIA 617.14", "GeForce RTX")


def test_the_copy_button_puts_the_text_on_the_clipboard(qtbot):
    copied = []
    # A stand-in for QClipboard; only setText is used.
    clipboard = SimpleNamespace(setText=copied.append)
    dialog = AboutDialog(INFO, clipboard=clipboard)
    qtbot.addWidget(dialog)

    dialog.copy_button.click()

    assert copied == [about_text(INFO)]


def test_help_about_opens_the_dialog_with_live_facts(main_window, monkeypatch):
    shown = []
    monkeypatch.setattr(about_module.AboutDialog, "exec", lambda self: shown.append(self.info) or 0)
    main_window._viewport.gl_info = GlInfo((4, 6), "4.6 test", "Test GPU")

    main_window._actions["help_about"].trigger()

    assert len(shown) == 1
    assert shown[0].gl_renderer == "Test GPU"
    assert Path(shown[0].log_dir).name == "logs"


def test_the_dialog_shows_the_copyright_notice(qtbot):
    dialog = AboutDialog(INFO)
    qtbot.addWidget(dialog)

    assert about_module.COPYRIGHT == "Copyright (C) 2026 Parrow Horrizon Studio"
    shown = [label.text() for label in dialog.findChildren(QLabel)]
    assert "Copyright (C) 2026 Parrow Horrizon Studio" in shown
    # Copy details is for bug reports and stays as it was.
    assert "Copyright" not in about_text(INFO)
