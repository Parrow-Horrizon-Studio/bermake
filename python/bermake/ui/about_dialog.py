"""Help > About Bermake (M7.9, spec 2.4).

Everything a bug report needs, in one place, with a button that copies it as
plain text for a GitHub issue.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout

from bermake.diagnostics.gl_check import GlInfo

NOT_INITIALISED = "not initialised yet"


@dataclass(frozen=True)
class AboutInfo:
    bermake: str
    qt: str
    pyside: str
    python: str
    gl_version: str
    gl_renderer: str
    compatibility_rendering: bool
    log_dir: str


def gather_about_info(
    gl_info: GlInfo | None, log_dir: Path, compatibility_rendering: bool
) -> AboutInfo:
    import PySide6
    from PySide6.QtCore import qVersion

    from bermake import __version__

    return AboutInfo(
        bermake=__version__,
        qt=qVersion(),
        pyside=PySide6.__version__,
        python=platform.python_version(),
        gl_version=gl_info.version_string if gl_info is not None else NOT_INITIALISED,
        gl_renderer=gl_info.renderer if gl_info is not None else NOT_INITIALISED,
        compatibility_rendering=compatibility_rendering,
        log_dir=str(log_dir),
    )


def about_text(info: AboutInfo) -> str:
    return "\n".join(
        [
            f"Bermake {info.bermake}",
            f"Qt {info.qt}",
            f"PySide6 {info.pyside}",
            f"Python {info.python}",
            f"OpenGL: {info.gl_version}",
            f"Renderer: {info.gl_renderer}",
            f"Compatibility rendering: {'on' if info.compatibility_rendering else 'off'}",
            f"Logs: {info.log_dir}",
        ]
    )


class AboutDialog(QDialog):
    def __init__(self, info: AboutInfo, parent=None, *, clipboard=None) -> None:
        super().__init__(parent)
        self.info = info
        self._clipboard = clipboard
        self.setWindowTitle("About Bermake")

        title = QLabel(f"<b>Bermake {info.bermake}</b>")
        details = QLabel(about_text(info))
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.copy_button = QPushButton("Copy details")
        buttons.addButton(self.copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        self.copy_button.clicked.connect(self.copy_details)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(details)
        layout.addWidget(buttons)

    def copy_details(self) -> None:
        clipboard = self._clipboard if self._clipboard is not None else QGuiApplication.clipboard()
        clipboard.setText(about_text(self.info))
