"""A note over the 3D view when OpenGL cannot draw it (M7.10, #139).

With no OpenGL context at all, QOpenGLWidget never runs initializeGL or
paintGL, so the viewport stays blank and nothing drawn with GL could explain
why. This is an ordinary child QLabel of the viewport, which needs no context.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QLabel, QWidget

NOTE_WITH_RESTART = (
    "Bermake cannot draw the 3D view on this computer. "
    "Turn on Help > Use Compatibility Rendering and restart Bermake."
)
NOTE_WITHOUT_RESTART = (
    "Bermake cannot draw the 3D view on this computer. "
    "Help > About Bermake has details to include in a bug report."
)

_MAX_WIDTH = 440
_MARGIN = 16


def note_text(restart_available: bool) -> str:
    return NOTE_WITH_RESTART if restart_available else NOTE_WITHOUT_RESTART


class GlUnavailableNote(QLabel):
    """Centred, word-wrapped, hidden until show_note().

    The text uses the palette's window text colour on its window colour, so it
    follows the theme and stays readable whatever the viewport behind it is
    (nothing, a grey clear colour, or a drawn scene).
    """

    def __init__(self, viewport: QWidget) -> None:
        super().__init__(viewport)
        self.setObjectName("GlUnavailableNote")
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMargin(_MARGIN)
        self.setAutoFillBackground(True)
        self.setForegroundRole(QPalette.ColorRole.WindowText)
        self.setBackgroundRole(QPalette.ColorRole.Window)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()
        viewport.installEventFilter(self)

    def show_note(self, restart_available: bool) -> None:
        self.setText(note_text(restart_available))
        self._centre()
        self.show()
        self.raise_()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 (Qt virtual)
        if event.type() == QEvent.Type.Resize and self.isVisible():
            self._centre()
        return False

    def _centre(self) -> None:
        parent = self.parentWidget()
        width = max(1, min(_MAX_WIDTH, parent.width() - 2 * _MARGIN))
        height = self.heightForWidth(width)
        self.setGeometry(
            (parent.width() - width) // 2, (parent.height() - height) // 2, width, height
        )
