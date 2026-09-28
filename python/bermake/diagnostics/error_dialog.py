"""The dialogs a tester sees when Bermake hits an uncaught error or cannot
draw the 3D view (M7.9, spec 2.4)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def show_error_dialog(log_dir: Path, parent=None) -> None:
    """Modal and synchronous, so it also works before the event loop runs."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Bermake")
    box.setText("Bermake hit an error.")
    box.setInformativeText(
        f"Details were saved to the log. Please attach it when you report the problem.\n\n{log_dir}"
    )
    open_folder = box.addButton("Open log folder", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    if box.clickedButton() is open_folder:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_dir)))


def show_gl_fallback_dialog(message: str, offer_restart: bool, parent=None) -> bool:
    """Explain an unusable OpenGL context; True to restart with compatibility rendering.

    Shared by MainWindow (the viewport's own check) and app.py's preflight,
    which runs before any window exists and so passes no parent. Modal and
    synchronous, like show_error_dialog.
    """
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Graphics problem")
    box.setText("Bermake cannot draw the 3D view on this computer.")
    if not offer_restart:
        box.setInformativeText(
            f"{message}\n\nDetails are in the log. Help > About Bermake shows where it is."
        )
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        return False
    box.setInformativeText(
        f"{message}\n\nBermake can restart using compatibility rendering, which works "
        "on almost any computer but may be slower."
    )
    restart = box.addButton(
        "Restart using compatibility rendering", QMessageBox.ButtonRole.AcceptRole
    )
    box.addButton(QMessageBox.StandardButton.Close)
    box.exec()
    return box.clickedButton() is restart


def _default_instance():
    from PySide6.QtCore import QCoreApplication

    return QCoreApplication.instance()


def _default_schedule(delay_ms: int, fn: Callable[[], None]) -> None:
    from PySide6.QtCore import QTimer

    QTimer.singleShot(delay_ms, fn)


def deferred_error_dialog(
    log_dir: Path,
    *,
    instance: Callable[[], object] = _default_instance,
    schedule: Callable[[int, Callable[[], None]], None] = _default_schedule,
    show: Callable[[Path], None] = show_error_dialog,
) -> None:
    """Show the dialog from the event loop rather than from inside the failing
    call, which may be a paint event. Without an application there is nothing
    to show it on; the error is already in the log."""
    if instance() is None:
        return
    schedule(0, lambda: show(Path(log_dir)))
