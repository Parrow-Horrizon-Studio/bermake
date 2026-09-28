"""The dialog a tester sees when Bermake hits an uncaught error (M7.9, spec 2.4)."""

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
