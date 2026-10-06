"""The startup recovery dialog (M7.12, #77; spec 4.5).

Shown modal over the main window when Bermake finds work autosaved by a
Bermake that is no longer running. `exec()` returns a `RecoveryChoice`
instead of a dialog code, so the caller never maps buttons itself.

Closing the dialog any way other than its buttons (Escape, the title bar's
close box) is Decide later: the work is kept, never discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from bermake.recovery.store import RecoverySession

RECOVER = "recover"
DISCARD = "discard"
LATER = "later"

# English month names regardless of the system locale: the rest of the UI is
# English, and strftime("%b") would follow the C runtime's locale.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class RecoveryChoice:
    action: str  # RECOVER, DISCARD or LATER
    session: RecoverySession | None  # None for LATER


def describe_saved_at(saved_at: str, now: datetime | None = None) -> str:
    """Return "today at 14:32", "yesterday at 14:32" or "4 Oct 2026 at 14:32",
    in local time. Never raises: an unusable timestamp is shown as written."""
    try:
        moment = datetime.fromisoformat(saved_at).astimezone()
        today = (now or datetime.now().astimezone()).astimezone().date()
        clock = f"{moment.hour:02d}:{moment.minute:02d}"
        if moment.date() == today:
            return f"today at {clock}"
        if moment.date() == today - timedelta(days=1):
            return f"yesterday at {clock}"
        return f"{moment.day} {_MONTHS[moment.month - 1]} {moment.year} at {clock}"
    except (ValueError, OverflowError, OSError):
        return saved_at


class RecoveryDialog(QDialog):
    """Recover, Discard or Decide later, for one session or a list of them.

    `sessions` must be non-empty and is shown in the order given
    (`recoverable_sessions` returns them newest first). `now` is for tests.
    """

    def __init__(
        self,
        sessions: list[RecoverySession],
        parent: QWidget | None = None,
        *,
        now: datetime | None = None,
    ) -> None:
        super().__init__(parent)
        if not sessions:
            raise ValueError("RecoveryDialog needs at least one session")
        self._sessions = list(sessions)
        self._choice = RecoveryChoice(LATER, None)
        self.setWindowTitle("Recover Work")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        message = QLabel(self)
        message.setObjectName("RecoveryMessage")
        message.setWordWrap(True)
        layout.addWidget(message)

        self._list: QListWidget | None = None
        if len(self._sessions) == 1:
            meta = self._sessions[0].meta
            message.setText(
                f"Bermake closed unexpectedly. Recover {meta.display_name}, "
                f"last autosaved {describe_saved_at(meta.saved_at, now)}?"
            )
        else:
            message.setText(
                "Bermake closed unexpectedly. Choose the autosaved work to recover. "
                "The rest is kept for next time."
            )
            self._list = QListWidget(self)
            self._list.setObjectName("RecoverySessions")
            self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            for session in self._sessions:
                meta = session.meta
                self._list.addItem(
                    f"{meta.display_name}, last autosaved {describe_saved_at(meta.saved_at, now)}"
                )
                if meta.original_path:
                    self._list.item(self._list.count() - 1).setToolTip(meta.original_path)
            self._list.setCurrentRow(0)
            self._list.itemSelectionChanged.connect(self._update_buttons)
            layout.addWidget(self._list)

        row = QHBoxLayout()
        row.addStretch(1)
        self._recover = QPushButton("Recover", self)
        self._discard = QPushButton("Discard", self)
        self._later = QPushButton("Decide later", self)
        for button in (self._recover, self._discard, self._later):
            button.setAutoDefault(False)
            row.addWidget(button)
        self._recover.setDefault(True)
        layout.addLayout(row)

        self._recover.clicked.connect(lambda: self._choose(RECOVER))
        self._discard.clicked.connect(lambda: self._choose(DISCARD))
        self._later.clicked.connect(self.reject)
        self._update_buttons()

    def _selected(self) -> RecoverySession | None:
        if self._list is None:
            return self._sessions[0]
        if not self._list.selectedItems():
            return None
        row = self._list.currentRow()
        return self._sessions[row] if 0 <= row < len(self._sessions) else None

    def _update_buttons(self) -> None:
        has_selection = self._selected() is not None
        self._recover.setEnabled(has_selection)
        self._discard.setEnabled(has_selection)

    def _choose(self, action: str) -> None:
        session = self._selected()
        if session is None:
            return
        self._choice = RecoveryChoice(action, session)
        self.accept()

    def exec(self) -> RecoveryChoice:  # type: ignore[override]
        super().exec()
        return self._choice
