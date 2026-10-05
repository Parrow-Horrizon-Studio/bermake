"""Where autosaves live (M7.12, spec 4.2, D1).

Callers resolve the folder through this module's attribute,
`paths.default_recovery_directory()`, never by importing the function by name.
The test suite's guard (tests/conftest.py) patches exactly this one attribute
so no test writes to the real folder, and a name imported elsewhere would
escape it.
"""

from __future__ import annotations

from pathlib import Path


def default_recovery_directory() -> Path:
    """`<AppLocalDataLocation>/recovery`, resolved like `default_log_directory()`.

    With the organisation and application names app.py sets, this is
    `%LOCALAPPDATA%/Parrow Horrizon Studio/Bermake/recovery` on Windows. The
    folder is not created here; `RecoveryStore.write` creates it on first use.
    """
    from PySide6.QtCore import QStandardPaths

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    return Path(base) / "recovery"
