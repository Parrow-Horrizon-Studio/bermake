"""The recovery folder: one `<id>.berm` plus one `<id>.json` per open document.

The `.berm` is written first, by the caller's `write_berm` (the existing
`save_document`, which is already atomic). The `.json` metadata is written
last and atomically, so a `.json` always describes a complete `.berm`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

_log = logging.getLogger(__name__)

RECOVERY_FORMAT = "bermake-recovery"
RECOVERY_VERSION = 1

_BERM = ".berm"
_JSON = ".json"
_BROKEN = ".broken."
_TMP = ".tmp"

# A live Bermake briefly leaves a lone .berm or .json between its two writes
# (first autosave, hand_over, delete). A second Bermake listing the folder must
# not quarantine that, so lone files younger than this are skipped, not
# quarantined and not offered.
LONE_FILE_GRACE_SECONDS = 60.0


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    original_path: str | None
    display_name: str
    saved_at: str  # ISO 8601 with offset, e.g. "2026-10-04T14:32:05+08:00"
    app_version: str
    pid: int
    process_started_at: str  # ISO 8601 with offset


@dataclass(frozen=True)
class RecoverySession:
    meta: SessionMeta
    berm_path: Path
    meta_path: Path


_META_FIELDS = tuple(f.name for f in fields(SessionMeta))


def _meta_to_json(meta: SessionMeta) -> str:
    data = {"format": RECOVERY_FORMAT, "version": RECOVERY_VERSION, **asdict(meta)}
    return json.dumps(data, indent=2)


def _meta_from_json(text: str) -> SessionMeta:
    """Parse and validate a metadata document; raises ValueError if unusable."""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("recovery metadata is not a JSON object")
    if data.get("format") != RECOVERY_FORMAT:
        raise ValueError(f"unexpected recovery format {data.get('format')!r}")
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"unsupported recovery version {version!r}")
    if not 1 <= version <= RECOVERY_VERSION:
        raise ValueError(f"unsupported recovery version {version!r}")
    missing = [name for name in _META_FIELDS if name not in data]
    if missing:
        raise ValueError(f"recovery metadata is missing {', '.join(missing)}")
    original_path = data["original_path"]
    if original_path is not None and not isinstance(original_path, str):
        raise ValueError("original_path must be a string or null")
    for name in ("session_id", "display_name", "saved_at", "app_version", "process_started_at"):
        if not isinstance(data[name], str):
            raise ValueError(f"{name} must be a string")
    pid = data["pid"]
    if not isinstance(pid, int) or isinstance(pid, bool):
        raise ValueError("pid must be an integer")
    if pid <= 0:
        raise ValueError(f"pid must be positive, not {pid}")
    for name in ("saved_at", "process_started_at"):
        try:
            # ValueError if it does not parse. The timestamp is what the startup
            # sort and the liveness check use, and it raises OSError or
            # OverflowError for an out-of-range date (for example year 1).
            datetime.fromisoformat(data[name]).timestamp()
        except (OSError, OverflowError) as exc:
            raise ValueError(f"{name} is out of range: {data[name]!r}") from exc
    return SessionMeta(**{name: data[name] for name in _META_FIELDS})


class RecoveryStore:
    def __init__(self, folder: Path, clock: Callable[[], float] = time.time) -> None:
        self.folder = Path(folder)
        self._clock = clock  # wall-clock seconds; injectable for tests

    def berm_path(self, session_id: str) -> Path:
        return self.folder / f"{session_id}{_BERM}"

    def _meta_path(self, session_id: str) -> Path:
        return self.folder / f"{session_id}{_JSON}"

    def write(self, meta: SessionMeta, write_berm: Callable[[Path], None]) -> None:
        """Create the folder if needed; call write_berm(<id>.berm) (save_document is
        already atomic); then write <id>.json atomically (temp file + os.replace).
        The JSON is written only after write_berm returns. Raises OSError on failure."""
        self.folder.mkdir(parents=True, exist_ok=True)
        write_berm(self.berm_path(meta.session_id))
        self._write_meta(meta)

    def _write_meta(self, meta: SessionMeta) -> None:
        target = self._meta_path(meta.session_id)
        tmp = target.with_name(target.name + _TMP)
        try:
            tmp.write_text(_meta_to_json(meta), encoding="utf-8")
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                tmp.unlink()

    def list_sessions(self) -> tuple[list[RecoverySession], list[str]]:
        """(valid sessions, ids that were quarantined). A .json without its .berm,
        a .berm without its .json, unreadable or wrong-format JSON: quarantined,
        never returned. Ignores *.broken.* and *.tmp files."""
        if not self.folder.is_dir():
            return [], []
        json_ids: set[str] = set()
        berm_ids: set[str] = set()
        for entry in self.folder.iterdir():
            name = entry.name
            if _BROKEN in name or not entry.is_file():  # *.tmp never matches .berm/.json
                continue
            if name.endswith(_JSON):
                json_ids.add(name[: -len(_JSON)])
            elif name.endswith(_BERM):
                berm_ids.add(name[: -len(_BERM)])

        sessions: list[RecoverySession] = []
        quarantined: list[str] = []
        for session_id in sorted(json_ids | berm_ids):
            if session_id not in berm_ids or session_id not in json_ids:
                lone = (
                    self._meta_path(session_id)
                    if session_id in json_ids
                    else self.berm_path(session_id)
                )
                if self._is_recent(lone):
                    continue  # possibly a live writer between its two writes
            if session_id not in berm_ids:
                self.quarantine(session_id, "metadata has no matching .berm")
                quarantined.append(session_id)
                continue
            if session_id not in json_ids:
                self.quarantine(session_id, ".berm has no matching metadata")
                quarantined.append(session_id)
                continue
            meta_path = self._meta_path(session_id)
            try:
                meta = _meta_from_json(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                # json.JSONDecodeError and UnicodeDecodeError are ValueErrors.
                self.quarantine(session_id, f"unreadable metadata: {exc}")
                quarantined.append(session_id)
                continue
            if meta.session_id != session_id:
                self.quarantine(
                    session_id, f"metadata names session {meta.session_id!r}, not {session_id!r}"
                )
                quarantined.append(session_id)
                continue
            sessions.append(RecoverySession(meta, self.berm_path(session_id), meta_path))
        return sessions, quarantined

    def _is_recent(self, path: Path) -> bool:
        try:
            age = self._clock() - path.stat().st_mtime
        except OSError:
            return True  # vanished while we looked: someone else is working on it
        return age < LONE_FILE_GRACE_SECONDS

    def delete(self, session_id: str) -> None:
        for path in (self.berm_path(session_id), self._meta_path(session_id)):
            path.unlink(missing_ok=True)

    def quarantine(self, session_id: str, reason: str) -> None:
        """Rename <id>.berm -> <id>.broken.berm and <id>.json -> <id>.broken.json
        (whichever exist), never delete; log the reason at WARNING."""
        _log.warning("Quarantining recovery session %s: %s", session_id, reason)
        sources = [
            self.folder / f"{session_id}{suffix}"
            for suffix in (_BERM, _JSON)
            if (self.folder / f"{session_id}{suffix}").exists()
        ]

        def target_for(source: Path, n: int) -> Path:
            tag = _BROKEN[:-1] if n == 1 else f"{_BROKEN}{n}"
            return self.folder / f"{session_id}{tag}{source.suffix}"

        # One generation number for the whole pass, so the .berm and .json of a
        # session always carry the same suffix; never overwrite an earlier one.
        n = 1
        while any(
            target_for(src, n).exists()
            for src in (self.folder / f"{session_id}{x}" for x in (_BERM, _JSON))
        ):
            n += 1
        for source in sources:
            target = target_for(source, n)
            try:
                os.replace(source, target)
            except OSError:
                _log.warning("Could not quarantine %s", source, exc_info=True)

    def hand_over(self, old_id: str, new_meta: SessionMeta) -> None:
        """Rename <old>.berm to <new>.berm and write <new>.json from new_meta;
        remove <old>.json. Used by Recover (spec 4.5)."""
        os.replace(self.berm_path(old_id), self.berm_path(new_meta.session_id))
        self._write_meta(new_meta)
        self._meta_path(old_id).unlink(missing_ok=True)
