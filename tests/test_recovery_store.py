"""RecoveryStore: the recovery folder (M7.12, spec 4.2 and 4.5)."""

import json
import os
from pathlib import Path

import pytest
from bermake.recovery.store import (
    RECOVERY_FORMAT,
    RECOVERY_VERSION,
    RecoverySession,
    RecoveryStore,
    SessionMeta,
)


def _meta(session_id: str = "a" * 32, **overrides) -> SessionMeta:
    values = {
        "session_id": session_id,
        "original_path": r"C:\Users\x\House.berm",
        "display_name": "House.berm",
        "saved_at": "2026-10-04T14:32:05+08:00",
        "app_version": "0.17.0",
        "pid": 12345,
        "process_started_at": "2026-10-04T13:58:41+08:00",
    }
    values.update(overrides)
    return SessionMeta(**values)


def _valid_json(session_id: str) -> str:
    return json.dumps(
        {
            "format": RECOVERY_FORMAT,
            "version": RECOVERY_VERSION,
            "session_id": session_id,
            "original_path": None,
            "display_name": "Untitled",
            "saved_at": "2026-10-04T14:32:05+08:00",
            "app_version": "0.17.0",
            "pid": 1,
            "process_started_at": "2026-10-04T13:58:41+08:00",
        }
    )


def _write_bytes(data: bytes):
    def write(path: Path) -> None:
        path.write_bytes(data)

    return write


def test_write_produces_both_files_and_round_trips(tmp_path):
    store = RecoveryStore(tmp_path / "recovery")  # folder does not exist yet
    meta = _meta()
    store.write(meta, _write_bytes(b"berm-bytes"))

    assert store.berm_path(meta.session_id) == tmp_path / "recovery" / f"{meta.session_id}.berm"
    assert store.berm_path(meta.session_id).read_bytes() == b"berm-bytes"
    raw = json.loads((tmp_path / "recovery" / f"{meta.session_id}.json").read_text("utf-8"))
    assert raw["format"] == RECOVERY_FORMAT == "bermake-recovery"
    assert raw["version"] == RECOVERY_VERSION == 1
    for key in (
        "session_id",
        "original_path",
        "display_name",
        "saved_at",
        "app_version",
        "pid",
        "process_started_at",
    ):
        assert key in raw
    sessions, quarantined = store.list_sessions()
    assert quarantined == []
    assert [s.meta for s in sessions] == [meta]
    assert isinstance(sessions[0], RecoverySession)
    assert sessions[0].berm_path == store.berm_path(meta.session_id)
    assert sessions[0].meta_path == tmp_path / "recovery" / f"{meta.session_id}.json"


def test_untitled_original_path_none_round_trips(tmp_path):
    store = RecoveryStore(tmp_path)
    meta = _meta(original_path=None, display_name="Untitled")
    store.write(meta, _write_bytes(b"x"))
    (session,), _ = store.list_sessions()
    assert session.meta == meta


def test_json_is_not_written_when_berm_fails(tmp_path):
    store = RecoveryStore(tmp_path)
    meta = _meta()

    def failing(_path: Path) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        store.write(meta, failing)
    assert not (tmp_path / f"{meta.session_id}.json").exists()


def test_json_is_written_after_write_berm_returns(tmp_path):
    store = RecoveryStore(tmp_path)
    meta = _meta()
    seen: list[bool] = []

    def write(path: Path) -> None:
        path.write_bytes(b"x")
        seen.append((tmp_path / f"{meta.session_id}.json").exists())

    store.write(meta, write)
    assert seen == [False]
    assert (tmp_path / f"{meta.session_id}.json").exists()


def test_atomic_json_write_keeps_previous_json_on_failure(tmp_path, monkeypatch):
    store = RecoveryStore(tmp_path)
    old = _meta(saved_at="2026-10-04T14:00:00+08:00")
    store.write(old, _write_bytes(b"old"))
    json_path = tmp_path / f"{old.session_id}.json"
    before = json_path.read_bytes()

    real_replace = os.replace

    def fail_on_json(src, dst):
        if str(dst).endswith(".json"):
            raise OSError("simulated crash")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_on_json)
    with pytest.raises(OSError, match="simulated crash"):
        store.write(_meta(saved_at="2026-10-04T15:00:00+08:00"), _write_bytes(b"new"))
    monkeypatch.undo()

    assert json_path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))  # temp file cleaned up
    (session,), _ = store.list_sessions()
    assert session.meta == old


def test_list_sessions_on_missing_folder_is_empty(tmp_path):
    assert RecoveryStore(tmp_path / "nope").list_sessions() == ([], [])


def test_list_sessions_quarantines_damaged_entries(tmp_path, caplog):
    store = RecoveryStore(tmp_path)
    store.write(_meta("1" * 32), _write_bytes(b"good"))

    # lone .json
    (tmp_path / ("2" * 32 + ".json")).write_text(_valid_json("2" * 32), "utf-8")
    # lone .berm
    (tmp_path / ("3" * 32 + ".berm")).write_bytes(b"orphan")
    # non-JSON .json
    (tmp_path / ("4" * 32 + ".berm")).write_bytes(b"b")
    (tmp_path / ("4" * 32 + ".json")).write_text("not json {", "utf-8")
    # wrong format
    (tmp_path / ("5" * 32 + ".berm")).write_bytes(b"b")
    wrong_format = json.loads(_valid_json("5" * 32))
    wrong_format["format"] = "something-else"
    (tmp_path / ("5" * 32 + ".json")).write_text(json.dumps(wrong_format), "utf-8")

    with caplog.at_level("WARNING"):
        sessions, quarantined = store.list_sessions()

    assert [s.meta.session_id for s in sessions] == ["1" * 32]
    assert "has no matching metadata" in caplog.text  # the lone .berm
    assert "has no matching .berm" in caplog.text  # the lone .json
    assert "unexpected recovery format" in caplog.text
    assert sorted(quarantined) == ["2" * 32, "3" * 32, "4" * 32, "5" * 32]
    names = sorted(p.name for p in tmp_path.iterdir())
    assert ("2" * 32 + ".broken.json") in names
    assert ("3" * 32 + ".broken.berm") in names
    assert ("4" * 32 + ".broken.json") in names and ("4" * 32 + ".broken.berm") in names
    assert ("5" * 32 + ".broken.json") in names and ("5" * 32 + ".broken.berm") in names
    for digit in "2345":
        assert not (tmp_path / (digit * 32 + ".json")).exists()
        assert not (tmp_path / (digit * 32 + ".berm")).exists()
    # Quarantined files are never offered again.
    assert store.list_sessions() == (sessions, [])


def test_newer_version_is_quarantined(tmp_path):
    store = RecoveryStore(tmp_path)
    sid = "6" * 32
    (tmp_path / f"{sid}.berm").write_bytes(b"b")
    data = json.loads(_valid_json(sid))
    data["version"] = RECOVERY_VERSION + 1
    (tmp_path / f"{sid}.json").write_text(json.dumps(data), "utf-8")
    sessions, quarantined = store.list_sessions()
    assert sessions == [] and quarantined == [sid]
    assert (tmp_path / f"{sid}.broken.berm").read_bytes() == b"b"


def test_json_missing_a_field_is_quarantined(tmp_path):
    store = RecoveryStore(tmp_path)
    sid = "7" * 32
    (tmp_path / f"{sid}.berm").write_bytes(b"b")
    data = json.loads(_valid_json(sid))
    del data["pid"]
    (tmp_path / f"{sid}.json").write_text(json.dumps(data), "utf-8")
    assert store.list_sessions() == ([], [sid])


def test_json_naming_a_different_session_is_quarantined(tmp_path):
    store = RecoveryStore(tmp_path)
    sid = "8" * 32
    (tmp_path / f"{sid}.berm").write_bytes(b"b")
    (tmp_path / f"{sid}.json").write_text(_valid_json("9" * 32), "utf-8")
    assert store.list_sessions() == ([], [sid])


def test_list_sessions_ignores_broken_and_tmp_files(tmp_path):
    store = RecoveryStore(tmp_path)
    (tmp_path / ("a" * 32 + ".broken.berm")).write_bytes(b"x")
    (tmp_path / ("a" * 32 + ".broken.json")).write_text("junk", "utf-8")
    (tmp_path / ("b" * 32 + ".berm.tmp")).write_bytes(b"x")
    (tmp_path / ("b" * 32 + ".json.tmp")).write_text("junk", "utf-8")
    assert store.list_sessions() == ([], [])
    assert len(list(tmp_path.iterdir())) == 4  # nothing renamed or removed


def test_delete_removes_both_files(tmp_path):
    store = RecoveryStore(tmp_path)
    meta = _meta()
    store.write(meta, _write_bytes(b"x"))
    store.delete(meta.session_id)
    assert list(tmp_path.iterdir()) == []


def test_delete_missing_session_is_a_no_op(tmp_path):
    RecoveryStore(tmp_path).delete("f" * 32)  # folder exists, files do not
    RecoveryStore(tmp_path / "absent").delete("f" * 32)  # folder does not exist


def test_delete_only_touches_its_own_session(tmp_path):
    store = RecoveryStore(tmp_path)
    store.write(_meta("a" * 32), _write_bytes(b"a"))
    store.write(_meta("b" * 32), _write_bytes(b"b"))
    store.delete("a" * 32)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["b" * 32 + ".berm", "b" * 32 + ".json"]


def test_quarantine_renames_and_never_deletes(tmp_path, caplog):
    store = RecoveryStore(tmp_path)
    meta = _meta()
    store.write(meta, _write_bytes(b"keep-me"))
    with caplog.at_level("WARNING"):
        store.quarantine(meta.session_id, "because tests")
    assert (tmp_path / f"{meta.session_id}.broken.berm").read_bytes() == b"keep-me"
    assert (tmp_path / f"{meta.session_id}.broken.json").exists()
    assert not (tmp_path / f"{meta.session_id}.berm").exists()
    assert not (tmp_path / f"{meta.session_id}.json").exists()
    assert "because tests" in caplog.text
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_quarantine_of_missing_session_does_nothing(tmp_path):
    RecoveryStore(tmp_path).quarantine("e" * 32, "nothing there")
    assert list(tmp_path.iterdir()) == []


def test_quarantine_twice_keeps_both_generations(tmp_path):
    store = RecoveryStore(tmp_path)
    sid = "c" * 32
    (tmp_path / f"{sid}.berm").write_bytes(b"first")
    store.quarantine(sid, "one")
    (tmp_path / f"{sid}.berm").write_bytes(b"second")
    store.quarantine(sid, "two")
    contents = sorted(p.read_bytes() for p in tmp_path.glob("*.berm"))
    assert contents == [b"first", b"second"]


def test_hand_over_moves_berm_and_rewrites_metadata(tmp_path):
    store = RecoveryStore(tmp_path)
    old = _meta("a" * 32)
    store.write(old, _write_bytes(b"\x00\x01recovered\xff"))
    new = _meta(
        "b" * 32,
        pid=999,
        process_started_at="2026-10-05T09:00:00+08:00",
        saved_at="2026-10-05T09:00:01+08:00",
    )

    store.hand_over(old.session_id, new)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["b" * 32 + ".berm", "b" * 32 + ".json"]
    assert store.berm_path(new.session_id).read_bytes() == b"\x00\x01recovered\xff"
    (session,), quarantined = store.list_sessions()
    assert quarantined == []
    assert session.meta == new
    assert session.meta.original_path == old.original_path
