"""The packaged smoke test's checks 1 to 3, run from the dev tree (M7.9, spec 2.6).

Check 4 (rendering) runs only in the package build: the offscreen platform the
suite uses under CI cannot create an OpenGL context (spec D11). This is a
stated gap, not a skipped test.
"""

import json
from pathlib import Path

import bermake.diagnostics.smoke as smoke

ROOT = Path(__file__).resolve().parent.parent


def test_the_three_dev_tree_checks_pass(qapp, tmp_path):
    report_path = tmp_path / "report.json"

    assert smoke.run_smoke(report_path, include_rendering=False) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert [c["name"] for c in report["checks"]] == ["imports", "resources", "document"]
    assert all(c["ok"] for c in report["checks"])


def test_resource_counts_match_the_source_tree(qapp):
    result = smoke.check_resources()
    icons = len(list((ROOT / "python/bermake/ui/icons").glob("*.svg")))
    shaders = len([p for p in (ROOT / "python/bermake/viewport/shaders").iterdir() if p.is_file()])
    assert result.data == {"icons": icons, "shaders": shaders}


def test_the_document_round_trip_compares_geometry(tmp_path):
    result = smoke.check_document_roundtrip(tmp_path)
    assert result.ok, result.detail
    assert result.data == {"faces": 1, "vertices": 4}


def test_a_failing_check_fails_the_run(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "check_imports", lambda: smoke.CheckResult("imports", False, "no"))

    assert smoke.run_smoke(tmp_path / "r.json", include_rendering=False) == 1
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["ok"] is False


def test_a_check_that_raises_is_recorded_as_a_failure(qapp, tmp_path, monkeypatch):
    def explode():
        raise RuntimeError("check blew up")

    monkeypatch.setattr(smoke, "check_resources", explode)

    assert smoke.run_smoke(tmp_path / "r.json", include_rendering=False) == 1
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    failed = [c for c in report["checks"] if not c["ok"]]
    assert [c["name"] for c in failed] == ["resources"]
    assert "check blew up" in failed[0]["detail"]


def test_rendering_without_mesa_is_a_failure_not_a_skip(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "default_mesa_dir", lambda: None)

    assert smoke.run_smoke(tmp_path / "r.json") == 1
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    rendering = [c for c in report["checks"] if c["name"] == "rendering"]
    assert rendering == [{"name": "rendering", "ok": False, "detail": "Mesa not found", "data": {}}]
