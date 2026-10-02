"""The packaged smoke test's checks 1 to 3, run from the dev tree (M7.9, spec 2.6).

Checks 4 and 5 (rendering and startup_check) run only in the package build: the
offscreen platform the suite uses under CI cannot create an OpenGL context
(spec D11). This is a stated gap, not a skipped test.
"""

import json
from pathlib import Path

import bermake.diagnostics.gl_preflight as gl_preflight
import bermake.diagnostics.smoke as smoke
import numpy as np
import pytest
from bermake.diagnostics.gl_check import GlInfo

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
    assert result.data == {"icons": icons, "shaders": shaders, "jpeg_decodes": True}


def test_the_resources_check_decodes_a_jpeg(qapp):
    assert smoke._jpeg_decodes() is True


def test_a_jpeg_that_does_not_decode_fails_the_resources_check(qapp, monkeypatch):
    monkeypatch.setattr(smoke, "_jpeg_decodes", lambda: False)

    result = smoke.check_resources()

    assert result.ok is False
    assert result.data["jpeg_decodes"] is False
    assert "jpeg" in result.detail


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
    gl_checks = [c for c in report["checks"] if c["name"] in ("rendering", "startup_check")]
    assert gl_checks == [
        {"name": "rendering", "ok": False, "detail": "Mesa not found", "data": {}},
        {"name": "startup_check", "ok": False, "detail": "Mesa not found", "data": {}},
    ]


def _stub_probe(monkeypatch, info):
    calls = []

    def probe(**kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr(gl_preflight, "probe_gl", probe)
    return calls


def test_startup_check_passes_on_a_good_context(monkeypatch):
    calls = _stub_probe(monkeypatch, GlInfo((4, 6), "4.6 Mesa", "llvmpipe"))
    result = smoke.check_startup_check()
    assert result.ok, result.detail
    # Strict, so a crash in the draw step fails the check rather than passing.
    assert calls == [{"strict": True}]
    assert result.data == {"version": "4.6", "renderer": "llvmpipe", "draw_error": None}


def test_startup_check_fails_when_the_draw_test_failed(monkeypatch):
    info = GlInfo((4, 6), "4.6", "bad", draw_error="a test image came back wrong")
    _stub_probe(monkeypatch, info)
    result = smoke.check_startup_check()
    assert not result.ok
    assert result.data["draw_error"] == "a test image came back wrong"


def test_startup_check_fails_without_a_context(monkeypatch):
    _stub_probe(monkeypatch, None)
    assert not smoke.check_startup_check().ok


def test_startup_check_fails_on_an_old_version(monkeypatch):
    _stub_probe(monkeypatch, GlInfo((3, 2), "3.2", "old"))
    assert not smoke.check_startup_check().ok


def test_identical_frames_have_no_changed_pixels():
    """The rendering check passes only when drawing the model changed the
    frame (final review M2); identical frames must count as no change."""
    frame = np.zeros((4 * 4, 4), dtype=np.uint8)
    assert smoke.changed_pixels(frame, frame.copy()) == 0


def test_changed_pixels_counts_pixels_not_channels():
    before = np.zeros((16, 4), dtype=np.uint8)
    after = before.copy()
    after[3] = (255, 255, 255, 255)
    after[7, 1] = 9
    assert smoke.changed_pixels(before, after) == 2


def test_frames_of_different_sizes_are_refused():
    with pytest.raises(ValueError, match="size"):
        smoke.changed_pixels(np.zeros((4, 4), np.uint8), np.zeros((8, 4), np.uint8))
