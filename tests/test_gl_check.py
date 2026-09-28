"""The OpenGL verdict (M7.9, spec 2.4)."""

import pytest
from bermake.diagnostics.gl_check import MIN_GL_VERSION, GlInfo, evaluate_gl


def _info(version, renderer="Test Renderer"):
    return GlInfo(version=version, version_string=f"{version[0]}.{version[1]}", renderer=renderer)


def test_the_minimum_is_the_version_the_shaders_declare():
    assert MIN_GL_VERSION == (3, 3)


@pytest.mark.parametrize("version", [(1, 1), (2, 1), (3, 0), (3, 2)])
def test_old_contexts_fail_naming_what_they_got(version):
    verdict = evaluate_gl(_info(version, "GDI Generic"), None)
    assert not verdict.ok
    assert f"OpenGL {version[0]}.{version[1]}" in verdict.message
    assert "GDI Generic" in verdict.message
    assert "3.3" in verdict.message


@pytest.mark.parametrize("version", [(3, 3), (4, 1), (4, 6)])
def test_new_enough_contexts_pass(version):
    assert evaluate_gl(_info(version), None).ok


def test_a_shader_failure_fails_with_only_its_first_line():
    error = "vertex shader compile failed:\n0:12(3): error: something long"
    verdict = evaluate_gl(_info((4, 6), "Odd Driver"), error)
    assert not verdict.ok
    assert "vertex shader compile failed:" in verdict.message
    assert "something long" not in verdict.message
    assert "Odd Driver" in verdict.message


def test_an_empty_shader_error_still_fails():
    verdict = evaluate_gl(_info((4, 6)), "")
    assert not verdict.ok
    assert "unknown error" in verdict.message
