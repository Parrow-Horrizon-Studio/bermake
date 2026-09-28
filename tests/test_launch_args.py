"""Command line parsing for Bermake's own flags (M7.9)."""

from pathlib import Path

from bermake.diagnostics.launch import COMPAT_FLAG, NO_COMPAT_FLAG, SMOKE_FLAG, parse_launch_args


def test_no_flags():
    args = parse_launch_args(["Bermake.exe"])
    assert (args.compatibility_rendering, args.no_compatibility_rendering) == (False, False)
    assert args.smoke_report is None
    assert args.qt_argv == ["Bermake.exe"]


def test_the_compatibility_flag_is_consumed():
    args = parse_launch_args(["Bermake.exe", COMPAT_FLAG])
    assert args.compatibility_rendering is True
    assert args.qt_argv == ["Bermake.exe"]


def test_the_smoke_flag_takes_a_report_path():
    args = parse_launch_args(["Bermake.exe", SMOKE_FLAG, "C:/out/report.json"])
    assert args.smoke_report == Path("C:/out/report.json")


def test_qt_arguments_and_paths_pass_through_untouched():
    args = parse_launch_args(
        ["Bermake.exe", "-style", "fusion", COMPAT_FLAG, "C:/models/my house.berm"]
    )
    assert args.compatibility_rendering is True
    assert args.qt_argv == ["Bermake.exe", "-style", "fusion", "C:/models/my house.berm"]


def test_an_abbreviated_flag_is_not_mistaken_for_ours():
    args = parse_launch_args(["Bermake.exe", "--compat"])
    assert args.compatibility_rendering is False
    assert args.qt_argv == ["Bermake.exe", "--compat"]


def test_the_no_compatibility_flag_is_consumed():
    args = parse_launch_args(["Bermake.exe", NO_COMPAT_FLAG, "C:/m/a.berm"])
    assert args.no_compatibility_rendering is True
    assert args.compatibility_rendering is False
    assert args.qt_argv == ["Bermake.exe", "C:/m/a.berm"]


def test_both_compatibility_flags_are_both_recorded():
    """Parsing keeps both; app.wants_compatibility_rendering decides."""
    args = parse_launch_args(["Bermake.exe", COMPAT_FLAG, NO_COMPAT_FLAG])
    assert (args.compatibility_rendering, args.no_compatibility_rendering) == (True, True)
    assert args.qt_argv == ["Bermake.exe"]


def test_an_abbreviated_no_flag_is_not_mistaken_for_ours():
    args = parse_launch_args(["Bermake.exe", "--no-compat"])
    assert args.no_compatibility_rendering is False
    assert args.qt_argv == ["Bermake.exe", "--no-compat"]
