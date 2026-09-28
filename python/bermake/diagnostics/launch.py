"""Bermake's own command line flags (M7.9).

Everything that is not one of these flags passes through untouched to
QApplication, which consumes Qt's own arguments (-style and friends).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

COMPAT_FLAG = "--compatibility-rendering"
# The way back when compatibility rendering itself stops Bermake starting:
# ignores the stored preference for one launch, and wins over COMPAT_FLAG.
NO_COMPAT_FLAG = "--no-compatibility-rendering"
SMOKE_FLAG = "--smoke-test"


@dataclass(frozen=True)
class LaunchArgs:
    compatibility_rendering: bool
    no_compatibility_rendering: bool
    smoke_report: Path | None
    qt_argv: list[str]


def parse_launch_args(argv: list[str]) -> LaunchArgs:
    # allow_abbrev=False: "--compat" must not silently mean our flag.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(COMPAT_FLAG, action="store_true", dest="compatibility_rendering")
    parser.add_argument(NO_COMPAT_FLAG, action="store_true", dest="no_compatibility_rendering")
    parser.add_argument(SMOKE_FLAG, type=Path, default=None, dest="smoke_report")
    known, rest = parser.parse_known_args(argv[1:])
    return LaunchArgs(
        compatibility_rendering=known.compatibility_rendering,
        no_compatibility_rendering=known.no_compatibility_rendering,
        smoke_report=known.smoke_report,
        qt_argv=[argv[0], *rest],
    )
