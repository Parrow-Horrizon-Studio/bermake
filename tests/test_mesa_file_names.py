"""The Mesa file names are spelled in four places; nothing else ties them
together, so a rename would surface only in the frozen smoke test at the end of
a full package build (#139)."""

import importlib.util
import re
from pathlib import Path

from bermake.diagnostics import compat_rendering

ROOT = Path(__file__).resolve().parent.parent


def _fetch_mesa_bundled_names():
    spec = importlib.util.spec_from_file_location("fetch_mesa", ROOT / "tools" / "fetch_mesa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return set(module.MESA_FILES.values())


def test_fetch_mesa_the_runtime_and_the_spec_agree_on_the_mesa_file_names():
    spec = (ROOT / "packaging" / "bermake.spec").read_text(encoding="utf-8")
    shipped = set(re.findall(r'MESA_DIR / "([^"]+)"', spec))
    named = re.search(r"^MESA_FILES = \{([^}]*)\}", spec, re.MULTILINE)
    assert named is not None, "packaging/bermake.spec no longer defines MESA_FILES"
    filtered = set(re.findall(r'"([^"]+)"', named.group(1)))

    runtime = {compat_rendering.MESA_LOADER, compat_rendering.MESA_DRIVER}
    fetched = _fetch_mesa_bundled_names()

    assert len(fetched) == 2
    assert runtime == fetched
    assert shipped == fetched
    assert filtered == fetched
