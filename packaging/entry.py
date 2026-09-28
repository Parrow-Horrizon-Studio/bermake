"""PyInstaller's entry script for Bermake.

Kept outside python/bermake deliberately: PyInstaller puts the entry script's
folder on the module search path during analysis, and python/bermake contains
an `io` package that would shadow the standard library's.
"""

import sys

from bermake.app import main

sys.exit(main())
