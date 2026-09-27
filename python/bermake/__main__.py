"""Allows `python -m bermake` to launch the application."""

import sys

from bermake.app import main

sys.exit(main())
