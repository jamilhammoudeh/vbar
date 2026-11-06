"""`python -m vbar`, the `vbar` command where the scripts folder is not on PATH."""

import sys

from .cli import main

sys.exit(main())
