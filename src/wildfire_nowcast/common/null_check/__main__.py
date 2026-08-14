"""``python -m wildfire_nowcast.common.null_check`` — see :mod:`.cli`."""

from __future__ import annotations

import sys

from wildfire_nowcast.common.null_check.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
