"""Earth Engine backed sources for C1 channels 1-13.

Nothing in this package imports ``ee`` at module import time, so the rest of the
data package stays importable on a machine with no Earth Engine credentials at
all. That is the property the whole test suite depends on.

Two standing constraints, both enforced in :mod:`.gee`:

* The Cloud project id is read from ``$WILDFIRE_GEE_PROJECT``. It is never
  hardcoded, never defaulted, and never written into a committed file. This is
  clause C7 in ``docs/interfaces.md``, which states it together with the scopes
  these credentials actually hold.
* The transport rule is NOT restated here, because a second copy of a rule is a
  second thing to go stale: :mod:`.gee` owns it, in its module docstring and in
  ``DEFAULT_EXPORT_TARGET``. An earlier version of this paragraph said exports
  "default to Drive", which was already false when it was read - the default is
  a synchronous chunked ``computePixels`` fetch, and ``Export.*.toDrive`` is not
  authorised for these scopes at all.
"""

from __future__ import annotations

__all__: list[str] = []
