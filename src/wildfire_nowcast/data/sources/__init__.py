"""Earth Engine backed sources for C1 channels 1-13.

Every module here is written but **not yet executed**: Earth Engine
authentication is human-gated (ADR-002) and blocked at the time of writing (see
docs/decisions.md). Nothing in this package imports ``ee`` at module
import time, so the rest of the data package stays usable while auth is
outstanding.

Two standing constraints, both enforced in :mod:`.gee`:

* The Cloud project id is read from ``$WILDFIRE_GEE_PROJECT``. It is never
  hardcoded, never defaulted, and never written into a committed file.
* Exports default to **Drive**, not Cloud Storage. Earth Engine compute on a
  noncommercial-registered project is free, but a GCS bucket bills for storage
  and egress independently of Earth Engine — so ``toCloudStorage`` is the one
  route that can put real charges on a project with live billing. The target is
  configurable (``$WILDFIRE_GEE_EXPORT_TARGET``) so this can be revisited
  without a rewrite, and the bucket name likewise comes from the environment.
* Region-sized work goes through ``ee.batch.Export.*`` (batch), never through
  ``getInfo()`` on a large computation. Batch EECU is cheaper and is the correct
  pattern regardless of billing state.
"""

from __future__ import annotations

__all__: list[str] = []
