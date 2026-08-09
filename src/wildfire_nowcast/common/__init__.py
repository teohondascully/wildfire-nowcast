"""Shared infrastructure: contracts, grid, io, config, run directories.

Anything two leads both need lives here, once. The submodules are:

``contract``  the executable form of INTERFACES.md C1/C2/C3 (constants + checks)
``states``    C1.1 ``fireline_v2`` — the ONE implementation of the state rule
``grid``      the EPSG:5070 1 km grid and its coordinate conventions
``zarr_io``   building, writing and reading C1/C2/C3 artifacts
``derive``    documented formulas for the derived channels (6-8, 11)
``splits``    C8 split fingerprint + C3.1 cross-fire fold checks
``synthetic`` C4, the synthetic fire generator
``config``    yaml config loading and composition (C7)
``runs``      ``runs/{run_id}/`` with resolved config + git SHA (C7)
``paths``     repository path resolution; no hardcoded paths in src/

Per C0 (ADR-007) anything the contract adjudicates has exactly one
implementation and it lives here. In particular the perimeter -> ``fire_state``
rule is ``wildfire_nowcast.common.states.apply_state_rule``; no other module may
re-derive it.

The convenience names below are resolved lazily (PEP 562). Importing a
submodule eagerly here would both drag xarray/pyproj into every import of this
package and make ``python -m wildfire_nowcast.common.contract`` emit a runpy
double-import warning — and that CLI is the contract-checking entry point other
leads use, so it needs to be quiet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from wildfire_nowcast.common.contract import (
        CELL_SIZE_M,
        CHANNEL_INDEX,
        CHANNEL_INDEX_OFFSET,
        CHANNELS,
        CRS_STRING,
        FEATURE_CHANNELS,
        N_CHANNELS,
        ContractReport,
        check_all,
        check_manifest,
        check_norm_stats,
        check_tensor,
    )
    from wildfire_nowcast.common.grid import Grid
    from wildfire_nowcast.common.splits import (
        assert_split_unchanged,
        check_run_split,
        check_split_assignment,
        split_fingerprint,
    )
    from wildfire_nowcast.common.states import apply_state_rule, fireline_v2

__all__ = [
    "CELL_SIZE_M",
    "CHANNELS",
    "CHANNEL_INDEX",
    "CHANNEL_INDEX_OFFSET",
    "CRS_STRING",
    "ContractReport",
    "FEATURE_CHANNELS",
    "Grid",
    "N_CHANNELS",
    "apply_state_rule",
    "assert_split_unchanged",
    "check_all",
    "check_manifest",
    "check_norm_stats",
    "check_run_split",
    "check_split_assignment",
    "check_tensor",
    "fireline_v2",
    "split_fingerprint",
]

_MODULE_OF: dict[str, str] = {
    "Grid": "grid",
    "apply_state_rule": "states",
    "fireline_v2": "states",
    "assert_split_unchanged": "splits",
    "check_run_split": "splits",
    "check_split_assignment": "splits",
    "split_fingerprint": "splits",
}
_LAZY: dict[str, str] = {name: _MODULE_OF.get(name, "contract") for name in __all__}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module_name}"), name)


def __dir__() -> list[str]:
    return sorted(__all__)
