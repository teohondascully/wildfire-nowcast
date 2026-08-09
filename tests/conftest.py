"""Shared fixtures, and the hook that lets the contract suite target ANY tensor.

By default the C1/C2/C3 suite runs against a freshly generated synthetic fire.
Point it at a real one instead::

    .venv/bin/pytest tests/test_contracts.py \\
        --tensor-path data/fires/2019_kincade/tensor.zarr

``--manifest-path`` and ``--norm-stats-path`` default to ``manifest.json`` beside
the store and to whatever the manifest's ``norm_stats_path`` resolves to.
``--labels-only`` relaxes *completeness* only (for ADR-003 interim label stores);
every grid, time and state rule is still enforced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import xarray as xr

from wildfire_nowcast.common import zarr_io as zio
from wildfire_nowcast.common.synthetic import SyntheticFire, make_synthetic_fire

_OPT_TENSOR = "--tensor-path"
_OPT_MANIFEST = "--manifest-path"
_OPT_NORM = "--norm-stats-path"
_OPT_LABELS_ONLY = "--labels-only"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("wildfire", "wildfire-nowcast contract checks")
    group.addoption(
        _OPT_TENSOR,
        action="store",
        default=None,
        metavar="PATH",
        help="run the C1/C2/C3 suite against this tensor.zarr instead of a synthetic fire",
    )
    group.addoption(
        _OPT_MANIFEST,
        action="store",
        default=None,
        metavar="PATH",
        help="manifest.json to check (default: alongside the tensor)",
    )
    group.addoption(
        _OPT_NORM,
        action="store",
        default=None,
        metavar="PATH",
        help="norm_stats.json to check (default: from the manifest)",
    )
    group.addoption(
        _OPT_LABELS_ONLY,
        action="store_true",
        default=False,
        help="only require the fire_state channel (interim label stores, ADR-003)",
    )


@pytest.fixture(scope="session")
def default_synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticFire:
    """One synthetic fire, generated once per session."""
    out = tmp_path_factory.mktemp("synthetic") / "tensor.zarr"
    return make_synthetic_fire(seed=20190901, n_hours=24, out=out)


@pytest.fixture(scope="session")
def using_custom_target(request: pytest.FixtureRequest) -> bool:
    """True when the suite was pointed at a caller-supplied tensor."""
    return request.config.getoption(_OPT_TENSOR) is not None


@pytest.fixture(scope="session")
def labels_only(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption(_OPT_LABELS_ONLY))


@pytest.fixture(scope="session")
def tensor_path(request: pytest.FixtureRequest, default_synthetic: SyntheticFire) -> Path:
    """The tensor store under test."""
    supplied = request.config.getoption(_OPT_TENSOR)
    if supplied:
        path = Path(supplied).expanduser().resolve()
        if not path.exists():
            pytest.fail(f"{_OPT_TENSOR} {path} does not exist")
        return path
    return default_synthetic.tensor_path


@pytest.fixture(scope="session")
def manifest_path(request: pytest.FixtureRequest, tensor_path: Path) -> Path:
    supplied = request.config.getoption(_OPT_MANIFEST)
    return (
        Path(supplied).expanduser().resolve() if supplied else tensor_path.parent / "manifest.json"
    )


@pytest.fixture(scope="session")
def norm_stats_path(request: pytest.FixtureRequest, manifest_path: Path) -> Path:
    supplied = request.config.getoption(_OPT_NORM)
    if supplied:
        return Path(supplied).expanduser().resolve()
    if manifest_path.is_file():
        declared = json.loads(manifest_path.read_text()).get("norm_stats_path")
        if isinstance(declared, str) and declared:
            candidate = Path(declared)
            if not candidate.is_absolute():
                candidate = manifest_path.parent / candidate
            return candidate.resolve()
    return manifest_path.parent / "norm_stats.json"


@pytest.fixture(scope="session")
def tensor_ds(tensor_path: Path) -> xr.Dataset:
    """The tensor under test, opened once."""
    return zio.open_tensor(tensor_path)


@pytest.fixture(scope="session")
def synthetic_ds(default_synthetic: SyntheticFire) -> xr.Dataset:
    """Always the synthetic fire, regardless of --tensor-path."""
    return zio.open_tensor(default_synthetic.tensor_path)


# --------------------------------------------------------------------------
# ADR-030 / A13 — one mutation-coverage report per playthrough per session
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _playthrough_reports() -> dict[str, object]:
    return {}


@pytest.fixture(scope="session")
def playthrough_report(_playthrough_reports: dict[str, object]):
    """Run a :class:`~wildfire_nowcast.common.playthrough.Playthrough` ONCE per session.

    The protocol costs ``(1 + n_defects)`` observations, and both the owning test
    file and ``tests/test_playthrough_registry.py`` want the same report. Without
    this the dispersion playthrough is scored twice for identical numbers, which
    is the same waste A12 removed from the null-check fixture.

    Memoised BY NAME and only within one session. Stated because A12 refused a
    disk cache for a good reason — *a cache that misses a dependency makes the
    controls pass on stale numbers* — and this is deliberately the weaker,
    safe version: nothing is persisted, nothing survives the process, and a
    playthrough is a pure function of code that cannot change mid-session.
    """
    from wildfire_nowcast.common import playthrough as pt_module

    def get(playthrough):
        key = playthrough.name
        if key not in _playthrough_reports:
            _playthrough_reports[key] = pt_module.run(playthrough)
        return _playthrough_reports[key]

    return get
