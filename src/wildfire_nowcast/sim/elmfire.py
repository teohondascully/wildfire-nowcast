"""ELMFIRE behind the C5 ``predict()`` signature - **NATIVE INPUTS, CONTRACT
OUTPUTS** (ADR-026 (3)).

"ELMFIRE Monte Carlo with default Rothermel parameters" is a non-negotiable
baseline of this project (``README.md``, *baselines*). It is not something the
model gets to opt out of comparing against, however it scores.
This module makes it callable exactly like every other
C5 model, so G5 can be scored through C6 on the same windows, members and masks
as the kernel and the ellipse. **The head-to-head is G5 and is NOT authorised
here.** Nothing in this file produces a comparison.

WHAT CHANGED, AND WHY IT HAD TO
-------------------------------
The original playbook mapped OUR 1 km tensor into ELMFIRE's inputs. Measured
consequence (ADR-025 (4)): **+2 cells against truth's +54** on the
highest-growth Kincade window, 2 distinct members of 6 - C6.2-degenerate
territory, which VOIDS G5 rather than losing it. ADR-026 (3) ruled that the
cause was the SPEC, not ELMFIRE: it is built for ~30 m LANDFIRE and 1 km fuel
channels lobotomise it.

    ELMFIRE now consumes **30 m LANDFIRE** (fuels, canopy AND topography) at its
    NATIVE resolution, with **crown fire ON**, and only its OUTPUT ensemble is
    coarsened to the C1 1 km lattice by :mod:`wildfire_nowcast.sim.coarsen` and
    wrapped in C5. Each model gets its best inputs; the comparison happens on the
    same truth.

The install, the provably-output-neutral ``PARSE_MAP_INFO`` patch and the 4/4
bitwise determinism from S3 all survive; only the input path changed.

HOW THE C5 CONTRACT IS STILL HONOURED
-------------------------------------
``predict(x0, static, weather, n_members, horizon_h, seed)`` is byte-for-byte the
signature the kernel and the ellipse implement, and the returned array is
``uint8[n_members, horizon_h, H, W]`` on the C1 grid. Two honest notes:

* geography is bound at CONSTRUCTION (``ElmfireNativeModel(grid=..., fire_year=
  ...)``), because C5's signature carries no CRS. Same pattern as
  ``load_model(path)``: the checkpoint is bound before ``predict``, not inside it.
* in ``InputMode.NATIVE`` the ``static`` argument is **not consumed** - fuels and
  terrain come from LANDFIRE at 30 m. It is still shape-validated, and the run
  JSON records ``static_consumed: false`` so this can never be a silent
  divergence. ``InputMode.LOBOTOMISED`` consumes it, because that mode exists to
  reproduce the degenerate S3 configuration as a negative control.

WEATHER
-------
``weather`` is C1 channels 1-4 + 11, which ARE RTMA. RTMA is natively **2.5 km**
and our lattice is 1 km, so putting RTMA on the C1 grid was an UPSAMPLING and
destroyed no information - which is why "raw RTMA" and "our weather channels" are
the same field here, and why no Earth Engine call is needed for the weather half
of ADR-026 (3). ELMFIRE reads weather rasters on their own grid with their own
cell size and interpolates (``GET_BILINEAR_INTERPOLATE_COEFFS``), so the weather
rasters are written at 1 km while the analysis grid is 30 m.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args
from wildfire_nowcast.sim.c5 import STATIC_C5, WEATHER_C5
from wildfire_nowcast.sim.coarsen import (
    COARSENING_RULE,
    DEFAULT_REFINE,
    OCCUPANCY_THRESHOLD,
    coarsen_occupancy,
    fine_grid,
)
from wildfire_nowcast.sim.landfire import NativeStack, fetch_native_stack

# ADR-103: a logger, and NOTHING else at import. `main` configures. This module
# shells out to a Fortran program whose wall-clock abort is SILENT, so what it
# has to say about a run is exactly the kind of thing that must not be a print
# inside a library.
logger = logging.getLogger(__name__)

__all__ = [
    "BUILD_NOTES",
    "MAPPING_COMPROMISES",
    "ElmfireNotInstalled",
    "InputMode",
    "ElmfireConfig",
    "ElmfireNativeModel",
    "find_binary",
    "build",
    "write_envi_bsq",
    "read_bil",
    "window_grids",
    "main",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = REPO_ROOT / "vendor" / "elmfire"

BUILD_NOTES: dict[str, str] = {
    "upstream": "https://github.com/lautenberger/elmfire",
    "vendored_at": "vendor/elmfire (OUTSIDE src/, per the P2 charter)",
    "compiler": "Homebrew gfortran + open-mpi (mpifort)",
    "delta_1": "drop `-unroll` at LINK time; Apple ld reads it as `-u nroll`",
    "delta_2": "add -L$(xcrun --show-sdk-path)/usr/lib so ld finds libSystem",
    "delta_3": (
        "no source file is modified. GDAL is NOT required: we write ENVI BSQ + .hdr "
        "directly and ELMFIRE's READ_BSQ_RASTER skips its gdal_translate shell-out "
        "when both files already exist."
    ),
    "patch_1": (
        "vendor/patches/0001-parse_map_info-uninitialised-num_values.patch — "
        "PROVABLY OUTPUT-NEUTRAL (every output of the subroutine is overwritten by "
        "its caller); without it ELMFIRE hangs >10 min under USE_BSQ_XML_HEADER=.FALSE."
    ),
}

#: Every place our world and ELMFIRE's do not line up, with a DIRECTION of bias.
#: An unfair baseline invalidates G5 in either direction, so these are data, not
#: caveats to be recalled later. Emitted into every run's JSON.
MAPPING_COMPROMISES: list[dict[str, str]] = [
    {
        "field": "fine lattice",
        "issue": "30 m does not divide 1000 m, so a true-30 m grid is not nested",
        "choice": f"exactly nested refine={DEFAULT_REFINE} -> 30.303 m (1.01% off native)",
        "bias": (
            "none directional. The alternative puts area conservation inside "
            "partial-overlap weights, where an area loss can hide."
        ),
    },
    {
        "field": "output coarsening",
        "issue": "C5 must return uint8 fire_state on the 1 km C1 lattice",
        "choice": COARSENING_RULE,
        "bias": (
            "measured, not asserted: exact on resolvable features; a fire finger "
            "narrower than 1 km is UNREPRESENTABLE at 1 km by any binary rule "
            "(500 m finger retains 67% of its area in 6 fragments). Biases ELMFIRE "
            "LOW on fine fingering, which is the flattering direction, so it is "
            "measured and not asserted: `python -m wildfire_nowcast.sim.coarsen` "
            "recomputes it from analytic scenes, with no corpus and no checkpoint, "
            "so a reader who clones this repository can reproduce the number."
        ),
    },
    {
        "field": "weather resolution",
        "issue": "ELMFIRE would ideally take RTMA on its own grid",
        "choice": "C1 channels 1-4 + 11 at 1 km, written as a separate coarser weather raster",
        "bias": (
            "none: RTMA is natively 2.5 km, so the 1 km C1 field is an UPSAMPLING "
            "and carries everything RTMA had."
        ),
    },
    {
        "field": "WX_BILINEAR_INTERPOLATION",
        "issue": (
            "elmfire_subs.f90:815 `J2 = MAX(MIN(J2+1, NROW_WX),1)` reads J2, an "
            "INTENT(OUT) argument, before it is set — the x-axis line above it "
            "correctly uses I1+1. A second uninitialised read, found by simviz."
        ),
        "choice": "set .FALSE. (nearest weather cell) rather than patch the physics",
        "bias": (
            "negligible at 1 km weather over a 2.5 km native product. Deliberately "
            "NOT patched: my standard for touching a baseline's source is that the "
            "change be provably output-neutral, and J1+1 would change results."
        ),
    },
    {
        "field": "m1 / m10 / m100",
        "issue": "C1 carries ONE fuel_moisture_proxy; Rothermel wants three dead classes",
        "choice": "m1 = proxy, m10 = proxy + 1 pt, m100 = proxy + 2 pt",
        "bias": "unknown, small; the +1/+2 offsets are conventional, not measured here",
    },
    {
        "field": "ws",
        "issue": "RTMA is 10 m wind in m/s; ELMFIRE wants 20 ft wind in mph",
        "choice": "log profile z0=0.03 m -> x0.8703, then x2.23694 m/s->mph",
        "bias": "none intended; the factor is stated so it can be disputed",
    },
    {
        "field": "wd",
        "issue": "our (u, v) is the vector the wind blows TOWARD; ELMFIRE wd is FROM",
        "choice": "wd = (270 - degrees(atan2(v, u))) mod 360",
        "bias": "an error here is a 180 deg flip and would be catastrophic and silent",
    },
    {
        "field": "initial front",
        "issue": "ELMFIRE normally starts from a point ignition",
        "choice": (
            "our x0 is replicated onto the fine lattice and written into ELMFIRE's "
            "level-set field phi (negative inside the burned region)"
        ),
        "bias": (
            "essential. A point-ignition mapping makes ELMFIRE re-grow the fire from "
            "scratch inside every window and look absurdly weak for a reason that is "
            "ours. Replication is exact — it invents no sub-cell detail."
        ),
    },
    {
        "field": "topography vintage",
        "issue": "LFPS publishes exactly one topographic vintage, LF2020",
        "choice": "LF2020 Elev/SlpD/Asp for every fire year",
        "bias": "none. Terrain is time-invariant at these scales; it is not a leak.",
    },
    {
        "field": "ensemble",
        "issue": "ELMFIRE is deterministic; its &MONTE_CARLO randomises ignition LOCATION",
        "choice": "M replicates with ws/wd/moisture perturbed from a declared PDF",
        "bias": "OUR construction. Any dispersion number is about the PDF we chose.",
    },
    {
        "field": "barrier / burn scar",
        "issue": "C1 channels 12 and 13 have no ELMFIRE equivalent",
        "choice": "not pushed into fuels; native FBFM40 passes through unchanged",
        "bias": "deliberate — ADR-022 (5) bars a hard non-burnable mask anywhere",
    },
]

_MS_TO_MPH = 2.2369362920544
_Z0 = 0.03
_WIND_10M_TO_20FT = math.log(6.096 / _Z0) / math.log(10.0 / _Z0)


class ElmfireNotInstalled(RuntimeError):
    """Raised when no ELMFIRE binary can be found. Never silently substituted."""


class InputMode(StrEnum):
    """Which input path a run uses. The negative control is a first-class mode."""

    #: ADR-026 (3): 30 m LANDFIRE fuels + canopy + topography, crown fire ON.
    NATIVE = "native"
    #: The S3 configuration, kept ONLY as playthrough 2's negative control:
    #: 1 km analysis grid, canopy layers zeroed so the crown fire model is OFF.
    LOBOTOMISED = "lobotomised"


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


def find_binary(explicit: str | Path | None = None) -> Path:
    """Locate the ELMFIRE executable. Env ``WILDFIRE_ELMFIRE_BIN`` wins (C7)."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("WILDFIRE_ELMFIRE_BIN")
    if env:
        candidates.append(Path(env))
    candidates += [
        VENDOR_ROOT / "build" / "linux" / "bin" / "elmfire",
        VENDOR_ROOT / "build" / "linux" / "elmfire_build" / "elmfire",
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c.resolve()
    raise ElmfireNotInstalled(
        "no ELMFIRE binary found. Build it with "
        "`python -m wildfire_nowcast.sim.elmfire --build`, or point "
        "WILDFIRE_ELMFIRE_BIN at one. Refusing to substitute another model: a "
        "silently-substituted baseline is how a gate gets decided against nothing."
    )


def build(*, source_root: Path = VENDOR_ROOT) -> Path:
    """Compile ELMFIRE with the two macOS link deltas. Returns the binary path."""
    src = source_root / "build" / "source"
    work = source_root / "build" / "linux" / "elmfire_build"
    out_bin = source_root / "build" / "linux" / "bin"
    work.mkdir(parents=True, exist_ok=True)
    out_bin.mkdir(parents=True, exist_ok=True)
    fc = shutil.which("mpifort") or shutil.which("mpif90")
    if fc is None:
        raise ElmfireNotInstalled(
            "mpifort not found. ELMFIRE is an MPI program; install open-mpi "
            "(`brew install open-mpi`). This is a system package manager, which "
            "the P2 charter allows; it is not a credential or a paid dependency."
        )
    sdk = subprocess.run(
        ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=False
    ).stdout.strip()
    defines = ["-D_SMOKE", "-D_WUI", "-D_UMDSPOTTING", "-D_SUPPRESSION"]
    fflags = [
        "-O3",
        "-frecord-marker=4",
        "-ffree-line-length-none",
        "-cpp",
        "-march=native",
        "-ffpe-summary=none",
    ]
    objects = [
        "elmfire_vars",
        "sort",
        "elmfire_init",
        "elmfire_namelists",
        "elmfire_subs",
        "elmfire_spread_rate",
        "elmfire_ignition",
        "elmfire_io",
        "elmfire_spotting",
        "elmfire_suppression",
        "elmfire_spotting_superseded",
        "elmfire_calibration",
        "elmfire_level_set",
        "elmfire",
    ]
    for stem in objects:
        source = src / f"{stem}.f90"
        if not source.exists():
            source = src / f"{stem}.for"
        subprocess.run([fc, "-c", *defines, *fflags, str(source)], cwd=work, check=True)
    link = [fc, *defines, *fflags]
    if sdk:
        link += ["-L", f"{sdk}/usr/lib"]
    else:
        # BUILD_NOTES delta_2 says this flag is why `ld` finds libSystem. Without
        # it the link fails several minutes later with an unrelated-looking error.
        logger.warning(
            "xcrun --show-sdk-path returned nothing; linking WITHOUT -L<sdk>/usr/lib "
            "(BUILD_NOTES delta_2). Expect ld to fail to find libSystem"
        )
    link += ["-o", "elmfire", *[f"{s}.o" for s in objects]]
    subprocess.run(link, cwd=work, check=True)
    final = out_bin / "elmfire"
    shutil.copy2(work / "elmfire", final)
    final.chmod(0o755)
    return final


# --------------------------------------------------------------------------
# ENVI BSQ / BIL
# --------------------------------------------------------------------------


def write_envi_bsq(
    path: Path,
    array: np.ndarray,
    *,
    x_left: float,
    y_top: float,
    cell_size: float,
    srs_name: str = "EPSG_5070",
    nodata: float = -9999.0,
) -> None:
    """Write ``[bands, rows, cols]`` (row 0 = NORTH) as ENVI BSQ + ``.hdr``.

    ``map info`` carries the UPPER-LEFT tie point, which is what
    ``READ_BSQ_RASTER`` parses before doing ``YLLCORNER = y - YDIM * NROWS``
    (``elmfire_io.f90:1311``). Our stores are C1.4 y-descending, i.e. row 0 IS
    north, so no flip happens here - and that is asserted by the geometry test,
    not assumed: getting it wrong mirrors every fire.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"expected 2-D or 3-D array, got shape {np.shape(array)}")
    bands, rows, cols = arr.shape
    if np.issubdtype(arr.dtype, np.integer):
        data, data_type = arr.astype("<i2"), 2
    else:
        data, data_type = arr.astype("<f4"), 4
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".bsq").write_bytes(data.tobytes(order="C"))
    hdr = "\n".join(
        [
            "ENVI",
            "description = {wildfire-nowcast ELMFIRE native input}",
            f"samples = {cols}",
            f"lines   = {rows}",
            f"bands   = {bands}",
            "header offset = 0",
            "file type = ENVI Standard",
            f"data type = {data_type}",
            "interleave = bsq",
            "byte order = 0",
            f"data ignore value = {nodata}",
            (
                "map info = {"
                f"{srs_name}, 1.0, 1.0, {x_left:.6f}, {y_top:.6f}, "
                f"{cell_size:.6f}, {cell_size:.6f}, units=Meters"
                "}"
            ),
            "",
        ]
    )
    path.with_suffix(".hdr").write_text(hdr)


def read_bil(stub: Path) -> np.ndarray:
    """Read an ELMFIRE ``.bil`` + ESRI ``.hdr`` output back as ``[rows, cols]``."""
    hdr_path = stub.with_suffix(".hdr")
    meta: dict[str, str] = {}
    for line in hdr_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meta[parts[0].strip().upper()] = parts[1].strip()
    rows, cols = int(meta["NROWS"]), int(meta["NCOLS"])
    nbands = int(meta.get("NBANDS", 1))
    nbits = int(meta.get("NBITS", 32))
    dtype = "<f4" if nbits == 32 else "<i2"
    raw = np.frombuffer(stub.with_suffix(".bil").read_bytes(), dtype=dtype)
    return raw.reshape(rows, nbands, cols)[:, 0, :].copy()


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    """A coarse sub-rectangle of the C1 grid and its exact fine refinement."""

    row0: int
    col0: int
    coarse: Grid
    fine: Grid
    refine: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.coarse.shape


def window_grids(
    grid: Grid, x0: np.ndarray, *, reach_cells: int, refine: int = DEFAULT_REFINE
) -> Window:
    """The sub-domain the fire can reach, snapped to whole 1 km cells.

    Running ELMFIRE at 30 m over a whole fire domain is wasteful and, worse,
    makes the run time depend on how much empty land a fire's bbox happens to
    contain. The window is the burned bbox grown by ``reach_cells``; because it
    is snapped to coarse cells, the fine grid stays EXACTLY nested and the
    coarsening block-mean remains exact.
    """
    burned = np.argwhere(np.asarray(x0) > 0)
    ny, nx = np.asarray(x0).shape
    if burned.size == 0:
        r0, c0, r1, c1 = 0, 0, ny, nx
    else:
        r0 = max(0, int(burned[:, 0].min()) - reach_cells)
        c0 = max(0, int(burned[:, 1].min()) - reach_cells)
        r1 = min(ny, int(burned[:, 0].max()) + reach_cells + 1)
        c1 = min(nx, int(burned[:, 1].max()) + reach_cells + 1)
    coarse = Grid(
        x_min=grid.x_min + c0 * grid.cell_size_m,
        y_max=grid.y_max - r0 * grid.cell_size_m,
        nx=c1 - c0,
        ny=r1 - r0,
        cell_size_m=grid.cell_size_m,
        crs=grid.crs,
    )
    return Window(r0, c0, coarse, fine_grid(coarse, refine), refine)


# --------------------------------------------------------------------------
# the C5 model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ElmfireConfig:
    """Everything that is not our tensor. Defaults are ELMFIRE's own (Rothermel)."""

    mode: InputMode = InputMode.NATIVE
    refine: int = DEFAULT_REFINE
    #: 1 km cells of head room around the t0 burned region, per forecast hour.
    #: 5 km/h is above ELMFIRE's own CROWN_FIRE_SPREAD_RATE_LIMIT (250 ft/min =
    #: 4.57 km/h), so the window cannot clip a physically reachable cell.
    reach_cells_per_hour: int = 5
    simulation_dt_s: float = 1.0
    live_herbaceous_moisture_pct: float = 30.0
    live_woody_moisture_pct: float = 60.0
    #: Per-member perturbation, one sigma. OUR ensemble construction, not ELMFIRE's.
    ws_sigma_frac: float = 0.15
    wd_sigma_deg: float = 15.0
    moisture_sigma_pct: float = 1.0
    mpi_ranks: int = 1
    keep_workdir: bool = False
    max_runtime_s: float = 600.0
    #: Explicit so a reader never has to infer it. NATIVE mode carries real
    #: CH/CBH/CBD, so the crown fire model has something to work with.
    crown_fire: bool = True
    #: A pre-supplied uniform stack (playthrough 2 runs with no network).
    stack: NativeStack | None = None
    #: [E1] A callable supplying the native stack for ONE window, so a whole-fire
    #: 30 m fetch can be SLICED instead of re-fetched per window. Takes precedence
    #: over :attr:`stack`. This is exact rather than an approximation:
    #: :func:`~wildfire_nowcast.sim.coarsen.fine_grid` shares the north-west corner
    #: and refines by an integer, and :func:`window_grids` snaps to whole coarse
    #: cells, so every window's fine grid is a sub-grid of the whole-domain fine
    #: grid with origin ``(row0 * refine, col0 * refine)`` and no remainder. The
    #: claim is CHECKED, not asserted, by
    #: ``sim.elmfire_stage.verify_slice_equivalence``, which re-fetches one window
    #: from LFPS and compares byte-for-byte against the slice.
    stack_provider: Callable[[Window], NativeStack] | None = None
    fire_year: int = 2020
    env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "refine": self.refine,
            "fine_cell_m": round(1000.0 / self.refine, 4),
            "crown_fire": self.crown_fire,
            "reach_cells_per_hour": self.reach_cells_per_hour,
            "simulation_dt_s": self.simulation_dt_s,
            "live_herbaceous_moisture_pct": self.live_herbaceous_moisture_pct,
            "live_woody_moisture_pct": self.live_woody_moisture_pct,
            "ws_sigma_frac": self.ws_sigma_frac,
            "wd_sigma_deg": self.wd_sigma_deg,
            "moisture_sigma_pct": self.moisture_sigma_pct,
            "coarsening_rule": COARSENING_RULE,
            "coarsening_threshold": OCCUPANCY_THRESHOLD,
            "stack_provider": self.stack_provider is not None,
        }


class ElmfireNativeModel:
    """ELMFIRE Monte Carlo behind C5 ``predict()``, on native 30 m inputs."""

    name = "elmfire"

    def __init__(
        self,
        grid: Grid,
        *,
        binary: str | Path | None = None,
        config: ElmfireConfig | None = None,
        name: str = "elmfire",
    ) -> None:
        self.grid = grid
        self.binary = find_binary(binary)
        self.config = config or ElmfireConfig()
        self.name = name
        self.last_run: dict[str, Any] = {}

    # -- inputs ----------------------------------------------------------

    def _stack_for(self, window: Window, static: np.ndarray | None) -> NativeStack:
        cfg = self.config
        if cfg.stack_provider is not None:
            return cfg.stack_provider(window)
        if cfg.stack is not None:
            return cfg.stack
        if cfg.mode is InputMode.NATIVE:
            return fetch_native_stack(window.fine, cfg.fire_year)
        # LOBOTOMISED: the S3 configuration, rebuilt from C1 at 1 km with the
        # canopy layers zeroed. It exists to be FLAGGED, not to be used.
        if static is None:
            raise ValueError("LOBOTOMISED mode needs the C5 `static` block")
        idx = {c: i for i, c in enumerate(STATIC_C5)}
        sub = np.asarray(static, dtype=np.float32)[
            :,
            window.row0 : window.row0 + window.coarse.ny,
            window.col0 : window.col0 + window.coarse.nx,
        ]
        asp = np.degrees(np.arctan2(sub[idx["aspect_sin"]], sub[idx["aspect_cos"]])) % 360.0
        zero = np.zeros(window.coarse.shape, dtype=np.int16)
        return NativeStack(
            grid=window.coarse,
            layers={
                "fbfm40": np.rint(sub[idx["fuel_model_id"]]).astype(np.int16),
                "cc": np.rint(np.clip(sub[idx["canopy_cover"]], 0, 100)).astype(np.int16),
                "ch": zero.copy(),
                "cbh": zero.copy(),
                "cbd": zero.copy(),
                "dem": np.rint(sub[idx["elevation"]]).astype(np.int16),
                "slp": np.rint(np.clip(sub[idx["slope"]], 0, 90)).astype(np.int16),
                "asp": np.rint(asp).astype(np.int16),
            },
            provenance={
                "source": "C1 tensor at 1 km, canopy zeroed - NEGATIVE CONTROL ONLY",
                "note": "reproduces the ADR-025 (4) configuration that gave +2 vs +54",
            },
        )

    @staticmethod
    def _weather_layers(
        weather: np.ndarray, rng: np.random.Generator, cfg: ElmfireConfig
    ) -> dict[str, np.ndarray]:
        idx = {c: i for i, c in enumerate(WEATHER_C5)}
        u = weather[:, idx["wind_u10"]]
        v = weather[:, idx["wind_v10"]]
        proxy = weather[:, idx["fuel_moisture_proxy"]]

        speed = np.hypot(u, v) * _WIND_10M_TO_20FT * _MS_TO_MPH
        # Meteorological convention: the direction the wind comes FROM.
        direction = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

        speed = speed * float(rng.normal(1.0, cfg.ws_sigma_frac))
        direction = (direction + float(rng.normal(0.0, cfg.wd_sigma_deg))) % 360.0
        dm = float(rng.normal(0.0, cfg.moisture_sigma_pct))
        m1 = np.clip(proxy + dm, 0.5, 60.0)
        return {
            "ws": np.maximum(speed, 0.0).astype(np.float32),
            "wd": direction.astype(np.float32),
            "m1": m1.astype(np.float32),
            "m10": np.clip(m1 + 1.0, 0.5, 60.0).astype(np.float32),
            "m100": np.clip(m1 + 2.0, 0.5, 60.0).astype(np.float32),
        }

    # -- namelist --------------------------------------------------------

    def _namelist(self, work: Path, analysis: Grid, n_t: int, horizon_h: int) -> str:
        cfg = self.config
        return f"""&INPUTS
FUELS_AND_TOPOGRAPHY_DIRECTORY = '{work}/inputs/'
ASP_FILENAME    = 'asp'
CBD_FILENAME    = 'cbd'
CBH_FILENAME    = 'cbh'
CC_FILENAME     = 'cc'
CH_FILENAME     = 'ch'
DEM_FILENAME    = 'dem'
FBFM_FILENAME   = 'fbfm40'
SLP_FILENAME    = 'slp'
ADJ_FILENAME    = 'adj'
PHI_FILENAME    = 'phi'
DT_METEOROLOGY  = 3600.0
WEATHER_DIRECTORY = '{work}/inputs/'
WS_FILENAME     = 'ws'
WD_FILENAME     = 'wd'
M1_FILENAME     = 'm1'
M10_FILENAME    = 'm10'
M100_FILENAME   = 'm100'
LH_MOISTURE_CONTENT = {cfg.live_herbaceous_moisture_pct}
LW_MOISTURE_CONTENT = {cfg.live_woody_moisture_pct}
CC_IN_PERCENT   = .TRUE.
CH_TIMES_10     = .TRUE.
CBH_TIMES_10    = .TRUE.
CBD_TIMES_100   = .TRUE.
USE_BSQ_XML_HEADER = .FALSE.
/

&OUTPUTS
OUTPUTS_DIRECTORY    = '{work}/outputs/'
DTDUMP               = 3600.
DUMP_TIME_OF_ARRIVAL = .TRUE.
CONVERT_TO_GEOTIFF   = .FALSE.
/

&COMPUTATIONAL_DOMAIN
A_SRS = 'EPSG:5070'
COMPUTATIONAL_DOMAIN_CELLSIZE  = {analysis.cell_size_m}
COMPUTATIONAL_DOMAIN_XLLCORNER = {analysis.x_min}
COMPUTATIONAL_DOMAIN_YLLCORNER = {analysis.y_min}
/

&TIME_CONTROL
SIMULATION_DT    = {cfg.simulation_dt_s}
SIMULATION_TSTOP = {float(horizon_h) * 3600.0}
/

&SIMULATOR
NUM_IGNITIONS = 0
MODE          = 1
CROWN_FIRE_MODEL = {1 if cfg.crown_fire else 0}
WX_BILINEAR_INTERPOLATION = .FALSE.
MAX_RUNTIME = {cfg.max_runtime_s}
/

&MONTE_CARLO
NUM_ENSEMBLE_MEMBERS = 1
NUM_METEOROLOGY_TIMES = {n_t}
SEED = 2024
/

&MISCELLANEOUS
MISCELLANEOUS_INPUTS_DIRECTORY = '{VENDOR_ROOT / "build" / "source"}/'
FUEL_MODEL_FILE = 'null'
PATH_TO_GDAL = '/usr/bin/'
SCRATCH = 'null'
/
"""

    # -- C5 --------------------------------------------------------------

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        cfg = self.config
        x0 = np.asarray(x0, dtype=np.uint8)
        height, width = x0.shape
        if (height, width) != self.grid.shape:
            raise ValueError(
                f"x0 is {x0.shape} but the model was constructed on a "
                f"{self.grid.shape} grid — refusing to guess the georeference"
            )
        static_arr = None if static is None else np.asarray(static, dtype=np.float32)
        wx = np.asarray(weather, dtype=np.float32)
        n_t = int(wx.shape[0])

        window = window_grids(
            self.grid,
            x0,
            reach_cells=max(1, cfg.reach_cells_per_hour * int(horizon_h)),
            refine=cfg.refine,
        )
        stack = self._stack_for(window, static_arr)
        analysis = stack.grid
        native = analysis.cell_size_m < self.grid.cell_size_m
        refine = window.refine if native else 1

        x0_win = x0[
            window.row0 : window.row0 + window.coarse.ny,
            window.col0 : window.col0 + window.coarse.nx,
        ]
        # Replicating a 1 km state onto sub-cells is EXACT: it invents no
        # sub-cell detail, it just states the same fact at a finer index.
        x0_fine = np.repeat(np.repeat(x0_win, refine, axis=0), refine, axis=1)
        phi = np.where(x0_fine > 0, -1.0, 1.0).astype(np.float32)
        adj = np.ones(analysis.shape, dtype=np.float32)

        wx_win = wx[
            :,
            :,
            window.row0 : window.row0 + window.coarse.ny,
            window.col0 : window.col0 + window.coarse.nx,
        ]

        out = np.zeros((n_members, horizon_h, height, width), dtype=np.uint8)
        runs: list[dict[str, Any]] = []
        for member in range(int(n_members)):
            rng = np.random.default_rng(int(seed) * 1_000_003 + member)
            work = Path(tempfile.mkdtemp(prefix=f"elmfire_m{member}_"))
            try:
                (work / "inputs").mkdir(parents=True, exist_ok=True)
                (work / "outputs").mkdir(parents=True, exist_ok=True)
                fuel_geom = {
                    "x_left": analysis.x_min,
                    "y_top": analysis.y_max,
                    "cell_size": analysis.cell_size_m,
                }
                for stub, layer in stack.layers.items():
                    write_envi_bsq(work / "inputs" / stub, layer, **fuel_geom)
                write_envi_bsq(work / "inputs" / "adj", adj, **fuel_geom)
                write_envi_bsq(work / "inputs" / "phi", phi, **fuel_geom)
                wx_geom = {
                    "x_left": window.coarse.x_min,
                    "y_top": window.coarse.y_max,
                    "cell_size": window.coarse.cell_size_m,
                }
                for stub, layer in self._weather_layers(wx_win, rng, cfg).items():
                    write_envi_bsq(work / "inputs" / stub, layer, **wx_geom)

                data = work / "elmfire.data"
                data.write_text(self._namelist(work, analysis, n_t, horizon_h))
                proc = subprocess.run(
                    [
                        "mpirun",
                        "-np",
                        str(cfg.mpi_ranks),
                        "--oversubscribe",
                        str(self.binary),
                        str(data),
                    ],
                    cwd=work,
                    capture_output=True,
                    text=True,
                    check=False,
                    env={**os.environ, **cfg.env},
                )
                if proc.returncode != 0:
                    # ELMFIRE's wall-clock abort is SILENT: elmfire_level_set.f90:1263
                    # sets T = TSTOP + 1 and no raster records that it happened. A
                    # non-zero return is the only signal we get, and it used to reach
                    # nothing but a field in the payload nobody reads until later.
                    logger.warning(
                        "member %d of %s returned %d; a truncated or absent arrival "
                        "raster is possible. stderr tail: %s",
                        member,
                        self.binary.name,
                        proc.returncode,
                        proc.stderr[-400:].strip(),
                    )
                arrival = self._read_arrival(work / "outputs", analysis.shape)
                if arrival is None:
                    raise RuntimeError(
                        "ELMFIRE produced no time_of_arrival raster.\n"
                        f"stdout tail:\n{proc.stdout[-3000:]}\n"
                        f"stderr tail:\n{proc.stderr[-3000:]}"
                    )
                for k in range(horizon_h):
                    reached = (arrival >= 0) & (arrival <= (k + 1) * 3600.0 + 1e-6)
                    burned_fine = reached | (x0_fine > 0)
                    coarse = coarsen_occupancy(burned_fine, refine) if refine > 1 else burned_fine
                    # Never lose a cell that was already burned at t0: the
                    # coarsening applies to what ELMFIRE ADDED, not to the state
                    # we handed it. Without this the rule could erase our own
                    # initial condition at a ragged perimeter.
                    coarse = coarse | (x0_win > 0)
                    out[
                        member,
                        k,
                        window.row0 : window.row0 + window.coarse.ny,
                        window.col0 : window.col0 + window.coarse.nx,
                    ] = coarse.astype(np.uint8)
                out[member] = np.maximum(out[member], x0[None])
                runs.append(
                    {
                        "member": member,
                        "returncode": proc.returncode,
                        "fine_new_cells": int(np.count_nonzero(reached & (x0_fine == 0))),
                    }
                )
            finally:
                if not cfg.keep_workdir:
                    shutil.rmtree(work, ignore_errors=True)
        # C1.1: fire is absorbing, so the sample sequence must be non-decreasing.
        out = np.maximum.accumulate(out, axis=1)
        self.last_run = {
            "binary": str(self.binary),
            "config": cfg.as_dict(),
            "mode": cfg.mode.value,
            "analysis_cell_size_m": round(analysis.cell_size_m, 4),
            "analysis_shape": list(analysis.shape),
            "window_origin_rowcol": [window.row0, window.col0],
            "window_coarse_shape": list(window.coarse.shape),
            "static_consumed": cfg.mode is InputMode.LOBOTOMISED,
            "stack_provenance": stack.provenance,
            "stack_summary": stack.summary(),
            "n_members": int(n_members),
            "horizon_h": int(horizon_h),
            "seed": int(seed),
            "members": runs,
            "mapping_compromises": MAPPING_COMPROMISES,
        }
        return out

    @staticmethod
    def _read_arrival(outputs: Path, shape: tuple[int, int]) -> np.ndarray | None:
        for cand in sorted(outputs.glob("time_of_arrival*.bil")):
            arr = read_bil(cand.with_suffix(""))
            if arr.shape != shape:
                raise RuntimeError(
                    f"ELMFIRE returned {arr.shape}, expected {shape} — the grid "
                    "mapping is wrong, which would silently mirror or crop the fire"
                )
            return arr
        return None


# --------------------------------------------------------------------------
# CLI - build, and a one-fire smoke run. NOT a comparison; G5 is not authorised.
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(prog="python -m wildfire_nowcast.sim.elmfire", allow_abbrev=False)
    ap.add_argument("--build", action="store_true", help="compile the vendored ELMFIRE")
    ap.add_argument("--tensor", default=None, help="a C1 tensor.zarr to smoke-run on")
    ap.add_argument("--mode", default="native", choices=[m.value for m in InputMode])
    ap.add_argument("--t0", type=int, default=None)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--refine", type=int, default=DEFAULT_REFINE)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", default="reports/figures/elmfire_native_smoke.json")
    add_logging_arguments(ap)
    args = ap.parse_args(argv)
    # ADR-103: the ONE place this program configures logging. INFO by default: a
    # build or a smoke run is long, and its narration is the point.
    configure_from_args(args, default_verbosity=1)

    if args.build:
        print(f"[elmfire] built {build()}")
    if not args.tensor:
        return 0

    from wildfire_nowcast.common.zarr_io import open_tensor  # noqa: PLC0415
    from wildfire_nowcast.sim.c5 import c5_inputs  # noqa: PLC0415

    ds = open_tensor(Path(args.tensor))
    state = np.asarray(ds["fire_state"].values, dtype=np.uint8)
    if args.t0 is None:
        growth = [
            (int((state[t + args.horizon] > 0).sum() - (state[t] > 0).sum()), t)
            for t in range(1, len(state) - args.horizon)
        ]
        t0 = max(growth)[1]
    else:
        t0 = int(args.t0)
    window = c5_inputs(ds, t0, args.horizon)

    grid = Grid.from_dataset(ds)
    year = int(str(ds.attrs.get("time_start_utc", "2020"))[:4])
    cfg = ElmfireConfig(
        mode=InputMode(args.mode),
        refine=args.refine,
        crown_fire=args.mode == InputMode.NATIVE.value,
        fire_year=year,
    )
    model = ElmfireNativeModel(grid, config=cfg)
    samples = model.predict(
        window.x0, window.static, window.weather, args.members, args.horizon, args.seed
    )

    burned0 = int((window.x0 > 0).sum())
    truth_new = int((window.truth[-1] > 0).sum()) - burned0
    per_member = [int((samples[m, -1] > 0).sum()) - burned0 for m in range(args.members)]
    payload = {
        "kind": "elmfire_native_smoke",
        "note": (
            "ADAPTER PROOF only. NOT a baseline comparison: G5 is not authorised "
            "and no C6 metric is computed here."
        ),
        "tensor": str(args.tensor),
        "mode": args.mode,
        "t0": t0,
        "horizon_h": args.horizon,
        "n_members": args.members,
        "grid_shape": list(window.x0.shape),
        "burned_at_t0": burned0,
        "truth_new_cells": truth_new,
        "elmfire_new_cells_per_member": per_member,
        "samples_are_absorbing": bool(np.all(np.diff(samples.astype(np.int16), axis=1) >= 0)),
        "distinct_members": len({s.tobytes() for s in samples}),
        "build_notes": BUILD_NOTES,
        "run": model.last_run,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(f"[elmfire] {out}")
    print(
        f"[elmfire] mode={args.mode} t0={t0} burned@t0={burned0} "
        f"truth new={truth_new} elmfire new per member={per_member}"
    )
    print(
        f"[elmfire] distinct members: {payload['distinct_members']}/{args.members}  "
        f"absorbing: {payload['samples_are_absorbing']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
