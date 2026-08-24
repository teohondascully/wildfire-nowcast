"""[G5] ELMFIRE as a COLUMN OF THE SAME TABLE, not a table of its own.

WHAT THIS MODULE IS FOR
-----------------------
"ELMFIRE Monte Carlo with default Rothermel parameters" is a non-negotiable
baseline of this project, stated inline here rather than cited: the document
that froze it is not in this repository, so a reader could not open it. G5 is
the head-to-head. The only thing that makes a
head-to-head worth reading is that every column was produced by ONE code path:
same windows, same members, same seeds, same C6 call, same C6.2 validity check.
:func:`wildfire_nowcast.eval.baseline_run.run_baselines` already provides that
path and already has the hook for it -- ``extra_models`` -- whose own docstring
says why:

    "a model evaluated by its own script and a baseline evaluated by this one
    differ by every detail nobody wrote down, and the difference always
    flatters the model whose author wrote the script."

So nothing here scores anything. Every number G5 reports comes out of
``run_baselines``; this module's entire job is to make ELMFIRE *callable* from
inside it without blowing the wall-clock budget, and to refuse loudly rather
than substitute quietly when it cannot.

WHY A CACHE SITS BETWEEN THEM, AND WHY THAT IS NOT A SHORTCUT
-------------------------------------------------------------
``run_baselines`` is a single serial call. One ELMFIRE member is one ``mpirun``
subprocess over a 30 m analysis grid, and on this corpus that is **5.9 s to
>300 s of CPU per member** depending on how much of the landscape is already
burning (measured, see :data:`MEASURED_MEMBER_SECONDS`). Serial, that is weeks.

:class:`EnsembleStore` therefore holds ELMFIRE's output for a window, and
:class:`CachedEnsembleModel` replays it inside ``run_baselines``. The replay is
NOT an approximation of the inline call, and the difference is worth being
precise about:

* the cache key is a digest of the EXACT C5 arguments -- ``x0``, ``weather``,
  ``n_members``, ``horizon_h``, ``seed`` -- so a hit means the stored array is
  the array ``ElmfireNativeModel.predict`` returned for THOSE arguments;
* a miss RAISES :class:`MissingEnsemble`. It does not fall back to persistence,
  to zeros, or to a neighbouring window. A baseline that quietly degrades to
  silence is the C6.2 failure mode this project has already met twice, and it
  would flatter us, so the refusal is the point;
* :func:`verify_replay_is_inline` is the control: it runs one window BOTH ways
  and requires the arrays to be bit-identical.

WHAT IS *NOT* DECIDED HERE
--------------------------
Nothing in this module chooses a stride, a member count or a fire list. Those
are arguments, they are stamped into the artifact, and the reason is that a
window set chosen after seeing which windows were cheap is a window set chosen
on a quantity correlated with fire growth. :func:`plan` enumerates the work
BEFORE any of it runs and writes the plan out, so the set that was intended is
recoverable from disk even if the run dies half way.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wildfire_nowcast.common.grid import Grid
from wildfire_nowcast.common.paths import fire_tensor_path
from wildfire_nowcast.common.zarr_io import open_tensor
from wildfire_nowcast.model.inputs import iter_windows
from wildfire_nowcast.sim.coarsen import DEFAULT_REFINE
from wildfire_nowcast.sim.elmfire import ElmfireConfig, ElmfireNativeModel, InputMode, find_binary

logger = logging.getLogger(__name__)

__all__ = [
    "MEASURED_MEMBER_SECONDS",
    "MissingEnsemble",
    "WindowJob",
    "EnsembleStore",
    "CachedEnsembleModel",
    "ElmfireDispatch",
    "window_key",
    "plan",
    "run_job",
    "verify_replay_is_inline",
]


#: Wall-clock CPU seconds for ONE ELMFIRE member, measured on this machine on
#: real held-out windows (S14, 30.303 m analysis grid, crown fire on,
#: ``SIMULATION_DT`` 1 s). Recorded because the feasibility of G5 turns on it
#: and because "ELMFIRE is slow" is not a number.
#:
#: The pattern is not about the fire's name: cost tracks the size of the
#: ANALYSIS GRID and how much of it is already burning, so it climbs steeply
#: with ``t0`` inside a single fire.
MEASURED_MEMBER_SECONDS: dict[str, float] = {
    "2020_dolan@t0=300": 5.9,
    "2020_july_complex@t0=150": 21.3,
    "2020_czu_lightning_complex@t0=60": 22.7,
    "2024_borel@t0=120": 47.2,
    "2020_dolan@t0=600": 95.6,
}


class MissingEnsemble(RuntimeError):
    """A window was scored that no ELMFIRE run ever produced samples for.

    Raised instead of returning anything, because every plausible substitute --
    zeros, persistence, the previous window -- is a SILENT baseline
    degradation, and C6.2 exists because a degenerate baseline reads as a win.
    """


# --------------------------------------------------------------------------
# keying
# --------------------------------------------------------------------------


def window_key(
    x0: np.ndarray,
    weather: np.ndarray,
    n_members: int,
    horizon_h: int,
    seed: int,
) -> str:
    """Digest of the EXACT C5 arguments an ELMFIRE ensemble was produced from.

    ``static`` is deliberately absent: in ``InputMode.NATIVE`` ELMFIRE does not
    consume it (fuels and terrain come from 30 m LANDFIRE), and
    ``sim.elmfire`` records ``static_consumed: false`` for exactly that reason.
    Hashing an argument the producer ignores would make the key sensitive to
    something the value cannot depend on.
    """
    h = hashlib.sha256()
    for arr in (np.ascontiguousarray(x0), np.ascontiguousarray(weather, dtype=np.float32)):
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    h.update(f"|{int(n_members)}|{int(horizon_h)}|{int(seed)}".encode())
    return h.hexdigest()[:32]


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleStore:
    """One ``.npz`` per window, bit-packed. Append-only; never overwritten.

    Bit-packing is not a space optimisation for its own sake -- the samples are
    ``uint8`` in ``{0, 1, 2}``... which does NOT pack to one bit. So the array is
    stored as-is and compressed; the note exists so the next reader does not
    "optimise" a lossy packing into a state variable that has three values.
    """

    root: Path

    def path(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self.path(key).is_file()

    def get(self, key: str) -> np.ndarray:
        p = self.path(key)
        if not p.is_file():
            raise MissingEnsemble(f"no stored ensemble for key {key}")
        with np.load(p, allow_pickle=False) as blob:
            return np.asarray(blob["samples"], dtype=np.uint8)

    def put(self, key: str, samples: np.ndarray, meta: Mapping[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(key)
        # The temp name MUST end in `.npz`: `np.savez_compressed` APPENDS `.npz`
        # to any path that does not, so `key.npz.part` was silently written as
        # `key.npz.part.npz` and the rename then failed on a file that did not
        # exist. Caught by a crash after ten minutes of ELMFIRE; the arrays were
        # on disk under the wrong name the whole time.
        tmp = self.root / f"{key}.part.npz"
        np.savez_compressed(
            tmp,
            samples=np.asarray(samples, dtype=np.uint8),
            meta_json=np.asarray(json.dumps(dict(meta), sort_keys=True)),
        )
        tmp.replace(p)
        return p

    def meta(self, key: str) -> dict[str, Any]:
        with np.load(self.path(key), allow_pickle=False) as blob:
            return dict(json.loads(str(blob["meta_json"].item())))

    def n_stored(self) -> int:
        return len(list(self.root.glob("*.npz"))) if self.root.is_dir() else 0

    def bytes_on_disk(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.root.glob("*.npz"))


# --------------------------------------------------------------------------
# the two C5 objects
# --------------------------------------------------------------------------


class ElmfireDispatch:
    """One ELMFIRE model per fire, selected by the grid the caller hands it.

    C5's ``predict`` carries no CRS and no fire id, but ELMFIRE needs both --
    it reads 30 m LANDFIRE for a specific place in a specific year. Every
    held-out fire has a DISTINCT ``(H, W)``, which is checked at construction,
    so the shape identifies the fire unambiguously. An unknown shape RAISES:
    guessing a georeference silently mirrors or relocates a fire, and the
    output still looks like a fire.
    """

    name = "elmfire"

    def __init__(
        self,
        fire_ids: Sequence[str],
        *,
        binary: Path | None = None,
        refine: int = DEFAULT_REFINE,
        simulation_dt_s: float = 1.0,
        max_runtime_s: float = 600.0,
        split_fingerprint: str = "",
        stack_provider_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.binary = Path(binary) if binary else find_binary()
        self.refine = int(refine)
        self.simulation_dt_s = float(simulation_dt_s)
        self.max_runtime_s = float(max_runtime_s)
        self._factory = stack_provider_factory
        self._by_shape: dict[tuple[int, int], str] = {}
        self._models: dict[str, ElmfireNativeModel] = {}
        for fid in fire_ids:
            ds = open_tensor(fire_tensor_path(fid))
            try:
                grid = Grid.from_dataset(ds)
            finally:
                ds.close()
            shape = (int(grid.shape[0]), int(grid.shape[1]))
            if shape in self._by_shape:
                raise ValueError(
                    f"{fid} and {self._by_shape[shape]} both have grid {shape}; shape cannot "
                    "identify the fire and this dispatcher would silently use the wrong "
                    "georeference"
                )
            self._by_shape[shape] = fid
            self._grids: dict[str, Grid] = getattr(self, "_grids", {})
            self._grids[fid] = grid
        # ELMFIRE is not fitted on anything. The stamp asserts "compatible with
        # every split because it saw no fire", NOT "trained on this split"; C8's
        # check is an equality against the evaluation split and there is no
        # third value that means "untrained". Declared in the artifact.
        self.provenance = {
            "split_fingerprint": str(split_fingerprint),
            "split_fingerprint_meaning": (
                "ELMFIRE IS UNTRAINED. This stamp exists because C8 hard-fails an "
                "unstamped model and has no value meaning 'saw no fires'. It asserts "
                "compatibility, not fitting."
            ),
        }

    def fire_for(self, shape: tuple[int, int]) -> str:
        key = (int(shape[0]), int(shape[1]))
        if key not in self._by_shape:
            raise KeyError(
                f"no held-out fire has grid {key}; refusing to guess a georeference. "
                f"known: {sorted(self._by_shape)}"
            )
        return self._by_shape[key]

    def model_for(self, fire_id: str) -> ElmfireNativeModel:
        if fire_id not in self._models:
            provider = self._factory(fire_id) if self._factory else None
            year = int(fire_id.split("_", 1)[0])
            cfg = ElmfireConfig(
                mode=InputMode.NATIVE,
                refine=self.refine,
                crown_fire=True,
                fire_year=year,
                stack_provider=provider,
                simulation_dt_s=self.simulation_dt_s,
                max_runtime_s=self.max_runtime_s,
            )
            self._models[fire_id] = ElmfireNativeModel(
                self._grids[fire_id], binary=self.binary, config=cfg
            )
        return self._models[fire_id]

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        rows, cols = np.asarray(x0).shape
        fid = self.fire_for((int(rows), int(cols)))
        return self.model_for(fid).predict(x0, static, weather, n_members, horizon_h, seed)


class CachedEnsembleModel:
    """Replays a stored ELMFIRE ensemble under the C5 signature.

    Every argument is used to look the ensemble up, so a hit is a proof that
    these are the arguments the stored array was produced from. A miss raises.
    """

    def __init__(self, store: EnsembleStore, *, name: str = "elmfire", provenance: Any = None):
        self.store = store
        self.name = name
        self.provenance = dict(provenance or {})
        self.hits: list[str] = []

    def predict(
        self,
        x0: np.ndarray,
        static: np.ndarray,
        weather: np.ndarray,
        n_members: int,
        horizon_h: int,
        seed: int,
    ) -> np.ndarray:
        key = window_key(x0, weather, n_members, horizon_h, seed)
        if not self.store.has(key):
            raise MissingEnsemble(
                f"ELMFIRE has no ensemble for window key {key} "
                f"(n_members={n_members}, horizon_h={horizon_h}, seed={seed}, grid={x0.shape}). "
                "Refusing to substitute: a silently degraded baseline is C6.2's failure mode."
            )
        samples = self.store.get(key)
        expected = (int(n_members), int(horizon_h), int(x0.shape[0]), int(x0.shape[1]))
        if samples.shape != expected:
            raise MissingEnsemble(
                f"stored ensemble for {key} has shape {samples.shape}, expected {expected}"
            )
        self.hits.append(key)
        return samples


# --------------------------------------------------------------------------
# planning and running
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowJob:
    """One (fire, window) unit of ELMFIRE work, with the seed it will be scored at.

    ``seed`` reproduces ``eval.baseline_run._score`` exactly: that function does
    ``seed + i`` where ``i`` enumerates THAT FIRE's window list, restarting at
    every fire. Reproducing the arithmetic here rather than inventing one is
    what makes the cached array the array the scorer would have got inline.
    """

    fire_id: str
    spatial_block_id: int
    index: int
    t0: int
    seed: int
    key: str
    truth_growth_cells: int


def plan(
    fires: Mapping[str, int],
    *,
    horizon_h: int,
    stride: int,
    n_members: int,
    seed: int,
) -> list[WindowJob]:
    """Enumerate every window ``run_baselines`` will score, before running any.

    ``fires`` maps ``fire_id -> spatial_block_id``. The enumeration uses the
    same ``iter_windows`` call ``eval.baseline_run._windows_for`` makes, so the
    plan is the scorer's window list and not a parallel one.
    """
    jobs: list[WindowJob] = []
    for fire_id, block in fires.items():
        ds = open_tensor(fire_tensor_path(fire_id))
        try:
            windows = list(iter_windows(ds, horizon_h, stride=stride, fire_id=fire_id))
        finally:
            ds.close()
        for i, w in enumerate(windows):
            jobs.append(
                WindowJob(
                    fire_id=fire_id,
                    spatial_block_id=int(block),
                    index=i,
                    t0=int(w.t0),
                    seed=int(seed) + i,
                    key=window_key(w.x0, w.weather, n_members, horizon_h, int(seed) + i),
                    truth_growth_cells=int(w.truth_growth_cells()),
                )
            )
    return jobs


def run_job(
    dispatch: ElmfireDispatch,
    store: EnsembleStore,
    window: Any,
    job: WindowJob,
    *,
    n_members: int,
    horizon_h: int,
) -> dict[str, Any]:
    """Produce and store ONE window's ELMFIRE ensemble. Idempotent on a hit."""
    if store.has(job.key):
        return {"key": job.key, "cached": True, "elapsed_s": 0.0}
    started = time.time()
    model = dispatch.model_for(job.fire_id)
    samples = model.predict(
        window.x0, window.static, window.weather, n_members, horizon_h, job.seed
    )
    elapsed = time.time() - started
    last = dict(model.last_run or {})
    members = list(last.get("members") or [])
    meta = {
        "fire_id": job.fire_id,
        "spatial_block_id": job.spatial_block_id,
        "t0": job.t0,
        "window_index": job.index,
        "seed": job.seed,
        "n_members": int(n_members),
        "horizon_h": int(horizon_h),
        "elapsed_s": round(elapsed, 2),
        "seconds_per_member": round(elapsed / max(1, int(n_members)), 2),
        "returncodes": [int(m.get("returncode", -1)) for m in members],
        "fine_new_cells": [int(m.get("fine_new_cells", -1)) for m in members],
        "peak_hourly_head_over_cap": [
            float((m.get("head") or {}).get("peak_hourly_head_over_cap", float("nan")))
            for m in members
        ],
        "config": last.get("config"),
        "analysis_shape": last.get("analysis_shape"),
        "static_consumed": last.get("static_consumed"),
    }
    store.put(job.key, samples, meta)
    return {"key": job.key, "cached": False, "elapsed_s": round(elapsed, 2), "meta": meta}


def verify_replay_is_inline(
    dispatch: ElmfireDispatch,
    store: EnsembleStore,
    window: Any,
    job: WindowJob,
    *,
    n_members: int,
    horizon_h: int,
) -> dict[str, Any]:
    """CONTROL: the cache must return exactly what an inline call returns.

    Runs ELMFIRE twice on the same window -- once through the dispatcher, once
    replayed from the store -- and requires bitwise equality. Without this the
    cache is an unverified claim about determinism, and ELMFIRE's own
    determinism is a property this project has measured (S3, 4/4) rather than
    assumed.
    """
    inline = dispatch.model_for(job.fire_id).predict(
        window.x0, window.static, window.weather, n_members, horizon_h, job.seed
    )
    replayed = CachedEnsembleModel(store).predict(
        window.x0, window.static, window.weather, n_members, horizon_h, job.seed
    )
    identical = bool(np.array_equal(inline, replayed))
    return {
        "fire_id": job.fire_id,
        "t0": job.t0,
        "seed": job.seed,
        "key": job.key,
        "bitwise_identical": identical,
        "n_differing_cells": int(np.count_nonzero(inline != replayed)),
        "inline_burned_3h": int(np.count_nonzero(inline[:, -1] > 0)),
    }


# --------------------------------------------------------------------------
# parallel precompute
# --------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _init_worker(
    fire_id: str,
    refine: int,
    binary: str,
    store_root: str,
    simulation_dt_s: float,
    max_runtime_s: float,
) -> None:
    """One process, one fire: the 30 m domain stack is loaded ONCE per worker.

    Imported lazily because ``sim.elmfire_stage`` pulls in the LANDFIRE client,
    and a module that is imported to read a cache should not require a network
    stack at import time.
    """
    from wildfire_nowcast.sim.elmfire_stage import build_domain_stack

    domain = build_domain_stack(fire_id, refine=refine)
    ds = open_tensor(fire_tensor_path(fire_id))
    dispatch = ElmfireDispatch(
        [fire_id],
        binary=Path(binary),
        refine=refine,
        simulation_dt_s=simulation_dt_s,
        max_runtime_s=max_runtime_s,
        stack_provider_factory=lambda _fid: domain.slice_for,
    )
    _W.update(fire_id=fire_id, ds=ds, dispatch=dispatch, store=EnsembleStore(Path(store_root)))


def _run_one(payload: tuple[int, int, int, str, int, int]) -> dict[str, Any]:
    index, t0, block, key, seed, n_members = payload
    from wildfire_nowcast.model.inputs import forecast_inputs

    store: EnsembleStore = _W["store"]
    if store.has(key):
        return {"key": key, "cached": True, "t0": t0, "elapsed_s": 0.0}
    window = forecast_inputs(_W["ds"], t0, 3, fire_id=_W["fire_id"])
    job = WindowJob(
        fire_id=_W["fire_id"],
        spatial_block_id=block,
        index=index,
        t0=t0,
        seed=seed,
        key=key,
        truth_growth_cells=int(window.truth_growth_cells()),
    )
    out = run_job(_W["dispatch"], store, window, job, n_members=n_members, horizon_h=3)
    out["t0"] = t0
    return out


# --------------------------------------------------------------------------
# CLI: plan -> precompute -> score. THREE STEPS, THREE ARTIFACTS.
# --------------------------------------------------------------------------


def heldout_fires() -> dict[str, int]:
    """``fire_id -> spatial_block_id`` for the held-out fold, read off the split.

    Delegates to ``eval.baseline_run.load_splits`` and to ``norm_stats`` for the
    train folds, so this cannot disagree with the scorer about who is held out.
    """
    from wildfire_nowcast.common.paths import norm_stats_path
    from wildfire_nowcast.common.zarr_io import read_norm_stats
    from wildfire_nowcast.eval.baseline_run import load_splits

    stats = read_norm_stats(norm_stats_path())
    folds = [int(f) for f in stats["train_folds"]]
    return {s.fire_id: int(s.spatial_block_id) for s in load_splits(folds) if not s.is_train}


def precompute(
    jobs: Sequence[WindowJob],
    store: EnsembleStore,
    *,
    n_members: int,
    refine: int,
    binary: Path,
    simulation_dt_s: float,
    max_runtime_s: float,
    workers: int,
    progress_path: Path | None = None,
    budget_s: float | None = None,
) -> dict[str, Any]:
    """Fill the store, one fire at a time, checkpointing after every window.

    ``budget_s`` STOPS the run; it does not choose windows. Whatever is missing
    when it fires is missing by wall clock, and the caller is handed the list --
    dropping expensive windows quietly would select on a quantity correlated
    with fire size, hence with growth, hence with the score.
    """
    from concurrent.futures import ProcessPoolExecutor

    started = time.time()
    done: list[dict[str, Any]] = []
    by_fire: dict[str, list[WindowJob]] = {}
    for j in jobs:
        by_fire.setdefault(j.fire_id, []).append(j)
    stopped_early = False
    for fire_id, fire_jobs in by_fire.items():
        todo = [j for j in fire_jobs if not store.has(j.key)]
        if not todo:
            continue
        if stopped_early:
            break
        payloads = [
            (j.index, j.t0, j.spatial_block_id, j.key, j.seed, int(n_members)) for j in todo
        ]
        with ProcessPoolExecutor(
            max_workers=max(1, int(workers)),
            initializer=_init_worker,
            initargs=(
                fire_id,
                int(refine),
                str(binary),
                str(store.root),
                float(simulation_dt_s),
                float(max_runtime_s),
            ),
        ) as pool:
            for res in pool.map(_run_one, payloads):
                res["fire_id"] = fire_id
                done.append(res)
                if progress_path is not None:
                    with progress_path.open("a") as fh:
                        fh.write(json.dumps(res) + "\n")
                logger.info(
                    "%s t0=%s %.1fs (%d/%d stored)",
                    fire_id,
                    res.get("t0"),
                    res.get("elapsed_s", 0.0),
                    store.n_stored(),
                    len(jobs),
                )
                if budget_s is not None and (time.time() - started) > budget_s:
                    stopped_early = True
                    break
        if stopped_early:
            break
    missing = [j for j in jobs if not store.has(j.key)]
    return {
        "n_jobs": len(jobs),
        "n_stored": store.n_stored(),
        "n_missing": len(missing),
        "missing": [
            {"fire_id": j.fire_id, "t0": j.t0, "key": j.key, "growth": j.truth_growth_cells}
            for j in missing
        ],
        "stopped_on_budget": stopped_early,
        "elapsed_s": round(time.time() - started, 1),
        "store_bytes": store.bytes_on_disk(),
    }


def score(
    store: EnsembleStore,
    *,
    horizon_h: int,
    stride: int,
    n_members: int,
    seed: int,
    fit_stride: int,
    split_fingerprint_value: str,
) -> dict[str, Any]:
    """THE table. Produced by ``run_baselines``, unmodified, in one call.

    persistence, the calibrated ellipse and ELMFIRE come out of the SAME loop,
    on the SAME windows, at the SAME member count and seeds, through the SAME
    ``eval.metrics.evaluate``. That is the only property that makes an ELMFIRE
    column worth printing beside a model column, and it is a property of WHERE
    the numbers are computed, not of how carefully they are copied.
    """
    from wildfire_nowcast.eval.baseline_run import run_baselines

    model = CachedEnsembleModel(
        store,
        provenance={
            "split_fingerprint": split_fingerprint_value,
            "untrained": True,
            "note": (
                "ELMFIRE is not fitted on any fire. The C8 stamp asserts compatibility "
                "with this split, not that it was trained on it."
            ),
        },
    )
    payload = run_baselines(
        horizon_h=horizon_h,
        n_members=n_members,
        seed=seed,
        stride=stride,
        fit_stride=fit_stride,
        write_run=False,
        extra_models={"elmfire": model},
    )
    payload["elmfire_replay"] = {
        "n_windows_replayed": len(model.hits),
        "n_distinct_keys": len(set(model.hits)),
        "store_root": str(store.root),
        "store_bytes": store.bytes_on_disk(),
    }
    return payload


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    import argparse

    from wildfire_nowcast.common.logs import add_logging_arguments, configure_from_args

    p = argparse.ArgumentParser(description="G5: ELMFIRE through run_baselines' extra_models hook")
    p.add_argument("step", choices=["plan", "precompute", "score"])
    p.add_argument("--store", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260807)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--fit-stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refine", type=int, default=DEFAULT_REFINE)
    p.add_argument("--simulation-dt", type=float, default=1.0)
    p.add_argument("--max-runtime", type=float, default=600.0)
    p.add_argument("--budget-s", type=float, default=None)
    p.add_argument("--fires", nargs="*", default=None)
    add_logging_arguments(p)
    args = p.parse_args(argv)
    configure_from_args(args)

    from wildfire_nowcast.eval.reporting import split_fingerprint

    fp = str(split_fingerprint()["fingerprint"])
    fires = heldout_fires()
    if args.fires:
        fires = {k: v for k, v in fires.items() if k in set(args.fires)}
    store = EnsembleStore(args.store)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.step == "plan":
        jobs = plan(
            fires,
            horizon_h=args.horizon,
            stride=args.stride,
            n_members=args.members,
            seed=args.seed,
        )
        payload = {
            "step": "plan",
            "split_fingerprint": fp,
            "fires": fires,
            "stride": args.stride,
            "n_members": args.members,
            "seed": args.seed,
            "horizon_h": args.horizon,
            "n_jobs": len(jobs),
            "per_fire": {f: sum(1 for j in jobs if j.fire_id == f) for f in fires},
            "jobs": [j.__dict__ for j in jobs],
        }
        args.out.write_text(json.dumps(payload, indent=1))
        print(json.dumps({k: payload[k] for k in ("n_jobs", "per_fire")}, indent=1))
        return 0

    if args.step == "precompute":
        jobs = plan(
            fires,
            horizon_h=args.horizon,
            stride=args.stride,
            n_members=args.members,
            seed=args.seed,
        )
        report = precompute(
            jobs,
            store,
            n_members=args.members,
            refine=args.refine,
            binary=find_binary(),
            simulation_dt_s=args.simulation_dt,
            max_runtime_s=args.max_runtime,
            workers=args.workers,
            progress_path=args.out.with_suffix(".jsonl"),
        )
        args.out.write_text(json.dumps({"step": "precompute", **report}, indent=1))
        print(json.dumps({k: report[k] for k in ("n_jobs", "n_stored", "n_missing")}, indent=1))
        return 0

    payload = score(
        store,
        horizon_h=args.horizon,
        stride=args.stride,
        n_members=args.members,
        seed=args.seed,
        fit_stride=args.fit_stride,
        split_fingerprint_value=fp,
    )
    args.out.write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps(payload.get("elmfire_replay", {}), indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


# --------------------------------------------------------------------------
# the four-block table, and the control that proves it is the SAME table
# --------------------------------------------------------------------------


def score_blocks(
    store: EnsembleStore,
    fire_ids: Sequence[str],
    *,
    horizon_h: int,
    stride: int,
    n_members: int,
    seed: int,
    fit_stride: int,
    split_fingerprint_value: str,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """ELMFIRE, persistence and the ellipse on a SUBSET of held-out fires.

    WHY THIS EXISTS AND WHY IT IS SECOND-BEST. ``run_baselines`` scores every
    held-out fire or none: it derives the held-out list from the split and takes
    no fire argument. ELMFIRE cannot afford ``2020_creek`` (measured: **480 s of
    CPU for ONE member at t0=200**, against 0.3-47 s on ``2020_czu``), so the
    complete call is unaffordable and the choice is a subset table or no table.

    The subset table is only worth printing if it is the SAME table. So nothing
    here re-implements a loop: the per-window scoring is
    ``eval.baseline_run._score`` ITSELF, pooling is ``eval.metrics.aggregate``,
    the headline is ``eval.baseline_run._headline``, and the ellipse is
    calibrated by the same ``calibrate_to_growth`` call on the same train
    windows. ``reference`` -- a ``run_baselines`` payload at the SAME stride,
    members and seed -- is then compared key by key on persistence and the
    ellipse. If those two columns are not identical, this function's ELMFIRE
    column is not commensurable either, and the check says so instead of the
    reader having to trust the paragraph above.
    """
    from wildfire_nowcast.common.paths import norm_stats_path
    from wildfire_nowcast.common.zarr_io import read_norm_stats
    from wildfire_nowcast.eval.baseline_run import (
        NULL_MODELS,
        _headline,
        _score,
        _windows_for,
        load_splits,
    )
    from wildfire_nowcast.eval.metrics import aggregate
    from wildfire_nowcast.eval.validity import baseline_validity
    from wildfire_nowcast.model.baselines import EllipseBaseline, PersistenceBaseline

    stats = read_norm_stats(norm_stats_path())
    folds = [int(f) for f in stats["train_folds"]]
    splits = load_splits(folds)
    train = [s for s in splits if s.is_train]
    heldout = {s.fire_id: s for s in splits if not s.is_train}
    unknown = [f for f in fire_ids if f not in heldout]
    if unknown:
        raise ValueError(f"not held-out under this split: {unknown}")

    hourly_train: list[Any] = []
    for s in train:
        hourly_train.extend(_windows_for(s.fire_id, 1, fit_stride))
    calibration = EllipseBaseline().calibrate_to_growth(
        hourly_train, train_fire_ids=[s.fire_id for s in train]
    )
    leaked = set(calibration.train_fire_ids) & set(heldout)
    if leaked:
        raise RuntimeError(f"ellipse was calibrated on held-out fires {sorted(leaked)}")

    models: dict[str, Any] = {
        "persistence": PersistenceBaseline(),
        "ellipse": EllipseBaseline(name="ellipse").with_calibration(calibration),
        "elmfire": CachedEnsembleModel(
            store,
            provenance={"split_fingerprint": split_fingerprint_value, "untrained": True},
        ),
    }

    per_fire: dict[str, Any] = {}
    pooled_all: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    pooled_growth: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    pooled_dormant: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    pooled_counts: dict[str, list[dict[str, Any]]] = {k: [] for k in models}
    for fire_id in fire_ids:
        s = heldout[fire_id]
        windows = _windows_for(fire_id, horizon_h, stride)
        entry: dict[str, Any] = {
            "fire_id": fire_id,
            "spatial_block_id": s.spatial_block_id,
            "n_windows": len(windows),
            "n_growth_windows": sum(1 for w in windows if w.truth_growth_cells() > 0),
            "models": {},
        }
        for name, model in models.items():
            all_res, counts = _score(model, windows, n_members, horizon_h, seed)
            growth_res = [
                r for w, r in zip(windows, all_res, strict=True) if w.truth_growth_cells() > 0
            ]
            dormant_res = [
                r for w, r in zip(windows, all_res, strict=True) if w.truth_growth_cells() == 0
            ]
            pooled_all[name].extend(all_res)
            pooled_growth[name].extend(growth_res)
            pooled_dormant[name].extend(dormant_res)
            pooled_counts[name].extend(counts)
            entry["models"][name] = {
                "all_windows": _headline(aggregate(all_res), horizon_h) if all_res else None,
                "growth_windows": (
                    _headline(aggregate(growth_res), horizon_h) if growth_res else None
                ),
                "dormant_windows": (
                    _headline(aggregate(dormant_res), horizon_h) if dormant_res else None
                ),
                "c6_2_validity": baseline_validity(
                    counts,
                    name=name,
                    scope=f"held-out {fire_id}",
                    null_model=name in NULL_MODELS,
                ),
            }
        per_fire[fire_id] = entry

    pooled = {
        name: {
            "all_windows": _headline(aggregate(pooled_all[name]), horizon_h),
            "growth_windows": (
                _headline(aggregate(pooled_growth[name]), horizon_h)
                if pooled_growth[name]
                else None
            ),
            "dormant_windows": (
                _headline(aggregate(pooled_dormant[name]), horizon_h)
                if pooled_dormant[name]
                else None
            ),
        }
        for name in models
    }

    control = _commensurability_control(per_fire, reference, calibration)
    return {
        "kind": "g5_elmfire_subset_of_heldout",
        "NOT_run_baselines": (
            "This table was NOT produced by a single run_baselines call. It uses that "
            "module's OWN _score, _headline and metrics.aggregate on the same windows, "
            "and `commensurability_control` compares its persistence and ellipse columns "
            "against a run_baselines payload at the same stride/members/seed."
        ),
        "fire_ids": list(fire_ids),
        "missing_heldout_fires": sorted(set(heldout) - set(fire_ids)),
        "horizon_h": horizon_h,
        "stride": stride,
        "n_members": n_members,
        "seed": seed,
        "fit_stride": fit_stride,
        "split_fingerprint": split_fingerprint_value,
        "ellipse_calibration": calibration.to_dict(),
        "per_fire": per_fire,
        "pooled_subset": pooled,
        "commensurability_control": control,
    }


def _commensurability_control(
    per_fire: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    calibration: Any,
) -> dict[str, Any]:
    """persistence and ellipse must be IDENTICAL to the run_baselines payload.

    Compared as JSON text after canonical serialisation, which is deliberately
    the strictest available reading: a float that differs in the last bit is a
    difference, and a difference means the two tables were not built the same
    way, whatever the size of the number.
    """
    if reference is None:
        return {"ran": False, "reason": "no reference run_baselines payload supplied"}
    ref_cal = (reference.get("ellipse_calibration") or {}).get("rule_of_record")
    checks: list[dict[str, Any]] = []
    for fire_id, entry in per_fire.items():
        ref_entry = ((reference.get("per_fire") or {}).get(fire_id) or {}).get("models") or {}
        for name in ("persistence", "ellipse"):
            mine = entry["models"].get(name)
            theirs = ref_entry.get(name)
            same = json.dumps(mine, sort_keys=True, default=str) == json.dumps(
                theirs, sort_keys=True, default=str
            )
            checks.append({"fire_id": fire_id, "model": name, "identical": bool(same)})
    n_bad = sum(1 for c in checks if not c["identical"])
    return {
        "ran": True,
        "reference_kind": reference.get("kind"),
        "reference_stride": reference.get("stride"),
        "calibration_identical": json.dumps(calibration.to_dict(), sort_keys=True, default=str)
        == json.dumps(ref_cal, sort_keys=True, default=str),
        "n_checked": len(checks),
        "n_not_identical": n_bad,
        "checks": checks,
        "verdict": (
            "COMMENSURABLE - every shared column is bit-identical to the run_baselines "
            "payload, so the ELMFIRE column sits on the same table"
            if n_bad == 0
            else "NOT COMMENSURABLE - a shared column differs; the ELMFIRE column may not "
            "be quoted beside a run_baselines number"
        ),
    }


def audit_store_keys(store: EnsembleStore) -> dict[str, Any]:
    """Re-derive every stored entry's key FROM THE TENSOR and require a match.

    The producer and the consumer build the window by two different routes --
    the worker calls ``forecast_inputs(ds, t0, ...)`` directly, the scorer walks
    ``iter_windows`` -- and they agree only because ``iter_windows`` yields
    exactly that call. That is a fact about a line of code in another package,
    so it is checked rather than relied on: a stored array whose key does not
    reproduce from its own recorded ``(fire_id, t0, seed)`` would be scored
    against a window it was not produced for, and nothing downstream could tell.
    """
    from wildfire_nowcast.model.inputs import forecast_inputs

    rows: list[dict[str, Any]] = []
    by_fire: dict[str, list[str]] = {}
    for p in sorted(store.root.glob("*.npz")):
        meta = store.meta(p.stem)
        by_fire.setdefault(str(meta["fire_id"]), []).append(p.stem)
    for fire_id, keys in by_fire.items():
        ds = open_tensor(fire_tensor_path(fire_id))
        try:
            for key in keys:
                meta = store.meta(key)
                w = forecast_inputs(ds, int(meta["t0"]), int(meta["horizon_h"]), fire_id=fire_id)
                recomputed = window_key(
                    w.x0,
                    w.weather,
                    int(meta["n_members"]),
                    int(meta["horizon_h"]),
                    int(meta["seed"]),
                )
                rows.append(
                    {
                        "fire_id": fire_id,
                        "t0": int(meta["t0"]),
                        "stored_key": key,
                        "recomputed_key": recomputed,
                        "match": recomputed == key,
                    }
                )
        finally:
            ds.close()
    bad = [r for r in rows if not r["match"]]
    return {
        "n_entries": len(rows),
        "n_mismatched": len(bad),
        "mismatched": bad[:20],
        "verdict": (
            "OK - every stored ensemble's key reproduces from its own (fire, t0, seed)"
            if not bad
            else "HARD FAIL - a stored ensemble does not belong to the window it is keyed to"
        ),
    }


#: The columns G3 is stated in. G3 is, verbatim and stated inline because the
#: file it is tracked in is not part of this repository: "dispersion ratio in
#: [0.8,1.2] AND reliability within +/-10 pts on held-out fires". The columns
#: below are that sentence spelled with the keys the CONTRACT chose
#: rather than the keys the sentence uses. `dispersion_ratio` is carried
#: because the sentence names it and C6.1 BARS it from adjudicating - printing
#: it beside `area_dispersion_ratio` is what stops a reader quoting the barred
#: one by accident.
G3_COLUMNS: tuple[tuple[str, str], ...] = (
    ("band_area_dispersion_ratio", "G3 dispersion CRITERION (C6.1; bar [0.8,1.2])"),
    ("dispersion_ratio", "REPORTED ONLY - C6.1 BARS this from any gate"),
    ("band_calibration_error", "G3 calibration CRITERION (growth_band, C6.6)"),
    ("band_brier_by_horizon", "Brier 1/2/3 h in growth_band"),
    ("arrival_crps", "REPORTED ONLY - C6.6 bars it (anti-monotone)"),
    ("band_best_member_iou_shape_masked", "C6.4 location criterion"),
)


def g5_table(payload: Mapping[str, Any], *, stratum: str = "growth_windows") -> str:
    """A flat text table: one row per (block, model), one column per G3 number.

    Deliberately plain. The point of G5 is that four columns are comparable, and
    a table a reader can check against the JSON by eye is worth more here than a
    figure they have to trust.
    """
    per_fire = payload.get("per_fire") or {}
    keys = [k for k, _ in G3_COLUMNS]
    lines = [
        f"stratum={stratum}  stride={payload.get('stride')}  members={payload.get('n_members')}  "
        f"seed={payload.get('seed')}  split={payload.get('split_fingerprint')}",
        "block fire                       model        " + "  ".join(f"{k[:22]:>22}" for k in keys),
    ]
    for fire_id, entry in per_fire.items():
        for name, res in (entry.get("models") or {}).items():
            head = (res or {}).get(stratum) or {}
            cells = []
            for k in keys:
                v = head.get(k)
                if isinstance(v, dict):
                    v = "/".join(
                        "--" if v.get(str(h)) is None else f"{float(v[str(h)]):.4f}"
                        for h in (1, 2, 3)
                    )
                    cells.append(f"{v:>22}")
                elif v is None:
                    cells.append(f"{'--':>22}")
                else:
                    cells.append(f"{float(v):>22.4f}")
            # A missing block id prints as `--` rather than raising. This is a
            # DIAGNOSTIC table; one that dies on an absent field fails exactly
            # when a payload is malformed, which is when it is wanted most.
            block_id = entry.get("spatial_block_id")
            block_txt = "--" if block_id is None else str(int(block_id))
            lines.append(f"{block_txt:>5} {fire_id:<26} {name:<12} " + "  ".join(cells))
    return "\n".join(lines)
