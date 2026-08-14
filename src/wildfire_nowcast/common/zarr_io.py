"""Reading and writing C1/C2/C3 artifacts.

The INTERFACES version this layout conforms to is
:data:`wildfire_nowcast.common.contract.CONTRACT_VERSION`, derived from
INTERFACES.md line 1 and stamped into every store's ``contract_version`` attr
below. It is not restated here: this docstring said ``v2.5`` for seven contract
versions, which is precisely how a reader ends up trusting a stale number.

Everyone writes tensors through :func:`build_tensor_dataset` +
:func:`write_tensor` (or the one-shot :func:`write_fire`) so that dtypes,
coordinate conventions, channel order and attrs are correct by construction
rather than by vigilance. Anything produced here passes
:mod:`wildfire_nowcast.common.contract`; if it ever does not, the contract test
is right and this module is wrong.

Store layout (C1, v2 — ADR-006 P2)::

    fire_state  uint8    (time, y, x)              values {0,1,2}
    features    float32  (time, channel, y, x)     C1 channels 1-13

A zarr array holds exactly one dtype, so v1's single "float32 except fire_state
uint8" array was unsatisfiable. Consumers that want the old per-channel view use
:func:`get_channel` / :func:`to_channel_dataset`; consumers that want the model
view use :func:`stack_channels`. Nobody should index ``features`` by a hardcoded
integer — ``channel_index_offset`` exists so the v1 indices still hold, and
:func:`get_channel` applies it for you.
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr

from wildfire_nowcast.common.contract import (
    ATTR_CHANNEL_INDEX_OFFSET,
    ATTR_CHANNEL_ORDER,
    ATTR_TIME_CONVENTION,
    ATTR_TIME_END,
    ATTR_TIME_START,
    CATEGORICAL_CHANNELS,
    CHANNEL_INDEX,
    CHANNEL_INDEX_OFFSET,
    CHANNELS,
    CONTRACT_VERSION,
    FEATURE_CHANNELS,
    FEATURES,
    FIRE_STATE,
    FIRE_STATE_VALUES,
    MANIFEST_KEYS,
    MIN_TRAIN_BLOCKS_FOR_REPORTING,
    STATIC_CHANNELS,
    TIME_CONVENTION,
    channel_dtype,
    feature_index,
)
from wildfire_nowcast.common.contract import (
    CHANNEL_UNITS as _UNITS,
)
from wildfire_nowcast.common.grid import Grid

__all__ = [
    "hourly_times",
    "utc_now_iso",
    "iso_naive",
    "build_tensor_dataset",
    "write_tensor",
    "open_tensor",
    "get_channel",
    "channel_values",
    "to_channel_dataset",
    "stack_channels",
    "to_stacked_dataarray",
    "build_manifest",
    "write_manifest",
    "read_manifest",
    "build_norm_stats",
    "write_norm_stats",
    "read_norm_stats",
    "compute_norm_stats",
    "write_fire",
    "fire_state_of",
]

_TIME_ENCODING = {"units": "seconds since 1970-01-01", "dtype": "int64"}

#: C3's identity transform for a class label (C3.2). Standardising an FBFM40
#: class id, or a {0,1,2} state, is meaningless arithmetic on a label.
CATEGORICAL_NOTE = (
    "fire_state (channel 0) and fuel_model_id (channel 9) are CATEGORICAL: mean=0 / std=1 is "
    "an identity transform, i.e. DO NOT standardise them. Embed the class id instead (C3.2)."
)


# --------------------------------------------------------------------------
# time helpers
# --------------------------------------------------------------------------


def iso_naive(t: Any) -> str:
    """Format a time as a naive-UTC ISO string (no ``Z``, no offset) per C1."""
    if isinstance(t, str):
        t = np.datetime64(t)
    if isinstance(t, datetime):
        t = t.replace(tzinfo=None) if t.tzinfo is None else t.astimezone(UTC).replace(tzinfo=None)
        t = np.datetime64(t)
    return str(np.datetime64(t, "s"))


def utc_now_iso() -> str:
    """Current time as a naive-UTC ISO string."""
    return iso_naive(datetime.now(UTC).replace(tzinfo=None))


def hourly_times(start: Any, n_hours: int) -> np.ndarray:
    """``n_hours`` hourly ``datetime64[s]`` steps beginning at ``start``.

    Times are END-OF-HOUR stamps (C1.3): ``t`` labels the hour ending at ``t``.
    """
    if n_hours < 1:
        raise ValueError(f"n_hours must be >= 1, got {n_hours}")
    t0 = np.datetime64(start, "s")
    return t0 + np.arange(n_hours, dtype="int64") * np.timedelta64(1, "h")


# --------------------------------------------------------------------------
# C1 tensor
# --------------------------------------------------------------------------


def build_tensor_dataset(
    channels: Mapping[str, np.ndarray],
    grid: Grid,
    times: Sequence[Any] | np.ndarray,
    *,
    attrs: Mapping[str, Any] | None = None,
    require_all: bool = True,
) -> xr.Dataset:
    """Assemble a C1-conformant :class:`xarray.Dataset` in the v2 layout.

    Parameters
    ----------
    channels
        Channel name -> array, keyed by the C1 names (not by index). Each array
        is either ``(time, y, x)`` or, for a static channel, ``(y, x)`` — 2-D
        input is broadcast over time for you ("static, repeated over time", C1).
    grid
        Spatial grid; supplies ``x``/``y`` coordinates and the CRS attrs.
    times
        Hourly, strictly increasing, naive-UTC, end-of-hour times.
    require_all
        Require all 14 C1 channels. Set ``False`` only for interim,
        deliberately-partial stores (ADR-003) that never land at the C1 path;
        such a store carries ``fire_state`` alone and no ``features`` array.

    dtypes are coerced to the contract (uint8 fire_state, float32 features);
    a mismatch in *shape* is an error rather than a silent reshape.
    """
    t = np.asarray(times)
    if t.dtype.kind in "US":
        t = t.astype("datetime64[s]")
    if not np.issubdtype(t.dtype, np.datetime64):
        raise TypeError(f"times must be datetime64 or ISO strings, got dtype {t.dtype}")
    t = t.astype("datetime64[s]")
    n_t = int(t.size)
    ny, nx = grid.shape

    if require_all:
        missing = [c for c in CHANNELS if c not in channels]
        if missing:
            raise ValueError(
                f"missing C1 channels {missing}. A partial tensor must not be written to the "
                f"C1 path (ADR-003); pass require_all=False and write to data/interim/."
            )
    unknown = [c for c in channels if c not in CHANNEL_INDEX]
    if unknown:
        raise ValueError(f"unknown channels {unknown}; C1 channel list is {CHANNELS}")
    if FIRE_STATE not in channels:
        raise ValueError(f"every C1 store carries {FIRE_STATE!r}; it is the label channel")

    def _conform(name: str) -> np.ndarray:
        arr = np.asarray(channels[name])
        dtype = channel_dtype(name)
        if arr.ndim == 2:
            if arr.shape != (ny, nx):
                raise ValueError(
                    f"channel {name!r}: static array shape {arr.shape} != grid {(ny, nx)}"
                )
            return np.broadcast_to(arr.astype(dtype, copy=False), (n_t, ny, nx))
        if arr.ndim == 3:
            if arr.shape != (n_t, ny, nx):
                raise ValueError(f"channel {name!r}: shape {arr.shape} != expected {(n_t, ny, nx)}")
            return arr.astype(dtype, copy=False)
        raise ValueError(f"channel {name!r}: expected 2-D or 3-D array, got {arr.ndim}-D")

    data_vars: dict[str, xr.DataArray] = {
        FIRE_STATE: xr.DataArray(
            _conform(FIRE_STATE),
            dims=("time", "y", "x"),
            attrs={
                "units": _UNITS[FIRE_STATE],
                "channel_index": CHANNEL_INDEX[FIRE_STATE],
                "flag_values": list(FIRE_STATE_VALUES),
                "flag_meanings": "unburned burning burned_out",
                "state_rule": "fireline_v2",
            },
        )
    }

    coords: dict[str, Any] = {
        "time": ("time", t),
        "y": ("y", grid.y_coords),
        "x": ("x", grid.x_coords),
    }

    present_features = [c for c in FEATURE_CHANNELS if c in channels]
    if present_features:
        if len(present_features) != len(FEATURE_CHANNELS) and require_all:  # pragma: no cover
            raise ValueError("require_all=True but not every feature channel was supplied")
        stack = np.empty((n_t, len(present_features), ny, nx), dtype=np.float32)
        for i, name in enumerate(present_features):
            stack[:, i] = _conform(name)
        data_vars[FEATURES] = xr.DataArray(
            stack,
            dims=("time", "channel", "y", "x"),
            attrs={
                # Position along `channel` + this == the v1 channel index (C1 v2).
                ATTR_CHANNEL_INDEX_OFFSET: CHANNEL_INDEX_OFFSET,
                "channel_units": [_UNITS[c] for c in present_features],
                "static_channels": [c for c in present_features if c in STATIC_CHANNELS],
            },
        )
        coords["channel"] = ("channel", np.array(present_features, dtype="<U24"))

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs.update(grid.attrs())
    ds.attrs[ATTR_TIME_START] = iso_naive(t[0])
    ds.attrs[ATTR_TIME_END] = iso_naive(t[-1])
    # C1.3: silently catastrophic if wrong, so the store self-describes rather
    # than relying on its manifest travelling with it.
    ds.attrs[ATTR_TIME_CONVENTION] = TIME_CONVENTION
    ds.attrs[ATTR_CHANNEL_ORDER] = list(CHANNELS)
    ds.attrs[ATTR_CHANNEL_INDEX_OFFSET] = CHANNEL_INDEX_OFFSET
    ds.attrs["contract"] = "C1"
    ds.attrs["contract_version"] = CONTRACT_VERSION
    ds["x"].attrs.update({"units": "m", "standard_name": "projection_x_coordinate"})
    ds["y"].attrs.update({"units": "m", "standard_name": "projection_y_coordinate"})
    if attrs:
        ds.attrs.update(dict(attrs))
    return ds


def write_tensor(
    ds: xr.Dataset,
    path: str | Path,
    *,
    mode: str = "w",
    time_chunk: int = 24,
) -> Path:
    """Write a C1 dataset to a zarr store and return the path.

    ``features`` is chunked one channel per chunk: every consumer that matters
    (the contract checker, per-channel QA, static-channel reads) touches a
    single channel at a time, and whole-tensor reads are unaffected.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    n_t = int(ds.sizes["time"])
    ny, nx = int(ds.sizes["y"]), int(ds.sizes["x"])
    t_chunk = min(time_chunk, n_t)
    encoding: dict[str, dict[str, Any]] = {}
    for name in ds.data_vars:
        # xarray types variable names as `Hashable`; C1 stores use `str`.
        key = cast("str", name)
        ndim = ds[name].ndim
        if ndim == 3:
            encoding[key] = {"chunks": (t_chunk, ny, nx)}
        elif ndim == 4:
            encoding[key] = {"chunks": (t_chunk, 1, ny, nx)}
    encoding["time"] = dict(_TIME_ENCODING)

    with warnings.catch_warnings():
        # zarr 3 warns that consolidated metadata is not in the v3 spec. We keep
        # it because it makes opening many-chunk stores far cheaper, and every
        # reader here goes through open_tensor(), which falls back gracefully.
        warnings.filterwarnings("ignore", message=".*[Cc]onsolidated metadata.*")
        # `mode` stays `str` in this signature: xarray's stub wants a Literal,
        # and narrowing a public parameter to satisfy a stub would push the
        # constraint onto every caller for no runtime benefit.
        ds.to_zarr(p, mode=mode, consolidated=True, encoding=encoding)  # type: ignore[call-overload]
    return p


def open_tensor(path: str | Path) -> xr.Dataset:
    """Open a tensor store read-only with time decoded to datetime64."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no tensor store at {p}")
    try:
        return cast("xr.Dataset", xr.open_zarr(p, consolidated=True, decode_timedelta=False))
    except Exception:
        return cast("xr.Dataset", xr.open_zarr(p, consolidated=False, decode_timedelta=False))


# --------------------------------------------------------------------------
# channel access (the v2 layout, made ordinary)
# --------------------------------------------------------------------------


def get_channel(ds: xr.Dataset, name: str) -> xr.DataArray:
    """One C1 channel as a ``(time, y, x)`` DataArray, by NAME.

    Works for ``fire_state`` (its own variable) and for any of the 13 feature
    channels (a slice of ``features``), so callers never have to know which is
    which, and never hardcode a channel integer.
    """
    if name == FIRE_STATE:
        return ds[FIRE_STATE]
    if name not in CHANNEL_INDEX:
        raise KeyError(f"{name!r} is not a C1 channel; expected one of {CHANNELS}")
    if FEATURES not in ds:
        raise KeyError(
            f"store has no {FEATURES!r} array, so channel {name!r} is unavailable "
            "(an ADR-003 interim label store carries fire_state only)"
        )
    if "channel" in ds.coords:
        names = [str(c) for c in np.atleast_1d(ds["channel"].values)]
        if name in names:
            return ds[FEATURES].isel(channel=names.index(name), drop=True)
    return ds[FEATURES].isel(channel=feature_index(name), drop=True)


def channel_values(ds: xr.Dataset, name: str, *, dtype: Any = None) -> np.ndarray:
    """:func:`get_channel` as a numpy array."""
    values = np.asarray(get_channel(ds, name).values)
    return values if dtype is None else values.astype(dtype, copy=False)


def to_channel_dataset(ds: xr.Dataset, channels: Sequence[str] = CHANNELS) -> xr.Dataset:
    """Explode the v2 store into one variable per channel (the v1 read view).

    Convenience for consumers that prefer named variables. It materialises the
    data, so prefer :func:`get_channel` in a loop for large stores.
    """
    return xr.Dataset(
        {name: get_channel(ds, name) for name in channels},
        coords={k: v for k, v in ds.coords.items() if k != "channel"},
        attrs=dict(ds.attrs),
    )


def stack_channels(
    ds: xr.Dataset,
    channels: Sequence[str] = CHANNELS,
    *,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Materialise the C1 ``(time, channel, y, x)`` view as one array.

    The index along axis 1 is the position in ``channels``; with the default it
    is exactly the v1 channel index, ``fire_state`` included.
    """
    missing = [
        c
        for c in channels
        if c not in CHANNEL_INDEX
        or (c == FIRE_STATE and FIRE_STATE not in ds)
        or (c != FIRE_STATE and FEATURES not in ds)
    ]
    if missing:
        raise KeyError(f"dataset is missing channels {missing}")
    return np.stack([channel_values(ds, c, dtype=dtype) for c in channels], axis=1)


def to_stacked_dataarray(ds: xr.Dataset, channels: Sequence[str] = CHANNELS) -> xr.DataArray:
    """Same as :func:`stack_channels` but labelled, dims ``(time, channel, y, x)``."""
    return xr.DataArray(
        stack_channels(ds, channels),
        dims=("time", "channel", "y", "x"),
        coords={
            "time": ds["time"],
            "channel": np.array(list(channels), dtype="<U24"),
            "y": ds["y"],
            "x": ds["x"],
        },
        attrs=dict(ds.attrs),
    )


def fire_state_of(ds: xr.Dataset) -> np.ndarray:
    """The ``(time, y, x)`` uint8 label array."""
    return np.asarray(ds[FIRE_STATE].values, dtype=np.uint8)


# --------------------------------------------------------------------------
# C2 manifest
# --------------------------------------------------------------------------


def build_manifest(
    *,
    fire_id: str,
    gofer_version: str,
    bbox_5070: Sequence[float],
    ignition_time_utc: Any,
    n_hours: int,
    cv_fold: int,
    spatial_block_id: int,
    provenance: Mapping[str, str],
    norm_stats_path: str,
    fuel_vintage_lag_years: int,
    n_ignition_components: int,
    created_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a C2 manifest dict. Extra keys are allowed and preserved (v2.1).

    ``spatial_block_id`` (v2, C3.1) is required: buffered domains overlap, so
    folds are blocked on connected components and a fire without a block id
    cannot be assigned to one safely.

    ``fuel_vintage_lag_years`` and ``n_ignition_components`` (v2.7, ADR-014) are
    required and have NO DEFAULT, deliberately. A default of 1 component would
    be silently wrong on exactly the two fires the clause exists for —
    ``2020_july_complex`` (2) and SCU (3) — and a wrong number a reader trusts
    is worse than a ``TypeError`` at build time. Same argument as
    ``spatial_block_id`` at A3: C2 compliance is structural, so a non-conformant
    manifest cannot be constructed through the one implementation (C0).
    """
    man: dict[str, Any] = {
        "fire_id": str(fire_id),
        "gofer_version": str(gofer_version),
        "bbox_5070": [float(v) for v in bbox_5070],
        "ignition_time_utc": iso_naive(ignition_time_utc),
        "n_hours": int(n_hours),
        "cv_fold": int(cv_fold),
        "spatial_block_id": int(spatial_block_id),
        "fuel_vintage_lag_years": int(fuel_vintage_lag_years),
        "n_ignition_components": int(n_ignition_components),
        "created_utc": created_utc or utc_now_iso(),
        "provenance": {str(k): str(v) for k, v in provenance.items()},
        "norm_stats_path": str(norm_stats_path),
    }
    if extra:
        man.update({k: v for k, v in extra.items() if k not in MANIFEST_KEYS})
    return man


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a C2 manifest.json."""
    return _write_json(dict(manifest), path)


def read_manifest(path: str | Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# C3 normalization stats
# --------------------------------------------------------------------------


def build_norm_stats(
    mean: Mapping[str, float],
    std: Mapping[str, float],
    train_folds: Sequence[int],
    n_train_blocks: int,
    *,
    note: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a C3 norm-stats dict: TOP-LEVEL ``channel_order``/``mean``/``std``.

    Categorical channels (0 and 9) are forced to the identity transform, and
    ``n_train_blocks`` is mandatory — with ``< 2`` the file is additionally
    marked ``bootstrap: true`` so a consumer reading only the JSON can tell that
    it is plumbing-grade, not reporting-grade (C3.2, C3.3).
    """
    missing = [c for c in CHANNELS if c not in mean or c not in std]
    if missing:
        raise ValueError(f"norm stats need one mean and one std per channel; missing {missing}")
    bad_blocks = isinstance(n_train_blocks, bool) or not isinstance(n_train_blocks, int)
    if bad_blocks or n_train_blocks < 1:
        raise ValueError(f"n_train_blocks must be a positive int (C3.3), got {n_train_blocks!r}")

    means = {c: (0.0 if c in CATEGORICAL_CHANNELS else float(mean[c])) for c in CHANNELS}
    stds = {c: (1.0 if c in CATEGORICAL_CHANNELS else float(std[c])) for c in CHANNELS}

    stats: dict[str, Any] = {
        "channel_order": list(CHANNELS),
        "mean": means,
        "std": stds,
        "train_folds": [int(f) for f in train_folds],
        "n_train_blocks": int(n_train_blocks),
        "created_utc": utc_now_iso(),
        "categorical_identity_note": CATEGORICAL_NOTE,
    }
    if n_train_blocks < MIN_TRAIN_BLOCKS_FOR_REPORTING:
        stats["bootstrap"] = True
        stats["bootstrap_note"] = (
            f"BOOTSTRAP: n_train_blocks={n_train_blocks} < {MIN_TRAIN_BLOCKS_FOR_REPORTING}. The "
            "only train fire is also the only landscape. Valid for plumbing; never for a number "
            "that appears in a gate (C3.3)."
        )
    if note:
        stats["note"] = note
    if extra:
        stats.update({k: v for k, v in extra.items() if k not in stats})
    return stats


def write_norm_stats(stats: Mapping[str, Any], path: str | Path) -> Path:
    return _write_json(dict(stats), path)


def read_norm_stats(path: str | Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(Path(path).read_text()))


def compute_norm_stats(
    datasets: Sequence[xr.Dataset],
    train_folds: Sequence[int],
    *,
    spatial_block_ids: Sequence[int] | None = None,
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """Per-channel mean/std over the given (TRAIN-fold) datasets.

    ``spatial_block_ids`` is the block id of each dataset; the count of distinct
    ids becomes ``n_train_blocks`` (C3.3). Omit it only when every dataset comes
    from one landscape — the result is then explicitly a bootstrap file.

    Categorical channels are emitted as the identity transform. Degenerate
    channels get ``std = max(std, min_std)`` so downstream division is safe.
    """
    if not datasets:
        raise ValueError("need at least one dataset to compute norm stats")
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for name in CHANNELS:
        total = 0.0
        total_sq = 0.0
        count = 0
        for ds in datasets:
            arr = channel_values(ds, name, dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            total += float(finite.sum())
            total_sq += float(np.square(finite).sum())
            count += int(finite.size)
        if count == 0:
            mean[name], std[name] = 0.0, min_std
            continue
        mu = total / count
        var = max(total_sq / count - mu * mu, 0.0)
        mean[name] = mu
        std[name] = max(float(np.sqrt(var)), min_std)

    n_blocks = len(set(int(b) for b in spatial_block_ids)) if spatial_block_ids else 1
    return build_norm_stats(mean, std, train_folds, n_blocks)


# --------------------------------------------------------------------------
# one-shot fire writer
# --------------------------------------------------------------------------


def write_fire(
    ds: xr.Dataset,
    manifest: Mapping[str, Any],
    out_dir: str | Path,
    *,
    tensor_name: str = "tensor.zarr",
    mode: str = "w",
) -> tuple[Path, Path]:
    """Write ``tensor.zarr`` + ``manifest.json`` into ``out_dir`` (the C1/C2 pair)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensor_path = write_tensor(ds, out / tensor_name, mode=mode)
    manifest_path = write_manifest(manifest, out / "manifest.json")
    return tensor_path, manifest_path


def _write_json(obj: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2, sort_keys=False)
            fh.write("\n")
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p
