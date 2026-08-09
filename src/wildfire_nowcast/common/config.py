"""Hydra-style yaml config loading, without the hydra runtime.

One experiment = one yaml in ``configs/`` (C7). A config may compose others::

    # configs/synthetic_smoke.yaml
    defaults: [base]
    data:
      source: synthetic

Merge order is: each entry of ``defaults`` in order, then the file's own keys,
then any command-line overrides. Mappings merge recursively; lists and scalars
replace wholesale (list-merging is the kind of cleverness that makes configs
impossible to reason about).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from wildfire_nowcast.common.paths import configs_dir

__all__ = [
    "load_yaml",
    "dump_yaml",
    "deep_merge",
    "parse_override",
    "apply_overrides",
    "load_config",
    "get_in",
    "to_plain",
]

_DEFAULTS_KEY = "defaults"
_MISSING = object()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a yaml file into a dict (an empty file yields ``{}``)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no config at {p}")
    data = yaml.safe_load(p.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{p} must contain a yaml mapping at the top level, got {type(data)}")
    return data


def to_plain(obj: Any) -> Any:
    """Coerce a config tree to plain builtins so ``yaml.safe_dump`` accepts it.

    Configs routinely pick up things that *look* like builtins but are not:
    numpy scalars, ``pathlib.Path``, and str subclasses such as
    ``torch.torch_version.TorchVersion``. ``SafeDumper`` dispatches on exact
    type and refuses all of them, which turns a provenance record into a crash
    at the end of a long run. Normalise once, here.
    """
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, str | bytes):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, Sequence | set | frozenset):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalars
        try:
            return to_plain(obj.item())
        except Exception:  # pragma: no cover - defensive
            pass
    if obj is None:
        return None
    return str(obj)


def dump_yaml(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a config to yaml, creating parent directories."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(to_plain(config), sort_keys=False, default_flow_style=False))
    return p


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base``; neither input is mutated."""
    out = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def parse_override(item: str) -> tuple[list[str], Any]:
    """Parse ``"a.b.c=value"``; the value is yaml-parsed so types survive."""
    if "=" not in item:
        raise ValueError(f"override {item!r} must be of the form key.path=value")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"override {item!r} has an empty key")
    return key.split("."), yaml.safe_load(raw)


def apply_overrides(config: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply dotted ``key.path=value`` overrides, creating intermediate dicts."""
    out = deepcopy(dict(config))
    for item in overrides:
        parts, value = parse_override(item)
        node: dict[str, Any] = out
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value
    return out


def _resolve_include(name: str, base_dir: Path, root: Path) -> Path:
    """Resolve a ``defaults:`` entry to a file, relative to the file then root."""
    candidates = []
    for stem in (name, f"{name}.yaml", f"{name}.yml"):
        candidates.extend([base_dir / stem, root / stem])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config include {name!r} not found near {base_dir} or {root}")


def _compose(path: Path, root: Path, seen: tuple[Path, ...]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in (*seen, resolved))
        raise ValueError(f"circular config includes: {chain}")
    raw = load_yaml(resolved)
    includes = raw.pop(_DEFAULTS_KEY, [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise TypeError(f"{resolved}: '{_DEFAULTS_KEY}' must be a string or list of strings")

    merged: dict[str, Any] = {}
    for name in includes:
        include_path = _resolve_include(str(name), resolved.parent, root)
        merged = deep_merge(merged, _compose(include_path, root, (*seen, resolved)))
    return deep_merge(merged, raw)


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
    *,
    configs_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load and fully resolve one experiment config.

    Returns a plain dict — the *resolved* config that
    :func:`wildfire_nowcast.common.runs.create_run_dir` records verbatim, so a
    run is reproducible from its own directory without consulting ``configs/``.
    """
    root = Path(configs_root) if configs_root else configs_dir()
    p = Path(path)
    if not p.is_file():
        candidate = _resolve_include(str(path), root, root)
        p = candidate
    config = _compose(p, root, ())
    if overrides:
        config = apply_overrides(config, overrides)
    return _stamp_interfaces_version(config, source=p)


#: [A14] The key ``runs/{run_id}/config.yaml`` carries so a result can never be
#: read under a different contract version than the one that produced it.
INTERFACES_VERSION_KEY = "interfaces_version"


def _stamp_interfaces_version(config: dict[str, Any], *, source: Path) -> dict[str, Any]:
    """Stamp — do not merely validate — the contract version into a resolved config.

    [A14, ADR-033 (1)] ``configs/base.yaml`` used to carry ``interfaces_version:
    v2.12`` as a LITERAL, which made it the THIRD place one fact was written down
    (INTERFACES.md line 1, ``contract.CONTRACT_VERSION``, here) and the third
    place it could go stale. ``tests/test_common.py`` pinned them equal, so a
    contract bump turned the build red until somebody edited a yaml — the same
    mechanically-forced edit that ADR-033 ruled must be fixed at the mechanism.

    Now the resolved config is STAMPED from
    :data:`~wildfire_nowcast.common.contract.CONTRACT_VERSION`, which is itself
    derived from INTERFACES.md line 1. A config that declares the key explicitly
    and DISAGREES is an error rather than an override: an experiment cannot
    unilaterally claim conformance to a contract version that is not the one in
    force, and silently honouring its claim would put a false attestation into
    every run directory it produced.

    Imported inside the function on purpose — ``contract`` pulls in xarray, and
    loading a yaml should not cost that.
    """
    from wildfire_nowcast.common.contract import CONTRACT_VERSION

    declared = config.get(INTERFACES_VERSION_KEY)
    if declared is not None and str(declared) != CONTRACT_VERSION:
        raise ValueError(
            f"{source}: declares {INTERFACES_VERSION_KEY}={declared!r} but the contract in force "
            f"is {CONTRACT_VERSION} (INTERFACES.md line 1). This key is STAMPED from the "
            "contract, not asserted by a config: a run directory recording a version its code "
            "did not enforce is the stale-checker hazard with a paper trail. Delete the key."
        )
    config[INTERFACES_VERSION_KEY] = CONTRACT_VERSION
    return config


def get_in(config: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    """Fetch ``config["a"]["b"]`` as ``get_in(config, "a.b")``."""
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise KeyError(f"{dotted!r} not present in config")
            return default
        node = node[part]
    return node
