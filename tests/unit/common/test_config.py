"""``common/config.py`` - hydra-style yaml, and the FORM the resolved copy is written in.

A resolved config is written into every run directory and is the thing a reader
opens first when asking what a run was. Both properties checked here are about
that copy being readable and diffable: keys stay in the order their author wrote
them, and nested blocks stay indented rather than collapsing into one line. Neither
changes what the config MEANS, which is why neither is visible to any round-trip
check, and both are one keyword apart in the source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wildfire_nowcast.common.config import (
    apply_overrides,
    deep_merge,
    dump_yaml,
    get_in,
    load_yaml,
    parse_override,
)


def test_a_written_config_keeps_its_key_order_and_its_indentation(tmp_path: Path) -> None:
    """Deliberately anti-alphabetical keys, so sorted output is visible as sorted.

    Two run configs are compared by eye and by ``diff``. Re-ordering the keys
    turns every diff into a whole-file diff, and flowing the nested blocks turns
    a readable tree into one long line, so a reviewer stops reading them. Neither
    is caught by loading the file back, because both round trip perfectly.
    """
    config = {"zeta": 1, "alpha": {"inner": 2, "beta": 3}, "middle": [4, 5]}
    path = dump_yaml(config, tmp_path / "nested" / "resolved.yaml")

    text = path.read_text()
    order = [
        line.split(":")[0] for line in text.splitlines() if line and not line.startswith((" ", "-"))
    ]
    assert order == ["zeta", "alpha", "middle"], (
        f"top-level keys were written as {order}, not in the order the author wrote them"
    )
    assert "\n  inner:" in text, (
        "the nested mapping was not written as an indented block; a flow-style config is "
        f"one line long and stops being reviewable:\n{text}"
    )
    assert load_yaml(path) == config, "the round trip changed the config"


def test_deep_merge_does_not_mutate_either_input() -> None:
    """An override that edited the base in place would leak into the next load."""
    base = {"model": {"lr": 0.1, "layers": 3}, "seed": 1}
    override = {"model": {"lr": 0.5}}
    merged = deep_merge(base, override)

    assert merged == {"model": {"lr": 0.5, "layers": 3}, "seed": 1}
    assert base == {"model": {"lr": 0.1, "layers": 3}, "seed": 1}
    assert override == {"model": {"lr": 0.5}}


def test_an_override_string_is_parsed_as_yaml_and_reaches_a_nested_key() -> None:
    """``a.b=3`` must set an int, not the string ``"3"``: a config is typed."""
    dotted, value = parse_override("model.lr=0.5")
    assert dotted == ["model", "lr"] and value == 0.5 and isinstance(value, float)

    out = apply_overrides({"model": {"lr": 0.1}}, ["model.lr=0.5", "seed=7"])
    assert out == {"model": {"lr": 0.5}, "seed": 7}
    assert get_in(out, "model.lr") == 0.5
    with pytest.raises(KeyError):
        get_in(out, "model.missing")
    assert get_in(out, "model.missing", default=None) is None
