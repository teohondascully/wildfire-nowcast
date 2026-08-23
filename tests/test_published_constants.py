"""Every published constant in ``model/`` and ``eval/``, pinned to the artifact it cites.

THE DEFECT THIS CLOSES
----------------------
``model/train.py``'s ``M6B_RHO`` is tagged *MEASURED, NOT CHOSEN*. Its comment
block names :func:`innovation_autocorrelation`, cites
``runs/m6_innovation_autocorr.json`` and quotes five numbers out of it. All five
were true when measured. Nothing checked them: the function had zero call sites
in ``src``, ``tests`` or ``tools``, so the constant, the prose and the artifact
were three independent copies of one fact and any one of them could drift while
the suite stayed green. ``tests/test_latent_published_branch.py`` pins ``0.5144``
into the published latent config, which makes four copies.

The same shape held for every other measured constant in the two packages. This
file makes the artifact the single source and derives everything else from it.

HOW A CLAIM IS CHECKED HERE, AND WHY IT IS NOT A GREP
-----------------------------------------------------
Two rules, both learned from a near miss: searching the artifact for the literal
string ``0.5144`` returns nothing, because the stored value is
``0.5143806420996772``. Stopping there would have produced a false accusation of
fabrication.

1. **Numbers are compared numerically, at a stated precision.** Never as strings.
2. **Prose is compared to a fragment BUILT FROM THE ARTIFACT.** The expected text
   is formatted out of the stored value at the precision the prose uses, then
   looked for in the comment block read off disk. A test that instead retyped
   ``2,350`` would be a fifth copy of the fact and would pin nothing.

The comment block is read from the source by :func:`_const_doc`, which walks the
module's AST and collects the contiguous ``#:`` lines above the assignment. It
returns the empty string where there is no block, so a fragment check on a
constant whose prose has been deleted fails rather than passing vacuously. Both
halves of that are asserted below before anything relies on them.

WHAT IS DELIBERATELY NOT DONE
-----------------------------
``M6B_RHO`` is NOT re-derived end to end. Doing that needs the M5 no-latent
checkpoint and the frozen corpus, neither of which exists in a clone, and the
corpus is frozen at ``b3e5dadad01eaef9`` and may not be regenerated.
:func:`innovation_autocorrelation` is instead exercised against closed-form
answers on synthetic fires, so the estimator is executable, tested and no longer
dead, while the published value stays pinned to the artifact that recorded it.
That function is the only executable description of how the number was obtained
and must not be deleted for want of a caller.
"""

from __future__ import annotations

import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_latent_published_branch import PUBLISHED_LATENT_CONFIG  # noqa: E402
from wildfire_nowcast.common.paths import repo_root  # noqa: E402
from wildfire_nowcast.eval import stage as ST  # noqa: E402
from wildfire_nowcast.model import labelnoise as LN  # noqa: E402
from wildfire_nowcast.model import train as TR  # noqa: E402
from wildfire_nowcast.model.train import (  # noqa: E402
    FireTensors,
    innovation_autocorrelation,
)

# --------------------------------------------------------------------------
# reading the source and the artifacts
# --------------------------------------------------------------------------

#: Module-level constants in ``model/`` and ``eval/`` whose own ``#:`` block
#: cites a ``runs/`` path. ENUMERATED BY A WALK, never listed by hand: a new
#: measured constant that nobody pinned turns
#: ``test_the_pinned_set_is_the_whole_cited_class`` red instead of joining the
#: class silently. Class members are pinned one test each below.
PINNED_CONSTANTS: tuple[tuple[str, str], ...] = (
    ("model/train.py", "M6B_RHO"),
    ("model/train.py", "M7_SPATIAL_LEVER_RMS"),
    ("model/train.py", "M8_ELLIPSE_TRANSFER"),
    ("eval/stage.py", "RATE_MINUS_GROWTH_ELASTICITY"),
)

#: Module DOCSTRINGS in the same two packages that cite a ``runs/`` path.
#: ``eval/stage.py`` quotes fifteen measured numbers and is pinned below.
#: ``eval/labelfloor.py`` cites the ``runs/`` DIRECTORY as a cache location and
#: quotes no number, so there is nothing to re-derive; it is in the tuple so that
#: the walk's answer stays complete rather than filtered.
PINNED_MODULE_DOCSTRINGS: tuple[str, ...] = ("eval/labelfloor.py", "eval/stage.py")

_SRC = Path(__file__).resolve().parents[1] / "src" / "wildfire_nowcast"


def _norm(text: str) -> str:
    """Collapse runs of whitespace, so a claim may wrap across comment lines."""
    return re.sub(r"\s+", " ", text).strip()


def _const_doc(rel: str, name: str) -> str:
    """The contiguous ``#:`` block immediately above module-level ``name``.

    Empty string when the constant has no block of its own, which is a real
    state: ``M8_GATE_EXTRA_TRANSFER_LOSS`` sits directly under
    ``M8_ELLIPSE_TRANSFER`` and shares its prose.
    """
    source = (_SRC / rel).read_text()
    lines = source.splitlines()
    for node in ast.parse(source).body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if name not in names:
            continue
        block: list[str] = []
        i = node.lineno - 2
        while i >= 0 and lines[i].strip().startswith("#:"):
            block.append(lines[i].strip()[2:].strip())
            i -= 1
        return _norm(" ".join(reversed(block)))
    raise AssertionError(f"{rel} has no module-level constant named {name}")


def _module_doc(rel: str) -> str:
    return _norm(ast.get_docstring(ast.parse((_SRC / rel).read_text())) or "")


def _cited_class() -> tuple[list[tuple[str, str]], list[str]]:
    """Walk both packages for ``runs/`` citations. The class, not a memory of it."""
    constants: list[tuple[str, str]] = []
    modules: list[str] = []
    for package in ("model", "eval"):
        for path in sorted((_SRC / package).glob("*.py")):
            rel = f"{package}/{path.name}"
            source = path.read_text()
            lines = source.splitlines()
            tree = ast.parse(source)
            if "runs/" in (ast.get_docstring(tree) or ""):
                modules.append(rel)
            for node in tree.body:
                names: list[str] = []
                if isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                if not names:
                    continue
                block: list[str] = []
                i = node.lineno - 2
                while i >= 0 and lines[i].strip().startswith("#:"):
                    block.append(lines[i].strip()[2:].strip())
                    i -= 1
                if "runs/" in " ".join(block):
                    constants.append((rel, names[0]))
    return constants, modules


def _artifact(rel: str) -> dict[str, Any]:
    """Load a TRACKED ``runs/`` artifact. Every one read here is in ``git ls-files``."""
    path = repo_root() / rel
    assert path.is_file(), f"{rel} is cited by published source and is not on disk"
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def _quotes(doc: str, fragment: str, why: str) -> None:
    assert fragment in doc, (
        f"the published prose no longer carries {fragment!r}, which is what the artifact "
        f"says ({why}). The artifact is the source; fix the prose, or if the ARTIFACT moved, "
        "raise it rather than retyping the comment."
    )


# --------------------------------------------------------------------------
# the readers themselves, before anything trusts them
# --------------------------------------------------------------------------


def test_the_comment_reader_reads_a_block_and_reports_nothing_where_there_is_none() -> None:
    """Positive and negative control for :func:`_const_doc`.

    Without the negative half every fragment assertion in this file could be
    satisfied by a reader that returned the whole source, and without the
    positive half by one that returned nothing and was never checked.
    """
    doc = _const_doc("model/train.py", "M6B_RHO")
    assert doc.startswith("[M6B] The AR(1) persistence used by the candidate.")
    assert "M6_ACTIVITY_PREREGISTRATION" not in doc, "the reader ran past its own block"

    assert _const_doc("model/train.py", "M8_GATE_EXTRA_TRANSFER_LOSS") == ""

    with pytest.raises(AssertionError):
        _const_doc("model/train.py", "NO_SUCH_CONSTANT_EXISTS")


def test_the_pinned_set_is_the_whole_cited_class() -> None:
    """A new constant citing an artifact fails here until someone pins it.

    This is the only assertion in the file that is about the FILE. Everything
    else pins one constant; this one pins the enumeration, so the class cannot
    grow a member that no test re-derives.
    """
    constants, modules = _cited_class()
    assert tuple(constants) == PINNED_CONSTANTS, (
        "the set of constants citing a runs/ artifact has changed. Add the new one to "
        "PINNED_CONSTANTS and give it a test that re-derives its number from the artifact."
    )
    assert tuple(modules) == PINNED_MODULE_DOCSTRINGS


# --------------------------------------------------------------------------
# M6B_RHO: the AR(1) persistence, and the five numbers beside it
# --------------------------------------------------------------------------

M6_PATH = "runs/m6_innovation_autocorr.json"


def test_M6B_RHO_is_the_pooled_autocorrelation_the_artifact_stores() -> None:
    art = _artifact(M6_PATH)
    assert TR.M6B_RHO == round(art["rho_lag1_pooled"], 4)


def test_the_M6B_comment_quotes_that_artifact_and_no_other_number() -> None:
    """Each fragment is FORMATTED OUT OF the artifact at the precision the prose uses."""
    art = _artifact(M6_PATH)
    doc = _const_doc("model/train.py", "M6B_RHO")
    positive = [f for f in art["per_fire"] if f["rho"] is not None and f["rho"] > 0]

    _quotes(doc, f"`{M6_PATH}`", "the artifact must be nameable by a reader")
    _quotes(doc, f"on the {art['n_fires']} TRAIN fires", "n_fires")
    _quotes(doc, f"{art['n_pairs']:,} consecutive hour-pairs", "n_pairs")
    _quotes(doc, f"innovation = **{art['rho_lag1_pooled']:.4f}**", "rho_lag1_pooled")
    _quotes(
        doc,
        f"per-fire {art['per_fire_rho_min']:.4f}-{art['per_fire_rho_max']:.4f}",
        "per_fire_rho_min / per_fire_rho_max",
    )
    _quotes(
        doc, f"positive on {len(positive)} of {art['n_fires']}", "the sign of every per-fire rho"
    )


def test_the_M6B_artifacts_own_totals_agree_with_its_per_fire_rows() -> None:
    """``2,350 hour-pairs`` and ``8 fires`` are only one claim if these agree."""
    art = _artifact(M6_PATH)
    rows = art["per_fire"]
    assert sum(int(r["n_pairs"]) for r in rows) == art["n_pairs"]
    assert len(rows) == art["n_fires"], "a fire with an undefined rho would break the pair"
    assert [r["rho"] for r in rows] == [r for r in (x["rho"] for x in rows) if r is not None]
    assert min(r["rho"] for r in rows) == art["per_fire_rho_min"]
    assert max(r["rho"] for r in rows) == art["per_fire_rho_max"]


def test_the_C3_claim_of_seven_spatial_blocks_is_the_partition_probes_own_split() -> None:
    """The M6B artifact records no split, so the C-3 claim is checked ACROSS artifacts.

    ``runs/m6_innovation_autocorr.json`` stores neither a split fingerprint nor a
    block list, so ``8 fires across 7 spatial blocks`` cannot be re-derived from
    it alone. It IS re-derivable from the fire list: the eight fires it scored are
    exactly the train side recorded by ``runs/m8_partition_probe.json``, which
    stores the blocks those fires occupy.
    """
    m6 = _artifact(M6_PATH)
    split = _artifact("runs/m8_partition_probe.json")["split_fingerprint"]
    scored = sorted(f["fire_id"] for f in m6["per_fire"])

    assert scored == sorted(split["train_fire_ids"])
    doc = _const_doc("model/train.py", "M6B_RHO")
    _quotes(
        doc,
        f"{len(scored)} fires across {len(split['train_blocks'])} spatial blocks",
        "train_fire_ids / train_blocks",
    )


def test_M6B_MATRIX_restates_the_measurement_and_every_arm_uses_the_constant() -> None:
    art = _artifact(M6_PATH)
    doc = _const_doc("model/train.py", "M6B_MATRIX")
    positive = [f for f in art["per_fire"] if f["rho"] is not None and f["rho"] > 0]

    _quotes(
        doc,
        f"({art['rho_lag1_pooled']:.4f} pooled over {art['n_pairs']:,} hour-pairs, "
        f"{art['n_fires']} fires, per-fire {art['per_fire_rho_min']:.3f}-"
        f"{art['per_fire_rho_max']:.3f}, all {len(positive)} positive)",
        "the same five numbers, restated at three places",
    )

    rho_arms = [
        cfg["latent_rho"]
        for matrix in (TR.M6B_MATRIX, TR.M7_MATRIX, TR.M8_MATRIX)
        for _name, cfg, _why in matrix
        if "latent_rho" in cfg
    ]
    assert rho_arms, "no arm sets latent_rho, so the constant would be decorative"
    assert set(rho_arms) == {TR.M6B_RHO}


def test_the_published_latent_config_carries_the_same_rho_as_the_artifact() -> None:
    """The fourth copy. ``tests/test_latent_published_branch.py`` pins rho as a literal.

    That file declines to read a run directory on purpose, because the checkpoints
    it describes are untracked and a test that read one would be vacuous in a
    clone. The artifact here IS tracked, so the literal can be tied to it without
    taking on that failure mode.
    """
    assert PUBLISHED_LATENT_CONFIG["rho"] == TR.M6B_RHO
    assert TR.M6B_RHO == round(_artifact(M6_PATH)["rho_lag1_pooled"], 4)


# --------------------------------------------------------------------------
# the estimator behind that number, run rather than described
# --------------------------------------------------------------------------


class _WeatherIsTheHazard:
    """A stand-in kernel whose one-step hazard IS weather channel 0.

    :func:`innovation_autocorrelation` calls ``step_probability(b0, weather[t+1],
    fields, None)`` and reduces the field to ``sum(p * unburned)``. Returning the
    weather channel verbatim therefore makes the predicted new-cell count an
    input of the test, which is what lets the residual sequence be chosen exactly.
    """

    def step_probability(
        self, burned: Any, weather_step: Any, fields: Any, latent: Any = None
    ) -> Any:
        return weather_step[0]


def _fire_with_residuals(fire_id: str, residuals: list[float], offset: float | None = None) -> Any:
    """A one-cell fire whose innovation sequence is ``residuals`` (up to a constant).

    The observed count is zero at every step, so ``e_t = -log1p(predicted)``.
    Setting ``predicted_t = expm1(offset - r_t)`` gives ``e_t = r_t - offset``,
    and ``offset >= max(r)`` keeps the predicted count non-negative. The estimator
    removes the per-fire mean, so the offset cannot reach the answer, which is
    the property :func:`test_the_estimator_removes_the_per_fire_mean` measures.
    """
    values = [float(x) for x in residuals]
    shift = max(values) if offset is None else float(offset)
    n = len(values)
    weather = torch.zeros(n + 1, 1, 1, 1, dtype=torch.float64)
    for t, value in enumerate(values):
        weather[t + 1, 0] = math.expm1(shift - value)
    return FireTensors(
        fire_id=fire_id,
        fields=None,
        weather=weather,
        burned=torch.zeros(n + 1, 1, 1, dtype=torch.float64),
        band=torch.ones(n + 1, 1, 1, dtype=torch.bool),
        t0_index=np.arange(n),
        shape=(1, 1),
    )


#: Four burning hours then four quiet ones: mean zero, six of seven consecutive
#: pairs agreeing in sign. ``rho`` is ``5 / 7`` exactly, by hand: six products of
#: ``+1`` and one of ``-1`` over a denominator of seven unit squares.
_PERSISTENT = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
#: Perfect alternation: every consecutive pair is exactly opposed, so ``rho`` is
#: ``-1`` exactly and no tolerance is needed to say so.
_ALTERNATING = [(-1.0) ** t for t in range(8)]


def test_the_estimator_reproduces_a_closed_form_lag_one_autocorrelation() -> None:
    """Known answers, not a re-implementation of the estimator beside itself."""
    out = innovation_autocorrelation(
        _WeatherIsTheHazard(), [_fire_with_residuals("f", _PERSISTENT)]
    )
    assert out["rho_lag1_pooled"] == 5 / 7
    assert out["n_pairs"] == len(_PERSISTENT) - 1
    assert out["n_fires"] == 1
    assert out["per_fire"][0]["rho"] == 5 / 7
    assert out["per_fire"][0]["innovation_sd"] == 1.0

    opposed = innovation_autocorrelation(
        _WeatherIsTheHazard(), [_fire_with_residuals("f", _ALTERNATING)]
    )
    assert opposed["rho_lag1_pooled"] == -1.0


def test_the_estimator_removes_the_per_fire_mean() -> None:
    """The docstring's own claim: a per-fire offset is calibration, not persistence.

    Left in, it would inflate rho toward 1 mechanically. Ten added to every
    residual must therefore change nothing, bitwise.
    """
    kernel = _WeatherIsTheHazard()
    tight = innovation_autocorrelation(
        kernel, [_fire_with_residuals("f", _PERSISTENT, offset=max(_PERSISTENT))]
    )
    shifted = innovation_autocorrelation(
        kernel, [_fire_with_residuals("f", _PERSISTENT, offset=max(_PERSISTENT) + 10.0)]
    )
    assert tight["rho_lag1_pooled"] == shifted["rho_lag1_pooled"] == 5 / 7


def test_the_estimator_pools_over_pairs_rather_than_averaging_per_fire_rho() -> None:
    """Two fires of equal length, one at ``5/7`` and one at ``-1``, pool to ``-1/7``.

    The mean of the two per-fire values is ``-1/7`` only because the lengths
    match; the point of the check is that the pooled figure is the ratio of
    summed cross-products, so ``per_fire_rho_min`` and ``max`` can straddle it.
    """
    out = innovation_autocorrelation(
        _WeatherIsTheHazard(),
        [_fire_with_residuals("a", _PERSISTENT), _fire_with_residuals("b", _ALTERNATING)],
    )
    assert out["rho_lag1_pooled"] == -1 / 7
    assert out["n_pairs"] == 2 * (len(_PERSISTENT) - 1)
    assert out["n_fires"] == 2
    assert out["per_fire_rho_min"] == -1.0
    assert out["per_fire_rho_max"] == 5 / 7


def test_a_variance_free_fire_contributes_pairs_but_no_rho_and_the_published_run_has_none() -> None:
    """A degenerate fire makes ``n_pairs`` and ``n_fires`` count different things.

    A constant innovation sequence has zero variance after mean removal, so the
    fire's own rho is ``None`` while its pairs still enter ``n_pairs``. That is
    the estimator's behaviour, recorded here rather than discovered later, and
    the second half states that the published run is NOT in that regime: all
    eight of its fires carry a rho, which is why ``2,350 hour-pairs`` and
    ``8 fires`` describe the same sample.
    """
    out = innovation_autocorrelation(
        _WeatherIsTheHazard(), [_fire_with_residuals("flat", [2.0] * 8)]
    )
    assert out["per_fire"][0]["rho"] is None
    assert out["n_pairs"] == 7
    assert out["n_fires"] == 0
    assert out["rho_lag1_pooled"] == 0.0

    rows = _artifact(M6_PATH)["per_fire"]
    assert all(r["rho"] is not None for r in rows)


# --------------------------------------------------------------------------
# M7_SPATIAL_LEVER_RMS
# --------------------------------------------------------------------------

M7_LEVER_PATH = "runs/m7_spatial_lever.json"


def test_M7_SPATIAL_LEVER_RMS_is_the_stored_pooled_rms() -> None:
    art = _artifact(M7_LEVER_PATH)
    assert TR.M7_SPATIAL_LEVER_RMS == tuple(round(v, 4) for v in art["rms_lever_pooled"])
    assert len(TR.M7_SPATIAL_LEVER_RMS) == len(art["mode_names"])


def test_the_M7_lever_comment_quotes_the_stored_means_scope_and_checkpoint() -> None:
    art = _artifact(M7_LEVER_PATH)
    doc = _const_doc("model/train.py", "M7_SPATIAL_LEVER_RMS")
    split = _artifact("runs/m8_partition_probe.json")["split_fingerprint"]

    _quotes(doc, f"``{M7_LEVER_PATH}``", "the artifact must be nameable by a reader")
    _quotes(
        doc,
        f"{art['n_windows_pooled']:,} windows, {len(art['per_fire'])} fires, "
        f"{len(split['train_blocks'])} spatial blocks",
        "n_windows_pooled / per_fire / the train blocks those fires occupy",
    )
    _quotes(
        doc,
        f"`{Path(art['checkpoint']).name.split('-')[0]}` checkpoint's own one-step hazard",
        "checkpoint",
    )

    east, north, radial = art["mode_names"]
    rms = art["rms_lever_pooled"]
    mean = art["mean_lever_pooled"]
    _quotes(doc, f"``{east}`` **{rms[0]:.4f}** (mean {mean[0]:+.3f})", "mode 0")
    _quotes(doc, f"``{north}`` **{rms[1]:.4f}** (mean {mean[1]:+.3f})", "mode 1")
    _quotes(doc, f"``{radial}`` **{rms[2]:.4f}** (mean **{mean[2]:+.4f}**)", "mode 2")
    _quotes(doc, f"a mode whose mean lever is {mean[2]:.4f} IS the global mode", "mode 2 again")


def test_the_M7_lever_comments_two_qualitative_claims_are_true_of_the_stored_values() -> None:
    """``mean lever ~0 and RMS ~0.30`` for the gradients, ``1.00`` for the radial."""
    art = _artifact(M7_LEVER_PATH)
    mean = art["mean_lever_pooled"]
    rms = art["rms_lever_pooled"]
    assert all(abs(m) < 0.15 for m in mean[:2]), mean
    assert all(abs(r - 0.30) < 0.01 for r in rms[:2]), rms
    assert round(mean[2], 2) == 1.00, mean


# --------------------------------------------------------------------------
# M8: the transfer constants
# --------------------------------------------------------------------------

M8_PARTITION_PATH = "runs/m8_partition_probe.json"
M8_TRANSFER_PATH = "runs/m8_transfer_probe.json"


def _ellipse_reference() -> dict[str, Any]:
    reference: dict[str, Any] = _artifact(M8_PARTITION_PATH)["observed_transfer_reference"]
    return reference


def test_M8_ELLIPSE_TRANSFER_is_the_quotient_the_artifacts_note_defines() -> None:
    """The two operands are parsed OUT OF the artifact, not retyped from the prose.

    ``ellipse_cal3h`` is stored to full float precision and the note beside it
    states the ratio and the held-out score it was computed from. The constant is
    the quotient of those two, so all three are checked against each other.
    """
    reference = _ellipse_reference()
    match = re.search(r"ratio ([0-9.]+), scoring ([0-9.]+) held-out", reference["note"])
    assert match, reference["note"]
    train_ratio, heldout = float(match.group(1)), float(match.group(2))

    assert TR.M8_ELLIPSE_TRANSFER == heldout / train_ratio
    assert abs(TR.M8_ELLIPSE_TRANSFER - reference["ellipse_cal3h"]) < 1e-12

    doc = _const_doc("model/train.py", "M8_ELLIPSE_TRANSFER")
    _quotes(doc, f"`{M8_PARTITION_PATH}`", "the artifact must be nameable by a reader")
    _quotes(doc, f"`{M8_TRANSFER_PATH}`", "the second artifact the block cites")
    _quotes(
        doc,
        f"TRAIN ratio **{train_ratio}** and scores held-out **{heldout}**",
        "the note's own two operands",
    )


def test_the_M8_comment_quotes_the_stored_per_arm_transfers() -> None:
    reference = _ellipse_reference()
    doc = _const_doc("model/train.py", "M8_ELLIPSE_TRANSFER")
    _quotes(
        doc,
        f"`m6_fair` {reference['m6_fair_mean']}, `m7_spatial` {reference['m7_spatial_mean']}, "
        f"gate arms {reference['gate_arms_mean']}",
        "m6_fair_mean / m7_spatial_mean / gate_arms_mean",
    )


def test_M8_GATE_EXTRA_TRANSFER_LOSS_is_the_gate_arms_share_of_the_ellipses() -> None:
    """The numerator is checked against the artifact; the denominator is the prose's.

    Which is why the two tests below exist: the denominator the published
    arithmetic uses is not what the ARTIFACT's stored float rounds to, and the
    reason turns out to be a defect in the artifact rather than in the prose.
    """
    reference = _ellipse_reference()
    doc = _const_doc("model/train.py", "M8_ELLIPSE_TRANSFER")
    numerator, denominator = (
        float(x) for x in re.search(r"EXTRA ([0-9.]+)/([0-9.]+) =", doc).groups()
    )

    assert numerator == reference["gate_arms_mean"]
    assert TR.M8_GATE_EXTRA_TRANSFER_LOSS == numerator / denominator
    _quotes(doc, f"= **{TR.M8_GATE_EXTRA_TRANSFER_LOSS:.3f}**", "the quotient the prose states")


def test_the_M8_artifacts_stored_transfer_is_a_QUOTIENT_OF_TWO_DISPLAY_VALUES() -> None:
    """The artifact and the prose differ at the third place, and the artifact is wrong.

    ``ellipse_cal3h`` looks like a measurement to sixteen digits. It is not: the
    script that wrote this artifact hard-codes ``0.8447 / 1.0086``, the two
    FOUR-PLACE DISPLAY values its own note quotes, so about five of those digits
    are real. No path to that script is given here because it is the one M8-era
    producing script that is not tracked, and a citation a reader cannot open is
    the defect the ``runs/`` citation check exists to catch. That quotient is
    ``0.8374975213166767`` and reads ``0.837`` at three places, while the prose
    says ``transfer 0.838``.

    Reading the artifact naively demands correcting the prose to 0.837. That would
    have introduced an error into public source, and the test below is why: the
    unrounded operands still exist, and the true quotient rounds to 0.838. This
    half is asserted everywhere, including a clone; the vindication needs an
    untracked run directory and is separated for that reason alone.
    """
    reference = _ellipse_reference()
    stored = reference["ellipse_cal3h"]
    match = re.search(r"ratio ([0-9.]+), scoring ([0-9.]+) held-out", reference["note"])
    assert match
    displays = (float(match.group(2)), float(match.group(1)))

    assert stored == displays[0] / displays[1], "the stored float is the DISPLAY quotient"
    assert all(len(x.split(".")[1]) == 4 for x in match.groups()), "four places, not sixteen"
    assert f"{stored:.3f}" == "0.837"
    _quotes(_const_doc("model/train.py", "M8_ELLIPSE_TRANSFER"), "transfer **0.838**", "the prose")


#: Where the unrounded operands live in a baselines record. Named as paths rather
#: than as values so that this file still carries no copy of either number.
_TRAIN_RATIO_KEYS = ("ellipse_calibration", "rule_of_record", "growth_ratio")
_HELDOUT_RATIO_KEYS = ("c6_2_validity", "ellipse_cal3h", "off_state", "growth_stratum_ratio")


def _dig(payload: Any, keys: tuple[str, ...]) -> Any:
    for key in keys:
        if not isinstance(payload, dict) or key not in payload:
            return None
        payload = payload[key]
    return payload


def test_the_unrounded_operands_EXIST_and_they_VINDICATE_the_M8_prose() -> None:
    """ADR-095 asked for option (a) first. The operands were found, and they decide.

    The two numbers the artifact's note quotes at four places are recorded at full
    precision in the baselines runs of the same era, under
    ``ellipse_calibration.rule_of_record.growth_ratio`` and
    ``c6_2_validity.ellipse_cal3h.off_state.growth_stratum_ratio``. Runs are
    selected by whether their operands DISPLAY as the pair the note quotes, so the
    selection is derived from the artifact and this file stores neither number.

    The true quotient is ``0.837527738869328``, which is ``0.838``. The published
    prose was right all along; what is wrong is the artifact, which stores the
    quotient of the roundings instead of the rounding of the quotient. The same
    holds one level down: the gate share is ``0.858``, as published.

    ``runs/baselines-*`` is untracked, so this skips in a clone. That is stated
    rather than hidden, and the half that gates CI is the test above, which needs
    only the tracked artifact.
    """
    reference = _ellipse_reference()
    match = re.search(r"ratio ([0-9.]+), scoring ([0-9.]+) held-out", reference["note"])
    assert match
    want = (float(match.group(1)), float(match.group(2)))

    found: set[tuple[float, float]] = set()
    for path in sorted((repo_root() / "runs").glob("baselines-*/results.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        train = _dig(payload, _TRAIN_RATIO_KEYS)
        heldout = _dig(payload, _HELDOUT_RATIO_KEYS)
        if train is None or heldout is None:
            continue
        if (round(train, 4), round(heldout, 4)) == want:
            found.add((float(train), float(heldout)))

    if not found:
        pytest.skip("no baselines record carrying the operands: this is a clone")
    assert len(found) == 1, f"the runs disagree about the operands: {found}"
    train, heldout = found.pop()

    assert (round(train, 4), round(heldout, 4)) == want, "the displays are the note's"
    assert train != round(train, 4) and heldout != round(heldout, 4), "full precision"

    truth = heldout / train
    doc = _const_doc("model/train.py", "M8_ELLIPSE_TRANSFER")
    _quotes(doc, f"transfer **{truth:.3f}**", "the TRUE quotient, not the artifact's")
    _quotes(doc, f"= **{reference['gate_arms_mean'] / truth:.3f}**", "the TRUE gate share")

    stored = reference["ellipse_cal3h"]
    assert f"{stored:.3f}" != f"{truth:.3f}", "the artifact and the truth differ at 3 places"
    assert abs(truth - stored) < 1e-4, "and only there: the displays are still the right pair"


def test_M8_CONDITIONAL_PRIOR_TRANSFER_is_the_stored_multiplier_mean() -> None:
    art = _artifact(M8_TRANSFER_PATH)
    doc = _const_doc("model/train.py", "M8_CONDITIONAL_PRIOR_TRANSFER")

    assert TR.M8_CONDITIONAL_PRIOR_TRANSFER == round(art["multiplier_mean"], 4)
    _quotes(
        doc,
        f"over all {len(art['per_arm'])} M7 gate seeds: {art['multiplier_mean']:.3f}",
        "multiplier_mean over per_arm",
    )
    _quotes(
        doc,
        f"~ {art['target_if_hypothesis_true']['observed_extra_gap']:.3f}` if true",
        "target_if_hypothesis_true.observed_extra_gap",
    )
    assert art["multiplier_mean"] > 1.0, "the block's whole point is the WRONG SIGN"


def test_the_off_state_ceiling_quoted_at_two_sites_is_the_stored_optimum() -> None:
    """``0.143`` appears in two comment blocks and is one measurement.

    Neither block names the artifact, so the linkage is stated here rather than
    assumed: ``optimal_dormant_off_rate`` is the best any policy of the declared
    form reaches, which is what both blocks call the ceiling from C1's covariates.
    """
    art = _artifact("runs/m7_offstate_optimum.json")
    ceiling = art["optimal_dormant_off_rate"]
    assert art["oracle_dormant_off_rate"] > ceiling, "an upper bound below the oracle"

    _quotes(
        _const_doc("model/train.py", "M8_OFF_STATE_IS_NOT_A_TARGET"),
        f"the ceiling at {ceiling:.3f}",
        "optimal_dormant_off_rate",
    )
    _quotes(
        _const_doc("model/train.py", "M8_MATRIX"),
        f"the ceiling from C1's covariates is {ceiling:.3f}",
        "optimal_dormant_off_rate, restated",
    )


# --------------------------------------------------------------------------
# eval/stage.py
# --------------------------------------------------------------------------

M9_PATH = "runs/m9_scaling.json"
M12_PATH = "runs/m12_frontier.json"


def test_RATE_MINUS_GROWTH_ELASTICITY_converts_the_stored_growth_elasticities() -> None:
    """Both trios in the block are derived: the growth one from the artifact, the
    rate one from the artifact THROUGH the constant.

    So a constant changed to anything but ``-1.0`` makes the quoted rate
    elasticities wrong, which is the whole reason the constant exists.
    """
    stored = _artifact(M9_PATH)["elasticities_from_bin_means"]
    growth = [stored[k]["elasticity"] for k in ("truth_growth", "model_growth", "ellipse_growth")]
    rate = [g + ST.RATE_MINUS_GROWTH_ELASTICITY for g in growth]
    doc = _const_doc("eval/stage.py", "RATE_MINUS_GROWTH_ELASTICITY")

    _quotes(doc, f"``{M9_PATH}``", "the artifact must be nameable by a reader")
    _quotes(doc, "RATE elasticities (" + " / ".join(f"{v:+.4f}" for v in rate) + ")", "converted")
    _quotes(doc, "GROWTH elasticities (" + " / ".join(f"{v:+.4f}" for v in growth) + ")", "stored")
    assert all(s["n_bins"] == 8 for s in stored.values())


def test_the_stage_module_docstrings_scope_is_the_m12_artifacts_scope() -> None:
    art = _artifact(M12_PATH)
    doc = _module_doc("eval/stage.py")
    blocks = sorted(art["licensed"], key=int)

    _quotes(doc, f"``{M12_PATH}``", "the artifact must be nameable by a reader")
    _quotes(doc, f"blocks {'/'.join(blocks)}, {art['n_heldout_windows']} windows", "fold scope")
    _quotes(doc, f"fingerprint ``{art['split_fingerprint']}``", "split_fingerprint")
    _quotes(
        doc,
        f"maximum absolute error of exactly {art['control_max_abs_error']}",
        "control_max_abs_error",
    )
    assert sum(v["n_windows"] for v in art["licensed"].values()) == art["n_heldout_windows"]
    assert art["control_can_fail"] is True


def test_the_stage_module_docstrings_per_block_numbers_are_the_m12_values() -> None:
    """Fire labels are resolved through the artifact's own ``fire_id``, not a map.

    So the prose cannot silently attach the right number to the wrong fire: the
    label has to appear inside the block's recorded id for the row to be found at
    all, and every block must be hit exactly once.
    """
    art = _artifact(M12_PATH)["licensed"]
    doc = _module_doc("eval/stage.py")

    def block_for(label: str) -> dict[str, Any]:
        hits = [v for v in art.values() if label.lower() in v["fire_id"]]
        assert len(hits) == 1, f"{label} matches {len(hits)} of the artifact's fires"
        return hits[0]

    frontier_span = doc.split("Frontier length RISES")[1].split("while growth falls")[0]
    frontier = re.findall(r"([+-]\d+\.\d{3}) ([A-Za-z]+)", frontier_span)
    assert len(frontier) == len(art), frontier
    for value, label in frontier:
        assert float(value) == round(block_for(label)["frontier_cells"]["stage_decay"], 3), label

    rate_span = doc.split("sits BELOW it on 5 of 5**:")[1].split("It is computed")[0]
    rates = re.findall(r"([+-]\d+\.\d{3}) vs ([+-]\d+\.\d{3}) ([A-Za-z]+)", rate_span)
    assert len(rates) == len(art), rates
    for rate, growth, label in rates:
        row = block_for(label)
        assert float(rate) == round(row["growth_per_frontier_cell"]["stage_decay"], 3), label
        assert float(growth) == round(row["growth"]["stage_decay"], 3), label

    assert {label for _v, label in frontier} == {label for *_r, label in rates}


def test_the_stage_module_docstrings_counting_claims_are_true_of_the_m12_values() -> None:
    """``5 of 5`` three times and ``4 of 5`` once, each re-counted from the rows."""
    art = _artifact(M12_PATH)["licensed"]
    doc = _module_doc("eval/stage.py")
    rows = list(art.values())
    n = len(rows)

    frontier_rises = sum(1 for r in rows if r["frontier_cells"]["stage_decay"] > 0)
    growth_falls = sum(1 for r in rows if r["growth"]["stage_decay"] < 0)
    sign_agrees = sum(
        1
        for r in rows
        if (r["growth_per_frontier_cell"]["stage_decay"] > 0) == (r["growth"]["stage_decay"] > 0)
    )
    rate_below = sum(
        1 for r in rows if r["growth_per_frontier_cell"]["stage_decay"] < r["growth"]["stage_decay"]
    )

    _quotes(doc, f"RISES on {frontier_rises} of {n} held-out blocks", "frontier_cells sign")
    _quotes(doc, f"while growth falls on {growth_falls} of {n}", "growth sign")
    _quotes(doc, f"agrees in SIGN with growth on {sign_agrees} of {n}", "sign agreement")
    _quotes(doc, f"sits BELOW it on {rate_below} of {n}", "rate below growth")


# --------------------------------------------------------------------------
# model/labelnoise.py: measured, and its citation is not public
# --------------------------------------------------------------------------

_LABEL_NOISE_INDEX = Path("data") / "interim" / "_index" / "label_noise_east_west.json"


def test_the_labelnoise_dataset_means_are_the_index_they_fall_back_to() -> None:
    """Both constants are dataset means and the block cites a file no reader can open.

    ``insights/data item 4 addendum`` is a coordination file and is not in the
    repository. The numbers ARE re-derivable, from the per-fire index the same
    function reads at run time, so they are checked against that here. The index
    is untracked, which is why this skips in a clone and is stated rather than
    hidden: the citation gap is reported to the maintainer, not repaired by me.
    """
    path = repo_root() / _LABEL_NOISE_INDEX
    if not path.is_file():
        pytest.skip("the per-fire label-noise index is untracked and absent")
    means = json.loads(path.read_text())["dataset_mean"]

    assert LN.DATASET_CENTROID_OFFSET_KM == round(means["centroid_offset_km_mean"], 2)
    assert LN.DATASET_RADIUS_MISMATCH_KM == round(means["equiv_radius_mismatch_km_mean"], 2)
