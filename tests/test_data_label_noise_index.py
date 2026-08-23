"""The label-noise index is TRACKED, and both published constants re-derive from it.

WHAT THIS CLOSES
----------------
``model/labelnoise.py`` publishes ``DATASET_CENTROID_OFFSET_KM = 1.64`` and
``DATASET_RADIUS_MISMATCH_KM = 0.91``. Those are the fallback magnitudes used
whenever a fire has no row in the per-fire index, so they size the observation
noise for exactly the fires nobody measured. Their comment block cited a
coordination file, which is excluded from this repository, so the citation
reached no reader outside this machine. The numbers were always correct; the
provenance stopped at the repository boundary, which is where a reader stands.

The fix is on the data side: the per-fire index the module already opens at run
time is now tracked (17,238 bytes), and this file re-derives both constants from
it. ``tests/test_published_constants.py`` also pins them, but skips when the
index is absent; that skip is now unreachable and this file is what makes the
absence a failure rather than a silence.

TWO RULES, BOTH LEARNED FROM NEAR MISSES
----------------------------------------
1. **Numerically, never as text.** The stored value is ``1.6393911428571428``.
   A search for the literal ``1.64`` does not fail quietly, which would be
   survivable; it matches ``"centroid_offset_km_mean": 1.641488``, which is
   ``2020_w_5_cold_springs``'s own value and not a dataset mean at all. A wrong
   hit that reads like a right one nearly produced a false accusation of
   fabricated provenance.
2. **By path, not by name.** The file this checks is the one
   :func:`wildfire_nowcast.model.labelnoise._index_path` returns, compared with
   :meth:`pathlib.Path.samefile`, so a second file of the same name somewhere
   else on disk cannot stand in for the one that is actually read.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from wildfire_nowcast.common.paths import repo_root
from wildfire_nowcast.model import labelnoise as LN

#: Repo-relative location of the artifact. Spelled out rather than taken from
#: the module, so that :func:`test_the_module_reads_the_tracked_file` compares
#: two independently obtained paths rather than one path with itself.
INDEX_RELPATH = Path("data") / "interim" / "_index" / "label_noise_east_west.json"

#: Both constants are the stored means rounded to 2 dp, so the derivation error
#: is bounded by half of the last place. Stated as a number rather than left to
#: ``round`` alone, because ``round`` would also agree with a constant that had
#: been re-typed from a different quantity that happens to round the same way.
ROUNDING_TOLERANCE_KM = 5e-3


def _tracked(relpath: Path) -> bool:
    """Whether git has this path in the index. Decided by git, not by exists()."""
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relpath)],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def _payload() -> dict[str, Any]:
    return dict(json.loads((repo_root() / INDEX_RELPATH).read_text()))


# --------------------------------------------------------------------------
# reachability
# --------------------------------------------------------------------------


def test_the_index_is_tracked_so_the_citation_survives_a_clone() -> None:
    """The whole point: a reader with a clone can open the cited file.

    Controlled. ``assignments.json`` sits in the same directory, is deliberately
    NOT tracked, and must come back False - otherwise this check is a query that
    says yes to everything and proves nothing about the file it names.
    """
    assert _tracked(INDEX_RELPATH), (
        f"{INDEX_RELPATH} is not tracked, so the two published dataset means "
        "cite a file nobody outside this machine can open"
    )
    sibling = INDEX_RELPATH.with_name("assignments.json")
    assert (repo_root() / sibling).is_file(), "the control file is missing; rewrite the control"
    assert not _tracked(sibling), (
        f"{sibling} reads as tracked, so this query cannot distinguish tracked "
        "from untracked and its positive answer above means nothing"
    )


def test_the_module_reads_the_tracked_file_and_not_a_namesake() -> None:
    """By path and inode, not by filename similarity."""
    runtime = LN._index_path()
    tracked = repo_root() / INDEX_RELPATH
    assert runtime.is_file(), f"{runtime} does not exist"
    assert runtime.resolve() == tracked.resolve(), (runtime, tracked)
    assert runtime.samefile(tracked), "same name, different file"


# --------------------------------------------------------------------------
# re-derivation
# --------------------------------------------------------------------------


def test_both_published_dataset_means_re_derive_from_the_tracked_index() -> None:
    """``1.64`` and ``0.91`` are the stored means, to 2 dp.

    No skip. The file is tracked, so absent means broken, and a skip here would
    restore exactly the silence this task was opened to remove.
    """
    means = dict(_payload()["dataset_mean"])

    stored_offset = float(means["centroid_offset_km_mean"])
    stored_mismatch = float(means["equiv_radius_mismatch_km_mean"])

    assert abs(LN.DATASET_CENTROID_OFFSET_KM - stored_offset) < ROUNDING_TOLERANCE_KM, (
        LN.DATASET_CENTROID_OFFSET_KM,
        stored_offset,
    )
    assert abs(LN.DATASET_RADIUS_MISMATCH_KM - stored_mismatch) < ROUNDING_TOLERANCE_KM, (
        LN.DATASET_RADIUS_MISMATCH_KM,
        stored_mismatch,
    )
    assert LN.DATASET_CENTROID_OFFSET_KM == round(stored_offset, 2)
    assert LN.DATASET_RADIUS_MISMATCH_KM == round(stored_mismatch, 2)

    # The tolerance must not be so loose that the two constants could be
    # swapped, or that either could be some other field of the same record.
    assert abs(LN.DATASET_CENTROID_OFFSET_KM - stored_mismatch) > ROUNDING_TOLERANCE_KM
    assert abs(LN.DATASET_RADIUS_MISMATCH_KM - stored_offset) > ROUNDING_TOLERANCE_KM
    assert abs(LN.DATASET_CENTROID_OFFSET_KM - float(means["iou_mean"])) > ROUNDING_TOLERANCE_KM


def test_the_dataset_means_are_the_mean_of_the_28_per_fire_rows() -> None:
    """Second, independent derivation: the summary must add up to its own rows.

    ``dataset_mean`` could have been typed in. The 28 per-fire records could not
    all have been. Their arithmetic mean reproduces both summary fields exactly
    in IEEE double, which is what makes the summary evidence rather than a note.
    """
    payload = _payload()
    per_fire = dict(payload["per_fire"])
    means = dict(payload["dataset_mean"])
    assert len(per_fire) == 28, len(per_fire)

    for key in ("centroid_offset_km_mean", "equiv_radius_mismatch_km_mean"):
        rows = [float(dict(v)[key]) for v in per_fire.values()]
        assert len(rows) == 28
        assert abs(statistics.fmean(rows) - float(means[key])) < 1e-12, key


def test_the_constants_are_what_an_unmeasured_fire_actually_gets() -> None:
    """Exercise the fallback the constants exist for, not just their values.

    ``noise_model_for`` reads the per-fire row when there is one. A fire with no
    row must degrade to the dataset's noise, never to zero noise, and this is the
    only path on which the two constants are load-bearing.
    """
    model = LN.noise_model_for("2099_a_fire_that_does_not_exist")
    assert "fallback" in model.source
    assert model.centroid_offset_km == LN.DATASET_CENTROID_OFFSET_KM
    assert model.radius_mismatch_km == LN.DATASET_RADIUS_MISMATCH_KM
    assert model.offset_sigma_cells > 0.0, "a missing calibration degraded to NO noise"
    assert model.p_morph > 0.0

    # Control: a fire that IS in the index must not take the fallback, or the
    # assertions above would hold no matter what the index contained.
    measured = LN.noise_model_for("2020_czu_lightning_complex")
    assert "fallback" not in measured.source
    assert measured.centroid_offset_km != LN.DATASET_CENTROID_OFFSET_KM
