"""C-4.2's coverage, made impossible to forget (ADR-057).

The defect this suite exists for is NOT "a check that cannot fail". It is a
check that **cannot discriminate**: ``_SCORING_CODE_MODULES`` was a hand-written
tuple of ten files, ``model/noiseoracle.py`` and ``model/direct.py`` were never
added to it, and a run therefore stamped "ONE version of the scoring code" while
an uncovered module was edited after the run finished. Every test was green
throughout, because the list was internally consistent — it just did not describe
the tree.

So the assertions here are about COVERAGE, not about behaviour on a fixed input:

* the fingerprinted set equals an INDEPENDENT walk of the tree (both directions:
  a module missing from the set is red, a set entry with no file is red);
* the two omitted modules are named explicitly, so that exact regression cannot
  come back quietly;
* a module inside a PACKAGE is covered, which is what makes splitting
  ``contract.py`` safe (ADR-047 (6)(7));
* planting a module changes the fingerprint and the during-run guard sees it —
  the ADR-057 scenario itself, reproduced hermetically;
* a scan that matches nothing RAISES rather than reporting a clean tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wildfire_nowcast.common import codefingerprint as F
from wildfire_nowcast.eval.reporting import (
    check_common_code_unchanged,
    common_code_fingerprint,
    scoring_code_fingerprint,
)

#: The two modules whose absence from the hand-written tuple produced ADR-057.
#: Named, because "derive it from the filesystem" is the mechanism and these are
#: the instances; a mechanism with no instance pinned is a mechanism nobody
#: notices losing.
ADR_057_OMISSIONS = ("model/noiseoracle.py", "model/direct.py")


def _walk(package_root: Path, subtree: str) -> set[str]:
    """An INDEPENDENT enumeration of a subtree, for the derived set to match."""
    return {
        p.relative_to(package_root).as_posix()
        for p in (package_root / subtree).rglob("*.py")
        if "__pycache__" not in p.parts
    }


# --------------------------------------------------------------------------
# 1. the real tree is covered, in both directions
# --------------------------------------------------------------------------


def test_the_scoring_set_is_the_tree_and_not_a_list() -> None:
    """Every ``*.py`` under ``eval/`` + ``model/`` is fingerprinted, and nothing else.

    Fails BOTH ways on purpose. A new module that the set does not cover is red
    (the ADR-057 defect). A set entry with no file behind it is red (the stale
    half, which an enumerated list can carry for months).
    """
    root = F.package_root()
    expected = _walk(root, "eval") | _walk(root, "model")

    # POSITIVE CONTROL. A walk that matched nothing would make the equality below
    # trivially true against an empty derived set, which is precisely how four of
    # this project's false negatives were produced.
    assert len(expected) >= 25, f"the independent walk found only {len(expected)} modules"

    payload = scoring_code_fingerprint()
    assert set(payload["modules"]) == expected
    assert set(payload["per_file"]) == expected


def test_the_two_modules_adr_057_missed_are_covered_by_name() -> None:
    payload = scoring_code_fingerprint()
    absent = [m for m in ADR_057_OMISSIONS if m not in payload["per_file"]]
    assert not absent, (
        f"{absent} are not fingerprinted. These are the exact modules whose omission let a run "
        "claim ONE version of the scoring code while one of them was edited afterwards."
    )


def test_the_common_set_is_the_tree_and_reaches_inside_packages() -> None:
    """``common/null_check/`` is a PACKAGE, and the old list could not see it.

    This is the property that unblocks splitting ``contract.py``: a module that
    becomes a directory stays covered, because the walk is recursive.
    """
    root = F.package_root()
    expected = _walk(root, "common")
    assert len(expected) >= 20, f"the independent walk found only {len(expected)} modules"

    payload = common_code_fingerprint()
    assert set(payload["modules"]) == expected
    assert "common/null_check/cli.py" in payload["per_file"], "package contents are not covered"
    assert "common/codefingerprint.py" in payload["per_file"], "the guard does not cover itself"


def test_every_digest_is_a_real_digest() -> None:
    """No sentinel, no empty string, no ``None`` masquerading as a hash."""
    for payload in (scoring_code_fingerprint(), common_code_fingerprint()):
        for name, digest in payload["per_file"].items():
            assert isinstance(digest, str) and len(digest) == F.DIGEST_CHARS, (name, digest)
            assert all(c in "0123456789abcdef" for c in digest), (name, digest)


# --------------------------------------------------------------------------
# 2. the ADR-057 scenario itself, hermetically
# --------------------------------------------------------------------------


def _fake_package(root: Path) -> Path:
    """A miniature package tree with the same subtree names as the real one."""
    for subtree in ("common", "eval", "model"):
        (root / subtree).mkdir(parents=True)
        (root / subtree / "__init__.py").write_text("")
    (root / "eval" / "metrics.py").write_text("SCORE = 1\n")
    (root / "model" / "kernel.py").write_text("KERNEL = 1\n")
    (root / "common" / "contract.py").write_text("C = 1\n")
    return root


def test_a_module_planted_after_the_fact_is_covered_with_no_list_edit(tmp_path: Path) -> None:
    """THE regression. A new module must enter the set by existing, not by being remembered."""
    root = _fake_package(tmp_path / "pkg")
    before = F.code_fingerprint(F.SCORING_SUBTREES, status="test", root=root)
    assert "eval/elasticity.py" not in before["per_file"]

    # exactly the shape of the next experiment: a NEW scoring module lands
    (root / "eval" / "elasticity.py").write_text("ELASTICITY = 1\n")
    after = F.code_fingerprint(F.SCORING_SUBTREES, status="test", root=root)

    assert "eval/elasticity.py" in after["per_file"], "a new scoring module was not picked up"
    assert after["fingerprint"] != before["fingerprint"], (
        "the combined fingerprint did not move when a scoring module appeared, so a run "
        "spanning that edit would still report ONE version of the scoring code"
    )


def test_the_during_run_guard_sees_an_edit_to_a_module_nobody_listed(tmp_path: Path) -> None:
    """C-4.2 end to end: sample, edit an UNLISTED module, sample again, get named.

    Under the enumerated tuple this comparison returned ``[]`` for any module
    outside the ten — the run looked clean because the instrument was not
    pointed at the thing that moved.
    """
    root = _fake_package(tmp_path / "pkg")
    (root / "model" / "noiseoracle.py").write_text("SEED = 1\n")
    before = F.code_fingerprint(F.SCORING_SUBTREES, status="test", root=root)

    (root / "model" / "noiseoracle.py").write_text("SEED = 2\n")
    after = F.code_fingerprint(F.SCORING_SUBTREES, status="test", root=root)

    changed = [n for n, d in after["per_file"].items() if d != before["per_file"].get(n)]
    assert changed == ["model/noiseoracle.py"], changed


def test_check_common_code_unchanged_reports_a_real_edit() -> None:
    """The wrapper the run payloads actually call, on the real tree."""
    now = common_code_fingerprint()
    assert check_common_code_unchanged(now)["changed_during_run"] == []

    tampered = {**now, "per_file": {**now["per_file"], "common/contract.py": "0" * 16}}
    report = check_common_code_unchanged(tampered)
    assert report["changed_during_run"] == ["common/contract.py"]
    assert report["warning"] and "MOVED DURING THIS RUN" in report["warning"]


# --------------------------------------------------------------------------
# 3. the scan cannot report a clean tree it never looked at
# --------------------------------------------------------------------------


def test_an_empty_scan_raises_instead_of_fingerprinting_nothing(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    (empty / "eval").mkdir(parents=True)
    with pytest.raises(F.EmptyFingerprintScanError, match="found no Python modules"):
        F.discover_modules(("eval",), root=empty)


def test_a_missing_subtree_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    root = _fake_package(tmp_path / "pkg")
    with pytest.raises(F.EmptyFingerprintScanError, match="not a directory"):
        F.discover_modules(("eval", "sim"), root=root)


# --------------------------------------------------------------------------
# 4. a missing target is LOUD, never a sentinel (ADR-047 (7))
# --------------------------------------------------------------------------


def test_a_missing_target_raises_instead_of_recording_a_sentinel(tmp_path: Path) -> None:
    """``"MISSING"`` used to be a legal value in a payload that reported success.

    That is the failure this repo keeps meeting under different names: an
    unevaluable answer formatted like an answer. A raise is the only outcome a
    reader cannot skim past, and it is what makes packaging a fingerprinted
    module safe (ADR-047 (6)).
    """
    root = _fake_package(tmp_path / "pkg")
    with pytest.raises(F.FingerprintTargetMissingError, match="is not a file"):
        F.fingerprint_modules(("eval/metrics.py", "eval/gone.py"), root=root)


def test_a_module_deleted_between_the_walk_and_the_read_raises(tmp_path: Path) -> None:
    """The only way the real, DISCOVERED set can hit this: code moving mid-run."""
    root = _fake_package(tmp_path / "pkg")
    modules = F.discover_modules(F.SCORING_SUBTREES, root=root)
    assert "eval/metrics.py" in modules
    (root / "eval" / "metrics.py").unlink()
    with pytest.raises(F.FingerprintTargetMissingError, match="eval/metrics.py"):
        F.fingerprint_modules(modules, root=root)


def test_no_real_payload_can_carry_the_sentinel() -> None:
    """Belt and braces on the live instrument: the token must not appear at all."""
    for payload in (scoring_code_fingerprint(), common_code_fingerprint()):
        assert "MISSING" not in json.dumps(payload)


def test_the_payload_is_json_serialisable_and_ordered(tmp_path: Path) -> None:
    """It is stamped into artifacts, so it must survive a round trip unchanged."""
    root = _fake_package(tmp_path / "pkg")
    payload = F.code_fingerprint(F.SCORING_SUBTREES, status="test", root=root)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["modules"] == sorted(payload["modules"])
    assert payload["subtrees"] == list(F.SCORING_SUBTREES)
