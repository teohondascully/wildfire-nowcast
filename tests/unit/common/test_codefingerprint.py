"""``common/codefingerprint.py`` - C-4.2, the hash that says which code produced a number.

Every published result in this repository carries one of these, so the two
failure modes that matter are a fingerprint over nothing (which hashes the empty
dict and looks exactly like agreement) and a fingerprint whose arguments were not
the ones the caller meant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wildfire_nowcast.common.codefingerprint import (
    EmptyFingerprintScanError,
    code_fingerprint,
    discover_modules,
    fingerprint_modules,
    package_root,
)


def test_a_scan_that_finds_nothing_is_a_failure_and_not_a_clean_tree() -> None:
    """The most repeated false negative in this project, made loud in one place."""
    with pytest.raises(EmptyFingerprintScanError):
        discover_modules(["no_such_subtree"])
    with pytest.raises(EmptyFingerprintScanError):
        code_fingerprint(["no_such_subtree"], status="probe")


def test_the_fingerprint_moves_when_a_module_changes_and_not_otherwise(tmp_path: Path) -> None:
    """A hash nobody has seen change is a hash nobody knows is wired up."""
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    (package / "sub" / "a.py").write_text("x = 1\n")
    (package / "sub" / "b.py").write_text("y = 2\n")

    before = code_fingerprint(["sub"], status="ok", root=package)
    assert before["modules"] == ["sub/a.py", "sub/b.py"]
    assert (
        code_fingerprint(["sub"], status="ok", root=package)["fingerprint"]
        == (before["fingerprint"])
    ), "the same tree hashed to two different values"

    (package / "sub" / "b.py").write_text("y = 3\n")
    after = code_fingerprint(["sub"], status="ok", root=package)
    assert after["fingerprint"] != before["fingerprint"]
    assert after["per_file"]["sub/a.py"] == before["per_file"]["sub/a.py"]
    assert after["per_file"]["sub/b.py"] != before["per_file"]["sub/b.py"]


def test_the_scan_root_and_the_status_cannot_be_passed_by_position() -> None:
    """A stray positional argument must be refused, not bound to something else.

    ``code_fingerprint(subtrees, "ok")`` reads like it names the status, and under
    a positional-capable signature it names the ROOT instead, so the payload would
    be a fingerprint of a different tree carrying a status of ``None``. That is a
    provenance stamp that is wrong and looks right, which is the failure this
    module exists to prevent.
    """
    with pytest.raises(TypeError):
        discover_modules(["common"], package_root())  # type: ignore[misc]
    with pytest.raises(TypeError):
        code_fingerprint(["common"], "ok")  # type: ignore[misc]
    with pytest.raises(TypeError):
        fingerprint_modules(["common/paths.py"], package_root())  # type: ignore[misc]

    # The control: the same arguments passed by keyword do work, so the three
    # assertions above are reporting the calling convention and not a bad call.
    assert discover_modules(["common"], root=package_root())
    assert code_fingerprint(["common"], status="ok")["status"] == "ok"
