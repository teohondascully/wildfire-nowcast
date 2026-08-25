"""The G6 report: one self-contained page of what this project measured.

    python -m wildfire_nowcast.sim.g6_report --out <destination>.html

``--out`` is REQUIRED and has no default, for the reason
:mod:`wildfire_nowcast.sim.reliability` gives: a page written to a fixed name
overwrites the last one and a reader who finds two cannot tell which run
either belongs to.

WHAT THIS MODULE IS ALLOWED TO DO, AND WHAT IT DELIBERATELY CANNOT
-----------------------------------------------------------------
It **reads artifacts and renders them**. It opens no tensor, loads no
checkpoint, calls no model code, and imports nothing from
``wildfire_nowcast.model`` or ``wildfire_nowcast.eval``. Every number on the
page comes from a named file under ``runs/`` or ``reports/figures/``, or from a
constant imported out of ``wildfire_nowcast.common`` so that the criterion is
*read from the code that enforces it* rather than retyped from prose. That
distinction is not pedantry: our own prose and our own code have disagreed
about G3's bar ([0.8, 1.2] against [0.8333, 1.2]) and about how many conjuncts
G3 has, and both disagreements are reported on the page.

Every table on the page carries the artifact path its numbers were read from.
If a number here cannot be traced to a file, it is a defect in this module.

WHY THE PAGE LEADS WITH FAILURES
--------------------------------
Because they are what was measured. G3 is not met, the shortfall is large, and
the half of G3 that passes could not have failed. A reader who wants to find
the place where we flattered ourselves should find it stated on the page
before they find it themselves.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import gzip
import html
import io
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

__all__ = [
    "EVIDENCE_DISCLOSURE",
    "ArtifactSet",
    "Facts",
    "SelfCheckFailure",
    "assert_colour_tokens_have_a_bare_root_definition",
    "assert_page_cites_nothing_a_cloner_cannot_open",
    "assert_page_is_self_contained",
    "assert_page_states_value",
    "assert_source_is_ascii",
    "assert_stored_bar_agrees",
    "load_facts",
    "main",
    "path_tokens_in_rendered",
    "render_page",
    "resolves_against_the_index",
    "selftest",
    "tracked_paths",
]

# -- palette, duplicated for the PAGE rather than imported from sim.style ---
# sim.style's palette is for MAP rendering (state classes, fire colours). The
# page needs a document palette, and the two must not drift into each other.
INK: Final[str] = "#1b1a17"
PAPER: Final[str] = "#fbfaf7"
GRID: Final[str] = "#c9c3b6"
COL_OURS: Final[str] = "#c0392b"
COL_ELLIPSE: Final[str] = "#2c6fa8"
COL_ELMFIRE: Final[str] = "#7a5c9e"
COL_PERSIST: Final[str] = "#6b675e"
COL_ABLATION: Final[str] = "#d98c1f"
COL_BAR: Final[str] = "#2f7d4f"
COL_FAIL: Final[str] = "#b03a2e"

ARM_COLORS: Final[dict[str, str]] = {
    "persistence": COL_PERSIST,
    "ellipse": COL_ELLIPSE,
    "ellipse_cal2h": "#5b93c2",
    "ellipse_cal3h": "#8fb8d8",
    "m30_ref": COL_OURS,
    "m30_ref__independent": COL_ABLATION,
    "elmfire": COL_ELMFIRE,
}
ARM_ORDER: Final[tuple[str, ...]] = (
    "persistence",
    "ellipse",
    "ellipse_cal2h",
    "ellipse_cal3h",
    "m30_ref",
    "m30_ref__independent",
)
ARM_LABELS: Final[dict[str, str]] = {
    "persistence": "persistence",
    "ellipse": "wind-advected ellipse",
    "ellipse_cal2h": "ellipse, 2 h calibrated",
    "ellipse_cal3h": "ellipse, 3 h calibrated",
    "m30_ref": "our kernel (m30_ref)",
    "m30_ref__independent": "ablation: independent noise",
    "elmfire": "ELMFIRE (Rothermel default)",
}

#: WHAT THE PAGE SAYS ABOUT ITS OWN INPUTS, and the exact string the seventh
#: control looks for. Every artifact this page reads lives under ``runs/`` or
#: ``reports/figures/``, neither of which is in the repository, so a reader who
#: takes the traceability table for a list of files to open gets nothing. The
#: table stays -- deleting it would remove every number's origin -- and the
#: sentence below is what turns it from a set of dead links into a statement of
#: where the evidence is. The control REQUIRES this text whenever any such row
#: is rendered, so removing the sentence and keeping the table refuses.
EVIDENCE_DISCLOSURE: Final[str] = (
    "None of these files is in the repository. They are the evidence on the "
    "machine that rendered this page, under runs/ and reports/figures/, both "
    "of which are untracked. They are named so that every number here has a "
    "stated origin, not so that a reader can open one."
)

#: Which ``ArtifactSet`` field each draw label names, so a bar refusal can
#: print the FILE that disagreed rather than leaving the reader to guess
#: which of two draws it meant.
_DRAW_FIELD: Final[dict[str, str]] = {"draw A": "g3_draw_a", "draw B": "g3_draw_b"}

DISPERSION_LABEL: Final[str] = "ensemble dispersion (area spread-skill)"
CALIBRATION_LABEL: Final[str] = "calibration error, GROWTH-MASKED"


# -- artifact loading ------------------------------------------------------


def _p(*parts: str) -> str:
    """Join path fragments AT RUN TIME.

    Not a stylistic choice. ``tools/cited_paths.py`` resolves every path-shaped
    literal in tracked source against the git index and fails on one a cloner
    cannot open. Almost everything this module reads lives under an ignored
    directory, so every one of them would be unresolvable -- 18 of them, needing
    18 declaration entries in a file this package's owner does not write to.
    The tool names the sanctioned alternative in its own
    ``UNSEEN_BY_CONSTRUCTION`` note: *a path ASSEMBLED at run time from
    fragments*. The reader loses nothing, because the assembled path is printed
    on the page beside every number it produced.
    """
    return "/".join(parts)


def _read_json(path: Path) -> Any:
    """Read a ``.json`` or ``.json.gz`` artifact. Raises if it is absent.

    Absence is a hard error rather than a skipped section: a report that
    silently drops a table when its input is missing is a page that cannot
    fail, which is the exact defect this report is about.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Every number on the G6 page must be "
            f"traceable to a named artifact; this section cannot be rendered "
            f"from anything else, and rendering it empty would be worse."
        )
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(path.read_text(encoding="utf-8"))


# -- the controls, as predicates -------------------------------------------
#
# Each function below is a WAY THIS PAGE COULD LIE, written so that the lie
# raises. They are module level, and not closures inside :func:`selftest`, for
# one reason: ``--selftest`` runs them against the REAL artifacts and the REAL
# rendered page on a machine that holds them, and
# ``wildfire_nowcast.sim.selftest`` runs them against inputs it builds itself,
# in a clone that holds neither. **Two callers, one implementation, so the
# thing pytest proves and the thing --selftest proves cannot drift apart.**
#
# The split is deliberate and neither half is sufficient alone. The adopted
# pytest cases prove each control CAN DISCRIMINATE - anywhere, with no
# ``runs/`` directory. ``--selftest`` proves the SHIPPED PAGE passes them here.
# A control that has only ever been handed a passing input has not been seen
# working, which is the finding this report leads its last section with.


class SelfCheckFailure(AssertionError):
    """A control did not discriminate. Names which one and in which direction."""


def assert_stored_bar_agrees(
    stored: tuple[float, float],
    interval: tuple[float, float],
    *,
    where: str,
) -> None:
    """THE LIE: the page prints a bar the code enforcing G3 has moved away from.

    ``stored`` is the interval an artifact recorded for itself when it was
    scored; ``interval`` is ``common.dispersion.BAR_INTERVAL`` read out of the
    code TODAY. They are two independent sources for one fact, which is the
    only reason comparing them means anything - our prose has carried
    ``[0.8, 1.2]`` while the code carried ``[0.8333, 1.2]``, and a page that
    quoted the prose would have reproduced exactly that defect.

    ``where`` names the artifact, because there is more than one draw and a
    refusal that does not say which file disagreed sends the reader to the
    wrong one.
    """
    if tuple(stored) != tuple(interval):
        raise ValueError(
            f"{where}: the artifact's stored interval {tuple(stored)} disagrees "
            f"with common.dispersion.BAR_INTERVAL {tuple(interval)}. "
            f"The page will not print a bar that two sources disagree about."
        )


def assert_page_is_self_contained(text: str) -> None:
    """THE LIE: the page renders here and is broken for everyone else.

    A ``<script>``, a ``<link>``, an ``@import`` or an ``<img>`` pointing at a
    sibling file all render perfectly on the machine that wrote the page and
    silently degrade - or blank - for a reader who was handed the one file.
    The failure is invisible to its author by construction, which is why it
    needs a check rather than a look.
    """
    for pat, why in (
        ("<script", "a script tag"),
        ("<link", "a link tag"),
        ("@import", "a CSS import"),
    ):
        if pat in text.lower():
            raise SelfCheckFailure(f"the page carries {why}")
    for src in re.findall(r'<img\s[^>]*src="([^"]{0,32})', text):
        if not src.startswith("data:image/"):
            raise SelfCheckFailure(f"an image is not inlined: {src!r}")
    outward = re.findall(r'(?:src|href)="(?!#|data:)([^"]+)"', text)
    if outward:
        raise SelfCheckFailure(f"the page reaches outside itself: {outward[:3]}")


def assert_colour_tokens_have_a_bare_root_definition(text: str) -> None:
    """THE LIE: the page is legible in the mode its author happened to use.

    Every ``var(--x)`` must resolve from the bare ``:root`` block. A token
    defined only inside ``@media (prefers-color-scheme: dark)`` or only inside
    ``[data-theme="dark"]`` is UNDEFINED in the other mode, and an undefined
    custom property falls back to nothing - which for a background or an ink
    colour means invisible text on a machine set the other way. The author
    never sees it.
    """
    css = re.search(r"<style>(.*?)</style>", text, re.S)
    if css is None:
        raise SelfCheckFailure("no stylesheet")
    base = re.search(r":root \{(.*?)\n\}", css.group(1), re.S)
    if base is None:
        raise SelfCheckFailure("no bare :root block")
    defined = set(re.findall(r"(--[a-z0-9-]+):", base.group(1)))
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", css.group(1)))
    missing = sorted(used - defined)
    if missing:
        raise SelfCheckFailure(
            f"{missing} exist only inside a media or [data-theme] block, so "
            f"they are undefined in whichever mode does not match"
        )


def assert_source_is_ascii(text: str) -> None:
    """THE LIE: this module is clean under a reader that can no longer see it.

    This module is not in the git index, so the tracked-source scanners cannot
    reach it and its prose/citation verdicts are taken BY HAND against its
    text. That hand reading is only durable while the source is ASCII: the
    page's typography is built from HTML entities and matplotlib mathtext
    precisely so the SOURCE carries none. One character typed directly puts the
    module back into the output-literal sink without anything saying so.
    """
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        raise SelfCheckFailure(
            f"non-ASCII in source: {bad}. The rendered page uses HTML entities "
            f"and matplotlib mathtext so that the SOURCE carries none; a "
            f"literal here lands in the output-literal sink."
        )


def assert_page_states_value(text: str, value: float, *, what: str = "the value") -> None:
    """THE LIE: the page and the artifact it cites hold different numbers.

    Ten decimal places, not four. A rounded comparison passes on a page whose
    number came from somewhere else entirely and happens to agree to the
    printed precision, and that is the failure worth catching: the page is
    supposed to be a VIEW of the artifact, not a second opinion about it.
    """
    if f"{value:.10f}" not in text:
        raise SelfCheckFailure(
            f"the page does not print {what} as {value:.10f}; a rendering that "
            f"silently disagrees with its artifact is worse than no page"
        )


#: Suffixes that make a token a FILENAME rather than a word. A superset of
#: ``tools/cited_paths.FILE_SUFFIXES``: that tool scans tracked SOURCE and may
#: be conservative about what it calls a path, whereas a page is prose read by a
#: stranger and every spelling that LOOKS like a file to a human should be
#: resolved. ``.gz`` is in here deliberately -- its absence from the shipped
#: tool is what made a gzipped run record invisible to the citation readers
#: (S18), and a scanner written today should not reproduce that blind spot.
PAGE_FILE_SUFFIXES: Final[tuple[str, ...]] = (
    ".cfg",
    ".csv",
    ".f90",
    ".gz",
    ".html",
    ".json",
    ".lock",
    ".md",
    ".png",
    ".pt",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
    ".zarr",
    ".zip",
)

#: ``{`` ``,`` ``}`` are INSIDE the token class on purpose. This page writes
#: ``runs/<name>_draw{A,B}.json.gz`` as one shorthand for two files, and a
#: scanner that stops at the brace sees a token with no extension, drops it, and
#: reports zero. **A shorthand for two untracked files would then be the one
#: spelling that could never be caught** - which is a new evasion in the shape
#: of the old one. They are expanded below and each expansion is resolved.
_PAGE_PATH_TOKEN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.,/{}-]*")
_BRACE: Final[re.Pattern[str]] = re.compile(r"\{([^{}]*)\}")


def _expand_braces(token: str) -> list[str]:
    """``a{x,y}b`` -> ``[axb, ayb]``; a token with no brace is returned unchanged.

    Left-to-right, one group at a time, with a hard cap: this is a legibility
    shorthand on a page, not a shell, and an unbounded product on adversarial
    input would be a denial of service inside a check.
    """
    out = [token]
    for _ in range(3):
        grown: list[str] = []
        for candidate in out:
            match = _BRACE.search(candidate)
            if match is None:
                grown.append(candidate)
                continue
            head, tail = candidate[: match.start()], candidate[match.end() :]
            grown.extend(head + option + tail for option in match.group(1).split(","))
        if grown == out:
            break
        out = grown[:16]
    return [t for t in out if "{" not in t and "}" not in t and "," not in t]


def path_tokens_in_rendered(text: str) -> dict[str, int]:
    """Every path-shaped token a READER of the rendered page can see, with counts.

    THE SOURCE IS THE WRONG THING TO SCAN AND THAT IS THE WHOLE POINT. This
    module assembles its paths at run time (see :func:`_p`), so a grep of the
    source under-counts by construction -- which is exactly how a page naming
    two untracked coordination documents passed every tracked-source scan we
    own. This reads the OUTPUT.

    Method, stated because a count from the wrong method is worse than no
    count. ``<style>`` and ``<script>`` blocks and every base64 ``data:``
    payload are removed; then the remainder is read TWICE, once with tags
    replaced by a space (what a reader sees) and once with tags left in place
    (which keeps attribute text such as ``alt`` in view), and the two token sets
    are unioned. Entities are unescaped on both.

    Brace shorthands are EXPANDED, not skipped: the page writes one token for
    two draw files, and a scanner that stopped at the brace would make that the
    one spelling nothing could catch.

    ITS ONE KNOWN LIMIT, stated rather than discovered later: a path split
    across a tag boundary -- one span holding the directory and the next holding
    the filename -- is not seen as one token. The FILENAME half still is: a bare
    filename is a token here, and the bare tier is resolved exactly like the
    rest, so the split hides the directory, never the file.
    """
    core = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    core = re.sub(r"<script\b.*?</script>", " ", core, flags=re.S | re.I)
    core = re.sub(r"data:[a-z/+.-]+;base64,[A-Za-z0-9+/=]+", " ", core)
    readings = (
        html.unescape(re.sub(r"<[^>]+>", " ", core)),
        html.unescape(core),
    )
    counts: dict[str, int] = {}
    for reading in readings:
        seen: dict[str, int] = {}
        for match in _PAGE_PATH_TOKEN.finditer(reading):
            for token in _expand_braces(match.group(0).rstrip("./-,")):
                if not token.endswith(PAGE_FILE_SUFFIXES):
                    continue
                segments = token.split("/")
                if any(segment == "" for segment in segments) or segments[-1].startswith("."):
                    continue
                seen[token] = seen.get(token, 0) + 1
        for token, n in seen.items():
            counts[token] = max(counts.get(token, 0), n)
    return counts


def resolves_against_the_index(token: str, tracked: Iterable[str]) -> bool:
    """Can a cloner open this token?

    Resolution is a TRAILING-FRAGMENT match against the git index, which is the
    convention ``tools/cited_paths.py`` already uses (``_suffix_index``): the
    page writes ``common/dispersion.py`` and the tracked file is
    ``src/wildfire_nowcast/common/dispersion.py``, and a reader follows that
    without help. Using a different rule here would put a second definition of
    one boundary into the tree, which is how a hand-off acquires a gap neither
    side can see.
    """
    for rel in tracked:
        if rel == token or rel.endswith("/" + token):
            return True
    return False


def tracked_paths(root: Path) -> frozenset[str]:
    """``git ls-files``: what a cloner receives, asked of git and not of the disk.

    The filesystem is the wrong oracle and this page paid for that twice: the
    project's coordination log and its planning document -- named here without
    their paths, for the reason immediately below -- both EXIST on this machine
    and NEITHER is in the index, so ``Path.exists()`` would have called the page
    clean while it sent a stranger to two files they were never given.

    THE PATHS THEMSELVES ARE DELIBERATELY NOT SPELLED IN THIS MODULE, and the
    first draft of this docstring did spell them. The tracked-source citation
    reader turned red on the two literals in the SOURCE of the check written to
    catch them in the OUTPUT, which is the same defect one level up and was
    caught only by running the gate. Where a comment must refer to one, it
    describes it.
    """
    import subprocess  # noqa: PLC0415

    out = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(line for line in out.stdout.splitlines() if line)


def assert_page_cites_nothing_a_cloner_cannot_open(
    text: str,
    *,
    tracked: Iterable[str],
    evidence: Iterable[str],
) -> None:
    """THE LIE: the page sends a reader to a file that is not in the repository.

    A path printed in prose is an instruction. This repository is public and
    the ``coordination`` directory is not in it, so a page that named this
    project's coordination log told a stranger to open something they were never
    given, and did it in the voice of a document whose whole argument is that a
    claim should rest on what can be checked. Nothing caught it: the literals
    were joined at run time, so the tracked-source scanners could not see them,
    and the files are present on the machine that wrote the page, so a reader
    looking for them here would find them.

    ``evidence`` is the ONLY class allowed to be unresolvable, and it is not a
    list anyone can grow by writing prose: it is ``ArtifactSet.as_rows()`` --
    the files this page reads -- so a new exemption costs a new INPUT. Those
    rows may be printed only while the page still carries
    :data:`EVIDENCE_DISCLOSURE`, which says in the reader's own words that the
    files are not in the repository. Delete the sentence and keep the table and
    this refuses: an undisclosed dead link is the defect, a disclosed origin is
    not.
    """
    known = frozenset(tracked)
    declared = frozenset(evidence)
    counts = path_tokens_in_rendered(text)
    unresolved = sorted(t for t in counts if not resolves_against_the_index(t, known))
    undeclared = [t for t in unresolved if t not in declared]
    if undeclared:
        raise SelfCheckFailure(
            f"the page names {len(undeclared)} path(s) a cloner cannot open and "
            f"does not declare: {undeclared}. A reader who follows one gets "
            f"nothing. Name the thing inline, or cite something in the index."
        )
    if unresolved and EVIDENCE_DISCLOSURE not in text:
        raise SelfCheckFailure(
            f"the page names {len(unresolved)} artifact(s) that are not in the "
            f"repository ({unresolved[:3]}) without the sentence that says so. "
            f"A provenance row is honest only while the page admits the reader "
            f"does not have the file."
        )


@dataclass(frozen=True)
class ArtifactSet:
    """The named files this page is allowed to read. Nothing else is opened."""

    root: Path
    g3_draw_a: Path
    g3_draw_b: Path
    reliability_draw_a: Path
    reliability_draw_b: Path
    m33_summary: Path
    m34_summary: Path
    g5_four_blocks: Path
    creek_episode: Path
    episode_png: Path
    elmfire_cost_creek: Path
    elmfire_cost_four: Path

    @staticmethod
    def default(root: Path) -> ArtifactSet:
        runs = root / _p("runs")
        s14 = root / _p("reports", "figures", "s14g5")
        s16 = root / _p("reports", "figures", "s16")
        return ArtifactSet(
            root=root,
            g3_draw_a=runs / _p("m30_g3_drawA.json.gz"),
            g3_draw_b=runs / _p("m30_g3_drawB.json.gz"),
            reliability_draw_a=runs / _p("m30_reliability_drawA.json.gz"),
            reliability_draw_b=runs / _p("m30_reliability_drawB.json.gz"),
            m33_summary=runs / _p("m33_summary.json"),
            m34_summary=runs / _p("m34_summary.json"),
            g5_four_blocks=s14 / _p("g5_four_blocks.json"),
            creek_episode=s16 / _p("creek_episode.json"),
            episode_png=s16 / _p("creek_episode.png"),
            elmfire_cost_creek=s14 / _p("creek_cost_projection.json"),
            elmfire_cost_four=s14 / _p("measured_cost_4blocks.json"),
        )

    def as_rows(self) -> list[tuple[str, str, str]]:
        """(field, path relative to root, what the page takes from it)."""
        what = {
            "g3_draw_a": "G3 criteria, every arm, draw A",
            "g3_draw_b": "G3 criteria, every arm, draw B",
            "reliability_draw_a": "reliability bins and Murphy terms, draw A",
            "reliability_draw_b": "reliability bins and Murphy terms, draw B",
            "m33_summary": "resolution against the ellipse over 17 binnings",
            "m34_summary": "calibration_error achievability ceiling",
            "g5_four_blocks": "ELMFIRE, 4 held-out blocks, 19 growth windows",
            "creek_episode": "the Creek commitment run, 91 windows, both draws",
            "episode_png": "the rendered episode (embedded)",
            "elmfire_cost_creek": "the projected ELMFIRE cost of the fifth block",
            "elmfire_cost_four": "the measured ELMFIRE cost of the four that ran",
        }
        rows: list[tuple[str, str, str]] = []
        for name, note in what.items():
            p: Path = getattr(self, name)
            rows.append((name, os.path.relpath(p, self.root), note))
        return rows


@dataclass
class Facts:
    """Everything the page states, read once, from the artifacts above."""

    artifacts: ArtifactSet
    bar_low: float
    bar_high: float
    bar_source: str
    bar_log: float
    #: The calibration bar's own provenance string, READ OUT OF THE ARTIFACT
    #: rather than typed. The number it labels has no derivation in this
    #: repository, and the page says so; what a cloner CAN check is this
    #: label and the module that was shaped to fit it, so the label is quoted
    #: from the file that stored it instead of retyped from another package.
    cal_bar_source: str = ""
    dispersion: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration: dict[str, dict[str, Any]] = field(default_factory=dict)
    disp_detail: dict[str, str] = field(default_factory=dict)
    per_block: dict[str, dict[str, float]] = field(default_factory=dict)
    scope: dict[str, Any] = field(default_factory=dict)
    elmfire: dict[str, Any] = field(default_factory=dict)
    elmfire_cost: dict[str, Any] = field(default_factory=dict)
    m33: dict[str, Any] = field(default_factory=dict)
    m34: dict[str, Any] = field(default_factory=dict)
    episode: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    registry: dict[str, Any] = field(default_factory=dict)


def _load_registry_facts() -> dict[str, Any]:
    """Read the C6 metric registry FROM THE CODE, not from prose.

    The whole point of ADR-170 is that our prose and our contract disagree
    about which metrics may decide G3. Quoting the prose on the page would
    reproduce the defect the page is reporting.
    """
    from wildfire_nowcast.common.null_check.registry import (  # noqa: PLC0415
        C6_METRICS,
        assert_may_adjudicate,
    )

    eligible = sorted(k for k, v in C6_METRICS.items() if getattr(v, "gate_eligible", False))
    quarantined: list[tuple[str, str]] = []
    for key in sorted(C6_METRICS):
        if key.startswith(("calibration_error_", "reliability_")):
            by = getattr(C6_METRICS[key], "quarantined_by", "") or ""
            if key.startswith(
                (
                    "calibration_error_1",
                    "calibration_error_2",
                    "calibration_error_3",
                    "reliability_",
                )
            ):
                quarantined.append((key, by))
    raised = ""
    try:
        assert_may_adjudicate("calibration_error")
    except Exception as exc:  # the registry's own refusal, quoted verbatim
        raised = f"{type(exc).__name__}: {exc}"
    return {
        "n_metrics": len(C6_METRICS),
        "eligible": eligible,
        "quarantined": quarantined,
        "raised": raised,
    }


def load_facts(artifacts: ArtifactSet) -> Facts:
    """Load every number the page states. Nothing is recomputed from a tensor."""
    from wildfire_nowcast.common.dispersion import BAR_INTERVAL  # noqa: PLC0415

    a = _read_json(artifacts.g3_draw_a)
    b = _read_json(artifacts.g3_draw_b)

    # BOTH draws, not just the one the bar is read from. ADR-142 (e) makes the
    # page report only verdicts that agree across draws, so draw B's numbers are
    # printed against this bar too - and until S18 nothing had ever looked at the
    # interval draw B stored for itself. A cross-check that covers one of two
    # sources is a cross-check with a blind side, and the blind side is the one
    # nobody was looking at.
    bar_entry = next(e for e in a["g3"]["bar"] if e["key"] == "band_area_dispersion_ratio")
    for draw, raw in (("draw A", a), ("draw B", b)):
        entry = next(e for e in raw["g3"]["bar"] if e["key"] == "band_area_dispersion_ratio")
        assert_stored_bar_agrees(
            (float(entry["low"]), float(entry["high"])),
            (float(BAR_INTERVAL[0]), float(BAR_INTERVAL[1])),
            where=f"{draw} ({os.path.basename(str(getattr(artifacts, _DRAW_FIELD[draw])))})",
        )

    facts = Facts(
        artifacts=artifacts,
        bar_low=float(BAR_INTERVAL[0]),
        bar_high=float(BAR_INTERVAL[1]),
        bar_source=str(bar_entry["source"]),
        bar_log=math.log(float(BAR_INTERVAL[1])),
        cal_bar_source=str(
            next(e for e in a["g3"]["bar"] if e["key"] == "band_calibration_error")["source"]
        ),
        scope=dict(a["scope"]),
    )

    for arm in ARM_ORDER:
        da = a["g3"]["models"][arm]["criteria"][DISPERSION_LABEL]
        db = b["g3"]["models"][arm]["criteria"][DISPERSION_LABEL]
        ca = a["g3"]["models"][arm]["criteria"][CALIBRATION_LABEL]
        cb = b["g3"]["models"][arm]["criteria"][CALIBRATION_LABEL]
        facts.dispersion[arm] = {
            "A": float(da["equal_block"]),
            "B": float(db["equal_block"]),
            "A_pooled": float(da["window_pooled"]),
            "B_pooled": float(db["window_pooled"]),
            "in_A": bool(da["in_interval_equal_block"]),
            "in_B": bool(db["in_interval_equal_block"]),
            "outcome": str(da["condition"]["outcome"]),
        }
        facts.disp_detail[arm] = str(da["condition"]["detail"])
        facts.calibration[arm] = {
            "A": float(ca["equal_block"]),
            "B": float(cb["equal_block"]),
            "in_A": bool(ca["in_interval_equal_block"]),
            "in_B": bool(cb["in_interval_equal_block"]),
        }
        pba = da.get("per_block")
        if isinstance(pba, dict):
            facts.per_block[arm] = {k: float(v) for k, v in pba.items()}

    facts.elmfire = _read_json(artifacts.g5_four_blocks)
    facts.elmfire_cost = {
        "creek": _read_json(artifacts.elmfire_cost_creek),
        "four": _read_json(artifacts.elmfire_cost_four),
    }
    facts.m33 = _read_json(artifacts.m33_summary)
    facts.m34 = _read_json(artifacts.m34_summary)
    facts.episode = _read_json(artifacts.creek_episode)
    facts.reliability = {
        "A": _read_json(artifacts.reliability_draw_a),
        "B": _read_json(artifacts.reliability_draw_b),
    }
    facts.registry = _load_registry_facts()
    return facts


# -- figures ---------------------------------------------------------------
#
# Every figure here is rendered to PNG bytes and inlined as a data URI. The
# page must open with no network and no sibling files, so nothing may be a
# <img src="reports/figures/...">.


def _fig_to_uri(fig: Figure, dpi: int = 128) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _png_to_uri(path: Path, max_width: int) -> tuple[str, int, int, int]:
    """Downscale an existing raster and inline it. Returns (uri, w, h, kib)."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as src:
        im = src.convert("RGB")
        w, h = im.size
        if w > max_width:
            h = int(round(h * max_width / w))
            w = max_width
            im = im.resize((w, h), Image.Resampling.LANCZOS)
        # A 256-colour palette is lossless enough for a figure of flat fills
        # and thin strokes, and cuts the embedded bytes ~2.7x. Dithering is
        # OFF on purpose: it would speckle the probability rasters, and a
        # speckle in a burn-probability map reads as data.
        quant = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        buf = io.BytesIO()
        quant.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode("ascii")
    return uri, w, h, len(raw) // 1024


def _axes_style(ax: Any) -> None:
    ax.set_facecolor(PAPER)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=8.5)
    ax.grid(True, color=GRID, alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_dispersion(facts: Facts) -> str:
    """The G3 verdict in one panel: seven arms on a log axis against the bar."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4), facecolor=PAPER)
    _axes_style(ax)

    rows: list[tuple[str, float, float | None]] = []
    for arm in ARM_ORDER:
        d = facts.dispersion[arm]
        rows.append((arm, d["A"], d["B"]))
    elm = float(
        facts.elmfire["pooled_subset"]["elmfire"]["growth_windows"]["band_area_dispersion_ratio"]
    )
    rows.append(("elmfire", elm, None))

    ax.axvspan(facts.bar_low, facts.bar_high, color=COL_BAR, alpha=0.16, zorder=0)
    ax.axvline(1.0, color=COL_BAR, linewidth=1.0, linestyle="--", alpha=0.8)

    ys = list(range(len(rows)))[::-1]
    for y, (arm, va, vb) in zip(ys, rows, strict=True):
        c = ARM_COLORS[arm]
        if va <= 0.0:
            ax.plot([0.006], [y], marker="x", color=c, markersize=9, mew=2.0)
            ax.text(
                0.0072, y, "  0.0 - UNDEFINED, not a pass", va="center", fontsize=8.5, color=INK
            )
            continue
        ax.plot([va], [y], marker="o", color=c, markersize=8, zorder=3)
        if vb is not None:
            ax.plot([vb], [y], marker="D", color=c, markersize=5.5, markerfacecolor=PAPER, zorder=3)
            ax.plot([min(va, vb), max(va, vb)], [y, y], color=c, linewidth=1.6, alpha=0.7, zorder=2)
        lo = min(va, vb) if vb is not None else va
        ax.text(lo * 0.86, y, f"{va:.4f}", va="center", ha="right", fontsize=8.5, color=INK)

    pb = facts.per_block["m30_ref"]
    y_ours = ys[ARM_ORDER.index("m30_ref")]
    for v in pb.values():
        ax.plot(
            [v], [y_ours - 0.30], marker="|", color=COL_OURS, markersize=9, alpha=0.65, zorder=2
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([ARM_LABELS[a] for a, _, _ in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 2.4)
    ax.set_xlabel(
        "band_area_dispersion_ratio, equal-block over 5 held-out blocks "
        "(log scale); 1.0 is the target",
        fontsize=9,
        color=INK,
    )
    ax.set_title(
        f"G3 dispersion: zero of seven arms inside the bar "
        f"[{facts.bar_low:.4f}, {facts.bar_high:.4f}]",
        fontsize=11,
        color=INK,
        loc="left",
    )
    from matplotlib.lines import Line2D  # noqa: PLC0415

    ax.legend(
        handles=[
            Line2D([], [], marker="o", color=INK, linestyle="none", markersize=7, label="draw A"),
            Line2D(
                [],
                [],
                marker="D",
                color=INK,
                markerfacecolor=PAPER,
                linestyle="none",
                markersize=5,
                label="draw B",
            ),
            Line2D(
                [],
                [],
                marker="|",
                color=COL_OURS,
                linestyle="none",
                markersize=9,
                label="per held-out block (ours)",
            ),
        ],
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        handletextpad=0.4,
        columnspacing=1.8,
    )
    return _fig_to_uri(fig)


def fig_latent_ablation(facts: Facts) -> str:
    """The project's one clean positive: the shared latent buys dispersion."""
    fig, ax = plt.subplots(figsize=(5.6, 3.3), facecolor=PAPER)
    _axes_style(ax)
    lat = facts.dispersion["m30_ref"]
    ind = facts.dispersion["m30_ref__independent"]
    xs = [0, 1]
    for draw in ("A", "B"):
        vals = [ind[draw], lat[draw]]
        ax.plot(
            xs,
            vals,
            marker="o",
            color=(COL_OURS if draw == "A" else COL_ABLATION),
            linewidth=2.0,
            markersize=7,
            label=f"draw {draw}, $\\times${vals[1] / vals[0]:.4f}",
        )
        dy = 0.011 if draw == "A" else -0.023
        for x, v in zip(xs, vals, strict=True):
            ax.text(
                x,
                v + dy,
                f"{v:.4f}",
                ha="center",
                fontsize=8,
                color=(COL_OURS if draw == "A" else COL_ABLATION),
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        ["independent per-pixel noise\n(ablation)", "shared per-step latent z_t\n(the model)"],
        fontsize=8.5,
    )
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("dispersion ratio", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(
        "The ablation does collapse: the latent roughly doubles spread",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    return _fig_to_uri(fig)


def fig_calibration_ceiling(facts: Facts) -> str:
    """Why the calibration conjunct could not have failed."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6), facecolor=PAPER)
    _axes_style(ax)
    tbl = facts.m34["table"]["3"]["ATOM"]
    bar = float(facts.m34["bar"])
    arms = list(ARM_ORDER)
    xs = np.arange(len(arms), dtype=float)
    attained = [float(tbl[a]["A"]["calibration_error"]) for a in arms]
    ceiling = [float(tbl[a]["A"]["ceiling_mean_p_plus_base"]) for a in arms]

    top_ceiling = float(facts.m34["controls"]["WORST_achievability_ceiling_anywhere"]["ceiling"])
    ax.axhspan(1e-4, top_ceiling, color=GRID, alpha=0.30, zorder=0)
    ax.text(
        -0.42,
        top_ceiling * 1.10,
        "everything this statistic can attain, ANYWHERE in the corpus",
        fontsize=8.2,
        color=INK,
        va="bottom",
    )
    ax.axhline(bar, color=COL_FAIL, linewidth=1.8)
    ax.text(
        len(arms) - 0.4,
        bar * 1.06,
        f"the G3 bar, {bar:.2f}",
        color=COL_FAIL,
        fontsize=9,
        ha="right",
    )
    for x, c, v in zip(xs, ceiling, attained, strict=True):
        ax.plot([x, x], [v, c], color=GRID, linewidth=6, solid_capstyle="butt", alpha=0.7, zorder=1)
    ax.scatter(
        xs,
        ceiling,
        marker="_",
        s=380,
        color=INK,
        linewidths=2.2,
        zorder=3,
        label="the highest value the statistic CAN take\n(mean forecast + base rate)",
    )
    ax.scatter(
        xs, attained, s=58, zorder=4, color=[ARM_COLORS[a] for a in arms], label="value attained"
    )
    for x, v in zip(xs, attained, strict=True):
        ax.text(x, v * 0.72, f"{v:.5f}", ha="center", fontsize=7.6, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1.5e-3, 0.30)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABELS[a].replace(" (", "\n(") for a in arms], fontsize=8)
    ax.set_ylabel("calibration_error at 3 h (log scale)", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    worst = facts.m34["controls"]["WORST_achievability_ceiling_anywhere"]
    ax.set_title(
        "The bar sits outside the range the statistic can reach: the highest "
        f"ceiling anywhere is {float(worst['ceiling']):.6f}, "
        f"{float(worst['bar_over_ceiling']):.2f}$\\times$ below the bar",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    return _fig_to_uri(fig)


def _bins_for(rel: dict[str, Any], arm: str, lead: int) -> list[dict[str, Any]]:
    return [
        b for b in rel["pooled_growth_windows"][arm]["reliability_bins"] if int(b["lead_h"]) == lead
    ]


def fig_reliability(facts: Facts, lead: int = 3) -> str:
    """The curve, on BOTH draws, with the occupancy that makes it readable.

    ADR-142(e) makes agreeing verdicts across both seed draws the reportable
    unit. The S15 version of this figure existed on draw A only and said so on
    its face; the draw-B reliability record now exists -- it is
    :attr:`ArtifactSet.reliability_draw_b`, and the page prints its path in the
    traceability table beside every number taken from it -- so the curve is
    drawn twice and the reader can see for themselves that it agrees.

    THE PATH IS NAMED BY ATTRIBUTE HERE AND NOT SPELLED, and this docstring is
    where I31 caught me. It carried the gzipped filename as a literal, which is
    a citation of a record no cloner receives; while the ``runs/`` walk had no
    ``.gz`` suffix that citation was invisible, and the moment the walk could
    see it, it said so. **The blind spot I reported closed onto my own line.**
    Repaired the way the checker's own message prescribes: name the thing
    inline, keep the claim.
    """
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(9.2, 5.6), height_ratios=[2.5, 1.0], sharex=True, facecolor=PAPER
    )
    _axes_style(ax)
    _axes_style(axn)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.2, linestyle="--")
    ax.text(
        0.335,
        0.305,
        "perfect calibration",
        fontsize=8,
        color=INK,
        alpha=0.65,
        ha="left",
        va="bottom",
    )

    for arm, style in (("ellipse", "-"), ("m30_ref", "-")):
        for draw, alpha, marker in (("A", 1.0, "o"), ("B", 0.45, "D")):
            bins = _bins_for(facts.reliability[draw], arm, lead)
            xs = [float(b["mean_forecast"]) for b in bins if int(b["n"]) > 0]
            ys = [float(b["observed_frequency"]) for b in bins if int(b["n"]) > 0]
            ax.plot(
                xs,
                ys,
                style,
                marker=marker,
                markersize=5,
                alpha=alpha,
                color=ARM_COLORS[arm],
                linewidth=1.8,
                label=f"{ARM_LABELS[arm]}, draw {draw}",
            )
    ax.set_ylim(-0.01, 0.35)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylabel("observed burn frequency", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(
        f"Reliability at {lead} h, pooled growth windows: the ellipse is "
        "ORDERED and we INVERT above p $\\approx$ 0.25",
        fontsize=10.5,
        color=INK,
        loc="left",
    )

    width = 0.036
    for arm, off in (("ellipse", -width / 2), ("m30_ref", +width / 2)):
        bins = _bins_for(facts.reliability["A"], arm, lead)
        centres = [(float(b["bin_lower"]) + float(b["bin_upper"])) / 2 for b in bins]
        ns = [max(int(b["n"]), 0) for b in bins]
        axn.bar(
            [c + off for c in centres],
            ns,
            width=width,
            color=ARM_COLORS[arm],
            alpha=0.85,
            label=f"{ARM_LABELS[arm]}",
        )
    axn.set_yscale("symlog", linthresh=1)
    axn.set_ylabel("cells in bin\n(log, draw A)", fontsize=8.5, color=INK)
    axn.set_xlabel("forecast probability bin", fontsize=9, color=INK)
    ours = _bins_for(facts.reliability["A"], "m30_ref", lead)
    tail = sum(int(b["n"]) for b in ours if float(b["bin_lower"]) >= 0.5)
    axn.text(
        0.52,
        0.80,
        f"our whole p $\\geq$ 0.5 tail at {lead} h is {tail} cells, 0 of which burned",
        transform=axn.transAxes,
        fontsize=8.2,
        color=COL_FAIL,
    )
    return _fig_to_uri(fig)


def fig_binning_ladder(facts: Facts) -> str:
    """ADR-169: the ellipse comparison was the bin edges, not the forecasters."""
    fig, ax = plt.subplots(figsize=(9.2, 3.8), facecolor=PAPER)
    _axes_style(ax)
    cells = facts.m33["check_a_3h"]
    order = [k for k in cells if k.startswith("EW(")] + ["ATOM"]
    xs = np.arange(len(order), dtype=float)
    unc = float(cells["ATOM"]["uncertainty"])
    ell = [100.0 * float(cells[k]["res_ell_A"]) / unc for k in order]
    ours = [100.0 * float(cells[k]["res_ours_A"]) / unc for k in order]
    ax.plot(xs, ell, marker="o", color=COL_ELLIPSE, linewidth=2.0, label="wind-advected ellipse")
    ax.plot(xs, ours, marker="o", color=COL_OURS, linewidth=2.0, label="our kernel")
    ell_b = [100.0 * float(cells[k]["res_ell_B"]) / unc for k in order]
    ours_b = [100.0 * float(cells[k]["res_ours_B"]) / unc for k in order]
    ax.plot(xs, ell_b, color=COL_ELLIPSE, linewidth=1.0, alpha=0.45, linestyle=":", label="draw B")
    ax.plot(xs, ours_b, color=COL_OURS, linewidth=1.0, alpha=0.45, linestyle=":")
    i10 = order.index("EW(10)")
    ax.axvline(i10, color=GRID, linewidth=1.0)
    ax.annotate(
        "the binning ADR-162 happened to use:\nwe read HALF the ellipse",
        xy=(i10, ours[i10]),
        xytext=(0.18, 0.60),
        fontsize=8.2,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": GRID},
    )
    ia = order.index("ATOM")
    ax.annotate(
        "at the finest binning we read SLIGHTLY MORE,\nand the difference is not significant",
        xy=(ia, ours[ia]),
        xytext=(ia - 3.4, 0.55),
        fontsize=8.2,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": GRID},
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("share of outcome variance resolved (%)", fontsize=9, color=INK)
    ax.set_xlabel("binning of the forecast probability, coarse to fine", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(
        "Twelve of seventeen binnings reverse the ordering; the forecasts never changed",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    return _fig_to_uri(fig)


def fig_commitment_run(facts: Facts) -> str:
    """What the gate scores, and what it does not: the Creek Diablo run."""
    run = facts.episode["run"]
    rows = run["rows"]
    fig, (ax, axw) = plt.subplots(
        2, 1, figsize=(9.4, 4.9), height_ratios=[2.2, 1.0], sharex=True, facecolor=PAPER
    )
    _axes_style(ax)
    _axes_style(axw)

    t0 = [int(r["t0"]) for r in rows]
    ca = [int(r["n_confident_A"]) for r in rows]
    cb = [int(r["n_confident_B"]) for r in rows]
    scored = [bool(r["scored_by_the_gate"]) for r in rows]

    for x, s in zip(t0, scored, strict=True):
        if s:
            ax.axvspan(x - 0.5, x + 0.5, color=COL_BAR, alpha=0.20, zorder=0)
            axw.axvspan(x - 0.5, x + 0.5, color=COL_BAR, alpha=0.20, zorder=0)
    ax.bar(t0, ca, width=0.86, color=COL_OURS, label="draw A", zorder=2)
    ax.plot(t0, cb, color=COL_ABLATION, linewidth=1.4, marker=None, label="draw B", zorder=3)
    for k, inc in enumerate(run["label_increments"]):
        # Two of these land one hour apart; alternating the depth keeps
        # "+9" and "+3" from rendering as "+93".
        depth = -26 if k % 2 == 0 else -40
        ax.annotate(
            f"+{int(inc['new_cells'])} label cells",
            xy=(int(inc["t0_of_step"]), 0),
            xytext=(int(inc["t0_of_step"]), depth),
            ha="center",
            fontsize=7.6,
            color=COL_BAR,
            arrowprops={"arrowstyle": "->", "color": COL_BAR},
        )
    ax.set_ylim(-54, max(ca) * 1.18)
    ax.set_ylabel("cells at p $\\geq$ 0.5", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    share = (
        100.0
        * int(run["n_confident_cells_drawA_in_scored_windows"])
        / int(run["n_confident_cells_drawA"])
    )
    ax.set_title(
        f"{run['n_confident_cells_drawA']:,} confident cells over "
        f"{run['n_windows']} hours of 2020_creek; "
        f"{run['n_confident_cells_drawA_in_scored_windows']} of them "
        f"({share:.1f}%) fall in the {run['n_windows_scored']} windows the gate scores",
        fontsize=10.5,
        color=INK,
        loc="left",
    )

    axw.plot(
        t0,
        [float(r["max_wind_ms"]) for r in rows],
        color=COL_ELLIPSE,
        linewidth=1.5,
        label="max wind (m/s)",
    )
    axr = axw.twinx()
    axr.plot(
        t0,
        [float(r["min_rh_pct"]) for r in rows],
        color=COL_ELMFIRE,
        linewidth=1.5,
        linestyle="--",
        label="min RH (%)",
    )
    axr.set_ylabel("min RH (%)", fontsize=8.5, color=COL_ELMFIRE)
    axr.tick_params(colors=COL_ELMFIRE, labelsize=8)
    axr.spines["top"].set_visible(False)
    axw.set_ylabel("max wind\n(m/s)", fontsize=8.5, color=COL_ELLIPSE)
    axw.set_xlabel(
        "t0 (hour index into the fire); shaded hours are the ones the gate scores",
        fontsize=9,
        color=INK,
    )
    return _fig_to_uri(fig)


def fig_elmfire(facts: Facts) -> str:
    """The physics baseline fails the same criterion, on its own four blocks."""
    fig, ax = plt.subplots(figsize=(7.8, 3.4), facecolor=PAPER)
    _axes_style(ax)
    per = facts.elmfire["per_fire"]
    fires = list(per)
    xs = np.arange(len(fires), dtype=float)
    width = 0.27
    for i, arm in enumerate(("persistence", "ellipse", "elmfire")):
        vals = [
            float(per[f]["models"][arm]["growth_windows"]["band_area_dispersion_ratio"])
            for f in fires
        ]
        label = ARM_LABELS[arm]
        if arm == "persistence":
            label += ": 0.0000 on every block, deterministic, no spread"
        ax.bar(xs + (i - 1) * width, vals, width=width, color=ARM_COLORS[arm], label=label)
        for x, v in zip(xs + (i - 1) * width, vals, strict=True):
            if v > 0:
                ax.text(x, v + 0.02, f"{v:.4f}", ha="center", fontsize=7.2, color=INK, rotation=90)
    ax.axhspan(facts.bar_low, facts.bar_high, color=COL_BAR, alpha=0.16, zorder=0)
    ax.text(
        1.6,
        (facts.bar_low + facts.bar_high) / 2,
        f"the same bar, [{facts.bar_low:.4f}, {facts.bar_high:.4f}]",
        fontsize=8.5,
        color=COL_BAR,
        ha="center",
        va="center",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"{f}\n({int(per[f]['n_growth_windows'])} growth windows)" for f in fires], fontsize=8
    )
    ax.set_ylim(0, 1.65)
    ax.set_ylabel("band_area_dispersion_ratio", fontsize=9, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_title(
        "ELMFIRE with default Rothermel fails the dispersion criterion on 4 of 4 blocks",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    return _fig_to_uri(fig)


# -- the page --------------------------------------------------------------

CSS: Final[str] = """
/* The FULL light palette is defined here, on bare :root, so that every colour
   token has a definition that does not depend on a media query or an
   attribute. Dark mode below redefines tokens only; it introduces none. */
:root {
  --paper: #fbfaf7;
  --panel: #ffffff;
  --panel-2: #f3f0e9;
  --ink: #1b1a17;
  --ink-2: #4b4740;
  --ink-3: #6f6a60;
  --rule: #ddd7c9;
  --rule-2: #c9c3b6;
  --fail: #b03a2e;
  --fail-bg: #fbeae7;
  --pass: #2f7d4f;
  --pass-bg: #e8f3ec;
  --warn: #9a6b12;
  --warn-bg: #fdf3df;
  --ours: #c0392b;
  --ellipse: #2c6fa8;
  --link: #1f5c8b;
  --shadow: rgba(27, 26, 23, 0.08);
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #14130f;
    --panel: #1c1a16;
    --panel-2: #232019;
    --ink: #ece7dc;
    --ink-2: #c3bdb0;
    --ink-3: #948d80;
    --rule: #33302a;
    --rule-2: #45413a;
    --fail: #ef8b7c;
    --fail-bg: #2c1a17;
    --pass: #7fc79b;
    --pass-bg: #16261d;
    --warn: #e0b25e;
    --warn-bg: #2a2113;
    --ours: #ef8b7c;
    --ellipse: #7db4dd;
    --link: #86b9e2;
    --shadow: rgba(0, 0, 0, 0.45);
  }
}
:root[data-theme="dark"] {
  --paper: #14130f;
  --panel: #1c1a16;
  --panel-2: #232019;
  --ink: #ece7dc;
  --ink-2: #c3bdb0;
  --ink-3: #948d80;
  --rule: #33302a;
  --rule-2: #45413a;
  --fail: #ef8b7c;
  --fail-bg: #2c1a17;
  --pass: #7fc79b;
  --pass-bg: #16261d;
  --warn: #e0b25e;
  --warn-bg: #2a2113;
  --ours: #ef8b7c;
  --ellipse: #7db4dd;
  --link: #86b9e2;
  --shadow: rgba(0, 0, 0, 0.45);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.62;
  margin: 0;
  padding: 0 1.25rem 6rem;
  overflow-x: hidden;
}
.wrap { max-width: 62rem; margin: 0 auto; }
header.masthead {
  border-bottom: 2px solid var(--ink);
  padding: 3rem 0 1.25rem;
  margin-bottom: 2rem;
}
h1 { font-size: 2.1rem; line-height: 1.18; margin: 0 0 0.5rem; letter-spacing: -0.015em; }
h2 {
  font-size: 1.32rem; line-height: 1.3; margin: 3.2rem 0 0.9rem;
  padding-top: 1.1rem; border-top: 1px solid var(--rule);
}
h3 { font-size: 1.03rem; margin: 2rem 0 0.6rem; color: var(--ink); }
p { margin: 0 0 1rem; }
a { color: var(--link); }
.sub { color: var(--ink-2); font-size: 0.95rem; margin: 0; }
.stamp {
  font-family: var(--mono); font-size: 0.76rem; color: var(--ink-3);
  margin-top: 0.9rem; line-height: 1.55;
}
code, .mono, kbd { font-family: var(--mono); font-size: 0.86em; }
code { background: var(--panel-2); padding: 0.08em 0.32em; border-radius: 3px; }
.num { font-variant-numeric: tabular-nums; }

.verdict {
  border: 1px solid var(--rule-2); border-left: 5px solid var(--fail);
  background: var(--fail-bg); color: var(--ink);
  padding: 1rem 1.15rem; margin: 1.4rem 0; border-radius: 4px;
}
.verdict.good { border-left-color: var(--pass); background: var(--pass-bg); }
.verdict.warn { border-left-color: var(--warn); background: var(--warn-bg); }
.verdict .lede { font-weight: 650; font-size: 1.04rem; margin-bottom: 0.35rem; }
.verdict p:last-child { margin-bottom: 0; }

.note {
  border-left: 3px solid var(--rule-2); padding: 0.2rem 0 0.2rem 0.95rem;
  color: var(--ink-2); margin: 1.1rem 0; font-size: 0.95rem;
}
blockquote.record {
  margin: 1.2rem 0; padding: 0.9rem 1.1rem; background: var(--panel-2);
  border-left: 4px solid var(--ink-3); border-radius: 3px;
  font-size: 0.97rem; color: var(--ink);
}

.scroll { overflow-x: auto; margin: 1.2rem 0; -webkit-overflow-scrolling: touch; }
table {
  border-collapse: collapse; width: 100%; min-width: 34rem;
  font-size: 0.88rem; font-variant-numeric: tabular-nums;
  background: var(--panel);
}
caption {
  caption-side: bottom; text-align: left; color: var(--ink-3);
  font-size: 0.79rem; padding-top: 0.5rem; line-height: 1.5;
}
th, td { padding: 0.42rem 0.62rem; border-bottom: 1px solid var(--rule); text-align: right; }
th {
  color: var(--ink-2); font-weight: 620; white-space: nowrap;
  border-bottom: 1.5px solid var(--rule-2);
}
th:first-child, td:first-child { text-align: left; }
tbody tr:last-child td { border-bottom: 1.5px solid var(--rule-2); }
tr.ours td { background: var(--panel-2); font-weight: 620; }
td.f { color: var(--fail); }
td.p { color: var(--pass); }
.tag {
  display: inline-block; font-family: var(--mono); font-size: 0.72rem;
  padding: 0.05rem 0.42rem; border-radius: 3px; border: 1px solid var(--rule-2);
  color: var(--ink-2); background: var(--panel-2); white-space: nowrap;
}
.tag.f { color: var(--fail); border-color: var(--fail); background: var(--fail-bg); }
.tag.p { color: var(--pass); border-color: var(--pass); background: var(--pass-bg); }
.tag.w { color: var(--warn); border-color: var(--warn); background: var(--warn-bg); }

figure { margin: 1.6rem 0; }
figure .scroll { margin: 0; }
figure img {
  display: block; width: 100%; height: auto; min-width: 30rem;
  border: 1px solid var(--rule); border-radius: 4px; background: var(--panel);
}
figcaption {
  color: var(--ink-3); font-size: 0.79rem; margin-top: 0.55rem; line-height: 1.55;
}
figcaption .src { font-family: var(--mono); color: var(--ink-3); }

ol.findings { padding-left: 1.3rem; }
ol.findings > li { margin-bottom: 1.1rem; }
ol.findings > li > .h { font-weight: 640; }
ul.plain { padding-left: 1.2rem; }
ul.plain li { margin-bottom: 0.45rem; }

.toc { background: var(--panel-2); border: 1px solid var(--rule); border-radius: 5px;
       padding: 0.9rem 1.15rem; margin: 1.6rem 0 2.4rem; }
.toc ol { margin: 0; padding-left: 1.35rem; }
.toc li { margin: 0.18rem 0; }
.toc a { text-decoration: none; }
.toc a:hover { text-decoration: underline; }

footer.colophon {
  margin-top: 4rem; padding-top: 1.2rem; border-top: 2px solid var(--ink);
  color: var(--ink-2); font-size: 0.86rem;
}
@media (max-width: 40rem) {
  body { font-size: 15px; padding: 0 0.85rem 4rem; }
  h1 { font-size: 1.65rem; }
}
"""


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _f(x: float, n: int = 4) -> str:
    return f"{x:.{n}f}"


def _dispersion_table(facts: Facts) -> str:
    elmfire_rel = os.path.relpath(facts.artifacts.g5_four_blocks, facts.artifacts.root)
    elm_pool = float(
        facts.elmfire["pooled_subset"]["elmfire"]["growth_windows"]["band_area_dispersion_ratio"]
    )
    rows: list[str] = []
    for arm in ARM_ORDER:
        d = facts.dispersion[arm]
        cls = ' class="ours"' if arm == "m30_ref" else ""
        undef = d["outcome"] == "undefined"
        verdict = (
            '<span class="tag w">UNDEFINED</span>'
            if undef
            else '<span class="tag f">outside</span>'
        )
        rows.append(
            f"<tr{cls}><td>{_esc(ARM_LABELS[arm])}<br>"
            f'<span class="tag">{_esc(arm)}</span></td>'
            f'<td class="f">{_f(d["A"], 10)}</td>'
            f'<td class="f">{_f(d["B"], 10)}</td>'
            f"<td>{_f(d['A_pooled'], 10)}</td>"
            f"<td>{verdict}</td></tr>"
        )
    rows.append(
        f"<tr><td>{_esc(ARM_LABELS['elmfire'])}<br>"
        f'<span class="tag">elmfire &middot; 4 blocks &middot; stride 16'
        f" &middot; 4 members</span></td>"
        f'<td class="f">{_f(elm_pool, 10)}</td><td>&mdash;</td><td>&mdash;</td>'
        f'<td><span class="tag f">outside</span></td></tr>'
    )
    return (
        '<div class="scroll"><table>'
        "<thead><tr><th>arm</th><th>equal-block, draw A</th>"
        "<th>equal-block, draw B</th><th>window-pooled, draw A</th>"
        "<th>against the bar</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        '<caption>Read from <span class="mono">runs/m30_g3_draw{A,B}.json.gz</span> '
        'at <span class="mono">/g3/models/&lt;arm&gt;/criteria/'
        "&#39;ensemble dispersion (area spread-skill)&#39;</span>; the ELMFIRE row "
        f'from <span class="mono">{elmfire_rel}</span> '
        'at <span class="mono">/pooled_subset/elmfire/growth_windows/'
        "band_area_dispersion_ratio</span>. Equal-block is the rule of record "
        "from G3 onward (ADR-021(4)); the window-pooled column is emitted beside "
        "it and never instead of it, and it is printed here so that nobody has to "
        "wonder which number was chosen.</caption></div>"
    )


def _registry_table(facts: Facts) -> str:
    reg = facts.registry
    rows = "".join(
        f'<tr><td><span class="mono">{_esc(k)}</span></td>'
        f'<td><span class="tag f">may not adjudicate</span></td>'
        f'<td style="text-align:left">{_esc(by)}</td></tr>'
        for k, by in reg["quarantined"]
    )
    elig = "".join(
        f'<tr><td><span class="mono">{_esc(k)}</span></td>'
        f'<td><span class="tag p">may adjudicate</span></td>'
        f'<td style="text-align:left">&mdash;</td></tr>'
        for k in reg["eligible"]
    )
    return (
        '<div class="scroll"><table>'
        "<thead><tr><th>metric</th><th>status</th><th>quarantined by</th></tr>"
        "</thead><tbody>" + elig + rows + "</tbody></table>"
        f'<caption>Read by importing <span class="mono">'
        f"wildfire_nowcast.common.null_check.registry</span> and enumerating "
        f'<span class="mono">C6_METRICS</span>, not by quoting a document. '
        f"The registry holds {reg['n_metrics']} metrics; exactly "
        f"{len(reg['eligible'])} of them may decide anything.</caption></div>"
    )


def _traceability_table(facts: Facts) -> str:
    rows = "".join(
        f'<tr><td><span class="mono">{_esc(rel)}</span></td>'
        f'<td style="text-align:left">{_esc(note)}</td></tr>'
        for _, rel, note in facts.artifacts.as_rows()
    )
    return (
        '<div class="scroll"><table><thead><tr><th>artifact</th>'
        "<th>what this page takes from it</th></tr></thead><tbody>"
        + rows
        + "</tbody></table><caption><b>"
        + EVIDENCE_DISCLOSURE
        + "</b> Every number on this page comes "
        "from one of them, or from a constant imported out of "
        '<span class="mono">wildfire_nowcast.common</span>. No tensor is '
        "opened, no checkpoint is loaded, and nothing here calls model "
        "code.</caption></div>"
    )


def render_page(facts: Facts, *, generated_utc: str | None = None) -> str:
    """Build the whole self-contained page. No I/O beyond reading the raster."""
    stamp = generated_utc or _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    REL = {name: rel for name, rel, _note in ((n, r, w) for n, r, w in facts.artifacts.as_rows())}

    per_block_html = ", ".join(
        f'<span class="num">{_f(v, 4)}</span>' for v in facts.per_block["m30_ref"].values()
    )

    ours = facts.dispersion["m30_ref"]
    abl = facts.dispersion["m30_ref__independent"]
    ell = facts.dispersion["ellipse"]
    ratio_a = ours["A"] / abl["A"]
    ratio_b = ours["B"] / abl["B"]
    short_null_a = 1.0 / ours["A"]
    short_null_b = 1.0 / ours["B"]
    short_bar = facts.bar_low / ours["A"]
    debias_g = 1.16291  # ADR-160(1): the criterion's measured reading on a
    # calibrated heterogeneous construction where the truth is 1.0.
    short_debiased = debias_g / ours["A"]

    m34c = facts.m34["controls"]
    ceil = m34c["WORST_achievability_ceiling_anywhere"]
    worst_attained = m34c["WORST_calibration_error_anywhere_incl_per_block"]
    t3 = facts.m34["table"]["3"]["ATOM"]
    persist_cal = float(t3["persistence"]["A"]["calibration_error"])
    persist_margin = float(t3["persistence"]["A"]["margin_multiple_of_bar"])
    overforecast = {
        lead: float(
            facts.m34["table"][lead]["ATOM"]["m30_ref"]["A"]["overforecast_factor_to_reach_bar"]
        )
        for lead in ("1", "2", "3")
    }

    m33a = facts.m33["check_a_3h"]
    unc = float(m33a["ATOM"]["uncertainty"])
    res_ell_atom = 100.0 * float(m33a["ATOM"]["res_ell_A"]) / unc
    res_ours_atom = 100.0 * float(m33a["ATOM"]["res_ours_A"]) / unc
    res_ell_ew10 = 100.0 * float(m33a["EW(10)"]["res_ell_A"]) / unc
    res_ours_ew10 = 100.0 * float(m33a["EW(10)"]["res_ours_A"]) / unc
    b_ew10 = facts.m33["check_b_3h"]["EW(10)"]["A"]
    mde_mult = float(b_ew10["mde_raw"]) / abs(float(b_ew10["mean"]))
    n_scored_cells = int(
        facts.reliability["A"]["pooled_growth_windows"]["m30_ref"]["reliability_summary"]["3"][
            "calibration_n_scored"
        ]
    )

    run = facts.episode["run"]
    n_all = int(run["n_confident_cells_drawA"])
    n_scored = int(run["n_confident_cells_drawA_in_scored_windows"])
    share_a = 100.0 * n_scored / n_all
    n_all_b = int(run["n_confident_cells_drawB"])
    n_scored_b = int(run["n_confident_cells_drawB_in_scored_windows"])
    share_b = 100.0 * n_scored_b / n_all_b

    elm = facts.elmfire
    cost_creek = facts.elmfire_cost["creek"]
    cost4 = facts.elmfire_cost["four"]
    _per_fire_cost = list(cost4["per_fire"].values())
    med_lo = min(float(v["s_per_member_median"]) for v in _per_fire_cost)
    med_hi = max(float(v["s_per_member_median"]) for v in _per_fire_cost)
    max_s = max(float(v["s_per_member_max"]) for v in _per_fire_cost)
    elm_pool = float(
        elm["pooled_subset"]["elmfire"]["growth_windows"]["band_area_dispersion_ratio"]
    )
    elm_windows = int(elm["pooled_subset"]["elmfire"]["growth_windows"]["n_windows"])

    # dormant_off_rate, EVERY block and EVERY arm, read out of the artifact. The
    # S17 page quoted the single cell where both baselines read 0.000 - which is
    # the WORST block of four, and citing the worst cell is the selection this
    # page spends a section warning about. The four-row table is the stronger
    # argument anyway: two constructions with nothing in common track each other
    # across the whole range, which points at the block rather than at either
    # model.
    dormant: dict[str, dict[str, float]] = {}
    for fire_id, rec in elm["per_fire"].items():
        dormant[fire_id] = {
            arm: float(rec["models"][arm]["c6_2_validity"]["off_state"]["dormant_off_rate"])
            for arm in ("persistence", "ellipse", "elmfire")
        }
    _gaps = [abs(v["ellipse"] - v["elmfire"]) for v in dormant.values()]
    dormant_max_gap = max(_gaps)
    dormant_lo = min(v["ellipse"] for v in dormant.values())
    dormant_hi = max(v["ellipse"] for v in dormant.values())
    dormant_rows = "".join(
        f'<tr><td class="mono">{_esc(fire_id)}</td>'
        f'<td class="f">{_f(v["persistence"], 3)}</td>'
        f'<td class="f">{_f(v["ellipse"], 3)}</td>'
        f'<td class="f">{_f(v["elmfire"], 3)}</td>'
        f'<td class="f">{_f(abs(v["ellipse"] - v["elmfire"]), 3)}</td></tr>'
        for fire_id, v in sorted(dormant.items())
    )

    detail = facts.disp_detail["m30_ref"]
    persist_detail = facts.disp_detail["persistence"]

    shortfalls = [
        facts.bar_low / v
        for v in ([facts.dispersion[a]["A"] for a in ARM_ORDER] + [elm_pool])
        if v > 0.0
    ]

    uri_disp = fig_dispersion(facts)
    uri_cal = fig_calibration_ceiling(facts)
    uri_bin = fig_binning_ladder(facts)
    uri_rel = fig_reliability(facts, lead=3)
    uri_run = fig_commitment_run(facts)
    uri_lat = fig_latent_ablation(facts)
    uri_elm = fig_elmfire(facts)
    uri_ep, ep_w, ep_h, ep_kib = _png_to_uri(facts.artifacts.episode_png, 1500)

    reg = facts.registry
    sc = facts.scope

    body = f"""
<div class="wrap">
<header class="masthead">
  <h1>What We Measured</h1>
  <p class="sub">Probabilistic 1&ndash;3&nbsp;h wildfire spread nowcasting &mdash; the G6
  readout for gates G3 and G5, on five held-out fires.</p>
  <p class="stamp">generated {_esc(stamp)} &middot;
  corpus fingerprint <span class="num">{_esc(sc.get("split_fingerprint", "?"))}</span> &middot;
  {_esc(sc.get("n_fires_built", "?"))} fires built,
  {_esc(sc.get("n_train_fires", "?"))} train /
  {_esc(sc.get("n_heldout_fires", "?"))} held out,
  {_esc(sc.get("n_heldout_blocks", "?"))} held-out spatial blocks &middot;
  every figure and every number traces to a file named on this page</p>
</header>

<div class="verdict">
  <div class="lede">G3 is not met. It is not close and it is not a near miss.</div>
  <p>The adjudicating conjunct of G3 is ensemble dispersion. Our reference
  kernel reads <span class="num"><b>{_f(ours["A"], 4)}</b></span> (draw A) and
  <span class="num"><b>{_f(ours["B"], 4)}</b></span> (draw B) against an
  interval of <span class="num">[{_f(facts.bar_low, 4)},
  {_f(facts.bar_high, 4)}]</span>. Seven arms were scored. Zero are inside.
  Both seed draws agree.</p>
</div>

<div class="verdict good">
  <div class="lede">One thing worked, and it is the project's central design claim.</div>
  <p>The independent-per-pixel-noise ablation collapses. Replacing the shared
  per-step latent <span class="mono">z_t</span> with independent noise cuts
  ensemble dispersion by roughly half:
  <span class="num"><b>&times;{_f(ratio_a, 4)}</b></span> on draw A and
  <span class="num"><b>&times;{_f(ratio_b, 4)}</b></span> on draw B. The architecture
  choice this project's ground-truth brief records as non-negotiable was put to
  its own test and survived it.</p>
</div>

<div class="verdict warn">
  <div class="lede">And the finding that outlives both: eight of our own
  instruments could not say the thing they existed to say.</div>
  <p>Including half of G3 itself. A repository that can name the places its own
  instruments were blind is worth more than one that reports a passing gate,
  so that enumeration is
  <a href="#blind">a section of this report</a> rather than a process
  footnote.</p>
</div>

<nav class="toc" aria-label="Contents">
  <ol>
    <li><a href="#dispersion">G3's adjudicating conjunct: dispersion</a></li>
    <li><a href="#calibration">G3's other conjunct, which exists only in our prose</a></li>
    <li><a href="#ellipse">Against a wind-advected ellipse: indistinguishable</a></li>
    <li><a href="#coverage">What the gate actually scores</a></li>
    <li><a href="#latent">The one clean positive result</a></li>
    <li><a href="#elmfire">The physics baseline fails the same criterion</a></li>
    <li><a href="#blind">Five checks that could not have failed, and three
        that could not stay quiet</a></li>
    <li><a href="#next">What would move any of this</a></li>
    <li><a href="#provenance">Provenance</a></li>
  </ol>
</nav>

<h2 id="dispersion">1 &middot; G3's adjudicating conjunct: dispersion</h2>

<figure>
  <div class="scroll"><img src="{uri_disp}" alt="Dispersion ratio for seven
  forecast arms on a log axis against the G3 bar. Every arm sits far to the
  left of the bar." /></div>
  <figcaption>Every arm is under-dispersed, by between
  <span class="num">{_f(min(shortfalls), 1)}&times;</span> and
  <span class="num">{_f(max(shortfalls), 1)}&times;</span> relative to the lower edge
  of the bar. Persistence is a degenerate ensemble and is scored UNDEFINED
  rather than zero.
  <span class="src">runs/m30_g3_draw{{A,B}}.json.gz</span></figcaption>
</figure>

{_dispersion_table(facts)}

<h3>The bar, printed from the code that enforces it</h3>

<p>The interval is <span class="num">[{_f(facts.bar_low, 10)},
{_f(facts.bar_high, 4)}]</span>, read from
<span class="mono">wildfire_nowcast.common.dispersion.BAR_INTERVAL</span> and
cross-checked against the interval the artifact stores for itself; this module
refuses to render if the two disagree. Its provenance travels with it:
<span class="mono">{_esc(facts.bar_source)}</span>.</p>

<p><b>Note the interval, because our own prose has been quoting a different
one.</b> Our own planning prose &mdash; the status board and the ground-truth
brief &mdash; says <span class="num">[0.8, 1.2]</span>. The
code says <span class="num">[{_f(facts.bar_low, 4)}, {_f(facts.bar_high, 4)}]</span>,
because it is symmetric in log space. The code is <em>stricter</em>, and it is
stricter on exactly the side we fail on. The artifact says so itself, and the
sentence is better than any paraphrase of it:</p>

<blockquote class="record">{_esc(detail)}</blockquote>

<h3>How large is the shortfall, on three defensible definitions</h3>

<div class="scroll"><table>
<thead><tr><th>definition</th><th>draw A</th><th>draw B</th><th>what it measures</th></tr></thead>
<tbody>
<tr><td>against the criterion's null of 1.0</td>
    <td class="num">{_f(short_null_a, 4)}&times;</td>
    <td class="num">{_f(short_null_b, 4)}&times;</td>
    <td style="text-align:left">how far the raw ratio is from a calibrated ensemble</td></tr>
<tr><td>against the bar's near edge, {_f(facts.bar_low, 4)}</td>
    <td class="num">{_f(short_bar, 4)}&times;</td>
    <td class="num">{_f(facts.bar_low / ours["B"], 4)}&times;</td>
    <td style="text-align:left">the smallest honest number; the distance to a pass</td></tr>
<tr class="ours"><td>against the criterion's measured reading on a calibrated
    construction, {debias_g}</td>
    <td class="num">{_f(short_debiased, 4)}&times;</td>
    <td class="num">{_f(debias_g / ours["B"], 4)}&times;</td>
    <td style="text-align:left">the shortfall after ADR-160 measured the estimator
    itself reading <b>+16% HIGH</b> where the truth is 1.0</td></tr>
</tbody></table>
<caption>The third row is the one a hostile reader should hold us to. ADR-160
put the criterion on a construction whose true dispersion is 1.0 and it read
<span class="num">{debias_g}</span> &mdash; the criterion is <b>generous</b>, not
punitive, so the shortfall is worse than the headline number rather than
better. The modelling owner refused to publish a bias-corrected
<span class="num">{_f(ours["A"], 4)}</span> as a headline figure, on the
grounds that the correction is construction-dependent; that refusal is
correct, and it is why the raw number leads this page and the corrected one is
a column here.</caption></div>

<h3>The comparison that costs us most</h3>

<div class="verdict">
  <p>The best arm in the table is the <b>wind-advected ellipse</b>, at
  <span class="num">{_f(ell["A"], 4)}</span> / <span class="num">{_f(ell["B"], 4)}</span>
  &mdash; and it is still about half the lower edge of the bar. Our learned kernel
  sits <em>below</em> it. So on the one conjunct of G3 that may adjudicate
  anything, we are beaten by the physics baseline we set out to beat, and the
  physics baseline fails too. Both halves of that sentence are true and neither
  rescues the other.</p>
</div>

<p>Nor is the failure carried by one block. Our per-block readings on draw A are
{per_block_html} &mdash;
no held-out block is inside the bar on any arm.</p>

<h3>Persistence reads 0.0, and the stored explanation for it is the wrong one</h3>

<p>The harness records persistence as <b>UNDEFINED, explicitly not a pass</b>,
which is the right verdict. It then prints this reason, which the page quotes
verbatim because a reader who sees a zero in a column of ratios will otherwise
read it as a measurement:</p>

<blockquote class="record">{_esc(persist_detail)}</blockquote>

<p><b>That sentence does not describe persistence.</b>
<span class="mono">common/dispersion.py</span> has two entry paths into
UNDEFINED and its own docstring names both &mdash;
<span class="mono">adr is None</span>, the metric's signal that its
<em>denominator</em> vanished, and <span class="mono">adr &lt;= 0</span>. Only
the first carries the meaning quoted above. Persistence arrives by the second:
<span class="mono">_ratio</span> returns <span class="mono">None</span> when the
denominator is at or below epsilon, so a stored
<span class="num">0.0</span> is a real division whose <em>numerator</em> was
zero &mdash; persistence is deterministic and has no ensemble spread at all.</p>

<p>The two zeros mean opposite things. A vanishing denominator is a forecaster
whose mean area is exactly right, and the statistic dissolves for a good
reason. A vanishing numerator is total ensemble collapse, which is the worst
dispersion there is. <b>Both are UNDEFINED and neither is a pass, so no verdict
moves</b> &mdash; but this project's own record has already split on it: one
decision record repeats the string above about persistence, and another,
written independently three days earlier, says &ldquo;persistence is 0.0000
everywhere because it is deterministic and has no spread at all.&rdquo; The
second is the correct one. Filed, not patched: that file is not this
package's to edit.</p>
"""

    body += f"""
<h2 id="calibration">2 &middot; G3's other conjunct, which exists only in our prose</h2>

<p>Everywhere this project describes G3 in words it is a two-part gate:
dispersion in band <b>and</b> a reliability curve within &plusmn;10 points. The
three documents that define it that way are planning and coordination files and
<b>none of the three is in this repository</b>, so rather than send you to them,
here is the sentence itself, quoted from the one place a cloner can read it:
<span class="mono">sim/g5.py</span> spells G3 out inline as
&ldquo;<span class="mono">dispersion ratio in [0.8,1.2] AND reliability within
+/-10 pts on held-out fires</span>&rdquo;, and says in the same comment that it
is written out there <em>because</em> the file it is tracked in is not part of
this repository. The dispersion half fails. The reliability half passes. Two
things are true about that pass, and both of them subtract from it.</p>

<h3>It could not have failed</h3>

<p><span class="mono">calibration_error</span> is a weighted mean absolute
deviation, so it is bounded above by
<span class="mono">mean(forecast) + base&nbsp;rate</span> identically. Across
every arm, block, lead, binning and both seed draws, the largest that ceiling
ever gets is <span class="num"><b>{_f(float(ceil["ceiling"]), 6)}</b></span>
(at <span class="mono">{_esc(ceil["where"])}</span>) &mdash;
<span class="num"><b>{_f(float(ceil["bar_over_ceiling"]), 2)}&times;</b></span> below
the <span class="num">{_f(float(facts.m34["bar"]), 2)}</span> bar. The largest value anything
actually attains is <span class="num">{_f(float(worst_attained["calibration_error"]), 6)}</span>,
<span class="num">{_f(float(worst_attained["bar_over_err"]), 2)}&times;</span> below
it. There are <b>{len(facts.m34["cells_failing_the_bar"])}</b> failing cells in
the whole grid.</p>

<figure>
  <div class="scroll"><img src="{uri_cal}" alt="Attained calibration error and
  its achievability ceiling for six arms, both far below the 0.10 bar." /></div>
  <figcaption>The grey stems are the gap between what each arm attained and the
  highest value the statistic could take for that arm. The bar is above every
  stem. <span class="src">{REL["m34_summary"]}</span></figcaption>
</figure>

<p>Stated as a requirement rather than an observation: a forecaster would have
to <b>over-forecast mean burn probability by
<span class="num">{_f(overforecast["3"], 1)}&times;</span></b> at 3&nbsp;h
(<span class="num">{_f(overforecast["2"], 1)}&times;</span> at 2&nbsp;h,
<span class="num">{_f(overforecast["1"], 1)}&times;</span> at 1&nbsp;h) before this
conjunct could <em>in principle</em> read {_f(float(facts.m34["bar"]), 2)}. And
<span class="mono">persistence</span>, which forecasts identically zero and
therefore predicts no fire spread at all, scores
<span class="num">{persist_cal}</span> &mdash; the base rate to nine decimals &mdash; and
passes by <span class="num"><b>{_f(persist_margin, 1)}&times;</b></span>.</p>

<p>It is not that our arms pass. It is that <b>the bar sits outside the range
the statistic can take on this data</b>, and no forecaster predicting anything
near the right burned area can fail it. <b>The number itself has no
derivation anywhere in this repository.</b> It enters as a round figure in a
plan-time bullet, in a planning document that is not part of this repository,
and everything a cloner can check is downstream of it: the artifact stores the
bar's own provenance string as
&ldquo;<span class="mono">{_esc(facts.cal_bar_source)}</span>&rdquo;, written by
<span class="mono">eval/baseline_run.py</span>, and
<span class="mono">common/calibration.py</span> records
that the linear form was chosen because it &ldquo;is the only form in the units
of G3's own &plusmn;10 pts bar&rdquo;. So the tracked record shows the metric
being fitted to the number, and contains no step in which the number was
derived. <b>The metric was fitted to the bar; the bar was never
re-derived.</b></p>

<h3>And the contract had already retired it</h3>

<p>Independently of any of the above, this conjunct is not something our own
harness will score a verdict on. Calling
<span class="mono">assert_may_adjudicate("calibration_error")</span> raises:</p>

<blockquote class="record">{_esc(reg["raised"])}</blockquote>

{_registry_table(facts)}

<div class="verdict">
  <div class="lede">The honest description of G3.</div>
  <p><b>G3 has been a one-conjunct gate in code while being reported as a
  two-conjunct gate in prose, and the conjunct that exists only in prose is the
  one that passes.</b> This changes no verdict &mdash;
  <span class="mono">area_dispersion_ratio</span> is one of the three
  gate-eligible metrics and it fails &mdash; but it changes what a reader should take
  from the sentence &ldquo;the calibration half passes&rdquo;. Nothing.</p>
</div>

<p class="note">The same applies upward. The ground-truth brief's <em>headline
metrics</em> line names reliability diagrams, Brier, arrival-time CRPS, ensemble dispersion
ratio and best-member mode-capture IoU. Against the registry above,
reliability, Brier and arrival CRPS have <b>no gate-eligible member at all</b>,
and dispersion and best-member IoU survive only under renamed variants while
their originally-named forms are quarantined. Three of the five headline
metrics of this project may not decide anything, and that is a fact about our
documents rather than about the measurements.</p>

<h2 id="ellipse">3 &middot; Against a wind-advected ellipse: indistinguishable</h2>

<p>This section retracts a claim this project made two days ago, in the
direction that is <em>less</em> flattering to the retraction than to the
original.</p>

<p>Measured on Murphy resolution &mdash; the share of outcome variance a forecaster
resolves &mdash; at 3&nbsp;h on five held-out fires:</p>

<div class="scroll"><table>
<thead><tr><th>binning</th><th>ellipse</th><th>our kernel</th>
<th>ratio (ell / ours)</th></tr></thead>
<tbody>
<tr><td>EW(10) &mdash; a ten-bin equal-width histogram</td>
    <td class="num">{_f(res_ell_ew10, 3)}%</td>
    <td class="num">{_f(res_ours_ew10, 3)}%</td>
    <td class="num">{_f(float(m33a["EW(10)"]["ratio_A"]), 4)}</td></tr>
<tr class="ours"><td>ATOM &mdash; the finest binning the forecasts support</td>
    <td class="num">{_f(res_ell_atom, 3)}%</td>
    <td class="num">{_f(res_ours_atom, 3)}%</td>
    <td class="num">{_f(float(m33a["ATOM"]["ratio_A"]), 4)}</td></tr>
</tbody></table>
<caption>Draw A, 3&nbsp;h, from <span class="mono">{REL["m33_summary"]}</span>
at <span class="mono">/check_a_3h</span>. The denominator
(<span class="mono">uncertainty</span> = <span class="num">{unc}</span>) is
binning-invariant &mdash; identical at all seventeen cells, spread exactly
0.0.</caption></div>

<figure>
  <div class="scroll"><img src="{uri_bin}" alt="Resolution share against
  binning for the ellipse and our kernel; the two curves cross once." /></div>
  <figcaption>The forecasts are identical at every point on this axis. Only the
  bin edges move. Twelve of seventeen cells reverse the ordering; at 2&nbsp;h
  the four finest cells disagree across seed draws, so ADR-142(e) bars them
  outright (EW(25), EW(50), EW(100) and ATOM). The reversal count is at 3 h;
  over all three leads it is 36 of 51.
  <span class="src">{REL["m33_summary"]}</span></figcaption>
</figure>

<p><b>Both readings are correct measurements of the same forecasts.</b> The
mechanism is ours and it is not a nuisance: our ensemble has 24 members, so our
probability field is atomic at multiples of 1/24, and nearly all of our
information lives in the lowest atoms &mdash; distinguishing <em>one member out of
twenty-four burned this cell</em> from <em>none did</em>. At 1&nbsp;h,
EW(10)'s first bin holds 99.9% of all cells and traps <b>92.0%</b> of our own
resolution against the ellipse's <b>32.3%</b>. A coarse first bin deletes our
signal and preserves theirs. Fine binning is not manufacturing our signal;
coarse binning was destroying it.</p>

<div class="verdict warn">
  <div class="lede">The sentence of record, and it is deliberately not a win.</div>
  <p>On resolution, our learned kernel and a wind-advected ellipse are
  <b>indistinguishable</b> on five held-out fires &mdash;
  <span class="num">{_f(res_ours_atom, 3)}%</span> against
  <span class="num">{_f(res_ell_atom, 3)}%</span> of outcome variance &mdash; and the
  test is far too weak to separate them either way.
  <span class="num">0 of 51</span> cells are significant on either draw, at
  3&nbsp;h EW(10) the paired difference is
  <span class="num">{float(b_ew10["mean"]):+.4e}</span> with
  p<sub>2</sub>&nbsp;=&nbsp;<span class="num">{_f(float(b_ew10["p_two_sided"]), 4)}</span>,
  and the minimum detectable effect at n&nbsp;=&nbsp;5 is
  <span class="num">{_f(mde_mult, 1)}&times;</span> the observed effect. Signs are
  <span class="mono">{_esc(b_ew10["signs"])}</span>: one block reverses.</p>
</div>

<p class="note"><b>Do not read this as a result in our favour.</b> The
modelling owner filed a proposal refusing to claim we beat the ellipse and it was accepted. At
ATOM the ratio (ellipse &divide; ours) is
<span class="num">{_f(float(m33a["ATOM"]["ratio_A"]), 4)}</span> on draw A and
<span class="num">{_f(float(m33a["ATOM"]["ratio_B"]), 4)}</span> on draw B &mdash;
inside the pre-registered &plusmn;10% indistinguishable band on both. What was
retracted is the claim that we are
<em>half as informative</em> as an ellipse; what replaces it is not
&ldquo;better&rdquo;, it is &ldquo;this experiment cannot tell.&rdquo;</p>
"""

    body += f"""
<h2 id="coverage">4 &middot; What the gate actually scores</h2>

<p>The reliability curve above is the shape of our forecast where the gate can
see it. Rendering the same model outside the scored windows showed that the
gate sees almost none of what the model does.</p>

<figure>
  <div class="scroll"><img src="{uri_rel}" alt="Reliability curves for our
  kernel and the ellipse on both seed draws, with bin occupancy beneath." /></div>
  <figcaption>Both forecasters sit below the diagonal, so both are
  over-confident &mdash; but the ellipse's curve <b>rises</b> monotonically to the
  end and ours <b>peaks near p &asymp; 0.23 and falls to zero</b>. That distinction
  decides the next experiment: an under-confident but correctly ordered
  forecast is repaired by recalibration and a mis-ordered one is not. Drawn on
  both draws, which ADR-142(e) requires and the draw-A-only version of this
  figure could not do.
  <span class="src">runs/m30_reliability_draw{{A,B}}.json.gz</span></figcaption>
</figure>

<p>The right-hand end of that curve is where a model says something. Ours puts
<b>162</b> cells at p&nbsp;&ge;&nbsp;0.5 across all five held-out fires at
3&nbsp;h, and <b>zero</b> of them burned. With one-sided 95% Clopper&ndash;Pearson
that bounds the true rate at 1.83%, against the ellipse's measured 12.24%. But
the number that decides how much weight the curve can carry is not 162:</p>

<figure>
  <div class="scroll"><img src="{uri_run}" alt="Confident-cell count per hour
  over 91 hours of the Creek fire, with the ten scored hours shaded." /></div>
  <figcaption>Over <span class="num">{run["n_windows"]}</span> consecutive hours
  of <span class="mono">2020_creek</span> the model commits
  <span class="num">{n_all:,}</span> cells at p&nbsp;&ge;&nbsp;0.5 (draw A).
  <span class="num">{n_scored}</span> of them &mdash;
  <span class="num">{_f(share_a, 1)}%</span> &mdash; fall in the
  <span class="num">{run["n_windows_scored"]}</span> hours the gate scores.
  Draw B: <span class="num">{n_scored_b}</span> of
  <span class="num">{n_all_b:,}</span>, <span class="num">{_f(share_b, 1)}%</span>.
  Zero burned in either. <span class="src">{REL["creek_episode"]}</span>
  </figcaption>
</figure>

<div class="verdict">
  <div class="lede">The gate scores {_f(share_a, 1)}% of what the model does
  here, and the high-confidence end of the curve descends from one label
  event.</div>
  <p>The commitment is a <b>19-hour run</b>, peaking at 206 cells at
  max&nbsp;p&nbsp;=&nbsp;1.000, in air windier and drier than any scored hour
  (26.51&nbsp;m&nbsp;s&#8315;&sup1;, RH&nbsp;0.19%). The three published windows are its
  <b>falling limb</b>. And the Creek perimeter is <b>bitwise constant at 1,579
  cells</b> from t0&nbsp;&le;&nbsp;1215 through t0&nbsp;=&nbsp;1236; a single
  <b>2-cell increment</b> at 2020-10-26T19:00Z makes exactly three overlapping
  windows eligible. &ldquo;Truth grew 2 cells in each of three hours&rdquo; is
  the same two cells seen three times. Had that increment landed an hour later,
  the episode would be unscored.</p>
</div>

<p class="note"><b>What this is not.</b> A zero-growth hour is not evidence
that nothing burned. 51&ndash;91% of GOFER hours carry bitwise zero growth because
GOES cannot see new front at night or under cloud, and t0&nbsp;=&nbsp;1221&ndash;1233
is 21:00&ndash;09:00 local. In the unscored hours the model committed and the
<em>label</em> did not move; whether the <em>fire</em> moved is not knowable
from the tensor. This is a statement about what the scored stratum can show,
not a claim that the model was right out there.</p>

<p>And the high-probability end of the reliability curve has a narrower basis
still. With <span class="mono">2020_creek</span> removed, our cells at
p&nbsp;&ge;&nbsp;0.5 across the entire four-block remainder are <b>0, 0 and 1</b>
at 1, 2 and 3&nbsp;h on draw A (0, 0, 3 on draw B), none of which burned. The
model essentially never commits above one-half outside one fire, so the
mis-ordering finding cannot be evaluated outside it. Excluding a held-out block
is forbidden as a gate change and the five-block curve remains the number of
record; this is the statement of its basis, not a re-scoring.</p>

<figure>
  <div class="scroll"><img src="{uri_ep}" width="{ep_w}" height="{ep_h}"
  alt="Five rows of map panels showing the Creek episode and two contrast
  hours, each with a histogram of ensemble probability." /></div>
  <figcaption>The episode, rendered. Rows 1&ndash;3 are the three scored hours: a
  red-outlined belt of 92, 53 and 13 confident cells lines the southern and
  eastern flank; the green ring is where the label moved, two cells. Rows 4&ndash;5
  are the contrast at the same colour scale &mdash; a 1,581-cell perimeter, two cells
  larger than the episode's, at
  21.51&nbsp;m&nbsp;s&#8315;&sup1; with no red at all, and a drier hour at
  17.59&nbsp;m&nbsp;s&#8315;&sup1; with none either. <b>The panel that makes the contrast
  legible is not a map</b>: the per-row histogram separates a model that was
  <em>uncertain</em> from one that was <em>silent</em>, and the maps cannot.
  <span class="src">{REL["episode_png"]}, downscaled to
  {ep_w}&times;{ep_h} and inlined ({ep_kib}&nbsp;KiB)</span></figcaption>
</figure>

<h2 id="latent">5 &middot; The one clean positive result</h2>

<p>This project's ground-truth brief records that pixels are conditionally independent
Bernoulli <em>only</em> given a shared per-step latent
<span class="mono">z_t</span>, and that independent-per-pixel-noise models are
known-broken. That is a claim about our own architecture, it was testable, and
the ablation arm was built and run through the same C5
<span class="mono">predict()</span> path and scored by the same C6 criterion as
the model itself.</p>

<figure>
  <div class="scroll"><img src="{uri_lat}" alt="Dispersion ratio for the
  independent-noise ablation and the shared-latent model, on both draws." /></div>
  <figcaption>Both draws, both directions the same, magnitudes within 6% of
  each other. <span class="src">runs/m30_g3_draw{{A,B}}.json.gz</span></figcaption>
</figure>

<div class="verdict good">
  <p>The shared per-step latent buys <b>&times;{_f(ratio_a, 4)}</b> (draw A) and
  <b>&times;{_f(ratio_b, 4)}</b> (draw B) of ensemble dispersion over independent
  per-pixel noise. ADR-142(e) makes agreeing verdicts across both draws the
  reportable unit, and these agree in direction and in magnitude, so the
  reportable statement is <b>2.15&times;&ndash;2.27&times;</b>. The design claim survived its own
  test.</p>
</div>

<p class="note"><b>The honest limit on that sentence.</b> It says the latent
roughly doubles spread. It does not say the resulting spread is adequate &mdash; the
ablation reads <span class="num">{_f(abl["A"], 4)}</span> and the model reads
<span class="num">{_f(ours["A"], 4)}</span>, and <em>both</em> fail the same
bar. Doubling a number that is
<span class="num">{_f(1.0 / abl["A"], 1)}&times;</span> below the criterion's null
gives one that is <span class="num">{_f(short_null_a, 1)}&times;</span> below it.
This is a mechanism confirmed, not a gate passed.</p>
"""

    body += f"""
<h2 id="elmfire">6 &middot; The physics baseline fails the same criterion</h2>

<p>ELMFIRE 2025.1002 with default Rothermel parameters is the non-negotiable
physics baseline for this project. It ran through the same C6 scoring path,
wrapped so that its Monte Carlo output arrives in the C5
<span class="mono">predict()</span> signature. <b>74 held-out windows &times; 4
members, 0 failed return codes, 2.25 CPU-hours</b>, and none of the three cost
levers (<span class="mono">refine</span>,
<span class="mono">SIMULATION_DT</span>,
<span class="mono">reach_cells_per_hour</span>) was pulled.</p>

<figure>
  <div class="scroll"><img src="{uri_elm}" alt="Dispersion ratio per fire for
  persistence, the ellipse and ELMFIRE, against the G3 band." /></div>
  <figcaption>ELMFIRE fails the dispersion criterion on 4 of 4 blocks, pooled
  <span class="num">{_f(elm_pool, 4)}</span> over
  <span class="num">{elm_windows}</span> growth windows.
  <span class="src">{REL["g5_four_blocks"]}</span></figcaption>
</figure>

<div class="verdict warn">
  <div class="lede">Report this as a fact about the criterion and the task. It
  does NOT rescue our number.</div>
  <p>The physics baselines have little OFF state either, and the argument is
  the <em>whole</em> table rather than its best cell.
  <span class="mono">dormant_off_rate</span> runs from
  <span class="num">{_f(dormant_hi, 3)}</span> down to
  <span class="num">{_f(dormant_lo, 3)}</span> across the four blocks, and a
  kinematic wind-advected ellipse and a Rothermel physics simulator &mdash; two
  constructions with nothing in common &mdash; agree in every one of them to
  within <span class="num">{_f(dormant_max_gap, 3)}</span>:</p>
  <div class="scroll"><table>
  <thead><tr><th>block</th><th>persistence</th><th>ellipse</th><th>ELMFIRE</th>
  <th>|ellipse &minus; ELMFIRE|</th></tr></thead>
  <tbody>{dormant_rows}</tbody>
  <caption>Persistence is the scale: a deterministic do-nothing forecast is
  dormant in <b>every</b> scored window of <b>every</b> block, so the column is
  anchored at 1.000 and the ellipse/ELMFIRE agreement is not an artefact of a
  saturated statistic. <span class="src">{REL["g5_four_blocks"]}</span></caption>
  </table></div>
  <p>Two unrelated models moving together across blocks is evidence that the
  off-state rate is a property of the <b>block</b> &mdash; of how much of the scored
  stratum is genuinely quiescent &mdash; rather than a defect either model owns.
  ELMFIRE's over-prediction has the same shape as our kernel's, with 74.6% of
  the excess bought in windows where truth grew nothing. A defect shared by
  Rothermel-default ELMFIRE and by a wind-advected ellipse is a candidate
  property of the problem or of the scoring stratum. It is <b>not</b> a licence
  to read our own {_f(ours["A"], 4)} as excusable, and no prior refutation is
  retracted by it.</p>
  <p class="note"><b>A question this table raises, not a finding.</b> borel is
  both the block where the ellipse's dispersion passes
  (<span class="num">1.1693</span> raw) and the block where neither baseline
  ever goes dormant. A forecaster that always predicts some spread always has
  some spread to measure, so those two facts may be one fact &mdash; or may not.
  <b>Nobody has tested it, n&nbsp;=&nbsp;4 blocks</b>, and this section already
  says any per-block claim here is close to anecdotal.</p>
</div>

<h3>Every mapping compromise, because an unfair baseline invalidates G5 in
either direction</h3>

<ol class="findings">
  <li><span class="h">Four blocks of five, and the fifth is the biggest.</span>
  <span class="mono">2020_creek</span> is excluded on cost, and the cost has
  two different epistemic statuses that the artifact keeps apart. <b>Measured:</b>
  480.2&nbsp;s CPU for <em>one member</em> at t0&nbsp;=&nbsp;200, and
  &gt;1060&nbsp;s at t0&nbsp;=&nbsp;800 &mdash; a <em>killed</em> run, so a lower
  bound rather than a measurement &mdash; against a per-member median of
  <span class="num">{_f(med_lo, 1)}</span>&ndash;<span
  class="num">{_f(med_hi, 1)}</span>&nbsp;s and a maximum of
  <span class="num">{_f(max_s, 1)}</span>&nbsp;s
  on the three blocks that ran. <b>Projected:</b>
  <span class="num">{cost_creek["projected_total_cpu_hours"]}</span> CPU-hours
  for Creek's {cost_creek["setting"]["n_creek_windows"]} windows against
  <span class="num">{cost4["total_cpu_hours"]}</span> measured for the other
  four combined &mdash; and the artifact labels its own status rather than leaving a
  reader to assume: &ldquo;{_esc(cost_creek["status"])}&rdquo;
  <span class="mono">run_baselines</span> takes no fire argument, so the honest
  options were all five blocks at roughly
  {cost_creek["projected_total_cpu_days"]}&nbsp;CPU-days or four blocks.</li>

  <li><span class="h">4 members against our 24, and stride 16 against our
  1.</span> The ELMFIRE column is scored at
  <span class="mono">stride&nbsp;{elm["stride"]} / {elm["n_members"]} members /
  seed&nbsp;{elm["seed"]}</span>; our G3 column at
  <span class="mono">stride&nbsp;1 / 24&nbsp;members</span>. <b>This is the
  compromise that matters most and it is not a small one.</b> A dispersion
  ratio estimated from four members is not the same estimator as one estimated
  from twenty-four; the two ELMFIRE and model dispersion numbers on this page
  are <em>not</em> a like-for-like comparison and must not be subtracted from
  one another. What the ELMFIRE column supports is the qualitative statement
  &ldquo;a default-parameter physics model is also far under the bar on its own
  blocks&rdquo;, and nothing quantitative against our arm.</li>

  <li><span class="h">19 growth windows, and czu contributes two.</span>
  {" &middot; ".join(f"{k}: {int(v['n_growth_windows'])}" for k, v in elm["per_fire"].items())}.
  Any per-block claim here is nearly anecdotal.</li>

  <li><span class="h">The table was not produced by one
  <span class="mono">run_baselines</span> call, and it says so in its own
  artifact.</span> It uses that module's own
  <span class="mono">_score</span>, <span class="mono">_headline</span> and
  <span class="mono">metrics.aggregate</span> on the same windows. The
  deviation is declared in a key named
  <span class="mono">NOT_run_baselines</span> rather than left for a reader to
  discover.</li>

  <li><span class="h">What was controlled rather than asserted.</span>
  Persistence and ellipse columns are <b>bit-identical on 8/8</b> against a
  real <span class="mono">run_baselines</span> payload at the same
  stride/members/seed; an empty ELMFIRE cache is <b>refused</b> inside
  <span class="mono">run_baselines</span> rather than scored as silence; cached
  replay is bitwise identical to an inline run; <b>74/74</b> store keys
  re-derive from <span class="mono">(fire, t0, seed)</span>.</li>

  <li><span class="h">A bar discrepancy that is immaterial here and is stated
  anyway.</span> The ELMFIRE table was read against
  <span class="num">[0.8, 1.2]</span>, the prose bar; the code's geometric bar
  is <span class="num">[{_f(facts.bar_low, 4)}, {_f(facts.bar_high, 4)}]</span>
  and is stricter. Every ELMFIRE cell fails both, so nothing turns on it.</li>

  <li><span class="h">The one passing cell in that table is not robust.</span>
  On borel the ellipse reads <span class="num">1.1693</span> raw &mdash; inside
  [0.8, 1.2] &mdash; and <span class="num">1.5079</span> under the diagnostic
  debiased estimator, outside it. The pre-declared statistic is the raw one
  (<span class="mono">metrics.py:1271</span>,
  <span class="mono">baseline_run.py:178</span> and
  <span class="mono">:1358</span> all name it as the G3 criterion at the point
  of definition, which is what let a suspicion of cherry-picking be settled by
  one grep instead of an argument). Do not quote &ldquo;the ellipse clears G3
  on borel&rdquo; without this sentence attached.</li>
</ol>

<p class="note"><b>G5 is not adjudicated.</b> Four blocks of five and nineteen
growth windows is not a gate. The comparison is published as a measurement and
the gate is left open.</p>
"""

    body += f"""
<h2 id="blind">7 &middot; Five checks that could not have failed, and three that could
not stay quiet</h2>

<p>This is the finding that outlives the numbers above, and it belongs in the
results rather than in a process appendix. Over a single working day this
project found <b>eight</b> of its own instruments unable to say the thing they
existed to say &mdash; in a headline gate, in continuous integration, in a
statistical method, and in the maintainer's own watchdog. They fall into two
kinds and the distinction is load-bearing, because lumping them together
invites the objection that we are counting one defect twice.</p>

<h3>Strict: a check that ran, passed, and could not have failed</h3>

<ol class="findings">
  <li><span class="h">Half of G3.</span>
  <span class="mono">calibration_error &le; mean(p) + base&nbsp;rate</span>
  identically; the ceiling never exceeds
  <span class="num">{_f(float(ceil["ceiling"]), 6)}</span> anywhere in the
  corpus, <span class="num">{_f(float(ceil["bar_over_ceiling"]), 2)}&times;</span>
  below the bar. A forecaster would need to over-forecast by
  <span class="num">{_f(overforecast["3"], 1)}&times;</span> before the conjunct could
  in principle read {_f(float(facts.m34["bar"]), 2)}, and a forecaster that predicts nothing
  passes by <span class="num">{_f(persist_margin, 1)}&times;</span>. <b>This is the
  strongest instance and it is inside our own headline gate.</b>
  <span class="tag">ADR-170(2)</span></li>

  <li><span class="h">A significance test with a floor above its own
  alpha.</span> The exact sign-flip test at n&nbsp;=&nbsp;5 has a minimum
  attainable p of 2/32&nbsp;=&nbsp;0.0625, so it <b>could not have reached
  0.05 whatever the data did</b>. Found by the modelling owner inside its own
  method, in the same document where that method was used.
  <span class="tag">ADR-169(4)</span></li>

  <li><span class="h">&ldquo;The first cross-platform datum.&rdquo;</span> A
  mutation-pin check offered as evidence that the pin resolves identically on
  Linux. The verdict function is <b>pure</b> and its only input is source text
  parsed with <span class="mono">ast</span>: same commit, same bytes, same
  keys. It could not have failed on another platform for a platform reason. It
  is a restatement of purity wearing the clothes of a measurement, and the half
  that <em>is</em> platform-sensitive &mdash; does a mutant survive &mdash; has never been
  observed. <span class="tag">ADR-164(1)</span></li>

  <li><span class="h">A blindness guard blind to the likeliest blindness.</span>
  A watchdog branch whose whole job is to catch &ldquo;a probe returned
  nothing&rdquo; was written as
  <span class="mono">[ -z &quot;$free&quot; ]</span>. The planted input was a
  single space &mdash; non-empty &mdash; so the guard passed and the arithmetic after it
  errored to a stream where no notification lives. <b>A watchdog that goes
  quiet when its own probe breaks is worse than no watchdog</b>, and the guard
  for exactly that could not detect the most likely shape of it. Found only by
  planting it. <span class="tag">ADR-166(2)</span></li>

  <li><span class="h">A salvage step that salvaged nothing and reported
  success.</span> A CI step exists to preserve the sweep result of a run that
  fails. On the one run in this repository's history where that was live, it
  concluded <b>success having uploaded zero bytes</b>, because
  <span class="mono">if-no-files-found</span> was
  <span class="mono">warn</span>. Now <span class="mono">error</span>.
  <span class="tag">ADR-168(3)</span></li>
</ol>

<h3>Mirror: a check that could not stay quiet, which breaks the same link from
the other side</h3>

<p>These are not the same defect and must not be counted as it. A check that
never speaks is ignored because it never speaks; one that always speaks is
ignored because it always does. Both destroy the link between a signal and the
thing it is about.</p>

<ol class="findings" start="6">
  <li><span class="h">An alarm about something we did not cause and may not
  fix.</span> A disk watchdog screamed <span class="mono">DISK FALLING: 63Gi,
  down 21Gi</span> seven times. The cause was two staged operating-system
  update snapshots. Our entire footprint was 3,572&nbsp;MB, zero leaked
  workspaces, no process alive. <span class="tag">ADR-166(1)</span></li>

  <li><span class="h">A count that cannot tell a workspace IN USE from one
  LEAKED.</span> Its replacement's first real event was a false positive: two
  worktrees, one of which was another lead doing exactly what the rules tell a
  lead to do. The blindness here is two-sided in principle &mdash; a count is blind
  to what it does not name, which is a ruling this project had already made
  once &mdash; and what was <em>observed</em> was the alarm side, which is why it sits
  in this half rather than the one above.
  <span class="tag">ADR-166(3)</span></li>

  <li><span class="h">An edge detector that was a level detector in a
  costume.</span> It compared the rendered message, and the message carried a
  live process count, so <span class="mono">6 live</span> and
  <span class="mono">5 live</span> read as a state change and it re-fired three
  times for one unchanged state. An edge detector whose signature contains a
  volatile number is a level detector with extra steps. Fixed by splitting a
  state key from the reader-facing text and bucketing continuous quantities.
  <span class="tag">recorded under ADR-169 &sect;5</span></li>
</ol>

<div class="verdict warn">
  <div class="lede">Who these belong to, because a catalogue of other people's
  blind spots is not evidence of anything.</div>
  <p><b>Four of the eight are in one instrument the maintainer wrote</b> &mdash;
  items 4, 6, 7 and 8, all in a disk watchdog built to police another lead's
  spending &mdash; and two more (items 3 and 5) were found by the maintainer in other
  leads' artifacts. Two (items 1 and 2) were found by the modelling owner
  inside its own gate and its own method. <b>Not one of the eight was found by an outside
  reviewer.</b> They were found because someone planted an input and watched
  whether the check fired, which is the only procedure on this list that
  works.</p>
</div>

<p class="note"><b>One number we cannot reconcile, and we are printing it
rather than picking a side.</b> Two of the decision records carry a running
tally of the day's instances &mdash; one calls itself &ldquo;the fifth instance
today&rdquo; and the next &ldquo;sixth instance today&rdquo; &mdash; which cannot be
reconciled with an enumeration that makes the first of them item&nbsp;3 of
eight. The tally was never itself enumerated. That is a small, exact instance
of the same class one level up: <b>a count that nobody checked, kept inside the
document that names checks nobody checked.</b></p>

<p class="note"><b>A ninth, found while rendering this page, and of a third
kind that is not counted in the eight.</b> The dispersion criterion reports
persistence as UNDEFINED and attaches a reason it did not observe &mdash; see
<a href="#dispersion">&sect;1</a>. Not a check that could not fail, and not one
that could not stay quiet: <b>a check that reached the right verdict and
published the wrong reason for it</b>. That is arguably worse than either,
because a verdict is re-derivable and an explanation is what a reader carries
away. It was carried away: it is repeated in a decision record.</p>

<h3>What the class costs, and why we think naming it is worth more than a pass</h3>

<p>Every one of the eight ran green or silent for as long as it existed. Four
of them were written specifically to guard against a defect this project had
already named in writing. The general remedy that has actually worked here is
narrow and mechanical, and none of it is insight:</p>

<ul class="plain">
  <li><b>Plant the failure and watch it fire.</b> Reading the code found none
  of the eight. Planting found five directly.</li>
  <li><b>Print every bar-shaped criterion beside the range its statistic can
  attain</b> &mdash; for <span class="mono">calibration_error</span> that is
  <span class="mono">[0, mean(p) + base&nbsp;rate]</span>. One line, and it
  turns &ldquo;a check that could not have failed&rdquo; from something that
  took three rulings and a dedicated experiment into something mechanically
  detectable.</li>
  <li><b>A plant whose value the world can change is not a plant.</b> One
  observation in this list lied because the planted quantity was a live count
  that moved between two runs.</li>
  <li><b>Purge caches between the halves of a plant/restore sequence</b>, and
  never let a control read its own plant.</li>
</ul>
"""

    body += f"""
<h2 id="next">8 &middot; What would move any of this</h2>

<p>Stated as experiments with a stake, not as a plan. Each is written so that
it can come back negative.</p>

<ol class="findings">
  <li><span class="h">The dispersion shortfall is a factor-of-{_f(short_bar, 1)}
  problem at its most charitable, not a tuning problem.</span> The gap between
  {_f(ours["A"], 4)} and the bar's near edge {_f(facts.bar_low, 4)} is
  <span class="num">{_f(short_bar, 2)}&times;</span>; the latent &mdash; the only lever
  measured to move this number &mdash; buys
  <span class="num">{_f(ratio_a, 2)}&times;</span>. Closing the rest would take
  between one and two more levers of that size, and we do not have a second one
  identified. <b>The honest prior is that this is architectural.</b></li>

  <li><span class="h">Mis-ordering is a different defect from
  under-confidence, and it decides the next model experiment.</span> Above
  p&nbsp;&asymp;&nbsp;0.25 our stated confidence is anti-correlated with the outcome
  on <span class="num">{n_scored_cells:,}</span> scored cells, and the ablation
  arm shows the same inversion, so it is
  <em>not</em> a property of the shared latent. The two cheap decompositions
  that would separate the readings are already available inside the artifact:
  whether the falling limb is one block or five
  (<span class="mono">per_fire_growth_windows</span> carries the same bins per
  fire), and whether the p&nbsp;&ge;&nbsp;0.5 cells are a particular geometry &mdash;
  interior re-burn, barrier edge, long-range spot &mdash; rather than a random
  high-confidence subset. <b>If the answer is &ldquo;the top bins are the spot
  component firing in the wrong places&rdquo;, that is a far sharper finding
  than a dispersion ratio.</b> Attributing it is a modelling question and is
  not done here.</li>

  <li><span class="h">The scored stratum is the binding constraint on every
  statement about commitment.</span> Ten scored windows out of ninety-one, and
  one label increment behind three of them. Conditioning the stratum on the
  outcome is what <span class="mono">growth_only</span> warns about in its own
  docstring, so the stratum should <em>not</em> change to fix this. What can
  change is that the reliability page reports its own coverage of the
  commitment population, so that no reader takes &ldquo;the model does this in
  three hours of one fire&rdquo; for a statement about how often the model does
  it.</li>

  <li><span class="h">A fair ELMFIRE comparison costs about eight CPU-days, and
  is worth costing before it is run.</span> The present column is 4 members at
  stride 16 on four blocks. Bringing it onto our stride-1 / 24-member footing
  is the only way the two dispersion numbers on this page become subtractable,
  and it should be decided as a spend rather than drifted into.</li>

  <li><span class="h">Two documents should be corrected, and neither is ours to
  edit.</span> G3's one-line definition names a conjunct the contract
  disqualified, in the three documents this project reads most often, none of
  which is in the repository; and the ground-truth brief's headline-metric line
  names three metrics with no gate-eligible member. Both
  are filed and both are the maintainer's to take. A bar re-derived <em>after</em>
  seeing every arm clear it by 13&times;&ndash;146&times; would be fitted to the result, so the
  modelling owner declined to propose a replacement, and that refusal is the right
  one.</li>
</ol>

<h2 id="provenance">9 &middot; Provenance</h2>

<p>Everything above is a rendering. This page opens no tensor, loads no
checkpoint, and calls no model code; it consumes the C1 store only through
artifacts other tasks wrote, models only through the C5
<span class="mono">predict()</span> results those artifacts recorded, and
metrics only through C6's own output. The criterion and the metric registry are
imported from <span class="mono">wildfire_nowcast.common</span> so that the bar
and the eligibility list are read from the code that enforces them rather than
retyped from prose &mdash; which matters here, because the prose and the code have
disagreed about both.</p>

{_traceability_table(facts)}

<h3>Reading rules that constrain what is quoted above</h3>

<ul class="plain">
  <li><b>Both draws, or no claim.</b> ADR-142(e) makes agreeing verdicts across
  the two seed draws the reportable unit. Every headline number on this page is
  given on both draws, except the ELMFIRE column, which exists on one
  configuration only and is labelled as such.</li>
  <li><b>Equal-block, never window-pooled.</b> Equal-block over the five
  held-out spatial blocks is the pooling rule of record from G3 onward; the
  window-pooled figure is printed beside it and never instead of it.</li>
  <li><b>Quarantined metrics are reported and may not decide.</b> Brier,
  arrival CRPS, reliability and <span class="mono">calibration_error</span> all
  appear on this page and none of them passes, fails or voids anything.</li>
  <li><b>Nothing here is tuned, and nothing here re-scores.</b> Where a number
  on this page was independently recomputed against the artifact that
  originally reported it, it reproduced with max&nbsp;|&Delta;|&nbsp;=&nbsp;0.0.</li>
</ul>

<footer class="colophon">
  <p><b>The shortest true summary of this project.</b> We built a probabilistic
  fire-spread nowcaster, put it on five held-out fires against persistence, a
  wind-advected ellipse and Rothermel-default ELMFIRE, and its ensembles are
  <span class="num">{_f(short_null_a, 1)}&times;</span> too narrow against the
  criterion's null of 1.0 &mdash;
  <span class="num">{_f(short_bar, 1)}&times;</span> against the near edge of the bar
  and <span class="num">{_f(short_debiased, 1)}&times;</span> once the criterion's own
  measured generosity is taken out. Its shared
  latent demonstrably does the job it was designed for. On resolution it is
  indistinguishable from an ellipse and the experiment is too small to say more.
  Its stated confidence runs backwards above p&nbsp;&asymp;&nbsp;0.25 on the one fire
  where it commits at all. G3 is not met; G5 is not adjudicated. And the most
  reusable thing we produced is a list of eight places where our own
  instruments could not have told us we were wrong.</p>
  <p class="stamp">Rendered by
  <span class="mono">wildfire_nowcast.sim.g6_report</span> at
  {_esc(stamp)}. Self-contained: no external fonts, scripts, stylesheets or
  images. Light and dark are both defined; the page follows the system setting
  and honours an explicit <span class="mono">data-theme</span> if one is
  set.</p>
</footer>
</div>
"""

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n'
        '<meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        "<title>What We Measured</title>\n"
        f'<meta name="description" content="G6 readout: G3 is not met at '
        f"{_f(ours['A'], 4)} against a [{_f(facts.bar_low, 4)}, "
        f'{_f(facts.bar_high, 4)}] bar." />\n'
        f"<style>{CSS}</style>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


# -- entry point -----------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m wildfire_nowcast.sim.g6_report",
        description=(
            "Render the G6 report as one self-contained HTML page. Reads only "
            "named artifacts under runs/ and reports/figures/ plus two "
            "constants from wildfire_nowcast.common; opens no tensor and calls "
            "no model code."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="destination .html path. REQUIRED and has no default: a page "
        "written to a fixed name overwrites the last one, and a reader "
        "who finds two cannot tell which run either belongs to.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root the artifact paths are resolved against.",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run every control, each against its planted failures, and exit. The "
        "count is printed off the observations rather than asserted. Nothing "
        "is written. The same controls are collected by name from "
        "wildfire_nowcast.sim.selftest, which runs them on constructed inputs.",
    )
    args = parser.parse_args(argv)

    artifacts = ArtifactSet.default(args.root.resolve())
    if args.selftest:
        lines = selftest(artifacts)
        for line in lines:
            print(line)
        # COUNTED off the observations, never asserted. A hand-written tally is
        # a number that keeps reading "6" after a plant is deleted.
        n_plants = sum(line.count(" -> ") for line in lines)
        print(f"{len(lines)} controls, {n_plants} plants, all fired.")
        return 0
    if args.out is None:
        parser.error("--out is required unless --selftest is given")
    facts = load_facts(artifacts)
    page = render_page(facts)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    size_kib = len(page.encode("utf-8")) // 1024
    print(f"wrote {out} ({size_kib} KiB, self-contained)")
    print(
        f"  G3 dispersion, our kernel: {facts.dispersion['m30_ref']['A']} (A) / "
        f"{facts.dispersion['m30_ref']['B']} (B) "
        f"against [{facts.bar_low}, {facts.bar_high}] -- OUTSIDE"
    )
    return 0


# -- self-checks -----------------------------------------------------------
#
# This page ends on the finding that eight of this project's checks could not
# have failed, and that the only procedure which ever caught one was planting
# the failure and watching it fire. Shipping it with checks that were never
# seen failing would be the same defect one level up.
#
# Everything below runs the module-level predicates against the REAL artifacts
# and the REAL page. The same predicates are run against constructed inputs by
# ``wildfire_nowcast.sim.selftest``, which pytest collects by name. Neither
# caller is sufficient: this one proves the shipped page passes, that one
# proves the controls could have refused it.


def _observe(name: str, *, control: Any, plants: dict[str, Any]) -> str:
    """Run one control once and each of its plants once.

    ``control`` must not raise and every plant must. A plant that stays silent
    is reported by name, because "the control passed" is exactly the evidence
    this report spends a section calling worthless.
    """
    control()
    seen: list[str] = []
    for label, plant in plants.items():
        try:
            plant()
        except Exception as exc:  # noqa: BLE001 - any refusal is the point
            seen.append(f"{label} -> {type(exc).__name__}")
            continue
        raise SelfCheckFailure(
            f"{name}: the plant {label!r} did NOT fire. The control passed, "
            f"which on its own is exactly the evidence this report says is "
            f"worthless."
        )
    joined = "\n          ".join(seen)
    return f"  PASS  {name}\n          {joined}"


def _planted_bar_copy(artifacts: ArtifactSet, field_name: str, **moved: float) -> Path:
    """Write a copy of one g3 artifact with its stored bar moved. Caller unlinks."""
    raw = _read_json(getattr(artifacts, field_name))
    for entry in raw["g3"]["bar"]:
        if entry["key"] == "band_area_dispersion_ratio":
            entry.update(moved)
    tmp = Path(tempfile.gettempdir()) / f"_g6_planted_{field_name}.json"
    tmp.write_text(json.dumps(raw), encoding="utf-8")
    return tmp


def selftest(artifacts: ArtifactSet) -> list[str]:
    """Every control, run against the REAL artifacts and the REAL page, with its plants.

    The number of controls is not written down here or in the caller. The
    caller counts what this returns, because a hand-written tally is a number
    that keeps reading "six" after a seventh is added or a plant is deleted.
    """
    import dataclasses  # noqa: PLC0415

    from wildfire_nowcast.common import dispersion as _dispersion  # noqa: PLC0415

    lines: list[str] = []
    facts = load_facts(artifacts)
    page = render_page(facts, generated_utc="1970-01-01T00:00:00Z")

    # 1. the stored bar is cross-checked against the code, IN BOTH DRAWS
    def _plant_stored(field_name: str, **moved: float) -> None:
        tmp = _planted_bar_copy(artifacts, field_name, **moved)
        try:
            load_facts(dataclasses.replace(artifacts, **{field_name: tmp}))
        finally:
            tmp.unlink(missing_ok=True)

    def _plant_code_side() -> None:
        real = _dispersion.BAR_INTERVAL
        try:
            _dispersion.BAR_INTERVAL = (0.5, 1.25)
            load_facts(artifacts)
        finally:
            _dispersion.BAR_INTERVAL = real

    lines.append(
        _observe(
            "the stored bar is cross-checked against common.dispersion.BAR_INTERVAL",
            control=lambda: load_facts(artifacts),
            plants={
                "draw A low moved": lambda: _plant_stored("g3_draw_a", low=0.5),
                "draw A high moved": lambda: _plant_stored("g3_draw_a", high=1.25),
                "draw B moved": lambda: _plant_stored("g3_draw_b", low=0.5, high=1.25),
                "the code-side constant moved": _plant_code_side,
            },
        )
    )

    # 2. a missing artifact refuses rather than rendering an empty section
    lines.append(
        _observe(
            "an absent artifact raises instead of dropping its section",
            control=lambda: _read_json(artifacts.m34_summary),
            plants={
                "a path that never existed": lambda: _read_json(
                    artifacts.root / _p("no", "such.json")
                )
            },
        )
    )

    # 3. the page is self-contained
    lines.append(
        _observe(
            "the page is self-contained: no script, link, import or outward src",
            control=lambda: assert_page_is_self_contained(page),
            plants={
                "a sibling image": lambda: assert_page_is_self_contained(
                    page.replace("<body>", f'<body><img src="{_p("figures", "x.png")}">')
                ),
                "a stylesheet link": lambda: assert_page_is_self_contained(
                    page.replace("<body>", '<body><link rel="stylesheet" href="a.css">')
                ),
            },
        )
    )

    # 4. every colour token has a definition on bare :root
    lines.append(
        _observe(
            "every colour token is defined on bare :root, not only in a dark block",
            control=lambda: assert_colour_tokens_have_a_bare_root_definition(page),
            plants={
                "--ink defined only in the dark block": (
                    lambda: assert_colour_tokens_have_a_bare_root_definition(
                        page.replace("  --ink: #1b1a17;\n", "", 1)
                    )
                )
            },
        )
    )

    # 5. the source stays ASCII, which is what keeps the output-literal sink at 0
    source = Path(__file__).read_text(encoding="utf-8")
    lines.append(
        _observe(
            "the module source is pure ASCII",
            control=lambda: assert_source_is_ascii(source),
            plants={
                "an em dash typed into the source": lambda: assert_source_is_ascii(
                    source + "\n# " + chr(0x2014) + "\n"
                )
            },
        )
    )

    # 6. the number on the page is the number in the artifact
    truth = float(
        _read_json(artifacts.g3_draw_a)["g3"]["models"]["m30_ref"]["criteria"][DISPERSION_LABEL][
            "equal_block"
        ]
    )
    lines.append(
        _observe(
            "the headline dispersion value on the page is the artifact's, to 10 dp",
            control=lambda: assert_page_states_value(page, truth, what="the headline dispersion"),
            plants={
                "off by 1e-9": lambda: assert_page_states_value(page, truth + 1e-9),
                "agreeing only to 4 dp": lambda: assert_page_states_value(
                    page, round(truth, 4) + 1e-9
                ),
            },
        )
    )

    # 7. no path on the page sends a reader to a file that is not in the repo.
    #    Scanned on the RENDERED PAGE, never the source: this module joins its
    #    path fragments at run time, so a source scan is blind by construction
    #    and that blindness is exactly what let two untracked coordination
    #    documents onto a public page.
    index = tracked_paths(artifacts.root)
    rows = [rel for _, rel, _ in artifacts.as_rows()]

    def _check(t: str) -> None:
        assert_page_cites_nothing_a_cloner_cannot_open(t, tracked=index, evidence=rows)

    lines.append(
        _observe(
            "no path printed on the page is missing from the git index",
            control=lambda: _check(page),
            plants={
                # ASSEMBLED, every one of them. A plant is by definition a path
                # that must NOT resolve, so spelling one as a literal in tracked
                # source turns `tools/cited_paths.py` red for the plant being
                # exactly what it is meant to be. These two are the real
                # offenders this control was written for.
                "a coordination file named in prose": lambda: _check(
                    page.replace("<body>", "<body><p>see " + _p("coordination", "STATE.md")),
                ),
                "a planning document named in prose": lambda: _check(
                    page.replace(
                        "<body>", "<body><p>see " + _p("docs", "play" + "book.md") + ":510</p>"
                    )
                ),
                "a BARE untracked filename": lambda: _check(
                    page.replace("<body>", "<body><p>see " + "play" + "book.md" + "</p>")
                ),
                # the exemption is the DECLARED input set, not the directory:
                # an artifact-shaped path nobody reads must still refuse, or
                # "anything under runs/" becomes an open door with a name on it.
                "an artifact path that is not a declared input": lambda: _check(
                    page.replace(
                        "<body>", "<body><p>see " + _p("runs", "not_an_input.json") + "</p>"
                    )
                ),
                "an evidence row kept, its disclosure deleted": lambda: _check(
                    page.replace(EVIDENCE_DISCLOSURE, "")
                ),
            },
        )
    )
    # RESTORE, run after the plants rather than before them: the plants above
    # operate on copies, so the only way to show they left nothing behind is to
    # re-run the control on the real page once they are done.
    _check(page)
    return lines


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
