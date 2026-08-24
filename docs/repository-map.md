# repository map

what is in this repository, directory by directory, and which directories are
here but hold nothing you can open. written because the shape of this tree is
unusual in three ways: one package defines what every other package is allowed to
mean, several of the directories a working machine has are outputs rather than
sources and are absent from a clone, and the largest single thing the project
depends on is not in the repository at all.

everything below is a statement about the tracked tree, i.e. about what a
`git clone` gives you. anything generated is marked as such.

## contents

- [top level](#top-level)
- [the source package](#the-source-package)
- [what is generated and not tracked](#what-is-generated-and-not-tracked)
- [where to look for a given question](#where-to-look-for-a-given-question)

## top level

```
src/wildfire_nowcast/   the package. five sub-packages, described below
tests/                  the test suite. see tests/README.md for where a test goes
tools/                  repository instruments: gates, scanners, guards
configs/                one yaml per experiment (C7)
docs/                   the contract, the decisions, and these guides
data/                   normalisation statistics only. the fire corpus is NOT here
runs/                   analysis scripts and the json records they produced
notebooks/              empty except for a .gitkeep. nothing in src imports a notebook
Makefile                every developer task. `make help` lists them
pyproject.toml          package metadata, ruff and mypy configuration
requirements.lock       the hash-pinned environment `make install` syncs to
.github/workflows/ci.yml   the published gate. it has one step and it is `make ci`
```

`data/` holds `data/norm_stats.json` and five leave-fold-out variants of it, and
one label-noise index under `data/interim/_index/`. it does not hold fires. the
per-fire tensors are large and reproducible from the ingestion pipeline, so they
are generated rather than tracked.

`runs/` is a mixed directory on purpose. most of it is ignored, and a small
number of analysis scripts and json records are force-added so that a published
number can be opened rather than only re-run. `tools/cited_runs.py` enumerates
every `runs/` path the tracked tree cites and fails if one of them is not
something a cloner can open.

## the source package

```
src/wildfire_nowcast/
  common/     everything the contract adjudicates. 22 modules plus a seven-module
              null_check sub-package. imported by all four others
  data/       ingestion and assembly: sources, rasterisation, folds, QA
  model/      the transition kernel, its training loop, and the baselines
  eval/       metrics, masks, the baseline runner, the reporting gate
  sim/        ensemble replay, diagnostics, figures, the ELMFIRE arm
```

the dependency rule is one-way and is the single most load-bearing fact about
this layout: `common/` imports none of the other four at module level, and the
other four import `common/` rather than reimplementing anything it defines. C0 in
`docs/interfaces.md` states it and gives the reason, which is that a producer
and a verifier computing geometry through different code is how a tensor passes
its check and is still wrong. the exact form of the rule, and the five deferred
imports that are its only exceptions, are measured in `docs/architecture.md`.

what lives in `common/`, by subject:

```
contract.py        the channel list, dtypes, grid and time semantics, and the
                   executable form of C1/C2/C3. CONTRACT_VERSION is parsed from
                   line 1 of docs/interfaces.md at import, with no fallback
zarr_io.py         the one writer and reader for C1/C2/C3 artifacts
grid.py            the analysis grid: EPSG:5070, 1 km cells, north-up
states.py          the ratified perimeter to fire_state rule
components.py      the one implementation of 8-connected component labelling
synthetic.py       C4, the synthetic fire generator
config.py          yaml config loading with a defaults list, no hydra runtime
runs.py            run directories: runs/{run_id}/ with resolved config + git SHA
paths.py           path resolution. no module in src may hardcode an absolute path
splits.py          C8, the split fingerprint and the cross-fire clauses
null_check/        C6.0, the do-nothing null every metric must beat
codefingerprint.py which code produced a number, discovered from the tree
playthrough.py     the rule that a playthrough which cannot fail must not ship
```

that is twelve of the twenty-two. the remaining nine are scoring machinery the
contract adjudicates (`calibration.py`, `dispersion.py`, `iou_terms.py`,
`pooling.py`, `separation.py`, `derive.py`) and cross-cutting utilities with the
same single-implementation obligation (`seeds.py`, `logs.py`, `environment.py`),
plus `__init__.py`.

## what is generated and not tracked

these directories exist on a working machine and are absent from a clone:

```
data/fires/     the per-fire tensors. about 700 MB, produced by the ingestion
                pipeline, which needs Earth Engine credentials
outputs/        whatever `make synth` and friends write. `make clean-outputs`
                removes it
reports/        rendered figures and html. regenerated from runs
vendor/         third-party inputs, including the ELMFIRE working tree
.venv/          the environment `make install` builds
```

the consequence worth stating plainly: a fresh clone can run the whole gate,
because the gate's data-shaped inputs are synthesised by C4 rather than read
from `data/fires/`. it cannot reproduce a headline number, because that needs
the corpus. `.github/workflows/ci.yml` says which clauses run against synthetic
data for exactly this reason.

## where to look for a given question

```
what does a tensor have to look like        docs/interfaces.md, C1 to C3
why is a threshold where it is              docs/decisions.md
what does the code actually check           src/wildfire_nowcast/common/contract.py
how do I run the gate                       docs/testing.md
where does a new test go                    tests/README.md
what broke and what does it mean            docs/troubleshooting.md
how is the code shaped, and why             docs/architecture.md
```
