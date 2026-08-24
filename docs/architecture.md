# architecture

how this code is shaped and why it is shaped that way. the science is in
`README.md` and the numbered contract is in `docs/interfaces.md`; this file is
about the software.

the short version: there is one package that defines what everything means, four
packages that consume it, a written contract those five agree on, and a set of
instruments whose job is to make a claim about a number checkable rather than
believable.

## contents

- [the one-way dependency rule](#the-one-way-dependency-rule)
- [the contract is the interface](#the-contract-is-the-interface)
- [what a fire is on disk](#what-a-fire-is-on-disk)
- [the sampling loop](#the-sampling-loop)
- [how a number is made checkable](#how-a-number-is-made-checkable)
- [the synthetic fire, and why it exists](#the-synthetic-fire-and-why-it-exists)
- [what is designed and not built](#what-is-designed-and-not-built)

## the one-way dependency rule

`src/wildfire_nowcast/common/` is the only implementation of anything the
contract adjudicates: the channel list, the state rule, the grid, the zarr io
layer, component labelling, the split fingerprint, the normalisation stats
reader, and the contract checker itself. `data/`, `model/`, `eval/` and `sim/`
import it and none of them reimplements it. `common/` imports none of the four at
module level, which is the exact form of the rule and is measured below.

this is C0 in `docs/interfaces.md` and it is not a style preference. the failure
it prevents is specific: if the code that writes a tensor computes its geometry
one way and the code that verifies the tensor computes it another, the tensor
passes its own check and is still wrong, and nothing anywhere goes red. the same
argument is why the perimeter-to-state rule, which looks like ingestion logic,
lives in `common/` rather than in `data/`.

the rule holds at module import time, and that is the precise claim, so here is
the measurement rather than the slogan. walking the syntax tree of every tracked
module in `common/` finds **zero** module-level runtime imports of `data/`,
`model/`, `eval/` or `sim/`. it finds six such imports in total, and every one of
them is either typing-only or deferred inside a function body:

```
common/synthetic.py:80            data.ignitions      TYPE_CHECKING only
common/synthetic.py:636           data.ignitions      deferred
common/null_check/forecasters.py  eval.masks          deferred, two sites
common/null_check/verdicts.py     eval.metrics        deferred, two sites
```

so `common/` imports cleanly on its own and the package graph has no cycle at
import time, and the null-check harness borrows the mask and metric
implementations it is scoring rather than growing a second copy of them. that is
a deliberate trade rather than an oversight, and it is worth knowing about before
adding a seventh.

## the contract is the interface

`docs/interfaces.md` is a numbered document and its first line carries the
version. the clause families are:

```
C0    single implementation: contract-adjudicated code lives in common/
C1    the per-fire tensor: dims, channel order, dtypes, CRS, hourly time
C2    the per-fire manifest: provenance, spatial_block_id, fold assignment
C3    normalisation statistics, computed over TRAIN folds only
C4    the synthetic fire generator
C5    the model prediction API, shared by the kernel and the baselines
C6    the metrics API, and C6.0's rule that every metric must beat a null
C7    config: one yaml per experiment, resolved config plus git SHA per run
C8    the split fingerprint, and what a mismatch means
C-1 to C-4   governance: check severity, ratification, n=1 thresholds,
             and what may move while a run is in flight
```

two properties of this arrangement are worth naming because they are unusual.

first, the version has exactly one home. `common/contract.py` derives
`CONTRACT_VERSION` from line 1 of `docs/interfaces.md` at import time with no
literal fallback, so a document that has drifted from the code fails loudly at
import rather than silently at read. it had drifted four times before that was
made mechanical.

second, ratification is not implementation. a clause can be agreed and written
down and still be enforced by nothing, which is a state this project has been in
and which C-2 exists to make visible. the executable form of C1, C2 and C3 is
`common/contract.py`, and `make contract TENSOR=path` runs it against any store.

## what a fire is on disk

one fire is a directory holding a zarr store and a manifest. the store has two
variables, not one, because a single zarr array has a single dtype and the state
channel is categorical:

```
fire_state   uint8    (time, y, x)                values {0, 1, 2}
features     float32  (time, channel, y, x)       13 named channels
```

the `channel` coordinate carries channel names rather than positions, and an
attribute records the offset that maps a position back to the original index, so
the naming change did not renumber anything.

the grid is EPSG:5070 at 1000 m, north-up. time is hourly UTC, monotone, with no
gaps. the state values are 0 unburned, 1 burning, 2 burned out, and the state is
absorbing: burned area never decreases. `common/states.py` holds the one
implementation of the perimeter-to-state rule that produces them.

normalisation statistics are a separate file, computed over training folds only,
and models consume them from that file rather than recomputing inline. the file
carries the fold list that produced it so that "training folds only" is
auditable rather than asserted.

## the sampling loop

`model/api.py` defines one prediction interface and everything implements it,
including the physics baselines:

```
predict(x0, static, weather, n_members, horizon_h, seed) -> samples
```

the shape of `samples` is `[n_members, horizon_h, H, W]` of `fire_state`. that
one signature is why a baseline and the learned kernel can be scored by the same
code path: `eval/metrics.py` takes samples and truth and never touches model
internals, so no argument choice inside a model can change what a headline
number means.

the ablation arm is part of the interface rather than a separate script.
`load_model(f"{path}__independent")` resolves to the same model with the shared
latent removed, and it shares the parent's parameters by identity, checked on
every call. an arm that merely held equal parameters would confound the sampler
with the fit and still read as a clean result.

## how a number is made checkable

this is where most of the tooling is, and it is the part of the repository a
reader is least likely to guess at. five mechanisms, each of which exists
because something went wrong without it.

**run directories.** `common/runs.py` is the only implementation of C7's rule
that a run writes its fully resolved config and a git SHA into
`runs/{run_id}/config.yaml`, plus a machine-readable copy in `run_meta.json`. a
run directory therefore means the same thing regardless of what produced it.

**code fingerprints.** `common/codefingerprint.py` records which code produced a
number, and it discovers the covered set by walking the tree rather than from a
hand-written list. the list it replaced omitted two modules, so a run stamped
"one version of the scoring code" while an omitted module was edited afterwards.
nothing was red at any point. the defect was not a check that could not fail; it
was a check that could not discriminate.

**the split fingerprint.** C8. every run stamps the split it used and a mismatch
between the training split and the evaluation split is a hard failure. this
exists because the split once moved mid-run and four fires crossed from train to
held-out while every individual tensor stayed conformant. no per-tensor check
could have seen it, because the defect is in the relation between tensors and a
running experiment.

**the do-nothing null.** C6.0. before any metric is allowed to adjudicate
anything, a forecaster that predicts nothing is scored against it. if the null
wins, the metric is broken rather than the model. three metrics here have failed
that way. `make null-check` runs it.

**playthroughs.** a playthrough is an end-to-end scenario whose correct answer is
known by construction, plus a scoring function returning pass or fail, plus a
planted defect the harness demonstrably detects. the third clause is the one that
does the work: a playthrough here was found to be one that could not have failed,
because its planted defect was invisible on smooth inputs, and the module that
makes the protocol mechanical records three separate occasions on which the rule
paid. `make playthrough` runs the registry, and `make playthrough-list` prints
what exists and which are slow.

## the synthetic fire, and why it exists

`common/synthetic.py` produces a complete, contract-conformant fire on disk in
well under five seconds, with no network, no credentials and no Earth Engine.
that is what let model and simulation work start against real-shaped data before
the ingestion path existed.

it deliberately exercises the awkward cases rather than the easy ones: all three
states occur, the perimeter-to-state mapping goes through the same canonical
implementation a real fire uses, a barrier crossing is scripted so that event
type is present, and a dormancy is scripted so frames with no burning cell occur
naturally rather than being relaxed out of a test.

the same generator is why the gate runs on a clean clone. `make synth` writes a
fire and `make contract` judges it with the checker that judges real fires, so
the C1 to C3 clauses are exercised on every push without the corpus being
present.

## what is designed and not built

stated here because an architecture document that only describes what exists is
a sales document.

* the long-range spot component is designed and is not built. the kernel records
  it as absent and nothing has been scored with one.
* `tests/contract/` is proposed in `tests/README.md` and does not exist, so
  there is still no route from a clause number to the code enforcing it.
* the mutation sweep is a real gate with a real budget and it is not part of
  `make ci`, because one sweep measured at over an hour and a half against a
  full suite of about four minutes. the comparison is gated; the value compared
  against is not.
