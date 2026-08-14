disclaimer. this is a personal experiment and is unrelated to my work. i built
it because i wanted to know whether short-range wildfire spread can be nowcast
probabilistically, and for no other reason. it uses public data only.

wildfire-nowcast

[![ci](https://github.com/teohondascully/wildfire-nowcast/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/teohondascully/wildfire-nowcast/actions/workflows/ci.yml)

short-range probabilistic wildfire spread. we learn a transition kernel
p(x_{t+1h} | x_t, features), sample monte carlo ensembles from it, and score
whether those ensembles are calibrated rather than merely sharp. the horizon is
one to three hours, which is the window where a forecast can still change what
an incident commander does.

setup. everything lives on a 1 km epsg:5070 grid at hourly utc steps, which
matches the label cadence exactly, so there is no temporal interpolation
anywhere. each cell is unburned, burning, or burned out. fire is absorbing, so
burned area never decreases and there is no stationary distribution to reason
about. labels are gofer hourly perimeters for california fires 2019-2021. gofer
stops publishing after 2021, so the later fires in the corpus use our own
reimplementation of the algorithm and are marked as such in every manifest. that
distinction is carried through to the results, because a gate spanning both is
partly measuring reimplementation fidelity.

features are rtma hourly weather, landfire fuels corrected with mtbs and nifc
burn scars, 3dep terrain, and water and barrier masks. the corpus is 21 fires
across 14 spatial blocks. cross-validation is leave-block-out rather than
leave-fire-out, because buffered fire domains overlap and two fires in the same
block share landscape, so scoring one while training on the other leaks.

the model has two components. the first is a short-range anisotropic contagion
kernel, a cnn over the burning neighbourhood, initialised as an elliptical
gaussian stretched by wind and slope, so it starts at the wind-ellipse physics
baseline and has to earn any deviation from it. the second is an explicit
long-range spot component with a learned ignition rate and a downwind dispersal
kernel, kept separate because a learned long-range kernel is not identifiable
from 21 fires.

pixels are conditionally independent bernoulli given a shared per-step latent
z_t. the latent is load-bearing. with independent per-pixel noise alone the
ensemble collapses, because ten thousand independent bernoullis average out and
total burned area concentrates on its mean. models without the shared latent
exist here only as ablations.

training is one-step nll plus a multi-step pushforward loss over three to six
hours scored with brier and crps. barrier-crossing windows are oversampled.
label noise is modelled explicitly by dilate and erode augmentation, because
gofer's effective resolution is about 2 km on a 1 km grid, which puts the labels
at roughly twice the cell size.

baselines are persistence, a wind-advected ellipse fitted per horizon, and
elmfire monte carlo with default rothermel parameters. elmfire takes native 30 m
inputs and only its output is coarsened, so the comparison does not handicap it.

metrics are reliability at one, two and three hours, brier, arrival-time crps,
ensemble dispersion ratio, and best-member mode-capture iou on documented wind
runs. everything is reported regime-stratified, dormant windows separately from
growth windows, because they are opposite-difficulty problems and an aggregate
over both is mostly noise.

where it stands. the kernel beats persistence, the rule opponent and the
wind-ellipse envelope on the shape-masked iou criterion and on arrival-time
crps, on every held-out block. the effect is about 1.8 standard deviations
measured across held-out blocks, which is the unit the design treats as
independent. an earlier version of that number divided by the spread across
training seeds instead and read about nine times larger. seed spread measures
how reproducible the optimiser is, not how uncertain the estimate is, so it is
the wrong denominator.

calibration is not there yet. the ensemble dispersion gate has failed four
times, and the reason is structural rather than a tuning problem. the elasticity
of growth rate to frontier length is about -0.78 in the data and about -0.04 in
the model. real fires decelerate as their perimeter grows, and a contagion
kernel is perimeter-proportional by construction, so it cannot. the wind ellipse
has the same defect, which is why a one-parameter baseline loses the same amount
out of sample that we do. the spread of log growth rates is 1.46 in truth and
0.64 in the model, and the model's growth is more predictable from covariates
than reality's is, so the ensemble is too narrow because the rate is wrong, not
because the noise is too small.

layout. src/wildfire_nowcast/common is the single implementation of anything the
contract adjudicates: the channel list, the state rule, the grid and lattice, the
zarr io layer, the split fingerprint, the c6 metric registries and the contract
checker itself. everything else imports it rather than reimplementing it, because
a producer and a verifier computing geometry through different code is how a
tensor passes its check and is still wrong. data/ ingests and assembles tensors,
model/ holds the kernel and the training loop, eval/ holds the metrics and the
baseline runner, sim/ holds the ensemble replay, the diagnostics and the figures.
tests/ mirrors the contract rather than the module tree: the c1-c3 suite takes
--tensor-path, so a real fire is judged by exactly the file that judges the
synthetic one.

running it. make install, then make test. make lint, make typecheck and make
null-check are part of the same gate, and make ci runs what github actions runs
on every push: lint, types, the full suite including the slow playthroughs, the
mutation-coverage gate over
every playthrough, a freshly generated synthetic fire judged by the real c1-c3
checker, and the do-nothing null check. what ci does not cover is written down in
.github/workflows/ci.yml rather than left to be discovered — the fire corpus is
not in the repo, so the cross-fire split clauses run against a synthetic corpus
there, elmfire is not built, and earth engine ingestion needs credentials. the
contract version is parsed from the first line of docs/interfaces.md at import
with no fallback, so a drifted version fails loudly instead of silently.

types. make typecheck runs mypy in strict mode over src, from a pinned isolated
environment rather than from .venv, so a developer tool cannot move the
interpreter the numbers are produced on. it is honest about what it does not
cover: 66 of the 107 modules are checked today, including all 26 of common, and
the other 41 are listed by name in pyproject.toml with the reason each one is
there. the list is a burn-down and it fails in both directions — adding a module
to it changes a pin in tests/test_typecheck_config.py, and a listed module that
has become clean fails the build until it is retired, because an exemption list
that only ever checks its ceiling turns into a permanent excuse. tests are not
type-checked yet and that gap is written down rather than hidden behind a
relaxed setting.

data is not in the repo. the fire tensors are about 1.3 gb and the run records
another 38 mb. both are reproducible from the ingestion pipeline but require
earth engine credentials, and the project id is read from wildfire_gee_project
rather than hardcoded.

caveats worth reading before trusting any number here. nine of the 21 fires use
our gofer reimplementation rather than the published product. one held-out fire
sits at 0.51 against its official perimeter, which is the published product's
own worst under-map. persistence wins at the one-hour horizon on window-pooled
band brier, and that is structural, since sub-cell movement at 1 km is
quantisation rather than skill. the spot component and the elmfire head-to-head
are not finished.

docs/interfaces.md is the data and api contract, and is load-bearing: the
package parses its version from line 1. docs/decisions.md records why the
thresholds and splits are where they are, including the ones that were retracted.

the adr-nnn citations throughout the source refer to a decision log kept outside
this repo, one numbered entry per ruling, written before the result it governs
wherever the ruling was a threshold or a prediction. docs/decisions.md is the
readable summary of the ones that still bind. the citations are left in the code
because knowing that a constant was argued about, and roughly when, is worth more
than a clean comment — but they will not resolve to anything you can open, and
that is a limitation of this repo rather than a hint that you missed a file.
