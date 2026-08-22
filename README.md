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
about. labels are gofer hourly perimeters for california fires. gofer publishes
2019 to 2021 and stops there, so the nine fires in the corpus from 2022 to 2025
use our own reimplementation of the algorithm and are marked as such in every
manifest. that distinction is carried through to the results, because a gate
spanning both is partly measuring reimplementation fidelity.

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
from 21 fires. that second component is designed and not built: the kernel
records it as absent, and nothing here has ever been scored with one.

pixels are conditionally independent bernoulli given a shared per-step latent
z_t. the latent is what stops the ensemble collapsing, because ten thousand
independent bernoullis average out and total burned area concentrates on its
mean. that is measured rather than assumed, and the measurement says it depends
on the arm: removing the latent narrows the ensemble by 3.7 to 7.8 times on 16
of 25 arms and by 1.1 to 1.6 times on the other nine, which had almost no
dispersion to lose. on the candidate of one dispersion-gate attempt it moved
1.09 times, because that arm's latent was already barely wider than none. models
without the shared latent exist here only as ablations.

training is one-step nll plus a multi-step pushforward loss scored with brier
and crps. the design allows a three to six hour pushforward and every run of
record uses three hours. oversampling barrier-crossing windows is in the design
and is not implemented: the crossing detector finds four admissible events
across all 21 fires, which is not a stratum anything can be sampled from. label
noise is modelled explicitly by dilate and erode augmentation, because gofer's
effective resolution is about 2 km on a 1 km grid, which puts the labels at
roughly twice the cell size.

baselines are persistence, a wind-advected ellipse fitted per horizon, and
elmfire monte carlo with default rothermel parameters. elmfire takes native 30 m
inputs and only its output is coarsened, so the comparison does not handicap it.

metrics are reliability at one, two and three hours, brier, arrival-time crps,
ensemble dispersion ratio, and best-member mode-capture iou on documented wind
runs. three of those, reliability, brier and arrival-time crps, are reported and
are not permitted to decide anything, and neither is the calibration error we
compute beside them: we measured all four running backwards against forecast
error on a degradation ladder, so a forecast that is badly wrong can outscore
one that is nearly right. the contract marks them non-adjudicating and the code
raises rather than warns if a gate tries to use one. everything is reported
regime-stratified, dormant windows separately from growth windows, because they
are opposite-difficulty problems and an aggregate over both is mostly noise.

where it stands. the kernel beats persistence, the rule opponent and the
wind-ellipse envelope on the shape-masked iou criterion, on every held-out
block: four of four on the corpus that criterion was adjudicated on, then five
of five on the corpus today, which is a one-sided sign test at p = 0.031. we do
not quote a standard deviation for that effect, and the reason is on the record
rather than tucked away. measured against the wind ellipse, which is what the
claim is actually about, the separation sits below the two standard deviation
bar the design set, so the claim rests on the unanimity across blocks and not on
its size. an earlier version of the same number divided by the spread across
training seeds instead of across held-out blocks, which is the unit the design
treats as independent, and read about nine times larger. seed spread measures
how reproducible the optimiser is, not how uncertain the estimate is, so it is
the wrong denominator. arrival-time crps used to be cited here as a second
criterion and has been withdrawn: we measured it running backwards against
forecast error, so it cannot support this claim in either direction.

calibration is not there yet. the ensemble dispersion gate has failed four
times, and the reason is structural rather than a tuning problem. real fires
decelerate as they age: 12 of the 14 spatial blocks decelerate, one-sided sign
test p = 0.0065, against a permutation null that centres on 7.01 of 14 and a
time-reversal control run through the same code that reads 2 of 14 at
p = 0.9991. the caveat travels with that number: on published gofer labels alone
the same read is 5 of 7 at p = 0.2266, so our reimplemented labels do not set
the direction but they do set the significance. a contagion kernel is
perimeter-proportional by construction, so it cannot decelerate, and the model
accelerates on 5 of 5 held-out blocks under both of the estimators we have. we
do not quote an elasticity for that gap: two of our own estimators agree on its
sign and disagree on its magnitude by a factor of 2.6, so the magnitude is not a
number we have. the wind ellipse also fails to decelerate materially, but we
cannot say it fails by the same amount as the kernel does. its estimate is about
four times less precise than the model's, wide enough to cover anything from
clear deceleration to acceleration, and our two estimators do not agree on which
of the two decelerates less. the spread of log growth rates is 1.46 in truth and
0.64 in the model, and the model's growth is more predictable from covariates
than reality's is, so the ensemble is too narrow because the rate is wrong, not
because the noise is too small.

the strongest result here is not about our model. stock elmfire, the field's
standard operational rothermel simulator, untuned by us, run on native 30 m
inputs with crown fire on, accelerates on all four of the held-out blocks it
completed. it accelerates more than our kernel does on four of four, and it ends
up farther from truth than our kernel does on three of four. in raw cells per
window rather than in a log ratio: on borel the real fire falls from 6.02 to
0.98 cells per window between its first and second half while elmfire runs 22.5
then 29.2, and on july the real fire is flat at 2.61 then 2.51 while elmfire
reaches 35.7. that is a thirtyfold and a fourteenfold over-prediction of late
growth, and it is the difference between saying a fire is finished and saying it
is still running, which is the call a one to three hour nowcast exists to
inform. the caveat travels with the number and neither half is usable without
the other: elmfire has no suppression component, and late-fire behaviour in
reality is dominated by containment lines and air attack, so the defensible
claim is that a pure physics model without suppression over-predicts late-fire
growth by 14 to 30 times on held-out california fires, not that rothermel is
wrong. czu is the weak leg, with five cells of growth across its entire early
half and elmfire under-predicting that half by a factor of 61; dropping it
leaves three of three and the verdict stands. creek is missing because elmfire's
runtime cap is wall-clock and its abort is silent, which is a reportable finding
about the tool rather than about the fire, and four positive blocks already
settle the count whatever creek would have done. the reading we take from it is
that the failure to decelerate belongs to perimeter-proportional spread models
as a class, and that adding real physics makes it worse rather than better.

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
null-check are part of the same gate, and make ci is what github actions runs on
every push — literally: the workflow has one gate step and it is make ci. that
sentence used to be an over-claim. until 2026-08-21 the workflow named its six
targets itself and a test asserted the two lists agreed, so make ci was a
faithful description of the gate rather than the gate. it now runs lint, types,
the full suite including the slow playthroughs, the mutation-coverage gate over
every playthrough, a freshly generated synthetic fire judged by the real c1-c3
checker, and the do-nothing null check. what ci does not cover is written down in
.github/workflows/ci.yml rather than left to be discovered — the fire corpus is
not in the repo, so the cross-fire split clauses run against a synthetic corpus
there, elmfire is not built, and earth engine ingestion needs credentials. the
contract version is parsed from the first line of docs/interfaces.md at import
with no fallback, so a drifted version fails loudly instead of silently.

the environment is installed, not resolved. make install runs uv pip sync
--require-hashes requirements.lock, so a laptop and the ci runner hold the same
package set by artifact hash rather than by a resolution each performs for
itself. this is recent and it was bought expensively: the lock previously sat in
the repo referenced by no install path, a clean clone resolved 15 of its 73
packages to other versions, and one of those differences — numpy 2.5.1 against
2.5.2, whose stubs disagree about a single overload — kept this badge red for
seven days while make typecheck exited 0 locally. a lock file nobody installs
from advertises a reproducibility that does not exist. make relock regenerates
it and never upgrades anything; upgrading a package means deleting its pin on
purpose.

a green make ci is a claim about your working copy, not about the repository.
make ci-status is the other claim: it asks github for the conclusion of the run
that built the exact commit on origin/main, prints whether your tree is dirty or
unpushed, and exits non-zero when it cannot tell. unknown is not green.

types. make typecheck runs mypy in strict mode over src, from a pinned isolated
environment rather than from .venv, so a developer tool cannot move the
interpreter the numbers are produced on. it is honest about what it does not
cover: 78 of the 119 modules are checked today, including all 28 of common, and
the other 41 are listed by name in pyproject.toml with the reason each one is
there. the list is a burn-down and it fails in both directions — adding a module
to it changes a pin in tests/test_typecheck_config.py, and a listed module that
has become clean fails the build until it is retired, because an exemption list
that only ever checks its ceiling turns into a permanent excuse. tests are not
type-checked yet and that gap is written down rather than hidden behind a
relaxed setting.

data is not in the repo. the fire tensors are about 700 mb, the ingestion cache
another 600 mb, and the run records 78 mb. all of it is reproducible from the
ingestion pipeline but requires earth engine credentials, and the project id is
read from wildfire_gee_project rather than hardcoded.

caveats worth reading before trusting any number here. nine of the 21 fires use
our gofer reimplementation rather than the published product. one held-out fire
sits at 0.51 against its official perimeter, which is the published product's
own worst under-map. persistence wins at the one-hour horizon on window-pooled
band brier, and that is structural, since sub-cell movement at 1 km is
quantisation rather than skill. the spot component is not built. the elmfire
head-to-head is finished and is reported above, on four of the five held-out
blocks.

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
