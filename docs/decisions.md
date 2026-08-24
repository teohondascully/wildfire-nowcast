# decisions

why the thresholds, splits and criteria are where they are. kept because a
number without its reason is not reproducible, and because several of these were
wrong the first time and the retractions matter more than the results.

## contents

- [splits are leave-block-out](#splits-are-leave-block-out)
- [the split fingerprint](#the-split-fingerprint)
- [membership is declared by fire id](#membership-is-declared-by-fire-id)
- [separation is measured across blocks and not across seeds](#separation-is-measured-across-blocks-and-not-across-seeds)
- [the dispersion bar is geometric](#the-dispersion-bar-is-geometric)
- [dispersion alone cannot decide the gate](#dispersion-alone-cannot-decide-the-gate)
- [ratios pool in log space](#ratios-pool-in-log-space)
- [an empty block is a hard failure](#an-empty-block-is-a-hard-failure)
- [undefined is its own outcome](#undefined-is-its-own-outcome)
- [every metric is scored against a do-nothing null](#every-metric-is-scored-against-a-do-nothing-null)
- [everything is reported stratified by regime](#everything-is-reported-stratified-by-regime)
- [the contract version has one home](#the-contract-version-has-one-home)
- [diagnosed cause of the calibration failure](#diagnosed-cause-of-the-calibration-failure)

## splits are leave-block-out

splits are leave-block-out, not leave-fire-out. fire domains are buffered and
overlap, so two fires in the same spatial block share landscape. scoring one
while training on the other leaks, and every per-fire check passes while it
happens. the corpus is 21 fires across 14 blocks; 16 fires in 9 blocks train,
5 fires in 5 blocks are held out.

## the split fingerprint

the split is fingerprinted and the fingerprint is pinned in a test. it is
supposed to fail when the corpus legitimately changes, at which point the new
value is written deliberately in the same commit. both the current and the
previous fingerprint are kept as named constants, because every earlier result
is bound to the old one and quoting across that boundary is the error the
fingerprint exists to prevent.

## membership is declared by fire id

train and held-out membership is declared by fire id, not by fold index, and a
non-empty intersection is a hard failure. the check this replaced could not
fail: it intersected two lists that the fingerprint routine constructs as a
partition, so it had been green since it was written and could never be
anything else. the leak it existed to catch, one fire worth 9.76 percent of
training mass, was found by reading the file. every check added since ships with
a planted defect that it demonstrably catches.

## separation is measured across blocks and not across seeds

separation is measured in standard deviations across held-out blocks, never
across training seeds. seed spread measures how reproducible the optimiser is,
and it shrinks toward zero as training becomes deterministic, which inflates any
ratio built on it without bound. the headline result was originally quoted in
seed units and read about nine times larger than it should have. it still holds
in block units, at about 1.8 sd with agreement on every block, but the magnitude
was wrong and is retracted.

## the dispersion bar is geometric

the dispersion bar is geometric: |log(ratio)| <= log(1.2), that is 0.8333 to
1.2. the earlier 0.8 to 1.2 was not symmetric. in log units it was 22 percent
more tolerant of under-dispersion, which is the only side this project has ever
failed on, so it was flattering the model.

## dispersion alone cannot decide the gate

dispersion alone cannot decide the calibration gate. the dispersion ratio
factorises, and the growth-calibration term dominates it, so a model can satisfy
the ratio through compensating first- and second-moment errors. the gate
therefore also requires that mean-growth mis-calibration be no worse than the
best physics baseline's. that condition is defined against a reference rather
than as an absolute number, so no threshold is fitted to any run's result.

## ratios pool in log space

ratios pool in log space. pooling a ratio arithmetically reintroduces, one level
down, the same asymmetry the geometric bar removes: a block over-predicting by
4x and one under-predicting by 4x average to 2.125 arithmetically and to 1.0
geometrically. this was decided while both poolings still agreed on every case,
specifically so the choice could not be made for its outcome.

## an empty block is a hard failure

a block that contributes nothing to an equal-block mean is a hard failure rather
than a silently smaller sample, and the pooled result carries the number of
blocks that actually contributed. silent partial coverage reads as full
coverage.

## undefined is its own outcome

the dispersion ratio is undefined at perfect mean calibration, because its
denominator is the model's own mean-area error and that is exactly zero there.
the metric goes undefined precisely as a model gets the first moment right.
undefined is reported as its own outcome and is never a pass.

## every metric is scored against a do-nothing null

no metric enters a gate until a do-nothing forecast has been scored against it.
the original shape metric ranked predicting nothing above predicting something,
and an untrained model beat its trained self under it. any higher-is-better
metric must pay a claims-nothing forecast the minimum of its range.

## everything is reported stratified by regime

everything is reported stratified by regime, dormant windows separately from
growth windows. they are opposite-difficulty problems and an aggregate over both
is mostly noise. relatedly, losing to persistence at the one-hour horizon is
structural and not a defect: at 1 km, sub-hour movement is quantisation.

## the contract version has one home

the contract version is parsed from line 1 of docs/interfaces.md at import, with
no literal fallback. it had drifted from the document four times, and a fallback
is how that drift stays hidden.

## diagnosed cause of the calibration failure

the elasticity of growth rate to frontier length is about -0.78 in the data and
about -0.04 in the model, so real fires decelerate as their perimeter grows and
this model cannot. a contagion kernel is perimeter-proportional by construction.
the wind-ellipse baseline has the same coefficient, which explains why a
one-parameter baseline loses the same amount out of sample. the spread of log
growth rates is 1.46 in truth against 0.64 in the model, and model growth is
more predictable from covariates than reality's is, so the ensemble is too
narrow because the rate is wrong rather than because the noise is too small. two
earlier hypotheses, a missing dormant state and a compressed wind response, were
pre-registered and both falsified.
