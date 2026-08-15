# INTERFACES v2.15 (bump version + DECISIONS.md entry to change anything here)
# v2.15 ADR-053 (1)(2): C6.6 — Brier, arrival-time CRPS, `calibration_error` and
#       reliability are NON-ADJUDICATING. They ran BACKWARDS on M11's ladder
#       (Spearman -0.45 / -0.34 / -0.14 / -0.80 against |log(area error)|), so a
#       forecast 40x TOO SMALL beats a 3%-correct one by 17% on Brier. This bump
#       only ever REMOVES a metric's power to pass something, which is why it is
#       the one bump permitted under the ADR-054 freeze. The flag is CARRIED by
#       the C6 registry in `common/` and an adjudicating call on a barred
#       channel RAISES — see C6.6. THIS BUMP WAS MADE BY infra UNDER AN EXPLICIT
#       DIRECTIVE (task I1), the same precedent as v2.12 below; no other line of
#       this contract was touched by that lead. [v2.15]
# v2.14 ADR-043: C6.5 (geometric G3 bar + first-moment condition) and C6.3's
#       pooling addition RATIFY — held back at v2.13 precisely because they were
#       implemented in `common/` but not CALLED. Verified before bumping:
#       `eval/baseline_run.py` imports `common.dispersion as g3`, invokes
#       `g3.first_moment_condition_from_blocks` and emits `g3_conditions`; the
#       local `equal_block_mean` is DELETED and `common/pooling.py` holds the
#       only definition (C0). A clause is ratified when the gate path RUNS it.
# v2.13 ADR-039/040: C3.5 — the DECLARED train/held-out membership is checked by
#       ID, and a non-empty intersection is a HARD failure. Bumped ONLY for the
#       clause that is LIVE: I planted `2020_creek` in both lists and watched
#       `check_split_assignment()` go False, then planted a missing key and
#       watched it go False again, then restored and watched it go True.
#       **The G3 geometric bar + first-moment condition (ADR-039 (4),(5)) is
#       DELIBERATELY NOT IN THIS BUMP.** It is implemented and unit-tested in
#       `common/dispersion.py`, but NOTHING IN `eval/` CALLS IT — I checked, and
#       the only importer is `common/pooling.py`. A clause is ratified when the
#       gate path runs it, not when the function exists (C-2). It lands at v2.14
#       after modelling wires it and I verify the wiring on a real run.
# v2.12 ADR-024: the INTERPRETER ENVIRONMENT joins C-4's frozen set (C-4.3).
#       Ratified as a DIRECTIVE at ADR-024 and deliberately left `PENDING`
#       because infra was barred from running and C-2 forbids a
#       ratified-but-unimplemented clause. IMPLEMENTED at A13 (2026-08-09):
#       `common/environment.py`, stamped structurally by `common/runs.py`,
#       hard-failed by `C8.environment_agrees_across_run`. THIS BUMP WAS MADE BY
#       infra UNDER THE EXPLICIT A13 DIRECTIVE ("bump CONTRACT_VERSION to
#       v2.12, register the clause, un-mark the PENDING"), which cannot be done
#       without it: `tests/test_clause_registry.py` reads line 1 of this file.
#       No other line of this contract was touched by that lead. [v2.12]
# v2.11 ADR-022: C-4 concurrency freeze; `eval/` ownership split is WRITTEN
#       DOWN (C-4.1); code fingerprints sampled BEFORE and AFTER (C-4.2).
#       Also: `make test` never hung — that was an maintainer error. [v2.11]
# v2.10 ADR-017: `best_member_iou` REWARDS SILENCE — unfit to adjudicate.
#       Third metric pathology of the same shape. [v2.10]
# v2.9 ADR-016: 9 of 26 clauses were FICTION. C-2 is now a build failure
#      (tests/test_clause_registry.py). Every clause carries an implementation
#      status. Clauses spanning MORE THAN ONE ARTIFACT need a home that sees
#      more than one — that blind spot is what hid all nine. [v2.9]
# v2.8 C8 split_fingerprint + per-horizon ellipse calibration — ADR-015. [v2.8]
# v2.7 C2 fuel_vintage_lag_years + n_ignition_components — ADR-014. [v2.7]
# v2.5 adds C6.1/C6.2/C6.3 gate-metric clauses — see ADR-011. Marked [v2.5].
# v2.6 adds C-2 governance + C-3 no-n=1 rules, defers C1.6 — ADR-013. [v2.6]
# v2.4 adds C1.7 physical-range clause — see ADR-010. Marked [v2.4].
# v2.3 ratifies infra A3 proposals — see ADR-009. Marked [v2.3].
# v2.2 ratifies data A4 proposals P6/P7/P8 — see ADR-008. Marked [v2.2].
# v2.1 ratifies infra A1 proposals — see ADR-007: terms ENFORCED BY TESTS
#      but never stated. Marked [v2.1].
# v2   ratifies data PROPOSALS P1/P2/P4 — see ADR-006.
# v1 channel indices are preserved throughout; nothing was ever renumbered.

## C-1. Check severity is TWO-TIER [v2.3] (ADR-009)
`fail`      — invariant violated OR unverifiable. Exits non-zero always.
`reporting` — verified and satisfied but recorded non-canonically, or scoped
              by contract to REPORTED results. Prints unconditionally, sets
              `reporting_ready:false`, exits non-zero under
              `--for-reporting` / `make contract-reporting`.
Corollary: declaring a weakness honestly (`n_train_blocks: 1`) is a reporting
gate; OMITTING it is a hard fail, because an unevaluable guard is strictly
worse than a declared-weak one.

## C-2. GOVERNANCE: ratification is NOT implementation [v2.6] (ADR-013)
[v2.9] NOW MECHANICALLY ENFORCED: `contract.CLAUSE_IMPLEMENTATIONS` +
`tests/test_clause_registry.py` parse THIS FILE and FAIL THE BUILD on any
unclassified clause; registry entries are verified against a real report so they
cannot be aspirational. This exists because C-2 ITSELF was unimplemented, along
with C-3 and seven others — 9 of 26 clauses, 35%, were fiction (ADR-016).
[v2.9] STRUCTURAL RULE: when adding a clause, ask whether it is checkable by
looking at ONE file. If not, it needs a home that sees more than one.
`check_tensor(one_store)` was the shape of our verification, so every clause
about RELATIONS BETWEEN ARTIFACTS or about PROCESS silently went unwritten —
which is precisely the shape of the bugs no single artifact can reveal.
C1.5's `features must be finite` was ratified at v2.3 and never implemented —
it sat on paper for two contract versions while an all-NaN channel passed all
56 checks and `+inf` was actively BLESSED by two green clauses
(`inf == round(inf)`; an all-inf slab is `array_equal` to itself).
RULE: every accepted clause names its implementer in the DECISIONS entry, and
the next status entry from that lead must confirm it exists in code. A clause
living only in this file is WORSE than no clause — everyone downstream believes
they are protected.
RULE: non-finite / None / empty-comparison are unpassable at the verdict choke
point, so clauses not yet written cannot be silently satisfied.

## C-3. NO THRESHOLD MAY BE CALIBRATED ON n=1 [v2.6] (ADR-013)
Three separate near-misses came from this one habit: C1.6's threshold from one
fire, G2's bar from one held-out block, norm stats from one train fire. Any
constant that decides a pass/fail MUST state the sample it was fitted on, and
that sample MUST span ≥2 spatial blocks. Each instance looked locally
reasonable; that is why this is a rule and not a reminder.

## C-4. CONCURRENCY: what may move while a lead is running [v2.11] (ADR-022)
Written after TWO maintainer failures of the same class, four days apart.
ADR-015: a CV fold changed while modelling was training, silently moving four
fires train→held-out. ADR-021: infra edited `eval/metrics.py` NINE MINUTES into
the run adjudicating G2. The first produced a rule about FOLDS; the second
happened because that rule named the instance instead of the class.

**FROZEN SET — while any lead holds a running experiment, no other lead may
modify: tensors, manifests, `norm_stats.json`, splits/folds, `common/`,
contract code, or SCORING CODE (`eval/`, `common/iou_terms.py`,
`common/calibration.py`, `common/null_check.py`).**
**C-4.3 THE INTERPRETER ENVIRONMENT IS IN THE FROZEN SET TOO** [v2.12] (ADR-024)
— site-packages, the lockfile and system deps. `pip install` during another lead's
run is a shared-state change; the enumerated file list above did not say so.
Extended by data (D5), which declined to install scipy mid-training and
wrote pure-numpy filters instead, reading C-4's INTENT over its text. A lead
extending a rule to a case it does not name, conservatively and with a stated
reason, is the behaviour this contract wants. Bursts touching a shared
code path are SERIALISED by the maintainer, not negotiated between leads.
A lead that needs a frozen file files a BLOCKER and stops.

**C-4.1 OWNERSHIP OF `eval/` IS SPLIT, AND THAT IS NOW WRITTEN DOWN.**
modelling owns `eval/`. infra MAY make edits to `eval/metrics.py` that are
strictly ADDITIVE (new keys, new functions; no existing key's value changes)
and ONLY where a ratified clause names C6 as its implementation site — C6.4 set
this precedent and the maintainer ratified it in ADR-020. Any non-additive
edit is a BLOCKER, never a patch. The additive edit MUST be declared in the
editing lead's status entry in the same session. Rationale: modelling learned
that another owner held part of its module FROM A FILE TIMESTAMP. A reader of the
ownership map and a reader of `eval/metrics.py` disagree about who owns it; folklore
is not an ownership model.

**C-4.2 A CODE FINGERPRINT MUST BE SAMPLED BEFORE *AND* AFTER THE RUN.**
`scoring_code_fingerprint` is stamped once, at payload construction — i.e. at the
END — so the M4 artifact on disk records the hash of a `metrics.py` that was
never executed. **A fingerprint sampled after the fact records the wrong code
precisely in the case it was built to catch.** Sample both ends and hard-fail on
disagreement, as `check_common_code_unchanged` already does for `common/`.
Self-reported by infra against its own instrument; C8 extends to cover it.

## C0. Single-implementation rule [v2.1] (ADR-007)
Anything the contract adjudicates has exactly ONE implementation, in
`common/`, owned by infra. `data/` imports it; it does not re-implement
it. Rationale: the producer and the verifier computing geometry through
different code is how a tensor passes its check and is still wrong.
The `fireline_v2` state rule (C1.1) is contractual and lives in `common/`.

## C1. Per-fire tensor store
Path: data/fires/{fire_id}/tensor.zarr
Grid EPSG:5070, 1000 m cells. Time hourly UTC, monotone, no gaps.

[v2] TWO variables in ONE store (v1 specified a single array that was
"float32 except fire_state uint8" — unsatisfiable, one zarr array has one
dtype):
  fire_state  uint8    (time, y, x)              values {0,1,2}
  features    float32  (time, channel, y, x)     channels 1-13 below
The `channel` coord carries channel NAMES; attr `channel_index_offset: 1`
maps position -> the v1 index. v1 indices are unchanged.

Channel order (fixed; index = v1 position):
 0 fire_state        {0,1,2}   [v2: separate uint8 variable, not in `features`]
 1 wind_u10          m/s   (RTMA)
 2 wind_v10          m/s   (RTMA)
 3 temp_2m           K     (RTMA)
 4 rh_2m             %     (RTMA)
 5 elevation         m     (3DEP, static, repeated over time)
 6 slope             deg   (static)
 7 aspect_sin        —     (static)
 8 aspect_cos        —     (static)
 9 fuel_model_id     int-as-float, FBFM40 class (static; embed in model)
                     [v2] SOURCE = USGS LFPS (lfps.usgs.gov), NOT Earth Engine
                     — FBFM40 is absent from the GEE catalog (ADR-005).
10 canopy_cover      %     (static)  [v2] SOURCE = USGS LFPS, same vintage as 9
11 fuel_moisture_proxy  —  (derived: RTMA-based dead FM estimate; document formula)
12 water_barrier_mask   {0,1} (static; rivers/lakes/roads-as-available)
13 recent_burn_scar     {0,1} (static per fire)
                     [v2.2] HISTORICAL PATH IS MTBS-ONLY. NIFC/WFIGS is
                     YearToDate (current season) and cannot supply within-
                     season scars for a 2019 fire; MTBS as of 2026 already
                     covers 2019-2024. NIFC applies to LIVE nowcasting only.
                     [v2.2] MANDATORY GUARD WINDOW, never a strict inequality:
                     exclude prior events within 14 days before ignition AND
                     drop the fire's own record by incident-name match. MTBS
                     dates are day-resolution and independently sourced, so a
                     fire's own scar can be stamped ~21 h BEFORE its GOFER
                     ignition and sail through `Ig_Date < ignition`. On Kincade
                     that put 86.5% of burned cells into the feature stack as
                     "already burned". See ADR-008 / insights item 13.

### [v2] C1.1 fire_state rule — `fireline_v2` (ADR-006 P1; supersedes the
### provisional rule, which is RETIRED and must not be used anywhere)
    ever(t)       = OR_{s<=t} inside perimeter(s)
    new(t)        = ever(t) and not ever(t-1)
    active(t)     = ever(t) and dilate(cfireLine(t, fconf=0.50), 1 cell)
    burning(t)    = (burning(t-1) or new(t)) and (new(t) or active(t))
    burned_out(t) = ever(t) and not burning(t)
Guarantees, all contract-tested: absorbing (0->1->2, never back), no 0->2
skips, one contiguous burning run per cell.
NOTE for consumers: state 1 is legitimately EMPTY in 6-37% of frames (long
dormancy closes every cell). The contagion source is the FRONTIER OF THE
BURNED REGION, not state 1 alone. Do not condition solely on state 1.
`fconf` has six levels; that set IS the label-perturbation ensemble for the
observation-noise augmentation.

### [v2.3] C1.5 Per-channel declarations are ENFORCED (ADR-009)
- Static channels must really be constant over time.
- `{0,1}` masks (12, 13) must really be binary.
- `fuel_model_id` must be INTEGRAL. A non-integral FBFM40 value means a class
  raster was resampled by interpolation — i.e. the pipeline invented fuel
  models that do not exist.
- `features` must be finite (no NaN/inf).

### [v2.4] C1.7 Physical range — HARD FAIL (ADR-010)
Static physical channels must lie in their definitional range:
  `canopy_cover`   ∈ [0, 100]        (it is a percentage)
  `fuel_model_id`  ∈ the FBFM40 class set (it is an enumeration)
HARD FAIL, unlike C1.6, because this is DEFINITIONAL rather than heuristic:
there is no legitimate value outside these ranges, so a sentinel is always a
bug. Where a domain genuinely lacks source coverage the answer is a documented
FILL POLICY, never a tolerated sentinel.
WHY THIS EXISTS: USGS LFPS returns `-9999` off the LANDFIRE coastline. It is
finite, integral and static, so it satisfies every C1.5 declaration — a CZU
tensor carrying it scored "OK — 42 checks passed (reporting-ready)" with a mean
canopy cover of **-3085%**. The contract checks STRUCTURE, not PLAUSIBILITY.
Fill policy of record: FBFM40 98 (NB8 Open Water, an existing class) and canopy
0, validated against the INDEPENDENT JRC water mask (`nodata_cells_not_water: 0`)
— validate a fill against a different source than the one that produced the hole.

### [v2.6] C1.6 IS DEFERRED — RATIFIED BUT DELIBERATELY NOT SHIPPED (ADR-013)
Its 0.6 threshold was calibrated on Kincade ALONE and flags three CONFORMANT
artifacts: CZU 0.714, Dolan 0.679, C4 synthetic 0.739. Shipping it would cry
wolf until someone disabled it. The checker prints `DEFERRED_CLAUSES` so the gap
is visible rather than silent.
TO SHIP: re-calibrate on ≥4 fires spanning ≥2 spatial blocks, with burnable-cell
masking, threshold justified by the observed distribution — not by one fire.
The clause text below is retained as the specification, NOT as active behaviour.

### [v2.3] C1.6 Leakage smoke alarm — `reporting` severity, NOT a hard fail
No STATIC channel may exceed |2·AUC−1| ≈ 0.6 against the final burn footprint.
Calibration: on the fixed Kincade tensor every static channel scores ≤ 0.244
(max = slope 0.217, physically expected); the channel-13 leak would have scored
MCC ≈ 0.92, ~4× the largest legitimate signal.
THIS IS A HEURISTIC AND A SMOKE ALARM ONLY. It is undefined for a fire whose
genuine prior-scar overlap is large. **The real fix is the C1 ch-13 guard
window, not this check.** Kept at `reporting` deliberately: a false-positive
hard fail on a legitimately scar-heavy fire would teach everyone to disable it.

### [v2.1] C1.4 Axis, monotonicity and attr names (enforced since A1)
- `y` DESCENDS, `x` ASCENDS (north-up). South-up silently mirrors every fire.
- `fire_state` is NON-DECREASING in time (cheapest detector of a broken
  perimeter rasterisation).
- Exact time attr names: `time_start_utc`, `time_end_utc` (naive-UTC ISO).

### [v2] C1.2 Extent and lattice
Per-fire domain = final-perimeter bbox buffered 10 km, snapped to a single
continental EPSG:5070 lattice so cell (i,j) means the same ground in every
fire. Canonical cell-size attr name: `cell_size_m`.

### [v2] C1.3 Time convention — READ THIS, it is silently catastrophic
GOFER `tUTC` is **end-of-hour**. RTMA must be lagged one hour to match.
Getting this wrong trains every fire one hour out of phase with its weather
and presents as a mediocre model, not as a bug. Time attrs are naive-UTC ISO
strings; the store records `time_convention: "end_of_hour"`.

## C2. Per-fire manifest
Path: data/fires/{fire_id}/manifest.json
Keys: fire_id, gofer_version, bbox_5070, ignition_time_utc, n_hours,
cv_fold (int), created_utc, provenance (dict of source → pull date),
norm_stats_path.
[v2] ADD `spatial_block_id` (int) — see C3.1.
[v2.1] Extra keys are PERMITTED (superset); synthetic and data both rely
on this. Stated rather than assumed.
[v2.7] ADD `fuel_vintage_lag_years` (int, machine-readable) — LFPS publishes NO
LF2020 product, so 2021 fires carry 5-YEAR and 2022 fires 6-YEAR stale fuels,
not the 1-2 the design assumed. Vintage still precedes ignition (no leakage),
but the MTBS correction is doing far more work than designed. Any result
spanning 2021+ fires MUST state the lag.
[v2.7] ADD `n_ignition_components` (int), DERIVED not defaulted, with method +
per-fire evidence in `provenance.ignition_components`. ROOT is canonical.
GOFER files SEPARATE LIGHTNING IGNITIONS UNDER ONE FIRE ID. That is not
spotting, it is two fires in a trenchcoat, and no contagion kernel can or
should reproduce it.
[v2.10 CORRECTION, ADR-019] Earlier text here said "SCU has 3". WRONG, and the
ESTIMAND was wrong, not just the value: that count used FINAL-FOOTPRINT
components, which CANNOT SEE MERGES. **SCU = 2 ignitions** (29.27 km apart in
the first burned frame, later merging) **PLUS 2 SPOT CANDIDATES** (5-6 km bodies
born h27/h89). `2020_july_complex` = 2 ignitions; its two first-frame bodies at
2.24 km are ONE ignition split by a one-cell rasterisation hole, and its real
second ignition is the 46.10 km h22 body.
BINDING ON P3: mining MUST distinguish SEPARATE IGNITIONS (filing artifact,
excluded) from SPOT EVENTS (real signal, RETAINED). Distance alone does not
separate them — use time, then genealogy, then distance as tiebreak. Excluding
all inter-component jumps would DELETE the signal G4 depends on.
**G4 SCOPE WARNING: the whole 12-fire corpus holds TWO never-merging spot
candidates**, against hundreds of 2.0-2.24 km rasterisation holes. G4 is not
adjudicable at n=2; growing the corpus (P2 GOFER extension) is on G4's
critical path.
[v2] `provenance` MUST record the per-fire LANDFIRE vintage (ADR-005) and the
state rule + `fconf` used.

## C3. Normalization
Path: data/norm_stats.json — per-channel mean/std computed over TRAIN folds
only. Models must consume stats from this file, never recompute inline.
[v2.1] MUST carry `train_folds` (makes "TRAIN folds only" auditable) and
`std > 0` for every channel (a constant channel otherwise NaNs every
normalisation).

### [v2.2] C3.2 File shape (was unspecified; two leads guessed differently)
CANONICAL shape = TOP-LEVEL dicts `channel_order`, `mean`, `std` — the shape
`common.contract` expects. Per C0 the adjudicator's shape wins. No nested
`channels` block.
Channels 0 (`fire_state`) and 9 (`fuel_model_id`) are CATEGORICAL: emit the
identity transform (mean=0, std=1) plus `categorical_identity_note`.
Standardising an FBFM40 class id is meaningless arithmetic on a label.

### [v2.4] C3.4 Per-fire QA is necessary, NOT sufficient (ADR-010)
Any per-fire defect propagates GLOBALLY through C3 shared norm stats: a fire
can be poisoned by a bug it does not contain. Measured case — CZU's `-9999`
sentinel is 33.1% of train cell-hours, which would have moved the TRAIN mean
canopy to -492.13% from 27.94%, corrupting the normalisation of two held-out
fires with no NoData at all. Norm-stats-level sanity is therefore a SEPARATE
check from per-fire QA, not an aggregate of it.

### [v2.13] C3.5 Declared membership BY ID — disjointness is HARD (ADR-038 (6))
MUST carry `train_fire_ids` and `heldout_fire_ids`. A MISSING key is a FAILURE,
not a skip. Their intersection MUST be EMPTY; a non-empty intersection is HARD
and names the offending ids.
**Why this clause exists, and it is not the reason you would guess.** A check
named `train_heldout_disjoint` ALREADY EXISTED and had been green since it was
written — because it intersected two lists that `split_fingerprint` CONSTRUCTS
as a partition. It was structurally incapable of returning False. The leak it
was supposed to catch (`2020_july_complex`, 9.76% of train mass, ADR-038) was
caught by a human reading a file, not by it.
This is the THIRD verification-that-cannot-fail in this project: an all-NaN
channel passed 56 checks; C1.5 sat unimplemented across two contract versions;
and now this. Fold INDICES are not identifying across corpus versions, so the
membership must be declared BY ID and checked against a source the fingerprint
does not itself build — hence `norm_stats.json`, read independently.
**A check that cannot fail is not a check.** Every clause added from here MUST
ship with a planted defect that it catches.

### [v2.2] C3.3 Bootstrap guard — ENFORCED, not advisory
MUST carry `n_train_blocks` (count of distinct `spatial_block_id` in the train
folds). Any REPORTED result must assert `n_train_blocks >= 2`. A norm-stats
file built when the only fire is also the only train fire satisfies C3's letter
and violates its spirit; it is marked `bootstrap: true` and is valid for
plumbing only, never for a number that appears in a gate.

### [v2] C3.1 Cross-validation is spatially blocked — effective n = 11
Buffered domains overlap: the 28 fires form 11 connected blocks (largest = 8
Klamath/Trinity fires; Kincade, Glass and LNU share a block). Overlapping
fires MUST share a fold — leave-one-fire-out across an overlapping pair is
landscape leakage. Accepted consequence: k=5 fold loads are imbalanced
(2,755-8,095 hours) and cannot be balanced without breaking spatial isolation.
**Effective n is 11, not 28. Do not quote CV spread as if n=28.**

## C4. Synthetic fire generator (the parallelism unlock)
src/wildfire_nowcast/common/synthetic.py
  make_synthetic_fire(seed: int, n_hours: int = 24) -> path to a zarr + manifest
Conforms exactly to C1/C2. Simple wind-driven ellipse growth + noise + one
scripted "river crossing" event so model/viz code exercises all states.
[v2] MUST implement the C1.1 `fireline_v2` state rule and the C1 two-variable
layout. (Follow-up A3.)
[v2.1] Return type is a NamedTuple
`(tensor_path, manifest_path, norm_stats_path, fire_id, geometry)` —
tuple-unpackable, so "path to a zarr + manifest" still reads true.
[v2.1] The synthetic fire MUST script a dormancy so empty-state-1 frames occur
naturally (C1.1 says 6-37% of real frames have none). Do NOT merely relax the
test — the fixture must exercise the phenomenon, not hide it.

## C5. Model prediction API (modelling implements; simviz consumes)
src/wildfire_nowcast/model/api.py
  predict(x0: uint8[H,W], static: f32[C_s,H,W], weather: f32[T,C_w,H,W],
          n_members: int, horizon_h: int, seed: int)
      -> samples: uint8[n_members, horizon_h, H, W]   # fire_state per member
Checkpoint loading behind load_model(path) -> object exposing predict.
Baselines (persistence, ellipse) implement the SAME signature.

## C6. Metrics API (modelling implements in eval/; everyone consumes)
src/wildfire_nowcast/eval/metrics.py
  evaluate(samples, truth: uint8[T,H,W]) -> dict with keys:
    brier_{1,2,3}h, reliability_bins (list), arrival_crps,
    dispersion_ratio, best_member_iou
Model-agnostic: operates only on samples + truth, never on model internals.

### [v2.10] C6.0 EVERY METRIC MUST BEAT A DO-NOTHING NULL (ADR-017)
Before ANY metric adjudicates ANY gate, score a null model that predicts
nothing. **If the null wins, the metric is broken — not the model.**
THREE pathologies of this exact shape have now been found:
 - Brier-fitting drove the ellipse to optimal SILENCE (ADR-011)
 - `dispersion_ratio` scores a COLLAPSED ensemble at 1.000 (ADR-011)
 - `best_member_iou` banks empty-vs-empty as IoU 1.0 (ADR-017)
Our scoring rules systematically reward NOT PREDICTING. This is R14's
persistence attractor reappearing in the INSTRUMENTS rather than in the model.

### [v2.10] C6.4 `best_member_iou` — REPORTED, never a gate criterion
Measured: null floor (persistence, ignites ZERO cells) 0.464/0.326/0.219 at
1/2/3 h; `kernel_m3` 0.351/0.264/0.217 — BELOW doing nothing at every horizon;
`kernel_init` UNTRAINED beats the trained kernel; the degenerate Brier-fit
ellipse ties the floor exactly. Cause: averaging over leads where 21.9% have
zero truth growth, and empty-vs-empty = IoU 1.0.
MUST be decomposed into a SHAPE term and a SILENCE term. Gates use the shape
term (or an empty-truth-masked variant). The undecomposed value stays REPORTED —
hiding it would hide the pathology.
IMPLEMENTED BY: infra, model-agnostically in C6, BEFORE any G2
re-adjudication (C-2 applies: this clause is fiction until confirmed in code).

### [v2.5] C6.1 Which metric adjudicates which gate (ADR-011)
**G3 IS ADJUDICATED ON `area_dispersion_ratio`, NOT `dispersion_ratio`.**
`dispersion_ratio` is ANTI-CORRELATED with collapse: measured, it scores the
COLLAPSED ensemble at exactly 1.000 and the healthy one at 1.051, so the
original "dispersion ratio in [0.8,1.2]" bar would have PASSED a collapsed
ensemble. Ensemble collapse is this project's central claim; adjudicating it
with a blind instrument, in the direction that flatters us, is the failure mode
this clause exists to prevent. `dispersion_ratio` remains a REPORTED
diagnostic and is never a gate criterion. `area_dispersion_ratio` separates
collapsed from healthy by ~106×.

### [v2.5] C6.2 Baseline validity — a degenerate baseline VOIDS its gate
The wind-advected ellipse is a PHYSICS baseline: its scale is CALIBRATED TO
REPRODUCE OBSERVED MEAN HOURLY GROWTH ON TRAIN FIRES, never fitted by pixelwise
Brier. Measured reason: precision on new cells is 0.09–0.43, always < 0.5, so a
hard 0/1 Brier makes a sub-coin-flip predictor optimally SILENT — the fitted
ellipse ignited ZERO cells while truth grew 782. Brier-fitting does not weaken
this baseline, it CONVERTS IT INTO PERSISTENCE, and persistence is already a
separate baseline.
BINDING: if a baseline ignites zero cells on the held-out set it is not a
distinct baseline, and any gate resting on beating it is **VOID, not passed**.

[v2.8] PER-HORIZON CALIBRATION (ADR-015). The ellipse's growth is super-linear
in horizon (0.82 at 1 h → 5.81 at 6 h) while the labels' is linear, so the
calibration horizon is a free parameter worth ~4.7× in over-prediction ratio.
**Calibrate the ellipse SEPARATELY AT EACH EVALUATION HORIZON on train fires;
the model must beat the ellipse's own best-calibrated form AT THAT HORIZON.**
We do not get to pick the horizon where our opponent is weakest. Report 1/2/3 h.
The barred Brier fit is retained as a CONTROL (still degenerate, 0.005×).

## C8. Split fingerprint [v2.8] (ADR-015) — HARD FAIL on mismatch
Every run stamps `split_fingerprint`; every reported number carries it. A
mismatch between the split used for TRAINING and the split used for EVALUATION
is a HARD FAIL.
WHY: the CV split moved mid-task (`train_folds [0,1,3]` → `[0,1,2,4]`) and four
fires silently crossed from train to held-out. **No per-tensor check could see
it** — every tensor was individually conformant throughout. This is the first
hazard created by PARALLELISM itself rather than by any lead's work.
MAINTAINER RULE: no fold change is authorised while any lead is training.

### [v2.14] C6.5 G3's bar is GEOMETRIC and carries a FIRST-MOMENT condition
The dispersion bar is `|log(adr)| <= log(1.2)`, i.e. `[0.8333, 1.2]`.
**The old `[0.8, 1.2]` was NOT symmetric:** in the bar's own log units it was
**22% more tolerant of UNDER-dispersion**, which is the only side this project
has ever failed on. It was flattering the candidate.
G3 additionally requires a **first-moment condition**: the candidate's
`|log(growth_calibration)|`, on the same held-out blocks under equal-block
weighting, MUST be no worse than the best physics baseline's. **Defined against
a REFERENCE, never as an absolute number**, so no threshold is fitted to any
arm's result (C-3). Ratios pool in LOG space — pooling a ratio arithmetically
reintroduces one level down the exact asymmetry this clause removes.
**G3 passes only if BOTH hold**, and both are always reported together: the
identity `adr = sqrt((M+1)/M) x ensemble_CV x growth_calibration x truth_shape
x relief` (residual 2.2e-16) means dispersion can otherwise be satisfied by
COMPENSATING first- and second-moment errors.
`area_dispersion_ratio` is **UNDEFINED at perfect mean calibration** — its
denominator is the model's own mean-area error, exactly 0 there — so the metric
goes undefined precisely as a model gets the first moment right. UNDEFINED is
its own outcome and is **NEVER a pass**.
IMPLEMENTED BY: `common/dispersion.py`, CALLED BY `eval/baseline_run.py`.

### [v2.15] C6.6 FOUR METRICS MAY BE REPORTED AND MAY NOT DECIDE A GATE
Brier (`brier_{1,2,3}h`), arrival-time CRPS (`arrival_crps`), `calibration_error`
(`calibration_error_{1,2,3}h`) and reliability (`reliability_{1,2,3}h`) are
**NON-ADJUDICATING**. They may appear in any report, table or figure. They may
not pass, fail, or void anything.
**THE MEASUREMENT THAT DISQUALIFIED THEM (ADR-053 (1)(2)).** M11's degradation
ladder — a forecast degraded across **0.053x to 8.0x area error**, scored on
**n=5 held-out blocks** — measured each channel's Spearman correlation against
`|log(area error)|`, i.e. against how wrong the forecast actually was:
```
  Brier               -0.45      arrival CRPS        -0.34
  calibration_error   -0.14      reliability         -0.80
```
Every one is the WRONG SIGN: on these channels a worse forecast scores better.
Concretely, **a forecast 40x TOO SMALL beats a 3%-correct one by 17% on Brier**,
and none of the four has a minimum detectable effect ANYWHERE on the ladder
(plateau 0.66-1.58 across the whole 0.053x-8.0x range). M10's +0.80 separation
was the CEILING of what these channels can express, not a near-miss of the 2.0
bar.
**THE ONLY CHANNELS THAT MAY ADJUDICATE TODAY ARE:**
```
  area_dispersion_ratio          (dispersion; MDE 1.021x)
  growth_calibration             (C6.5's first moment; MDE 1.056x, DEGENERATE on
                                  an AREA ladder because it IS the perturbed
                                  quantity there)
  best_member_iou_shape_masked   (the only LOCATION-sensitive channel; MDE 13.0x,
                                  underpowered rather than blind)
```
This clause SUBTRACTS and never adds: it can only stop a gate from being passed,
never let one be passed. That is why it is the single bump permitted while the
contract is otherwise frozen (ADR-054).
**IMPLEMENTED BY** `common/null_check/registry.py`: `MetricSpec.gate_eligible`
carries the ruling per channel, `adjudicating_metrics()` DERIVES the permitted
set from those flags rather than restating it, and `assert_may_adjudicate()`
**RAISES `NonAdjudicatingMetricError`** — it does not warn. The failure being
repaired is a human quoting a number while forgetting a ruling made two weeks
earlier, and a warning is read by exactly the person who already forgot.
The flag is already load-bearing: `MetricVerdict.is_failure` and
`is_reporting_gap` read `gate_eligible`, so C6.0's harness (`make null-check`,
run by `make ci`) changes tier for these four channels at this bump.
An UNREGISTERED metric raises as well — an unknown channel is not a permitted
one (C-2 one level down).

### [v2.14] C6.3 (addition) A block contributing NOTHING is a HARD FAIL
A block that contributes nothing to an equal-block mean is a HARD failure, not
a silently smaller sample. The pooled result MUST carry the number of blocks
that actually contributed. A caller that legitimately expects gaps opts in
explicitly, per call. Silent partial coverage reads as full coverage.
IMPLEMENTED BY: `common/pooling.equal_block_mean(..., allow_missing_blocks=False)`
raising `IncompleteBlockCoverageError`; the local copy in `eval/` is DELETED (C0).

### [v2.5] C6.3 Reporting requires spatial-block coverage
Every reported result states the number of DISTINCT HELD-OUT SPATIAL BLOCKS,
not the number of held-out fires. No CV spread may be quoted at n=1 block.
G2 requires ≥4 distinct held-out blocks. More fires from the same block are the
same evidence with false confidence, not more evidence.

## C7. Config
configs/*.yaml, one experiment = one yaml. Every training run logs its resolved
config + git SHA into runs/{run_id}/config.yaml. No hardcoded paths in src/.
[v2] No hardcoded GCP project id either — read `WILDFIRE_GEE_PROJECT` (ADR-003).
GEE credentials hold `earthengine,cloud-platform` scopes ONLY: `Export.*.toDrive`
is NOT authorized and GCS is forbidden. Use synchronous chunked fetch (ADR-004).
