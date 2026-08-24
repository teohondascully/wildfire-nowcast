# Where a test goes, and how to find the one you want

This suite had 29 files against 119 source modules and only 9 modules with a
same-named test. The cost of that is not a coverage number. It is that a reader
who wants to know what is asserted about `common/grid.py` has to grep the whole
suite, and a lead adding a module has no place the next reader will look. The
mutation survivors cluster in exactly the packages with no name-based route from
a module to its test, which is the measurement behind this layout.

## The three tiers

```
tests/unit/<package>/test_<module>.py    one file per source module, same tree shape
tests/contract/test_<clause>.py          one file per contract clause family
tests/test_<subject>.py                  cross-cutting: hygiene, tooling, playthroughs
```

### 1. `tests/unit/` mirrors the source tree

`src/wildfire_nowcast/common/null_check/windows.py` is tested by
`tests/unit/common/null_check/test_windows.py`. The rule is mechanical and has no
exceptions: prefix `test_` onto the module's own file name, and put it in a
directory that mirrors the module's own directory. `__main__.py` therefore maps to
`test___main__.py`, which is ugly and is kept anyway, because a convention with an
exception in it is a convention nobody can check.

Every directory under `tests/unit/` carries an `__init__.py`, and those files are
load-bearing rather than tidiness. Under pytest's default `prepend` import mode
two test files with the same basename and no package between them collide at
import and the whole suite errors, so `tests/unit/common/test_states.py` and
`tests/test_states.py` can only coexist because both directories are packages.

`common/` is the worked example and the only package under the rule today.
`model/`, `eval/` and `sim/` belong to other leads. The convention is written here
so it can be adopted, not imposed: a lead who wants it creates
`tests/unit/<their package>/`, adds the `__init__.py`, and extends the check in
`tests/test_hygiene.py` with their own burn-down list.

### 2. `tests/contract/` mirrors the contract clauses

Proposed, not yet built. `docs/interfaces.md` is the numbered contract, and the
tests that enforce it are currently spread across `tests/test_contracts.py`,
`tests/test_clause_registry.py`, `tests/test_null_check.py` and others, so there
is no route from a clause number to the code that enforces it either. The shape
that fixes it is one file per clause family:

```
tests/contract/test_c1_tensor.py       C1: dims, channel order, dtypes, CRS, hourly time
tests/contract/test_c2_manifest.py     C2: provenance keys, spatial_block_id, ignition components
tests/contract/test_c3_norm_stats.py   C3: shared normalisation, n_train_blocks, atomicity
tests/contract/test_c6_metrics.py      C6: the metric registry, what may adjudicate
tests/contract/test_c8_splits.py       C8: the split fingerprint and what it binds
```

This has NOT been done because moving `tests/test_contracts.py` is outside infra's
file allocation for this task and three leads are editing the tree in parallel.
It is a serialised move, not a refactor to do while others are running.

### 3. Top-level files stay where they are

A test whose subject is not one module and not one clause stays at the top level:
`test_hygiene.py` (repository rules), `test_lockfile.py`, `test_ci_status.py`,
`test_push_guard.py`, the `test_playthrough_*.py` family. These are named after
what they check rather than after a file, and mirroring a source tree would put
them somewhere arbitrary.

## The burn-down list

`tests/test_hygiene.py` holds `_COMMON_MODULES_WITHOUT_A_MIRRORING_TEST`. It
fails in both directions, like the mypy exemption list and the mutation survivor
set: a module with no mirror that is not listed turns it red, and a module that
gains a mirror while still listed turns it red too. So an entry cannot outlive its
reason and the list can only shrink through a visible edit.

To retire an entry: write `tests/unit/common/test_<module>.py`, delete the module
from the frozenset, and commit both together.

## Running one

```
make test                       # the suite, minus `slow`
.venv/bin/pytest tests/unit/common -q
.venv/bin/pytest tests/unit/common/test_grid.py::test_padding_grows_the_domain_outward_on_all_four_sides
```

Use `make test`, not a bare `pytest`, unless you know why. The Makefile sets
`PYTHONDONTWRITEBYTECODE=1` and purges `__pycache__` first, and both halves are
load-bearing: CPython invalidates a `.pyc` on source mtime in WHOLE SECONDS plus
size, so a same-length edit made and reverted inside one second is invisible and
the stale bytecode runs instead. A file that verifiably reads `max(a, b)` then
executes `min(a, b)`. Anyone planting a defect to check that a test catches it is
in that hazard's blast radius, and the way out is to read the value back through
the interpreter that will run the test rather than to trust the file write.

## What a test here is expected to do

* Name the defect it catches, in its docstring, in terms of what goes wrong in a
  result rather than in terms of a line of code.
* Carry its own control where an assertion could pass vacuously. A scan that
  matches nothing and a check that cannot fail are this project's most repeated
  false negatives, and several tests here fail on purpose if their fixture stops
  being able to separate the two cases.
* Never be written to kill a mutant that cannot be killed. If a mutation sweep
  survivor is provably equivalent, prove it and record it in
  `EQUIVALENT_MUTANTS` in `tools/mutation.py` with the node id of the proof. A
  test written to make a survivor go away is a test that cannot fail, which is
  the defect this repository has catalogued nine times.
