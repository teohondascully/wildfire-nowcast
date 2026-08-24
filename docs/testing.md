# testing

how the gates work, which one adjudicates, and what a new check has to do before
it counts. where a test file goes is a different question and is answered in
`tests/README.md`; this file is about running them and about adding a check that
can actually fail.

## contents

- [the layers](#the-layers)
- [which target produced the number](#which-target-produced-the-number)
- [the non-pytest gates](#the-non-pytest-gates)
- [git add is an input to this suite](#git-add-is-an-input-to-this-suite)
- [what a new check has to do](#what-a-new-check-has-to-do)
- [pins and burn-down lists](#pins-and-burn-down-lists)
- [what CI does not cover](#what-ci-does-not-cover)

## the layers

```
make test           the suite minus `slow`. the fast local loop
make test-all       the whole suite. this is what CI runs
make test-isolated  the suite in a detached worktree at HEAD
make check          lint + typecheck + test-all + playthrough
make ci             check, plus synth, contract and null-check. THE gate
make ci-status      ask GitHub about the published head, not this machine
```

`make test` deselects the `slow` marker. that is one case today: the ELMFIRE
playthrough runs a real Fortran simulator several times. nothing is hidden in a
pytest default, so a bare `pytest` still runs everything.

`make test-isolated` exists because several people and several plant-and-revert
protocols share one working tree. a plant deliberately breaks a file for a few
seconds; a full-suite run inside that window measures the plant, and the tree is
clean again by the time anyone looks at it. isolation cannot be achieved by being
careful, because the reader cannot know the window was open. use it for any
number you are going to report.

## which target produced the number

a suite count is meaningless without the target that produced it, because the
targets do not run the same set. state both. three measurements of the same
commit, `169c6f2`, taken the same evening:

```
make test-all      in a working copy with the corpus     1476 passed,  0 skipped
make test-isolated in a detached worktree, no corpus     1460 passed, 16 skipped
```

they reconcile exactly, 1460 + 16 = 1476, and the 16 are the data-dependent cases
that skip on a machine with no built fires; each prints its own reason. neither
number is wrong and neither can be quoted as the other. `make test` reports a
smaller number again, because it deselects `slow`.

quoting a weaker gate's figure in support of a stronger claim has gone wrong here
before, which is why the rule is written down rather than assumed.

and a green local gate is a claim about a working copy, not about the
repository. `make ci-status` is the other claim: it asks GitHub for the
conclusion of the run that built the exact commit on `origin/main`, prints
whether your tree is dirty or unpushed, and exits non-zero when it cannot tell.
unknown is not green.

## the non-pytest gates

```
make lint         ruff check AND ruff format --check, over src tests tools runs
make typecheck    mypy --strict over src and tools, plus an exemption audit
make prose        typographic punctuation in the tracked tree, by region
make synth        generate one synthetic fire
make contract     judge any tensor.zarr against C1 to C3
make null-check   score a do-nothing forecaster against every C6 metric
make playthrough  the playthrough registry, including the slow arm
make mutation     the survivor budget over common/ and eval/. NOT in `make ci`
```

two of these have a trap in the name. `make lint` runs **both** halves, `check`
and `format --check`; a report that says "lint clean" from anything narrower is a
weaker claim than it reads. and `make format` is the writing form: it rewrites
files, which moves the code fingerprint every run artifact stamps at both ends,
so it is scoped to your own directory and is never what a gate runs.

`make mutation` is deliberately outside `make ci`. it was measured, not
estimated: one sweep took 110.4 minutes against a full suite of about four,
which is more than an order of magnitude on every push. the comparison it makes
is still gated, because the pure verdict function is exercised in both
directions inside the suite. only the value compared against is not.

## git add is an input to this suite

sixteen tracked files decide what to check by enumerating the git index, through
`git ls-files`. for every one of them, staging a file is a semantic change to
the test run, not bookkeeping.

the failure mode is sharp and has happened: a suite was run, reported green,
then two files were staged and committed, and CI went red on a count pin that
nothing in the source had touched. both statements were true of different trees.

so: **run the gate after staging, and run `make test-all`, not `make test`.**
adding or removing a tracked file, especially one that cites a path or a `runs/`
artifact, will move a pin. that is the pin working.

when a count pin fails, the first question is not which number to change. it is
whether the thing being counted is supposed to exist. a pin is an alarm, and the
two ways to silence an alarm are to fix the fault and to cut the wire.

## what a new check has to do

three requirements, all of which exist because a check here failed to meet them.

**name the defect it catches**, in its docstring, in terms of what goes wrong in
a result rather than in terms of a line of code.

**carry its own control.** an assertion that could pass vacuously needs a probe
that shows the instrument can still discriminate. a scan that matches nothing and
a check that cannot fail are this project's most repeated false negatives. a
control that returns the same value as the subject is not a control: a heading
scan was once verified against a file that also had zero headings, which proved
nothing about the scan.

**plant the defect and watch it go red.** this is the one that keeps paying. a
guard whose only evidence is that it was added is not evidence. a pipefail guard
here was specified in a form that macOS make silently ignores, and it was caught
only because the plant was run instead of the prescription being trusted. plant
in both directions where the pin is two-sided.

never write a test to kill a mutation survivor that cannot be killed. if a
survivor is provably equivalent, prove it and record it with the node id of the
proof; a test written to make a survivor go away is a test that cannot fail.

## pins and burn-down lists

several checks are pinned to an exact number rather than to an upper bound, and
they fail in both directions on purpose:

* the mypy exemption list. a listed module that has become clean fails the build
  until it is retired, so the list cannot become a permanent excuse.
* the typographic-punctuation counts, one for docstrings and comments and one for
  tracked prose files.
* the tell scan's burn-down, per file.
* the mutation survivor budget, which may be lowered by a commit that kills one
  and never raised.
* the `runs/` citation class, which fails when a citation stops being tracked and
  when a new one appears unaccounted for.

a pin larger than the debt is an allowance for the next one, which is why the
"good news, lower it in this commit" direction is an assertion rather than a
comment.

## what CI does not cover

written down in `.github/workflows/ci.yml` rather than left to be discovered.
the fire corpus is not in the repository, so the cross-fire split clauses run
against a synthetic corpus there; ELMFIRE is not built; and Earth Engine
ingestion needs credentials. the tests are not themselves type-checked yet, and
that gap is recorded in `pyproject.toml` rather than hidden behind a relaxed
setting.
