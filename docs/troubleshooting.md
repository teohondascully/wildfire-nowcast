# troubleshooting

failures this repository actually produces, what each one means, and what to do.
every entry below is a real failure mode of the tree as it stands, not a generic
checklist. if a symptom is not here, the honest answer is that it has not
happened yet.

## contents

- [a gate is green locally and red in CI](#a-gate-is-green-locally-and-red-in-ci)
- [a count pin failed after git add](#a-count-pin-failed-after-git-add)
- [a path check fails on a file you have not staged](#a-path-check-fails-on-a-file-you-have-not-staged)
- [a file verifiably says one thing and does another](#a-file-verifiably-says-one-thing-and-does-another)
- [make lint passed and the commit was still refused](#make-lint-passed-and-the-commit-was-still-refused)
- [the contract version fails at import](#the-contract-version-fails-at-import)
- [a typecheck exemption turns the build red for being clean](#a-typecheck-exemption-turns-the-build-red-for-being-clean)
- [a test result changed and nothing was edited](#a-test-result-changed-and-nothing-was-edited)
- [ingestion refuses to run](#ingestion-refuses-to-run)
- [ELMFIRE work skips or aborts](#elmfire-work-skips-or-aborts)
- [disk filled up, or worktrees are left behind](#disk-filled-up-or-worktrees-are-left-behind)
- [ci-status will not answer](#ci-status-will-not-answer)

## a gate is green locally and red in CI

first check which target you ran. `make test` deselects `slow`; CI runs
`make ci`, which reaches `make test-all`, which is a bare `pytest`. a figure from
the weaker selector is not evidence about the gate.

second check whether your `.venv` matches the lock. `make install` runs
`uv pip sync --require-hashes requirements.lock`, which removes anything not in
the lock. an environment that was built with a plain install can differ from the
runner by a patch version, and a single overload disagreement between two numpy
patch releases once kept the badge red for seven days while the local typecheck
exited 0.

## a count pin failed after git add

expected, and it is the pin working. sixteen tracked files decide what to check
by enumerating the git index, so staging a file changes what the suite measures.

do not start by changing the number. ask first whether the thing being counted is
supposed to exist. if a newly tracked file cites a path or a `runs/` artifact,
the pin moved because a real citation entered the class; if it cites something
incidental, the citation is what should go. a pin is an alarm, and the two ways
to silence an alarm are to fix the fault and to cut the wire.

then re-run the gate **after** staging. a green run taken before `git add`
answers a question about a tree nobody has.

## a path check fails on a file you have not staged

the mirror image of the entry above, and it bites in the middle of ordinary work
rather than at commit time. the path scanner reads the CONTENTS of tracked files
from your working tree, and resolves what they cite against the git INDEX. so the
moment you add a reference to a brand new module inside a file that is already
tracked, and before you stage the new module, the scanner reports that a tracked
file cites a path a cloner cannot open, and the suite goes red.

it is a real defect for exactly as long as the new file is unstaged: a commit of
the citing file alone would publish a dangling reference. staging the new module
clears it. observed live in this tree, and cleared within minutes, while these
docs were being written.

## a file verifiably says one thing and does another

stale bytecode. CPython invalidates a `.pyc` on the source's mtime in **whole
seconds** plus its size, so a same-length edit made and reverted inside one
wall-clock second is invisible and the cached module runs instead. a file that
reads `max(a, b)` can execute `min(a, b)`.

this is why the Makefile sets `PYTHONDONTWRITEBYTECODE=1` and purges
`__pycache__` before the suite rather than trusting it to be fresh, and it is
why you should run tests through `make` rather than a bare `pytest` unless you
know why. anyone planting a defect to check that a test catches it is directly in
this hazard's blast radius: read the value back through the interpreter that will
run the test rather than trusting the file write.

## make lint passed and the commit was still refused

`make lint` runs both halves, `ruff check` and `ruff format --check`. if a hook
refused a commit that `make lint` accepted, the two were run against different
scopes, so compare what each covered. do not quote a gate's name as its coverage.

to fix formatting, run `make format DIRS=<your directory>`. the writing form is
deliberately never what a gate runs: rewriting source during a gate run moves the
code fingerprint that every run artifact stamps at both ends, which turns a
harmless reformat into a provenance change no reader of that stamp can
distinguish from a real one.

## the contract version fails at import

`common/contract.py` parses `CONTRACT_VERSION` from **line 1** of
`docs/interfaces.md` and has no literal fallback. if that line is unreadable or
has been reformatted, importing the package raises.

this is deliberate. the version had drifted between the document and the code
four times, and a fallback is precisely how that drift stays hidden. the fix is
to repair line 1, never to add a default.

## a typecheck exemption turns the build red for being clean

`make typecheck` runs mypy and then re-runs it with every exemption removed. a
module on the exemption list that has become clean fails the build until the
entry is retired. that direction is not a bug: an exemption list that only ever
checks its ceiling turns into a permanent excuse.

the same shape appears in the typographic-punctuation pins and in the mutation
survivor budget. the message will tell you which direction moved and what to
edit.

## a test result changed and nothing was edited

if several people or several processes share this working tree, a full-suite run
can land inside somebody's plant-and-revert window and measure the plant. the
tree is clean again by the time you look, so the evidence is gone.

`make test-isolated` runs the suite in a detached worktree at HEAD and is immune
to this. use it for any number you are going to report.

## ingestion refuses to run

Earth Engine ingestion needs credentials and a registered Cloud project, and the
project id is read from the `WILDFIRE_GEE_PROJECT` environment variable rather
than hardcoded anywhere. the data package ships a probe that reports separately
whether the environment variable is set and whether the credentials file is
present, so those two are answered rather than guessed at. the third requirement
is per-project Earth Engine registration, which the probe cannot see directly and
which is the step most often missed: enabling the API is not the same as
registering the project.

none of this blocks the gate. the corpus is not in the repository and the whole
of `make ci` runs on a clean clone, because the data-shaped inputs it needs are
synthesised by C4.

## ELMFIRE work skips or aborts

ELMFIRE is a vendored Fortran simulator that is not built by this repository, so
its playthrough skips on a clean checkout. the skip is declared and a test
asserts that it is declared, so it cannot become silent.

if a run is missing a fire rather than skipping, note that the simulator's
runtime cap is wall-clock and its abort is silent. a missing block is a
reportable property of the tool, not of the fire, and should be recorded as one.

## disk filled up, or worktrees are left behind

`make mutation` applies each mutant in a git worktree rather than in your working
copy, and it cleans up in a `finally`. `SIGKILL` does not run a `finally`, so a
sweep that is killed rather than interrupted leaves worktrees and a temp root
behind. one killed sweep here left four registered worktrees and a 43 MB root.

check with `git worktree list` and reclaim with `git worktree remove --force`
followed by `git worktree prune`. the sweep also reports leaked temp entries, and
in a shared tree that count measures the tree rather than the process, so treat
it as a prompt to look rather than as an attribution.

`make clean-outputs` removes the generated `outputs/` directory.

## ci-status will not answer

`make ci-status` exits 0 for green, 1 for red and **3 when it cannot tell**, and
the third case is the one that matters. unknown is not green.

it answers about one commit. filter to the sha you mean; a row for any other sha
is not evidence however green it is, and no matching row means no run yet rather
than "the top row must be it". a stale poll once returned a truthful green for a
commit eleven behind HEAD, which satisfied every rule then in force and
adjudicated the wrong tree. the tool now prints the queried commit's subject and
its distance from HEAD, and tags every HEAD-relative line, for exactly that
reason.

never pipe it. under the default shell a pipeline reports the exit status of its
last element, so a gate that rejected its own argument and did nothing at all
reads as a pass. the Makefile pins `SHELL` with `pipefail` for this, and the flag
is on `SHELL` rather than on `.SHELLFLAGS` because `.SHELLFLAGS` arrived in GNU
make 3.82 and macOS ships 3.81, which parses that assignment and never reads it.
