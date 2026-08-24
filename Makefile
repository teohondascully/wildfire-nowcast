# wildfire-nowcast - developer tasks.
#
# Every target runs against the repo-root .venv (ADR-001: uv, CPython 3.12).
# Never invoke bare `python`/`pytest`/`ruff` - they may be system Python.

# SHELL AND .SHELLFLAGS ARE A GUARD, NOT A STYLE CHOICE (ADR-149).
# make's default is `/bin/sh -c`, and a pipeline's exit status there is its LAST
# element. So `<gate> | tail -5` reports tail's success: a gate that rejected its
# own arguments, made no network call and did nothing at all still reads as a
# pass. That is not hypothetical - it is how a verdict was taken here, and the
# reader of `$?` had no way to see it. No recipe in this file pipes a gate today,
# so this is prophylactic; the hazard is one edit away in every recipe, and the
# edit that introduces it looks like a formatting change.
# bash rather than /bin/sh because pipefail is NOT POSIX: /bin/sh is dash on the
# ubuntu-latest runner and `set -o pipefail` is an error there, so pinning the
# interpreter is what makes the flag portable rather than locally true.
#
# THE FLAG IS ON `SHELL`, NOT ON `.SHELLFLAGS`, AND THAT IS THE WHOLE POINT.
# `.SHELLFLAGS` arrived in GNU make 3.82. macOS ships **3.81**, which parses the
# assignment, stores it and NEVER READS IT. Written the obvious way
# (`SHELL := /bin/bash` + `.SHELLFLAGS := -o pipefail -c`) this guard is live on
# the CI runner and INERT on the machine every lead actually works on: measured
# here, `false | tail -1` exited **0** under that spelling and **2** under this
# one. A prophylactic that is a no-op exactly where the human is typing is the
# defect class this repository keeps re-finding, and it would have shipped had
# the guard not been watched catching something.
# make splits a multi-word SHELL in both versions, so this spelling is honoured
# by 3.81 and by 4.x. `.SHELLFLAGS := -c` is set anyway - it is 3.82+'s default,
# it is what 3.81 hard-codes, and pinning it stops a later edit there from
# silently dropping the `-c` that makes the recipe a command.
#
# DELIBERATELY NOT `-e`: errexit changes the meaning of the multi-command recipes
# below - `contract-all-fires` runs a loop whose per-fire failures are counted
# and reported rather than fatal - and make already checks the status of each
# recipe line. One change, one property.
# `tests/test_ci_matches_makefile.py` EXECUTES a failing pipeline through THIS
# file rather than reading these two lines, because a guard nobody watched catch
# something is the class of check this repository keeps finding in its own past.
SHELL        := /bin/bash -o pipefail
.SHELLFLAGS  := -c

VENV   := .venv
PY     := $(VENV)/bin/python

# PYTHONDONTWRITEBYTECODE IS LOAD-BEARING AND IS NOT A TIDINESS SETTING.
# CPython invalidates a `.pyc` on (source mtime in WHOLE SECONDS, source size),
# so a same-length edit made and reverted inside one wall-clock second is
# INVISIBLE and the stale bytecode runs instead: a file that verifiably reads
# `max(a, b)` executes `min(a, b)`. Reproduced here 2026-08-22, together with
# the whole class it breaks - any by-hand edit-run-revert check, which is how a
# mutant gets recorded as a survivor it never was, and how a red result gets
# recorded against a tree byte-identical to HEAD.
# Nothing here depends on a person remembering to clear a cache. `pytest` is
# invoked through `env` so the setting travels to every target, to `make ci`, and
# to the CI runner, all of which reach the suite through this one variable.
PYTEST := env PYTHONDONTWRITEBYTECODE=1 $(VENV)/bin/pytest
RUFF   := $(VENV)/bin/ruff
UV     ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

PY_VERSION ?= 3.12

# mypy is PINNED and run ISOLATED, not installed into .venv. Pinned because an
# unpinned type checker adds rules on its own schedule and turns CI red on a day
# nobody changed anything. Isolated because C-4.3 puts the interpreter
# environment in the frozen set: a developer tool must not be able to move
# site-packages under another lead's running experiment. `--python-executable`
# below still points mypy at .venv, so it checks against the real numpy/xarray/
# torch rather than against stubs it guessed at.
# `--python $(PY_VERSION)` pins the interpreter mypy ITSELF runs on, which is a
# different thing from the version it type-checks FOR. It is not cosmetic:
# uv defaults to the newest interpreter it can find (3.11.2 on this machine),
# and mypy parses .pyi files with its host's ast, so numpy 2.x's stubs fail to
# parse with "Invalid syntax" and mypy exits 2 with only nine errors reported.
# A checker that stops early looks exactly like a clean tree from the outside.
MYPY_VERSION ?= 2.3.0
MYPY         := $(UV) tool run --python $(PY_VERSION) --from mypy==$(MYPY_VERSION) mypy

# A fire is a directory: tensor.zarr + manifest.json + norm_stats.json together,
# so that C1/C2/C3 can be checked as a unit.
OUT    ?= outputs/synthetic_fire/tensor.zarr
TENSOR ?= outputs/synthetic_fire/tensor.zarr
MOVIE  ?= outputs/fire.mp4

.PHONY: help venv install relock hooks test test-all test-isolated purge-bytecode lint typecheck format-check format synth contract contract-reporting \
        contract-real contract-split contract-all-fires prose null-check check ci ci-status movie clean-outputs \
        playthrough playthrough-list playthrough-dispersion playthrough-off-state \
        playthrough-separation playthrough-harness playthrough-coarsening playthrough-baseline

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## venv: create .venv with uv on the pinned interpreter (ADR-001)
venv:
	$(UV) venv --python $(PY_VERSION) $(VENV)

## install: install EXACTLY `requirements.lock` into .venv, then the project
##         itself, then wire the git hooks.
##         `pip sync`, NOT `pip install`. Two things follow from that word and
##         both are the point: the venv ends up holding the lock's set and
##         NOTHING ELSE (a package someone added by hand is removed, which is
##         C-4.3's frozen-environment clause made mechanical), and no version is
##         ever resolved at install time.
##         WHY THIS CHANGED (I8): until 2026-08-21 this line was
##         `uv pip install -e ".[dev]"` and `requirements.lock` was named by no
##         install path at all -- two pyproject comments, a ruff-pin test and an
##         environment fingerprint, none of which install anything. A clean
##         clone therefore RE-RESOLVED, and measured against the lock it moved
##         15 of 73 packages and dropped a 16th. That is how the same source
##         line was green here on numpy 2.5.1 and red in CI on numpy 2.5.2 for
##         seven days. A lock nobody installs from is worse than no lock,
##         because it advertises a reproducibility that does not exist.
##         `--require-hashes` makes the pins binding on the artifact and not
##         only on the version string. `-e . --no-deps` is separate because an
##         editable path install cannot be hash-checked; its dependencies are
##         already present, and `tests/test_lockfile.py` fails if pyproject
##         declares a dependency the lock does not carry.
install: venv
	VIRTUAL_ENV=$(CURDIR)/$(VENV) $(UV) pip sync --python $(PY) --require-hashes requirements.lock
	VIRTUAL_ENV=$(CURDIR)/$(VENV) $(UV) pip install --python $(PY) --no-deps -e .
	@$(MAKE) --no-print-directory hooks

## relock: regenerate requirements.lock. IT NEVER UPGRADES ANYTHING.
##         The command constrains the resolution BY THE EXISTING LOCK (`-c
##         requirements.lock`), so it is a fixpoint: running it twice on an
##         unchanged tree reproduced the file byte for byte at I8 (sha equal
##         across passes 2 and 3; pass 1 differs only because it converted a
##         73-row file into a 94-row one). That is a MEASUREMENT, not a test --
##         asserting it in the suite would put a network resolve on the gate.
##         It exists to pick up a NEW dependency added to pyproject.toml, not
##         to drift the old ones.
##         TO UPGRADE ONE PACKAGE: delete its pin from requirements.lock, run
##         this, and commit the result with the reason. That makes every version
##         change a deliberate, reviewable line in a diff instead of a side
##         effect of whoever cloned most recently.
##         `--universal` resolves for every platform at once with environment
##         markers, so the macOS dev machine and the Linux CI runner install
##         from ONE file (the 21 CUDA rows are torch's Linux dependencies).
##         `--python-version 3.12` is load-bearing: without it uv bounds the
##         resolution by whatever interpreter it finds, so the same command
##         succeeds in a directory with a .venv and fails in one without.
relock:
	$(UV) pip compile pyproject.toml --extra dev --universal --generate-hashes \
	  --python-version $(PY_VERSION) -c requirements.lock -o requirements.lock

## hooks: install the pre-commit, commit-msg AND pre-push hooks (all three types,
##         see `default_install_hook_types` in .pre-commit-config.yaml), plus the
##         AUTHORITATIVE half of the pre-push publication guard. Run by
##         `make install` rather than left as an instruction: a guard that only
##         works if someone remembers to install it is the guard that failed.
##         The second command is not a duplicate of the first. pre-commit
##         consumes git's pre-push stdin inside its own wrapper and exports only
##         the FIRST ref of the push, so a hook wired through the config alone
##         cannot see `git push --all` -- measured, see tools/push_guard.py.
##         `--install` writes the full-ref-list hook into the slot pre-commit
##         chains to, and exits NON-ZERO if that chain is not in place.
##         Skipped, loudly, outside a git checkout (a source tarball has no hooks
##         to install and that is not an error).
hooks: | $(PY)
	@if [ -d .git ]; then \
	  $(VENV)/bin/pre-commit install; \
	  $(PY) tools/push_guard.py --install; \
	else \
	  echo "not a git checkout: no hooks installed"; \
	fi

# Fail loudly (and usefully) instead of falling back to system Python.
$(PY):
	@echo "ERROR: $(PY) not found. Run 'make install' first (ADR-001)." >&2
	@exit 1

## test: run the test suite (deselects `slow`; see `make test-all`)
##         `slow` is one case today: the ELMFIRE playthrough runs a real Fortran
##         simulator six times (33 s). Nothing is hidden in a pytest default -
##         bare `pytest` still runs everything, and `make check` runs it too.
test: purge-bytecode | $(PY)
	$(PYTEST) -m "not slow"

## test-all: the whole suite INCLUDING slow playthroughs. What a release runs.
test-all: purge-bytecode | $(PY)
	$(PYTEST)

## test-isolated: the suite in a detached worktree at HEAD, immune to neighbours.
##         Four leads share this tree and three run plant-and-revert protocols, in
##         which a file is deliberately broken for a few seconds. A full-suite run
##         inside that window measures the plant, and `git status` is clean again
##         by the time anyone looks. Two leads hit this independently before it was
##         understood. Isolation cannot be achieved by being careful, because the
##         reader cannot know the window was open - so it is a target.
##         Use it for any number you are going to REPORT. `make test` is still the
##         right thing for a fast local loop.
ISOLATED_ARGS ?= -q
test-isolated: | $(PY)
	$(PY) tools/isolated_suite.py $(ISOLATED_ARGS)

## purge-bytecode: delete every __pycache__ under the source trees.
##         The second half of the guard above, and it is not the same half.
##         PYTHONDONTWRITEBYTECODE stops a `.pyc` being WRITTEN; it does not stop
##         one already on disk being READ, and this tree had 11 such directories
##         when the defect was found. So the caches are removed before the suite
##         runs rather than trusted to be fresh. It costs nothing in steady
##         state: with writing disabled none are created, so after the first run
##         this finds nothing. Scoped to our own trees - the .venv is not ours to
##         churn, and its packages do not move under us mid-edit.
purge-bytecode:
	@find src tests tools -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ONE SCOPE, READ BY BOTH HALVES OF `lint` AND BY `format-check`. Spelling the
# roots three times is how the halves come to disagree about what they cover.
# `runs` resolves to the five tracked analysis scripts and nothing else: ruff
# respects .gitignore, and /runs/* is ignored except for the negated entries. So
# the scope maintains itself - track a new runs/ script and it is linted, with
# no second list to update.
LINT_PATHS := src tests tools runs

## lint: BOTH halves of what the commit hook enforces - `ruff check` AND
##         `ruff format --check` - over $(LINT_PATHS). This is the CI gate.
##
##         THE SECOND LINE IS NEW AT I20 AND EVERY "LINT CLEAN" BEFORE IT WAS A
##         WEAKER CLAIM THAN IT READ. This target was `ruff check` alone while
##         `.pre-commit-config.yaml` ran `ruff` AND `ruff-format --check`, so the
##         gate a developer runs and reports was strictly narrower than the gate
##         that decides whether the work can land. It surfaced the only way it
##         could: the stronger check refused a commit seconds after this target
##         reported zero. Nobody was going to catch it by reading reports,
##         because the reports were accurate.
##         `tests/test_format_hook.py` now DERIVES the hook's subcommands from
##         the config and asserts this recipe runs all of them, so the two
##         cannot part again without turning the suite red.
##
##         `--check` IS LOAD-BEARING, NOT A PREFERENCE. A bare `ruff format`
##         here would rewrite source during a gate run, which moves
##         `scoring_code_fingerprint` under every artifact bound to it (C-4.2).
##         `make format` is how you format, scoped to your own directory.
lint: | $(PY)
	$(RUFF) check $(LINT_PATHS)
	$(RUFF) format --check $(LINT_PATHS)

## typecheck: mypy --strict over src, PLUS an audit of the burn-down list that
##         lets it pass. The audit is the half that cannot rot: it re-runs mypy
##         with every exemption removed and fails if a listed module has become
##         clean without being retired. Config and the list live in
##         pyproject.toml under [tool.mypy]; the list is pinned by
##         tests/test_typecheck_config.py, so adding to it is a visible edit.
typecheck: | $(PY)
	MYPY="$(MYPY)" $(PY) tools/typecheck.py --python-executable $(PY)

## format-check: the second half of `make lint`, on its own, for a fast loop.
##         IT IS NO LONGER ADVISORY. It was, and the reason it was is worth
##         keeping: when the hook was changed, 88 of 150 files were
##         non-conformant, so wiring it into the gate then would have turned CI
##         red for reasons unrelated to anyone's science. That debt was
##         discharged deliberately, in its own commit (see
##         `tests/test_code_fingerprint_pins.py`, which records the fingerprint
##         transition it caused), and the tree has been conformant since. `lint`
##         now runs this, and `ci` depends on `lint`, so the debt cannot silently
##         return.
##         The pre-commit hook REPORTS the same thing and does NOT fix it: a hook
##         that rewrites files during a commit moves `scoring_code_fingerprint`,
##         which every run artifact stamps at both ends. See
##         .pre-commit-config.yaml.
format-check: | $(PY)
	$(RUFF) format --check $(LINT_PATHS)

## format: autofix lint + formatting. Scope this to YOUR directory, e.g.
##         make format DIRS="src/wildfire_nowcast/data" - never reformat another lead's code.
DIRS ?= src tests tools runs
format: | $(PY)
	$(RUFF) check --fix $(DIRS)
	$(RUFF) format $(DIRS)

## synth: generate one synthetic fire -> $(OUT)
synth: | $(PY)
	$(PY) -m wildfire_nowcast.common.synthetic --out $(OUT)

## contract: check ANY tensor.zarr against C1-C3 -> make contract TENSOR=path
contract: | $(PY)
	$(PY) -m wildfire_nowcast.common.contract $(TENSOR)

## contract-reporting: same, but reporting-gate clauses are HARD failures.
##         Run this before any number from a tensor appears in a gate: it is what
##         enforces C3.3 (n_train_blocks >= 2) instead of leaving it as a note.
contract-reporting: | $(PY)
	$(PY) -m wildfire_nowcast.common.contract $(TENSOR) --for-reporting

## contract-real: run the FULL pytest contract suite against a real fire, not
##         just the CLI -> make contract-real TENSOR=data/fires/2019_kincade/tensor.zarr
contract-real: | $(PY)
	$(PYTEST) tests/test_contracts.py --tensor-path $(TENSOR)

## contract-split: C8 + C3.1. With no RUN, print the current split fingerprint and
##         check the fold assignment across ALL fires. With RUN=runs/xxx, also
##         assert that run's stamps agree with each other, with the runs it
##         consumed, and with the split on disk. These are the clauses NO
##         per-tensor check can see: the split moved mid-task once already and
##         every tensor was conformant throughout (ADR-015).
RUN ?=
contract-split: | $(PY)
	$(PY) -m wildfire_nowcast.common.splits $(if $(RUN),--run $(RUN),)

## contract-all-fires: the C1-C3 CLI over every built fire, then C8/C3.1 once.
##         Exits non-zero if ANY fire fails, and names the ones that did.
##
##         THE SCRATCH FILE IS PER INVOCATION AND IS NOT NEGOTIABLE (ADR-098).
##         It was `/tmp/_c.txt`: a fixed, predictable name in a world-writable
##         directory, rewritten once per fire, inside the reporting path of this
##         project's flagship check. Four leads share this tree and have run this
##         target concurrently, and two runs interleave writes to one file - so
##         the `[FAIL]` lines printed can belong to a DIFFERENT fire than the one
##         `echo "FAIL $$d"` names beside them. The exit code stays correct,
##         which is what makes it the worse failure: a wrong exit code gets
##         investigated, and a plausible failure filed under the wrong fire gets
##         ACTED ON. `mktemp` per invocation, removed by a trap on every exit
##         path including the interrupt that a long run over 21 fires invites.
contract-all-fires: | $(PY)
	@rc=0; c=$$(mktemp "$${TMPDIR:-/tmp}/wfnc-contract.XXXXXX"); \
	trap 'rm -f "$$c"' EXIT INT TERM; \
	for d in data/fires/*/; do \
	  $(PY) -m wildfire_nowcast.common.contract $$d/tensor.zarr > "$$c" 2>&1 \
	    || { rc=1; echo "FAIL $$d"; grep -E '^\s+\[FAIL\]' "$$c" | cut -c1-140; }; \
	done; $(PY) -m wildfire_nowcast.common.splits || rc=1; exit $$rc

## prose: ADR-097 - classify non-ASCII punctuation in the TRACKED tree by the
##         region it sits in, and FAIL on the ones a reader sees. Every excluded
##         category is printed beside the verdict, with what it means and what
##         the scanner cannot see. Add `--region output` to list them.
prose: | $(PY)
	$(PY) tools/prose_scan.py --repo .

## null-check: C6.0 - score a DO-NOTHING null against every C6 metric.
##         Run this before any metric adjudicates any gate. If the null wins, the
##         metric is broken, not the model (ADR-017). Three metrics have already
##         failed this way. Add STRICT=1 to promote SILENCE_FAVOURING reporting
##         gaps to hard failures, or TENSOR=... to score real windows instead of
##         the generated label sequence.
STRICT ?=
null-check: | $(PY)
	$(PY) -m wildfire_nowcast.common.null_check \
	  $(if $(TENSOR_NULL),--tensor $(TENSOR_NULL),) $(if $(STRICT),--strict,)

## mutation: plan Task 5.6 - the SURVIVOR BUDGET over common/ and eval/. Runs the
##         suite once per single-token mutant and fails unless the survivor count
##         equals `SURVIVOR_BUDGET` in tools/mutation.py: it may be lowered by a
##         commit that kills one, never raised, and a pin ABOVE the debt is as
##         loud as a regression because an over-estimate forgives the next gap.
##         Mutants are applied in a git WORKTREE, never in this working copy -
##         the sweep edits eval/ by construction and C-4 freezes that while any
##         lead is running. Add `--pristine` for a number quotable against a sha,
##         `--only common/states.py` to look at one module.
##         DELIBERATELY NOT IN `make ci`, AND THE REASON IS NOW A MEASUREMENT
##         RATHER THAN THE "~40 minutes" THAT STOOD HERE (I25). Measured, not
##         estimated: `--pristine --workers 4` (this target's own default) at
##         `68ebd2f` took **110.4 min / 6625.4 s** for 138 mutants, on a 12-core
##         machine with two other leads working in it. The number it replaces was
##         inherited from `MEASURED_AT`'s 43.3 min at `1a7c480`; the suite has
##         grown by roughly 350 tests since, and every one of them runs 138 times
##         here. Wiring that into `make ci` multiplies the gate by more than an
##         order of magnitude and puts a git worktree per worker on the runner.
##         A SEPARATE SCHEDULED target is the right shape; a prerequisite of `ci`
##         is not. `make check` is the same argument.
##         THE GAP IS NARROWER THAN "not in ci" SOUNDS, and the distinction is
##         worth keeping straight: `budget_verdict` is a pure function and
##         `tests/test_hygiene.py` exercises it in BOTH directions in
##         milliseconds, inside `make ci`. The COMPARISON is gated. Only the
##         VALUE compared against is not - and that value has already moved:
##         the same sweep read **25 survivors against a budget of 21**, four of
##         them in three `eval/` modules that did not exist when the pin was
##         taken. It is NOT raised here; see BLOCKERS.md.
MUTATION_ARGS ?= --workers 4
mutation: | $(PY)
	$(PY) tools/mutation.py $(MUTATION_ARGS)

## playthrough: ADR-030 - run EVERY playthrough in the repo and require that each
##         one's planted defects are all detected. This is the mutation-coverage
##         gate: a playthrough that cannot fail turns this red. Includes the slow
##         ELMFIRE arm, which `make test` deselects.
##         Add ARGS='-k separation' to narrow it.
ARGS ?=
playthrough: | $(PY)
	$(PYTEST) -s tests/test_playthrough_registry.py $(ARGS)

## playthrough-list: which playthroughs exist, who owns them, and which are slow.
##         Reads the registry rather than a hand-kept list, so it cannot drift.
playthrough-list: | $(PY)
	@$(PY) -c "import sys; sys.path.insert(0, 'tests'); \
	from test_playthrough_registry import PLAYTHROUGHS as P; \
	[print(f'{n:44s} owner={e.owner:28s} {\"SLOW\" if e.slow else \"\"}') for n, e in sorted(P.items())]"

## playthrough-dispersion: G3's dispersion half - area_dispersion_ratio vs a closed form.
playthrough-dispersion: | $(PY)
	$(PYTEST) -s tests/test_playthrough_dispersion.py

## playthrough-off-state: can the kernel say "nothing will happen this hour"?
playthrough-off-state: | $(PY)
	$(PYTEST) -s tests/test_playthrough_off_state.py

## playthrough-separation: G3's calibration half as a separation test (ADR-032 (7)).
playthrough-separation: | $(PY)
	$(PYTEST) -s tests/test_playthrough_separation.py

## playthrough-harness: the mutation-coverage harness run through its own protocol.
playthrough-harness: | $(PY)
	$(PYTEST) -s tests/test_playthrough_harness.py

## playthrough-coarsening: simviz PLAYTHROUGH 1 - the 30 m -> 1 km rule (its own CLI).
playthrough-coarsening: | $(PY)
	$(PY) -m wildfire_nowcast.sim.coarsen

## playthrough-baseline: simviz PLAYTHROUGH 2 - ELMFIRE non-degeneracy (needs the binary).
playthrough-baseline: | $(PY)
	$(PY) -m wildfire_nowcast.sim.playthrough

## check: lint + types + the full suite + every playthrough. The developer gate.
check: lint typecheck test-all playthrough

## ci: THE gate. `.github/workflows/ci.yml` has one gate step and it is
##         `make ci`, so this list is not a copy of the workflow's list -- it is
##         the only list. Until I8 the workflow named the same six targets
##         itself and a test asserted the two agreed; agreement between two
##         copies is a weaker property than having one copy, and the cost of
##         the change is that GitHub's UI now shows one step instead of six
##         (make still names the failing target in `*** [Makefile:NN: t] Error`).
##         `check` plus the two artifact-level checks that need no data: the
##         synthetic fire judged by the real C1-C3 checker, and C6.0's
##         do-nothing null.
##         SCOPE, because this is exactly what was over-claimed: a green
##         `make ci` is a statement about THIS WORKING COPY on THIS MACHINE.
##         It is not a statement about the repository. `make ci-status` is.
ci: lint typecheck test-all playthrough synth contract null-check

## ci-status: ask GITHUB whether the published head is green, rather than
##         asking this machine whether it is happy. Exits non-zero unless the
##         workflow run for the exact commit on origin/main concluded
##         `success` -- and non-zero, loudly, when it cannot tell. UNKNOWN IS
##         NOT GREEN: a badge stayed red here for seven days and thirteen
##         commits while five local gate runs exited 0, because every one of
##         those runs answered a question about a laptop.
ci-status:
	@$(PY) tools/ci_status.py $(ARGS)

## movie: render a fire movie from a tensor path -> $(MOVIE)
movie: | $(PY)
	$(PY) -m wildfire_nowcast.sim.movie --tensor $(TENSOR) --out $(MOVIE)

## clean-outputs: remove generated outputs/
clean-outputs:
	rm -rf outputs
