# wildfire-nowcast — developer tasks.
#
# Every target runs against the repo-root .venv (ADR-001: uv, CPython 3.12).
# Never invoke bare `python`/`pytest`/`ruff` — they may be system Python.

VENV   := .venv
PY     := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
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

.PHONY: help venv install hooks test test-all lint typecheck format-check format synth contract contract-reporting \
        contract-real contract-split contract-all-fires null-check check ci movie clean-outputs \
        playthrough playthrough-list playthrough-dispersion playthrough-off-state \
        playthrough-separation playthrough-harness playthrough-coarsening playthrough-baseline

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'

## venv: create .venv with uv on the pinned interpreter (ADR-001)
venv:
	$(UV) venv --python $(PY_VERSION) $(VENV)

## install: install the project + dev extras into .venv, and wire the git hooks
install: venv
	VIRTUAL_ENV=$(CURDIR)/$(VENV) $(UV) pip install --python $(PY) -e ".[dev]"
	@$(MAKE) --no-print-directory hooks

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
##         simulator six times (33 s). Nothing is hidden in a pytest default —
##         bare `pytest` still runs everything, and `make check` runs it too.
test: | $(PY)
	$(PYTEST) -m "not slow"

## test-all: the whole suite INCLUDING slow playthroughs. What a release runs.
test-all: | $(PY)
	$(PYTEST)

## lint: lint src, tests and tools (the CI gate)
lint: | $(PY)
	$(RUFF) check src tests tools

## typecheck: mypy --strict over src, PLUS an audit of the burn-down list that
##         lets it pass. The audit is the half that cannot rot: it re-runs mypy
##         with every exemption removed and fails if a listed module has become
##         clean without being retired. Config and the list live in
##         pyproject.toml under [tool.mypy]; the list is pinned by
##         tests/test_typecheck_config.py, so adding to it is a visible edit.
typecheck: | $(PY)
	MYPY="$(MYPY)" $(PY) tools/typecheck.py --python-executable $(PY)

## format-check: also verify formatting (advisory; pre-commit fixes it per-file)
format-check: | $(PY)
	$(RUFF) format --check src tests tools

## format: autofix lint + formatting. Scope this to YOUR directory, e.g.
##         make format DIRS="src/wildfire_nowcast/data" — never reformat another lead's code.
DIRS ?= src tests tools
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
contract-all-fires: | $(PY)
	@rc=0; for d in data/fires/*/; do \
	  $(PY) -m wildfire_nowcast.common.contract $$d/tensor.zarr > /tmp/_c.txt 2>&1 \
	    || { rc=1; echo "FAIL $$d"; grep -E '^\s+\[FAIL\]' /tmp/_c.txt | cut -c1-140; }; \
	done; $(PY) -m wildfire_nowcast.common.splits || rc=1; exit $$rc

## null-check: C6.0 — score a DO-NOTHING null against every C6 metric.
##         Run this before any metric adjudicates any gate. If the null wins, the
##         metric is broken, not the model (ADR-017). Three metrics have already
##         failed this way. Add STRICT=1 to promote SILENCE_FAVOURING reporting
##         gaps to hard failures, or TENSOR=... to score real windows instead of
##         the generated label sequence.
STRICT ?=
null-check: | $(PY)
	$(PY) -m wildfire_nowcast.common.null_check \
	  $(if $(TENSOR_NULL),--tensor $(TENSOR_NULL),) $(if $(STRICT),--strict,)

## playthrough: ADR-030 — run EVERY playthrough in the repo and require that each
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

## playthrough-dispersion: G3's dispersion half — area_dispersion_ratio vs a closed form.
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

## playthrough-coarsening: simviz PLAYTHROUGH 1 — the 30 m -> 1 km rule (its own CLI).
playthrough-coarsening: | $(PY)
	$(PY) -m wildfire_nowcast.sim.coarsen

## playthrough-baseline: simviz PLAYTHROUGH 2 — ELMFIRE non-degeneracy (needs the binary).
playthrough-baseline: | $(PY)
	$(PY) -m wildfire_nowcast.sim.playthrough

## check: lint + types + the full suite + every playthrough. The developer gate.
check: lint typecheck test-all playthrough

## ci: exactly what .github/workflows/ci.yml runs, in one command, so the gate
##         can be reproduced locally instead of read off a log. `check` plus the
##         two artifact-level checks that need no data: the synthetic fire judged
##         by the real C1-C3 checker, and C6.0's do-nothing null.
##         tests/test_ci_matches_makefile.py fails the build if the two drift.
ci: lint typecheck test-all playthrough synth contract null-check

## movie: render a fire movie from a tensor path -> $(MOVIE)
movie: | $(PY)
	$(PY) -m wildfire_nowcast.sim.movie --tensor $(TENSOR) --out $(MOVIE)

## clean-outputs: remove generated outputs/
clean-outputs:
	rm -rf outputs
