# contributing

this is a single-author personal experiment. it is public because the result is
worth reading and because a repository nobody can check is a repository nobody
should believe, not because it is looking for contributors. there is no roadmap
to join and no issue triage.

that said, the mechanics below are real and they are what a change has to satisfy
whoever makes it. if you have cloned this to run it, read the first two sections
and stop; the rest is about editing.

## contents

- [getting a working copy](#getting-a-working-copy)
- [the gate](#the-gate)
- [branching and releases](#branching-and-releases)
- [the hooks, and what they refuse](#the-hooks-and-what-they-refuse)
- [rules a change has to satisfy](#rules-a-change-has-to-satisfy)
- [where the documents are](#where-the-documents-are)

## getting a working copy

```
make install     build .venv on the pinned interpreter and sync the lock
make hooks       install the git hooks
make test        the fast loop
```

`make install` runs `uv pip sync --require-hashes requirements.lock`, not
`pip install`. two things follow from that word and both are the point: the
environment ends up holding the lock's set and nothing else, and no version is
ever resolved at install time. `make relock` regenerates the lock and never
upgrades anything; upgrading a package means deleting its pin on purpose.

never invoke a bare `python`, `pytest` or `ruff`. every target goes through the
repository `.venv`, and the Makefile fails loudly rather than falling back to a
system interpreter.

## the gate

```
make ci
```

that is the whole gate, and `.github/workflows/ci.yml` has one step which is
`make ci`. the workflow does not restate the target list; there is one list and
the Makefile holds it.

run it **after** `git add`, not before. staging is an input to this suite: see
`docs/testing.md` for why, and for the difference between `make test` and
`make test-all`.

a green `make ci` is a claim about your working copy. `make ci-status` is the
other claim and asks GitHub about the published head.

## branching and releases

`main` is the only branch and the only publishable ref, and the push guard
derives that from what the remote already carries rather than from a branch name
typed into a file. there is no develop branch and no release branch, and nothing
in this repository configures branch protection, because there is one author.

**there is no release process, and this section is the whole of it.** the package
version is `0.0.0`, there are zero tags, there is no changelog, and nothing here
publishes an artifact anywhere. a wheel is built during the test suite, but only
so that a test can open it and assert that the non-Python files the package needs
at run time actually arrive in it; that build is a check, not a release. if this
ever ships something, that will need a document, and inventing one now would be
describing a process that does not exist.

## the hooks, and what they refuse

`make hooks` installs pre-commit plus a push guard.

**on commit**, ruff lint and ruff format run in report-only mode, along with the
standard large-file, merge-conflict, yaml, toml, json and private-key checks. the
format hook reports and does **not** rewrite: a hook that rewrites files during a
commit moves the code fingerprint that every run artifact stamps at both ends,
which is a provenance change disguised as tidiness.

**on the commit message**, a guard rejects attribution. it derives its rules
rather than listing tells. six of them: no trailer block at all, adjudicated by
git's own trailer grammar; no identity other than the ones git itself reports for
this repository; no URL outside this repository's own remotes; no pictographs or
invisible formatting; a small declared set of attribution constructions, which is
the one maintained surface and says so; and no typographic punctuation. nothing is
allow-listed on the first rule, so a co-authorship trailer and a sign-off trailer
fail for the same reason, which is that they are trailers.

**on push**, a guard refuses to publish any ref the remote does not already
carry. the allowed set is read from the remote-tracking refs, so `git push --all`
and `git push --mirror` both stop there. this exists because an unrelated
local-only history is present in this checkout and publishing it is one flag
away.

## rules a change has to satisfy

**one implementation.** anything the contract adjudicates lives in
`src/wildfire_nowcast/common/` and is imported, not reimplemented. a producer and
a verifier computing the same geometry through different code is how an artifact
passes its own check and is still wrong.

**a contract change is a document change first.** `docs/interfaces.md` is
numbered and versioned and its first line is parsed at import. changing what a
clause means requires bumping that version and recording the reason, and the code
follows the document rather than the other way round.

**a new check ships with the defect it catches.** plant it, watch the check go
red, then restore. a guard whose only evidence is that it was added is not
evidence; the most recent example here was a shell guard whose prescribed
spelling is silently ignored by the make version that ships on macOS, and it was
caught only because the plant was run.

**do not reformat somebody else's directory.** `make format DIRS=<your
directory>` exists for this. a tree-wide reformat moves fingerprints that
published numbers are bound to.

**quote the target with the number.** "the suite passes" is not a claim; "`make
test-all` at `<sha>` reports N passed" is. the targets do not run the same set,
and a figure from the weaker one has been quoted in support of the stronger claim
here before.

## where the documents are

```
README.md                    what the project is, and where the science stands
docs/architecture.md         how the code is shaped and why
docs/repository-map.md       what is in the tree, directory by directory
docs/testing.md              the gates, and how to add a check that can fail
docs/troubleshooting.md      failures this repository actually produces
docs/interfaces.md           the numbered, versioned contract
docs/decisions.md            why the thresholds and splits are where they are
tests/README.md              where a test file goes
```
