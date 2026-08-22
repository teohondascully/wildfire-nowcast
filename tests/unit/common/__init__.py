"""Unit tests mirroring ``src/wildfire_nowcast/common/``, one file per module.

A package rather than a bare directory so that ``tests/unit/common/test_states.py``
and ``tests/test_states.py`` can coexist: under pytest's default ``prepend``
import mode two test files with the same basename and no package collide.

See ``tests/README.md`` for the convention and ``tests/test_hygiene.py`` for the
check that enforces it.
"""
