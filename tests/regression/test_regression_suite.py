"""Pytest gate for the regression check scripts.

Every ``check_*.py`` file next to this module is a standalone regression body:
it prints its own PASS/FAIL summary and exits non-zero when a check fails.
This module is the single pytest entry point for all of them - it runs each
script as a subprocess from the repository root and fails when a child exits
non-zero, so the one command gate is::

    .venv/bin/python -m pytest -q

Checks that need measurement data which only exists on the maintainer's
workstation are marked ``localdata``: the local gate above selects everything,
while CI runs the data-independent subset::

    .venv/bin/python -m pytest -q -m "not localdata"

The classification is mechanical, never optimistic: a script is ``localdata``
as soon as it references a local absolute path (``/Users/``), which is exactly
how the data-dependent checks reach the real scans. A newly added check that
reads such a path is therefore deselected in CI automatically instead of
turning the CI job red.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPTS = sorted(Path(__file__).resolve().parent.glob("check_*.py"))
_TAIL_LINES = 60
LOCAL_PATH_MARK = "/Users/"


def _tail(text: str) -> str:
    """Return the last lines of ``text`` (keep failure reports readable)."""
    lines = text.splitlines()
    return "\n".join(lines[-_TAIL_LINES:])


def _failure_report(script: Path, completed: subprocess.CompletedProcess) -> str:
    return (
        f"{script.relative_to(REPO_ROOT)} exited with {completed.returncode}\n"
        f"--- stdout (last {_TAIL_LINES} lines) ---\n{_tail(completed.stdout)}\n"
        f"--- stderr (last {_TAIL_LINES} lines) ---\n{_tail(completed.stderr)}"
    )


def needs_local_data(script: Path) -> bool:
    """True when ``script`` references a local absolute path (``/Users/``)."""
    return LOCAL_PATH_MARK in script.read_text(encoding="utf-8")


def _param(script: Path) -> pytest.param:
    """One parametrized case, tagged ``localdata`` when it needs local data."""
    marks = [pytest.mark.localdata] if needs_local_data(script) else []
    return pytest.param(script, marks=marks, id=script.stem)


@pytest.mark.parametrize("script", [_param(script) for script in CHECK_SCRIPTS])
def test_check_script_exits_zero(script: Path) -> None:
    """Each regression check script must exit 0 when run from the repo root."""
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, _failure_report(script, completed)
