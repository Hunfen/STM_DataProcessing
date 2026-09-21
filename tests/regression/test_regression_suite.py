"""Pytest gate for the regression check scripts.

Every ``check_*.py`` file next to this module is a standalone regression body:
it prints its own PASS/FAIL summary and exits non-zero when a check fails.
This module is the single pytest entry point for all of them - it runs each
script as a subprocess from the repository root and fails when a child exits
non-zero, so the one command gate is::

    .venv/bin/python -m pytest -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPTS = sorted(Path(__file__).resolve().parent.glob("check_*.py"))
_TAIL_LINES = 60


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


@pytest.mark.parametrize("script", CHECK_SCRIPTS, ids=lambda path: path.stem)
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
