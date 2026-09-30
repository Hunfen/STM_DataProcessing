#!/usr/bin/env python
"""List the grid h5 channels and let the user map roles (current / didv / ...).

This is the **first stage** of the ``grid-sym-fft`` skill.  It reads a grid
``.h5`` produced by ``python -m stm_data_processing.io.grid2h5`` and prints the
full channel table from ``/channels``: the index, the channel name, its unit
(from ``/channel_units``) and the forward/backward pairing derived from the
name.  The user then decides which channels to extract and what role each plays
(``current``, ``didv``, ...) -- roles are free-form labels chosen by the user.

Channel names are resolved **by name** (``list.index(name)`` at runtime); the
script never contains channel-index literals, and the index is always a derived,
printed column, never a lookup key the user must type.  A name the user gives
must match exactly one entry: if it matches none, or more than one, the script
lists every available name and exits non-zero instead of guessing.

Backward pairing is derived purely from the name: ``X`` and ``X [bwd]`` are the
forward/backward pair of one physical channel.  If the user names a forward
channel that has a ``[bwd]`` twin, the extraction stage averages the two; naming
the ``[bwd]`` one directly is rejected (the forward name is the canonical handle).

Usage::

    .venv/bin/python <this script> GRID.h5 [--map ROLE="CHANNEL NAME"]...

Without ``--map`` the script prompts on stdin for the channel names and their
roles.  With ``--map role="name"`` it takes the same mapping non-interactively
(any number of ``--map``), printing the table and the resolved roles, then
exits 0.  It only *prints* the mapping -- the extraction itself is done by the
sibling ``grid_sym.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grid_sym_mplconfig")
)

import h5py

#: Marker that separates a forward channel name from its backward twin.
BWD_MARKER = "[bwd]"


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _split_unit(name: str):
    """Split a trailing parenthetical unit, e.g. 'Current (A)' -> ('Current', ' (A)')."""
    if name.endswith(")") and " (" in name:
        head, _, tail = name.rpartition(" (")
        return head, " (" + tail
    return name, ""


def _bwd_twin(name: str) -> str | None:
    """The backward-scan twin of a forward channel name, by name convention.

    ``X [bwd] (U)`` is the backward twin of ``X (U)``: the marker sits before
    the trailing parenthetical unit.  The forward name always comes first in the
    h5 channel list; the twin is derived from the name, never read from an index.
    """
    if BWD_MARKER in name:
        return None  # already a backward name
    head, unit = _split_unit(name)
    twin = f"{head} {BWD_MARKER}{unit}"
    return None if twin == name else twin


class Channels:
    """The full channel table of a grid h5, resolved by name."""

    def __init__(self, h5_path: Path):
        with h5py.File(h5_path, "r") as handle:
            names = _decode(handle["channels"][:])
            units = _decode(handle["channel_units"][:])
        if len(names) != len(units):
            raise SystemExit(
                f"ERROR: {h5_path}: {len(names)} channels vs "
                f"{len(units)} units -- inconsistent h5"
            )
        self.names = names
        self.units = units

    def table_lines(self) -> list[str]:
        """Human-readable full table: index, name, unit, forward twin."""
        lines = []
        lines.append(f"{'idx':>4}  {'channel name':<40} {'unit':<8} pair")
        lines.append("-" * 72)
        for i, name in enumerate(self.names):
            twin = _bwd_twin(name)
            notes = ""
            if BWD_MARKER in name:
                notes = "(backward)"
            elif twin and twin in self.names:
                notes = f"fwd of {twin}"
            elif twin:
                notes = f"(no {twin!r} in this file -- forward only)"
            lines.append(f"{i:>4}  {name:<40} {self.units[i]:<8} {notes}")
        return lines

    def resolve(self, role: str, name: str):
        """Resolve a channel name to its role mapping, fail-closed on ambiguity.

        Returns ``(role, forward_name, backward_name)`` where ``forward_name``
        is the canonical (non-``[bwd]``) handle and ``backward_name`` may be
        ``None`` when there is no backward twin in the file.  Naming a ``[bwd]``
        channel directly is rejected; the forward name is canonical.
        """
        if BWD_MARKER in name:
            raise SystemExit(
                f"ERROR: role {role!r} names the backward channel {name!r}. "
                "Give the forward channel name (e.g. without ' [bwd]'); the "
                "pipeline averages the forward and backward directions itself. "
                "Available names:\n  " + "\n  ".join(self.names)
            )
        matches = [n for n in self.names if n == name]
        if len(matches) == 0:
            raise SystemExit(
                f"ERROR: no channel named {name!r} for role {role!r}. "
                "Available names:\n  " + "\n  ".join(self.names)
            )
        if len(matches) > 1:
            raise SystemExit(
                f"ERROR: {name!r} matches {len(matches)} channels -- ambiguous. "
                "Available names:\n  " + "\n  ".join(self.names)
            )
        twin = _bwd_twin(name)
        back = twin if twin in self.names else None
        return role, name, back


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="grid .h5 file (grid2h5 output)")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar='ROLE="CHANNEL NAME"',
        help="role=channel mapping (repeatable); non-interactive mode",
    )
    return parser.parse_args(argv)


def _parse_map_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f'ERROR: --map must be ROLE="CHANNEL NAME", got {spec!r}')
    role, name = spec.split("=", 1)
    role = role.strip()
    name = name.strip()
    if not role or not name:
        raise SystemExit(f"ERROR: empty role or name in --map {spec!r}")
    return role, name


def _interactive_prompt(channels: Channels) -> list[tuple[str, str]]:
    print("\nEnter channel->role mappings, one per line (role = channel name).")
    print("  e.g.  current = Current (A)")
    print("  e.g.  didv = DSP 7280 Y (%)")
    print("Enter a blank line when done.", flush=True)
    mappings = []
    while True:
        try:
            line = input("role = channel name > ")
        except EOFError:
            break
        if not line.strip():
            break
        try:
            role, name = _parse_map_spec(line)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            continue
        mappings.append((role, name))
    if not mappings:
        raise SystemExit("ERROR: no channel/role mapping given")
    return mappings


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        raise SystemExit(f"ERROR: {src} does not exist")

    channels = Channels(src)
    for line in channels.table_lines():
        print(line)

    if args.map:
        # normalize: split each --map spec into role/name through the same parser
        parsed = []
        for spec in args.map:
            role, name = _parse_map_spec(spec)
            parsed.append((role, name))
    else:
        print("\nInteractive mode: no --map given.", flush=True)
        parsed = _interactive_prompt(channels)

    print("\nResolved role -> channel mapping:")
    for role, name in parsed:
        _, fwd, back = channels.resolve(role, name)
        extra = f" (averaged with backward twin {back})" if back else ""
        print(f"  {role} <- {fwd}{extra}")

    print("\n# channel listing for grid_sym.py --map flags:")
    for role, name in parsed:
        print(f'#   --map {role}="{name}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
