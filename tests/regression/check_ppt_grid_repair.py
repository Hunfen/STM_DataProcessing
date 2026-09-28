"""Regression checks for the 3ds repair-copy hygiene and the dI/dV channel selection.

Run from the repository root:

    .venv/bin/python tests/regression/check_ppt_grid_repair.py

The script is self-contained: every input is a synthetic file written into a
temporary directory, no real measurement data is touched and nothing is read
from the network. It prints one ``PASS <name>`` line per check and exits
non-zero as soon as one of them fails.

Guarded behaviours:

  A. ``read_grid()`` does not raise on a ``.3ds`` whose header holds a
     multi-line string value, and the caller's directory is exactly the same
     before and after the call (in particular no ``*__repaired_*.3ds`` copy is
     left behind).
  B. The repaired copy is written into the module's process-level temporary
     directory. This is asserted on its own (the temporary directory was
     created, the copy is inside it and the reported path is there) and never
     inferred from the absence of residue in the caller's directory.
  C. A well-formed synthetic ``.3ds`` really round-trips through nanonispy:
     the channel array equals the floats decoded from the file itself, so a
     passing check cannot come from values written into the test.
  D. An input without ``:HEADER_END:`` degrades to "warn with the file name
     and skip" (``read_grid`` returns None) instead of raising.
  E. ``parse_channel_answer()`` is 0-based, comma separated and de-duplicating,
     and rejects an empty, non-integer or out-of-range answer with ValueError.
  F. ``average_channels()`` returns the element-wise arithmetic mean of the
     selected channels for one, two and three selections.
  G. A channel index this file does not have, and an index whose name differs
     from the reference table the user was shown, are reported (file name,
     index, reference name and actual name) instead of being used silently.
"""

import contextlib
import logging
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

from stm_data_processing.utils import plot_funcs
from stm_data_processing.utils.nanonis_ppt_generator import parse_channel_answer
from stm_data_processing.utils.plot_funcs import (
    average_channels,
    file_lacks_channels,
    read_grid,
    resolve_channel_arrays,
)

PLOT_LOGGER = "stm_data_processing.utils.plot_funcs"

# Minimal .3ds header nanonispy's Grid needs (Grid dim / Grid settings /
# Sweep Signal / Fixed parameters / Experiment parameters / # Parameters /
# Experiment size / Points / Channels / Delay + the four metadata fields).
# A grid of 2 x 1 pixels is written as 2 * (5 parameters + 3 sweep points);
# nanonispy's _extract_topo reads parameter column 4, hence 5 parameters.
CHANNEL = "Current (A)"
NUM_PARAMETERS = 5
NUM_POINTS = 3
GRID_DIM = (2, 1)
HEADER = {
    "Grid dim": "2 x 1",
    "Grid settings": "0.0;0.0;5e-08;5e-08;0.0",
    "Sweep Signal": "Bias (V)",
    "Fixed parameters": "Z (m)",
    "Experiment parameters": "Z (m);Current (A);LI X (A);LI Y (A);Bias (V)",
    "# Parameters (4 byte)": str(NUM_PARAMETERS),
    "Experiment size (bytes)": str(4 * GRID_DIM[0] * GRID_DIM[1] * 8),
    "Points": str(NUM_POINTS),
    "Channels": CHANNEL,
    "Delay before measuring (s)": "0.0",
    "Experiment": "synthetic",
    "Start time": "00:00:00",
    "End time": "00:00:01",
    "User": "regression",
    "Comment": "synthetic grid",
}
# Values are small integers, so float32 and float64 representations agree and
# the "max |delta| == 0" comparisons below are exact.
DATA = [float(value) for value in range(1, 17)]
# First header line of a value that Nanonis continues on the next line: the
# opening quote sits on the line carrying '=', the closing quote sits alone on
# a following line that has no '=' at all.
MULTILINE_VALUE_FIRST_LINE = "20250910 anneal@1.35W for 30mins"


def write_3ds(path: Path, header: dict, floats) -> None:
    """Write a synthetic .3ds file (key=value header + big-endian f4 data)."""
    with path.open("wb") as handle:
        for key, value in header.items():
            handle.write(f"{key}={value}\r\n".encode())
        handle.write(b":HEADER_END:\r\n")
        handle.write(np.asarray(floats, dtype=">f4").tobytes())


def write_multiline_3ds(path: Path, header: dict, floats) -> None:
    """Write a .3ds whose 'Comment' value continues on the next header line."""
    with path.open("wb") as handle:
        for key, value in header.items():
            if key == "Comment":
                continue
            handle.write(f"{key}={value}\r\n".encode())
        handle.write(f'Comment="{MULTILINE_VALUE_FIRST_LINE}\r\n'.encode())
        handle.write(b'"\r\n')
        handle.write(b":HEADER_END:\r\n")
        handle.write(np.asarray(floats, dtype=">f4").tobytes())


def read_data_section(path: Path) -> np.ndarray:
    """Decode the float32 data section without using nanonispy.

    The expectation of the round-trip check is taken from the bytes actually
    stored in the file, decoded independently of the reader under test.
    """
    raw = path.read_bytes()
    tag = b":HEADER_END:\r\n"
    offset = raw.index(tag) + len(tag)
    return np.frombuffer(raw, dtype=">f4", offset=offset).astype(float)


def snapshot(directory: Path) -> list:
    """Describe every entry below ``directory`` (name, kind, size, mtime)."""
    entries = []
    for path in sorted(directory.rglob("*")):
        stat = path.stat()
        entries.append(
            (
                str(path.relative_to(directory)),
                path.is_dir(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return entries


class _CollectHandler(logging.Handler):
    """Collect the messages emitted on one logger."""

    def __init__(self, sink: list) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record.getMessage())


@contextlib.contextmanager
def captured_warnings(*logger_names: str):
    """Yield a list that collects every message logged on ``logger_names``."""
    messages: list = []
    installed = []
    for name in logger_names:
        target = logging.getLogger(name)
        handler = _CollectHandler(messages)
        installed.append((target, handler, target.level, target.propagate))
        target.addHandler(handler)
        target.setLevel(logging.DEBUG)
        target.propagate = False
    try:
        yield messages
    finally:
        for target, handler, level, propagate in installed:
            target.removeHandler(handler)
            target.setLevel(level)
            target.propagate = propagate


def make_caller_dir(prefix: str) -> Path:
    """Create a fresh directory that plays the role of the data folder."""
    return Path(tempfile.mkdtemp(prefix=prefix))


def test_read_grid_leaves_no_residue_in_caller_dir():
    """A: a multi-line header is repaired without leaving residue behind."""
    caller_dir = make_caller_dir("check_ppt_grid_repair_a_")
    try:
        grid_path = caller_dir / "broken_grid_a.3ds"
        write_multiline_3ds(grid_path, HEADER, DATA)

        before = snapshot(caller_dir)
        # The call must not raise for either outcome (Grid or None).
        with captured_warnings(PLOT_LOGGER):
            read_grid(grid_path)
        after = snapshot(caller_dir)

        assert after == before, (
            f"read_grid changed the caller's directory: before={before} after={after}"
        )
        stray = sorted(path.name for path in caller_dir.rglob("*__repaired_*"))
        assert not stray, f"repaired copy left in the caller's directory: {stray}"
        assert list(caller_dir.iterdir()) == [grid_path], (
            f"unexpected extra entries: {sorted(p.name for p in caller_dir.iterdir())}"
        )
    finally:
        shutil.rmtree(caller_dir, ignore_errors=True)


def test_repaired_copy_lives_in_process_temp_dir():
    """B: the repaired copy is written into the process-level temp directory."""
    caller_dir = make_caller_dir("check_ppt_grid_repair_b_")
    try:
        grid_path = caller_dir / "broken_grid_b.3ds"
        write_multiline_3ds(grid_path, HEADER, DATA)

        with captured_warnings(PLOT_LOGGER) as messages:
            grid = read_grid(grid_path)

        repair_dir = plot_funcs._REPAIRED_GRID_DIR
        assert repair_dir is not None, (
            "read_grid did not create the process-level repair directory"
        )
        repair_dir = Path(repair_dir)
        assert repair_dir.is_dir(), f"{repair_dir} is not a directory"

        resolved_repair = repair_dir.resolve()
        resolved_caller = caller_dir.resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        assert resolved_repair.parent == temp_root, (
            f"repair directory {resolved_repair} is not directly inside the "
            f"process temporary directory {temp_root}"
        )
        assert resolved_repair != resolved_caller, (
            f"repair directory {resolved_repair} is the caller's directory"
        )
        assert resolved_caller not in resolved_repair.parents, (
            f"repair directory {resolved_repair} is below the caller's "
            f"directory {resolved_caller}"
        )

        copies = sorted(
            path.name
            for path in repair_dir.iterdir()
            if path.name.startswith(grid_path.stem) and "__repaired_" in path.name
        )
        assert copies, (
            f"no repaired copy for {grid_path.name} inside {repair_dir}; "
            f"contents: {sorted(p.name for p in repair_dir.iterdir())}"
        )
        assert any(str(repair_dir) in message for message in messages), (
            "no warning pointed at the process-level repair directory; "
            f"messages: {messages}"
        )
        # The repaired copy must itself be readable: the check proves the copy
        # exists in the temporary directory and is what the reader parsed.
        assert grid is not None, (
            "the multi-line header was not repaired into a readable grid"
        )
        assert CHANNEL in grid.signals, (
            f"repaired grid lacks channel {CHANNEL!r}: {sorted(grid.signals)}"
        )
    finally:
        shutil.rmtree(caller_dir, ignore_errors=True)


def test_synthetic_grid_data_round_trips():
    """C: a well-formed synthetic grid yields exactly the stored floats."""
    caller_dir = make_caller_dir("check_ppt_grid_repair_c_")
    try:
        grid_path = caller_dir / "valid_grid_c.3ds"
        write_3ds(grid_path, HEADER, DATA)

        values = read_data_section(grid_path)
        per_pixel = NUM_PARAMETERS + NUM_POINTS
        assert values.size == GRID_DIM[0] * GRID_DIM[1] * per_pixel, (
            f"the file holds {values.size} floats, expected "
            f"{GRID_DIM[0] * GRID_DIM[1] * per_pixel}"
        )
        expected = (
            values.reshape(GRID_DIM[1], GRID_DIM[0], per_pixel)[
                :, :, NUM_PARAMETERS : NUM_PARAMETERS + NUM_POINTS
            ]
            .astype(float)
            .ravel()
        )

        with captured_warnings(PLOT_LOGGER):
            grid = read_grid(grid_path)
        assert grid is not None, "read_grid returned None for a valid grid"
        channel = np.asarray(grid.signals[CHANNEL], dtype=float)
        assert list(grid.header["channels"]) == [CHANNEL], (
            f"parsed channels {list(grid.header['channels'])}"
        )
        assert channel.size == expected.size, (
            f"channel holds {channel.size} values, expected {expected.size}"
        )
        max_delta = float(np.max(np.abs(channel.ravel() - expected)))
        assert max_delta == 0.0, (
            f"channel data differs from the stored floats by {max_delta}"
        )
        # The parameter block is a different slice of the same file: the
        # comparison above must not be satisfied by every slice.
        params = np.asarray(grid.signals["params"], dtype=float).ravel()
        assert not np.array_equal(params, channel.ravel()), (
            "the parameter block equals the channel block; the slices are not "
            "discriminating"
        )
    finally:
        shutil.rmtree(caller_dir, ignore_errors=True)


def test_unrepairable_input_warns_and_skips():
    """D: bytes without ':HEADER_END:' warn with the file name and return None."""
    caller_dir = make_caller_dir("check_ppt_grid_repair_d_")
    try:
        garbage_path = caller_dir / "garbage_grid_d.3ds"
        garbage_path.write_bytes(bytes(range(256)) * 4)

        with captured_warnings(PLOT_LOGGER) as messages:
            result = read_grid(garbage_path)

        assert result is None, f"read_grid returned {result!r} for garbage input"
        assert messages, "no warning was logged for the unreadable file"
        assert any(garbage_path.name in message for message in messages), (
            f"no warning names {garbage_path.name!r}; messages: {messages}"
        )
    finally:
        shutil.rmtree(caller_dir, ignore_errors=True)


def test_channel_answer_parsing():
    """E: the answer is 0-based, comma separated, de-duplicated and validated."""
    accepted = [
        ("7", 9, [7]),
        (" 7 , 6 ", 9, [7, 6]),
        ("7,7", 9, [7]),
        ("0", 1, [0]),
        ("0,1,2", 3, [0, 1, 2]),
    ]
    for answer, n_channels, expected in accepted:
        parsed = parse_channel_answer(answer, n_channels)
        assert parsed == expected, (
            f"parse_channel_answer({answer!r}, {n_channels}) returned {parsed!r}, "
            f"expected {expected!r}"
        )

    rejected = [
        ("", 9, "empty answer"),
        ("   ", 9, "blank answer"),
        ("x", 9, "non-integer entry"),
        ("7.5", 9, "float entry"),
        ("-1", 9, "negative index"),
        ("999", 9, "index above the channel count"),
        ("3", 3, "index equal to the channel count"),
    ]
    for answer, n_channels, why in rejected:
        try:
            parsed = parse_channel_answer(answer, n_channels)
        except ValueError:
            continue
        raise AssertionError(
            f"parse_channel_answer({answer!r}, {n_channels}) returned {parsed!r} "
            f"instead of raising ValueError ({why})"
        )


def test_average_channels_matches_arithmetic_mean():
    """F: several selected channels are averaged element-wise.

    The arrays are chosen so that the mean differs from every input, which is
    asserted for N >= 2 (no coincidental equality).
    """
    cases = [
        [np.array([1.0, 2.0, 4.0])],
        [np.array([1.0, 2.0, 3.0]), np.array([3.0, 6.0, 9.0])],
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 9.0, 15.0]),
            np.array([0.0, 3.0, 3.0]),
        ],
    ]
    for arrays in cases:
        result = np.asarray(average_channels(arrays), dtype=float)
        expected = sum(array for array in arrays) / len(arrays)
        assert result.shape == arrays[0].shape, (
            f"mean shape {result.shape} != input shape {arrays[0].shape}"
        )
        max_delta = float(np.max(np.abs(result - np.asarray(expected, dtype=float))))
        assert max_delta == 0.0, (
            f"N={len(arrays)}: mean differs from the hand-computed value by {max_delta}"
        )
        if len(arrays) >= 2:
            for array in arrays:
                assert not np.array_equal(result, np.asarray(array, dtype=float)), (
                    f"N={len(arrays)}: the mean coincides with one of the inputs; "
                    "the case does not distinguish averaging from selection"
                )


def test_missing_channel_index_warns_with_file_name():
    """G: an index this file does not have is reported, then the file is skipped."""
    signals = {"Current (A)": np.zeros(4), "LI Demod 1 Y (A)": np.ones(4)}
    source = "Grid Spectroscopy003.3ds"

    with captured_warnings(PLOT_LOGGER) as messages:
        lacks = file_lacks_channels(signals, [5], source)
        arrays = resolve_channel_arrays(signals, [5], source=source)

    assert lacks is True, "file_lacks_channels did not report the missing index"
    assert arrays is None, f"resolve_channel_arrays returned {arrays!r}"
    assert messages, "no warning was logged for the missing channel index"
    assert any(source in message for message in messages), (
        f"no warning names the file {source!r}; messages: {messages}"
    )
    assert any("5" in message for message in messages), (
        f"no warning names the missing index 5; messages: {messages}"
    )


def test_reference_name_mismatch_warns():
    """G: a name mismatch against the reference table is reported, not hidden."""
    actual_name = "LI Demod 1 Y (A)"
    reference_name = "LI Demod 1 Y (V)"
    signals = {"LI Demod 1 X (A)": np.zeros(4), actual_name: np.ones(4)}
    source = "Grid Spectroscopy001.3ds"

    with captured_warnings(PLOT_LOGGER) as messages:
        arrays = resolve_channel_arrays(
            signals,
            [1],
            reference_names=["LI Demod 1 X (A)", reference_name],
            source=source,
        )

    assert arrays is not None, "the selected index exists; it must not be skipped"
    assert len(arrays) == 1, f"expected one array, got {len(arrays)}"
    max_delta = float(np.max(np.abs(np.asarray(arrays[0], dtype=float) - 1.0)))
    assert max_delta == 0.0, "the returned array is not the stored channel"
    assert messages, "no warning was logged for the name mismatch"
    joined = " | ".join(messages)
    for needle in (source, "1", reference_name, actual_name):
        assert needle in joined, (
            f"the mismatch warning does not mention {needle!r}: {messages}"
        )

    # The matching case must stay silent (otherwise the warning is noise).
    with captured_warnings(PLOT_LOGGER) as clean_messages:
        resolve_channel_arrays(
            signals,
            [0, 1],
            reference_names=["LI Demod 1 X (A)", actual_name],
            source=source,
        )
    assert clean_messages == [], (
        f"matching reference names must not warn; got {clean_messages}"
    )
    # Without a reference table there is nothing to compare against.
    with captured_warnings(PLOT_LOGGER) as no_reference_messages:
        resolve_channel_arrays(signals, [0, 1], source=source)
    assert no_reference_messages == [], (
        f"no reference table must not warn; got {no_reference_messages}"
    )


TESTS = [
    test_read_grid_leaves_no_residue_in_caller_dir,
    test_repaired_copy_lives_in_process_temp_dir,
    test_synthetic_grid_data_round_trips,
    test_unrepairable_input_warns_and_skips,
    test_channel_answer_parsing,
    test_average_channels_matches_arithmetic_mean,
    test_missing_channel_index_warns_with_file_name,
    test_reference_name_mismatch_warns,
]


def main() -> int:
    """Run every check, print one PASS/FAIL line each and return an exit code."""
    failed = 0
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # report every failing check and keep going
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} grid-repair regression checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
