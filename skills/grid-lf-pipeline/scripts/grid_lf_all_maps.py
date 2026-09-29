#!/usr/bin/env python
"""Run the whole grid -> Lawler-Fujita -> corrected-maps chain on one grid file.

One command does everything:

1. **topography** -- extract ``/params['Z (m)']`` and preprocess it with the
   sibling ``grid_topo`` module, writing ``<out>/<stem>_topo.csv``
   (``<stem>`` is the input's file name without suffix; a ``.3ds`` source is
   converted to h5 first, printing the converter command);
2. **fit** -- ``skills/lawler-fujita-correction/scripts/stm_lf_correct.py`` on
   that CSV (``--lambda-nm``, default 3 nm), saving the transferable
   displacement-field bundle to ``<out>/lf/<stem>_topo_transform.json``
   (``+ .h5``);
3. **apply** -- one ``scripts/apply_lf_transform.py`` run per map, for the
   current map and the dI/dV map of every bias frame, writing
   ``<out>/current_corrected/<i>_<bias>meV_corrected.csv`` and
   ``<out>/didv_corrected/<i>_<bias>meV_corrected.csv``.

The maps that go into step 3 are the raw per-channel averages, written as plain
square CSVs under ``<out>/current`` and ``<out>/didv``; nothing is resampled
here -- ``apply_lf_transform.py`` does the ``flipud`` + ``warp_by_field``
resampling itself and never touches the values.

**Orientation (top-down convention).**  A ``.3ds`` records its pixels bottom-up
and the loader's 3ds path flips nothing: it walks one flat pixel index and
writes it into ``/params`` and into the grid channels alike, so ``/data`` and
``/params`` are in one and the same frame.  The loader's sxm path
(``_reform_sxm_data``) is the one that normalises an image -- ``fliplr`` on the
backward-scan rows, ``flipud`` when ``SCAN_DIR`` is ``up`` -- i.e. sxm comes out
top-down.  This skill adopts the sxm convention for every grid product, so the
flip is made at the source (``grid_topo.load_z_map`` returns a top-down map, and
the topography CSV is top-down) and the rest of the chain follows it:

* the channel average of every bias frame is written with one ``np.flipud``, so
  the raw maps are top-down, exactly like the topography CSV the displacement
  field was fitted on -- ``apply_lf_transform.py`` requires its input in that
  same frame;
* the applier works in u's index frame (it flips its input on the way in) and
  leaves its product there, so the product is flipped with ``np.flipud`` once
  before it lands -- ``<out>/<kind>/<label>.csv`` and
  ``<out>/<kind>_corrected/<label>_corrected.csv`` are then both top-down and
  comparable point by point.

Neither flip is a frame fix: ``/data`` and ``/params`` were never mirrored with
respect to each other.  Both exist only so that the whole chain -- topography
CSV, raw maps, corrected maps -- comes out in the same top-down convention as
the sxm products.

Channels are resolved **by name** out of ``/channels`` (no channel index is
hard-coded anywhere; ``list.index(name)`` is used after the name is found), and
a forward channel is always taken together with its mirrored sweep partner,
Nanonis' ``' [bwd]'`` twin (``'Current (A)'`` <-> ``'Current [bwd] (A)'``):

* ``--current-ch`` / ``--didv-ch`` name the channels to average -- this is the
  primary path (comma separated names, each expanded to its ``[bwd]`` twin via
  the pairing rule below; bare 0-based indices are accepted as a shortcut);
* without them the choice is a convenience default: **current** = the forward
  channel whose name contains ``Current``, **dI/dV** = decided by the header,
  not by the channel name alone -- with ``Lock-in>Lock-in status = OFF`` the
  measurement used an *external* lock-in, so the Y output that is not the
  built-in one is meant (here ``DSP 7280 Y (%)``), and with ``= ON`` the
  built-in ``LI Demod 1 Y`` is meant.  A default that matches no channel, or
  more than one, is **not** guessed: the run stops (exit 1) and prints the
  ``--didv-ch '<name>'`` line to use instead.

Usage::

    .venv/bin/python <this script> GRID.h5 -o OUT [-L 30] [--lambda-nm 3] \
        [--current-ch 'Current (A),Current [bwd] (A)'] \
        [--didv-ch 'DSP 7280 Y (%),DSP 7280 Y [bwd] (%)']
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grid_lf_mplconfig")
)

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grid_topo  # noqa: E402

LF_SKILL = Path("skills") / "lawler-fujita-correction" / "scripts" / "stm_lf_correct.py"
APPLY_SCRIPT = Path("scripts") / "apply_lf_transform.py"
#: Nanonis tags the backward sweep of a recorded channel with this, just before
#: the unit: 'Current (A)' -> 'Current [bwd] (A)'.
BWD_TAG = " [bwd]"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="grid .h5 (or .3ds) file")
    parser.add_argument("-o", "--out", required=True, help="output directory")
    parser.add_argument(
        "-L",
        "--size-nm",
        type=float,
        default=None,
        help="scan size in nm; default: read from the file header",
    )
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=3.0,
        help="real-space scale kept by the LF lock-in low-pass (default 3.0, "
        "the value this grid recipe uses; the LF skill's own default of 30 nm "
        "is too wide for a 30 nm field and it warns about that)",
    )
    parser.add_argument(
        "--current-ch",
        default=None,
        help="current channels by name (comma separated; indices also "
        "accepted); default: the name containing 'Current' plus its [bwd] twin",
    )
    parser.add_argument(
        "--didv-ch",
        default=None,
        help="dI/dV channels by name (comma separated; indices also accepted); "
        "default: from the Lock-in status, see the module docstring",
    )
    return parser.parse_args(argv)


def repo_root() -> Path:
    """STM_DataProcessing root, from where ``stm_data_processing`` is installed."""
    import stm_data_processing

    return Path(stm_data_processing.__file__).resolve().parents[2]


def read_header(h5_path: Path):
    """``(/channels, Lock-in status, /bias volts)`` of ``h5_path``."""
    with h5py.File(h5_path, "r") as handle:
        names = [
            name.decode() if isinstance(name, bytes) else str(name)
            for name in handle["channels"][:]
        ]
        lockin = handle.get("header/Lock-in")
        status = (
            str(lockin.attrs["Lock-in status"]).strip().upper()
            if lockin is not None and "Lock-in status" in lockin.attrs
            else None
        )
        bias = np.asarray(handle["bias"][:], dtype=float)
    return names, status, bias


def split_unit(name: str) -> tuple[str, str]:
    """``'Current (A)' -> ('Current', ' (A)')``; no trailing unit -> ``(name, '')``."""
    match = re.search(r"\([^()]*\)\s*$", name)
    if match is None:
        return name, ""
    return name[: match.start()].rstrip(), name[match.start() :]


def backward_twin(name: str) -> str:
    """Name of the backward-sweep twin: ``'Current (A)' -> 'Current [bwd] (A)'``."""
    base, unit = split_unit(name)
    return f"{base}{BWD_TAG} {unit}".rstrip()


def paired_indices(names: list[str], forward: str) -> list[int]:
    """The forward channel and its ``[bwd]`` twin, located by name in ``names``."""
    indices = [names.index(forward)]
    twin = backward_twin(forward)
    if twin in names:
        indices.append(names.index(twin))
    return indices


def channel_family(names: list[str], wanted, flag: str) -> list[int]:
    """Indices of the one forward channel satisfying ``wanted`` plus its ``[bwd]`` twin.

    The automatic choice is a convenience only: when the default rule matches no
    channel, or more than one, the run stops (exit 1, hint on stderr) and asks
    for the explicit ``--current-ch`` / ``--didv-ch`` name instead of picking one
    silently.
    """
    forwards = [name for name in names if BWD_TAG not in name and wanted(name)]
    if len(forwards) != 1:
        listed = ", ".join(repr(name) for name in names)
        hint = f"--{flag} '<name>[,<name>]'"
        if not forwards:
            raise SystemExit(
                f"ERROR: no channel matched the default rule; pass {hint}.\n"
                f"/channels: {listed}"
            )
        raise SystemExit(
            f"ERROR: {len(forwards)} channels matched the default rule "
            f"({', '.join(repr(name) for name in forwards)}); pick one with "
            f"{hint}.\n/channels: {listed}"
        )
    return paired_indices(names, forwards[0])


def is_y_channel(name: str) -> bool:
    """True for a lock-in Y output (``DSP 7280 Y (%)``, ``LI Demod 1 Y (A)``)."""
    return re.search(r"(?<![A-Za-z])Y(?![A-Za-z])", name) is not None


def current_channels(names: list[str]) -> list[int]:
    """Current channels: the name containing ``Current`` plus its ``[bwd]`` twin."""
    return channel_family(names, lambda name: "Current" in name, "current-ch")


def didv_channels(names: list[str], status: str | None) -> list[int]:
    """dI/dV channels as the header says the lock-in was hooked up."""
    if status is None:
        raise SystemExit(
            "ERROR: the header carries no Lock-in status; pass --didv-ch '<name>'"
        )
    if status.startswith("ON"):
        builtin = True
    elif status.startswith("OFF"):
        builtin = False
    else:
        raise SystemExit(
            f"ERROR: unknown Lock-in status {status!r}; pass --didv-ch '<name>'"
        )

    def wanted(name: str) -> bool:
        return is_y_channel(name) and ("LI Demod" in name) == builtin

    return channel_family(names, wanted, "didv-ch")


def resolve_channels(spec: str, names: list[str]) -> list[int]:
    """``--*-ch`` value -> indices; names are the primary spelling.

    Every spelling is resolved to itself plus, when ``/channels`` has it, its
    ``[bwd]`` twin, so one name is enough to average both sweep directions;
    duplicates are dropped.
    """
    picked: list[int] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token in names:
            expanded = paired_indices(names, token)
        elif token.isdigit() and int(token) < len(names):
            expanded = paired_indices(names, names[int(token)])
        else:
            raise SystemExit(
                f"ERROR: no channel named {token!r} in /channels; "
                f"available: {', '.join(repr(name) for name in names)}"
            )
        for index in expanded:
            if index not in picked:
                picked.append(index)
    if not picked:
        raise SystemExit(f"ERROR: no channel given ({spec!r})")
    return picked


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)


def write_map(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", fmt="%.10e")


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    h5_path = grid_topo.convert_3ds(src) if src.suffix.lower() == ".3ds" else src
    h5_path = h5_path.resolve()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = repo_root()
    lf_script = root / LF_SKILL
    apply_script = root / APPLY_SCRIPT
    missing = [str(path) for path in (lf_script, apply_script) if not path.is_file()]
    if missing:
        raise SystemExit(
            "ERROR: not found in the STM_DataProcessing checkout at "
            f"{root}: {', '.join(missing)}"
        )

    size_nm = (
        args.size_nm if args.size_nm is not None else grid_topo.scan_size_nm(h5_path)
    )
    stem = h5_path.stem

    # which channels go into the maps; resolved first so a channel choice that
    # cannot be made unambiguously stops the run before the fit is computed
    names, status, bias = read_header(h5_path)
    current = (
        resolve_channels(args.current_ch, names)
        if args.current_ch
        else current_channels(names)
    )
    didv = (
        resolve_channels(args.didv_ch, names)
        if args.didv_ch
        else didv_channels(names, status)
    )
    print(
        f"# channels: current {[names[i] for i in current]}, "
        f"dI/dV {[names[i] for i in didv]} (Lock-in status {status})"
    )

    # 1. topography (extract + preprocess)
    topo_csv = out / f"{stem}_topo.csv"
    topo = grid_topo.preprocess(grid_topo.load_z_map(h5_path))
    np.savetxt(topo_csv, topo, delimiter=",", fmt="%.10e")
    print(
        f"# topography: {topo_csv}  ({topo.shape[0]} x {topo.shape[1]} px, {size_nm:g} nm)"
    )

    # 2. Lawler-Fujita fit + displacement-field bundle
    lf_dir = out / "lf"
    lf_dir.mkdir(parents=True, exist_ok=True)
    transform_json = lf_dir / f"{stem}_topo_transform.json"
    run(
        [
            sys.executable,
            str(lf_script),
            str(topo_csv),
            "-L",
            f"{size_nm:g}",
            "-o",
            str(lf_dir),
            "--lambda-nm",
            f"{args.lambda_nm:g}",
            "--save-transform",
            str(transform_json),
            "--stm-lib",
            str(root / "src"),
        ]
    )

    # 3. apply the bundle to every current and dI/dV map
    with h5py.File(h5_path, "r") as handle:
        cube = handle["data"][:]

    for index, volts in enumerate(bias):
        label = f"{index}_{volts * 1e3:g}meV"
        for kind, channels in (("current", current), ("didv", didv)):
            # Top-down convention (see the module docstring): a grid's /data and
            # /params share one frame -- the 3ds loader flips neither -- whereas
            # the sxm loader normalises its images top-down.  Every grid product
            # of this pipeline follows the sxm convention, and the topography
            # CSV the applier was fitted on is top-down too, so the channel
            # average gets one np.flipud here.
            raw_csv = out / kind / f"{label}.csv"
            write_map(raw_csv, np.flipud(cube[:, :, channels, index].mean(axis=2)))
            corrected_dir = out / f"{kind}_corrected"
            run(
                [
                    sys.executable,
                    str(apply_script),
                    str(raw_csv),
                    "--transform",
                    str(transform_json),
                    "-L",
                    f"{size_nm:g}",
                    "-o",
                    str(corrected_dir),
                ]
            )
            # The applier works in u's index frame (it flipped its input on the
            # way in) and leaves its product there, so flip it back with
            # np.flipud before it lands: the corrected map is then top-down like
            # raw_csv above and the two are comparable point by point.
            corrected_csv = corrected_dir / f"{label}_corrected.csv"
            corrected = np.loadtxt(corrected_csv, delimiter=",")
            write_map(corrected_csv, np.flipud(corrected))

    print(f"# done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
