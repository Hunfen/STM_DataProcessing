#!/usr/bin/env python
"""Symmetrized-FFT pipeline for grid spectroscopy maps (Skill B, stage 2).

This is the **second stage** of the ``grid-sym-fft`` skill.  It takes the role
mapping produced by the sibling ``grid_channels.py`` (roles applied to channel
names), and for every bias point of the grid runs the full chain::

    a. extract   each role = mean(forward channel, its [bwd] twin)
    b. raw FFT   F = fftshift(fft2(x)), x = map with NaN filled by nanmedian
    c. correct   resample the map along the LF displacement field (resampling
                 ONLY, target values untouched) -> corrected FFT
    d. Z3        Z = (F + R120 F + R240 F)/3 in the complex domain, rotation
                 about the DC pixel (the only place the spectrum is rotated)
    e. C3(dc)    A = |F|; C3dc = (A + R120 A + R240 A)/3 (the only difference
                 from Z3 is the modulus is taken first)

Every stage writes its data and an image (data next to image, one directory per
quantity), and every FFT-producing stage writes complex128 ``_fft2.npy`` plus a
``log|...|`` plot.  The correction stage uses Skill A's transferable
displacement-field bundle and only re-samples: ``corrected(r) = T(r + u(r))``,
with **no** flip of the target map: ``/data`` and ``/params`` are written on one
flat pixel index, so a map read by channel name is already in the frame the u
field lives in (see the "coordinate frames" section of ``SKILL.md`` and
``apply_transform()``).  It never touches the target values -- no
``stm_lf_apply.py`` (that one removes a plane), only ``lf_lib.warp_by_field``.

**Two transform branches.**  ``--transform`` accepts either transferable bundle:

* an **LF displacement-field bundle** (Skill A: ``lawler-fujita-correction``) --
  ``corrected(r) = T(r + u(r))`` through ``lf_lib.warp_by_field``, with the flat
  layout of ``SKILL.md`` §3;
* an **affine transform bundle** (``affine-correction``: schema
  ``topo-correction-transform``) -- ``A = P M^-1 P`` plus the offset/canvas side of
  the affine skill's ``_image_transform``, applied **directly to the map as read
  from ``/data``, with no flip at all** (the affine skill's apply stage goes
  through the topo *CSV* frame and flips there; the h5 frame is that frame's flip,
  so copying that ``flipud`` here would mirror the correction).  Every product of
  this branch is prefixed with ``<out>/affine/`` and only the c-e stages are
  written (``corrected`` / ``fft_corrected`` / ``z3`` / ``c3dc``): there is no
  ``raw/``, ``fft_raw/`` or ``window_candidates/``, so no colour-candidate run
  exists for this branch and ``--window-candidates`` is fail-closed.  The per-role
  colour windows are inherited from the LF run's ``<out>/fft_window.json`` (or from
  the branch's own ``<out>/affine/fft_window.json`` once written) and this run's
  selection is written back there.  Provenance is stated truthfully in
  ``pipeline.log`` and ``<out>/affine/affine_provenance.json`` (the fit's
  ``anchor_verdict`` / ``n_labelled`` are passed through as they are, and the
  anchor self-check is *not* re-run here).

**Two-run color-scale protocol.**  The user looks at real candidate figures
before committing to a FFT colormap (they dislike formula-derived windows):

1. Run ``--window-candidates N`` (N is required with this flag): stages a+b
   run and, **per role**, N candidate FFT images + that role's own contact
   sheet are written into ``window_candidates/<role>/``; the per-role candidate
   table is printed and saved to ``<OUTDIR>/fft_window_candidates.json``, then
   the script exits 0 (no window chosen yet).
2. Run ``--pick-window K`` (candidate K of *every* role), or
   ``--fft-window-role ROLE VMIN VMAX`` (one role at a time; repeatable), or
   ``--fft-window VMIN VMAX`` (the same numbers for every role): the full chain
   a+b+c+d+e runs, ``<OUTDIR>/fft_window.json`` is written, and every FFT plot
   (fft_raw / fft_corrected / z3 / c3dc) of a role uses **that role's** window.

Windows are never shared across roles: ``current`` and ``didv`` carry different
units, and their log|F| medians are ~26 apart, so one window pins one role at
the top of the colormap and the other at the bottom.  Within a role the window
is shared by all its bias points and by all four FFT plot families.

If ``fft_window.json`` already exists it is reused (idempotent).  Running the
full chain with no window selected yet is fail-closed.

**Fail-closed.**  A missing/ambiguous channel name, a ``--transform`` that is
missing or not unique inside its directory, a grid h5 without ``/params``'s
``Z (m)`` column, a saved candidate table from the pooled-window version, or
requesting the full chain before a window is chosen all exit non-zero without
writing a half-finished pipeline.

Usage::

    .venv/bin/python <this script> GRID.h5 \\
        --map current="Current (A)" --map didv="DSP 7280 Y (%)" \\
        --transform <dir|bundle.json> -o OUT [--window-candidates 9]
    (then)  ... -o OUT --pick-window 3
    (or)    ... -o OUT --fft-window-role current -22 -16 \\
                          --fft-window-role didv 4 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# stm_data_processing / lf_lib / matplotlib would rather have a writable config dir.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grid_sym_mplconfig")
)

import h5py
import numpy as np
import scipy.ndimage

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is a hard dependency for the contact sheet
    Image = None


def repo_root() -> Path:
    import stm_data_processing

    return Path(stm_data_processing.__file__).resolve().parents[2]


ROOT = repo_root()
LF_SCRIPTS = ROOT / "skills" / "lawler-fujita-correction" / "scripts"
sys.path.insert(0, str(LF_SCRIPTS))
import lf_lib  # noqa: E402  (needs LF_SCRIPTS on sys.path)

BWD_MARKER = "[bwd]"
Z_COLUMN = "Z (m)"
FFT_WINDOW_FILE = "fft_window.json"
CANDIDATES_FILE = "fft_window_candidates.json"

#: Affine branch (``--transform`` = an ``affine-correction`` bundle).
#: Schema of the affine skill's transform JSON; the discriminator between the two
#: bundle kinds, together with a positive-definite ``affine_q``.
AFFINE_SCHEMA = "topo-correction-transform"
#: ``<out>/affine`` -- every product of the affine branch is prefixed with it (the
#: caller still passes ``-o <out>``), and only the c-e stages are written.
AFFINE_DIRNAME = "affine"
AFFINE_PROVENANCE_FILE = "affine_provenance.json"
#: Maps physical (x, y) order to array (row, col) order and back; ``A = P M^-1 P``.
AXIS_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])

# Color-scale candidate generation: (vmin percentile, vmax percentile) of **one
# role's own** log|F| distribution.  Windows are never pooled across roles --
# ``current`` and ``didv`` sit on different units and their log|F| medians differ
# by ~26 (e.g. -19.95 vs +6.27 on 2025-11-15), so a pooled window pins one role
# at the top of the colormap and the other at the bottom.
#
# The pair is chosen so that the fraction of pixels pinned at a window endpoint,
# ``vmin% + (100 - vmax)%``, stays below 5 %: a candidate whose picture is one
# flat colour cannot be judged.  Index 0 is the widest window (least clipping),
# the last is the contrastiest one that still stays inside that budget.
_WINDOW_CANDIDATE_FRACS = [
    (0.05, 99.95),
    (0.10, 99.80),
    (0.20, 99.50),
    (0.40, 99.20),
    (0.70, 98.90),
    (1.00, 98.60),
    (1.40, 98.30),
    (1.80, 98.00),
    (2.20, 97.70),
]

#: Spatial (data) windows applied to the map *before* ``fft2``.  ``hamming`` is
#: the default because the user's reference series carries the windowed variants
#: (``*_fft_hamming``, ``*_C3dc_hamming``, ``*_Z3/hamming_z3``); ``none`` is the
#: historical behaviour and keeps every unwindowed file byte-identical.
SPATIAL_WINDOWS = {"none": None, "hamming": np.hamming, "hanning": np.hanning}

#: Default spatial window (the user's reference series carries the windowed
#: variants, so the windowed set is what they asked for).
DEFAULT_SPATIAL_WINDOW = "hamming"


#: ``<label>`` -> windowed file names.  The unwindowed set keeps the plain
#: ``<label>_fft2.npy`` / ``<label>_fft.png`` names; the windowed set adds the
#: window name, so both live side by side in the same family directory.
def windowed_npy_name(label: str, window: str) -> str:
    return f"{label}_fft2_{window}.npy"


def windowed_png_name(label: str, window: str) -> str:
    return f"{label}_fft_{window}.png"


#: A windowed plane whose endpoint-clip fraction differs from the unwindowed one
#: by more than this many percentage points gets a warning (the colour window was
#: chosen on the *unwindowed* candidate figures).
WINDOW_CLIP_WARN_PP = 0.5

#: Colormap of every FFT image (log|F|, log|Z|, log|C3dc| and the candidate
#: figures).  The user's reference series -- ``data_processing/20251110/
#: 20251110_grid1_30mV500pA_30nm/{didv_corrected_upscaling_fft*,
#: didv_original_fft, rmap_fft, zmap_fft}/`` -- is plasma: nearest-LUT median
#: distance 0.8 against plasma vs >=33 for magma, 40 for inferno, 43 for turbo,
#: 52 for cividis and 55 for viridis (5 images measured directly, 23 by the
#: captain).  Single source: changing the convention is one edit here.
FFT_CMAP = "plasma"

#: Colormap of the real-space maps (raw / corrected) -- unchanged.
MAP_CMAP = "viridis"

#: Percentiles used to pick the plane that represents a role (see
#: ``representative_bias_index``).
_REPRESENTATIVE_QUANTILES = (0.5, 50.0, 99.5)

#: Slack (percentage points) allowed between a candidate's design budget and the
#: endpoint fraction measured on the plane that is actually drawn.  The two are
#: not the same number (the budget comes from the role's pooled log|F|, the plane
#: is one member of that pool), so a small excess is expected; a larger one is
#: reported as a warning instead of being hidden.
PLANE_CLIP_TOL_PCT = 0.5


def _decode(values):
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _split_unit(name: str):
    """Split a trailing parenthetical unit, e.g. 'Current (A)' -> ('Current', ' (A)')."""
    if name.endswith(")") and " (" in name:
        head, _, tail = name.rpartition(" (")
        return head, " (" + tail
    return name, ""


def _bwd_twin(name: str) -> str | None:
    """Backward twin of a forward channel name, by the h5 naming convention.

    The marker goes *before* the trailing unit: ``Current (A)`` pairs with
    ``Current [bwd] (A)``.  The twin is derived from the name only, never read
    from an index; ``None`` means the name is already a backward channel or has
    no backward twin.
    """
    if BWD_MARKER in name:
        return None
    head, unit = _split_unit(name)
    if f"{head} {BWD_MARKER}{unit}" == name:
        return None
    return f"{head} {BWD_MARKER}{unit}"


# --------------------------------------------------------------------------
# input resolution
# --------------------------------------------------------------------------


def load_channel_table(h5_path: Path):
    with h5py.File(h5_path, "r") as handle:
        names = _decode(handle["channels"][:])
        units = _decode(handle["channel_units"][:])
    return names, units


def load_bias_meV(h5_path: Path) -> np.ndarray:
    """Bias labels in meV, read only from ``/bias`` (never hard-coded)."""
    with h5py.File(h5_path, "r") as handle:
        bias_v = np.asarray(handle["bias"][:]).astype(float)
    return bias_v * 1000.0


def require_z_column(h5_path: Path) -> None:
    """Fail-closed unless ``/params`` carries the ``Z (m)`` column."""
    with h5py.File(h5_path, "r") as handle:
        columns = _decode(handle["param_columns"][:])
    if Z_COLUMN not in columns:
        raise SystemExit(
            f"ERROR: {h5_path} has no {Z_COLUMN!r} in /param_columns "
            f"(found {columns}); cannot tie the maps to the LF topography"
        )


def resolve_transform(transform: str) -> Path:
    """Locate the transform JSON from ``--transform`` (a file or a directory).

    A directory must hold exactly one bundle: ``*_transform.json`` (the LF naming
    of Skill A) or ``transform.json`` (the affine skill's).  Zero or more than one
    candidate is fail-closed -- and the ambiguous case lists the candidates.
    """
    p = Path(transform)
    if p.is_file():
        return p
    if p.is_dir():
        found = list(p.glob("*_transform.json"))
        plain = p / "transform.json"
        if plain.is_file():
            found.append(plain)
        # one entry per resolved path, in file-name order
        matches = sorted({m.resolve(): m for m in found}.values(), key=lambda m: m.name)
        if len(matches) == 0:
            raise SystemExit(
                f"ERROR: {transform} contains no *_transform.json / transform.json "
                f"bundle (found: {[x.name for x in p.iterdir()]})"
            )
        if len(matches) > 1:
            raise SystemExit(
                f"ERROR: {transform} contains {len(matches)} transform bundles "
                f"({[m.name for m in matches]}); pass the exact file instead"
            )
        return matches[0]
    raise SystemExit(f"ERROR: --transform {transform} does not exist")


def read_json_object(path: Path):
    """Read ``path`` as a JSON object; fail-closed on anything else."""
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot read {path}: {exc}") from exc
    except ValueError as exc:  # json.JSONDecodeError is a ValueError
        raise SystemExit(f"ERROR: {path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"ERROR: {path} must hold a JSON object, got {type(payload).__name__}"
        )
    return payload


def affine_q_problem(payload):
    """Why ``payload`` is not an affine bundle, or ``None`` when it is one.

    An affine bundle is ``schema == AFFINE_SCHEMA`` **plus** a finite 2x2
    ``affine_q`` with a positive determinant (the map has to be invertible).  The
    ``fallback`` key is deliberately not part of this test: an affine bundle
    carries ``fallback: true`` exactly when the fit solved nothing, and it is still
    an affine bundle.
    """
    if payload.get("schema") != AFFINE_SCHEMA:
        return f"schema is {payload.get('schema')!r}, not {AFFINE_SCHEMA!r}"
    raw = payload.get("affine_q")
    try:
        q = np.asarray(raw, dtype=float)
    except TypeError, ValueError:
        return f"affine_q is not numeric ({raw!r})"
    if q.shape != (2, 2) or not np.all(np.isfinite(q)):
        return f"affine_q must be a finite 2x2 matrix, got {raw!r}"
    det = float(np.linalg.det(q))
    if det <= 0.0:
        return f"affine_q must have a positive determinant, got det = {det!r}"
    return None


def classify_transform(payload, bundle_json: Path) -> str:
    """``"affine"`` or ``"lf"`` for one bundle JSON, fail-closed when ambiguous.

    ``schema`` first: ``topo-correction-transform`` with a valid ``affine_q`` is the
    affine branch; a ``u_field_file`` key is the LF branch.  Both at once (an affine
    bundle that also points at a u field) and neither are fail-closed with the
    reason, so a bundle is never silently sent down the wrong stage.
    """
    problem = affine_q_problem(payload)
    is_affine = problem is None
    is_lf = "u_field_file" in payload
    if is_affine and is_lf:
        raise SystemExit(
            f"ERROR: {bundle_json} looks like BOTH an affine transform bundle "
            f"(schema {payload.get('schema')!r} with a positive-definite affine_q) "
            f"and an LF displacement-field bundle (it carries 'u_field_file'); "
            f"refusing to guess which stage to apply -- pass the bundle of the "
            f"branch you want"
        )
    if not is_affine and not is_lf:
        raise SystemExit(
            f"ERROR: {bundle_json} is neither an affine transform bundle "
            f"({problem}) nor an LF displacement-field bundle (no 'u_field_file' "
            f"key); pass the JSON Skill A or affine-correction wrote"
        )
    return "affine" if is_affine else "lf"


def affine_geometry(payload, bundle_json: Path):
    """The affine numbers the resampling stage needs, validated once."""
    problem = affine_q_problem(payload)
    if problem is not None:
        raise SystemExit(f"ERROR: {bundle_json} is not an affine bundle: {problem}")
    return {
        "affine_q": np.asarray(payload["affine_q"], dtype=float),
        "pad": int(payload.get("pad", 10)),
        "order": int(payload.get("order", 3)),
        # same semantics as the affine skill's apply stage: nothing was solved
        "fallback": bool(payload.get("fallback", False))
        or payload.get("method") == "identity_fallback",
        "method": payload.get("method"),
    }


def image_transform(stretch, n, pad):
    """Array matrix, canvas side and offset of the affine resampling.

    **Local mirror** of ``stm_apply_transform._image_transform`` in the
    ``affine-correction`` skill (``skills/affine-correction/scripts/
    stm_apply_transform.py``), which itself mirrors the package's
    ``bragg_peak.correct._image_transform``: the map is ``A = P M^-1 P`` (``P`` the
    axis swap), the canvas holds every transformed corner of the ``n``-pixel input
    plus ``pad`` pixels and the offset aligns the two centres.

    It is mirrored instead of imported so this skill keeps its single dependency
    (``lf_lib``) and never reaches into another skill's script, and the mirror
    agrees with the original bit for bit -- that agreement is a self-check in
    ``SKILL.md`` §8 item 11.
    """
    matrix = AXIS_SWAP @ np.linalg.inv(np.asarray(stretch, dtype=float)) @ AXIS_SWAP
    corners = (
        np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]) * (n - 1) / 2.0
    )
    extent = np.abs(np.linalg.inv(matrix) @ corners.T).max(axis=1)
    n_out = 2 * (int(np.ceil(float(extent.max()))) + int(pad)) + 1
    offset = (n - 1) / 2.0 - matrix @ np.array([(n_out - 1) / 2.0, (n_out - 1) / 2.0])
    return matrix, n_out, offset


def apply_affine(role_map: np.ndarray, geometry):
    """Apply the fitted affine map to a map **in the h5 frame**: 0 flips.

    ``A = P M^-1 P`` and the offset of :func:`image_transform` are applied straight
    to the map as read from ``/data``.  That is the frame ``A`` was fitted in: the
    h5 ``/data`` and ``/params`` arrays share one flat pixel index, ``grid_topo.py``
    flips the topo CSV it writes, and the affine fit (``stm_topo_correct.py``) flips
    that CSV back before fitting.  The affine skill's own apply stage reaches the same
    frame by flipping *its* CSV input before resampling (and flipping the product back
    since 2026-10-14) -- copying that input flip here would apply the mirrored
    correction.  Measured on 2025-11-15 (n = 304, marker at row 20): the two routings
    differ by the pure mirror ``n_out - 1 - 2r`` ~ 290 px, and a real map's high-pass
    cross-correlation with the flipped routing drops to ~0.

    ``method == identity_fallback`` means the fit solved nothing: the map is copied
    onto its own canvas with no NaN added, exactly like the affine skill's apply
    stage.
    """
    n = int(role_map.shape[0])
    if geometry["fallback"]:
        return np.array(role_map, dtype=float, copy=True), n
    matrix, n_out, offset = image_transform(geometry["affine_q"], n, geometry["pad"])
    corrected = scipy.ndimage.affine_transform(
        role_map,
        matrix,
        offset=offset,
        output_shape=(n_out, n_out),
        order=int(geometry["order"]),
        mode="constant",
        cval=np.nan,
        prefilter=int(geometry["order"]) > 1,
    )
    return corrected, n_out


def read_window_file(path: Path, role_names):
    """Per-role colour windows of a saved ``fft_window.json``.

    Accepts the current ``{"roles": {...}}`` document and the legacy flat
    ``{"vmin", "vmax"}`` one (applied to every role after a warning: those windows
    were pooled across roles, hence wrong for each of them).  A file without a
    window for every role is fail-closed.
    """
    saved = json.loads(path.read_text())
    if "roles" in saved:
        windows = {r: dict(w) for r, w in saved["roles"].items()}
    else:
        windows = {
            r: {"vmin": saved["vmin"], "vmax": saved["vmax"]} for r in role_names
        }
        print(
            f"# WARNING: {path} uses the legacy single-window format; "
            f"applying it to every role.  Rerun --window-candidates N to get "
            f"per-role windows."
        )
    missing = [r for r in role_names if r not in windows]
    if missing:
        raise SystemExit(
            f"ERROR: {path} has no window for role(s) {missing}; it "
            f"holds {sorted(windows)}"
        )
    return windows


def load_transform(bundle_json: Path):
    """Read the transferable bundle: JSON metadata + the u field h5 + validity."""
    payload = json.loads(bundle_json.read_text())
    if payload.get("fallback") or not payload.get("u_field_file"):
        raise SystemExit(
            f"ERROR: {bundle_json} carries no displacement field "
            f"(method={payload.get('method')}, fallback={payload.get('fallback')})"
        )
    u_path = Path(payload["u_field_file"])
    if not u_path.is_absolute():
        u_path = bundle_json.parent / u_path
    with h5py.File(u_path, "r") as handle:
        u_nm = np.stack([handle["u_x"][:], handle["u_y"][:]]).astype(float)
        valid = np.asarray(handle["valid"][:], dtype=bool)
    return payload, u_nm, valid, u_path


def resolve_roles(h5_path: Path, specs):
    """Turn ``--map ROLE="CHANNEL NAME"`` specs into resolved role descriptors."""
    names, units = load_channel_table(h5_path)
    roles = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f'ERROR: --map must be ROLE="CHANNEL NAME", got {spec!r}')
        role, name = spec.split("=", 1)
        role = role.strip()
        name = name.strip().strip('"')
        if not role or not name:
            raise SystemExit(f"ERROR: empty role/name in --map {spec!r}")
        if BWD_MARKER in name:
            raise SystemExit(
                f"ERROR: role {role!r} names backward channel {name!r}; give the "
                f"forward name. Available:\n  " + "\n  ".join(names)
            )
        matches = [n for n in names if n == name]
        if len(matches) == 0:
            raise SystemExit(
                f"ERROR: no channel named {name!r} for role {role!r}. "
                f"Available:\n  " + "\n  ".join(names)
            )
        if len(matches) > 1:
            raise SystemExit(
                f"ERROR: {name!r} matches {len(matches)} channels -- ambiguous. "
                f"Available:\n  " + "\n  ".join(names)
            )
        twin = _bwd_twin(name)
        back = twin if twin in names else None
        roles.append(
            {
                "role": role,
                "fwd": name,
                "back": back,
                "fwd_idx": names.index(name),
                "back_idx": names.index(back) if back else None,
                "unit": units[names.index(name)],
            }
        )
    if not roles:
        raise SystemExit('ERROR: at least one --map ROLE="CHANNEL NAME" is required')
    return roles


# --------------------------------------------------------------------------
# data / FFT helpers
# --------------------------------------------------------------------------


def extract_role_map(data, role_desc, bias_idx: int) -> np.ndarray:
    """Mean of the forward channel and its [bwd] twin, one map per bias point."""
    fwd = data[:, :, role_desc["fwd_idx"], bias_idx].astype(float)
    if role_desc["back_idx"] is not None:
        back = data[:, :, role_desc["back_idx"], bias_idx].astype(float)
        return np.nanmean(np.stack([fwd, back]), axis=0)
    return fwd


def _nanmedian_fill(x: np.ndarray) -> np.ndarray:
    med = float(np.nanmedian(x))
    return np.where(np.isnan(x), med, x)


def compute_fft(x: np.ndarray, window: str = "none") -> np.ndarray:
    """Complex128, fftshifted, NaN filled by nanmedian of valid pixels.

    ``window`` names a spatial window applied to the NaN-filled map *before*
    ``fft2``, as the separable 2D outer product ``np.outer(w, w)`` with
    ``w = np.hamming(n)`` / ``np.hanning(n)``; ``"none"`` reproduces the
    historical unwindowed transform byte for byte.
    """
    arr = _nanmedian_fill(x)
    taps = SPATIAL_WINDOWS[window]
    if taps is not None:
        w = taps(arr.shape[0])
        arr = arr * np.outer(w, w)
    return np.fft.fftshift(np.fft.fft2(arr)).astype(np.complex128)


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def rotate_about_dc(arr: np.ndarray, angle_deg: float, order=3) -> np.ndarray:
    """Rotate ``arr`` by ``angle`` about the DC pixel (after fftshift).

    ``fftshift`` puts the DC of an even ``N x N`` spectrum at index ``N/2``.
    The affine recipe ``matrix=R, offset=c - R c`` with ``c = (N/2, N/2)``
    (floating point) makes ``output(c) = input(c)`` exactly, so the DC
    survives a cubic-spline rotation (verified: preserved to ~1e-16).
    """
    n = arr.shape[0]
    c = np.asarray([n / 2.0, n / 2.0])
    R = _rotation_matrix(angle_deg)
    offset = c - R @ c
    return scipy.ndimage.affine_transform(
        arr,
        matrix=R,
        offset=offset,
        output_shape=arr.shape,
        order=order,
        mode="nearest",
        prefilter=order > 1,
    )


def z3_symmetrize(F: np.ndarray) -> np.ndarray:
    """Z = (F + R120 F + R240 F) / 3 in the complex domain, about the DC pixel."""
    return (F + rotate_about_dc(F, 120.0) + rotate_about_dc(F, 240.0)) / 3.0


def c3dc_symmetrize(F: np.ndarray) -> np.ndarray:
    """C3dc = (|F| + R120 |F| + R240 |F|) / 3 (modulus taken first)."""
    A = np.abs(F)
    return (A + rotate_about_dc(A, 120.0) + rotate_about_dc(A, 240.0)) / 3.0


def _residual_metric(vals: np.ndarray) -> float:
    """max|v - R120 v| / max|v| -- reduction of the broken-120-round residual."""
    r120 = rotate_about_dc(vals, 120.0)
    denom = float(np.max(np.abs(vals)))
    if denom <= 0:
        return 0.0
    return float(np.max(np.abs(vals - r120))) / denom


def _dc_preservation(vals: np.ndarray, reference: np.ndarray) -> float:
    n = vals.shape[0]
    c = n // 2
    r = complex(reference[c, c])
    if abs(r) == 0:
        return float("nan")
    return abs(complex(vals[c, c])) / abs(r)


def _energy_ratio(out: np.ndarray, ref: np.ndarray) -> float:
    """Retained spectral power ``sum|out|^2 / sum|ref|^2`` after symmetrization."""
    denom = float(np.sum(np.abs(ref) ** 2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(out) ** 2)) / denom


# --------------------------------------------------------------------------
# correction (resample only)
# --------------------------------------------------------------------------


def apply_transform(role_map: np.ndarray, size_nm: float, payload, u_nm, valid):
    """Resample ``role_map`` along u -- never a value-domain change.

    No flip here.  The h5 ``/params`` and ``/data`` arrays share one flat pixel
    index (row 0 is the smallest physical Y), and the LF displacement field is
    fitted in exactly that frame: ``grid_topo.py`` flips the CSV it writes, and
    ``stm_lf_correct.py`` flips it back before fitting, so ``u`` is defined on
    the h5 frame the maps are read in.  Feeding ``warp_by_field`` a flipped map
    would apply a mirrored u.  (``stm_lf_apply.py`` does flip its input, because
    *its* input is the CSV frame, not the h5 frame.)
    """
    n = role_map.shape[0]
    reference_n = int(payload["n_px_reference"])
    reference_fov = float(payload["field_of_view_nm_reference"])
    fov_matches = abs(float(size_nm) - reference_fov) <= 1e-9 * max(
        1.0, abs(reference_fov)
    )
    if n != reference_n or not fov_matches:
        u_nm, valid = lf_lib.resample_field(
            u_nm, valid, reference_fov, n, float(size_nm)
        )
    corrected, n_out, half, magnitude = lf_lib.warp_by_field(
        role_map,
        u_nm,
        float(size_nm),
        valid=valid,
        pad=int(payload["pad"]),
        order=int(payload["order"]),
    )
    return corrected, n_out, half, magnitude


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------


def _logmag(F: np.ndarray) -> np.ndarray:
    return np.log(np.abs(F).astype(np.float64, copy=False))


def save_array_image(out_png: Path, mat, vmin, vmax, cmap: str, png_dpi=300):
    """Write ``mat`` **1:1**: one PNG pixel per array element, nothing else.

    No colour bar, no axes, ticks or title, no cropping (``pad_inches=0`` and no
    ``bbox_inches``): the canvas is exactly ``n_rows x n_cols`` pixels, so the PNG
    can be used directly (overlaid, averaged, fed to another tool) without
    stripping a frame.  ``interpolation="nearest"`` keeps the mapping exact.
    ``vmin``/``vmax`` are the caller's chosen window -- never re-derived here.
    ``png_dpi`` only sets the physical-size metadata: the canvas is ``n_rows x
    n_cols`` pixels whatever it is (300 dpi, same as the LF family and the
    user's reference series).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = int(mat.shape[0]), int(mat.shape[1])
    # No-data pixels (the corrected canvas is NaN outside the valid displacement
    # field) get the Lawler-Fujita family's opaque bad colour instead of the
    # default transparent/white, so the picture is a plain opaque RGB image with
    # no white frame -- same convention as stm_lf_correct.py:1029
    # (cmap_bad.set_bad(color=lf.BAD_COLOR); find it by that anchor if the number
    # drifts).
    cmap_bad = plt.get_cmap(cmap).copy()
    cmap_bad.set_bad(lf_lib.BAD_COLOR)
    fig = plt.figure(
        figsize=(n_cols / png_dpi, n_rows / png_dpi), dpi=png_dpi, frameon=False
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    # Same display convention as the Lawler-Fujita family: row 0 (the smallest
    # physical Y) at the bottom, so corrected maps and the fitted topo agree.  The
    # LF pair there is stm_lf_correct.py:1031/1047 and stm_lf_apply.py:398/414 (the
    # maps' ax.imshow(corrected, cmap=cmap_bad, origin="lower") line and the FFT
    # figure's origin="lower", line; find them by those anchors if the numbers move).
    ax.imshow(
        mat,
        cmap=cmap_bad,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_axis_off()
    fig.savefig(out_png, dpi=png_dpi, pad_inches=0)
    plt.close(fig)


def save_map_image(out_png: Path, mat, vmin, vmax, cmap: str = MAP_CMAP):
    """Real-space map (raw / corrected) -> 1:1 PNG, ``MAP_CMAP``."""
    save_array_image(out_png, mat, vmin, vmax, cmap)


def save_fft_image(out_png: Path, F, window, cmap: str = FFT_CMAP, log_fn=_logmag):
    """FFT plane -> 1:1 PNG of ``log|F|`` under the caller's window, ``FFT_CMAP``."""
    save_array_image(out_png, log_fn(F), window["vmin"], window["vmax"], cmap)


def candidate_window_list(logvals, n):
    """Build ``n`` (vmin, vmax) candidate windows from **one role's** log|F| values.

    ``logvals`` must hold a single role's values: the two roles are different
    units and must never be pooled (that bug is what made every candidate a flat
    colour).

    The windows -- and the ``*_pooled`` clip percentages reported with them --
    come from that role's *pooled* log|F| (every bias plane of the role), because
    that is the distribution the window is applied to.  The candidate *figure*,
    however, can only show one plane, so its own numbers are reported separately
    by ``plane_clip_pct`` (``*_plane``); the two differ slightly and must never
    share one field name.
    """
    nlist = []
    for i in range(n):
        fmin, fmax = _WINDOW_CANDIDATE_FRACS[min(i, len(_WINDOW_CANDIDATE_FRACS) - 1)]
        vmin = float(np.percentile(logvals, fmin))
        vmax = float(np.percentile(logvals, fmax))
        if vmax <= vmin:
            vmax = float(np.max(logvals))
            vmin = float(np.min(logvals))
        clip_below = 100.0 * float(np.mean(logvals < vmin))
        clip_above = 100.0 * float(np.mean(logvals > vmax))
        nlist.append(
            {
                "index": i,
                "vmin": vmin,
                "vmax": vmax,
                "clip_below_pct_pooled": clip_below,
                "clip_above_pct_pooled": clip_above,
                "clip_total_pct_pooled": clip_below + clip_above,
                # the endpoint budget this candidate was designed for (= the two
                # percentile fractions it was built from)
                "clip_budget_pct": fmin + (100.0 - fmax),
            }
        )
    return nlist


def plane_clip_pct(plane_vals, vmin, vmax):
    """``(below %, above %)`` of the plane that is actually drawn for a candidate.

    Data side only: it counts values of the representative plane, not rendered
    pixels (the figure's axes, text and colour bar are not part of it).
    """
    below = 100.0 * float(np.mean(plane_vals < vmin))
    above = 100.0 * float(np.mean(plane_vals > vmax))
    return below, above


def representative_bias_index(planes, pooled):
    """Index of the plane whose log|F| distribution is closest to the role's pooled one.

    The windows come from the pooled distribution (they are applied to every
    plane of the role), but the candidate *picture* has to be an actual plane, so
    it must be the one that represents that distribution -- otherwise the figure
    would be drawn from a different distribution than the window, which is the
    defect this function exists to prevent.
    """
    target = np.array([np.percentile(pooled, q) for q in _REPRESENTATIVE_QUANTILES])
    best, best_score = 0, None
    for i, plane in enumerate(planes):
        finite = plane[np.isfinite(plane)]
        got = np.array([np.percentile(finite, q) for q in _REPRESENTATIVE_QUANTILES])
        score = float(np.abs(got - target).sum())
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def role_fft_log_data(outdir: Path, role: str, window: str = "none"):
    """One role's log|F| planes from ``fft_raw/<role>`` -> (planes, pooled values).

    ``window`` selects which set is read: the candidate table and the candidate
    figures must be built on the **same** planes the final canvas will use, so the
    default spatial window (hamming) is also the default basis here.  The
    unwindowed names never end in ``_<window>``, so the two sets cannot be
    confused by the glob.
    """
    d = outdir / "fft_raw" / role
    pattern = "*_fft2.npy" if window == "none" else f"*_fft2_{window}.npy"
    planes = []
    for path in sorted(d.glob(pattern)):
        F = np.load(path)
        planes.append(np.log(np.abs(F)))
    if not planes:
        raise SystemExit(
            f"ERROR: no {pattern} under {d}; run --window-candidates after the "
            f"raw FFT stage has produced them"
        )
    pooled = np.concatenate([p[np.isfinite(p)] for p in planes])
    return planes, pooled


def render_candidates(candidates, outdir: Path, role: str):
    """Render N candidate FFT images + tiled contact sheet, all same canvas.

    The candidates all show the *same role's* log|F| plane under a different
    (vmin, vmax); with a fixed figure size and no ``bbox_inches`` crop every PNG
    shares one canvas, so they are pixel-for-pixel comparable (asserted below
    with PIL before the contact sheet is assembled).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    sizes = []
    paths = []
    for cand in candidates:
        out = outdir / f"window_{cand['index']:02d}.png"
        plane = np.asarray(cand["_plane"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(
            f"{role} window {cand['index']}: vmin={cand['vmin']:.3g} "
            f"vmax={cand['vmax']:.3g}\n"
            f"clip pooled {cand['clip_below_pct_pooled']:.1f}/"
            f"{cand['clip_above_pct_pooled']:.1f} %  "
            f"plane {cand['clip_below_pct_plane']:.2f}/"
            f"{cand['clip_above_pct_plane']:.2f} %",
            fontsize=8,
        )
        im = ax.imshow(plane, cmap=FFT_CMAP, vmin=cand["vmin"], vmax=cand["vmax"])
        fig.colorbar(im, ax=ax, shrink=0.9)
        fig.savefig(out, dpi=120)
        w, h = Image.open(out).size
        sizes.append((w, h))
        paths.append(out)
        plt.close(fig)

    # every candidate must have the same pixel canvas (PIL-asserted)
    if len(sizes) > 1 and len(set(sizes)) != 1:
        raise SystemExit(f"ERROR: candidate PNG canvas sizes differ: {sorted(sizes)}")
    if not paths:
        return [], None
    w, h = sizes[0]
    cols = int(np.ceil(np.sqrt(len(candidates))))
    rows = int(np.ceil(len(candidates) / cols))
    sheet = Image.new("RGB", (w * cols, h * rows), "white")
    for i, p in enumerate(paths):
        tile = Image.open(p)
        sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
    sheet_path = outdir / "contact_sheet.png"
    sheet.save(sheet_path)
    return paths, sheet_path


def default_fft_window(roles, h5_path, bias_meV, window="none"):
    """Per-role p50..p99.9 window used only as a provisional render fallback.

    Returns ``{role: {"vmin", "vmax"}}``; the roles are never pooled (see
    ``_WINDOW_CANDIDATE_FRACS``).
    """
    with h5py.File(h5_path, "r") as handle:
        data = handle["data"][:]
    windows = {}
    for role_desc in roles:
        role = role_desc["role"]
        chunks = []
        for i in range(len(bias_meV)):
            m = extract_role_map(data, role_desc, i)
            F = compute_fft(m, window)
            chunks.append(np.log(np.abs(F))[np.isfinite(F)])
        pooled = np.concatenate(chunks) if chunks else np.zeros(1)
        windows[role] = {
            "vmin": float(np.percentile(pooled, 50)),
            "vmax": float(np.percentile(pooled, 99.9)),
        }
    return windows


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------


class Pipeline:
    def __init__(
        self,
        h5_path,
        roles,
        outdir,
        size_nm,
        payload,
        u_nm,
        valid,
        windows,
        spatial_window="none",
        candidate_basis=None,
        affine=None,
    ):
        self.h5_path = h5_path
        self.roles = roles
        self.outdir = outdir
        self.size_nm = size_nm
        self.payload = payload
        self.u_nm = u_nm
        self.valid = valid
        # {role: {"vmin", "vmax"}} -- one colour window per role, never shared:
        # the roles carry different units (see _WINDOW_CANDIDATE_FRACS).
        self.windows = windows
        # name of the spatial window ("none" / "hamming" / "hanning") and the list
        # of windowed files actually written (recorded in fft_window.json)
        self.spatial_window = spatial_window
        # which window's planes the candidate table/figures were built on (they
        # follow the spatial window, so this is normally the same value)
        self.candidate_basis = candidate_basis or spatial_window
        # ``None`` = the LF displacement-field branch (unchanged); a geometry dict
        # = the affine branch (``A = P M^-1 P``, no flip, products under <out>/affine)
        self.affine = affine
        # machine-readable provenance of an affine run, written by finish_log()
        self.affine_provenance = None
        self.windowed_files = []
        self.windowed_taps = set()
        self.log_lines = []
        self.log(
            f"pairing rule: forward name + {BWD_MARKER!r} suffix, resolved by "
            f"_bwd_twin()/list.index(name); a role without a twin uses the "
            f"forward map alone"
        )
        for r in self.roles:
            self.log(
                f"extract {r['role']}: fwd {r['fwd']!r} (index "
                f"{r['fwd_idx']}) + bwd {r['back']!r} (index {r['back_idx']}) "
                f"-> nanmean; unit {r['unit']!r}"
            )
        for role, win in self.windows.items():
            if win["vmin"] is not None:
                self.log(
                    f"fft color window {role}: [{win['vmin']:.6g}, {win['vmax']:.6g}] "
                    f"(per role, never pooled across roles)"
                )
        self.log(
            f"spatial FFT window: {self.spatial_window!r} "
            f"(applied to the map before fft2 as the outer product of the 1D "
            f"window; 'none' keeps the unwindowed files)"
        )

    def window_for(self, role):
        win = self.windows.get(role)
        if win is None or win.get("vmin") is None:
            raise SystemExit(f"ERROR: no FFT colour window selected for role {role!r}")
        return win

    def save_windowed_plane(
        self, directory, label, role, arr, plain_plane, log_fn=_logmag
    ):
        """Write the windowed ``.npy`` + 1:1 PNG and report its endpoint clipping.

        The windowed set is written *next to* the unwindowed one (same family
        directory) with the window name in the file name.  The colour window is
        the role's already-chosen one -- never recomputed here -- and the plane's
        clipping is compared with the unwindowed plane: the candidates the user
        picked from were unwindowed, so a windowed plane can clip noticeably more.
        """
        win = self.spatial_window
        self.windowed_taps.add(int(arr.shape[0]))
        npy = directory / windowed_npy_name(label, win)
        png = directory / windowed_png_name(label, win)
        np.save(npy, arr)
        save_fft_image(png, arr, self.window_for(role), log_fn=log_fn)
        for path in (npy, png):
            self.windowed_files.append(str(path.relative_to(self.outdir)))
        colour = self.window_for(role)
        b1, a1 = plane_clip_pct(log_fn(arr), colour["vmin"], colour["vmax"])
        b0, a0 = plane_clip_pct(plain_plane, colour["vmin"], colour["vmax"])
        delta = (b1 + a1) - (b0 + a0)
        self.log(
            f"fft window clip {directory.name} {role} {label}: "
            f"windowed {b1:.3f}/{a1:.3f} % vs unwindowed {b0:.3f}/{a0:.3f} % "
            f"(total delta {delta:+.3f} pp)"
        )
        if abs(delta) > WINDOW_CLIP_WARN_PP:
            # the candidates follow the spatial window, so the *calibrated* set is
            # the windowed one; the unwindowed figures (or, if the candidates were
            # made with another --spatial-window, the windowed ones) are off basis
            basis = self.candidate_basis
            off = "unwindowed" if basis == win else f"{win}-windowed"
            self.log(
                f"WARNING fft window {directory.name} {role} {label}: the colour "
                f"window is calibrated on the {basis} planes; the {off} plane pins "
                f"{b0 + a0:.3f} % of its pixels at a colour endpoint vs "
                f"{b1 + a1:.3f} % for the {win}-windowed plane ({delta:+.3f} pp > "
                f"{WINDOW_CLIP_WARN_PP:.2f} pp) -- expect the off-basis figures to "
                f"clip differently"
            )

    def log(self, msg):
        self.log_lines.append(msg)
        print(msg)

    def run_extract(self, write=True):
        """Stage a: per-role maps per bias point -> raw CSV + PNG + data held.

        ``write=False`` is the affine branch, which keeps the maps in memory only:
        that branch writes the c-e stages, so it must leave no ``raw/`` (and no
        ``fft_raw/``) behind -- there is then no raw plane either for a colour
        candidate run to be built on, which is why it is fail-closed there.
        """
        with h5py.File(self.h5_path, "r") as handle:
            data = handle["data"][:]
        bias_meV = load_bias_meV(self.h5_path)
        buf = {}
        for role_desc in self.roles:
            role = role_desc["role"]
            if write:
                d = self.outdir / "raw" / role
                d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                m = extract_role_map(data, role_desc, i)
                label = f"{i}_{round(b)}meV"
                if write:
                    np.savetxt(d / f"{label}.csv", m, delimiter=",", fmt="%.10e")
                # raw map colormap: nanmedian +/- 5 sigma of that map
                finite = m[np.isfinite(m)]
                med = float(np.median(finite))
                std = float(np.std(finite))
                win = (med - 5 * std, med + 5 * std)
                if write:
                    save_map_image(d / f"{label}.png", m, win[0], win[1])
                buf[(role, i)] = m
                self.log(
                    f"raw {role} {i} meV {round(b)}: median +/- 5 sigma window "
                    f"[{win[0]:.6g}, {win[1]:.6g}] shape {m.shape}"
                    + ("" if write else " (in memory only: this branch writes c-e)")
                )
        return buf, bias_meV

    def run_raw_fft(self, maps, bias_meV):
        """Stage b: complex FFT per raw map -> npy + log|F| plot."""
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "fft_raw" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                m = maps[(role, i)]
                F = compute_fft(m)
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", F)
                save_fft_image(d / f"{label}_fft.png", F, self.window_for(role))
                if self.spatial_window != "none":
                    self.save_windowed_plane(
                        d, label, role, compute_fft(m, self.spatial_window), _logmag(F)
                    )
                self.log(
                    f"fft_raw {role} {i} meV {round(b)}: "
                    f"F[{self.window_for(role)['vmin']:.3g},"
                    f"{self.window_for(role)['vmax']:.3g}]"
                )

    def run_correct(self, raw_maps, bias_meV):
        """Stage c: correct every map (LF resample or affine resample) -> CSV+PNG+FFT."""
        corrected = {}
        for role_desc in self.roles:
            role = role_desc["role"]
            cdir = self.outdir / "corrected" / role
            fdir = self.outdir / "fft_corrected" / role
            cdir.mkdir(parents=True, exist_ok=True)
            fdir.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                m = raw_maps[(role, i)]
                if self.affine is None:
                    corr, n_out, half, mag = apply_transform(
                        m,
                        self.size_nm,
                        self.payload,
                        self.u_nm,
                        self.valid,
                    )
                else:
                    corr, n_out = apply_affine(m, self.affine)
                label = f"{i}_{round(b)}meV"
                np.savetxt(cdir / f"{label}.csv", corr, delimiter=",", fmt="%.10e")
                finite = corr[np.isfinite(corr)]
                med = float(np.median(finite))
                std = float(np.std(finite))
                save_map_image(
                    cdir / f"{label}.png", corr, med - 5 * std, med + 5 * std
                )
                FC = compute_fft(corr)
                np.save(fdir / f"{label}_fft2.npy", FC)
                save_fft_image(fdir / f"{label}_fft.png", FC, self.window_for(role))
                FCW = None
                if self.spatial_window != "none":
                    FCW = compute_fft(corr, self.spatial_window)
                    self.save_windowed_plane(fdir, label, role, FCW, _logmag(FC))
                corrected[(role, i)] = (FC, corr, FCW)
                if self.affine is None:
                    self.log(
                        f"correct {role} {i}: canvas {m.shape[0]}->{n_out}, half "
                        f"{half}, max|u| {mag:.3g} px"
                    )
                elif self.affine["fallback"]:
                    self.log(
                        f"correct {role} {i}: canvas {m.shape[0]}->{n_out} "
                        f"(identity_fallback: the map is copied, no resampling, no "
                        f"NaN added)"
                    )
                else:
                    self.log(
                        f"correct {role} {i}: canvas {m.shape[0]}->{n_out}, affine "
                        f"A = P M^-1 P applied in the h5 frame (no flip), pad "
                        f"{self.affine['pad']}, order {self.affine['order']}"
                    )
                self.log(
                    f"fft_corrected {role} {i}: F[{self.window_for(role)['vmin']:.3g},"
                    f"{self.window_for(role)['vmax']:.3g}]"
                )
        return corrected

    def run_z3(self, corrected, bias_meV):
        """Stage d: Z3 = (F + R120 F + R240 F)/3 in the complex domain."""
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "z3" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                F, _, FW = corrected[(role, i)]
                Z = z3_symmetrize(F)
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", Z)
                save_fft_image(d / f"{label}_fft.png", Z, self.window_for(role))
                if FW is not None:
                    ZW = z3_symmetrize(FW)
                    self.save_windowed_plane(d, label, role, ZW, _logmag(Z))
                    self.log(
                        f"z3 windowed {role} {i}: residual_in {_residual_metric(FW):.4g} "
                        f"-> residual_out {_residual_metric(ZW):.4g}; "
                        f"energy_ratio {_energy_ratio(ZW, FW):.4f}"
                    )
                dc_out = _dc_preservation(Z, F)
                # reference: the 120 deg rotation must itself leave the DC pixel of
                # the real spectrum alone.  ``offset = c - R c`` makes output(c) =
                # input(c) by construction, so this ratio is 1.0000 for any input;
                # anchoring at the (N-1)/2 geometric centre instead gives a value
                # that swings with canvas size and data (measured 0.085 / 0.42 /
                # 0.92 / 1.23 on the same spectra at different N) -- no single
                # number is a usable reference, so treat this ratio as a
                # construction check only, not as evidence.
                dc_rot = _dc_preservation(rotate_about_dc(F, 120.0), F)
                resid_in = _residual_metric(F)
                resid_out = _residual_metric(Z)
                energy = _energy_ratio(Z, F)
                self.log(
                    f"z3 {role} {i}: DC_retained {dc_out:.4f} "
                    f"(R120 keeps {dc_rot:.4f} of the input DC); "
                    f"residual_in {resid_in:.4g} -> residual_out {resid_out:.4g}; "
                    f"energy_ratio {energy:.4f}"
                )

    def run_c3dc(self, corrected, bias_meV):
        """Stage e: C3dc = (|F| + R120 |F| + R240 |F|)/3 (modulus first)."""
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "c3dc" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                F, _, FW = corrected[(role, i)]
                C = c3dc_symmetrize(F)
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", C)
                if FW is not None:
                    CW = c3dc_symmetrize(FW)
                    self.save_windowed_plane(
                        d,
                        label,
                        role,
                        CW,
                        _logmag(C),
                        log_fn=lambda x: np.log(np.abs(x)),
                    )
                    self.log(
                        f"c3dc windowed {role} {i}: residual_in "
                        f"{_residual_metric(np.abs(FW)):.4g} -> residual_out "
                        f"{_residual_metric(CW):.4g}; energy_ratio "
                        f"{_energy_ratio(CW, np.abs(FW)):.4f}"
                    )
                neg = int(np.count_nonzero(C < 0))
                neg_mag = float(np.abs(C[C < 0]).max()) if neg else 0.0
                scale = float(np.max(np.abs(C))) if C.size else 1.0
                rel_neg = neg_mag / scale if scale > 0 else 0.0
                # log|C3dc| -- negatives are spline-ringing, take abs inside log
                save_fft_image(
                    d / f"{label}_fft.png",
                    C,
                    self.window_for(role),
                    log_fn=lambda x: np.log(np.abs(x)),
                )
                dc_out = _dc_preservation(C, np.abs(F))
                # reference: the rotation applied to the real |F| must itself leave
                # the DC pixel alone (see run_z3).  Passing |F| twice would be a
                # self-comparison; this form is not one, but it is still 1.0000 by
                # construction -- a construction check, not evidence.
                dc_rot = _dc_preservation(rotate_about_dc(np.abs(F), 120.0), np.abs(F))
                resid_in = _residual_metric(np.abs(F))
                resid_out = _residual_metric(C)
                energy = _energy_ratio(C, np.abs(F))
                self.log(
                    f"c3dc {role} {i}: DC_retained {dc_out:.4f} "
                    f"(R120 keeps {dc_rot:.4f} of the input DC); "
                    f"residual_in {resid_in:.4g} -> residual_out {resid_out:.4g}; "
                    f"energy_ratio {energy:.4f}; neg_pixels {neg}, "
                    f"neg_rel_mag {rel_neg:.2e}"
                )

    def announce_affine(self, bundle_json, payload, n_in):
        """Log + build the provenance document of one affine run (single source).

        Every number here is stated once and reused by both the ``pipeline.log``
        section and ``affine_provenance.json``, so the two cannot drift.  The
        declarations the branch owes its reader are part of the document:
        ``applied_in_the_h5_frame_no_flip`` and ``anchor_self_check_re_run = False``
        -- the fit's ``anchor_verdict`` is passed through as it is and is *not*
        re-verified here.
        """
        geometry = self.affine
        n_out = (
            n_in
            if geometry["fallback"]
            else image_transform(geometry["affine_q"], n_in, geometry["pad"])[1]
        )
        size_out = float(self.size_nm) * n_out / n_in

        def shown(value):
            return "missing" if value is None else f"{value:.6g}"

        doc = {
            "branch": "affine",
            "transform_file": str(bundle_json),
            "schema": payload.get("schema"),
            "schema_version": payload.get("schema_version"),
            "method": geometry["method"],
            "bundle_fallback_flag": bool(payload.get("fallback", False)),
            "corrected_this_run": not geometry["fallback"],
            "affine_q": [[float(v) for v in row] for row in geometry["affine_q"]],
            "pad": int(geometry["pad"]),
            "order": int(geometry["order"]),
            "n_px_in": int(n_in),
            "n_out": int(n_out),
            "field_of_view_nm": float(self.size_nm),
            "corrected_field_of_view_nm": float(size_out),
            "n_labelled": payload.get("n_labelled"),
            "anchor_ring": payload.get("anchor_ring"),
            "anchor_verdict": payload.get("anchor_verdict"),
            "stretch_scale_sqrt_det": payload.get("stretch_scale_sqrt_det"),
            "a_nm": payload.get("a_nm"),
            "a_ref_nm": payload.get("a_ref_nm"),
            "source_report": payload.get("source_report"),
            "source_input": payload.get("source_input"),
            "applied_in_the_h5_frame_no_flip": True,
            "anchor_self_check_re_run": False,
            "notes": [
                "applied in the h5 frame, no flip",
                "anchor self-check NOT re-run",
            ],
        }
        self.affine_provenance = doc
        self.log(f"# affine transform: {doc['transform_file']}")
        self.log(
            f"#   schema {doc['schema']} v{doc['schema_version']}; method "
            f"{doc['method']}; bundle fallback flag {doc['bundle_fallback_flag']} "
            f"-> corrected_this_run {doc['corrected_this_run']}"
        )
        self.log(f"#   affine_q M = {doc['affine_q']}")
        self.log(
            f"#   pad {doc['pad']}, order {doc['order']}; canvas {doc['n_px_in']} -> "
            f"{doc['n_out']} px; field of view {doc['field_of_view_nm']:g} -> "
            f"{doc['corrected_field_of_view_nm']:.6g} nm"
        )
        self.log(
            f"#   fit provenance: n_labelled {doc['n_labelled']}, anchor_ring "
            f"{doc['anchor_ring']}, anchor_verdict {doc['anchor_verdict']}, stretch "
            f"scale |det M|^(1/2) {shown(doc['stretch_scale_sqrt_det'])}"
        )
        self.log(
            "#   applied in the h5 frame, no flip (A = P M^-1 P with the "
            "_image_transform offset, straight on the map read from /data)"
        )
        self.log(
            "#   anchor self-check NOT re-run (the verdict above is the fit stage's "
            "and is the only evidence for this transform; the products below are "
            "corrected by it, not verified by it)"
        )
        if geometry["fallback"]:
            self.log(
                "# WARNING affine: method = identity_fallback -- the fit solved "
                "nothing, so the maps are copied unchanged (n_out = n, field of view "
                "unchanged, no NaN added); do not read the corrected/z3/c3dc "
                "products as corrected"
            )

    def finish_log(self, include_window=True):
        with (self.outdir / "pipeline.log").open("w") as fh:
            fh.write("\n".join(self.log_lines) + ("\n" if self.log_lines else ""))
        if self.affine_provenance is not None:
            with (self.outdir / AFFINE_PROVENANCE_FILE).open("w") as fh:
                json.dump(self.affine_provenance, fh, indent=2)
                fh.write("\n")
        if include_window:
            # ``roles`` is the colour window (read back by later runs); the
            # ``spatial_window`` block records which data window this run applied
            # and the exact list of windowed files it wrote.
            with (self.outdir / FFT_WINDOW_FILE).open("w") as fh:
                json.dump(
                    {
                        "window_version": 2,
                        "roles": self.windows,
                        "spatial_window": {
                            "name": self.spatial_window,
                            "suffix": None
                            if self.spatial_window == "none"
                            else f"_{self.spatial_window}",
                            "candidate_basis": self.candidate_basis,
                            "taps": sorted(self.windowed_taps),
                            "files": self.windowed_files,
                        },
                    },
                    fh,
                    indent=2,
                )
                fh.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --fft-window is ONLY the colour window (two floats).  A single token (e.g.
    # ``--fft-window 3``) is a usage error, and the message must say where the
    # spatial window lives now -- one flag with two meanings bit us once already
    # (see SKILL.md 2.5), so keep the failure loud and pointed.
    _error = parser.error

    def _error_with_spatial_hint(message):
        if "--fft-window" in message and "--spatial-window" not in message:
            message = (
                f"{message}  --fft-window is the FFT COLOUR window and takes "
                f"exactly two floats VMIN VMAX; to pick a SPATIAL window use "
                f"--spatial-window {{{', '.join(sorted(SPATIAL_WINDOWS))}}}."
            )
        _error(message)

    parser.error = _error_with_spatial_hint
    parser.add_argument("input", help="grid .h5 file")
    parser.add_argument(
        "--map", action="append", default=[], metavar='ROLE="CHANNEL NAME"'
    )
    parser.add_argument(
        "--transform",
        required=True,
        help="LF displacement-field bundle JSON (Skill A, with u_field_file) or "
        "affine-correction transform bundle JSON (schema topo-correction-transform), "
        "or a directory holding one of them",
    )
    parser.add_argument("-o", "--outdir", required=True, help="output directory")
    parser.add_argument(
        "--window-candidates",
        type=int,
        default=None,
        metavar="N",
        help="generate N FFT color-scale candidate figures + contact sheet, "
        "print the table, exit 0 (no window chosen yet); N is required here "
        "(there is no default)",
    )
    parser.add_argument(
        "--pick-window",
        type=int,
        default=None,
        metavar="K",
        help="pick candidate K from the saved --window-candidates table, write "
        "fft_window.json and run the full chain",
    )
    parser.add_argument(
        "--spatial-window",
        choices=sorted(SPATIAL_WINDOWS),
        default=DEFAULT_SPATIAL_WINDOW,
        help="spatial window applied to the map BEFORE fft2, as the outer product "
        f"of the 1D window (default {DEFAULT_SPATIAL_WINDOW!r}).  Every FFT family "
        "then writes two sets: the unwindowed files keep their names, the windowed "
        "ones add the window name (e.g. 0_50meV_fft2_hamming.npy / "
        "0_50meV_fft_hamming.png); 'none' writes only the unwindowed set.  The "
        "candidate table and figures are built on this window's planes, so the "
        "candidates you choose from are the canvas you get.",
    )
    parser.add_argument(
        "--fft-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="explicit FFT COLOUR window applied to EVERY role; write "
        "fft_window.json and run the full chain.  The roles carry different "
        "units, so this only does what you want when they happen to match -- use "
        "--fft-window-role for a per-role window.  Takes exactly two floats; the "
        "spatial window is a separate flag (--spatial-window).",
    )
    parser.add_argument(
        "--fft-window-role",
        action="append",
        nargs=3,
        default=[],
        metavar=("ROLE", "VMIN", "VMAX"),
        help="explicit FFT color window for one role (repeatable; overrides "
        "--fft-window for that role)",
    )
    return parser.parse_args(argv)


def _scan_size_nm(h5_path: Path) -> float:
    with h5py.File(h5_path, "r") as handle:
        raw = None
        scan = handle.get("header/Scan")
        if scan is not None and "Scanfield" in scan.attrs:
            raw = scan.attrs["Scanfield"]
        if raw is None and "Grid settings" in handle["header"].attrs:
            raw = handle["header"].attrs["Grid settings"]
        if raw is None:
            raise SystemExit(
                f"ERROR: {h5_path} carries no Scanfield / Grid settings; "
                f"cannot determine L (nm)"
            )
    fields = str(raw).split(";")
    if len(fields) <= 2:
        raise SystemExit(f"ERROR: cannot read a scan size out of {raw!r}")
    # The h5 header stores the field as a float32 decimal (e.g. "3e-8" -> 30 nm
    # comes back as 29.999999999999996).  Round to nm-level precision so the
    # grid-consistency test against the bundle's reference field of view is a
    # real comparison instead of a bit-for-bit float equality.
    return round(float(fields[2]) * 1e9, 9)


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.input)
    if not src.is_file():
        raise SystemExit(f"ERROR: {src} does not exist")
    require_z_column(src)

    # ---- which branch: LF displacement field or affine stretch ----------
    bundle_json = resolve_transform(args.transform)
    probe = read_json_object(bundle_json)
    kind = classify_transform(probe, bundle_json)
    affine_mode = kind == "affine"
    root_outdir = Path(args.outdir)
    parent_window_file = root_outdir / FFT_WINDOW_FILE
    if affine_mode:
        # Fail-closed BEFORE anything is created: this branch writes no raw/fft_raw
        # plane, so there is no plane set for a candidate table, and a --pick-window
        # would read a table that can never exist here.
        if args.window_candidates is not None:
            raise SystemExit(
                f"ERROR: --window-candidates N has no meaning in the affine branch: "
                f"that branch writes only the c-e stages under "
                f"{root_outdir / AFFINE_DIRNAME}, so no fft_raw/<role>/*_fft2.npy "
                f"exists for a candidate table to be built on.  It inherits the "
                f"per-role colour window from {parent_window_file} (or from its own "
                f"{root_outdir / AFFINE_DIRNAME / FFT_WINDOW_FILE} once written); to "
                f"override it use --fft-window-role ROLE VMIN VMAX (or --fft-window "
                f"VMIN VMAX for every role)."
            )
        if args.pick_window is not None:
            raise SystemExit(
                f"ERROR: --pick-window K has no meaning in the affine branch: the "
                f"candidate table is written by a --window-candidates run, which "
                f"this branch does not have (it writes no fft_raw planes).  Inherit "
                f"the window from {parent_window_file} or pass "
                f"--fft-window-role ROLE VMIN VMAX."
            )
    outdir = root_outdir / AFFINE_DIRNAME if affine_mode else root_outdir
    outdir.mkdir(parents=True, exist_ok=True)
    # Every affine product is prefixed with <out>/affine/ by moving the pipeline's
    # output root; the LF branch keeps the historical flat layout.

    if affine_mode:
        geometry = affine_geometry(probe, bundle_json)
        payload, u_nm, valid, u_path = probe, None, None, None
    else:
        geometry = None
        payload, u_nm, valid, u_path = load_transform(bundle_json)
    roles = resolve_roles(src, args.map)
    size_nm = _scan_size_nm(src)

    # ---- decide the window and mode ------------------------------------
    # A window is always a {role: (vmin, vmax)} table: the roles carry different
    # units, so one shared window is meaningless (see _WINDOW_CANDIDATE_FRACS).
    role_names = [r["role"] for r in roles]
    window_file = outdir / FFT_WINDOW_FILE
    # An explicit command-line entry ALWAYS beats the file on disk: fft_window.json
    # records the *previous* selection, and letting it win (the old behaviour)
    # silently ignored --pick-window, so re-picking meant deleting the file by hand.
    explicit = (
        args.pick_window is not None
        or args.fft_window is not None
        or bool(args.fft_window_role)
    )
    # ``--window-candidates N`` is a request to REDO the table, so it takes
    # priority over the saved selection too: letting the on-disk fft_window.json
    # win made the whole candidate run a silent no-op (exit 0, nothing written).
    candidates_mode = args.window_candidates is not None
    if candidates_mode and explicit:
        raise SystemExit(
            "ERROR: --window-candidates N cannot be combined with --pick-window / "
            "--fft-window / --fft-window-role: the candidate run exits before the "
            "full chain (it only produces the table to pick from).  Run "
            "--window-candidates N first, then --pick-window K."
        )
    pre_notes = []
    if explicit and window_file.is_file():
        try:
            previous = json.loads(window_file.read_text())
        except OSError:
            previous = None
        except ValueError:
            previous = None
        if not isinstance(previous, dict):
            was = "unreadable content"
        elif "roles" in previous:
            was = ", ".join(
                f"{r} [{w['vmin']:g}, {w['vmax']:g}]"
                for r, w in previous["roles"].items()
            )
        else:
            was = f"legacy single window [{previous['vmin']:g}, {previous['vmax']:g}]"
        if args.pick_window is not None:
            entry = f"--pick-window {args.pick_window}"
            ignored = [
                flag
                for flag, given in (
                    ("--fft-window", args.fft_window is not None),
                    ("--fft-window-role", bool(args.fft_window_role)),
                )
                if given
            ]
            suffix = (
                f" ({' and '.join(ignored)} also given but ignored, as before)"
                if ignored
                else ""
            )
        else:
            parts = []
            if args.fft_window is not None:
                parts.append(
                    f"--fft-window {args.fft_window[0]:g} {args.fft_window[1]:g}"
                )
            parts += [
                f"--fft-window-role {spec[0]} {float(spec[1]):g} {float(spec[2]):g}"
                for spec in args.fft_window_role
            ]
            entry = " + ".join(parts)
            suffix = ""
        pre_notes.append(
            f"# NOTE: {window_file} existed with {was}; overridden by {entry}{suffix} "
            f"(the file is rewritten with this selection)"
        )
    if candidates_mode:
        cand_file = outdir / CANDIDATES_FILE
        if cand_file.is_file():
            pre_notes.append(
                f"# NOTE: {cand_file} is recomputed and overwritten by "
                f"--window-candidates {args.window_candidates} "
                f"(basis --spatial-window {args.spatial_window})"
            )
        if window_file.is_file():
            pre_notes.append(
                f"# NOTE: {window_file} exists and is left untouched by the "
                f"candidate run; the next --pick-window / --fft-window* run "
                f"overwrites it with the new selection"
            )
        mode = "candidates"
        windows = {r: {"vmin": None, "vmax": None} for r in role_names}
    elif window_file.is_file() and not explicit:
        windows = read_window_file(window_file, role_names)
        mode = "full"
    elif args.pick_window is not None:
        cand_file = outdir / CANDIDATES_FILE
        if not cand_file.is_file():
            raise SystemExit(
                f"ERROR: --pick-window {args.pick_window} needs the saved "
                f"candidate table {cand_file}, produced by a --window-candidates "
                f"run first"
            )
        cands = json.loads(cand_file.read_text())
        if "roles" not in cands:
            raise SystemExit(
                f"ERROR: {cand_file} holds a pooled (pre per-role) candidate "
                f"table, whose windows are wrong for every role; rerun "
                f"--window-candidates N to regenerate it per role"
            )
        windows = {}
        for role in role_names:
            entries = cands["roles"].get(role, {}).get("candidates")
            if not entries:
                raise SystemExit(
                    f"ERROR: {cand_file} has no candidate table for role {role!r} "
                    f"(it holds {sorted(cands['roles'])})"
                )
            if not (0 <= args.pick_window < len(entries)):
                raise SystemExit(
                    f"ERROR: --pick-window {args.pick_window} out of range "
                    f"[0, {len(entries) - 1}] for role {role!r} in {cand_file}"
                )
            windows[role] = {
                "vmin": entries[args.pick_window]["vmin"],
                "vmax": entries[args.pick_window]["vmax"],
            }
        mode = "full"
    elif args.fft_window is not None or args.fft_window_role:
        if args.fft_window is not None:
            vmin, vmax = args.fft_window
            if not (vmin < vmax):
                raise SystemExit(
                    f"ERROR: --fft-window requires VMIN < VMAX, got {vmin},{vmax}"
                )
            windows = {r: {"vmin": vmin, "vmax": vmax} for r in role_names}
        else:
            windows = {}
        for spec in args.fft_window_role:
            role, vmin, vmax = spec[0], float(spec[1]), float(spec[2])
            if role not in role_names:
                raise SystemExit(
                    f"ERROR: --fft-window-role names unknown role {role!r}; this "
                    f"run has {role_names}"
                )
            if not (vmin < vmax):
                raise SystemExit(
                    f"ERROR: --fft-window-role {role} requires VMIN < VMAX, got "
                    f"{vmin},{vmax}"
                )
            windows[role] = {"vmin": vmin, "vmax": vmax}
        missing = [r for r in role_names if r not in windows]
        if missing:
            raise SystemExit(
                f"ERROR: no window given for role(s) {missing}; pass --fft-window "
                f"VMIN VMAX (all roles) or --fft-window-role ROLE VMIN VMAX"
            )
        mode = "full"
    else:
        # The affine branch has no candidate run of its own: it inherits the LF
        # run's per-role window file one level up, and writes this run's selection
        # back into its own <out>/affine/fft_window.json (finish_log does that).
        if affine_mode and parent_window_file.is_file():
            windows = read_window_file(parent_window_file, role_names)
            mode = "full"
            print(
                f"# affine branch: inheriting the per-role colour windows from "
                f"{parent_window_file} (written back to {window_file})"
            )
        elif affine_mode:
            raise SystemExit(
                f"ERROR: no FFT color window selected and none on disk.  The affine "
                f"branch inherits the per-role windows from {parent_window_file} "
                f"(or from its own {window_file}); run the LF branch first, or pass "
                f"--fft-window-role ROLE VMIN VMAX (or --fft-window VMIN VMAX for "
                f"every role)."
            )
        else:
            raise SystemExit(
                "ERROR: no FFT color window selected and none on disk. Run "
                "--window-candidates N first (choose from the printed table), then "
                "--pick-window K (or --fft-window VMIN VMAX, or one "
                "--fft-window-role ROLE VMIN VMAX per role), or place "
                f"{FFT_WINDOW_FILE} in the output directory."
            )

    if affine_mode:
        print(f"# grid: {src}")
        print(f"# transform: {bundle_json} (affine; products under {outdir})")
    else:
        print(f"# grid: {src}")
        print(f"# transform: {bundle_json} -> {u_path}")
    print(f"# field of view: {size_nm} nm")
    print(
        f"# spatial FFT window: {args.spatial_window}"
        + (
            " (unwindowed files unchanged; windowed copies add "
            f"'_{args.spatial_window}' to the name)"
            if args.spatial_window != "none"
            else " (only the unwindowed set is written)"
        )
    )

    # which window's planes the picked colour window came from: read it back from
    # the candidate table when there is one (it follows the spatial window, so a
    # mismatch means the candidates were made with a different --spatial-window)
    cand_file = outdir / CANDIDATES_FILE
    candidate_basis = args.spatial_window
    if cand_file.is_file():
        try:
            doc = json.loads(cand_file.read_text())
        except OSError:
            doc = {}
        except ValueError:  # json.JSONDecodeError is a ValueError
            doc = {}
        # flat key first (t18), then the nested block (t16); a table written
        # before either carries no basis at all -- then the basis is unknown and
        # the reader must be told rather than silently assumed to match
        recorded = doc.get("candidate_basis")
        if recorded is None:
            recorded = doc.get("spatial_window", {}).get("name")
        if recorded is None:
            print(
                f"# NOTE: {cand_file} carries no candidate_basis field (written "
                f"before the basis was recorded); its windows may have been "
                f"calibrated on another spatial window than "
                f"{args.spatial_window!r}"
            )
        elif recorded != args.spatial_window:
            candidate_basis = recorded
            print(
                f"# NOTE: the colour window was picked from candidates built on "
                f"the {candidate_basis!r} planes, but this run draws "
                f"{args.spatial_window!r} planes"
            )

    pipe = Pipeline(
        src,
        roles,
        outdir,
        size_nm,
        payload,
        u_nm,
        valid,
        windows,
        spatial_window=args.spatial_window,
        candidate_basis=candidate_basis,
        affine=geometry,
    )
    for note in pre_notes:
        pipe.log(note)

    if affine_mode:
        # The affine branch writes only the c-e stages: it extracts the maps in
        # memory, states its provenance, then corrects and symmetrizes.
        with h5py.File(src, "r") as handle:
            n_in = int(handle["data"].shape[0])
        pipe.announce_affine(bundle_json, probe, n_in)
        raw_maps, bias_meV = pipe.run_extract(write=False)
        corrected = pipe.run_correct(raw_maps, bias_meV)
        pipe.run_z3(corrected, bias_meV)
        pipe.run_c3dc(corrected, bias_meV)
        pipe.finish_log(include_window=True)
        print(
            f"# affine branch done; products in {outdir}; per-role windows: "
            + ", ".join(
                f"{role} [{windows[role]['vmin']:.3g}, {windows[role]['vmax']:.3g}]"
                for role in role_names
            )
        )
        print(f"# provenance: {outdir / AFFINE_PROVENANCE_FILE}")
        return 0

    # stages a + b always run first (extraction + raw FFT)
    raw_maps, bias_meV = pipe.run_extract()
    if mode == "candidates":
        # No window chosen yet: render the b-stage fft_raw plots with the
        # provisional per-role window so they are viewable; the candidate figures
        # give the user the final choices and the full rerun redraws every FFT
        # plot with the picked window.
        pipe.windows = default_fft_window(roles, src, bias_meV, args.spatial_window)
        for role, win in pipe.windows.items():
            pipe.log(
                f"provisional fft color window {role}: "
                f"[{win['vmin']:.6g}, {win['vmax']:.6g}] (this role's p50..p99.9; "
                f"the picked window replaces it)"
            )
    pipe.run_raw_fft(raw_maps, bias_meV)

    if mode == "candidates":
        # ---- candidate color-scale generation, ONE SET PER ROLE ----------
        n = args.window_candidates
        cand_dir = outdir / "window_candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        # the flat window_NN.png / contact_sheet.png files this script wrote
        # before windows became per-role would otherwise be mistaken for the new
        # candidates; remove exactly those legacy names.
        for stale in sorted(cand_dir.glob("window_[0-9][0-9].png")) + sorted(
            cand_dir.glob("contact_sheet.png")
        ):
            stale.unlink()
            pipe.log(f"removed legacy pooled candidate figure {stale}")
        saved_roles = {}
        for role_desc in roles:
            role = role_desc["role"]
            # The candidates must live on the SAME planes the final canvas uses,
            # so the basis is the current spatial window (default hamming).  The
            # pure (unwindowed) planes are read too, only to report what the same
            # window would pin there -- never to build the table or the figures.
            planes, pooled = role_fft_log_data(outdir, role, args.spatial_window)
            if args.spatial_window == "none":
                planes_none = planes
            else:
                planes_none, _ = role_fft_log_data(outdir, role, "none")
                if len(planes_none) != len(planes):
                    raise SystemExit(
                        f"ERROR: {role}: {len(planes)} windowed vs "
                        f"{len(planes_none)} unwindowed fft_raw planes; rerun the "
                        f"whole stage so both sets cover the same bias points"
                    )
            cands = candidate_window_list(pooled, n)
            rep = representative_bias_index(planes, pooled)
            plane_vals = planes[rep][np.isfinite(planes[rep])]
            plane_none_vals = planes_none[rep][np.isfinite(planes_none[rep])]
            for c in cands:
                c["_plane"] = planes[rep]
                # the numbers that belong to the canvas: measured on the very
                # plane this candidate draws, not on the pooled distribution the
                # window percentiles come from
                below, above = plane_clip_pct(plane_vals, c["vmin"], c["vmax"])
                c["clip_below_pct_plane"] = below
                c["clip_above_pct_plane"] = above
                c["clip_total_pct_plane"] = below + above
                c["plane_over_budget_pct"] = (
                    c["clip_total_pct_plane"] - c["clip_budget_pct"]
                )
                # same window, same bias, but on the unwindowed plane (what the
                # window would pin if it had been picked there); reported for
                # comparison only
                nb, na = plane_clip_pct(plane_none_vals, c["vmin"], c["vmax"])
                c["clip_below_pct_plane_none"] = nb
                c["clip_above_pct_plane_none"] = na
                c["clip_total_pct_plane_none"] = nb + na
            paths, sheet = render_candidates(cands, cand_dir / role, role)
            saved_roles[role] = {
                "unit": role_desc["unit"],
                "pooled_log_p50": float(np.percentile(pooled, 50)),
                "representative_bias_index": rep,
                "representative_bias_meV": float(bias_meV[rep]),
                "plane_clip_tolerance_pct": PLANE_CLIP_TOL_PCT,
                "contact_sheet": str(sheet),
                "figures": [str(p) for p in paths],
                "candidates": [
                    {k: v for k, v in c.items() if k != "_plane"} for c in cands
                ],
            }
            for c in cands:
                if c["plane_over_budget_pct"] > PLANE_CLIP_TOL_PCT:
                    pipe.log(
                        f"WARNING candidates {role} window {c['index']:02d}: the drawn "
                        f"plane pins {c['clip_total_pct_plane']:.3f} % of its pixels at a "
                        f"window endpoint, {c['plane_over_budget_pct']:.3f} pp above its "
                        f"{c['clip_budget_pct']:.2f} % design budget (tolerance "
                        f"{PLANE_CLIP_TOL_PCT:.2f} pp); pooled-side numbers are "
                        f"{c['clip_below_pct_pooled']:.2f}/{c['clip_above_pct_pooled']:.2f} %. "
                        f"Window values are NOT changed (vmin={c['vmin']:.6g} "
                        f"vmax={c['vmax']:.6g}, still the pooled percentiles)"
                    )
            pipe.log(
                f"candidates {role} ({role_desc['unit']}): log|F| over "
                f"{len(planes)} planes, p50 {np.percentile(pooled, 50):.3f}, "
                f"preview plane {rep} ({round(bias_meV[rep])} meV), "
                f"{len(cands)} windows -> {sheet}"
            )
        saved = {
            "source": str(src),
            # flat, unambiguous basis marker: the planes this table was built on
            # (asserted by value in SKILL.md 8/10)
            "candidate_basis": args.spatial_window,
            "spatial_window": {
                "name": args.spatial_window,
                "candidate_basis": args.spatial_window,
            },
            "roles": saved_roles,
        }
        (outdir / CANDIDATES_FILE).write_text(json.dumps(saved, indent=2) + "\n")

        print("\nCandidate FFT color windows, per role (choose with --pick-window K):")
        basis = args.spatial_window
        print(
            "column meanings: vmin/vmax = the window (percentiles of the role's "
            "pooled log|F|);\n"
            f"  pooled<,> = pixels outside the window in that pooled distribution "
            f"(what the window is designed for);\n"
            f"  plane_{basis}<,> = pixels outside the window on the plane the "
            f"candidate figure actually draws ({basis} basis -- the number that "
            f"belongs to the canvas);\n"
            f"  plane_none<,> = the same window measured on the unwindowed plane of "
            f"the same bias point (for comparison; equal to the above when "
            f"--spatial-window none);\n"
            f"  budget = the design budget of that candidate; over = plane total - "
            f"budget (marked ! and warned about when it exceeds "
            f"{PLANE_CLIP_TOL_PCT:.2f} pp)."
        )
        for role in role_names:
            info = saved_roles[role]
            print(
                f"\n-- {role} (unit {info['unit']!r}; pooled log|F| p50 "
                f"{info['pooled_log_p50']:.3f}; preview plane bias "
                f"{info['representative_bias_index']} = "
                f"{info['representative_bias_meV']:g} meV)"
            )
            print(
                f"{'#':>3} {'vmin':>11} {'vmax':>11} "
                f"{'pooled< %':>9} {'pooled> %':>9} "
                f"{f'plane_{basis}< %':>14} {f'plane_{basis}> %':>14} "
                f"{'plane_none< %':>13} {'plane_none> %':>13} "
                f"{'budget %':>9} {'over pp':>8}"
            )
            for c in info["candidates"]:
                flag = " !" if c["plane_over_budget_pct"] > PLANE_CLIP_TOL_PCT else ""
                print(
                    f"{c['index']:>3} {c['vmin']:>11.4g} {c['vmax']:>11.4g} "
                    f"{c['clip_below_pct_pooled']:>9.2f} {c['clip_above_pct_pooled']:>9.2f} "
                    f"{c['clip_below_pct_plane']:>14.3f} {c['clip_above_pct_plane']:>14.3f} "
                    f"{c['clip_below_pct_plane_none']:>13.3f} "
                    f"{c['clip_above_pct_plane_none']:>13.3f} "
                    f"{c['clip_budget_pct']:>9.2f} {c['plane_over_budget_pct']:>+8.3f}{flag}"
                )
        print(
            "\nRerun with --pick-window K (candidate K of every role) or "
            "--fft-window-role ROLE VMIN VMAX (per role) to run the full chain."
        )
        pipe.finish_log(include_window=False)
        return 0

    # ---- full chain -----------------------------------------------------
    corrected = pipe.run_correct(raw_maps, bias_meV)
    pipe.run_z3(corrected, bias_meV)
    pipe.run_c3dc(corrected, bias_meV)
    pipe.finish_log(include_window=True)
    print(
        f"# full chain done; products in {outdir}; per-role windows: "
        + ", ".join(
            f"{role} [{windows[role]['vmin']:.3g}, {windows[role]['vmax']:.3g}]"
            for role in role_names
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
