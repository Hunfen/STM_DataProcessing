#!/usr/bin/env python
"""Symmetrized-FFT pipeline for grid spectroscopy maps (Skill B, stage 2).

This is the **second stage** of the ``grid-sym-fft`` skill.  It takes the role
mapping produced by the sibling ``grid_channels.py`` (roles applied to channel
names), and for every bias point of the grid runs the full chain::

    a. extract   each role = mean(forward channel, its [bwd] twin)
    b. raw FFT   F = fftshift(fft2(x)), x = map with NaN filled by nanmedian
    c. correct   resample the map along the LF displacement field (flipud +
                 resampling ONLY, target values untouched) -> corrected FFT
    d. Z3        Z = (F + R120 F + R240 F)/3 in the complex domain, rotation
                 about the DC pixel (the only place the spectrum is rotated)
    e. C3(dc)    A = |F|; C3dc = (A + R120 A + R240 A)/3 (the only difference
                 from Z3 is the modulus is taken first)

Every stage writes its data and an image (data next to image, one directory per
quantity), and every FFT-producing stage writes complex128 ``_fft2.npy`` plus a
``log|...|`` plot.  The correction stage uses Skill A's transferable
displacement-field bundle and only re-samples: ``corrected(r) = T(r + u(r))``
after a coordinate-only ``flipud`` (see ``scripts/apply_lf_transform.py`` in
the repo root for the convention).  It never touches the target values -- no
``stm_lf_apply.py`` (that one removes a plane), only ``lf_lib.warp_by_field``.

**Two-run color-scale protocol.**  The user looks at real candidate figures
before committing to a FFT colormap (they dislike formula-derived windows):

1. Run ``--window-candidates N`` (N is required with this flag): stages a+b
   run and N candidate FFT images + a contact sheet are written into
   ``window_candidates/``, the candidate table
   is printed and saved to ``<OUTDIR>/fft_window_candidates.json``, then the
   script exits 0 (no window chosen yet).
2. Run ``--pick-window K`` (or ``--fft-window VMIN VMAX``): the full chain
   a+b+c+d+e runs, ``<OUTDIR>/fft_window.json`` is written, and **all** FFT
   plots (fft_raw / fft_corrected / z3 / c3dc) share that one window.

If ``fft_window.json`` already exists it is reused (idempotent).  Running the
full chain with no window selected yet is fail-closed.

**Fail-closed.**  A missing/ambiguous channel name, a ``--transform`` that is
missing or not unique inside its directory, a grid h5 without ``/params``'s
``Z (m)`` column, or requesting the full chain before a window is chosen all
exit non-zero without writing a half-finished pipeline.

Usage::

    .venv/bin/python <this script> GRID.h5 \\
        --map current="Current (A)" --map didv="DSP 7280 Y (%)" \\
        --transform <dir|bundle.json> -o OUT [--window-candidates 9]
    (then)  ... -o OUT --pick-window 3
    (or)    ... -o OUT --fft-window 0 6.5
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

# Color-scale candidate generation (see contain below): percentile fractions of
# the pooled log|F| data, vmin ascending, vmax ascending.  Index 0 is the
# tightest/darkest window, the last is the brightest.
_VMIN_FRAC = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
_VMAX_FRAC = [0.90, 0.95, 0.98, 0.99, 0.995, 0.997, 0.999, 0.9995, 0.9999]


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

    A directory must contain exactly one ``*_transform.json``; zero or more
    than one is fail-closed.
    """
    p = Path(transform)
    if p.is_file():
        return p
    if p.is_dir():
        matches = sorted(p.glob("*_transform.json"))
        if len(matches) == 0:
            raise SystemExit(
                f"ERROR: {transform} contains no *_transform.json bundle "
                f"(found: {[x.name for x in p.iterdir()]})"
            )
        if len(matches) > 1:
            raise SystemExit(
                f"ERROR: {transform} contains {len(matches)} *_transform.json "
                f"bundles; pass the exact file instead. Found: "
                f"{[m.name for m in matches]}"
            )
        return matches[0]
    raise SystemExit(f"ERROR: --transform {transform} does not exist")


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


def compute_fft(x: np.ndarray) -> np.ndarray:
    """Complex128, fftshifted, NaN filled by nanmedian of valid pixels."""
    return np.fft.fftshift(np.fft.fft2(_nanmedian_fill(x))).astype(np.complex128)


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
    """flipud + resample along u -- never a value-domain change.

    Mirrors ``scripts/apply_lf_transform.py``: the grid maps and the fitted
    topography share one frame (``/data`` and ``/params`` were written from the
    same flat pixel index), and the LF field is fitted on the top-down topo, so
    the map is flipped into ``u``'s index frame before ``warp_by_field``.  That
    flip is a coordinate convention, not a value operation.
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
    array = np.flipud(role_map)
    corrected, n_out, half, magnitude = lf_lib.warp_by_field(
        array,
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


def plot_matrix(fig, ax, mat, vmin, vmax, cmap="viridis", title=""):
    ax.set_title(title, fontsize=9)
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    return im, cbar


def save_raw_image(out_png: Path, mat, vmin, vmax, title, dpi=120):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_matrix(fig, ax, mat, vmin, vmax, cmap="viridis", title=title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def save_fft_image(out_png: Path, F, window, title, dpi=120, log_fn=_logmag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logv = log_fn(F)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_matrix(
        fig, ax, logv, window["vmin"], window["vmax"], cmap="viridis", title=title
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def candidate_window_list(logvals_flat, n):
    """Build ``n`` (vmin, vmax) candidate windows from pooled log|F| values."""
    fracs_min = _VMIN_FRAC
    fracs_max = _VMAX_FRAC
    nlist = []
    for i in range(n):
        fmin = fracs_min[min(i, len(fracs_min) - 1)] * 100.0
        fmax = fracs_max[min(i, len(fracs_max) - 1)] * 100.0
        if fmax <= fmin:
            fmax = fmin + 1.0
        vmin = float(np.percentile(logvals_flat, fmin))
        vmax = float(np.percentile(logvals_flat, fmax))
        clip_below = 100.0 * float(np.mean(logvals_flat < vmin))
        clip_above = 100.0 * float(np.mean(logvals_flat > vmax))
        nlist.append(
            {
                "index": i,
                "vmin": vmin,
                "vmax": vmax,
                "clip_below_pct": clip_below,
                "clip_above_pct": clip_above,
            }
        )
    return nlist


def render_candidates(candidates, outdir: Path):
    """Render N candidate FFT images + tiled contact sheet, all same canvas.

    The candidate figures all show the same pooled log|F| plane under a
    different (vmin, vmax); with a fixed figure size and no ``bbox_inches``
    crop every PNG shares one canvas, so they are pixel-for-pixel comparable
    (asserted below with PIL before the contact sheet is assembled).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = []
    paths = []
    for cand in candidates:
        out = outdir / f"window_{cand['index']:02d}.png"
        plane = np.asarray(cand["_plane"])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(
            f"window {cand['index']}: vmin={cand['vmin']:.3g} "
            f"vmax={cand['vmax']:.3g}  clip {cand['clip_below_pct']:.1f}/"
            f"{cand['clip_above_pct']:.1f} %",
            fontsize=8,
        )
        im = ax.imshow(plane, cmap="viridis", vmin=cand["vmin"], vmax=cand["vmax"])
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


def default_fft_window(roles, h5_path, bias_meV):
    """Pooled p50..p99.9 window used only as a provisional render fallback."""
    with h5py.File(h5_path, "r") as handle:
        data = handle["data"][:]
    pooled = []
    for role_desc in roles:
        for i in range(len(bias_meV)):
            m = extract_role_map(data, role_desc, i)
            F = compute_fft(m)
            pooled.append(np.log(np.abs(F))[np.isfinite(F)])
    pooled = np.concatenate(pooled) if pooled else np.zeros(1)
    return {
        "vmin": float(np.percentile(pooled, 50)),
        "vmax": float(np.percentile(pooled, 99.9)),
    }


# --------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------


class Pipeline:
    def __init__(self, h5_path, roles, outdir, size_nm, payload, u_nm, valid, window):
        self.h5_path = h5_path
        self.roles = roles
        self.outdir = outdir
        self.size_nm = size_nm
        self.payload = payload
        self.u_nm = u_nm
        self.valid = valid
        self.window = window
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

    def log(self, msg):
        self.log_lines.append(msg)
        print(msg)

    def run_extract(self):
        """Stage a: per-role maps per bias point -> raw CSV + PNG + data held."""
        with h5py.File(self.h5_path, "r") as handle:
            data = handle["data"][:]
        bias_meV = load_bias_meV(self.h5_path)
        buf = {}
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "raw" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                m = extract_role_map(data, role_desc, i)
                label = f"{i}_{round(b)}meV"
                np.savetxt(d / f"{label}.csv", m, delimiter=",", fmt="%.10e")
                # raw map colormap: nanmedian +/- 5 sigma of that map
                finite = m[np.isfinite(m)]
                med = float(np.median(finite))
                std = float(np.std(finite))
                win = (med - 5 * std, med + 5 * std)
                save_raw_image(
                    d / f"{label}.png",
                    m,
                    win[0],
                    win[1],
                    f"{role} {i} bias {round(b)} meV  "
                    f"(raw, median +/- 5 sigma [{win[0]:.3g}, {win[1]:.3g}])",
                )
                buf[(role, i)] = m
                self.log(
                    f"raw {role} {i} meV {round(b)}: median +/- 5 sigma window "
                    f"[{win[0]:.6g}, {win[1]:.6g}] shape {m.shape}"
                )
        return buf, bias_meV

    def run_raw_fft(self, maps, bias_meV):
        """Stage b: complex FFT per raw map -> npy + log|F| plot."""
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "fft_raw" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                F = compute_fft(maps[(role, i)])
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", F)
                save_fft_image(
                    d / f"{label}_fft.png",
                    F,
                    self.window,
                    f"{role} {i} bias {round(b)} meV  |FFT| (raw)",
                )
                self.log(
                    f"fft_raw {role} {i} meV {round(b)}: "
                    f"F[{self.window['vmin']:.3g},{self.window['vmax']:.3g}]"
                )

    def run_correct(self, raw_maps, bias_meV):
        """Stage c: correct every map (flipud+resample only) -> CSV+PNG+FFT."""
        corrected = {}
        for role_desc in self.roles:
            role = role_desc["role"]
            cdir = self.outdir / "corrected" / role
            fdir = self.outdir / "fft_corrected" / role
            cdir.mkdir(parents=True, exist_ok=True)
            fdir.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                m = raw_maps[(role, i)]
                corr, n_out, half, mag = apply_transform(
                    m,
                    self.size_nm,
                    self.payload,
                    self.u_nm,
                    self.valid,
                )
                label = f"{i}_{round(b)}meV"
                np.savetxt(cdir / f"{label}.csv", corr, delimiter=",", fmt="%.10e")
                finite = corr[np.isfinite(corr)]
                med = float(np.median(finite))
                std = float(np.std(finite))
                save_raw_image(
                    cdir / f"{label}.png",
                    corr,
                    med - 5 * std,
                    med + 5 * std,
                    f"{role} {i} bias {round(b)} meV  (LF corrected)",
                )
                FC = compute_fft(corr)
                np.save(fdir / f"{label}_fft2.npy", FC)
                save_fft_image(
                    fdir / f"{label}_fft.png",
                    FC,
                    self.window,
                    f"{role} {i} bias {round(b)} meV  |FFT| (LF corrected)",
                )
                corrected[(role, i)] = (FC, corr)
                self.log(
                    f"correct {role} {i}: canvas {m.shape[0]}->{n_out}, half {half}, "
                    f"max|u| {mag:.3g} px"
                )
                self.log(
                    f"fft_corrected {role} {i}: F[{self.window['vmin']:.3g},"
                    f"{self.window['vmax']:.3g}]"
                )
        return corrected

    def run_z3(self, corrected, bias_meV):
        """Stage d: Z3 = (F + R120 F + R240 F)/3 in the complex domain."""
        for role_desc in self.roles:
            role = role_desc["role"]
            d = self.outdir / "z3" / role
            d.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(bias_meV):
                F, _ = corrected[(role, i)]
                Z = z3_symmetrize(F)
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", Z)
                save_fft_image(
                    d / f"{label}_fft.png",
                    Z,
                    self.window,
                    f"{role} {i} bias {round(b)} meV  log|Z| (C3, complex)",
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
                F, _ = corrected[(role, i)]
                C = c3dc_symmetrize(F)
                label = f"{i}_{round(b)}meV"
                np.save(d / f"{label}_fft2.npy", C)
                neg = int(np.count_nonzero(C < 0))
                neg_mag = float(np.abs(C[C < 0]).max()) if neg else 0.0
                scale = float(np.max(np.abs(C))) if C.size else 1.0
                rel_neg = neg_mag / scale if scale > 0 else 0.0
                # log|C3dc| -- negatives are spline-ringing, take abs inside log
                save_fft_image(
                    d / f"{label}_fft.png",
                    C,
                    self.window,
                    f"{role} {i} bias {round(b)} meV  log|C3dc| "
                    f"(neg {neg} px, mag {rel_neg:.2e})",
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

    def finish_log(self, include_window=True):
        with (self.outdir / "pipeline.log").open("w") as fh:
            fh.write("\n".join(self.log_lines) + ("\n" if self.log_lines else ""))
        if include_window:
            with (self.outdir / FFT_WINDOW_FILE).open("w") as fh:
                json.dump(self.window, fh, indent=2)
                fh.write("\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="grid .h5 file")
    parser.add_argument(
        "--map", action="append", default=[], metavar='ROLE="CHANNEL NAME"'
    )
    parser.add_argument(
        "--transform",
        required=True,
        help="Skill A transform bundle JSON, or a directory holding one",
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
        "--fft-window",
        nargs=2,
        type=float,
        default=None,
        metavar=("VMIN", "VMAX"),
        help="explicit FFT color window; write fft_window.json and run the full chain",
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
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle_json = resolve_transform(args.transform)
    payload, u_nm, valid, u_path = load_transform(bundle_json)
    roles = resolve_roles(src, args.map)
    size_nm = _scan_size_nm(src)

    # ---- decide the window and mode ------------------------------------
    window_file = outdir / FFT_WINDOW_FILE
    if window_file.is_file():
        window = json.loads(window_file.read_text())
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
        entries = cands["candidates"]
        if not (0 <= args.pick_window < len(entries)):
            raise SystemExit(
                f"ERROR: --pick-window {args.pick_window} out of range "
                f"[0, {len(entries) - 1}] from {cand_file}"
            )
        window = {
            "vmin": entries[args.pick_window]["vmin"],
            "vmax": entries[args.pick_window]["vmax"],
        }
        mode = "full"
    elif args.fft_window is not None:
        vmin, vmax = args.fft_window
        if not (vmin < vmax):
            raise SystemExit(
                f"ERROR: --fft-window requires VMIN < VMAX, got {vmin},{vmax}"
            )
        window = {"vmin": vmin, "vmax": vmax}
        mode = "full"
    elif args.window_candidates is not None:
        mode = "candidates"
        window = {"vmin": None, "vmax": None}  # filled after candidate generation
    else:
        raise SystemExit(
            "ERROR: no FFT color window selected and none on disk. Run "
            "--window-candidates N first (choose from the printed table), then "
            "--pick-window K (or --fft-window VMIN VMAX), or place "
            f"{FFT_WINDOW_FILE} in the output directory."
        )

    print(f"# grid: {src}")
    print(f"# transform: {bundle_json} -> {u_path}")
    print(f"# field of view: {size_nm} nm")

    pipe = Pipeline(src, roles, outdir, size_nm, payload, u_nm, valid, window)

    # stages a + b always run first (extraction + raw FFT)
    raw_maps, bias_meV = pipe.run_extract()
    if mode == "candidates":
        # No window chosen yet: render the b-stage fft_raw plots with a
        # provisional pooled window so they are viewable; the candidate figures
        # give the user the final choices and the full rerun redraws every FFT
        # plot with the picked window.
        pipe.window = default_fft_window(roles, src, bias_meV)
    pipe.run_raw_fft(raw_maps, bias_meV)

    if mode == "candidates":
        # ---- candidate color-scale generation --------------------------
        n = args.window_candidates
        # pool log|F| from all raw FFT maps to pick meaningful range
        pooled = []
        for role_desc in roles:
            d = outdir / "fft_raw" / role_desc["role"]
            for i in range(len(bias_meV)):
                path = d / f"{i}_{round(bias_meV[i])}meV_fft2.npy"
                F = np.load(path)
                pooled.append(np.log(np.abs(F))[np.isfinite(F)])
        pooled = np.concatenate(pooled)
        cands = candidate_window_list(pooled, n)
        # attach the first raw FFT plane for comparable candidate rendering
        first = None
        for role_desc in roles:
            d = outdir / "fft_raw" / role_desc["role"]
            p0 = d / f"0_{round(bias_meV[0])}meV_fft2.npy"
            if p0.is_file():
                first = np.log(np.abs(np.load(p0)))
                break
        for c in cands:
            c["_plane"] = first
        cand_dir = outdir / "window_candidates"
        cand_dir.mkdir(parents=True, exist_ok=True)
        render_candidates(cands, cand_dir)
        # persist candidate table (mini-json without the plane)
        saved = {
            "source": str(src),
            "candidates": [
                {k: v for k, v in c.items() if k != "_plane"} for c in cands
            ],
        }
        (outdir / CANDIDATES_FILE).write_text(json.dumps(saved, indent=2) + "\n")

        print("\nCandidate FFT color windows (choose with --pick-window K):")
        print(f"{'#':>3} {'vmin':>12} {'vmax':>12} {'clip< %':>8} {'clip> %':>8}")
        for c in cands:
            print(
                f"{c['index']:>3} {c['vmin']:>12.4g} {c['vmax']:>12.4g} "
                f"{c['clip_below_pct']:>8.2f} {c['clip_above_pct']:>8.2f}"
            )
        print(
            "\nRerun with --pick-window K (or --fft-window VMIN VMAX) to "
            "run the full chain with that window."
        )
        pipe.finish_log(include_window=False)
        return 0

    # ---- full chain -----------------------------------------------------
    corrected = pipe.run_correct(raw_maps, bias_meV)
    pipe.run_z3(corrected, bias_meV)
    pipe.run_c3dc(corrected, bias_meV)
    pipe.finish_log(include_window=True)
    print(
        f"# full chain done; products in {outdir}; window: "
        f"[{window['vmin']:.3g}, {window['vmax']:.3g}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
