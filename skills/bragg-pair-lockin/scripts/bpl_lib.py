"""Bragg-pair lock-in: local fields and real-space maps for arbitrary q(h, k).

The layer implemented here is four steps

    a set of Bragg points q  ->  the local (complex) field at q  ->  the
    real-space map B(r) of that reflection pair  ->  the phase-difference
    coherence C(dr) of that point's own field (xi / residual phase ramp)

and nothing else: no cross-point integration, no pairwise or triple-product
statistics and no figure atlas (those belong to the sibling ``phase-analysis``
skill).

Two equivalent ways of giving a point are accepted and both are printed
everywhere:

    (h, k)              ``--q 1/3,1/3``      q = h b1 + k b2
    n x basis + index   ``--g 3:1,1``        q = (1 b1 + 1 b2) / 3

``(h, k)`` is the single internal truth (an exact ``Fraction``), so the two
spellings of one point agree bit for bit.

Engine
------
The local fields are produced by the sibling skill ``local-q-map``: its
``localqmap`` module is imported **in-process** (never copied, never run in a
subprocess) and ``localqmap``'s engine is the only place where a local field is
computed in this skill.  ``h5io`` of the same skill supplies the HDF5 writing
convention (gzip/4, explicit chunks, schema/generator attributes).

Routes
------
``analytic``      B(r) = 2 Re[ psi_q(r) exp(i q.r) ], the default;
``two_sided``     B(r) = Re{ IFFT[ FFT[T] * (G(k - q) + G(k + q)) ] }, the strict
                  band-pass on the raw spectrum, used when one of the three
                  applicability criteria of the analytic form fails.

The criteria are reported per point with their numbers and thresholds, so the
choice of route is always reproducible from the report alone.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt

SKILL_VERSION = "1.0"
GENERATOR = "stm_bragg_pair_lockin.py"

ROUTE_ANALYTIC = "analytic"
ROUTE_TWO_SIDED = "two_sided"

#: The factor two of the analytic reconstruction.  ``psi_q`` carries ``A/2``
#: for a map containing ``A cos(q.r + phi)``, so the q-component of the map is
#: ``2 Re[psi exp(i q.r)]``.
ANALYTIC_FACTOR = 2.0

#: The analytic form is accepted only when the Friedel partner is suppressed to
#: at most this weight in the demodulated frame, ``exp(-2 (lambda |q|)^2)``.
FRIEDEL_SUPPRESSION_MAX = 1e-3
FRIEDEL_EXPONENT_MIN = float(-math.log(FRIEDEL_SUPPRESSION_MAX))

#: The envelope must change by at most this fraction of its own size over one
#: carrier cycle.
SLOW_VARIATION_MAX = 1.0

#: Default cap of the denominator of the rational (h, k) attribution.
DEFAULT_DENOMINATOR_MAX = 12
#: Rows of the candidate table closer than this relative radius share a ring.
DEFAULT_RING_TOL = 0.05
#: |q_measured - q_rational| / |q| below this marks a commensurate point.
COMMENSURATE_REL_TOL = 0.02
#: Mask threshold as a fraction of the median amplitude (sibling convention).
DEFAULT_AMPLITUDE_FRACTION = 0.1
DEFAULT_DELIMITER = ","
#: Detector candidates kept for the candidate table.
MAX_CANDIDATES = 512

STM_LIB_DEFAULT = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"

#: The four figures of the generalized step set, one set per selected point.
STEP_FIGURES = (
    "step1_fft_marks.png",
    "step2_q_zoom.png",
    "step3_modulated_field.png",
    "step4_lowpassed_field.png",
    "step5_phase_coherence.png",
)

FORMULA_ANALYTIC = "B(r) = 2 Re[ psi_q(r) exp(i q.r) ]"
FORMULA_TWO_SIDED = "B(r) = Re{ IFFT[ FFT[T] * (G(k - q) + G(k + q)) ] }"

EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_DEGENERATE = 4
#: ``--replot`` refuses to draw from products that do not match the report.
EXIT_REPLOT = 5

# --------------------------------------------------------------------------- #
# step 5 (phase coherence) constants
# --------------------------------------------------------------------------- #
#: ``|C|`` is sampled at these fractions of ``max_lag_nm``.
COHERENCE_ANCHOR_FRACTIONS = (0.10, 0.25, 0.50, 0.75)
#: Pixels closer than this multiple of lambda to an invalid pixel are dropped.
COHERENCE_BORDER_FACTOR = 0.65
COHERENCE_MAX_LAG_RULE = "L / sqrt(2)"
COHERENCE_FIT_RULE = "(2/3) * max_lag_nm"
COHERENCE_DEFINITION = (
    "C(dr) = <exp(i[theta(r)-theta(r+dr)])> over valid pairs; A/N normalisation"
)
COHERENCE_CRITERION = (
    "hidden order present <=> |C(dr)| stays close to 1 up to the field scale"
)
#: Lattice constant and valley coefficient of the pseudo-gauge amplitude:
#: ``A = 2 beta / (sqrt(3) a_nm)`` in nm^-1 (notes/kekule_phase_extraction.md
#: section 7: 1 % uniaxial strain -> 2|A| ~ 0.028 Angstrom^-1).
#: Lattice constant (nm) and valley coefficient of the pseudo-gauge amplitude.
#: Everything else below is *computed* from these two numbers: no seven digit
#: constant of ``A`` (or of its divisor) is written down anywhere in this file.
STRAIN_A0_NM = 0.246
STRAIN_BETA = 3.0
#: ``A = 2 beta / (sqrt(3) a_nm)`` in nm^-1 (notes/kekule_phase_extraction.md
#: section 7: 1 % uniaxial strain -> 2|A| ~ 0.028 Angstrom^-1).
STRAIN_A_NM_INV = 2.0 * STRAIN_BETA / (math.sqrt(3.0) * STRAIN_A0_NM)
#: ``1 nm^-1 = 10 Angstrom^-1``, so ``slope_per_angstrom = slope_rad_per_nm / 10``
#: and ``strain = |slope_per_angstrom| / (2 * A)``.  The divisor is written as
#: ``2 * 10 * A = 20 * A`` so that the rad/nm and Angstrom^-1 forms are exactly
#: equal.
STRAIN_A_PER_ANGSTROM_INV = STRAIN_A_NM_INV / 10.0
STRAIN_DIVISOR_PER_NM = 20.0 * STRAIN_A_PER_ANGSTROM_INV
#: Kept for the historical comparison only: the one-off script used 1.4082.
STRAIN_A_HISTORICAL_PER_ANGSTROM_INV = 1.4082
STRAIN_CONVERSION = (
    "strain = |slope_per_angstrom| / (2 * A) = |slope_rad_per_nm| / (20 * A) "
    f"with A = 2 * beta / (sqrt(3) * a_nm) = {STRAIN_A_PER_ANGSTROM_INV:.6f} "
    f"Angstrom^-1 and 20 * A = {STRAIN_DIVISOR_PER_NM:.5f} per nm"
)
STRAIN_CONVERSION_NOTE = (
    f"A = 2 * beta / (sqrt(3) * a_nm) = {STRAIN_A_NM_INV:.9f} nm^-1 = "
    f"{STRAIN_A_PER_ANGSTROM_INV:.9f} Angstrom^-1 with a_nm = {STRAIN_A0_NM} nm "
    f"and beta = {STRAIN_BETA}; the rad/nm divisor is 20 * A = "
    f"{STRAIN_DIVISOR_PER_NM:.9f} per nm.  The historical one-off script used the "
    "rounded A = 1.4082 (and read rad/nm as rad/Angstrom), so its strain is the "
    f"correct one times {STRAIN_A_HISTORICAL_PER_ANGSTROM_INV / STRAIN_A_PER_ANGSTROM_INV:.9f}"
    " = 10 * (1 + 2.0308e-05)"
)
#: The ramp cut is capped at this many times the smallest mask inset (px): a
#: circular autocorrelation folds the far side of the canvas into ``arg C`` once
#: the lag exceeds twice the inset of the valid region.
COHERENCE_CAP_RULE = "min(L / 2, 2 * min(inset_x_px, inset_y_px)) * nm_per_px"
#: A lag is only valid when its pair count reaches this floor: the FFT of a 0/1
#: mask leaves ~1e-10 of floating point residue where the true count is zero.
COHERENCE_PAIR_FLOOR = 0.5


# --------------------------------------------------------------------------- #
# the engine of the sibling skill local-q-map (in-process, never reimplemented)
# --------------------------------------------------------------------------- #
def fail(message, code=EXIT_USAGE):
    """Report an error on stderr and exit with ``code`` (never a tuple code).

    ``SystemExit`` is raised with a single integer, so the process status is
    exactly ``code``; passing a message and a code to ``SystemExit`` at once
    would turn the status into a tuple and collapse to 1.
    """
    print(f"# ERROR: {message}", file=sys.stderr, flush=True)
    raise SystemExit(int(code))


def local_q_map_scripts():
    """Directory holding the engine of the sibling skill ``local-q-map``.

    The skills are installed side by side, so the engine sits either next to
    this skill directory (``<skills>/bragg-pair-lockin/scripts`` ->
    ``<skills>/local-q-map/scripts``, which covers a repository checkout *and*
    the installed library ``~/.agents/skills``) or in the installed library
    while this skill runs from a scratch tree.  Both are searched; nothing is
    copied and the engine is never modified.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "local-q-map" / "scripts",
        Path.home() / ".agents" / "skills" / "local-q-map" / "scripts",
    ]
    for candidate in candidates:
        if (candidate / "localqmap.py").is_file():
            return candidate
    fail(
        "the engine of the sibling skill local-q-map was not found; expected "
        + " or ".join(str(path) for path in candidates)
    )


LQ_SCRIPTS = local_q_map_scripts()
if str(LQ_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(LQ_SCRIPTS))

import h5io  # noqa: E402
import localqmap as lq  # noqa: E402

TWO_PI = lq.TWO_PI


def local_field(image, q_rad_px, lambda_nm, nm_per_px):
    """The complex local field of one q, from the sibling skill's engine."""
    return lq.demodulate(image, q_rad_px, lambda_nm, nm_per_px)


# --------------------------------------------------------------------------- #
# input, basis, field of view
# --------------------------------------------------------------------------- #
def load_topo(path, delimiter=DEFAULT_DELIMITER):
    """Square topography matrix plus its NaN fraction (sibling loader)."""
    array, nan_fraction = lq.load_topo(path, delimiter)
    if float(nan_fraction) >= 1.0:
        fail(
            f"{path}: the canvas has no finite pixel (NaN fraction "
            f"{nan_fraction:.6f}); there is nothing to analyse",
            EXIT_INPUT,
        )
    return array, float(nan_fraction)


def resolve_field_of_view(size_nm, size_nm_from_log, basis, n_px):
    """Field of view in nm plus a one-line provenance string.

    Precedence: ``-L``, then ``--size-nm-from-log``, then the correction report
    itself (``corrected_nm_per_px`` times the canvas, else
    ``corrected_field_of_view_nm``).  Neither present is an error.
    """
    if size_nm is not None:
        value = float(size_nm)
        return value, f"command line -L/--size-nm: {value:.4f} nm"
    if size_nm_from_log:
        value, source = lq.field_of_view_from_log(size_nm_from_log)
        return float(value), f"--size-nm-from-log: {value:.4f} nm ({source})"
    report_scale = basis.get("corrected_nm_per_px")
    if report_scale:
        per_px = float(report_scale)
        return per_px * float(n_px), (
            f"correction report corrected_nm_per_px {per_px:.8f} nm/px times "
            f"the {int(n_px)} px canvas"
        )
    report_fov = basis.get("corrected_field_of_view_nm")
    if report_fov:
        return float(report_fov), "correction report corrected_field_of_view_nm"
    fail(
        "a field of view is required: pass -L <nm> or --size-nm-from-log",
        EXIT_USAGE,
    )


# --------------------------------------------------------------------------- #
# the two spellings of a point
# --------------------------------------------------------------------------- #
def parse_q_flag(text):
    """``'H,K'`` -> ``(h, k)`` as exact Fractions (``'1/3'``, ``'0.25'``, ``'2'``)."""
    parts = [item.strip() for item in str(text).split(",")]
    if len(parts) != 2:
        fail(f"--q {text!r}: expected 'H,K', got {len(parts)} field(s)")
    out = []
    for token in parts:
        try:
            out.append(Fraction(token))
        except ValueError, ZeroDivisionError:
            fail(f"--q {text!r}: cannot read the number {token!r}")
    return out[0], out[1]


def parse_g_flag(text):
    """``'N:I,J'`` -> ``(i/N, j/N)`` as exact Fractions (``'3:1,1'``, ``'3:2,-1'``)."""
    body = str(text).strip()
    if ":" not in body:
        fail(f"--g {text!r}: expected 'N:I,J' (for example '3:1,1')")
    scale_text, _, index_text = body.partition(":")
    try:
        scale = int(scale_text)
    except ValueError:
        fail(f"--g {text!r}: the basis multiple {scale_text!r} is not an integer")
    if scale == 0:
        fail(f"--g {text!r}: the basis multiple must not be zero")
    parts = [item.strip() for item in index_text.split(",")]
    if len(parts) != 2:
        fail(f"--g {text!r}: expected two coefficients, got {len(parts)}")
    try:
        indices = [int(item) for item in parts]
    except ValueError:
        fail(f"--g {text!r}: the coefficients must be integers")
    return Fraction(indices[0], scale), Fraction(indices[1], scale)


def fraction_text(value):
    """``'1/3'`` for a Fraction, ``'1'`` for an integer."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def common_denominator(h, k):
    """Smallest d with d*h and d*k integers."""
    h = Fraction(h)
    k = Fraction(k)
    return int(h.denominator * k.denominator / math.gcd(h.denominator, k.denominator))


def writings(h, k):
    """Both spellings of one point, plus the exact command-line flags."""
    h = Fraction(h)
    k = Fraction(k)
    scale = common_denominator(h, k)
    i, j = int(h * scale), int(k * scale)
    hk = f"{fraction_text(h)},{fraction_text(k)}"
    n_basis = f"{scale}:{i},{j}"
    return {
        "hk": hk,
        "n_times_basis": n_basis,
        "cli_hk": f"--q {hk}",
        "cli_n_times_basis": f"--g {n_basis}",
        "text": f"(h, k) = ({fraction_text(h)}, {fraction_text(k)}) = {n_basis} x basis",
    }


def is_self_conjugate(h, k):
    """True when ``2q`` is a reciprocal lattice vector (``q = 0`` included)."""
    return (2 * Fraction(h)).denominator == 1 and (2 * Fraction(k)).denominator == 1


def q_of(h, k, basis):
    """``q = h b1 + k b2`` in nm^-1 (rad/nm) from a resolved basis."""
    b1 = np.asarray(basis["b1_nm_inv"], dtype=float)
    b2 = np.asarray(basis["b2_nm_inv"], dtype=float)
    return float(h) * b1 + float(k) * b2


@dataclass
class Point:
    """One requested Bragg point: the (h, k) truth plus its per-point settings."""

    label: str
    h: Fraction
    k: Fraction
    source: str
    lambda_nm: float
    aliases: list = field(default_factory=list)

    @property
    def writings(self):
        return writings(self.h, self.k)

    @property
    def phase_degenerate(self):
        return is_self_conjugate(self.h, self.k)


def build_points(q_specs, g_specs, labels, lambda_nm, lambda_overrides):
    """Parse both spellings, merge explicit ``-q`` partners, apply per-point lambda.

    Returns ``(points, notices)``.  A point whose exact ``-q`` partner is also
    given is merged into one pair: the first spelling stays the representative
    and the second becomes an alias of the same pair.
    """
    requested = []
    for text in q_specs:
        h, k = parse_q_flag(text)
        requested.append((h, k, f"--q {text}"))
    for text in g_specs:
        h, k = parse_g_flag(text)
        requested.append((h, k, f"--g {text}"))
    if labels and len(labels) != len(requested):
        fail(
            f"--label: {len(labels)} label(s) for {len(requested)} point(s)",
            EXIT_USAGE,
        )
    names = list(labels) if labels else [f"q{index}" for index in range(len(requested))]
    if len(set(names)) != len(names):
        fail(f"--label: duplicated label in {names}", EXIT_USAGE)

    notices = []
    points = []
    by_key = {}
    for name, (h, k, source) in zip(names, requested, strict=True):
        partner_key = (-h, -k)
        if partner_key in by_key:
            host = by_key[partner_key]
            host.aliases.append(
                {
                    "label": name,
                    "h": float(h),
                    "k": float(k),
                    "writings": writings(h, k),
                    "source": source,
                }
            )
            notices.append(
                f"{name} {writings(h, k)['cli_hk']} is the explicit -q partner of "
                f"{host.label} ({host.writings['cli_hk']}): merged into the single "
                f"pair {host.label} / -{host.label}"
            )
            continue
        if (h, k) in by_key:
            host = by_key[(h, k)]
            host.aliases.append(
                {
                    "label": name,
                    "h": float(h),
                    "k": float(k),
                    "writings": writings(h, k),
                    "source": source,
                    "note": "duplicate point",
                }
            )
            notices.append(f"{name} repeats {host.label}: merged")
            continue
        point = Point(
            label=name,
            h=Fraction(h),
            k=Fraction(k),
            source=source,
            lambda_nm=float(lambda_nm),
        )
        by_key[(h, k)] = point
        points.append(point)

    if not points:
        fail("no point was selected (empty point set)", EXIT_USAGE)

    for override in lambda_overrides or []:
        name, _, value_text = str(override).partition("=")
        name = name.strip()
        matched = [point for point in points if point.label == name]
        if not matched:
            aliases = [alias["label"] for point in points for alias in point.aliases]
            hint = (
                " (it is an alias of a merged -q partner; set the lambda on the "
                "representative point)"
                if name in aliases
                else ""
            )
            fail(
                f"--lambda-nm-label {override!r}: no point is labelled {name!r}{hint}",
                EXIT_USAGE,
            )
        try:
            value = float(value_text)
        except ValueError:
            fail(
                f"--lambda-nm-label {override!r}: cannot read the lambda {value_text!r}",
                EXIT_USAGE,
            )
        if not (value > 0.0):
            fail(f"--lambda-nm-label {override!r}: lambda must be positive", EXIT_USAGE)
        matched[0].lambda_nm = value
    return points, notices


# --------------------------------------------------------------------------- #
# candidate table (bragg_peak detector -> rational (h, k) -> rings)
# --------------------------------------------------------------------------- #
def detect_peaks(canvas, size_nm, stm_lib=STM_LIB_DEFAULT):
    """Run the ``bragg_peak`` detector of the sibling skills on the canvas."""
    if str(stm_lib) not in sys.path:
        sys.path.insert(0, str(stm_lib))
    try:
        from stm_data_processing.utils.bragg_peak import detect_bragg_peaks
    except Exception as exc:  # an import problem is reported, not raised
        fail(
            "the candidate table needs the bragg_peak detector "
            f"(stm_data_processing.utils.bragg_peak with --stm-lib {stm_lib}): {exc}",
            EXIT_USAGE,
        )
    return detect_bragg_peaks(
        np.asarray(canvas, dtype=float),
        float(size_nm),
        subtract_plane=False,
        max_candidates=int(MAX_CANDIDATES),
        return_fft2=True,
    )


def sample_magnitude(magnitude, n_px, q_px):
    """Bilinear |F| at a signed fftshift offset (circular, so any q is defined)."""
    center = n_px // 2
    y = (center + float(q_px[1])) % n_px
    x = (center + float(q_px[0])) % n_px
    y0, x0 = int(np.floor(y)), int(np.floor(x))
    y1, x1 = (y0 + 1) % n_px, (x0 + 1) % n_px
    fy, fx = y - y0, x - x0
    return float(
        (magnitude[y0, x0] * (1 - fx) + magnitude[y0, x1] * fx) * (1 - fy)
        + (magnitude[y1, x0] * (1 - fx) + magnitude[y1, x1] * fx) * fy
    )


def solve_hk(q_nm_inv, basis):
    """Real (h, k) with ``q = h b1 + k b2`` (the exact, not rounded, attribution)."""
    b1 = np.asarray(basis["b1_nm_inv"], dtype=float)
    b2 = np.asarray(basis["b2_nm_inv"], dtype=float)
    matrix = np.array([[b1[0], b2[0]], [b1[1], b2[1]]], dtype=float)
    h, k = np.linalg.solve(matrix, np.asarray(q_nm_inv, dtype=float))
    return float(h), float(k)


def rational_hk(h_real, k_real, denominator_max):
    """Nearest fractions with denominator <= ``denominator_max`` (componentwise)."""
    cap = int(denominator_max)
    return (
        Fraction(float(h_real)).limit_denominator(cap),
        Fraction(float(k_real)).limit_denominator(cap),
    )


def cluster_rings(rows, ring_tol):
    """Group candidate rows by radius (ascending, relative tolerance)."""
    rings = []
    for row in sorted(rows, key=lambda item: item["q_nm_inv_abs"]):
        radius = float(row["q_nm_inv_abs"])
        placed = False
        for ring in rings:
            reference = float(ring["radius_nm_inv_mean"])
            if abs(radius - reference) <= float(ring_tol) * reference:
                ring["members"].append(row)
                ring["radius_nm_inv_mean"] = float(
                    np.mean([member["q_nm_inv_abs"] for member in ring["members"]])
                )
                placed = True
                break
        if not placed:
            rings.append({"members": [row], "radius_nm_inv_mean": radius})
    for index, ring in enumerate(rings):
        radii = np.array([member["q_nm_inv_abs"] for member in ring["members"]])
        amplitudes = np.array([member["f_amplitude"] for member in ring["members"]])
        mean_amplitude = float(np.mean(amplitudes))
        ring.update(
            ring_id=f"ring{index}",
            n_members=len(ring["members"]),
            radius_nm_inv_mean=float(np.mean(radii)),
            radius_nm_inv_min=float(np.min(radii)),
            radius_nm_inv_max=float(np.max(radii)),
            radius_spread_relative=float(
                (np.max(radii) - np.min(radii)) / np.mean(radii)
            ),
            f_amplitude_mean=mean_amplitude,
            f_amplitude_max=float(np.max(amplitudes)),
            amplitude_residual_max=float(
                np.max(np.abs(amplitudes - mean_amplitude)) / mean_amplitude
            )
            if mean_amplitude > 0.0
            else float("nan"),
        )
        for row in ring["members"]:
            row["ring_id"] = ring["ring_id"]
            row["ring_radius_nm_inv"] = float(ring["radius_nm_inv_mean"])
            row["amplitude_residual_ring"] = (
                float(abs(row["f_amplitude"] - mean_amplitude) / mean_amplitude)
                if mean_amplitude > 0.0
                else float("nan")
            )
    return rings


def candidate_table(
    canvas,
    size_nm,
    basis,
    *,
    denominator_max=DEFAULT_DENOMINATOR_MAX,
    ring_tol=DEFAULT_RING_TOL,
    lambda_nm=None,
    selected=None,
    stm_lib=STM_LIB_DEFAULT,
    friedel_tolerance_px=2.0,
):
    """The candidate Bragg table of a canvas, in the 1x1 basis of the report.

    Every detected peak gets its (h, k) by a rational approximation with a
    denominator cap, both spellings of that (h, k), its radius, its measured
    |F|, its azimuth, the strength and asymmetry of its Friedel partner, the
    residual of the rational attribution (shown as-is for non-commensurate
    points) and the cross-talk flag against the already selected points.
    """
    detection = detect_peaks(canvas, size_nm, stm_lib=stm_lib)
    peaks = list(detection.peaks)
    n_px = int(np.asarray(canvas).shape[0])
    magnitude = (
        np.abs(detection.fft2)
        if detection.fft2 is not None
        else np.abs(
            np.fft.fftshift(np.fft.fft2(np.nan_to_num(np.asarray(canvas, dtype=float))))
        )
    )
    rows = []
    for index, peak in enumerate(peaks):
        q_px = np.asarray(peak.q_px, dtype=float)
        q_nm_inv = np.asarray(peak.q_nm_inv, dtype=float)
        radius = float(np.hypot(*q_nm_inv))
        h_real, k_real = solve_hk(q_nm_inv, basis)
        h_fraction, k_fraction = rational_hk(h_real, k_real, denominator_max)
        q_rational = q_of(h_fraction, k_fraction, basis)
        residual = float(np.hypot(*(np.asarray(q_nm_inv) - q_rational)))
        record = _friedel_partner(
            peaks, index, q_px, n_px, magnitude, friedel_tolerance_px
        )
        rows.append(
            {
                "peak_index": index,
                "q_px": [float(q_px[0]), float(q_px[1])],
                "q_nm_inv": [float(q_nm_inv[0]), float(q_nm_inv[1])],
                "q_nm_inv_abs": radius,
                "h": float(h_fraction),
                "k": float(k_fraction),
                "h_k_real": [h_real, k_real],
                "h_k_fraction": [str(h_fraction), str(k_fraction)],
                "writings": writings(h_fraction, k_fraction),
                "rational_residual_nm_inv": residual,
                "rational_residual_relative": float(residual / radius)
                if radius
                else float("nan"),
                "rational_residual_px": residual * n_px / TWO_PI,
                "commensurate": bool(
                    radius > 0.0 and residual / radius <= COMMENSURATE_REL_TOL
                ),
                "azimuth_deg": float(np.degrees(np.arctan2(q_px[1], q_px[0])) % 360.0),
                "f_amplitude": sample_magnitude(magnitude, n_px, q_px),
                "peak_amplitude": float(peak.amplitude),
                "snr": float(peak.snr),
                "quality": str(peak.quality),
                "independent": bool(peak.independent),
                "friedel_partner": record,
            }
        )
    rings = cluster_rings(rows, ring_tol)
    for row in rows:
        row["cross_talk_with_selected"] = cross_talk(
            row["q_nm_inv"],
            selected or [],
            lambda_nm,
            coincidence_rad_per_nm=float(detection.dq_nm_inv),
        )
    return {
        "detector": "bragg_peak.detect_bragg_peaks",
        "n_peaks": len(peaks),
        "n_candidates_before_cap": int(detection.n_candidates),
        "noise_sigma": float(detection.noise_sigma),
        "dq_nm_inv": float(detection.dq_nm_inv),
        "q_nyquist_nm_inv": float(detection.q_nyquist_nm_inv),
        "denominator_max": int(denominator_max),
        "ring_tol": float(ring_tol),
        "commensurate_relative_tolerance": COMMENSURATE_REL_TOL,
        "friedel_tolerance_px": float(friedel_tolerance_px),
        "lambda_nm_used_for_cross_talk": None
        if lambda_nm is None
        else float(lambda_nm),
        "cross_talk_coincidence_tolerance_rad_per_nm": float(detection.dq_nm_inv),
        "rings": [
            {key: value for key, value in ring.items() if key != "members"}
            | {
                "members": [
                    {
                        "peak_index": member["peak_index"],
                        "h": member["h"],
                        "k": member["k"],
                        "writings": member["writings"],
                        "q_nm_inv_abs": member["q_nm_inv_abs"],
                        "f_amplitude": member["f_amplitude"],
                    }
                    for member in ring["members"]
                ]
            }
            for ring in rings
        ],
        "candidates": rows,
    }


def _friedel_partner(peaks, index, q_px, n_px, magnitude, tolerance_px):
    """Nearest detected peak at ``-q`` plus the asymmetry residual."""
    target = -np.asarray(q_px, dtype=float)
    best_index = None
    best_amplitude = None
    best_distance = float("inf")
    for other_index, other in enumerate(peaks):
        if other_index == index:
            continue
        distance = float(np.hypot(*(np.asarray(other.q_px, dtype=float) - target)))
        if distance < best_distance:
            best_index, best_amplitude, best_distance = (
                int(other_index),
                float(other.amplitude),
                distance,
            )
    own = sample_magnitude(magnitude, n_px, q_px)
    partner_magnitude = sample_magnitude(magnitude, n_px, target)
    found = bool(best_index is not None and best_distance <= float(tolerance_px))
    strongest = max(own, partner_magnitude)
    return {
        "detected_peak_found": found,
        "detected_peak_index": best_index if found else None,
        "detected_peak_distance_px": None if best_index is None else best_distance,
        "detected_peak_amplitude": best_amplitude if found else None,
        "q_px_at_minus_q": [float(target[0]), float(target[1])],
        "amplitude_at_minus_q": float(partner_magnitude),
        "amplitude": float(partner_magnitude),
        "asymmetry_residual": float(abs(own - partner_magnitude) / strongest)
        if strongest > 0.0
        else float("nan"),
    }


def cross_talk(q_nm_inv, selected, lambda_nm, coincidence_rad_per_nm=None):
    """Cross-talk of one candidate against the already selected points.

    The threshold is the sibling window law ``|dq| < 2 / lambda`` (rad/nm).  A
    candidate within ``coincidence_rad_per_nm`` (the FFT bin spacing by default)
    of a selected point is that very reflection, not a second one, and is
    reported separately.
    """
    if not selected or lambda_nm is None:
        return {
            "threshold_rad_per_nm": None
            if lambda_nm is None
            else 2.0 / float(lambda_nm),
            "coincidence_tolerance_rad_per_nm": None,
            "min_separation_rad_per_nm": None,
            "min_separation_with": None,
            "violations": [],
            "same_reflection": [],
            "cross_talk": False,
            "note": "no selected point to compare with",
        }
    threshold = 2.0 / float(lambda_nm)
    # a peak of the canvas that is the selected reflection itself is not
    # cross-talk: only a *different* reflection inside the window is
    coincidence = (
        float(coincidence_rad_per_nm)
        if coincidence_rad_per_nm is not None
        else 0.25 * threshold
    )
    best = None
    best_separation = float("inf")
    violations = []
    same = []
    for label, q_other in selected:
        for sign in (+1.0, -1.0):
            separation = float(
                np.hypot(
                    *(
                        np.asarray(q_nm_inv, dtype=float)
                        - sign * np.asarray(q_other, dtype=float)
                    )
                )
            )
            record = {
                "label": str(label),
                "sign": int(sign),
                "separation_rad_per_nm": separation,
            }
            if separation < best_separation:
                best, best_separation = record, separation
            if separation <= coincidence:
                same.append(record)
            elif separation < threshold:
                violations.append(record)
    return {
        "threshold_rad_per_nm": threshold,
        "coincidence_tolerance_rad_per_nm": coincidence,
        "min_separation_rad_per_nm": best_separation,
        "min_separation_with": best,
        "violations": violations,
        "same_reflection": same,
        "cross_talk": bool(violations),
    }


# --------------------------------------------------------------------------- #
# the three applicability criteria and the route
# --------------------------------------------------------------------------- #
def slow_variation(field, valid, q_rad_px):
    """Relative change of the local field over one carrier cycle (dimensionless).

    The carrier period ``2 pi / |q|`` in pixels is rounded to an integer shift
    along the carrier direction; the reported number is
    ``median |psi(r + delta) - psi(r)| / median |psi(r)|`` over the valid
    pixels, i.e. how much the envelope has changed after one carrier cycle.
    """
    array = np.asarray(field)
    q_abs = float(np.hypot(*np.asarray(q_rad_px, dtype=float)))
    if q_abs <= 0.0:
        return float("inf")
    period_px = TWO_PI / q_abs
    step = max(1, round(period_px))
    unit = np.asarray(q_rad_px, dtype=float) / q_abs
    dx, dy = round(step * unit[0]), round(step * unit[1])
    if dx == 0 and dy == 0:
        dx = 1
    shifted = np.roll(np.roll(array, -dy, axis=0), -dx, axis=1)
    reference = float(np.median(np.abs(array)[valid]))
    if not (reference > 0.0):
        return float("inf")
    return float(np.median(np.abs(shifted - array)[valid]) / reference)


def criteria(field, valid, q_rad_px, q_nm_inv, lambda_nm, others):
    """The three applicability criteria of the analytic route, with numbers."""
    q_abs = float(np.hypot(*np.asarray(q_nm_inv, dtype=float)))
    lambda_q = float(lambda_nm) * q_abs
    two_lambda2_q2 = float(2.0 * lambda_q**2)
    suppression = float(math.exp(-two_lambda2_q2))
    threshold = 2.0 / float(lambda_nm)
    best, best_separation = None, float("inf")
    for label, q_other in others:
        for sign in (+1.0, -1.0):
            separation = float(
                np.hypot(
                    *(
                        np.asarray(q_nm_inv, dtype=float)
                        - sign * np.asarray(q_other, dtype=float)
                    )
                )
            )
            if separation < best_separation:
                best_separation = separation
                best = {"label": str(label), "sign": int(sign)}
    if others:
        separated = bool(best_separation >= threshold)
    else:
        separated, best_separation = True, None
    variation = slow_variation(field, valid, q_rad_px)
    q_rad_abs = float(np.hypot(*np.asarray(q_rad_px, dtype=float)))
    record = {
        "two_lambda2_q2": two_lambda2_q2,
        "friedel_suppression": suppression,
        "friedel_suppression_max": FRIEDEL_SUPPRESSION_MAX,
        "friedel_ok": bool(two_lambda2_q2 >= FRIEDEL_EXPONENT_MIN),
        "crosstalk_threshold_rad_per_nm": threshold,
        "crosstalk_min_separation_rad_per_nm": best_separation,
        "crosstalk_min_separation_with": best,
        "crosstalk_ok": separated,
        "carrier_period_nm": float(TWO_PI / q_abs) if q_abs > 0.0 else float("inf"),
        "carrier_period_px": float(TWO_PI / q_rad_abs)
        if q_rad_abs > 0.0
        else float("inf"),
        "slow_variation_per_cycle": variation,
        "slow_variation_max": SLOW_VARIATION_MAX,
        "slow_ok": bool(variation <= SLOW_VARIATION_MAX),
    }
    for key, value in record.items():
        if isinstance(value, float) and not math.isfinite(value):
            record[key] = None
    record["route"] = decide_route(record)
    record["route_reason"] = route_reason(record)
    return record


def number_text(value):
    """Readable number, with ``None`` for a quantity that is not defined."""
    if value is None:
        return "not defined"
    return f"{float(value):.6g}"


def decide_route(record):
    """The default analytic route unless one of the three criteria fails."""
    if record["friedel_ok"] and record["crosstalk_ok"] and record["slow_ok"]:
        return ROUTE_ANALYTIC
    return ROUTE_TWO_SIDED


def route_reason(record):
    """Which criterion sent the point to the strict two-sided reconstruction."""
    if record["route"] == ROUTE_ANALYTIC:
        return (
            "2 lambda^2 |q|^2 = "
            f"{number_text(record['two_lambda2_q2'])} >= "
            f"{FRIEDEL_EXPONENT_MIN:.6g}, the smallest |dq| to another point is "
            + (
                "not defined (single point)"
                if record["crosstalk_min_separation_rad_per_nm"] is None
                else f"{number_text(record['crosstalk_min_separation_rad_per_nm'])} "
                f"nm^-1 >= 2 / lambda = "
                f"{number_text(record['crosstalk_threshold_rad_per_nm'])} nm^-1"
            )
            + ", and the envelope changes by "
            f"{number_text(record['slow_variation_per_cycle'])} of its size per "
            f"carrier cycle <= {SLOW_VARIATION_MAX:.6g}"
        )
    failed = [
        name
        for name, key in (
            ("Friedel partner suppression 2 lambda^2 |q|^2", "friedel_ok"),
            ("cross-talk |dq| vs 2 / lambda", "crosstalk_ok"),
            ("slow envelope per carrier cycle", "slow_ok"),
        )
        if not record[key]
    ]
    return (
        "the strict two-sided reconstruction is used because "
        + "; ".join(failed)
        + " fails: 2 lambda^2 |q|^2 = "
        + number_text(record["two_lambda2_q2"])
        + f" (needs >= {FRIEDEL_EXPONENT_MIN:.6g}), smallest |dq| = "
        + number_text(record["crosstalk_min_separation_rad_per_nm"])
        + " nm^-1 (needs >= "
        + number_text(record["crosstalk_threshold_rad_per_nm"])
        + " nm^-1), envelope change per carrier cycle = "
        + number_text(record["slow_variation_per_cycle"])
        + f" (needs <= {SLOW_VARIATION_MAX:.6g})"
    )


# --------------------------------------------------------------------------- #
# the two reconstructions
# --------------------------------------------------------------------------- #
def analytic_map(field, q_rad_px):
    """``B = 2 Re[psi exp(i q.r)]``: the q-component of the map (default route)."""
    array = np.asarray(field)
    n_px = int(array.shape[0])
    rows = np.arange(n_px, dtype=float)[:, None]
    columns = np.arange(n_px, dtype=float)[None, :]
    carrier = np.exp(1j * (float(q_rad_px[0]) * columns + float(q_rad_px[1]) * rows))
    return ANALYTIC_FACTOR * np.real(array * carrier)


def two_sided_map(canvas, q_rad_px, lambda_nm, nm_per_px, nan_fill=0.0):
    """Strict band-pass on the raw spectrum: mask both ``+q`` and ``-q``.

    Returns ``(B, imaginary_rms)``; the imaginary part of the inverse transform
    is exactly zero for a Hermitian spectrum and is reported as a check.
    """
    array = np.asarray(canvas, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"the canvas must be square, got {array.shape}")
    n_px = int(array.shape[0])
    filled = np.where(np.isfinite(array), array, float(nan_fill))
    spectrum = np.fft.fft2(filled)
    frequency = TWO_PI * np.fft.fftfreq(n_px)
    width = lq.lambda_px(lambda_nm, nm_per_px)
    qx, qy = float(q_rad_px[0]), float(q_rad_px[1])
    delta = (frequency[None, :] - qx) ** 2 + (frequency[:, None] - qy) ** 2
    partner = (frequency[None, :] + qx) ** 2 + (frequency[:, None] + qy) ** 2
    window = np.exp(-0.5 * width**2 * delta) + np.exp(-0.5 * width**2 * partner)
    transformed = np.fft.ifft2(spectrum * window)
    return np.real(transformed), float(np.max(np.abs(np.imag(transformed))))


def cross_check(analytic, two_sided, valid):
    """Point-by-point comparison of the analytic and the two-sided route."""
    first = np.asarray(analytic, dtype=float)[valid]
    second = np.asarray(two_sided, dtype=float)[valid]
    if first.size < 2 or first.std() == 0.0 or second.std() == 0.0:
        return {
            "n_pixels": int(first.size),
            "correlation": float("nan"),
            "rms_ratio_two_sided_over_analytic": float("nan"),
            "max_abs_difference": float("nan"),
            "rms_analytic": float(first.std()) if first.size else float("nan"),
            "rms_two_sided": float(second.std()) if second.size else float("nan"),
        }
    return {
        "n_pixels": int(first.size),
        "correlation": float(np.corrcoef(first, second)[0, 1]),
        "rms_ratio_two_sided_over_analytic": float(second.std() / first.std()),
        "max_abs_difference": float(np.max(np.abs(first - second))),
        "rms_analytic": float(first.std()),
        "rms_two_sided": float(second.std()),
    }


# --------------------------------------------------------------------------- #
# step 5: the phase-difference coherence of one point's own field
# --------------------------------------------------------------------------- #
@dataclass
class CoherenceResult:
    """The JSON record of one point's coherence plus the arrays the figure draws."""

    record: dict
    c: np.ndarray
    profile: np.ndarray
    lag_nm: np.ndarray
    bin_pairs: np.ndarray
    max_lag_nm: float
    fit_range_nm: float
    border_px: float


def circular_autocorrelation(field):
    """Complex circular autocorrelation, fftshifted so that lag 0 is at the centre."""
    spectrum = np.fft.fft2(field)
    return np.fft.fftshift(np.fft.ifft2(spectrum * np.conj(spectrum)))


def coherence_radius_px(n_px):
    """Distance of every pixel from the canvas centre, in px (lag 0 at the centre)."""
    rows, columns = np.mgrid[0:n_px, 0:n_px]
    centre = n_px // 2
    return np.hypot(rows - centre, columns - centre)


def shell_profile(correlation, pair_count, radius_px, nm_per_px, max_lag_nm):
    """Pair-count-weighted shell mean of a COMPLEX correlation.

    One bin per pixel of lag (``edges = 0, 1, ... px`` up to ``max_lag_nm``); the
    real and the imaginary part are binned apart and recombined.  A bin without a
    single valid pair is NaN.  Returns ``(centres_nm, profile, bin_pairs)`` where
    ``bin_pairs`` is the number of valid pixel pairs collected by each bin.
    """
    edges = np.arange(0.0, max_lag_nm / nm_per_px + 1.0, 1.0)
    index = np.digitize(radius_px, edges) - 1
    keep = (
        (index >= 0) & (index < len(edges) - 1) & (pair_count >= COHERENCE_PAIR_FLOOR)
    )
    bins = index[keep]
    weight = pair_count[keep]
    n_bins = len(edges) - 1
    denominator = np.bincount(bins, weights=weight, minlength=n_bins)
    real = np.bincount(
        bins, weights=np.real(correlation)[keep] * weight, minlength=n_bins
    )
    imag = np.bincount(
        bins, weights=np.imag(correlation)[keep] * weight, minlength=n_bins
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        profile = (real + 1j * imag) / denominator
    centres = 0.5 * (edges[:-1] + edges[1:]) * nm_per_px
    return centres, profile, denominator


def mask_insets_px(valid):
    """Distance of the valid region to the canvas edge, per axis, in px.

    ``(inset_x_px, inset_y_px)``: the smallest number of columns (rows) between
    the valid region and the left/right (top/bottom) canvas edge.  A circular
    autocorrelation starts to fold the shifted copy of the valid region onto
    itself once the lag reaches twice the smaller of the two, which is why the
    ramp cut is capped there.
    """
    rows = np.flatnonzero(np.asarray(valid).any(axis=1))
    columns = np.flatnonzero(np.asarray(valid).any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return 0, 0
    n_px = int(np.asarray(valid).shape[0])
    return (
        int(min(columns[0], n_px - 1 - columns[-1])),
        int(min(rows[0], n_px - 1 - rows[-1])),
    )


def ramp_slopes(correlation, nm_per_px, *, cut_n_px, fit_max_nm):
    """``arg C(dr)`` slope along +x and +y, in rad/nm.

    The 1D cuts run ``cut_n_px`` samples from the centre along +x and +y (lags
    ``0 .. cut_n_px - 1`` px) and the fit uses the samples with
    ``lag <= fit_max_nm``.  Both are explicit arguments so that the historical
    numbers (400 samples, 50 nm fit window, written when the profile range was
    60 nm on a 2120 px canvas) can be reproduced for the reconciliation of the
    frozen design; the command line never exposes them.
    """
    centre = int(correlation.shape[0]) // 2
    length = int(min(int(cut_n_px), int(correlation.shape[0]) - centre))
    lags = np.arange(length) * float(nm_per_px)
    out = {}
    for name, cut in (
        ("x", correlation[centre, centre : centre + length]),
        ("y", correlation[centre : centre + length, centre]),
    ):
        phase = np.unwrap(np.angle(cut))
        good = np.isfinite(phase) & (lags <= float(fit_max_nm))
        out[name] = (
            float(np.polyfit(lags[good], phase[good], 1)[0])
            if int(np.count_nonzero(good)) > 5
            else float("nan")
        )
    return out


def phase_coherence(
    theta,
    mask,
    *,
    nm_per_px,
    lambda_nm,
    max_lag_nm,
    fit_range_nm,
    anchor_fractions=COHERENCE_ANCHOR_FRACTIONS,
    border_factor=COHERENCE_BORDER_FACTOR,
    ramp_cut_nm=None,
    ramp_fit_max_nm=None,
):
    """The phase-difference coherence of one point's own demodulated field.

    ``C(dr) = <exp(i[theta(r) - theta(r + dr)])>`` over the valid pixel pairs,
    estimated as the circular autocorrelation ``A`` of ``u = exp(i theta)`` divided
    by the pair count ``N`` of the mask, both masked.  The mask is the point's own
    validity mask eroded by ``0.65 lambda`` so that the periodic-FFT border band is
    excluded.  The radial profile is the pair-count-weighted complex shell mean;
    ``xi_nm`` is the exponential decay length fitted over every non-empty bin with
    ``0 < dr <= fit_range_nm`` (no sample-count filter); the residual phase ramp is
    the slope of ``arg C`` along +x and +y, reported in rad/nm, Angstrom^-1 and as
    an equivalent strain.

    The ramp cut is ``min(fit_range_nm, cap)`` with
    ``cap = min(L / 2, 2 * min(inset_x_px, inset_y_px)) * nm_per_px``: beyond
    twice the inset of the valid region a circular autocorrelation folds the far
    side of the canvas into ``arg C``, which biases the slope (the frozen design
    addendum section 1).  ``ramp_cut_nm`` overrides that cut explicitly, which the
    historical reconciliation needs and the command line never exposes.  The
    ``xi`` fit window and the ``|C|`` anchors are not touched by the cap.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 2 or theta.shape[0] != theta.shape[1]:
        raise ValueError(f"theta must be a square 2D array, got {theta.shape}")
    n_px = int(theta.shape[0])
    valid0 = np.asarray(mask, dtype=float) > 0.5
    border_px = float(border_factor) * float(lambda_nm) / float(nm_per_px)
    valid = valid0 & (distance_transform_edt(valid0) > border_px)
    indicator = valid.astype(float)
    unit = np.where(valid, np.exp(1j * theta), 0.0 + 0.0j)
    numerator = circular_autocorrelation(unit)
    pair_count = np.real(circular_autocorrelation(indicator))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(pair_count >= COHERENCE_PAIR_FLOOR, numerator / pair_count, np.nan)

    lag_nm, profile, bin_pairs = shell_profile(
        c, pair_count, coherence_radius_px(n_px), nm_per_px, float(max_lag_nm)
    )
    magnitude = np.abs(profile)
    fitted = (
        (lag_nm > 0)
        & (lag_nm <= float(fit_range_nm))
        & np.isfinite(magnitude)
        & (magnitude > 0)
    )
    if int(np.count_nonzero(fitted)) >= 5:
        slope = float(np.polyfit(lag_nm[fitted], np.log(magnitude[fitted]), 1)[0])
        xi_nm = float(-1.0 / slope) if slope < 0 else float("inf")
    else:
        xi_nm = float("nan")

    anchors = {
        f"{float(fraction):.2f}": float(
            np.interp(float(fraction) * float(max_lag_nm), lag_nm, magnitude)
        )
        for fraction in anchor_fractions
    }
    finite_bins = np.flatnonzero(np.isfinite(magnitude))
    if finite_bins.size:
        last = int(finite_bins[-1])
        last_lag_nm, last_pairs = float(lag_nm[last]), float(bin_pairs[last])
    else:
        last_lag_nm, last_pairs = float("nan"), 0.0

    inset_x_px, inset_y_px = mask_insets_px(valid)
    wrap_free_lag_nm = 2.0 * float(min(inset_x_px, inset_y_px)) * float(nm_per_px)
    cap_nm = min(0.5 * n_px * float(nm_per_px), wrap_free_lag_nm)
    if ramp_cut_nm is None:
        cut_nm = min(float(fit_range_nm), cap_nm)
    else:
        cut_nm = float(ramp_cut_nm)
    cut_truncated = bool(cut_nm < float(fit_range_nm))
    # ``cut_n_px`` samples cover the lags ``0 .. cut_n_px - 1`` px, so a cut of
    # ``n`` px uses ``n`` samples (which is how the historical script sliced its
    # 1D cut); rounding keeps the samples strictly inside the wrap free lag.
    cut_n_px = max(2, round(cut_nm / float(nm_per_px)))
    fit_max_nm = (
        float(fit_range_nm) if ramp_fit_max_nm is None else float(ramp_fit_max_nm)
    )
    slopes = ramp_slopes(c, nm_per_px, cut_n_px=cut_n_px, fit_max_nm=fit_max_nm)
    ramp = {
        name: {
            "slope_rad_per_nm": float(slope),
            "slope_per_angstrom": float(slope) / 10.0,
            "strain_equivalent": abs(float(slope)) / STRAIN_DIVISOR_PER_NM,
            "cut_nm": float(cut_nm),
            "cut_truncated": cut_truncated,
        }
        for name, slope in slopes.items()
    }
    record = {
        "n_valid_px": int(np.count_nonzero(valid)),
        "valid_fraction": float(np.mean(valid)) if n_px else float("nan"),
        "border_erosion_px": border_px,
        "inset_x_px": int(inset_x_px),
        "inset_y_px": int(inset_y_px),
        "wrap_free_lag_nm": wrap_free_lag_nm,
        "ramp_cap_nm": float(cap_nm),
        "anchors": anchors,
        "xi_nm": xi_nm,
        "lag_profile": {
            "lag_nm": [float(value) for value in lag_nm],
            "C_real": [float(value) for value in np.real(profile)],
            "C_imag": [float(value) for value in np.imag(profile)],
        },
        "profile_last_lag_nm": last_lag_nm,
        "profile_last_pairs": last_pairs,
        "ramp": ramp,
        "ramp_cut_nm": float(cut_nm),
        "ramp_cut_px": int(cut_n_px),
        "ramp_cut_truncated": cut_truncated,
        "ramp_fit_max_nm": fit_max_nm,
    }
    return CoherenceResult(
        record=record,
        c=c,
        profile=profile,
        lag_nm=lag_nm,
        bin_pairs=bin_pairs,
        max_lag_nm=float(max_lag_nm),
        fit_range_nm=float(fit_range_nm),
        border_px=border_px,
    )


def coherence_config(field_of_view_nm, lambda_nm, nm_per_px):
    """The top level ``coherence`` configuration block of the report."""
    max_lag_nm = float(field_of_view_nm) / math.sqrt(2.0)
    return {
        "definition": COHERENCE_DEFINITION,
        "criterion": COHERENCE_CRITERION,
        "max_lag_nm": max_lag_nm,
        "fit_range_nm": (2.0 / 3.0) * max_lag_nm,
        "border_erosion_px": COHERENCE_BORDER_FACTOR
        * float(lambda_nm)
        / float(nm_per_px),
        "anchor_fractions": list(COHERENCE_ANCHOR_FRACTIONS),
        "max_lag_rule": COHERENCE_MAX_LAG_RULE,
        "fit_rule": COHERENCE_FIT_RULE,
        "ramp_cut_rule": COHERENCE_CAP_RULE,
        "pair_floor": COHERENCE_PAIR_FLOOR,
        "a_nm": STRAIN_A0_NM,
        "beta": STRAIN_BETA,
        "A_per_strain_angstrom_inv": STRAIN_A_PER_ANGSTROM_INV,
        "strain_divisor_per_nm": STRAIN_DIVISOR_PER_NM,
        "strain_conversion": STRAIN_CONVERSION,
        "strain_conversion_note": STRAIN_CONVERSION_NOTE,
        "no_sample_count_filter": True,
    }


# --------------------------------------------------------------------------- #
# products (the HDF5 convention of the sibling skill)
# --------------------------------------------------------------------------- #
def write_point_products(path, *, field, amplitude, theta, mask):
    """The four datasets of one point: field / amplitude / theta / mask."""
    return lq.save_products(
        path,
        field=np.asarray(field, dtype=np.complex128),
        amplitude=np.asarray(amplitude, dtype=np.float64),
        theta=np.asarray(theta, dtype=np.float64),
        mask=np.asarray(mask, dtype=np.float64),
        generator=GENERATOR,
    )


def write_pair_products(path, *, B, mask, attributes):
    """The B(r) map of one pair plus the provenance attributes of the pair."""
    target = Path(path)
    with h5py.File(target, "w") as handle:
        h5io.write_file_metadata(handle, generator=GENERATOR, extra=attributes)
        h5io.create_dataset(handle, "B", np.asarray(B, dtype=np.float64))
        h5io.create_dataset(handle, "mask", np.asarray(mask, dtype=np.float64))
    return target


def statistical_summary(field, amplitude, valid):
    """The amplitude/theta summary of one point (the sibling conventions)."""
    amplitude = np.asarray(amplitude, dtype=float)
    field = np.asarray(field)
    valid = np.asarray(valid, dtype=bool)
    if not np.any(valid):
        fail(
            "the amplitude mask of this point has no valid pixel: the amplitude "
            "is identically zero on the canvas",
            EXIT_DEGENERATE,
        )
    positive = amplitude[valid]
    median, fwhm = lq.linear_median_fwhm(positive)
    mean_rad, resultant, _ = lq.circular_mean_rad(field, valid)
    median_rad, span_deg, minimisers = lq.circular_median_rad(
        np.angle(field)[valid], weight=positive
    )
    return {
        "amplitude": {
            "median": float(median),
            "fwhm": float(fwhm),
            "max": float(np.max(amplitude)),
            "min_valid": float(np.min(positive)) if positive.size else float("nan"),
        },
        "theta": {
            "circular_mean_deg": float(np.degrees(mean_rad) % 360.0),
            "circular_median_deg": float(np.degrees(median_rad) % 360.0),
            "circular_median_span_deg": float(span_deg),
            "n_median_minimisers": int(minimisers),
            "resultant_R": float(resultant),
        },
        "mask_coverage_fraction": float(np.mean(valid)),
    }


def dump_json(path, payload):
    """Write a report as UTF-8 JSON with numpy scalars converted."""
    Path(path).write_text(
        json.dumps(payload, indent=2, default=float, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return Path(path)
