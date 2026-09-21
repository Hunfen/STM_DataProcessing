"""Engine of the local-q-map skill: Gaussian-windowed local Fourier analysis.

One definition, used by every script of this skill.  With a square canvas of
``N x N`` pixels, array coordinates ``(row = y, col = x)`` and a field of view
``L`` (nm, hence ``nm_per_px = L / N``), a corrected topography ``T(r)`` is
demodulated at the wave vector ``q`` (rad/px) into the complex local field

    psi_q(r) = FFT^-1{ FFT[ T(r) * exp(-i q.r) ] * exp(-Lambda^2 |k|^2 / 2) }

``Lambda`` is ``--lambda-nm`` (default 3.0 nm, i.e. the paper's 30 Angstrom
window; it is NOT the ``lambda_nm`` of the lawler-fujita-correction skill, which
defaults to 30 nm and plays a different role there).  ``|k|`` is the angular
frequency of the FFT grid in rad/px, so the Gaussian is evaluated with
``Lambda_px = Lambda_nm / nm_per_px``.

For ``T`` containing ``A cos(q.r + phi_q)`` the demodulated carrier sits at
``k = 0`` and the window keeps it with weight exactly one, so

    psi_q(r) ~ (A_q / 2) exp(+i phi_q(r))       amplitude ~ A_q / 2
                                                theta_q(r) = arg psi_q ~ +phi_q

i.e. the map carries no ``q.r`` ramp (the demodulated convention) and
``theta_q`` is wrapped into ``(-pi, pi]``.  Three identities hold to machine
precision and are used by the self-test:

    sum_r psi_q(r)              = sum_r T(r) exp(-i q.r)      (window independent)
    psi_{-q}(r)                 = conj(psi_q(r))              (T real, Friedel)
    psi_q(r - delta)            = exp(-i q.delta) psi_q(r)    (image translation)

Wave vectors are addressed by the 1x1-ring hexagonal reciprocal basis
``{b1, b2}`` (60 degrees apart, the ``bragg_peak`` ``_HEXAGONAL_UNIT``
convention) with arbitrary real ``h, k``:

    q(h, k) = h * b1 + k * b2
    ring_1x1 members  +/- (1, 0), (0, 1), (1, -1)
    ring_r3  members  +/- (1/3, 1/3), (2/3, -1/3), (1/3, -2/3)

A correction report fixes only the RADIUS of that basis reliably: its
``orientation_deg`` (affine) and its lock-in directions (lawler-fujita) live in
the INPUT frame of the correction, while this skill analyses the corrected
canvas.  The corrected canvas therefore decides the orientation
(``canvas_ring_orientation``, one ``bragg_peak.detect_bragg_peaks`` run over the
ring_1x1 neighbourhood), and the resolved basis is validated against the canvas
unconditionally before any artifact is written (``basis_canvas_check``).

Everything here is geometry and signal processing: no physical statement is made
anywhere, and no ring is ever named after a k-space high-symmetry point.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

SKILL_VERSION = "1.0"
DEFAULT_LAMBDA_NM = 3.0
DEFAULT_AMPLITUDE_FRACTION = 0.10
# Periodic-FFT pollution depth of the window, in units of Lambda (measured on the
# sibling lawler-fujita-correction skill); the border band is reported, not cut.
BOUNDARY_MARGIN_FACTOR = 0.65
# Relative tolerances of the CLI guards: the canvas nm/px (L / N) against the
# nm/px of the basis source, and the ring identity of a report's radius keys.
NM_PER_PX_REL_TOLERANCE = 1e-3
RING_MATCH_REL_TOLERANCE = 0.02
# Basis against the canvas (review F6).  A correction report's orientation_deg
# describes the INPUT canvas, so the corrected canvas itself is the authority on
# the ring_1x1 directions of the product.  The orientation is measured on the
# canvas with the bragg_peak detector and replaces the report's value when the two
# disagree by more than BASIS_RING_ORIENTATION_TOLERANCE_DEG (below that the
# report's own value is kept bit for bit); the resolved basis is then validated
# against the canvas unconditionally:
#   * the members must reach BASIS_RING_STRENGTH_FRACTION of the strongest
#     spectrum magnitude of the annulus radius_px * (1 +/- BASIS_RING_BAND_FRACTION)
#     (the |psi| ratio: |psi_q| is |ifft2(window * FFT)|, so the 1 / N^2 factor
#     cancels);
#   * their arc distance to the measured ring direction must stay below
#     BASIS_RING_OFFSET_FACTOR window radii, i.e. below the offset at which the
#     demodulation window attenuates a member to that same fraction
#     (offset_tolerance_px = BASIS_RING_OFFSET_FACTOR * n / (2 pi Lambda_px) with
#     BASIS_RING_OFFSET_FACTOR = sqrt(2 ln(1 / 0.2))).
BASIS_RING_BAND_FRACTION = 0.10
BASIS_RING_STRENGTH_FRACTION = 0.20
BASIS_RING_OFFSET_FACTOR = 1.79
BASIS_RING_MEMBER_HALF_PX = 2.0
BASIS_RING_ORIENTATION_BAND = 0.15
BASIS_RING_ORIENTATION_TOLERANCE_DEG = 1.0
BASIS_RING_DETECTION_PATCH_HALF = 3
BASIS_RING_DETECTION_MAX_CANDIDATES = 512
TWO_PI = 2.0 * np.pi
BAD_COLOR = "#b0b0b0"

# Members of the two rings in the {b1, b2} basis (the negatives complete them).
RING_1X1_MEMBERS = ((1.0, 0.0), (0.0, 1.0), (1.0, -1.0))
RING_R3_MEMBERS = ((1.0 / 3.0, 1.0 / 3.0), (2.0 / 3.0, -1.0 / 3.0),
                   (1.0 / 3.0, -2.0 / 3.0))

FOV_FROM_LOG_HELP = (
    "read the field of view from a correction log; the "
    "'# corrected canvas: ... field of view <value> nm' line wins, otherwise the "
    "last 'field of view <value> nm' line of the log is used")


# --------------------------------------------------------------------------- #
# style and artifact writing (mirrors lf_lib.save_map / lf_lib.save_npy)
# --------------------------------------------------------------------------- #
def setup_style():
    """Plot style of the skill (no usetex: broken with mpl 3.10 + TeX Live 2026)."""
    import matplotlib

    matplotlib.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_map(path, data, cmap="inferno", vmin=None, vmax=None, dpi=150, size=4.0):
    """Write one map as a bare PNG preview (no axes, no frame, origin lower)."""
    import matplotlib.pyplot as plt

    cmap_object = plt.get_cmap(cmap).copy()
    cmap_object.set_bad(color=BAD_COLOR)
    finite = np.asarray(data, dtype=float)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(finite, cmap=cmap_object, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_npy(path, array):
    np.save(path, np.asarray(array))


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #
def load_topo(path, delimiter=","):
    """Square topography matrix plus the NaN fraction; first row = scan start."""
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"{source}: no such file")
    array = np.loadtxt(source, delimiter=delimiter)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise SystemExit(f"{source}: expected a square 2D matrix, got {array.shape}")
    array = np.asarray(array, dtype=float)
    return array, float(np.mean(~np.isfinite(array)))


def parse_q_spec(text):
    """``'H,K'`` (floats or fractions such as ``'2/3,1/3'``) -> ``(h, k)``."""
    parts = [item.strip() for item in str(text).split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--q {text!r}: expected 'H,K', got {len(parts)} field(s)")

    def value(token):
        if "/" in token:
            numerator, _, denominator = token.partition("/")
            try:
                return float(numerator) / float(denominator)
            except (ValueError, ZeroDivisionError):
                raise SystemExit(
                    f"--q {text!r}: cannot read the fraction {token!r}") from None
        try:
            return float(token)
        except ValueError:
            raise SystemExit(
                f"--q {text!r}: cannot read the number {token!r}") from None

    return value(parts[0]), value(parts[1])


def format_hk(h, k):
    """Compact English label of an (h, k) pair for the log."""
    def one(token):
        if abs(token - round(token)) < 1e-9:
            return f"{round(token)}"
        for denominator in (3, 2):
            scaled = token * denominator
            if abs(scaled - round(scaled)) < 1e-9:
                return f"{round(scaled)}/{denominator}"
        return f"{token:g}"

    return f"({one(h)}, {one(k)})"


# --------------------------------------------------------------------------- #
# field of view from a correction log (precedence of the phase-analysis skill)
# --------------------------------------------------------------------------- #
def parse_fov_log(text):
    """All 'field of view <value> nm' occurrences of a correction log, in order."""
    pattern = re.compile(r"field of view\s+([\d.]+)\s+nm")
    matches = []
    for line in text.splitlines():
        hit = pattern.search(line)
        if not hit:
            continue
        matches.append({"value": float(hit.group(1)),
                        "corrected": "corrected canvas" in line,
                        "label": ("corrected canvas line" if "corrected canvas" in line
                                  else "input canvas line"),
                        "line": line.strip()})
    return matches


def field_of_view_from_log(path):
    """The corrected-canvas field of view of a correction log, else its last line."""
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"--size-nm-from-log {source}: no such file")
    try:
        text = source.read_text()
    except OSError as exc:
        raise SystemExit(
            f"--size-nm-from-log {source}: cannot read it ({exc})") from exc
    matches = parse_fov_log(text)
    if not matches:
        raise SystemExit(f"no 'field of view <value> nm' line in {source}")
    chosen = next((item for item in matches if item["corrected"]), matches[-1])
    return chosen["value"], f"log: {chosen['label']} of {source}"


# --------------------------------------------------------------------------- #
# reciprocal basis sources
# --------------------------------------------------------------------------- #
def _pair(values):
    array = np.asarray(values, dtype=float).ravel()
    return array if array.size == 2 and bool(np.all(np.isfinite(array))) else None


def hexagonal_unit():
    """(2, 2) hexagonal basis in units of |b1|, b2 at +60 degrees.

    The package's ``_HEXAGONAL_UNIT`` is used when importable so the convention
    cannot drift; the identical local copy keeps the skill runnable without it.
    """
    try:
        from stm_data_processing.utils.bragg_peak.lattice_fit import _HEXAGONAL_UNIT

        return np.asarray(_HEXAGONAL_UNIT, dtype=float)
    except Exception:  # any import problem falls back to the copy
        return np.array([[1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


def rotate_basis(basis, orientation_deg):
    """Rotate the rows of ``basis`` counter-clockwise by ``orientation_deg``.

    Mirrors ``stm_data_processing.utils.bragg_peak.lattice_fit.rotate_basis``
    (``rows @ rot.T``); the package version is used when importable.
    """
    array = np.asarray(basis, dtype=float)
    if orientation_deg is None:
        return array
    try:
        from stm_data_processing.utils.bragg_peak.lattice_fit import (
            rotate_basis as package_rotate,
        )

        return np.asarray(package_rotate(array, float(orientation_deg)), dtype=float)
    except Exception:  # any import problem falls back to the copy
        theta = np.radians(float(orientation_deg))
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        return array @ rot.T


def hexagon_from_radius(radius_nm_inv, orientation_deg):
    """``{b1, b2}`` (nm^-1) with ``|b1| = radius`` and b2 at +60 degrees from b1."""
    return rotate_basis(float(radius_nm_inv) * hexagonal_unit(), orientation_deg)


def _normalise_anchor_ring(value):
    """Ring named by an affine report's ``anchor_ring`` key.

    Returns ``'1x1'``, ``'r3'``, or ``None`` when the report names no ring this
    skill can act on (an unlabelled radius is read as the ring_1x1 radius).
    """
    if value is None:
        return None
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if token in ("1x1", "ring_1x1", "1x1_ring", "one_by_one"):
        return "1x1"
    if token in ("r3", "ring_r3", "r3_ring", "sqrt3", "sqrt_3", "root3"):
        return "r3"
    return None


def ring_radii(payload):
    """``radius_nm_inv`` of every ``rings_after`` entry of a report, in order."""
    entries = payload.get("rings_after")
    if not isinstance(entries, list):
        return []
    radii = []
    for item in entries:
        value = item.get("radius_nm_inv") if isinstance(item, dict) else None
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            radii.append(value)
    return radii


def _close(first, second, tolerance=RING_MATCH_REL_TOLERANCE):
    """Whether two positive radii agree to a relative ``tolerance``."""
    scale = max(abs(float(first)), abs(float(second)))
    return bool(scale > 0.0 and abs(float(first) - float(second)) / scale <= tolerance)


def anchored_ring(payload, radius, labelled):
    """Ring the affine radius keys refer to, plus a one-line English reason.

    The report's own ``anchor_ring`` wins.  Without it the ``rings_after`` list
    decides: a radius sitting on the inner member of a ratio-1/sqrt(3) pair is the
    anchored ring_r3, not the ring_1x1 ring the basis needs.
    """
    if labelled is not None:
        return labelled, f"report anchor_ring = {labelled}"
    radii = ring_radii(payload)
    if radius is None or not radii:
        return None, "the report names none (no anchor_ring, no usable rings_after)"
    for small in radii:
        if not _close(small, radius):
            continue
        for big in radii:
            if big > small and _close(big / small, np.sqrt(3.0)):
                return "r3", ("rings_after shows the ratio-1/sqrt(3) pair "
                              f"{small:.6f} and {big:.6f} nm^-1 and the radius keys "
                              "match the inner one, so they refer to ring_r3")
    return "1x1", ("rings_after shows no ratio-1/sqrt(3) pair containing the radius "
                   "keys, so they refer to ring_1x1")


def basis_ring_warning(payload, radius, anchored, anchored_radius):
    """WARNING text when a resolved radius still matches the anchored ring_r3.

    ``anchored_radius`` is the radius read from the report keys, i.e. the radius of
    the ring the report was anchored on; the ring_1x1 radius is sqrt(3) times it.
    The report's own ``rings_after`` list confirms both radii when it is present.
    """
    if anchored != "r3" or not anchored_radius:
        return None
    inner = float(anchored_radius)
    radii = ring_radii(payload)
    if radii:
        nearest = min(radii, key=lambda value: abs(np.log(value / inner)))
        if not _close(nearest, inner):
            return None
        inner = nearest
    outer = float(np.sqrt(3.0) * inner)
    if _close(radius, inner) and not _close(radius, outer):
        return (f"the resolved |b1| = {float(radius):.6f} nm^-1 matches the report's "
                f"anchored ring_r3 at {inner:.6f} nm^-1, which sits at 1/sqrt(3) of "
                f"the ring_1x1 ring at {outer:.6f} nm^-1: the chosen basis is the r3 "
                "ring, not the ring_1x1 ring")
    return None


def basis_from_report(path):
    """Resolve the 1x1 reciprocal basis from a correction report (key-based).

    Two report shapes are recognised:

    * lawler-fujita-correction: ``q_a_nm_inv`` and ``q_b_nm_inv`` are a 120 degree
      pair of the ring_1x1 hexagon, so ``b1 = q_a`` and ``b2 = q_a + q_b = -q_c``;
    * affine-correction: ``orientation_deg`` plus ``b1_measured_nm_inv_after``
      (fallback ``b1_ideal_nm_inv``, fallback ``4 pi / (sqrt(3) a_1x1)`` from
      ``implied_lattice_after``), so ``b1 = R (cos t, sin t)`` and ``b2`` is b1
      rotated by +60 degrees.  The radius keys describe the ring the report was
      anchored on (``anchor_ring``): when that is ``r3``, the ring_1x1 radius is
      recovered (``implied_lattice_after.a_1x1_nm``, else ``sqrt(3) * radius``).

    Anything else is an error: the skill never guesses a basis from a lattice
    constant alone.
    """
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"--basis-from {source}: no such file")
    try:
        payload = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"--basis-from {source}: cannot read the JSON report ({exc})") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"--basis-from {source}: expected a JSON object")

    lockin = payload.get("lockin")
    lockin_lambda = (float(lockin["lambda_nm"])
                     if isinstance(lockin, dict) and lockin.get("lambda_nm") is not None
                     else None)
    common = {
        "path": str(source),
        "corrected_nm_per_px": (float(payload["corrected_nm_per_px"])
                                if payload.get("corrected_nm_per_px") else None),
        "corrected_field_of_view_nm": (float(payload["corrected_field_of_view_nm"])
                                       if payload.get("corrected_field_of_view_nm")
                                       else None),
        "report_lockin_lambda_nm": lockin_lambda,
        "notes": [],
        "warnings": [],
    }

    q_a = _pair(payload.get("q_a_nm_inv"))
    q_b = _pair(payload.get("q_b_nm_inv"))
    if q_a is not None and q_b is not None:
        b1 = q_a
        b2 = q_a + q_b
        notes = ["lawler-fujita style report: b1 = q_a, b2 = q_a + q_b = -q_c"]
        angle = float(np.degrees(np.arccos(np.clip(
            float(q_a @ q_b) / (float(np.hypot(*q_a)) * float(np.hypot(*q_b))),
            -1.0, 1.0))))
        notes.append(f"measured angle between q_a and q_b: {angle:.3f} deg "
                     f"(120 deg expected for the ring_1x1 pair)")
        q_c = _pair(payload.get("q_c_nm_inv"))
        if q_c is not None:
            notes.append(f"|q_c - (-(q_a + q_b))| = "
                         f"{float(np.hypot(*(q_c + q_a + q_b))):.6g} nm^-1")
        detected = ["q_a_nm_inv", "q_b_nm_inv"] + (["q_c_nm_inv"] if q_c is not None
                                                   else [])
        return dict(common, kind="correction_report_lawler_fujita",
                    b1_nm_inv=[float(value) for value in b1],
                    b2_nm_inv=[float(value) for value in b2],
                    detected_keys=detected, notes=notes)

    orientation = payload.get("orientation_deg")
    measured = payload.get("b1_measured_nm_inv_after")
    ideal = payload.get("b1_ideal_nm_inv")
    implied = payload.get("implied_lattice_after")
    a_1x1 = (implied.get("a_1x1_nm") if isinstance(implied, dict) else None)
    if orientation is not None and (measured or ideal or a_1x1):
        notes = []
        warnings = []
        labelled = _normalise_anchor_ring(payload.get("anchor_ring"))
        if measured:
            radius_key = "b1_measured_nm_inv_after"
            radius_raw = float(measured)
        elif ideal:
            radius_key = "b1_ideal_nm_inv"
            radius_raw = float(ideal)
        else:
            radius_key = None
            radius_raw = None
        anchored, reason = anchored_ring(payload, radius_raw, labelled)
        notes.append(f"anchored ring of the radius keys: {reason}")
        if a_1x1 is not None and anchored == "r3":
            radius = 4.0 * np.pi / (np.sqrt(3.0) * float(a_1x1))
            notes.append("affine report anchored on ring_r3 -> 1x1 basis radius from "
                         "implied_lattice_after.a_1x1_nm = 4 pi / (sqrt(3) a_1x1) = "
                         f"{radius:.6f} nm^-1 (anchor_ring = r3, the a_1x1 key is the "
                         "ring_1x1 lattice constant by definition)")
        elif radius_raw is None:
            radius = 4.0 * np.pi / (np.sqrt(3.0) * float(a_1x1))
            notes.append("radius from implied_lattice_after.a_1x1_nm = "
                         f"{float(a_1x1):.6f} nm -> 4 pi / (sqrt(3) a) = "
                         f"{radius:.6f} nm^-1")
        elif anchored == "r3":
            radius = float(np.sqrt(3.0) * radius_raw)
            notes.append("affine report anchored on ring_r3 -> 1x1 basis radius = "
                         f"sqrt(3) * {radius_key} = {radius:.6f} nm^-1")
        else:
            radius = radius_raw
            notes.append(f"radius from {radius_key} = {radius:.6f} nm^-1")
        basis = hexagon_from_radius(radius, float(orientation))
        notes.append("affine style report: b1 = R (cos t, sin t) with "
                     f"orientation_deg = {float(orientation):.6f} deg, b2 = b1 "
                     "rotated by +60 deg (bragg_peak rotate_basis)")
        cross_check = basis_ring_warning(payload, radius, anchored, radius_raw)
        if cross_check:
            warnings.append(cross_check)
        detected = [name for name, value in (("orientation_deg", orientation),
                                             ("b1_measured_nm_inv_after", measured),
                                             ("b1_ideal_nm_inv", ideal),
                                             ("implied_lattice_after", implied),
                                             ("anchor_ring",
                                              payload.get("anchor_ring")))
                    if value is not None]
        if ring_radii(payload):
            detected.append("rings_after")
        return dict(common, kind="correction_report_affine",
                    b1_nm_inv=[float(value) for value in basis[0]],
                    b2_nm_inv=[float(value) for value in basis[1]],
                    detected_keys=detected, notes=notes, warnings=warnings)

    raise SystemExit(
        f"--basis-from {source}: no recognised reciprocal basis in the report. "
        "Expected either the lawler-fujita keys 'q_a_nm_inv' and 'q_b_nm_inv' (a "
        "120 deg ring_1x1 pair), or the affine keys 'orientation_deg' plus "
        "'b1_measured_nm_inv_after' (fallback 'b1_ideal_nm_inv', fallback "
        "'implied_lattice_after.a_1x1_nm'). Use --basis-px for an explicit basis.")


def basis_from_px(text):
    """``'b1x,b1y;b2x,b2y'`` signed fftshift offsets in px -> resolved basis dict."""
    blocks = [block.strip() for block in str(text).split(";") if block.strip()]
    if len(blocks) != 2:
        raise SystemExit(f"--basis-px {text!r}: expected 'b1x,b1y;b2x,b2y'")
    vectors = []
    for block in blocks:
        try:
            values = [float(item) for item in block.split(",")]
        except ValueError:
            raise SystemExit(f"--basis-px {text!r}: cannot read '{block}'") from None
        if len(values) != 2:
            raise SystemExit(f"--basis-px {text!r}: '{block}' is not a pair")
        vectors.append(values)
    return {"kind": "command_line_px", "path": None, "b1_px": vectors[0],
            "b2_px": vectors[1], "detected_keys": ["--basis-px"],
            "report_lockin_lambda_nm": None, "corrected_nm_per_px": None,
            "corrected_field_of_view_nm": None,
            "warnings": [],
            "notes": ["command line escape hatch: b1, b2 are signed fftshift "
                      "offsets in px on the corrected canvas"]}


def resolve_basis(basis, n_px, nm_per_px):
    """Resolve a basis dict into nm^-1, rad/px and px representations."""
    if basis.get("b1_px") is not None:
        b1_px = np.asarray(basis["b1_px"], dtype=float)
        b2_px = np.asarray(basis["b2_px"], dtype=float)
        b1_rad_px = px_offsets_to_rad_px(b1_px, n_px)
        b2_rad_px = px_offsets_to_rad_px(b2_px, n_px)
        scale = nm_per_px
        b1_nm_inv = b1_rad_px / scale
        b2_nm_inv = b2_rad_px / scale
        provenance = "command line --basis-px (fftshift px offsets)"
    else:
        b1_nm_inv = np.asarray(basis["b1_nm_inv"], dtype=float)
        b2_nm_inv = np.asarray(basis["b2_nm_inv"], dtype=float)
        # the report's own nm/px is preferred for the rad/nm -> rad/px conversion
        report_scale = basis.get("corrected_nm_per_px")
        scale = float(report_scale) if report_scale else float(nm_per_px)
        b1_rad_px = nm_inv_to_rad_px(b1_nm_inv, scale)
        b2_rad_px = nm_inv_to_rad_px(b2_nm_inv, scale)
        provenance = ("correction report corrected_nm_per_px"
                      if report_scale else "canvas nm/px = L / N")
    resolved = dict(basis)
    resolved.update(
        b1_nm_inv=[float(value) for value in b1_nm_inv],
        b2_nm_inv=[float(value) for value in b2_nm_inv],
        b1_rad_px=[float(value) for value in b1_rad_px],
        b2_rad_px=[float(value) for value in b2_rad_px],
        b1_px=[float(value) for value in rad_px_to_px_offsets(b1_rad_px, n_px)],
        b2_px=[float(value) for value in rad_px_to_px_offsets(b2_rad_px, n_px)],
        basis_nm_per_px=float(scale),
        basis_conversion=provenance,
        b1_angle_deg=float(np.degrees(np.arctan2(b1_nm_inv[1], b1_nm_inv[0]))),
        b2_angle_deg=float(np.degrees(np.arctan2(b2_nm_inv[1], b2_nm_inv[0]))),
        abs_b1_nm_inv=float(np.hypot(*b1_nm_inv)),
    )
    resolved["angle_between_deg"] = float(np.degrees(np.arccos(np.clip(
        float(b1_nm_inv @ b2_nm_inv)
        / (float(np.hypot(*b1_nm_inv)) * float(np.hypot(*b2_nm_inv))), -1.0, 1.0))))
    return resolved


# --------------------------------------------------------------------------- #
# basis against the canvas: which frame a report describes, and the validation
# --------------------------------------------------------------------------- #
def fft_offsets_px(n_px):
    """Signed fftshift offsets (px) of an FFT grid, in fftshift order."""
    n = int(n_px)
    return np.fft.fftshift(np.fft.fftfreq(n)) * float(n)


def canvas_spectrum_magnitude(image, nan_fill=0.0):
    """``|FFT2(T - mean T)|`` of the NaN-filled canvas, on the shifted grid.

    This is the canvas' own answer to "where is the ring": ``psi_q`` is
    ``ifft2(FFT[T exp(-i q.r)] * window)``, so the local peak of ``|psi_q|`` at a
    ring member equals this magnitude at the member divided by ``N^2``, a constant
    factor that cancels in every ratio taken here.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"the canvas must be square, got {array.shape}")
    filled = np.where(np.isfinite(array), array, float(nan_fill))
    return np.abs(np.fft.fftshift(np.fft.fft2(filled - float(np.mean(filled)))))


def ring_1x1_members_rad_px(b1_rad_px, b2_rad_px):
    """The six signed ring_1x1 members (1,0), (0,1), (1,-1) and their negatives."""
    base = [q_vector(h, k, b1_rad_px, b2_rad_px) for h, k in RING_1X1_MEMBERS]
    return [sign * member for sign in (1.0, -1.0) for member in base]


def fold_deg(angle_deg, period_deg=60.0):
    """Wrap an angle into ``[0, period)``."""
    return float(np.mod(float(angle_deg), float(period_deg)))


def fold_residual_deg(angle_deg, reference_deg, period_deg=60.0):
    """``angle - reference`` wrapped into ``(-period/2, period/2]``."""
    half = 0.5 * float(period_deg)
    return float((float(angle_deg) - float(reference_deg) + half)
                 % float(period_deg) - half)


def canvas_ring_orientation(image, size_nm, radius_px, reference_deg,
                            patch_half=BASIS_RING_DETECTION_PATCH_HALF,
                            band_fraction=BASIS_RING_ORIENTATION_BAND,
                            max_candidates=BASIS_RING_DETECTION_MAX_CANDIDATES):
    """Orientation of the canvas' own ring_1x1, measured on the corrected canvas.

    The detector is ``bragg_peak.detect_bragg_peaks`` (the call of the sibling
    skills), restricted to the ring_1x1 neighbourhood (``q_max_px = 1.5 *
    radius_px``) and to ``max_candidates`` localised candidates, because only the
    member azimuths within ``band_fraction`` of the resolved radius are used.  The
    azimuths are folded onto the 60 degree period of the hexagonal ring and
    combined with an amplitude-weighted circular mean; the result is expressed as a
    rotation relative to ``reference_deg`` (the orientation the report asks for),
    which keeps the (h, k) label gauge of the report.

    Returns a dict with ``status`` ('measured', 'no_members' or 'no_detector'),
    ``orientation_deg``, ``delta_deg``, ``n_members``, ``spread_deg``,
    ``member_angles_deg``, ``radius_px``, ``detector`` and an English ``message``.
    """
    reference = float(reference_deg)
    record = {"status": "no_members", "orientation_deg": reference, "delta_deg": 0.0,
              "n_members": 0, "spread_deg": float("nan"), "member_angles_deg": [],
              "members": [], "radius_px": float(radius_px),
              "band_fraction": float(band_fraction), "patch_half": int(patch_half),
              "max_candidates": int(max_candidates),
              "detector": "bragg_peak.detect_bragg_peaks", "message": ""}
    try:
        from stm_data_processing.utils.bragg_peak import detect_bragg_peaks
    except Exception as exc:  # an import problem is reported, not raised
        record.update(status="no_detector",
                      message=("the bragg_peak detector is not importable "
                               f"({exc}); the orientation of the report is kept"))
        return record
    detection = detect_bragg_peaks(np.asarray(image, dtype=float), float(size_nm),
                                   q_max_px=1.5 * float(radius_px),
                                   max_candidates=int(max_candidates),
                                   patch_half=int(patch_half), subtract_plane=False,
                                   return_fft2=False)
    for peak in detection.peaks:
        q_px = np.asarray(peak.q_px, dtype=float)
        member_radius = float(np.hypot(*q_px))
        if member_radius <= 0.0:
            continue
        if abs(member_radius - float(radius_px)) > float(band_fraction) * float(radius_px):
            continue
        angle = float(np.degrees(np.arctan2(q_px[1], q_px[0])))
        record["members"].append({"q_px": [float(q_px[0]), float(q_px[1])],
                                  "radius_px": member_radius, "angle_deg": angle,
                                  "folded_deg": fold_deg(angle),
                                  "residual_deg": fold_residual_deg(angle, reference),
                                  "amplitude": float(peak.amplitude)})
    if not record["members"]:
        record["message"] = (f"no ring_1x1 member within "
                             f"{100.0 * float(band_fraction):.0f} % of "
                             f"{float(radius_px):.3f} px was detected on the canvas")
        return record
    weight = np.asarray([member["amplitude"] for member in record["members"]],
                        dtype=float)
    residual = np.radians([member["residual_deg"] for member in record["members"]])
    z = complex(np.sum(weight * np.exp(6j * residual)))  # 6 = 2 pi / (60 deg in rad)
    if abs(z) == 0.0:
        record["message"] = "the detected ring_1x1 members cancel in the circular mean"
        return record
    delta = float(np.degrees(np.angle(z)) / 6.0)
    spread = max(abs(fold_residual_deg(member["residual_deg"], delta))
                 for member in record["members"])
    record.update(status="measured", orientation_deg=reference + delta,
                  delta_deg=delta, n_members=len(record["members"]),
                  spread_deg=float(spread),
                  member_angles_deg=[member["angle_deg"] for member in record["members"]],
                  message=(f"{len(record['members'])} ring_1x1 member(s) within "
                           f"{100.0 * float(band_fraction):.0f} % of "
                           f"{float(radius_px):.3f} px; the amplitude-weighted "
                           f"azimuth difference to the report orientation is "
                           f"{delta:+.4f} deg (spread {spread:.4f} deg)"))
    return record


def rebase_orientation(basis, delta_deg, note):
    """Rigidly rotate the nm^-1 vectors of a basis by ``delta_deg`` (counter-clockwise).

    Radius, 60 degree relation and every other number of the report are kept; only
    the frame of the two vectors changes.  ``note`` is appended to the basis notes
    (and therefore to ``basis_source.notes`` and to the log).
    """
    vectors = rotate_basis(np.asarray([basis["b1_nm_inv"], basis["b2_nm_inv"]],
                                      dtype=float), float(delta_deg))
    updated = dict(basis)
    updated["b1_nm_inv"] = [float(value) for value in vectors[0]]
    updated["b2_nm_inv"] = [float(value) for value in vectors[1]]
    updated["notes"] = [*list(basis.get("notes", [])), str(note)]
    return updated


def basis_canvas_check(image, members_rad_px, radius_px, n_px, lambda_px_value,
                       band_fraction=BASIS_RING_BAND_FRACTION,
                       member_half_px=BASIS_RING_MEMBER_HALF_PX, nan_fill=0.0):
    """Unconditional validation of a resolved ring_1x1 basis against the canvas.

    The strongest spectrum magnitude inside the annulus
    ``radius_px * (1 +/- band_fraction)`` is the canvas' own ring.  Every resolved
    ring_1x1 member must reach ``BASIS_RING_STRENGTH_FRACTION`` of it within
    ``member_half_px`` px (the |psi| ratio) and must sit closer to the measured ring
    direction than ``BASIS_RING_OFFSET_FACTOR`` window radii, the arc at which the
    demodulation window attenuates the member to that same fraction.

    Returns a dict with both directions, the ratio, the offsets, ``mismatch``
    (bool), an English ``reason`` (None when nothing was flagged) and a one-line
    ``summary``.  "No evidence" cases (no FFT pixel in the annulus, a constant
    canvas) are recorded with ``mismatch`` False and their reason.
    """
    n = int(n_px)
    offsets = fft_offsets_px(n)
    columns = offsets[None, :]
    rows = offsets[:, None]
    radial = np.hypot(columns, rows)
    band = ((radial >= float(radius_px) * (1.0 - float(band_fraction)))
            & (radial <= float(radius_px) * (1.0 + float(band_fraction))))
    record = {
        "band_fraction": float(band_fraction), "radius_px": float(radius_px),
        "member_half_px": float(member_half_px),
        "strength_fraction": float(BASIS_RING_STRENGTH_FRACTION),
        "offset_factor": float(BASIS_RING_OFFSET_FACTOR),
        "window_radius_px": float(n / (TWO_PI * float(lambda_px_value))),
        "annulus_pixels": int(np.count_nonzero(band)),
        "annulus_peak_px": None, "annulus_peak_radius_px": None,
        "annulus_peak_angle_deg": None, "measured_angle_deg": None,
        "resolved_angle_deg": None, "annulus_strength": 0.0,
        "member_strength": 0.0, "strength_ratio": float("nan"),
        "worst_offset_px": None, "offset_tolerance_px": None,
        "mismatch": False, "reason": None, "summary": "",
    }
    if not bool(band.any()):
        record["reason"] = (f"no FFT pixel in the comparison annulus "
                            f"{float(radius_px):.3f} px +/- "
                            f"{100.0 * float(band_fraction):.0f} %")
        record["summary"] = f"not applicable: {record['reason']}"
        return record
    magnitude = canvas_spectrum_magnitude(image, nan_fill)
    band_values = np.where(band, magnitude, 0.0)
    peak_row, peak_column = np.unravel_index(int(np.argmax(band_values)),
                                             band_values.shape)
    annulus_strength = float(band_values[peak_row, peak_column])
    if annulus_strength <= 0.0:
        record["reason"] = ("the comparison annulus carries no spectrum magnitude "
                            "(a constant canvas)")
        record["summary"] = f"not applicable: {record['reason']}"
        return record
    peak_px = [float(columns[0, peak_column]), float(rows[peak_row, 0])]
    measured_angle = float(np.degrees(np.arctan2(peak_px[1], peak_px[0])))
    member_strength = 0.0
    member_angles = []
    for member in members_rad_px:
        q_px = rad_px_to_px_offsets(member, n)
        window = ((np.abs(columns - q_px[0]) <= float(member_half_px))
                  & (np.abs(rows - q_px[1]) <= float(member_half_px)))
        if bool(window.any()):
            member_strength = max(member_strength, float(magnitude[window].max()))
        member_angles.append(float(np.degrees(np.arctan2(q_px[1], q_px[0]))))
    ratio = member_strength / annulus_strength
    worst_offset = max(float(radius_px)
                       * abs(np.radians(fold_residual_deg(angle, measured_angle)))
                       for angle in member_angles)
    tolerance = BASIS_RING_OFFSET_FACTOR * record["window_radius_px"]
    mismatch = bool(ratio < BASIS_RING_STRENGTH_FRACTION or worst_offset > tolerance)
    record.update(
        annulus_peak_px=peak_px,
        annulus_peak_radius_px=float(np.hypot(*peak_px)),
        annulus_peak_angle_deg=measured_angle,
        measured_angle_deg=fold_deg(measured_angle),
        resolved_angle_deg=sorted({round(fold_deg(angle), 9) for angle in member_angles}),
        annulus_strength=annulus_strength, member_strength=member_strength,
        strength_ratio=float(ratio), worst_offset_px=float(worst_offset),
        offset_tolerance_px=float(tolerance), mismatch=mismatch)
    record["summary"] = (
        f"annulus {float(radius_px):.4f} px +/- "
        f"{100.0 * float(band_fraction):.0f} %, annulus peak {annulus_strength:.6g} at "
        f"{record['annulus_peak_radius_px']:.4f} px = {measured_angle:.4f} deg (folded "
        f"{record['measured_angle_deg']:.4f} deg), the resolved members reach "
        f"{member_strength:.6g} -> ratio {ratio:.6f} (threshold "
        f"{BASIS_RING_STRENGTH_FRACTION:g}), worst member offset {worst_offset:.4f} px "
        f"(tolerance {tolerance:.4f} px) -> "
        + ("MISMATCH" if mismatch else "OK"))
    if mismatch:
        record["reason"] = (
            f"the resolved ring_1x1 members (folded to 60 deg: "
            f"{record['resolved_angle_deg']} deg) reach only {100.0 * ratio:.3f} % of "
            f"the strongest spectrum magnitude of the annulus (threshold "
            f"{100.0 * BASIS_RING_STRENGTH_FRACTION:.0f} %) and the worst member sits "
            f"{worst_offset:.3f} px from the measured ring direction (tolerance "
            f"{tolerance:.3f} px), while the canvas' strongest in-band peak is at "
            f"{measured_angle:.4f} deg (folded {record['measured_angle_deg']:.4f} deg, "
            f"{record['annulus_peak_radius_px']:.3f} px): the resolved basis does not "
            "describe this canvas")
    return record


# --------------------------------------------------------------------------- #
# unit conversions of wave vectors
# --------------------------------------------------------------------------- #
def px_offsets_to_rad_px(offsets_px, n_px):
    """Signed fftshift offsets in px -> angular frequency in rad/px."""
    return np.asarray(offsets_px, dtype=float) * (TWO_PI / float(n_px))


def rad_px_to_px_offsets(rad_px, n_px):
    """Angular frequency in rad/px -> signed fftshift offsets in px."""
    return np.asarray(rad_px, dtype=float) * (float(n_px) / TWO_PI)


def nm_inv_to_rad_px(nm_inv, nm_per_px):
    """Angular frequency in rad/nm -> rad/px (rad/nm * nm/px)."""
    return np.asarray(nm_inv, dtype=float) * float(nm_per_px)


def q_vector(h, k, b1_rad_px, b2_rad_px):
    """``q(h, k) = h b1 + k b2`` in rad/px for arbitrary real ``h`` and ``k``."""
    return (float(h) * np.asarray(b1_rad_px, dtype=float)
            + float(k) * np.asarray(b2_rad_px, dtype=float))


# --------------------------------------------------------------------------- #
# the engine
# --------------------------------------------------------------------------- #
def lambda_px(lambda_nm, nm_per_px):
    """Window width in px (the Gaussian ``exp(-Lambda^2 |k|^2 / 2)`` uses rad/px)."""
    return float(lambda_nm) / float(nm_per_px)


def demodulate(image, q_rad_px, lambda_nm, nm_per_px, nan_fill=0.0):
    """One demodulated local field ``psi_q`` (complex128).

    ``T`` is multiplied by ``exp(-i q.r)``, which removes the carrier, and the
    result is low-passed with the Gaussian ``exp(-Lambda^2 |k|^2 / 2)`` evaluated
    on the FFT angular frequencies in rad/px.  NaN pixels of the input are filled
    with ``nan_fill`` (the skill's rule is zero) before the FFT; the caller keeps
    their positions in the validity mask.  The ``k = 0`` weight is exactly one, so
    ``sum_r psi_q(r) = sum_r T_fill(r) exp(-i q.r)`` holds exactly.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"the canvas must be square, got {array.shape}")
    n = int(array.shape[0])
    filled = np.where(np.isfinite(array), array, float(nan_fill))
    qx, qy = float(q_rad_px[0]), float(q_rad_px[1])
    columns = np.arange(n, dtype=float)[None, :]
    rows = np.arange(n, dtype=float)[:, None]
    carrier = np.exp(-1j * (qx * columns + qy * rows))
    spectrum = np.fft.fft2(filled * carrier)
    frequency = TWO_PI * np.fft.fftfreq(n)  # rad/px, unshifted FFT order
    width = lambda_px(lambda_nm, nm_per_px)
    window = np.exp(-0.5 * (width ** 2)
                    * (frequency[None, :] ** 2 + frequency[:, None] ** 2))
    return np.fft.ifft2(spectrum * window)


def demodulated_spectrum(image, q_rad_px, nan_fill=0.0):
    """``FFT[T exp(-i q.r)]`` before the window (used by the self-checks)."""
    array = np.asarray(image, dtype=float)
    n = int(array.shape[0])
    filled = np.where(np.isfinite(array), array, float(nan_fill))
    qx, qy = float(q_rad_px[0]), float(q_rad_px[1])
    columns = np.arange(n, dtype=float)[None, :]
    rows = np.arange(n, dtype=float)[:, None]
    carrier = np.exp(-1j * (qx * columns + qy * rows))
    return np.fft.fft2(filled * carrier)


def demodulated_sum(image, q_rad_px, nan_fill=0.0):
    """``sum_r T_fill(r) exp(-i q.r)``, the exact value of ``sum_r psi_q(r)``.

    The window's ``k = 0`` weight is one, so the two sums agree whatever the
    window width is; the self-test uses this identity as its strongest check.
    """
    array = np.asarray(image, dtype=float)
    n = int(array.shape[0])
    filled = np.where(np.isfinite(array), array, float(nan_fill))
    qx, qy = float(q_rad_px[0]), float(q_rad_px[1])
    columns = np.arange(n, dtype=float)[None, :]
    rows = np.arange(n, dtype=float)[:, None]
    return complex(np.sum(filled * np.exp(-1j * (qx * columns + qy * rows))))


def valid_mask(amplitude, nan_region, amplitude_fraction):
    """Validity mask: invalid = input NaN region UNION amplitude below threshold.

    The threshold is ``amplitude_fraction * median(amplitude)`` with the median
    taken over the whole canvas (the sibling convention).  Returns
    ``(valid, threshold)``.
    """
    reference = float(np.median(np.asarray(amplitude, dtype=float)))
    threshold = float(amplitude_fraction) * reference
    valid = (np.asarray(amplitude, dtype=float) >= threshold) & ~np.asarray(nan_region,
                                                                           dtype=bool)
    return valid, threshold


# --------------------------------------------------------------------------- #
# circular statistics (weighted, mirroring the phase-analysis conventions)
# --------------------------------------------------------------------------- #
def wrap_pm_pi(angle):
    """Wrap angles into ``(-pi, pi]``."""
    return (np.asarray(angle, dtype=float) + np.pi) % TWO_PI - np.pi


def circular_mean_rad(field, valid):
    """Amplitude-weighted circular mean ``arg sum psi`` plus the resultant ``R``.

    Weighting a phase sample by its own amplitude is exactly the same operation as
    summing the complex field, so the estimator is ``arg sum_r psi(r)`` over the
    selected pixels, and ``R = |sum psi| / sum |psi|``.
    """
    values = np.asarray(field)[np.asarray(valid, dtype=bool)]
    if values.size == 0:
        return float("nan"), float("nan"), 0.0
    total = float(np.sum(np.abs(values)))
    z = np.sum(values)
    if total <= 0.0 or float(np.abs(z)) == 0.0:
        return float("nan"), 0.0, total
    return float(np.angle(z)), float(np.abs(z) / total), total


def circular_median_rad(phi, weight=None, bins=3600, candidates=64):
    """Weighted circular median, the minimiser of ``sum_i w_i |wrap(t - phi_i)|``.

    Two stages: a coarse grid of ``bins`` candidate angles evaluated with the
    weighted histogram (the objective is Lipschitz, so the coarse minimum is
    within one bin), then an exact evaluation at the sample angles inside that
    bin.  The objective is piecewise linear with its kinks at the samples, so the
    exact stage lands on the true minimiser whenever fewer than ``candidates``
    samples fall in the bin.  Returns ``(median_rad, span_deg, n_minimisers)``.
    """
    phi = np.mod(np.asarray(phi, dtype=float).ravel(), TWO_PI)
    if weight is None:
        weight = np.ones_like(phi)
    weight = np.asarray(weight, dtype=float).ravel()
    keep = np.isfinite(phi) & np.isfinite(weight) & (weight > 0.0)
    phi, weight = phi[keep], weight[keep]
    if phi.size == 0:
        return float("nan"), float("nan"), 0
    if phi.size == 1:
        return float(phi[0]), 0.0, 1
    counts, _ = np.histogram(phi, bins=bins, range=(0.0, TWO_PI), weights=weight)
    centres = (np.arange(bins) + 0.5) * (TWO_PI / bins)
    objective = np.empty(bins)
    for start in range(0, bins, 256):
        block = centres[start:start + 256][:, None]
        objective[start:start + 256] = np.sum(
            counts[None, :] * np.abs(wrap_pm_pi(block - centres[None, :])), axis=1)
    coarse = float(centres[int(np.argmin(objective))])
    step = TWO_PI / bins
    inside = np.flatnonzero(np.abs(wrap_pm_pi(phi - coarse)) <= step)
    if inside.size == 0:
        return float(np.mod(coarse, TWO_PI)), 0.0, 1
    if inside.size > candidates:
        inside = inside[np.argsort(np.abs(wrap_pm_pi(phi[inside] - coarse)),
                                   kind="stable")[:candidates]]
    cand = phi[inside]
    values = np.array([float(np.sum(weight * np.abs(wrap_pm_pi(angle - phi))))
                       for angle in cand])
    best = float(values.min())
    spread = float(np.max(np.abs(wrap_pm_pi(cand - cand[int(np.argmin(values))]))))
    minimisers = int(np.count_nonzero(values <= best + 1e-12))
    order = np.argsort(values, kind="stable")
    ties = cand[order[:minimisers]]
    centre = float(np.angle(np.mean(np.exp(1j * ties))))
    return float(np.mod(centre, TWO_PI)), float(np.degrees(spread)), minimisers


def linear_median_fwhm(x, weight=None, bins=512, smooth_bins=2.0):
    """Median and FWHM (same unit as ``x``) of a non-circular weighted sample."""
    x = np.asarray(x, dtype=float).ravel()
    if weight is None:
        weight = np.ones_like(x)
    weight = np.asarray(weight, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(weight) & (weight > 0.0)
    x, weight = x[keep], weight[keep]
    if x.size == 0 or float(np.sum(weight)) <= 0.0:
        return float("nan"), float("nan")
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], weight[order]
    cumulative = np.cumsum(ws) / float(np.sum(ws))
    k = int(np.searchsorted(cumulative, 0.5, side="left"))
    median = float(xs[min(k, xs.size - 1)])
    lo, hi = float(xs[0]), float(xs[-1])
    if hi <= lo:
        return median, 0.0
    edges = np.linspace(lo, hi, bins + 1)
    hist, _ = np.histogram(xs, bins=edges, weights=ws)
    hist = _smooth_bins(hist, smooth_bins)
    # zero padding at both ends so that the half level is always reached, even when
    # the mode sits in the first or the last bin of the range
    pad = max(2, int(np.ceil(4.0 * smooth_bins)))
    hist = np.concatenate([np.zeros(pad), hist, np.zeros(pad)])
    top = float(hist.max())
    if top <= 0.0:
        return median, float("nan")
    peak = int(np.argmax(hist))
    half = top / 2.0
    step = (hi - lo) / bins

    def crossing(direction):
        k = 1
        while 0 <= peak + direction * k < hist.size:
            prev = hist[peak + direction * (k - 1)]
            here = hist[peak + direction * k]
            if here <= half <= prev and prev > here:
                return (k - 1) + (prev - half) / (prev - here)
            if here > half:
                k += 1
                continue
            return float(k)
        return None

    right, left = crossing(+1), crossing(-1)
    if right is None or left is None:
        return median, float("nan")
    return median, float((right + left) * step)


def _smooth_bins(values, sigma_bins):
    """Gaussian smoothing of a histogram (wrapped kernel, harmless off-circle)."""
    array = np.asarray(values, dtype=float)
    if sigma_bins <= 0.0:
        return array.copy()
    n = array.size
    half = max(1, min(int(np.ceil(4.0 * sigma_bins)), n - 1))
    offsets = np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([array[-half:], array, array[:half]])
    return np.convolve(padded, kernel, mode="same")[half:half + n]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def pair_cross_talk(q_rad_px_list, lambda_nm, nm_per_px):
    """Pairs whose separation is below ``2 / Lambda`` (the window separation law).

    All separations are reported in rad/nm (radian is dimensionless, so this is
    the same unit as ``1 / Lambda``); the condition ``|dq| >> 1 / Lambda`` of the
    windowed demodulation is enforced at the ``2 / Lambda`` level.
    """
    scale = float(nm_per_px)
    records = []
    for first in range(len(q_rad_px_list)):
        for second in range(first + 1, len(q_rad_px_list)):
            delta = ((np.asarray(q_rad_px_list[first], dtype=float)
                      - np.asarray(q_rad_px_list[second], dtype=float)) / scale)
            separation = float(np.hypot(*delta))
            records.append({"q_i": first, "q_j": second,
                            "separation_rad_per_nm": separation,
                            "threshold_rad_per_nm": 2.0 / float(lambda_nm),
                            "cross_talk": bool(separation < 2.0 / float(lambda_nm))})
    return records
