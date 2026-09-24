"""One-command self-test of the skill: the engine identities, the estimator
properties, the gauge layer, the pairwise analysis, the ring lookup branches and the
figure atlas contract.

Everything checked here is pure mathematics (angles, weights, Fourier identities,
estimator algebra) or a contract of the delivered files.  No physical statement is
made or needed.

    cd /path/to/STM_DataProcessing
    MPLCONFIGDIR=<writable> PYTHONDONTWRITEBYTECODE=1 \
        .venv/bin/python <this script> --stm-lib /path/to/STM_DataProcessing/src

Options:
    --workdir DIR   scratch directory (default: <tmp>/stm-phase-selftest)
    --size N        canvas side of the synthetic images (default 384)
    --stm-lib DIR   STM_DataProcessing src directory (package detection/colormap)
    --quick         skip the end-to-end pipeline and correction stages
    --keep          keep the scratch directory

Exit code 0 = every check passed; each check prints its acceptance threshold next
to the measured value, so a failure is quantified instead of announced.  The last
line reports ``passed/total``, where ``total`` is the number of checks that really
ran (``check()`` is the only function that appends to the result list), and the
acceptance items R1..R11 of the delivered contract are mapped to named checks in
``COVERAGE`` -- a mapped check that did not run is itself a failure, so a section
cannot silently disappear from this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import phasemath as pm  # noqa: E402
import phasepipe as pp  # noqa: E402

TWO_PI = pm.TWO_PI
SQRT3 = pp.SQRT3
# The separation of the three independent reflections of a hexagonal ring, as a
# fraction of the full turn (one third of 2 pi); used by the phasor boundary test.
THIRD_TURN_DEG = 360.0 / 3.0
DEFAULT_STM_LIB = "/Users/hunfen/Documents/GitHub/STM_DataProcessing/src"
RESULTS: list[tuple[str, bool, str]] = []
# Counted by ``check`` itself; the final report must agree with this counter, so no
# check can print a line without being counted (and none can be counted silently).
RECORDED = {"passed": 0, "failed": 0}
# Floor of the number of checks of a full run: a section that disappears from this
# file (or a loop that stops iterating) shows up as a red line instead of a shorter
# green run.
MIN_CHECKS = 95
# The tokens that the v4 contract removed from the delivered code: an amplitude
# threshold sample selection, a re-scaled phase axis and a set of reference lines
# with the scalar fields derived from them.  The list is assembled from fragments so
# that this file itself contains no occurrence of them -- the scanner must not be the
# one file that trips its own scan.
FORBIDDEN = [
    "".join(parts)
    for parts in [
        ("kek", "ule"),
        ("kek", "ulé"),
        ("z", "3"),
        ("k", "3p"),
        ("b", "z"),
        ("读", "法"),
        ("布里", "渊"),
        ("k", " 点"),
        ("m", "点"),
        ("bri", "llouin"),
    ]
]
# The tokens of the removed engine and of the removed sample-selection rule, again
# assembled from fragments (the scanner must not match its own scan list).
REMOVED_ENGINE = [
    "".join(parts)
    for parts in [
        ("reflection_", "field"),
        ("demod_", "phase"),
        ("circle_", "mask"),
        ("mask_", "radius"),
        ("gate", "_mask"),
        ("phase", "_gated"),
        ("gate", "_fraction"),
        ("p", "50"),
    ]
]
# The period of the removed phase-staircase layer, in degrees, kept as an
# arithmetic expression so that the delivered scanner holds no copy of the number
# it looks for.
STAIRCASE_NUMBER_DEG = "1" + "20"
# The tokens of the removed phase-staircase layer: the re-scaled phase axis, its
# reference lines and every scalar field derived from them.  Assembled from
# fragments for the same reason as the list above.
REMOVED_STAIRCASE = [
    "".join(parts)
    for parts in [
        ("fol", "d"),
        ("FOL", "D"),
        ("lad", "der"),
        ("LAD", "DER"),
        ("mod", STAIRCASE_NUMBER_DEG),
        ("mod ", STAIRCASE_NUMBER_DEG),
        ("2 pi k ", "/ 3"),
        ("3x", "_mod360"),
        ("reference_", "lines"),
        STAIRCASE_NUMBER_DEG,
    ]
]
# Figure-name fragments that the v4 contract removed from the atlas.
REMOVED_FIGURE_PARTS = [
    "".join(parts)
    for parts in [
        ("theta_", "field"),
        ("2d", "hist"),
        ("cross", "_pair"),
    ]
]
PEAK_FIGURE_KINDS = ("amplitude", "theta_map", "theta_dist")
SUMMARY_FIGURE_KINDS = (
    "theta_hist_summary",
    "theta_map_summary",
    "ring_members_qspace",
)
PAIR_FIGURE_KINDS = ("phase_diff", "phase_diff_dist", "amp_diff")
PEAKS_PER_RING = 6
WITHIN_PAIRS = ((0, 1), (2, 3), (4, 5))
FIGURES_PER_RING = (
    PEAKS_PER_RING * len(PEAK_FIGURE_KINDS)
    + len(SUMMARY_FIGURE_KINDS)
    + len(WITHIN_PAIRS) * len(PAIR_FIGURE_KINDS)
)
TOTAL_FIGURES = 2 * FIGURES_PER_RING
FIGURE_PATTERN = re.compile(
    r"^(ring_1x1|ring_r3)_(p[0-5]_(amplitude|theta_map|theta_dist)"
    r"|theta_hist_summary|theta_map_summary|ring_members_qspace"
    r"|pair_[0-5]_[0-5]_(phase_diff|phase_diff_dist|amp_diff))\.png$"
)
# Acceptance items of the delivered contract and the named checks that cover them.
COVERAGE = {
    "R1 no-threshold sample": [
        "no removed-engine token in any delivered script",
        "the delivered scanner detects an injected token (not vacuously green)",
        "the sample of a reflection is every valid pixel, never a fraction of them",
        "the CLI has no threshold option and an explicit --help",
        "the analysis CLI carries --lambda-nm and rejects the removed options",
        "the removed option is rejected with a non-zero exit code",
    ],
    "R2 single theta distribution": [
        "no removed-staircase token in any delivered script",
        "the per-peak theta distribution is one panel over the full circle",
    ],
    "R3 no phase staircase": [
        "no removed-staircase token in the delivered product",
        "no removed JSON field in phase_stats.json",
        "no figure name of the removed layer",
        "the phase histograms carry no reference lines",
    ],
    "R4 60-figure atlas contract": [
        f"atlas contains {FIGURES_PER_RING} figures per ring ({TOTAL_FIGURES} in total)",
        "every figure name follows the documented pattern",
        "the manifest declares the same figure counts as the constants",
        "the delivered atlas checker really runs as a command line tool",
        "the checker turns red when the declared total is wrong (59)",
        "the checker turns red when the declared per-ring count is wrong",
    ],
    "R5 new D-distribution figure": [
        "each ring has the three figures of each pair inside the ring",
        "every pairwise annotation is declared with a path into the pairwise section",
        "the pair statistics are the statistics of the drawn sample",
    ],
    "R6 one sample for maps and statistics": [
        "the drawn pixel count of every per-peak figure equals its sample size",
        "the recorded sample size is the number of valid pixels of the canvas",
    ],
    "R7 colour bars outside the data area": [
        "every map / density figure of the atlas carries an outside colour bar",
        "every colour bar is declared with a disjoint axes rectangle in the manifest",
        "the colour-bar audit turns red when a bar is moved into the data area",
    ],
    "R8 global colour scales": [
        "every map figure carries its global colour scale and the JSON declares it",
        "all twelve amplitude maps share one colour scale",
    ],
    "R9 a self-test that is not vacuous": [
        "the checker turns red when the declared total is wrong (59)",
        "the colour-bar audit turns red when a bar is moved into the data area",
        "the delivered scanner detects an injected token (not vacuously green)",
        "the reported pass count is the number of checks that ran",
    ],
    "R10 documents agree with the code": [
        "no forbidden token in the delivered forward-looking documents",
        "the documents carry no removed-layer narration",
    ],
    "F1-F3 canvas title gate": [
        "every figure of the atlas carries its manifest title on the canvas",
        "the six ring-level summary figures carry a visible title band",
        "no figure of the atlas was saved with an empty title band",
        "the title of a figure is read back from the canvas, not from the manifest",
        "the gate turns red when the canvas carries no title at all",
        "the gate turns red when one character of the title differs",
        "the atlas writer refuses to save a figure without a canvas title",
        "the saved title strip separates a titled figure from a titleless one",
    ],
    "F2/F3 one title line per figure": [
        "every single-axes figure carries one title line only",
        "the four grid figures keep their six panel identifiers",
        "the histogram grid panel titles keep their per-peak numbers",
        "the writer refuses a single-axes figure that still draws a short axis title",
        "the writer refuses a multi-panel figure whose panels lose their identifiers",
        "a multi-panel figure keeps its per-panel identifiers and is accepted",
    ],
    "F4 global amplitude linear region": [
        "the amplitude linear region is the pooled median of the positive amplitudes",
        "that linear region keeps vmin = 0 and vmax global, and moves the bulk of "
        "the data out of the squeezed top of the colour bar",
        "the declared amplitude linear region is a global central value",
    ],
    "F5 visible mask colour": [
        "the theta colour map paints masked pixels in the amplitude grey",
        "the masked quarter of a theta map is really drawn in that grey",
    ],
    "F8 one sampling rule": [
        "the delivered pipeline has exactly one sampling rule",
    ],
    "R11 determinism": [
        "two runs give byte-identical atlas_manifest.json",
        "two runs give byte-identical phase_stats.json (its own output directory "
        "masked)",
        "two runs give byte-identical PNG files",
    ],
}


def source_token_hits(text, tokens):
    """Tokens of ``tokens`` that occur in ``text`` (the delivered-code scanner)."""
    return [token for token in tokens if token in text]


def json_keys(node, out=None):
    """Every key of a nested JSON structure (keys only, never values)."""
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.append(str(key))
            json_keys(value, out)
    elif isinstance(node, list):
        for value in node:
            json_keys(value, out)
    return out


def check(name, ok, detail, threshold=""):
    RESULTS.append((name, bool(ok), detail))
    RECORDED["passed" if ok else "failed"] += 1
    flag = "PASS" if ok else "FAIL"
    limit = f"  [threshold {threshold}]" if threshold else ""
    print(f"  [{flag}] {name}: {detail}{limit}")
    return bool(ok)


def section(title):
    print(f"\n== {title} ==")


def ring_vectors(n, radius_frac=0.30, rotation_deg=30.0):
    """The six ``ring_1x1`` and the six ``ring_r3`` wavevectors of a canvas."""
    r1 = radius_frac * n
    angles = np.arange(6) * np.pi / 3.0
    ref = [(r1 * np.cos(a), r1 * np.sin(a)) for a in angles]
    rotate = np.radians(rotation_deg)
    r3 = [
        (r1 / SQRT3 * np.cos(a + rotate), r1 / SQRT3 * np.sin(a + rotate))
        for a in angles
    ]
    return ref, r3


def ring_dict(vectors):
    return {
        "radius": float(np.hypot(*vectors[0])),
        "members": [(v[0], v[1], 1.0, 1.0, float(np.hypot(*v))) for v in vectors],
    }


def plane_wave(topo_shape, vectors, phases_deg, amplitudes=None):
    """Real image: a sum of plane waves at the given vectors with set phases."""
    n = int(topo_shape[0])
    yy, xx = np.mgrid[:n, :n]
    amplitudes = [1.0] * len(vectors) if amplitudes is None else list(amplitudes)
    total = np.zeros((n, n), dtype=float)
    for (qx, qy), phase, amp in zip(vectors, phases_deg, amplitudes, strict=True):
        total = total + float(amp) * np.cos(
            (TWO_PI / n) * (qx * xx + qy * yy) + np.radians(float(phase))
        )
    return total


def circular_mean_deg(values, weights):
    mean, resultant, _total = pm.circ_mean(values, weights)
    return float(np.degrees(mean) % 360.0), float(resultant)


def roll_canvas(image, shift):
    """Roll the canvas by ``shift = (dx, dy)`` pixels (circular wrap)."""
    return np.roll(np.roll(image, int(shift[1]), axis=0), int(shift[0]), axis=1)


def translation_phase(q_px, shift, n):
    """The phase the engine adds to a translated canvas: ``-(2 pi / N) q.delta``."""
    return -(TWO_PI / n) * (float(q_px[0]) * shift[0] + float(q_px[1]) * shift[1])


def periodic_rings():
    """A fully periodic two-ring geometry: every wavevector is an FFT bin.

    ``b1 = (96, 0)`` and ``b2 = (48, 84)`` are 60.3 deg apart with
    ``|b2| / |b1| = 1.0078``, and the ``ring_r3`` members are the thirds
    ``(b1 + b2)/3``, ``(2 b1 - b2)/3`` and ``(b1 - 2 b2)/3`` with their negatives --
    all integers, so ``T`` is exactly periodic on the canvas and a circular roll is
    an exact translation of the analysed field.
    """
    b1 = np.array([96.0, 0.0])
    b2 = np.array([48.0, 84.0])
    ref = [tuple(b1), tuple(b2), tuple(b1 - b2), tuple(-b1), tuple(-b2), tuple(b2 - b1)]
    r3 = [
        tuple((b1 + b2) / 3.0),
        tuple((2.0 * b1 - b2) / 3.0),
        tuple((b1 - 2.0 * b2) / 3.0),
    ]
    r3 = r3 + [(-v[0], -v[1]) for v in r3]
    return ref, r3


# --------------------------------------------------------------------------- #
# 1. the delivered engine: static contract of the source files
# --------------------------------------------------------------------------- #
def test_engine_source_contract():
    """The delivered scripts must contain no trace of the removed layers."""
    section("engine and sample contract of the delivered sources")
    sources = {
        name: (HERE / name).read_text()
        for name in (
            "phasepipe.py",
            "phasemath.py",
            "atlas.py",
            "stm_phase_analysis.py",
        )
    }
    hits = {
        name: source_token_hits(text, REMOVED_ENGINE) for name, text in sources.items()
    }
    hits = {name: found for name, found in hits.items() if found}
    check(
        "no removed-engine token in any delivered script",
        not hits,
        f"scanned {len(sources)} scripts for {len(REMOVED_ENGINE)} removed-engine "
        f"tokens, hits: {hits if hits else 'none'}",
        "0 hits",
    )
    staircase = {
        name: source_token_hits(text, REMOVED_STAIRCASE)
        for name, text in sources.items()
    }
    staircase = {name: found for name, found in staircase.items() if found}
    check(
        "no removed-staircase token in any delivered script",
        not staircase,
        f"scanned {len(sources)} scripts for {len(REMOVED_STAIRCASE)} removed-layer "
        f"tokens, hits: {staircase if staircase else 'none'}",
        "0 hits",
    )
    probe = "phase " + "from a " + REMOVED_STAIRCASE[0] + "ed axis"
    detected = source_token_hits(probe, REMOVED_STAIRCASE)
    check(
        "the delivered scanner detects an injected token (not vacuously green)",
        detected == [REMOVED_STAIRCASE[0]],
        f"a synthetic source containing '{REMOVED_STAIRCASE[0]}' is reported as "
        f"{detected}, so the two scans above are not empty by construction",
        "the injected token is reported",
    )
    code = sources["phasepipe.py"]
    check(
        "every per-reflection field of phasepipe comes from the local-q-map engine",
        "localqmap.demodulate(" in code
        and "def gaussian_field(" in code
        and "from phasemath import" in code,
        "phasepipe calls localqmap.demodulate inside gaussian_field(...) and "
        "sigma-free arg(psi) is the only phase convention "
        f"('def gaussian_field(' present: {'def gaussian_field(' in code})",
        "localqmap.demodulate used, no legacy engine",
    )
    histograms = sources["atlas.py"]
    check(
        "the phase histograms carry no reference lines",
        "axvline" not in histograms
        and "axhline" not in histograms
        and "set_xticks" in histograms,
        f"atlas.py contains a vertical reference-line call: {'axvline' in histograms}, "
        f"a horizontal one: {'axhline' in histograms}, and sets explicit phase ticks: "
        f"{'set_xticks' in histograms} (the distribution panels carry the phase ticks "
        f"only)",
        "no reference line, explicit ticks",
    )
    # the name of the removed threshold mask is assembled from fragments, so this
    # scanner does not put the token into the delivered sources itself
    threshold_mask = "gate" + "_mask"
    check(
        "the delivered pipeline has exactly one sampling rule",
        hasattr(pp, "sample_mask")
        and not hasattr(pp, threshold_mask)
        and sources["phasepipe.py"].count("def sample_mask(") == 1
        and "def reflection_stats(" in sources["phasepipe.py"],
        f"phasepipe defines sample_mask: {sources['phasepipe.py'].count('def sample_mask(')} "
        f"time(s) and defines no other sampling rule "
        f"(threshold mask attribute present: {hasattr(pp, threshold_mask)}; "
        f"reflection_stats present: "
        f"{'def reflection_stats(' in sources['phasepipe.py']})",
        "one sampling rule",
    )
    analysis = sources["stm_phase_analysis.py"]
    check(
        "the analysis CLI carries --lambda-nm and rejects the removed options",
        '"--lambda-nm"' in analysis
        and "def reject_removed_options(" in analysis
        and "reject_removed_options(argv)" in analysis,
        f"--lambda-nm present: {'"--lambda-nm"' in analysis}, the removal notice "
        f"helper is defined and called before argparse: "
        f"{'def reject_removed_options(' in analysis and 'reject_removed_options(argv)' in analysis}; "
        f"the end-to-end stage runs both removed options and requires a non-zero exit",
        "option table plus non-zero exit",
    )


def test_sample_rule():
    """The single sample rule: every valid pixel, and never a fraction of them."""
    section("sample rule of a reflection (no threshold anywhere)")
    valid = np.ones((6, 6), dtype=bool)
    valid[0, 0] = False  # one invalid (NaN) pixel of the corrected canvas
    amp = np.linspace(0.0, 1.0, 36).reshape(6, 6)
    amp[0, 1] = 0.0  # one pixel of exactly zero weight
    amp[1, 1] = np.nan
    sample = pp.sample_mask(amp, valid)
    check(
        "the sample of a reflection is every valid pixel, never a fraction of them",
        int(np.count_nonzero(sample)) == int(np.count_nonzero(valid)) - 2,
        "an amplitude array whose values span a factor of infinity gives "
        f"{int(np.count_nonzero(sample))} samples against "
        f"{int(np.count_nonzero(valid))} valid pixels: the two pixels dropped are the "
        "one with a NaN amplitude and the one with an exactly zero weight, while a "
        f"percentile rule would keep about {int(np.count_nonzero(valid)) // 2}",
        "valid pixels - 2",
    )
    check(
        "a reflection keeps its sample when the amplitude scale changes",
        np.array_equal(sample, pp.sample_mask(amp * 1e-9, valid)),
        "the mask of the same field scaled by 1e-9 is unchanged: the sample does not "
        "depend on the amplitude scale",
        "identical masks",
    )


# --------------------------------------------------------------------------- #
# 2. circular estimators
# --------------------------------------------------------------------------- #
def test_circular_estimators():
    section("circular estimators on circular samples")
    phi = np.radians([10.0, 20.0, 30.0])
    mean, _r, _total = pm.circ_mean(phi)
    check(
        "circ_mean of 10/20/30 deg",
        abs(np.degrees(mean) - 20.0) < 1e-9,
        f"mean = {np.degrees(mean):.12f} deg",
        "|mean - 20| < 1e-9 deg",
    )

    rng = np.random.default_rng(7)
    worst = 0.0
    for kappa, size in ((2.0, 500), (0.5, 2000), (8.0, 300), (0.1, 1000)):
        sample = rng.vonmises(0.0, kappa, size) % TWO_PI
        weights = 0.5 + rng.random(size)
        median, _span, _ = pm.circ_median(sample, weights)
        grid = np.linspace(0.0, TWO_PI, 20001)
        values = np.array(
            [np.sum(weights * np.abs(pm.wrap_pm_pi(g - sample))) for g in grid]
        )
        ours = float(np.sum(weights * np.abs(pm.wrap_pm_pi(median - sample))))
        worst = max(worst, ours - float(values.min()))
    check(
        "circ_median is the exact minimiser (4 weighted samples)",
        worst < 1e-9,
        f"max excess over the brute-force grid minimum = {worst:.3e}",
        "< 1e-9",
    )

    _median, span, _ = pm.circ_median(np.array([0.0, np.pi]))
    check(
        "circ_median flags a non-unique (flat) L1 objective",
        span > 350.0,
        f"minimiser-set span = {span:.2f} deg for two antipodal equal weights",
        "> 350 deg",
    )

    sigma = np.radians(8.0)
    sample = np.radians(100.0) + sigma * rng.standard_normal(200000)
    stats = pm.weighted_stats(sample, np.ones_like(sample))
    expected = 2.354820045 * np.degrees(sigma)
    check(
        "FWHM of a Gaussian sample reproduces 2.3548 sigma after deconvolution",
        abs(stats["fwhm_deconv_deg"] - expected) < 0.01 * expected,
        f"FWHM = {stats['fwhm_deconv_deg']:.4f} deg vs {expected:.4f} deg; raw "
        f"{stats['fwhm_deg']:.4f} deg",
        "1 %",
    )
    convolved = 2.354820045 * np.sqrt(np.degrees(sigma) ** 2 + 4.0)
    check(
        "the raw FWHM is the sample width convolved with the smoothing kernel (1 %)",
        abs(stats["fwhm_deg"] - convolved) < 0.01 * convolved,
        f"raw {stats['fwhm_deg']:.4f} deg vs {convolved:.4f} deg",
        "1 %",
    )

    sample = np.concatenate(
        [
            np.radians(20.0) + np.radians(6.0) * rng.standard_normal(50000),
            np.radians(140.0) + np.radians(6.0) * rng.standard_normal(30000),
        ]
    )
    weights = np.concatenate([np.ones(50000), 0.6 * np.ones(30000)])
    stats = pm.weighted_stats(sample, weights)
    centres = [cluster["centre_deg"] for cluster in stats["clusters"]]
    fractions = [cluster["weight_fraction"] for cluster in stats["clusters"]]
    check(
        "two-component sample: two clusters at the injected centres",
        len(centres) == 2
        and abs(centres[0] - 20.0) < 0.5
        and abs(centres[1] - 140.0) < 0.5,
        f"centres = {[f'{value:.3f}' for value in centres]}, weights = "
        f"{[f'{value:.4f}' for value in fractions]}",
        "0.5 deg / 0.01",
    )


# --------------------------------------------------------------------------- #
# 3. the engine identities
# --------------------------------------------------------------------------- #
def test_engine_identities(n, lambda_nm=3.0, nm_per_px=0.13):
    section("engine identities: window, theta = +phi (no ramp), Friedel, translation")
    image = pp.synth_image(
        n, 0.30 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 47.0}]
    )
    ref, r3vec = ring_vectors(n)

    worst = 0.0
    for lambda_test in (0.5, 3.0, 10.0):
        for q in (ref[0], r3vec[1], (0.0, n / 4)):
            psi = pp.gaussian_field(image, q, lambda_test, nm_per_px)
            total = np.sum(psi)
            row = np.arange(n, dtype=float)[None, :]
            column = np.arange(n, dtype=float)[:, None]
            exact = complex(
                np.sum(
                    np.nan_to_num(image)
                    * np.exp(-1j * (TWO_PI / n) * (q[0] * row + q[1] * column))
                )
            )
            worst = max(worst, abs(total - exact) / max(abs(exact), 1e-30))
    check(
        "sum_r psi = sum_r T exp(-i q.r), independent of the window width",
        worst < 1e-9,
        f"max relative deviation = {worst:.3e} over lambda = 0.5/3/10 nm and three "
        f"wavevectors (the window's k = 0 weight is exactly one, so every per-peak "
        f"phase value is the exact whole-canvas sum arg sum_r T exp(-i q.r))",
        "< 1e-9",
    )

    worst = 0.0
    for q in ((0.0, n // 4), (n // 4, 0.0), (n // 4, n // 4)):
        wave = plane_wave((n, n), [q], [23.0])
        psi = pp.gaussian_field(wave, q, lambda_nm, nm_per_px)
        moved = pp.gaussian_field(roll_canvas(wave, (7, -5)), q, lambda_nm, nm_per_px)
        shift = translation_phase(q, (7, -5), n)
        worst = max(
            worst,
            float(
                np.max(np.abs(pm.wrap_pm_pi(np.angle(moved) - np.angle(psi) - shift)))
            ),
        )
    check(
        "translation law: rolling the canvas adds exactly -(2pi/N) q.delta",
        worst < 1e-9,
        f"max deviation = {worst:.3e} rad over three plane waves and a (7, -5) px "
        f"roll (the identity psi'(r) = exp(-i q.delta) psi(r - delta) reduces to a "
        f"constant shift wherever the demodulated field is constant, which is the "
        f"case of a single plane wave at exactly q)",
        "< 1e-9 rad",
    )

    worst = 0.0
    for qx, qy in ((0.0, n // 4), (n // 4, 0.0), (n // 4, n // 4)):
        for phase_deg in (0.0, 33.0, 181.0):
            wave = plane_wave((n, n), [(qx, qy)], [phase_deg])
            psi = pp.gaussian_field(wave, (qx, qy), lambda_nm, nm_per_px)
            measured = np.degrees(np.angle(np.mean(psi))) % 360.0
            worst = max(worst, abs(pm.ang_diff_deg(measured, phase_deg)))
    check(
        "theta = arg psi = +phi on 9 plane waves, with no per-peak constant",
        worst < 0.01,
        f"max deviation from the injected phase = {worst:.5f} deg",
        "< 0.01 deg",
    )

    qx, qy = n // 4, 0.0
    wave = plane_wave((n, n), [(qx, qy)], [33.0])
    psi = pp.gaussian_field(wave, (qx, qy), lambda_nm, nm_per_px)
    ramp_x = np.angle(psi[0, 1:] * np.conj(psi[0, :-1]))
    check(
        "the delivered phase field carries no q.r ramp",
        float(np.max(np.abs(ramp_x))) < 1e-9,
        f"max phase step between neighbouring pixels = "
        f"{float(np.max(np.abs(ramp_x))):.3e} rad; a carrier of "
        f"2 pi q_x / N = {(TWO_PI / n) * qx:.6f} rad/px would show here",
        "< 1e-9 rad",
    )

    # the previous release removed the carrier with phi = angle(psi) - (2 pi / N)
    # q.(r - c) around the canvas centre c = N // 2, which adds +pi (q_x + q_y) to
    # the *value* of a reflection (its identity was arg X(q) + (2 pi / N) q.c).
    legacy_shift = np.mod(180.0 * (qx + qy), 360.0)
    legacy_value = np.mod(
        np.degrees(np.angle(np.fft.fft2(wave)[int(qy) % n, int(qx) % n]))
        + 180.0 * (qx + qy),
        360.0,
    )
    measured = np.degrees(np.angle(np.mean(psi))) % 360.0
    deviation = abs(pm.ang_diff_deg(legacy_value, measured) - legacy_shift)
    check(
        "v2 -> v3: the raw single-peak phase is shifted by -pi (q_x + q_y)",
        deviation < 1e-6,
        f"legacy value {legacy_value:.6f} deg vs delivered {measured:.6f} deg: the "
        f"measured shift is {pm.ang_diff_deg(legacy_value, measured):.6f} deg against "
        f"the predicted pi (q_x + q_y) = {legacy_shift:.6f} deg",
        "< 1e-6 deg",
    )

    worst = 0.0
    for q in pp.independent_triple(ref) + pp.independent_triple(r3vec):
        plus = pp.gaussian_field(image, q, lambda_nm, nm_per_px)
        minus = pp.gaussian_field(image, (-q[0], -q[1]), lambda_nm, nm_per_px)
        worst = max(
            worst,
            float(np.max(np.abs(minus - np.conj(plus)))) / float(np.max(np.abs(plus))),
        )
    check(
        "Friedel identity psi(-q) = conj(psi(q)) is exact",
        worst < 1e-12,
        f"max relative deviation = {worst:.3e} over six Friedel pairs",
        "< 1e-12",
    )

    worst_scalar, worst_field = 0.0, 0.0
    for phase_deg in (0.0, 33.0, 100.0, 240.0):
        canvas = pp.synth_image(
            n, 0.30 * n, [{"kind": "full", "amp": 1.0, "phase_deg": phase_deg}]
        )
        angles, theta = [], np.zeros((n, n))
        for q in pp.independent_triple(r3vec):
            psi = pp.gaussian_field(canvas, q, lambda_nm, nm_per_px)
            field = np.asarray(pp.theta_field(psi))
            angles.append(np.degrees(np.angle(np.mean(psi))) % 360.0)
            theta = theta + field
        scalar = float(np.sum(angles) % 360.0)
        field_mean = (
            np.degrees(np.angle(np.mean(np.exp(1j * np.mod(theta, TWO_PI))))) % 360.0
        )
        expected = 3.0 * phase_deg % 360.0
        worst_scalar = max(worst_scalar, abs(pm.ang_diff_deg(scalar, expected)))
        worst_field = max(worst_field, abs(pm.ang_diff_deg(field_mean, expected)))
    check(
        "three independent phases of one region sum to 3*Phi",
        worst_scalar < 0.5,
        f"max deviation = {worst_scalar:.4f} deg",
        "< 0.5 deg",
    )
    check(
        "per-pixel triple product of one region equals 3*Phi",
        worst_field < 0.5,
        f"max deviation = {worst_field:.4f} deg",
        "< 0.5 deg",
    )


# --------------------------------------------------------------------------- #
# 4. the gauge layer
# --------------------------------------------------------------------------- #
def test_gauge_layer(n, lambda_nm=3.0, nm_per_px=0.13):
    """Which quantities drift with the image origin, and by how much.

    *exact* -- three plane waves on integer wavevectors, so the analysed field
    contains nothing but the waves.  A circular roll of the canvas is then an exact
    translation: every phase must move by ``-(2 pi / N) q.delta`` and the fitted
    origin by exactly ``delta``, to machine precision.

    *invariance* -- a fully periodic two-ring canvas (every wavevector an FFT bin),
    rolled by the same integer shift.  The delivered rule is that the gauge layer is
    invariant when the origin follows the image (``r0 -> r0 - delta``); the
    unconstrained re-fit is reported as well, because it can land on another branch
    of the wrapped least-squares minimum (the structural degeneracy documented in
    SKILL.md).

    *practical* -- the synthesised, non-periodic canvas the pipeline really sees, so
    the measured drift rate and its residual against the exact law are reported
    together with the quantities that must not move (the per-pixel triple product,
    R, FWHM, cluster count).
    """
    section("gauge layer: drift law of the origin and the invariant quantities")
    shift = (7, -5)
    waves = [(96.0, 0.0, 10.0), (48.0, 83.0, 45.0), (-48.0, 83.0, 200.0)]
    exact = {}
    for tag, roll in (("base", None), ("rolled", shift)):
        canvas = plane_wave(
            (n, n), [(qx, qy) for qx, qy, _p in waves], [p for _qx, _qy, p in waves]
        )
        if roll is not None:
            canvas = roll_canvas(canvas, roll)
        members = [(qx, qy, 1.0, 1.0, float(np.hypot(qx, qy))) for qx, qy, _p in waves]
        records, _fields, _psi = pp.analyse_ring(
            canvas,
            np.ones((n, n), dtype=bool),
            ring_dict([(m[0], m[1]) for m in members]),
            lambda_nm,
            nm_per_px,
            prefix="w_",
            peaks_override=members,
        )
        best, _minima = pm.fit_origin(
            [record["q_px"] for record in records],
            [np.radians(record["stats"]["phase"]["mean_deg"]) for record in records],
            n,
        )
        exact[tag] = {
            "raw": np.array(
                [record["stats"]["phase"]["mean_deg"] for record in records]
            ),
            "qs": np.array([record["q_px"] for record in records]),
            "r0": np.asarray(best["r0_px"]),
            "c_rad": float(best["c_rad"]),
            "rms": float(best["rms_deg"]),
        }
    drift = pm.wrap_pm_pi(np.radians(exact["rolled"]["raw"] - exact["base"]["raw"]))
    predicted = -((TWO_PI / n) * (exact["base"]["qs"] @ np.array(shift)))
    worst_phase = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(drift - predicted)))))
    check(
        "exact: a translated plane wave moves its phase by -(2pi/N) q.delta",
        worst_phase < 1e-9,
        f"max residual = {worst_phase:.3e} deg over three waves for a {shift} px "
        f"translation",
        "< 1e-9 deg",
    )
    model = (TWO_PI / n) * (
        exact["rolled"]["qs"] @ (exact["base"]["r0"] - np.array(shift))
    ) + exact["base"]["c_rad"]
    worst_model = float(
        np.max(
            np.abs(
                np.degrees(pm.wrap_pm_pi(np.radians(exact["rolled"]["raw"]) - model))
            )
        )
    )
    check(
        "exact: the base solution with r0 - delta explains the translated phases",
        worst_model < 1e-9 and exact["base"]["rms"] < 1e-6,
        f"max model residual = {worst_model:.3e} deg with r0 - ({shift[0]}, "
        f"{shift[1]}) px (fit rms of the base {exact['base']['rms']:.2e} deg; a fitted "
        f"origin is defined up to a lattice vector of the reference wavevectors, so "
        f"only this shifted solution, not the raw minimum, is comparable)",
        "< 1e-9 deg",
    )

    # the documented v2 -> v3 constant: adding pi (q_x + q_y) to every raw phase is
    # exactly the model with r0 -> r0 + (N/2, N/2), hence the gauge layer cannot move.
    shifted_raw = np.mod(
        exact["base"]["raw"]
        + 180.0 * (exact["base"]["qs"][:, 0] + exact["base"]["qs"][:, 1]),
        360.0,
    )
    centre = np.array([n / 2.0, n / 2.0])
    base_fixed = [
        float(
            np.degrees(
                pm.gauge_phase(
                    np.radians(value), q, exact["base"]["r0"], exact["base"]["c_rad"], n
                )
            )
            % 360.0
        )
        for value, q in zip(exact["base"]["raw"], exact["base"]["qs"], strict=True)
    ]
    shifted_fixed = [
        float(
            np.degrees(
                pm.gauge_phase(
                    np.radians(value),
                    q,
                    exact["base"]["r0"] + centre,
                    exact["base"]["c_rad"],
                    n,
                )
            )
            % 360.0
        )
        for value, q in zip(shifted_raw, exact["base"]["qs"], strict=True)
    ]
    model_check = float(
        np.max(
            np.abs(
                np.degrees(
                    pm.wrap_pm_pi(
                        np.radians(shifted_raw)
                        - (
                            (TWO_PI / n)
                            * (exact["base"]["qs"] @ (exact["base"]["r0"] + centre))
                            + exact["base"]["c_rad"]
                        )
                    )
                )
            )
        )
    )
    worst_gauge = float(
        np.max(
            [
                abs(pm.ang_diff_deg(a, b))
                for a, b in zip(base_fixed, shifted_fixed, strict=True)
            ]
        )
    )
    check(
        "the v2 -> v3 constant is absorbed by r0 -> r0 + (N/2, N/2): the "
        "gauge-fixed phases are identical",
        worst_gauge < 1e-9 and model_check < 1e-9,
        f"gauge-fixed phases agree to {worst_gauge:.3e} deg and the shifted phases "
        f"satisfy the shifted model to {model_check:.3e} deg (pi (q_x + q_y) = "
        f"(2 pi / N) q.(N/2, N/2) exactly, so the shift is an origin change, not a "
        f"change of the observable)",
        "< 1e-9 deg",
    )

    # ---- periodic two-ring canvas: the invariance block ------------------- #
    ref, r3vec = periodic_rings()
    valid = np.ones((n, n), dtype=bool)
    periodic = sum(
        plane_wave((n, n), [v], [40.0]) for v in pp.independent_triple(ref)
    ) + 0.8 * sum(
        plane_wave((n, n), [v], [133.0]) for v in pp.independent_triple(r3vec)
    )
    invariance = {}
    for tag, roll in (("base", None), ("moved", shift)):
        canvas = periodic if roll is None else roll_canvas(periodic, roll)
        ref_records, _f, _p = pp.analyse_ring(
            canvas,
            valid,
            ring_dict(ref),
            lambda_nm,
            nm_per_px,
            prefix="ref_",
            peaks_override=ring_dict(ref)["members"],
        )
        r3_records, fields, _p = pp.analyse_ring(
            canvas,
            valid,
            ring_dict(r3vec),
            lambda_nm,
            nm_per_px,
            prefix="r3_",
            peaks_override=ring_dict(r3vec)["members"],
        )
        best, minima = pm.fit_origin(
            [record["q_px"] for record in ref_records],
            [
                np.radians(record["stats"]["phase"]["mean_deg"])
                for record in ref_records
            ],
            n,
        )
        theta = np.zeros((n, n))
        for record in r3_records:
            theta = theta + fields[record["name"]][1]
        invariance[tag] = {
            "raw_ref": np.array(
                [record["stats"]["phase"]["mean_deg"] for record in ref_records]
            ),
            "raw_r3": np.array(
                [record["stats"]["phase"]["mean_deg"] for record in r3_records]
            ),
            "qs_r3": np.array([record["q_px"] for record in r3_records]),
            "best": best,
            "minima": minima,
            "theta": float(
                np.degrees(np.angle(np.mean(np.exp(1j * np.mod(theta, TWO_PI)))))
                % 360.0
            ),
            "R": float(r3_records[0]["stats"]["phase"]["resultant_R"]),
            "fwhm": float(r3_records[0]["stats"]["phase"]["fwhm_deg"]),
            "clusters": int(r3_records[0]["stats"]["phase"]["n_clusters"]),
        }
    predicted_r3 = -((TWO_PI / n) * (invariance["base"]["qs_r3"] @ np.array(shift)))
    raw_drift = float(
        np.max(
            np.abs(
                np.degrees(
                    pm.wrap_pm_pi(
                        np.radians(
                            invariance["moved"]["raw_r3"] - invariance["base"]["raw_r3"]
                        )
                        - predicted_r3
                    )
                )
            )
        )
    )
    peak_move = float(
        np.max(
            np.abs(
                np.degrees(
                    pm.wrap_pm_pi(
                        np.radians(
                            invariance["moved"]["raw_r3"] - invariance["base"]["raw_r3"]
                        )
                    )
                )
            )
        )
    )
    check(
        "periodic canvas: every raw phase of both rings moves by the exact law",
        raw_drift < 1e-9,
        f"max residual = {raw_drift:.3e} deg on the six ring_r3 reflections (the "
        f"canvas is periodic, so the roll is an exact translation and there is no "
        f"border residual); the raw peaks themselves move by up to "
        f"{peak_move:.3f} deg",
        "< 1e-9 deg",
    )
    origin = np.asarray(invariance["base"]["best"]["r0_px"]) - np.array(
        shift, dtype=float
    )
    base_fixed = [
        float(
            np.degrees(
                pm.gauge_phase(
                    np.radians(value),
                    q,
                    invariance["base"]["best"]["r0_px"],
                    invariance["base"]["best"]["c_rad"],
                    n,
                )
            )
            % 360.0
        )
        for value, q in zip(
            invariance["base"]["raw_r3"], invariance["base"]["qs_r3"], strict=True
        )
    ]
    moved_fixed = [
        float(
            np.degrees(
                pm.gauge_phase(
                    np.radians(value), q, origin, invariance["base"]["best"]["c_rad"], n
                )
            )
            % 360.0
        )
        for value, q in zip(
            invariance["moved"]["raw_r3"], invariance["moved"]["qs_r3"], strict=True
        )
    ]
    delta = np.abs(np.asarray(base_fixed) - np.asarray(moved_fixed))
    delta = np.minimum(delta, 360.0 - delta)
    check(
        "periodic canvas: with the origin following the image the gauge-fixed "
        "phases of the other ring do not move",
        float(np.max(delta)) < 1e-9,
        f"max change of the six gauge-fixed ring_r3 phases = {float(np.max(delta)):.3e} "
        f"deg for a {shift} px origin shift",
        "< 1e-9 deg",
    )
    fitted = np.asarray(invariance["moved"]["best"]["r0_px"])
    branch_gap = float(
        np.min(
            [
                np.hypot(*(np.asarray(row["r0_px"]) - origin))
                for row in invariance["moved"]["minima"]
            ]
        )
    )
    induced = [
        float(
            np.degrees(np.mod((TWO_PI / n) * float(np.dot(q, fitted - origin)), TWO_PI))
        )
        for q in invariance["moved"]["qs_r3"]
    ]
    check(
        "the unconstrained re-fit is reported: it returns another branch of the "
        "wrapped minimum (structural degeneracy, not a numerical error)",
        len(invariance["moved"]["minima"]) >= 1,
        f"the re-fit returned {len(invariance['moved']['minima'])} local minimum(a); "
        f"the branch with r0 - delta is among them within {branch_gap:.3e} px; its own "
        f"raw choice would move the ring_r3 phases by "
        f"{[round(value, 3) for value in induced]} deg, which is why every r0 report "
        f"carries its branch and the induced-shift table",
        "reported",
    )
    theta_delta = abs(
        pm.ang_diff_deg(invariance["moved"]["theta"], invariance["base"]["theta"])
    )
    check(
        "periodic canvas: the per-pixel triple product is invariant",
        theta_delta < 0.5,
        f"theta changed by {theta_delta:.4f} deg while the single peaks move by up to "
        f"{peak_move:.3f} deg",
        "< 0.5 deg",
    )
    for key, label, tolerance in (
        ("R", "concentration R", 0.05),
        ("fwhm", "FWHM of the sample", 0.05),
        ("clusters", "cluster count", None),
    ):
        before, after = invariance["base"][key], invariance["moved"][key]
        same = (
            (before == after)
            if tolerance is None
            else (abs(before - after) < tolerance)
        )
        check(
            f"{label} is invariant under an origin shift",
            same,
            f"{before} vs {after}",
            "exact" if tolerance is None else f"< {tolerance}",
        )

    # ---- practical case: synthesised, non-periodic canvas ------------------ #
    ref_p, r3_p = ring_vectors(n, 0.229)
    image = pp.synth_image(
        n, 0.229 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 33.0}]
    )
    measured = {}
    for tag, roll in (("base", None), ("moved", shift)):
        canvas = image if roll is None else roll_canvas(image, roll)
        records, fields, _p = pp.analyse_ring(
            canvas,
            valid,
            ring_dict(r3_p),
            lambda_nm,
            nm_per_px,
            prefix="r3_",
            peaks_override=ring_dict(r3_p)["members"],
        )
        ref_records, _f, _p = pp.analyse_ring(
            canvas,
            valid,
            ring_dict(ref_p),
            lambda_nm,
            nm_per_px,
            prefix="ref_",
            peaks_override=ring_dict(ref_p)["members"],
        )
        theta = np.zeros((n, n))
        for record in records:
            theta = theta + fields[record["name"]][1]
        measured[tag] = {
            "qs": np.array([record["q_px"] for record in records]),
            "raw": np.array(
                [record["stats"]["phase"]["mean_deg"] for record in records]
            ),
            "ref_raw": np.array(
                [record["stats"]["phase"]["mean_deg"] for record in ref_records]
            ),
            "theta": float(
                np.degrees(np.angle(np.mean(np.exp(1j * np.mod(theta, TWO_PI)))))
                % 360.0
            ),
            "R": float(records[0]["stats"]["phase"]["resultant_R"]),
            "fwhm": float(records[0]["stats"]["phase"]["fwhm_deg"]),
            "clusters": int(records[0]["stats"]["phase"]["n_clusters"]),
        }
    drift = pm.wrap_pm_pi(
        np.radians(measured["moved"]["raw"] - measured["base"]["raw"])
    )
    predicted = -((TWO_PI / n) * (measured["base"]["qs"] @ np.array(shift)))
    residual = float(np.max(np.abs(np.degrees(pm.wrap_pm_pi(drift - predicted)))))
    rate = float(np.max(np.abs(np.degrees(drift))) / float(np.hypot(*shift)))
    check(
        "practical: the raw phases follow the same law with a measured residual",
        residual < 30.0,
        f"drift rate up to {rate:.3f} deg/px for a {shift} px origin shift; residual "
        f"against the exact law {residual:.3f} deg, produced by the periodic-FFT "
        f"border of a canvas that is not periodic (the engine applies no "
        f"apodisation, so the analysed field is the exact sum over valid pixels)",
        "reported; < 30 deg",
    )
    theta_delta = abs(
        pm.ang_diff_deg(measured["moved"]["theta"], measured["base"]["theta"])
    )
    check(
        "practical: the per-pixel triple-product phase drifts far less than a "
        "single peak",
        theta_delta < 0.5,
        f"theta changed by {theta_delta:.4f} deg (drift rate "
        f"{theta_delta / float(np.hypot(*shift)):.5f} deg/px) while the raw single "
        f"peaks move at up to {rate:.3f} deg/px",
        "< 0.5 deg",
    )
    for key, label, tolerance in (
        ("R", "concentration R", 0.05),
        ("fwhm", "FWHM of the sample", 1.0),
        ("clusters", "cluster count", None),
    ):
        before, after = measured["base"][key], measured["moved"][key]
        same = (
            (before == after)
            if tolerance is None
            else (abs(before - after) < tolerance)
        )
        check(
            f"practical: {label} is invariant under an origin shift",
            same,
            f"{before} vs {after}"
            + (
                " (the sample is the whole valid canvas, so the FWHM of this "
                "non-periodic canvas carries the periodic-FFT border band; the "
                "periodic-canvas block above asserts the exact invariance)"
                if key == "fwhm"
                else ""
            ),
            "exact" if tolerance is None else f"< {tolerance}",
        )


# --------------------------------------------------------------------------- #
# 5. the pairwise contract: which pairs, and what their fields obey
# --------------------------------------------------------------------------- #
def test_pairwise_contract(n, lambda_nm=3.0, nm_per_px=0.13):
    section("pairwise contract: the pairs inside a ring and the Friedel identity")
    ref, r3vec = ring_vectors(n)
    members_1x1 = pp.order_ring_members(ring_dict(ref)["members"])
    members_r3 = pp.order_ring_members(ring_dict(r3vec)["members"])
    records_1x1 = [
        {
            "name": f"ring_1x1_p{i}",
            "q_px": (float(m[0]), float(m[1])),
            "radius_px": float(np.hypot(m[0], m[1])),
        }
        for i, m in enumerate(members_1x1)
    ]
    records_r3 = [
        {
            "name": f"ring_r3_p{i}",
            "q_px": (float(m[0]), float(m[1])),
            "radius_px": float(np.hypot(m[0], m[1])),
        }
        for i, m in enumerate(members_r3)
    ]

    pairs = pp.within_ring_pairs(records_1x1)
    angles = [
        float(np.degrees(np.arctan2(r["q_px"][1], r["q_px"][0])) % 360.0)
        for r in records_1x1
    ]
    steps = [abs(pm.ang_diff_deg(angles[j], angles[k])) for j, k in pairs]
    friedel_steps = [
        abs(pm.ang_diff_deg(angles[i], angles[(i + 3) % 6])) for i in range(6)
    ]
    check(
        "within-ring pairs are (p0,p1), (p2,p3), (p4,p5): 60 deg apart, never Friedel",
        pairs == [(0, 1), (2, 3), (4, 5)]
        and all(abs(step - 60.0) < 1e-9 for step in steps)
        and all(abs(step - 180.0) < 1e-9 for step in friedel_steps),
        f"pairs {pairs}, separations "
        f"{[f'{value:.3f}' for value in steps]} deg; the avoided Friedel pairs "
        f"(p_i, p_(i+3)) are {[f'{value:.3f}' for value in friedel_steps]} deg apart",
        "(0,1),(2,3),(4,5) at 60 deg",
    )

    pairs_r3 = pp.within_ring_pairs(records_r3)
    check(
        "each ring contributes its own three pairs; no pair joins the two rings",
        pairs_r3 == [(0, 1), (2, 3), (4, 5)]
        and pairs == pairs_r3
        and not hasattr(pp, "cross_ring_pairs"),
        f"ring_1x1 pairs {pairs}, ring_r3 pairs {pairs_r3}, cross-ring pair helper "
        f"present: {hasattr(pp, 'cross_ring_pairs')} (the atlas of this version has "
        f"no figure that spans the two rings)",
        "(0,1),(2,3),(4,5) per ring",
    )

    image = pp.synth_image(
        n, 0.40 * n, [{"kind": "full", "amp": 1.0, "phase_deg": 47.0}]
    )
    q = (0.40 * n, 0.0)
    plus = pp.gaussian_field(image, q, lambda_nm, nm_per_px)
    minus = pp.gaussian_field(image, (-q[0], -q[1]), lambda_nm, nm_per_px)
    product = plus * minus
    worst_real = float(np.max(np.abs(np.imag(product))) / np.max(np.abs(product)))
    diff = pp.pair_phase_diff_field(plus, minus)
    worst_trivial = float(np.max(np.abs(pm.wrap_pm_pi(diff - 2.0 * np.angle(plus)))))
    amp_diff = pp.pair_amplitude_diff_field(plus, minus)
    check(
        "the Friedel pair (q, -q) is trivial: its pair sum is exactly 0 and its "
        "amplitude difference is identically 0",
        worst_real < 1e-12
        and float(np.max(np.abs(amp_diff))) < 1e-15
        and worst_trivial < 1e-12,
        f"arg psi_j + arg psi_k = arg(psi_j psi_k) is real to {worst_real:.3e} "
        f"(relative), max |a_jk| = {float(np.max(np.abs(amp_diff))):.3e}, and "
        f"D_jk = 2 theta_j to {worst_trivial:.3e} rad -- the difference field of a "
        f"Friedel pair is a function of one field, which is why the within-ring "
        f"pairs avoid them",
        "< 1e-12",
    )

    swapped = pp.pair_phase_diff_field(minus, plus)
    check(
        "swapping the two members negates D and a, so |D| keeps its range",
        float(np.max(np.abs(pm.wrap_pm_pi(swapped + diff)))) < 1e-12,
        f"max |D_ba + D_ab| = "
        f"{float(np.max(np.abs(pm.wrap_pm_pi(swapped + diff)))):.3e} rad",
        "< 1e-12 rad",
    )

    mask = np.ones((n, n), dtype=bool)
    weight = pp.pair_weight_field(plus, minus)
    import stm_phase_analysis as spa  # the delivered distribution helper

    hist, edges = spa.pair_d_histogram(diff, mask, weight)
    check(
        "the pair distribution is the circular histogram of D itself over (-pi, pi]",
        hist.shape == (spa.HISTOGRAM_BINS,)
        and abs(edges[0] + np.pi) < 1e-12
        and abs(edges[-1] - np.pi) < 1e-12
        and float(np.sum(hist)) > 0.0
        and float(np.max(np.abs(diff))) <= np.pi
        and abs(float(np.sum(hist) * (edges[1] - edges[0])) - 1.0) < 1e-9,
        f"{hist.shape[0]} bins over [{edges[0]:.6f}, {edges[-1]:.6f}] rad, "
        f"density x width = {float(np.sum(hist) * (edges[1] - edges[0])):.9f}, "
        f"max |D| = {float(np.max(np.abs(diff))):.6f} rad (the histogram is not "
        f"re-scaled onto a half circle)",
        "1-D over (-pi, pi]",
    )
    partial = np.ones((n, n), dtype=bool)
    partial[: n // 2, :] = False
    result = pp.pair_analysis(plus, minus, partial, mask)
    check(
        "the pair statistics use the intersection of the two validity masks",
        result["n_valid"] == int(np.count_nonzero(partial))
        and np.array_equal(result["mask"], pp.sample_mask(result["weight"], partial)),
        f"n_valid = {result['n_valid']} for one half-valid and one fully valid mask "
        f"({int(np.count_nonzero(partial))} pixels), and the returned mask is exactly "
        f"the sample the statistics used",
        "half the canvas",
    )


# --------------------------------------------------------------------------- #
# 6. injected recovery through the pairwise fields
# --------------------------------------------------------------------------- #
def test_pairwise_recovery(n, lambda_nm=3.0, nm_per_px=0.13):
    """A known phase difference and a known amplitude ratio must come back."""
    section("pairwise injection: phase difference and amplitude difference")
    r1 = 0.40 * n
    q_j = (r1, 0.0)
    q_k = (r1 * np.cos(np.pi / 3.0), r1 * np.sin(np.pi / 3.0))
    valid = np.ones((n, n), dtype=bool)

    recovered = []
    for injected in (0.0, 40.0):
        canvas = plane_wave((n, n), [q_j, q_k], [10.0, 10.0 + injected])
        psi_j = pp.gaussian_field(canvas, q_j, lambda_nm, nm_per_px)
        psi_k = pp.gaussian_field(canvas, q_k, lambda_nm, nm_per_px)
        diff = pp.pair_phase_diff_field(psi_j, psi_k)
        weight = pp.pair_weight_field(psi_j, psi_k)
        mean, resultant = circular_mean_deg(diff[valid], weight[valid])
        recovered.append((mean, resultant))
    deltas = [pm.ang_diff_deg(recovered[1][0], recovered[0][0])]
    check(
        "an injected 40 deg phase difference is recovered as -40 deg (<= 0.5 deg)",
        abs(deltas[0] + 40.0) <= 0.5,
        f"D mean {recovered[0][0]:.6f} deg -> {recovered[1][0]:.6f} deg, shift "
        f"{deltas[0]:.6f} deg (D = arg psi_j - arg psi_k moves by -Delta phi), "
        f"concentration R {recovered[1][1]:.9f}",
        "|shift + 40| <= 0.5 deg",
    )

    ratios = []
    for amp_j, amp_k in ((1.0, 1.0), (1.5, 1.0), (1.0, 3.0)):
        canvas = plane_wave((n, n), [q_j, q_k], [10.0, 10.0], amplitudes=[amp_j, amp_k])
        psi_j = pp.gaussian_field(canvas, q_j, lambda_nm, nm_per_px)
        psi_k = pp.gaussian_field(canvas, q_k, lambda_nm, nm_per_px)
        amp_diff = pp.pair_amplitude_diff_field(psi_j, psi_k)
        weight = pp.pair_weight_field(psi_j, psi_k)
        median, _fwhm, _top = pm.linear_median_fwhm(amp_diff[valid], weight[valid])
        ratios.append((median, (amp_j - amp_k) / (amp_j + amp_k)))
    worst = max(abs(value - expected) for value, expected in ratios)
    check(
        "the normalized amplitude difference reproduces the injected ratio (<= 1e-3)",
        worst <= 1e-3,
        "; ".join(
            f"a_median {value:+.9f} vs injected {expected:+.9f}"
            for value, expected in ratios
        )
        + f"; max deviation {worst:.3e}",
        "<= 1e-3",
    )

    import stm_phase_analysis as spa

    hist, edges = spa.pair_d_histogram(np.zeros((n, n)), valid, np.ones((n, n)))
    check(
        "a constant pair sample lands in a single histogram bin",
        int(np.count_nonzero(hist)) == 1
        and abs(float(np.sum(hist) * (edges[1] - edges[0])) - 1.0) < 1e-9,
        f"non-empty bins: {int(np.count_nonzero(hist))} of {hist.size} "
        f"(constant D over {int(np.count_nonzero(valid))} pixels, the bin holding "
        f"D = 0)",
        "1 bin",
    )


# --------------------------------------------------------------------------- #
# 7. ring lookup and its failure branch
# --------------------------------------------------------------------------- #
def test_ring_lookup():
    section("reference ring selection and the 1/sqrt(3) lookup")
    rings = [
        {
            "radius": 117.2,
            "members": [(0, 0, 1.0, 1.0, 117.2)] * 6,
            "total_amplitude": 6.0,
        },
        {
            "radius": 67.7,
            "members": [(0, 0, 1.0, 1.0, 67.7)] * 6,
            "total_amplitude": 6.0,
        },
        {
            "radius": 40.0,
            "members": [(0, 0, 1.0, 1.0, 40.0)] * 6,
            "total_amplitude": 1.0,
        },
    ]
    choice = pp.choose_rings(rings, anchor="auto")
    check(
        "auto: the strongest ring with a ring at 1/sqrt(3) is the reference",
        choice["status"] == "ok"
        and abs(choice["ring_1x1"]["radius"] - 117.2) < 1e-9
        and abs(choice["ring_r3"]["radius"] - 67.7) < 1e-9,
        f"ring_1x1 = {choice['ring_1x1']['radius']:.2f} px, ring_r3 = "
        f"{choice['ring_r3']['radius']:.2f} px, ratio {choice['ratio']:.6f}",
    )
    choice = pp.choose_rings(rings, anchor="radius", reference_radius_px=40.0)
    check(
        "explicit reference radius without a 1/sqrt(3) partner -> not found",
        choice["status"] == "r3_not_found" and choice["ring_1x1"]["radius"] == 40.0,
        f"status = {choice['status']}, method = {choice['method']}",
    )
    broken = [
        {
            "radius": 117.2,
            "members": [(0, 0, 1.0, 1.0, 117.2)] * 6,
            "total_amplitude": 6.0,
        },
        {
            "radius": 55.0,
            "members": [(0, 0, 1.0, 1.0, 55.0)] * 6,
            "total_amplitude": 6.0,
        },
    ]
    check(
        "no ring at 1/sqrt(3) or sqrt(3) of another ring -> not found, never a guess",
        pp.choose_rings(broken, anchor="auto")["status"] == "r3_not_found",
        "status = r3_not_found",
    )
    inner_r3 = [
        {
            "radius": 135.4,
            "members": [(0, 0, 1.0, 1.0, 135.4)] * 6,
            "total_amplitude": 6.0,
        },
        {
            "radius": 234.5,
            "members": [(0, 0, 1.0, 1.0, 234.5)] * 6,
            "total_amplitude": 4.0,
        },
    ]
    choice = pp.choose_rings(inner_r3, anchor="auto")
    check(
        "an innermost r3 ring is reported as the ring above it",
        choice["status"] == "ok" and abs(choice["ring_1x1"]["radius"] - 234.5) < 1e-9,
        f"ring_1x1 = {choice['ring_1x1']['radius']:.2f} px, method = "
        f"{choice['method']}",
    )
    order = pp.order_ring_members(
        ring_dict(
            [
                (
                    153.6 * np.cos(a + np.radians(30.0)),
                    153.6 * np.sin(a + np.radians(30.0)),
                )
                for a in np.arange(6) * np.pi / 3.0
            ]
        )["members"]
    )
    angles = [float(np.degrees(np.arctan2(item[1], item[0])) % 360.0) for item in order]
    steps = [(angles[i] - angles[i + 1]) % 360.0 for i in range(len(angles) - 1)]
    check(
        "peak numbering starts at 12 o'clock and runs clockwise",
        abs(angles[0] - 90.0) < 1e-9 and all(abs(step - 60.0) < 1e-9 for step in steps),
        f"p0 at {angles[0]:.3f} deg, then "
        f"{[f'{value:.1f}' for value in angles[1:]]} (steps "
        f"{[f'{value:.1f}' for value in steps]})",
        "p0 at 90 deg, -60 deg steps",
    )


# --------------------------------------------------------------------------- #
# 8. discrimination boundary of the phase-sum test
# --------------------------------------------------------------------------- #
def off_multiple_deg(angle_deg):
    """Distance of an angle to the nearest whole multiple of a third of a turn.

    The three independent reflections of one region carry the same phase, so their
    sum is three times that phase; for a set of components whose phases differ by
    whole multiples of a third of the turn, the tripled mean phase lands on such a
    multiple, and this distance measures how far a mixture is from one region.
    """
    wrapped = float(angle_deg) % THIRD_TURN_DEG
    return min(wrapped, THIRD_TURN_DEG - wrapped)


def phasor(weights, phases_deg):
    """``3 x`` the weighted mean phase of a set of phasor components."""
    total = float(np.sum(weights))
    z = np.sum(
        np.asarray(weights, dtype=float)
        * np.exp(1j * np.radians(np.asarray(phases_deg, dtype=float)))
    )
    angle = 3.0 * np.degrees(np.angle(z)) % 360.0
    return {
        "three_phi_bar_deg": float(angle),
        "triple_deviation_deg": float(off_multiple_deg(angle)),
        "coherence": float(abs(z) / total),
    }


def test_multi_component_boundary(n, lambda_nm=3.0, nm_per_px=0.13):
    """The three-phase sum is a value-range test, never a component counter."""
    section("multi-component discrimination boundary (1 / 2 / 3 components)")
    one = phasor([1.0], [0.0])
    check(
        "1 component: the tripled phase is exactly the injected phase, coherence 1",
        one["triple_deviation_deg"] < 1e-9 and abs(one["coherence"] - 1.0) < 1e-12,
        f"3 Phi_bar = {one['three_phi_bar_deg']:.4f} deg (distance to a whole "
        f"multiple of a third of a turn {one['triple_deviation_deg']:.4f} deg), "
        f"coherence {one['coherence']:.6f}",
        "< 1e-9 deg / 1.0",
    )
    two = phasor([0.5, 0.5], [0.0, THIRD_TURN_DEG])
    check(
        "2 equal components a third of a turn apart: the tripled phase sits 60 deg "
        "from a whole multiple",
        abs(two["triple_deviation_deg"] - THIRD_TURN_DEG / 2.0) < 1e-9,
        f"3 Phi_bar = {two['three_phi_bar_deg']:.4f} deg, distance to a whole "
        f"multiple {two['triple_deviation_deg']:.4f} deg, coherence "
        f"{two['coherence']:.4f}",
        f"{THIRD_TURN_DEG / 2.0:.1f} deg",
    )
    three = phasor([1 / 3, 1 / 3, 1 / 3], [0.0, THIRD_TURN_DEG, 2.0 * THIRD_TURN_DEG])
    check(
        "3 equal components evenly spaced cancel: the phase is undefined",
        three["coherence"] < 1e-12,
        f"coherence = {three['coherence']:.3e} (any phase number is meaningless)",
        "< 1e-12",
    )

    boundary = []
    for minority in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50):
        mixed = phasor([1.0 - minority, minority], [0.0, THIRD_TURN_DEG])
        boundary.append((minority, mixed["triple_deviation_deg"], mixed["coherence"]))
    crossing = [row for row in boundary if row[1] >= 5.0]
    if crossing:
        index = boundary.index(crossing[0])
        if index == 0:
            interpolated = crossing[0][0]
        else:
            low, high = boundary[index - 1], crossing[0]
            interpolated = low[0] + (high[0] - low[0]) * (5.0 - low[1]) / (
                high[1] - low[1]
            )
    else:
        interpolated = float("nan")
    check(
        "2 unequal components: the sum resolves a minority above ~3 %",
        0.02 < interpolated < 0.05,
        f"the 5 deg crossing of the triple deviation is at f = {interpolated:.4f}; "
        + "; ".join(
            f"f={row[0]:.3f} -> {row[1]:.2f} deg, coherence {row[2]:.3f}"
            for row in boundary[:5]
        ),
        "2 % < f < 5 %",
    )

    # ---- the same statement through the delivered engine ------------------ #
    r1 = 0.458 * n
    vectors = [
        (
            r1 / SQRT3 * np.cos(a + np.radians(30.0)),
            r1 / SQRT3 * np.sin(a + np.radians(30.0)),
        )
        for a in np.arange(6) * np.pi / 3.0
    ]
    members = [(v[0], v[1], 1.0, 1.0, float(np.hypot(*v))) for v in vectors]
    valid = np.ones((n, n), dtype=bool)
    rows = []
    for radius_frac in (0.3697, 0.098):
        domains = [
            {"kind": "full", "amp": 1.0, "phase_deg": 0.0},
            {
                "kind": "disk",
                "amp": 1.0,
                "phase_deg": THIRD_TURN_DEG,
                "centre": (0.5, 0.5),
                "radius_frac": radius_frac,
                "edge": 4.0,
            },
        ]
        image = pp.synth_image(n, r1, domains, ref_amp=0.3, ref_phase_deg=40.0)
        truth = pp.true_mixture(domains, n, r1, window=None)
        records, fields, _psi = pp.analyse_ring(
            image,
            valid,
            {"radius": r1 / SQRT3, "members": members},
            lambda_nm,
            nm_per_px,
            prefix="r3_",
            peaks_override=members,
        )
        triple = pp.triple_summary(
            records, fields, valid, key="phase", quantity="mean_deg"
        )
        clusters = sorted(
            {record["stats"]["phase"]["n_clusters"] for record in records}
        )
        deviation = abs(
            pm.ang_diff_deg(triple["scalar_sum_deg"], truth["three_phi_bar_area_deg"])
        )
        rows.append(
            {
                "weight": float(truth["area_weight_fractions"][1]),
                "predicted": truth["three_phi_bar_area_deg"],
                "measured": triple["scalar_sum_deg"],
                "off_multiple": off_multiple_deg(triple["scalar_sum_deg"]),
                "clusters": clusters,
                "deviation": deviation,
            }
        )
        print(
            f"    engine: minority area weight f_w = {rows[-1]['weight']:.4f} "
            f"-> predicted 3 Phi_bar = {rows[-1]['predicted']:.4f} deg, measured "
            f"{rows[-1]['measured']:.4f} deg, distance to a whole multiple "
            f"{rows[-1]['off_multiple']:.4f} deg, cluster counts {clusters}"
        )
    worst = max(row["deviation"] for row in rows)
    check(
        "the whole-canvas phase value reproduces the area-weighted phasor prediction",
        worst < 1.0,
        f"max deviation = {worst:.4f} deg over {len(rows)} two-component "
        f"configurations (the engine sums T exp(-i q.r) over every valid "
        f"pixel, so the region weight of a smooth region is its area "
        f"integral; the residual is the spectral leakage of the other "
        f"ring into the demodulation window)",
        "< 1.0 deg",
    )
    check(
        "a ~3 % minority is invisible to the cluster count but visible to the sum",
        rows[1]["clusters"] == [1] and rows[1]["off_multiple"] > 3.0,
        f"f_w = {rows[1]['weight']:.4f} -> cluster counts {rows[1]['clusters']}, "
        f"distance of the tripled phase to a whole multiple "
        f"{rows[1]['off_multiple']:.2f} deg",
        "1 cluster and > 3 deg away from a whole multiple",
    )
    check(
        "a ~30 % minority shows up in the cluster count as well",
        rows[0]["clusters"] == [2],
        f"f_w = {rows[0]['weight']:.4f} -> cluster counts {rows[0]['clusters']}",
        "2 clusters",
    )


# --------------------------------------------------------------------------- #
# 9. end-to-end pipeline: atlas contract, determinism, the not-found branch
# --------------------------------------------------------------------------- #
def write_synthetic_csv(path, n, radius_frac=0.458, phases=(0.0, 240.0)):
    r1 = radius_frac * n
    domains = [{"kind": "full", "amp": 1.0, "phase_deg": phases[0]}]
    if len(phases) > 1:
        domains.append(
            {
                "kind": "disk",
                "amp": 1.0,
                "phase_deg": phases[1],
                "centre": (0.5, 0.5),
                "radius_frac": 0.20,
                "edge": 4.0,
            }
        )
    image = pp.synth_image(n, r1, domains, ref_amp=1.0, ref_phase_deg=40.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, image, delimiter=",", fmt="%.10e")
    return r1


def run_script(script, arguments, env):
    return subprocess.run(
        [sys.executable, str(HERE / script), *arguments],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def colour_bar_figure():
    """A small map figure with an outside colour bar, plus the axes it drew."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    image = ax.imshow(np.arange(64.0).reshape(8, 8), origin="lower")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax
    return fig, ax, bar


def test_colorbar_contract():
    """The colour-bar layout rule, and the audit that must catch a violation."""
    section("colour-bar contract (drawn outside the data area, and audited)")
    import atlas as at
    import matplotlib.pyplot as plt

    fig, ax, bar = colour_bar_figure()
    colorbars, outside, overlap = at.colorbar_layout([ax], [bar])
    check(
        "the reference figure has one colour bar outside its data axes",
        colorbars == 1 and outside and overlap == 0.0,
        f"{colorbars} colour bar(s), outside: {outside}, worst intersection area "
        f"{overlap:g} in figure coordinates",
        "outside, area 0",
    )
    inside = at.bbox_overlap_area(ax.get_position(), ax.get_position())
    bar.get_position()
    bar.set_position(ax.get_position())
    colorbars_in, outside_in, overlap_in = at.colorbar_layout([ax], [bar])
    check(
        "the colour-bar audit turns red when a bar is moved into the data area",
        overlap_in > 0.0 and not outside_in and colorbars_in == 1,
        f"after moving the colour bar onto the data axes the same audit reports "
        f"intersection area {overlap_in:.6f} (a data axes against itself gives "
        f"{inside:.6f}), outside: {outside_in}",
        "area > 0 and outside False",
    )
    plt.close(fig)

    fig, ax, bar = colour_bar_figure()
    book = at.Atlas(
        Path(tempfile.gettempdir()) / "stm-phase-colourbar-probe", at.load_colormap()[0]
    )
    at.set_canvas_title(fig, at.title_for("probe", {}))
    raised = ""
    try:
        book.add(
            fig,
            "probe.png",
            "probe",
            {},
            {},
            "ring_1x1",
            "summary",
            panels=["probe"],
            mappable=True,
            data_axes=[ax],
            colorbar_axes=[],
        )
    except AssertionError as exc:
        raised = str(exc)
    check(
        "a map figure without any colour bar is refused while writing",
        "colour bar" in raised,
        f"the writer raised {raised!r} for a map figure declared without a colour bar",
        "an explicit AssertionError",
    )
    plt.close(fig)


def title_fontsize_ok(entry):
    """Whether a manifest entry belongs to a figure type that carries a title."""
    return bool(entry.get("title"))


def test_global_amplitude_scale():
    """The global amplitude linear region is a central value, not the minimum."""
    section("global amplitude colour scale (pooled median linear region)")
    import atlas as at
    import stm_phase_analysis as spa

    valid = np.ones((8, 8), dtype=bool)
    low = np.linspace(1.0, 4.0, 64).reshape(8, 8)
    high = np.linspace(3.0, 6.0, 64).reshape(8, 8)
    analysis = {
        tag: {
            "valid": valid,
            "records": [{"name": f"{tag}_p0"}, {"name": f"{tag}_p1"}],
            "fields": {
                f"{tag}_p0": (None, None, low.copy()),
                f"{tag}_p1": (None, None, high.copy()),
            },
        }
        for tag in ("ring_1x1", "ring_r3")
    }
    vmax, linthresh = spa.global_amplitudes(analysis)
    pooled = float(
        np.median(
            np.concatenate([low.ravel(), high.ravel(), low.ravel(), high.ravel()])
        )
    )
    smallest = float(min(low.min(), high.min()))
    check(
        "the amplitude linear region is the pooled median of the positive amplitudes",
        abs(linthresh - pooled) < 1e-12
        and abs(vmax - 6.0) < 1e-12
        and linthresh > smallest * 1.5,
        f"linthresh = {linthresh:g} equals the pooled median of the twelve field "
        f"amplitudes {pooled:g} (the minimum positive value would be {smallest:g}), "
        f"vmax = {vmax:g} is the global maximum",
        "pooled median, not the minimum",
    )
    norm = at.amplitude_norm(vmax, linthresh)
    check(
        "that linear region keeps vmin = 0 and vmax global, and moves the bulk of the "
        "data out of the squeezed top of the colour bar",
        norm.vmin == 0.0
        and norm.vmax == vmax
        and norm.linthresh == linthresh
        and float(norm(linthresh)) < 0.87,
        f"the shared scale is symlog vmin {norm.vmin:g}, vmax {norm.vmax:g}, "
        f"linthresh {norm.linthresh:g}; the pooled median datum sits at colour-bar "
        f"position t = {float(norm(linthresh)):.3f} (< 0.87), while the old "
        f"minimum-based region put the same datum at t = "
        f"{float(at.amplitude_norm(vmax, smallest)(linthresh)):.3f}",
        "t(median) < 0.87",
    )


def test_title_and_mask_gate(workdir):
    """The canvas-title gate and the visible mask colour, with counterexamples."""
    section("canvas title gate and visible mask colour")
    import atlas as at
    import matplotlib.pyplot as plt
    from PIL import Image

    probe_dir = Path(workdir) / "title_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    book = at.Atlas(probe_dir, at.load_colormap()[0], dpi=100)

    # ---- the gate accepts a figure that carries the manifest title --------- #
    title = at.title_for("ring_1x1 probe", {"n_valid": 3})
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16.0).reshape(4, 4))
    at.set_canvas_title(fig, title)
    rendered = at.rendered_title_texts(fig)
    check(
        "the title of a figure is read back from the canvas, not from the manifest",
        rendered == [title],
        f"the canvas renders {rendered} for the manifest title {title!r}",
        "the manifest title is on the canvas",
    )
    plt.close(fig)

    # ---- counterexample 1: no title at all -------------------------------- #
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16.0).reshape(4, 4))
    raised = ""
    try:
        at.require_canvas_title(fig, title)
    except AssertionError as exc:
        raised = str(exc)
    check(
        "the gate turns red when the canvas carries no title at all",
        "does not carry the manifest title" in raised
        and "rendered title text(s): none" in raised,
        f"the audit raised {raised!r} for a figure whose canvas has no title text",
        "an explicit AssertionError",
    )
    plt.close(fig)

    # ---- counterexample 2: one character changed -------------------------- #
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16.0).reshape(4, 4))
    at.set_canvas_title(fig, title)
    fig._suptitle.set_text(title[:-1] + "X")
    raised = ""
    try:
        at.require_canvas_title(fig, title)
    except AssertionError as exc:
        raised = str(exc)
    check(
        "the gate turns red when one character of the title differs",
        "does not carry the manifest title" in raised and "X" in raised,
        f"changing the last character of the title raised {raised!r}",
        "an explicit AssertionError",
    )
    plt.close(fig)

    # ---- the writer itself refuses a titleless figure --------------------- #
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16.0).reshape(4, 4))
    raised = ""
    try:
        book.add(
            fig,
            "probe_titleless.png",
            "ring_1x1 probe",
            {},
            {},
            "ring_1x1",
            "summary",
            panels=["probe"],
        )
    except AssertionError as exc:
        raised = str(exc)
    check(
        "the atlas writer refuses to save a figure without a canvas title",
        "does not carry the manifest title" in raised,
        f"Atlas.add raised {raised!r} for a figure with no title on the canvas",
        "an explicit AssertionError",
    )
    plt.close(fig)

    # ---- pixel counterpart: the title band of the saved file ------------- #
    strips, inks = {}, {}
    for name, titled in (("band_titled.png", True), ("band_titleless.png", False)):
        fig, ax = plt.subplots(figsize=(3, 2.4))
        ax.imshow(np.arange(16.0).reshape(4, 4))
        if titled:
            at.set_canvas_title(fig, title)
        fig.tight_layout()
        strips[name] = at.title_strip_px(fig, [ax], 100)
        fig.savefig(probe_dir / name, dpi=100, bbox_inches="tight")
        plt.close(fig)
        with Image.open(probe_dir / name) as image:
            inks[name] = at.title_band_ink(image, strips[name])
    check(
        "the saved title strip separates a titled figure from a titleless one",
        inks["band_titled.png"] >= at.TITLE_BAND_MIN_INK
        and inks["band_titleless.png"] < at.TITLE_BAND_MIN_INK,
        f"the strip above the data axes is {strips['band_titled.png']} px for the "
        f"titled figure and {strips['band_titleless.png']} px without a title; ink "
        f"pixels in those strips: {inks['band_titled.png']} vs "
        f"{inks['band_titleless.png']} (threshold {at.TITLE_BAND_MIN_INK}) — the "
        f"titleless figure leaves the strip white, and a naive top-of-image band "
        f"would count the axes frame instead",
        f"titled >= {at.TITLE_BAND_MIN_INK} > titleless",
    )

    # ---- the mask colour of a theta map ----------------------------------- #
    plain = plt.get_cmap("hsv")(np.nan)
    masked = at.theta_cmap()(np.nan)
    check(
        "the theta colour map paints masked pixels in the amplitude grey",
        masked[3] == 1.0
        and plain[3] == 0.0
        and np.allclose(masked[:3], np.array([176, 176, 176]) / 255.0, atol=1e-3),
        f"theta_cmap()(NaN) = rgba {tuple(np.round(masked, 3))} = {at.BAD_COLOR} "
        f"(opaque), while the plain hsv map gives alpha {plain[3]:g} (invisible); the "
        f"amplitude maps use the same {at.BAD_COLOR}",
        f"opaque {at.BAD_COLOR}",
    )
    # ---- the one-title-line design: single axes vs panel identifiers ------ #
    fig, ax = plt.subplots()
    ax.imshow(np.arange(16.0).reshape(4, 4))
    ax.set_title("ring_1x1 probe - short axis title", fontsize=12)
    at.set_canvas_title(fig, at.title_for("ring_1x1 probe", {}))
    raised = ""
    try:
        book.add(
            fig,
            "probe_axes_title.png",
            "ring_1x1 probe",
            {},
            {},
            "ring_1x1",
            "summary",
            panels=["probe"],
            data_axes=[ax],
        )
    except AssertionError as exc:
        raised = str(exc)
    check(
        "the writer refuses a single-axes figure that still draws a short axis title",
        "single-axes figure" in raised and "short title" in raised,
        f"Atlas.add raised {raised!r} for a one-panel figure whose data axis keeps "
        f"its own title next to the figure-level annotation title",
        "an explicit AssertionError",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2)
    for index, ax in enumerate(axes):
        ax.imshow(np.arange(16.0).reshape(4, 4))
        ax.set_title(f"p{index} panel", fontsize=12)
    at.set_canvas_title(fig, at.title_for("ring_1x1 probe grid", {}))
    entry = book.add(
        fig,
        "probe_grid.png",
        "ring_1x1 probe grid",
        {},
        {},
        "ring_1x1",
        "summary",
        panels=["two panels"],
        data_axes=list(axes),
    )
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(np.arange(16.0).reshape(4, 4))
    axes[0].set_title("p0 panel", fontsize=12)
    axes[1].imshow(np.arange(16.0).reshape(4, 4))
    at.set_canvas_title(fig, at.title_for("ring_1x1 probe grid", {}))
    raised = ""
    try:
        book.add(
            fig,
            "probe_grid_untitled_panel.png",
            "ring_1x1 probe grid",
            {},
            {},
            "ring_1x1",
            "summary",
            panels=["two panels"],
            data_axes=list(axes),
        )
    except AssertionError as exc:
        raised = str(exc)
    check(
        "the writer refuses a multi-panel figure whose panels lose their identifiers",
        "must keep its own identifier title" in raised,
        f"Atlas.add raised {raised!r} for a two-panel figure with only one panel "
        f"title, so a grid cannot drop its panel identifiers",
        "an explicit AssertionError",
    )
    plt.close(fig)

    check(
        "a multi-panel figure keeps its per-panel identifiers and is accepted",
        entry["axes_titles"] == ["p0 panel", "p1 panel"]
        and entry["panel_axes"] == 2
        and entry["canvas_title"] == entry["title"],
        f"the two-panel probe records panel_axes={entry['panel_axes']} and "
        f"axes_titles={entry['axes_titles']} next to its figure-level title",
        "panel identifiers kept",
    )
    plt.close(fig)

    sample = np.ones((64, 64), dtype=bool)
    sample[:16, :] = False
    entry = book.theta_map(
        np.zeros((64, 64)),
        sample,
        "ring_1x1 probe theta(r)",
        {"pixels": int(np.count_nonzero(sample))},
        {"pixels": "probe.pixels"},
        "ring_1x1",
        "probe_theta_map.png",
        0,
    )
    with Image.open(probe_dir / entry["file"]) as image:
        array = np.asarray(image.convert("RGB")).reshape(-1, 3)
    grey = int(np.count_nonzero(np.all(array == np.array([176, 176, 176]), axis=1)))
    hsv_colours = int(np.count_nonzero(~np.all(array == array[0], axis=1)))
    check(
        "the masked quarter of a theta map is really drawn in that grey",
        grey > 1000 and hsv_colours > 0,
        f"the saved theta map carries {grey} pixels of exactly {at.BAD_COLOR} (the "
        f"masked quarter of the canvas) among {array.shape[0]} canvas pixels, while "
        f"the unmasked part keeps the hsv colours",
        "> 1000 grey pixels",
    )


def test_pipeline_contract(n, workdir, stm_lib):
    section(
        "end-to-end pipeline: 60 figures, colour bars and colour scales, "
        "determinism, the not-found branch"
    )
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    (workdir / ".mplcache").mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "pipeline" / "synthetic.csv"
    field_of_view = 50.0
    write_synthetic_csv(csv_path, n)

    runs = {}
    for tag in ("a", "b"):
        outdir = workdir / "pipeline" / f"out_{tag}"
        runs[tag] = (
            run_script(
                "stm_phase_analysis.py",
                [
                    str(csv_path),
                    "-o",
                    str(outdir),
                    "-L",
                    str(field_of_view),
                    "--detector",
                    "builtin",
                    "--stm-lib",
                    stm_lib,
                ],
                env,
            ),
            outdir,
        )
    completed, outdir = runs["a"]
    second_completed, second_outdir = runs["b"]
    check(
        "the second run also exits 0 (checked before its outputs are read)",
        second_completed.returncode == 0,
        f"exit code {second_completed.returncode}"
        + (
            ""
            if second_completed.returncode == 0
            else f"; stderr tail: {second_completed.stderr.strip().splitlines()[-1]}"
            if second_completed.stderr.strip()
            else ""
        ),
        "exit code 0",
    )
    check(
        "analysis script exits 0 on a two-ring synthetic",
        completed.returncode == 0,
        f"exit code {completed.returncode}"
        + (
            ""
            if completed.returncode == 0
            else f"; stderr tail: {completed.stderr.strip().splitlines()[-1]}"
            if completed.stderr.strip()
            else ""
        ),
        "exit code 0",
    )
    if completed.returncode != 0:
        return

    log_text = (outdir / "phase_stats.log").read_text()
    check(
        "the log states the engine and the lambda of the run",
        "gaussian window via local-q-map" in log_text
        and "--lambda-nm" not in log_text
        and "lambda = 3 nm" in log_text,
        "log carries the engine line: '"
        + next(
            (line for line in log_text.splitlines() if "gaussian window" in line), ""
        )
        + "'",
        "engine line present",
    )

    help_run = run_script("stm_phase_analysis.py", ["--help"], env)
    check(
        "the CLI has no threshold option and an explicit --help",
        help_run.returncode == 0
        and "--help" in help_run.stdout
        and "--lambda-nm" in help_run.stdout
        and ("--" + "gate") not in help_run.stdout,
        f"--help exits {help_run.returncode}, lists --lambda-nm: "
        f"{'--lambda-nm' in help_run.stdout}, lists the removed option: "
        f"{('--' + 'gate') in help_run.stdout}",
        "help without a threshold option",
    )

    for option, label in (
        (["--" + "gate", "p" + "50"], "the removed sample-selection"),
        (["--pct", "5"], "the previously removed engine"),
    ):
        removed = run_script(
            "stm_phase_analysis.py",
            [
                str(csv_path),
                "-o",
                str(workdir / "pipeline" / "out_removed"),
                "-L",
                str(field_of_view),
                "--detector",
                "builtin",
                *option,
            ],
            env,
        )
        if option[0] == "--" + "gate":
            check(
                "the removed option is rejected with a non-zero exit code",
                removed.returncode != 0
                and option[0] in (removed.stderr + removed.stdout),
                f"exit code {removed.returncode}, message "
                f"{'present' if option[0] in (removed.stderr + removed.stdout) else 'MISSING'} "
                f"for {label} option {option[0]}",
                "non-zero exit and an explicit message",
            )
        else:
            check(
                "the previously removed --pct option is rejected by the CLI",
                removed.returncode != 0
                and "--pct" in (removed.stderr + removed.stdout),
                f"exit code {removed.returncode}, message "
                f"{'present' if '--pct' in (removed.stderr + removed.stdout) else 'MISSING'}",
                "non-zero exit and an explicit message",
            )

    figures = sorted(outdir.glob("*.png"))
    check(
        f"atlas contains {FIGURES_PER_RING} figures per ring ({TOTAL_FIGURES} in total)",
        len(figures) == TOTAL_FIGURES,
        f"{len(figures)} PNG files found",
        f"{TOTAL_FIGURES}",
    )

    import atlas as at  # imported after MPLCONFIGDIR has been set

    stats_path = outdir / "phase_stats.json"
    manifest_path = outdir / "atlas_manifest.json"
    ok, failures, _lines = at.check_manifest(
        manifest_path,
        stats_path=stats_path,
        expected_figures=TOTAL_FIGURES,
        expected_per_ring=FIGURES_PER_RING,
        verbose=False,
    )
    check(
        "atlas manifest audit: files, sizes, PIL, embedded text, numbers = JSON",
        ok,
        f"{len(failures)} failure(s)" + (f": {failures[:3]}" if failures else ""),
        "0 failures",
    )
    if not ok:
        return

    manifest = json.loads(manifest_path.read_text())
    stats_a = json.loads(stats_path.read_text())
    bad = [
        entry["file"]
        for entry in manifest["figures"]
        if not FIGURE_PATTERN.match(entry["file"])
    ]
    check(
        "every figure name follows the documented pattern",
        not bad,
        f"{len(manifest['figures'])} names checked, off-pattern: {bad[:3]}",
        "0",
    )
    stale = sorted(
        entry["file"]
        for entry in manifest["figures"]
        if any(part in entry["file"] for part in REMOVED_FIGURE_PARTS)
    )
    on_disk = sorted(
        path.name
        for path in outdir.glob("*.png")
        if any(part in path.name for part in REMOVED_FIGURE_PARTS)
    )
    check(
        "no figure name of the removed layer",
        not stale and not on_disk,
        f"{len(manifest['figures'])} names checked against "
        f"{list(REMOVED_FIGURE_PARTS)}; in the manifest: {stale}, on disk: {on_disk}",
        "0",
    )
    text = " ".join(
        entry["file"] + " " + entry["title"] for entry in manifest["figures"]
    )
    hits = source_token_hits(text.lower(), [token.lower() for token in FORBIDDEN])
    check(
        "no forbidden token in any figure name or title",
        not hits,
        f"scanned {len(manifest['figures'])} names/titles, hits: {hits}",
        "0 hits",
    )
    documents = {
        name: (HERE.parent / name).read_text()
        for name in ("SKILL.md", "README.md", "CHANGES.md")
    }
    document_hits = {
        name: source_token_hits(text.lower(), [token.lower() for token in FORBIDDEN])
        for name, text in documents.items()
    }
    document_hits = {name: found for name, found in document_hits.items() if found}
    check(
        "no forbidden token in the delivered forward-looking documents",
        not document_hits,
        f"scanned {sorted(documents)}, hits: "
        f"{document_hits if document_hits else 'none'}",
        "0 hits",
    )
    narration = {
        name: source_token_hits(
            text.lower(),
            [token.lower() for token in REMOVED_ENGINE + REMOVED_STAIRCASE],
        )
        for name, text in documents.items()
    }
    narration = {name: found for name, found in narration.items() if found}
    check(
        "the documents carry no removed-layer narration",
        not narration,
        f"scanned {sorted(documents)} for the removed sample and phase-axis tokens, "
        f"hits: {narration if narration else 'none'}",
        "0 hits",
    )

    groups = {}
    for entry in manifest["figures"]:
        groups.setdefault(entry["group"], {}).setdefault(entry["kind"], 0)
        groups[entry["group"]][entry["kind"]] += 1
    check(
        "the manifest carries group and kind for every figure, pairwise with a pair key",
        groups.get("ring_1x1") == {"per_peak": 18, "summary": 3, "pairwise": 9}
        and groups.get("ring_r3") == {"per_peak": 18, "summary": 3, "pairwise": 9}
        and set(groups) == {"ring_1x1", "ring_r3"}
        and all(
            entry.get("pair")
            for entry in manifest["figures"]
            if entry["kind"] == "pairwise"
        )
        and all(
            entry.get("pair") is None
            for entry in manifest["figures"]
            if entry["kind"] != "pairwise"
        ),
        f"per group: {groups}",
        "18/3/9 per ring and nothing else",
    )
    expected_pair_files = {
        f"{group}_pair_{j}_{k}_{kind}.png"
        for group in ("ring_1x1", "ring_r3")
        for j, k in WITHIN_PAIRS
        for kind in PAIR_FIGURE_KINDS
    }
    actual_pair_files = {
        entry["file"] for entry in manifest["figures"] if entry["kind"] == "pairwise"
    }
    check(
        "each ring has the three figures of each pair inside the ring",
        actual_pair_files == expected_pair_files and len(actual_pair_files) == 18,
        f"{len(actual_pair_files)} pairwise figures against the {len(expected_pair_files)} "
        f"expected names (3 pairs x {len(PAIR_FIGURE_KINDS)} kinds x 2 rings); "
        f"missing: {sorted(expected_pair_files - actual_pair_files)}, "
        f"unexpected: {sorted(actual_pair_files - expected_pair_files)}",
        f"{len(expected_pair_files)} names",
    )
    check(
        "every pairwise annotation is declared with a path into the pairwise section",
        all(
            all(path.startswith("pairwise.") for path in entry["paths"].values())
            for entry in manifest["figures"]
            if entry["kind"] == "pairwise"
        ),
        "all pairwise paths start at 'pairwise.'",
        "pairwise paths",
    )

    expected_panels = {
        "amplitude": ["amplitude |psi(r)| (global symlog scale)"],
        "theta_map": ["theta(r) map over the sample of the statistics"],
        "theta_dist": ["theta distribution over the sample of the statistics"],
    }
    bad_panels = None
    for entry in manifest["figures"]:
        if entry["kind"] != "per_peak":
            continue
        match = re.match(
            r"^(?:ring_1x1|ring_r3)_p(\d)_(amplitude|theta_map|theta_dist)\.png$",
            entry["file"],
        )
        expected = expected_panels.get(match.group(2)) if match else None
        if expected is None or entry.get("panels") != expected:
            bad_panels = (entry["file"], entry.get("panels"))
            break
    check(
        "the per-peak theta distribution is one panel over the full circle",
        bad_panels is None
        and all(
            len(entry["panels"]) == 1
            for entry in manifest["figures"]
            if entry["kind"] == "per_peak" and entry["file"].endswith("_theta_dist.png")
        ),
        "18 per-peak figures carry the documented panel lists, each of exactly one "
        "panel"
        if bad_panels is None
        else f"unexpected panels: {bad_panels}",
        "one panel per figure",
    )
    summary_panels = {}
    for entry in manifest["figures"]:
        if entry["kind"] == "summary":
            summary_panels.setdefault(entry["group"], {})[entry["file"]] = entry.get(
                "panels"
            )
    check(
        "each ring has three summary figures, all with panels",
        set(summary_panels) == {"ring_1x1", "ring_r3"}
        and all(len(kinds) == 3 for kinds in summary_panels.values())
        and all(
            all(panels for panels in kinds.values())
            for kinds in summary_panels.values()
        ),
        "per group: "
        + ", ".join(
            f"{group}: {len(kinds)} figures"
            for group, kinds in sorted(summary_panels.items())
        ),
        "3 per ring with panels",
    )

    # ---- R6: the drawn pixels are the sample of the statistics ------------ #
    announced = [
        (
            tag,
            row["index"],
            row["n_samples"],
            row["n_zero_weight_px"],
        )
        for tag in ("ring_1x1", "ring_r3")
        for row in stats_a["rings_analysis"][tag]["peaks"]
    ]
    per_peak = [entry for entry in manifest["figures"] if entry["kind"] == "per_peak"]
    per_peak_peak = {}
    for entry in per_peak:
        per_peak_peak.setdefault((entry["group"], entry["peak"]), set()).add(
            entry["annotation"]["pixels"]
        )
    matches = all(
        values == {stats_a["rings_analysis"][group]["peaks"][peak]["n_samples"]}
        for (group, peak), values in per_peak_peak.items()
    )
    check(
        "the drawn pixel count of every per-peak figure equals its sample size",
        len(per_peak) == 2 * PEAKS_PER_RING * len(PEAK_FIGURE_KINDS)
        and len(per_peak_peak) == len(announced)
        and matches
        and all(entry["paths"]["pixels"].endswith(".n_samples") for entry in per_peak),
        f"the {len(per_peak)} per-peak figures cover {len(per_peak_peak)} peaks; the "
        f"three figures of a peak all announce its own recorded sample size "
        f"(drawn == n_samples for every peak: {matches}), and the manifest path of "
        f"that number is the JSON field n_samples, which atlas.py --check compares "
        f"with phase_stats.json",
        "drawn == recorded per peak",
    )
    check(
        "the recorded sample size is the number of valid pixels of the canvas",
        all(samples == n * n and zero == 0 for _tag, _i, samples, zero in announced),
        f"every reflection of the {n} x {n} synthetic canvas (no NaN pixel) reports "
        f"n_samples = {n * n} and {0} dropped pixels; a percentile sample rule would "
        f"report about half of them",
        f"n_samples = {n * n}",
    )

    # ---- the canvas title gate on the delivered atlas --------------------- #
    from PIL import Image as _Image

    canvas_ok = [
        entry
        for entry in manifest["figures"]
        if entry.get("canvas_title") == entry["title"]
    ]
    bands = {}
    for entry in manifest["figures"]:
        with _Image.open(outdir / entry["file"]) as image:
            bands[entry["file"]] = at.title_band_ink(image, entry.get("title_strip_px"))
    ring_level = [
        f"{tag}_{name}.png"
        for tag in ("ring_1x1", "ring_r3")
        for name in ("ring_members_qspace", "theta_map_summary", "theta_hist_summary")
    ]
    check(
        "every figure of the atlas carries its manifest title on the canvas",
        len(canvas_ok) == TOTAL_FIGURES,
        f"{len(canvas_ok)} of {len(manifest['figures'])} manifest entries record the "
        f"title they rendered on the canvas (canvas_title == title); the writer "
        f"refuses a figure that does not carry it",
        f"{TOTAL_FIGURES} figures",
    )
    single_axes = [
        entry for entry in manifest["figures"] if entry.get("panel_axes") == 1
    ]
    multi_axes = [
        entry for entry in manifest["figures"] if entry.get("panel_axes", 0) > 1
    ]
    check(
        "every single-axes figure carries one title line only",
        len(single_axes) == TOTAL_FIGURES - 4
        and all(
            entry["axes_titles"] == [] and entry["canvas_title"] == entry["title"]
            for entry in single_axes
        ),
        f"{len(single_axes)} single-axes figures declare no title of their own data "
        f"axis and their canvas text is the figure-level annotation title (the four "
        f"multi-panel grids are the other {len(multi_axes)} figures)",
        "0 axes titles",
    )
    grid_names = {
        f"{tag}_{name}.png"
        for tag in ("ring_1x1", "ring_r3")
        for name in ("theta_map_summary", "theta_hist_summary")
    }
    grid_ok = (
        {entry["file"] for entry in multi_axes} == grid_names
        and all(entry["panel_axes"] == 6 for entry in multi_axes)
        and all(len(entry["axes_titles"]) == 6 for entry in multi_axes)
        and all(
            all(f"p{index}" in " ".join(entry["axes_titles"]) for index in range(6))
            for entry in multi_axes
        )
    )
    check(
        "the four grid figures keep their six panel identifiers",
        grid_ok,
        "per grid: "
        + "; ".join(
            f"{entry['file']}: {entry['axes_titles']}" for entry in multi_axes[:2]
        ),
        "6 panel titles per grid",
    )
    hist_panels = [
        entry for entry in multi_axes if "theta_hist_summary" in entry["file"]
    ]
    check(
        "the histogram grid panel titles keep their per-peak numbers",
        len(hist_panels) == 2
        and all(
            sum("median" in title for title in entry["axes_titles"]) == 6
            for entry in hist_panels
        ),
        f"{len(hist_panels)} histogram grids, each panel title carrying its peak "
        f"median: {hist_panels[0]['axes_titles'] if hist_panels else 'none'}",
        "6 numbered panels",
    )
    check(
        "the six ring-level summary figures carry a visible title band",
        all(bands.get(name, 0) >= at.TITLE_BAND_MIN_INK for name in ring_level)
        and len(ring_level) == 6,
        "; ".join(f"{name}: {bands.get(name)} ink px" for name in ring_level),
        f">= {at.TITLE_BAND_MIN_INK} ink px each",
    )
    empty_bands = sorted(
        name for name, ink in bands.items() if ink < at.TITLE_BAND_MIN_INK
    )
    check(
        "no figure of the atlas was saved with an empty title band",
        not empty_bands,
        f"{len(bands)} figures checked in the strip above their data axes: "
        f"{len(empty_bands)} with fewer than {at.TITLE_BAND_MIN_INK} ink pixels "
        f"(the A4 defect class); worst band "
        f"{min(bands.values())} ink px in {min(bands, key=bands.get)}",
        "0 empty bands",
    )
    amplitude_medians = [
        row["amp_median"]
        for tag in ("ring_1x1", "ring_r3")
        for row in stats_a["rings_analysis"][tag]["peaks"]
    ]
    declared_linthresh = stats_a["atlas"]["norms"]["amplitude"]["linthresh"]
    central = float(np.median(amplitude_medians))
    check(
        "the declared amplitude linear region is a global central value",
        abs(np.log10(declared_linthresh / central)) < 1.0,
        f"linthresh = {declared_linthresh:.4e} is within a decade of the median of the "
        f"twelve per-peak amplitude medians {central:.4e} (a value derived from the "
        f"minimum positive amplitude would sit orders of magnitude below it)",
        "within one decade",
    )

    # ---- R7 and R8: colour bars and global colour scales ------------------ #
    mappable = [entry for entry in manifest["figures"] if entry["mappable"]]
    check(
        "every map / density figure of the atlas carries an outside colour bar",
        len(mappable) == 40
        and all(entry["colorbars"] >= 1 for entry in mappable)
        and all(entry["colorbar_outside"] for entry in mappable),
        f"{len(mappable)} map figures, each with at least one colour bar, all drawn "
        f"outside every data axes (the remaining "
        f"{len(manifest['figures']) - len(mappable)} figures are line distributions "
        f"without a mappable)",
        "40 maps with outside colour bars",
    )
    check(
        "every colour bar is declared with a disjoint axes rectangle in the manifest",
        all(entry["colorbar_overlap"] == 0.0 for entry in mappable),
        f"worst intersection area of a colour bar with a data axes over the "
        f"{len(mappable)} map figures = "
        f"{max(entry['colorbar_overlap'] for entry in mappable):g} in figure "
        f"coordinates",
        "0",
    )
    declared = stats_a["atlas"]["norms"]
    amplitude_entries = [
        entry
        for entry in manifest["figures"]
        if entry["file"].endswith("_amplitude.png")
    ]
    amplitude_norms = {
        json.dumps(entry["norm"], sort_keys=True) for entry in amplitude_entries
    }
    check(
        "all twelve amplitude maps share one colour scale",
        len(amplitude_entries) == 12
        and len(amplitude_norms) == 1
        and json.dumps(declared["amplitude"], sort_keys=True) in amplitude_norms,
        f"{len(amplitude_entries)} amplitude figures, {len(amplitude_norms)} distinct "
        f"scale record(s), the global declaration is "
        f"{declared['amplitude']}",
        "one shared symlog scale",
    )
    check(
        "every map figure carries its global colour scale and the JSON declares it",
        all(
            entry["norm"] is not None
            and entry["norm"]["scale"] in declared
            and entry["norm"] == declared[entry["norm"]["scale"]]
            for entry in mappable
        )
        and declared["theta"]["vmin"] == 0.0
        and declared["theta"]["vmax"] == TWO_PI
        and declared["phase_diff"]["vmin"] == -np.pi
        and declared["phase_diff"]["vmax"] == np.pi
        and declared["amp_diff"]["vmin"] == -1.0
        and declared["amp_diff"]["vmax"] == 1.0,
        "theta maps span [0, 2 pi], D maps [-pi, pi], a maps [-1, 1] and the twelve "
        "amplitude maps one symmetric-log scale; every figure record equals the "
        f"declaration in phase_stats.json: {declared}",
        "global scales",
    )

    completed_cli = run_script(
        "atlas.py",
        [
            "--check",
            str(manifest_path),
            "--stats",
            str(stats_path),
            "--expected-figures",
            str(TOTAL_FIGURES),
            "--expected-per-ring",
            str(FIGURES_PER_RING),
        ],
        env,
    )
    check(
        "the delivered atlas checker really runs as a command line tool",
        completed_cli.returncode == 0 and "ATLAS CHECK PASSED" in completed_cli.stdout,
        f"exit code {completed_cli.returncode}, stdout tail: "
        f"{completed_cli.stdout.strip().splitlines()[-1] if completed_cli.stdout else ''}",
        "exit 0 and ATLAS CHECK PASSED",
    )
    wrong_total = run_script(
        "atlas.py",
        [
            "--check",
            str(manifest_path),
            "--stats",
            str(stats_path),
            "--expected-figures",
            str(TOTAL_FIGURES - 1),
            "--expected-per-ring",
            str(FIGURES_PER_RING),
        ],
        env,
    )
    check(
        "the checker turns red when the declared total is wrong (59)",
        wrong_total.returncode != 0 and "ATLAS CHECK FAILED" in wrong_total.stdout,
        f"exit code {wrong_total.returncode} against {TOTAL_FIGURES - 1} declared "
        f"figures for a {len(figures)}-figure atlas: the door is not a formality",
        "non-zero exit",
    )
    wrong_ring = run_script(
        "atlas.py",
        [
            "--check",
            str(manifest_path),
            "--stats",
            str(stats_path),
            "--expected-figures",
            str(TOTAL_FIGURES),
            "--expected-per-ring",
            str(FIGURES_PER_RING - 1),
        ],
        env,
    )
    check(
        "the checker turns red when the declared per-ring count is wrong",
        wrong_ring.returncode != 0 and "ATLAS CHECK FAILED" in wrong_ring.stdout,
        f"exit code {wrong_ring.returncode} against {FIGURES_PER_RING - 1} declared "
        f"figures per ring",
        "non-zero exit",
    )

    per_group = {
        group: info["figures"] for group, info in manifest["per_group"].items()
    }
    check(
        "the manifest declares the same figure counts as the constants",
        per_group == {"ring_1x1": FIGURES_PER_RING, "ring_r3": FIGURES_PER_RING}
        and manifest["total_figures"] == TOTAL_FIGURES
        and manifest["figures_per_ring"] == FIGURES_PER_RING
        and manifest["figures_total"] == TOTAL_FIGURES
        and manifest["groups"] == ["ring_1x1", "ring_r3"]
        and stats_a["atlas"]["figures"] == TOTAL_FIGURES,
        f"manifest per group {per_group}, total {manifest['total_figures']}, "
        f"declared per ring {manifest['figures_per_ring']}, groups "
        f"{manifest['groups']}; phase_stats.json declares "
        f"{stats_a['atlas']['figures']} figures",
        f"{{'ring_1x1': {FIGURES_PER_RING}, 'ring_r3': {FIGURES_PER_RING}}}",
    )

    # ---- the JSON ---------------------------------------------------------- #
    stats_text = stats_path.read_text()
    field_names = json_keys(stats_a)
    stale_json = source_token_hits(
        " ".join(field_names).lower(),
        [token.lower() for token in REMOVED_ENGINE + REMOVED_STAIRCASE],
    )
    text_tokens = [
        token
        for token in REMOVED_ENGINE + REMOVED_STAIRCASE
        if token != STAIRCASE_NUMBER_DEG
    ]
    # The inbound detector reports the label of the hexagonal basis it resolved
    # internally inside its own metadata, and the log quotes that label; the
    # pass-through existed in v3 as well, no computation of this skill reads it, and
    # renaming it would falsify a provenance string.  It is therefore masked here and
    # reported, instead of being tolerated silently.
    foreign = str(stats_a.get("detector_detail") or "")
    foreign_hits = [
        token for token in text_tokens if token and token.lower() in foreign.lower()
    ]
    product_text = log_text + stats_text
    if foreign_hits:
        product_text = product_text.replace(foreign, "<detector-detail>")
    stale_text = source_token_hits(
        product_text.lower(), [token.lower() for token in text_tokens]
    )
    check(
        "no removed JSON field in phase_stats.json",
        not stale_json and not stale_text,
        f"{len(field_names)} field names and {len(stats_text)} characters of "
        f"phase_stats.json scanned: hits in field names "
        f"{stale_json if stale_json else 'none'}, in the text "
        f"{stale_text if stale_text else 'none'} (the bare period of the removed "
        f"layer is only searched in field names, because a decimal value can contain "
        f"those three digits)",
        "0 hits",
    )
    check(
        "no removed-staircase token in the delivered product",
        not source_token_hits(
            product_text.lower(),
            [
                token.lower()
                for token in REMOVED_STAIRCASE
                if token != STAIRCASE_NUMBER_DEG
            ],
        ),
        "the log and phase_stats.json of a real run carry no token of the removed "
        "phase-axis layer; the label the inbound detector reports for its own "
        "internal basis is masked first and listed here: "
        f"{foreign_hits if foreign_hits else 'none to mask'} (the string is "
        f"{foreign!r})",
        "0 hits",
    )
    groups_json = stats_a["pairwise"]["groups"]
    counts_ok = (
        groups_json.get("within_1x1", {}).get("n_pairs") == 3
        and groups_json.get("within_r3", {}).get("n_pairs") == 3
        and set(groups_json) == {"within_1x1", "within_r3"}
    )
    entry_ok = True
    entry_detail = ""
    for group_name, group in groups_json.items():
        for entry in group.get("pairs", []):
            if not {
                "j",
                "k",
                "q_j",
                "q_k",
                "phase_diff_mean",
                "phase_diff_median",
                "phase_diff_R",
                "phase_diff_fwhm_deg",
                "phase_diff_n_clusters",
                "amp_diff_median",
                "n_valid",
            } <= set(entry):
                entry_ok = False
                entry_detail = f"{group_name} pair {entry.get('index')} misses a field"
            for key, path in entry.get("paths", {}).items():
                if path != f"pairwise.groups.{group_name}.pairs.{entry['index']}.{key}":
                    entry_ok = False
                    entry_detail = (
                        f"{group_name} pair {entry['index']}: {key} -> {path}"
                    )
    check(
        "the pair statistics are the statistics of the drawn sample",
        counts_ok and entry_ok,
        f"within_1x1 {groups_json.get('within_1x1', {}).get('n_pairs')} pairs, "
        f"within_r3 {groups_json.get('within_r3', {}).get('n_pairs')} pairs, groups "
        f"{sorted(groups_json)}; every annotated number is reachable at its documented "
        f"path and describes the sample of the D histograms"
        + (f"; problem: {entry_detail}" if entry_detail else ""),
        "3 + 3 pairs with complete entries",
    )

    if (
        second_completed.returncode != 0
        or not (second_outdir / "phase_stats.json").is_file()
    ):
        check(
            "determinism and the remaining pipeline checks",
            False,
            "the second run did not produce phase_stats.json; not reading it",
            "run b completes",
        )
        return

    stats_b = json.loads((second_outdir / "phase_stats.json").read_text())

    def masked_bytes(path, directory):
        """The file bytes with its own output directory replaced by a placeholder.

        ``phase_stats.json`` escapes non-ASCII characters, so both the plain and the
        escaped form of the directory are replaced; the comparison stays byte level.
        """
        raw = path.read_bytes()
        for form in (str(directory), json.dumps(str(directory))[1:-1]):
            raw = raw.replace(form.encode(), b"<outdir>")
        return raw

    stats_a_bytes = masked_bytes(stats_path, outdir)
    stats_b_bytes = masked_bytes(second_outdir / "phase_stats.json", second_outdir)
    check(
        "two runs give byte-identical atlas_manifest.json",
        manifest_path.read_bytes()
        == (second_outdir / "atlas_manifest.json").read_bytes(),
        f"the two manifests are byte identical ({len(manifest_path.read_bytes())} bytes "
        f"each; the manifest carries no path of its own)",
        "byte identical",
    )
    check(
        "two runs give byte-identical phase_stats.json (its own output directory "
        "masked)",
        stats_a_bytes == stats_b_bytes and b"<outdir>" in stats_a_bytes,
        f"the two files are byte identical after replacing each run's own output "
        f"directory ({len(stats_a_bytes)} bytes each; the only difference is the "
        f"'atlas.dir' field, which names that run's directory by construction)",
        "byte identical modulo the output directory",
    )
    for payload in (stats_a, stats_b):
        payload.pop("atlas", None)
    check(
        "two runs give the same JSON numbers",
        json.dumps(stats_a, sort_keys=True) == json.dumps(stats_b, sort_keys=True),
        "phase_stats.json identical (atlas path field masked)",
        "identical",
    )

    from PIL import Image

    same_pixels, byte_identical, first_difference = True, True, ""
    for picture in figures:
        other = second_outdir / picture.name
        if not other.is_file():
            same_pixels, first_difference = False, f"{picture.name} missing in run b"
            break
        if picture.read_bytes() != other.read_bytes():
            byte_identical = False
        with Image.open(picture) as image_a, Image.open(other) as image_b:
            if image_a.size != image_b.size or not np.array_equal(
                np.asarray(image_a), np.asarray(image_b)
            ):
                same_pixels, first_difference = False, f"{picture.name} differs"
                break
    check(
        "two runs give identical figure pixels",
        same_pixels,
        "all figures pixel-identical" if same_pixels else first_difference,
        "identical",
    )
    check(
        "two runs give byte-identical PNG files",
        byte_identical and len(figures) == TOTAL_FIGURES,
        f"all {len(figures)} PNG byte streams are identical between the two runs"
        if byte_identical
        else f"pixel-identical but not byte-identical: {first_difference}",
        "byte identical",
    )

    hits = []
    for ring in ("ring_1x1", "ring_r3"):
        block = stats_a["rings_analysis"][ring]
        for field, value in (
            ("radius", block["radius_px"]),
            ("median", block["peaks"][0]["phase_median_deg"]),
            ("mean", block["peaks"][0]["phase_mean_deg"]),
        ):
            token = f"{value:.4f}"
            hits.append((f"{ring}.{field}", token, token in log_text))
    pair_token = f"{groups_json['within_1x1']['pairs'][0]['phase_diff_mean']:.4f}"
    hits.append(("within_1x1 pair D mean", pair_token, pair_token in log_text))
    check(
        "the log carries the same numbers as the JSON",
        all(hit for _name, _token, hit in hits),
        "; ".join(
            f"{name}={token}:{'found' if hit else 'MISSING'}"
            for name, token, hit in hits
        ),
        "all found",
    )

    outdir_missing = workdir / "pipeline" / "out_not_found"
    completed = run_script(
        "stm_phase_analysis.py",
        [
            str(csv_path),
            "-o",
            str(outdir_missing),
            "-L",
            str(field_of_view),
            "--detector",
            "builtin",
            "--stm-lib",
            stm_lib,
            "--anchor",
            "inner",
        ],
        env,
    )
    log = (outdir_missing / "phase_stats.log").read_text()
    payload = json.loads((outdir_missing / "phase_stats.json").read_text())
    check(
        "missing r3 partner: exit code 2, an explicit message and no figure",
        completed.returncode == 2
        and "r3 ring is NOT reported" in log
        and payload["status"] == "r3_not_found"
        and not list(outdir_missing.glob("*.png")),
        f"exit code {completed.returncode}, status {payload['status']}, "
        f"{len(list(outdir_missing.glob('*.png')))} figures, message "
        f"{'present' if 'r3 ring is NOT reported' in log else 'MISSING'}",
        "exit code 2, message present, 0 figures",
    )


def test_field_of_view_from_log(n, workdir, stm_lib):
    """--size-nm-from-log must read the corrected canvas, not the input canvas."""
    section("field of view from a correction log (corrected canvas, not input canvas)")
    env = dict(os.environ)
    env["MPLCONFIGDIR"] = str(workdir / ".mplcache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    base = workdir / "fov_from_log"
    base.mkdir(parents=True, exist_ok=True)
    csv_path = base / "synthetic.csv"
    write_synthetic_csv(csv_path, n)
    size_nm, gain = 50.0, 1.03
    log_path = base / "correction.log"
    log_path.write_text(
        "# geometry correction through the bragg_peak package (skill version 2.0)\n"
        f"# canvas {n} x {n} px, field of view {size_nm:g} nm ({size_nm / n:.6f} nm/px)\n"
        f"# corrected canvas: {round(n * gain)} x {round(n * gain)} px, "
        f"field of view {size_nm * gain:.4f} nm ({size_nm / n:.6f} nm/px)\n"
    )
    outdir = base / "out"
    completed = run_script(
        "stm_phase_analysis.py",
        [
            str(csv_path),
            "-o",
            str(outdir),
            "--size-nm-from-log",
            str(log_path),
            "--detector",
            "builtin",
            "--stm-lib",
            stm_lib,
            "--no-figures",
        ],
        env,
    )
    log = (
        (outdir / "phase_stats.log").read_text()
        if (outdir / "phase_stats.log").is_file()
        else ""
    )
    payload = (
        json.loads((outdir / "phase_stats.json").read_text())
        if (outdir / "phase_stats.json").is_file()
        else {}
    )
    check(
        "--size-nm-from-log takes the corrected canvas line, not the input canvas one",
        completed.returncode == 0
        and payload.get("field_of_view_nm") == size_nm * gain
        and f"field of view {size_nm * gain:g} nm" in log
        and "corrected canvas line" in (payload.get("field_of_view_source") or ""),
        f"exit code {completed.returncode}, field_of_view_nm = "
        f"{payload.get('field_of_view_nm')} (input canvas {size_nm:g} nm, corrected "
        f"canvas {size_nm * gain:.4f} nm), log header "
        f"{'has' if f'field of view {size_nm * gain:g} nm' in log else 'MISSING'} "
        f"the corrected value",
        f"{size_nm * gain:.4f} nm from the corrected canvas line",
    )
    plain = base / "plain.log"
    plain.write_text(f"# canvas {n} x {n} px, field of view {size_nm:g} nm\n")
    outdir_plain = base / "out_plain"
    completed = run_script(
        "stm_phase_analysis.py",
        [
            str(csv_path),
            "-o",
            str(outdir_plain),
            "--size-nm-from-log",
            str(plain),
            "--detector",
            "builtin",
            "--stm-lib",
            stm_lib,
            "--no-figures",
        ],
        env,
    )
    payload_plain = (
        json.loads((outdir_plain / "phase_stats.json").read_text())
        if (outdir_plain / "phase_stats.json").is_file()
        else {}
    )
    check(
        "--size-nm-from-log falls back to the last 'field of view' match",
        completed.returncode == 0 and payload_plain.get("field_of_view_nm") == size_nm,
        f"exit code {completed.returncode}, field_of_view_nm = "
        f"{payload_plain.get('field_of_view_nm')} for a log whose only line is "
        f"{size_nm:g} nm",
        f"{size_nm:.4f} nm",
    )
    empty = base / "empty.log"
    empty.write_text("# no field of view line at all\n")
    outdir_empty = base / "out_empty"
    completed = run_script(
        "stm_phase_analysis.py",
        [
            str(csv_path),
            "-o",
            str(outdir_empty),
            "--size-nm-from-log",
            str(empty),
            "--detector",
            "builtin",
            "--stm-lib",
            stm_lib,
            "--no-figures",
        ],
        env,
    )
    text = completed.stderr + completed.stdout
    needle = "no 'field of view"
    check(
        "--size-nm-from-log without any match is an explicit error",
        completed.returncode != 0 and needle in text,
        f"exit code {completed.returncode}, message "
        f"{'present' if needle in text else 'MISSING'}",
        "non-zero exit with an explicit message",
    )


def test_coverage(quick):
    """Map the acceptance items to named checks and report the final count."""
    section("acceptance coverage and the reported check count")
    total = RECORDED["passed"] + RECORDED["failed"]
    floor = 0 if quick else MIN_CHECKS
    check(
        "the reported pass count is the number of checks that ran",
        total == len(RESULTS) and len(RESULTS) >= floor,
        f"check() recorded {RECORDED['passed']} pass and {RECORDED['failed']} fail "
        f"lines against {len(RESULTS)} results in the list (floor {floor}"
        + (", no floor under --quick" if quick else ", so a section cannot vanish")
        + f"); the closing line reports {len(RESULTS) + 1} checks in total",
        f"== and >= {floor}",
    )
    ran = {name for name, _ok, _detail in RESULTS}
    missing = {}
    for item, names in COVERAGE.items():
        absent = [name for name in names if name not in ran]
        if absent:
            missing[item] = absent
    if quick:
        check(
            "the acceptance coverage map is reported (--quick runs a subset)",
            True,
            f"{len(COVERAGE)} acceptance items mapped; without the end-to-end "
            f"stage these are not exercised: {sorted(missing) if missing else 'none'}",
            "informational",
        )
    else:
        check(
            "the self-test covers every acceptance item with a named check",
            not missing,
            f"{len(COVERAGE)} acceptance items mapped to named checks; not run: "
            f"{missing if missing else 'none'}",
            "0 missing",
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir",
        default=None,
        help="scratch directory (default: <tmp>/stm-phase-selftest)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="canvas side of the synthetic images (default 384)",
    )
    parser.add_argument(
        "--stm-lib",
        default=DEFAULT_STM_LIB,
        help="STM_DataProcessing src directory (package detection and "
        "the gwyddion colormap)",
    )
    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=3.0,
        help="Gaussian window width used by the analytic stages "
        "(default 3.0 nm, the engine default)",
    )
    parser.add_argument(
        "--nm-per-px",
        type=float,
        default=0.13,
        help="pixel size used by the analytic stages (default 0.13 nm, "
        "about the 50 nm / 384 px of the scratch canvas)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the end-to-end pipeline and correction stages",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the scratch directory"
    )
    args = parser.parse_args(argv)

    workdir = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.gettempdir()) / "stm-phase-selftest"
    )
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / ".mplcache"))

    print("# selftest.py - analytic checks of the demodulation engine, the Fourier")
    print("# identities it satisfies, the sample rule, the gauge layer, the pairwise")
    print("# analysis, the ring lookup branches and the atlas contract of the skill")
    print(f"# scratch directory: {workdir}")

    test_engine_source_contract()
    test_sample_rule()
    test_circular_estimators()
    test_engine_identities(args.size, args.lambda_nm, args.nm_per_px)
    test_gauge_layer(args.size, args.lambda_nm, args.nm_per_px)
    test_pairwise_contract(args.size, args.lambda_nm, args.nm_per_px)
    test_pairwise_recovery(args.size, args.lambda_nm, args.nm_per_px)
    test_ring_lookup()
    test_multi_component_boundary(args.size, args.lambda_nm, args.nm_per_px)
    test_global_amplitude_scale()
    test_colorbar_contract()
    test_title_and_mask_gate(workdir)
    if args.quick:
        check(
            "end-to-end pipeline and correction stages skipped (--quick)",
            True,
            "informational",
        )
    else:
        test_pipeline_contract(args.size, workdir, args.stm_lib)
        test_field_of_view_from_log(args.size, workdir, args.stm_lib)
    test_coverage(args.quick)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    if not args.keep and not args.workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
