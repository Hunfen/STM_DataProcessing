"""Ring clustering and reference-ring inference (stage 3b).

Physical model: the |FFT| of a two-dimensional lattice shows hexagonal rings of
six spots (three independent half-plane representatives) and the ring radii follow
the ladder ``|b| * sqrt(h^2 + k^2 + hk)``.  A ring is *defined* by three measured
peaks at ``theta``, ``theta + 60`` and ``theta + 120`` degrees with a common
radius.

Reference selection (``lattice=None``): every detected ring is tried in turn -
anchored on its own three peaks, see :func:`ring_model` - and scored by how many
detected rings its integer labelling explains.  The reference is the **largest**
ring that explains at least ``_LADDER_MIN_RINGS`` rings, i.e. that carries its own
harmonic ladder; otherwise the best-explaining ring wins, with the strongest ring
as the last resort.  That resolves the nesting ambiguity: a
``sqrt(3) x sqrt(3) R30`` superstructure ring sits at ``|b| / sqrt(3)`` and every
reflection of the fundamental lattice is also one of the superlattice, so counting
explained reflections alone always prefers the finer cell.  Only ring members keep
a label; superstructure and moire rings stay in the result as unlabelled peaks.
"""

from __future__ import annotations

import numpy as np

from .lattice_fit import gls_fit, hexagon_basis, match_labels, spec_model

__all__ = ["infer_reference", "reference_model", "ring_clusters", "ring_model"]

# Radius match of a 60-degree triple (loose: a real ring is an ellipse whose
# members sit 2.1 % apart in radius on the 30 nm standard scan) and its tolerance.
_RING_RADIUS_TOL = 0.03
_RING_ANGLE_TOL_DEG = 5.0
_RING_DUPLICATE_TOL = 0.08
# A reference ring must carry a ladder: itself plus at least two further rings.
_LADDER_MIN_RINGS = 3
# The three half-plane members of a first ring, in order of increasing angle.
_FIRST_RING_LABELS = ((1, 0), (0, 1), (-1, 1))


def angle_distance(first: float, second: float) -> float:
    """Distance between two directions modulo 180 degrees."""
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _triple_around(
    angle: np.ndarray, members: np.ndarray, snr: np.ndarray, seed: int, tol_deg: float
):
    """``(i, j, k)`` triple around ``seed`` at 60 degree spacing, or None."""
    partners = []
    for target in (angle[seed] + 60.0, angle[seed] + 120.0):
        candidates = [
            index
            for index in members.tolist()
            if index != seed and angle_distance(angle[index], target) < tol_deg
        ]
        if not candidates:
            return None
        partners.append(
            max(
                candidates,
                key=lambda index: (
                    float(snr[index]),
                    -angle_distance(angle[index], target),
                ),
            )
        )
    if partners[0] == partners[1]:
        return None
    ordered = sorted([seed, *partners], key=lambda index: angle[index])
    gaps = [
        angle_distance(angle[ordered[0]], angle[ordered[1]]),
        angle_distance(angle[ordered[1]], angle[ordered[2]]),
        angle_distance(angle[ordered[2]], angle[ordered[0]]),
    ]
    if max(abs(gap - 60.0) for gap in gaps) >= tol_deg:
        return None
    return tuple(int(index) for index in ordered)


def ring_clusters(
    q_obs: np.ndarray,
    snr: np.ndarray,
    *,
    radius_tol: float = _RING_RADIUS_TOL,
    angle_tol_deg: float = _RING_ANGLE_TOL_DEG,
) -> list[dict]:
    """Group peaks into hexagonal rings (radius tolerance + 60 degree grouping)."""
    radius = np.hypot(q_obs[:, 0], q_obs[:, 1])
    angle = np.degrees(np.arctan2(q_obs[:, 1], q_obs[:, 0])) % 180.0
    used = np.zeros(q_obs.shape[0], dtype=bool)
    rings: list[dict] = []
    for seed in np.argsort(-snr, kind="stable"):
        if used[seed] or radius[seed] <= 0.0:
            continue
        members = np.nonzero(
            ~used & (np.abs(radius - radius[seed]) <= radius_tol * radius[seed])
        )[0]
        if members.size < 3:
            continue
        triple = _triple_around(angle, members, snr, int(seed), angle_tol_deg)
        if triple is None:
            continue
        used[list(triple)] = True
        rings.append(
            {
                "radius": float(np.mean(radius[list(triple)])),
                "triple": triple,
                "snr": float(np.mean(snr[list(triple)])),
            }
        )
    rings.sort(key=lambda ring: -ring["snr"])
    return [
        ring
        for index, ring in enumerate(rings)
        if not any(
            abs(ring["radius"] - other["radius"])
            <= _RING_DUPLICATE_TOL * other["radius"]
            for other in rings[:index]
        )
    ]


def ring_model(
    q_obs: np.ndarray, covs: list[np.ndarray], ring: dict
) -> tuple[np.ndarray, np.ndarray] | None:
    """Reference basis and fitted affine anchored on one ring.

    The triple is ordered by folded angle (mod 180), so its label sequence is a
    cyclic rotation of ``_FIRST_RING_LABELS``; pairing a fixed sequence with a
    fixed anchor mislabels a ring whose members span the +/-q boundary and blows
    the GLS chi-square up to ~1e8 with no member left inside the labelling
    tolerance.  All three rotations are therefore tried, each anchored on the
    *true* (unfolded) direction of the member that takes the (1, 0) label, and the
    assignment with the smallest chi2 wins.
    """
    triple = list(ring["triple"])
    best = None
    for shift in range(3):
        labels = _FIRST_RING_LABELS[shift:] + _FIRST_RING_LABELS[:shift]
        anchor = triple[labels.index((1, 0))]
        theta = float(np.degrees(np.arctan2(q_obs[anchor, 1], q_obs[anchor, 0])))
        basis = hexagon_basis(ring["radius"], theta)
        fit = gls_fit(q_obs[triple], [covs[i] for i in triple], list(labels), basis)
        if fit is not None and (best is None or fit[2] < best[0]):
            best = (fit[2], basis, fit[0])
    return None if best is None else (best[1], best[2])


def _ladder_rings(labels, rings) -> list[dict]:
    """Rings whose triple carries at least two labels (its own ring included)."""
    return [
        ring
        for ring in rings
        if sum(1 for member in ring["triple"] if labels[int(member)] is not None) >= 2
    ]


def infer_reference(
    q_obs: np.ndarray,
    covs: list[np.ndarray],
    rings: list[dict],
    h_max: int,
    q_max: float,
):
    """Infer the reference basis (see the module docstring)."""
    evaluated = []
    for ring in rings:
        anchored = ring_model(q_obs, covs, ring)
        if anchored is None:
            continue
        basis, affine = anchored
        labels = match_labels(q_obs, basis @ affine, h_max, q_max)
        evaluated.append(
            {
                "ring": ring,
                "basis": basis,
                "affine": affine,
                "labels": labels,
                "ladder": len(_ladder_rings(labels, rings)),
            }
        )
    if not evaluated:
        return [], {
            "basis_source": "no_rings",
            "ring_ladder": [],
            "reference_radius_px": None,
        }
    supported = [item for item in evaluated if item["ladder"] >= _LADDER_MIN_RINGS]
    if supported:
        chosen = max(
            supported, key=lambda item: (item["ring"]["radius"], item["ring"]["snr"])
        )
        reason = "harmonic_ladder"
    else:
        chosen = max(evaluated, key=lambda item: (item["ladder"], item["ring"]["snr"]))
        reason = "strongest_ring"
    diagnostics = {
        "basis_source": reason,
        "ring_ladder": [int(item["ladder"]) for item in evaluated],
        "reference_radius_px": round(chosen["ring"]["radius"], 3),
    }
    return [chosen], diagnostics


def reference_model(
    q_obs: np.ndarray,
    covs: list[np.ndarray],
    snr: np.ndarray,
    rings: list[dict],
    *,
    spec,
    h_max: int,
    q_max: float,
    dq_nm_inv: float,
) -> dict:
    """Choose the reference model, label the peaks and keep the ladder rings."""
    if spec is not None:
        model, basis_source = spec_model(spec, q_obs, snr, 1.0 / dq_nm_inv)
        labels = (
            match_labels(q_obs, model, h_max, q_max)
            if model is not None
            else [None] * len(q_obs)
        )
        ladder = _ladder_rings(labels, rings)
        reference_radius_px = float(np.hypot(*model[0])) if model is not None else None
        symmetry = spec.symmetry
        rotation_is_absolute = spec.orientation_deg is not None
        ring_ladder: list[int] = []
    else:
        chosen, diagnostics = infer_reference(q_obs, covs, rings, h_max, q_max)
        if not chosen:
            return {
                "model": None,
                "labels": [None] * q_obs.shape[0],
                "members": [],
                "ladder_rings": [],
                "symmetry": "hexagonal",
                "rotation_is_absolute": False,
                **diagnostics,
            }
        item = chosen[0]
        model, labels = item["basis"] @ item["affine"], item["labels"]
        ladder = _ladder_rings(labels, rings)
        reference_radius_px = diagnostics["reference_radius_px"]
        symmetry = "hexagonal"
        rotation_is_absolute = False
        basis_source = diagnostics["basis_source"]
        ring_ladder = diagnostics["ring_ladder"]
    if spec is not None:
        # An explicit reference states the symmetry and the scale, so every
        # labelled peak is a fit member.  Restricting the fit to hexagonal ladder
        # rings would drop a square or rectangular lattice (no 60 degree triple)
        # and a hexagonal one whose basis sits further than 2 % from the data.
        members = [
            int(index) for index, label in enumerate(labels) if label is not None
        ]
    else:
        members = [int(member) for ring in ladder for member in ring["triple"]]
    return {
        "model": model,
        "labels": labels,
        "members": members,
        "ladder_rings": [list(ring["triple"]) for ring in ladder],
        "symmetry": symmetry,
        "rotation_is_absolute": bool(rotation_is_absolute),
        "reference_radius_px": reference_radius_px,
        "basis_source": basis_source,
        "ring_ladder": ring_ladder,
    }
