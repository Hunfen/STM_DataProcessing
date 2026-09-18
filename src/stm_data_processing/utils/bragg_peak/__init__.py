"""Modular FFT Bragg peak detection for STM topographs.

Canonical import: ``from stm_data_processing.utils.bragg_peak import
detect_bragg_peaks``.  The old flat module
``stm_data_processing.utils.bragg_peak_detection`` is a compatibility shim
re-exporting this package.
"""

from __future__ import annotations

from .correct import correct_bragg_peaks
from .detect import detect_candidates, radial_background, robust_rayleigh_scale
from .fft import compute_fft2, q_axis_limits
from .lattice_fit import fit_lattice, gls_fit, hexagon_basis, ideal_basis, match_labels
from .localize import finalize_peaks, localize_peak
from .models import (
    BraggDetectionResult,
    BraggPeak,
    CorrectionResult,
    LatticeFit,
    LatticeSpec,
)
from .pipeline import BraggPeakDetector, detect_bragg_peaks
from .preprocess import load_image, validate_image, window_array
from .rings import reference_model, ring_clusters

__all__ = [
    "BraggDetectionResult",
    "BraggPeak",
    "BraggPeakDetector",
    "CorrectionResult",
    "LatticeFit",
    "LatticeSpec",
    "compute_fft2",
    "correct_bragg_peaks",
    "detect_bragg_peaks",
    "load_image",
]
