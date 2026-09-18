"""Compatibility shim for the modular :mod:`...utils.bragg_peak` package.

The implementation moved to ``stm_data_processing.utils.bragg_peak``; import
that path in new code.  This module only re-exports the public names so that
existing ``from stm_data_processing.utils.bragg_peak_detection import ...``
imports keep working.
"""

from __future__ import annotations

from .bragg_peak import (
    BraggDetectionResult,
    BraggPeak,
    BraggPeakDetector,
    LatticeFit,
    LatticeSpec,
    compute_fft2,
    detect_bragg_peaks,
)

__all__ = [
    "BraggDetectionResult",
    "BraggPeak",
    "BraggPeakDetector",
    "LatticeFit",
    "LatticeSpec",
    "compute_fft2",
    "detect_bragg_peaks",
]
