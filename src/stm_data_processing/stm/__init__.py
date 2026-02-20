"""
STM (Scanning Tunneling Microscopy) module for Quasiparticle Interference (QPI) analysis.

This module provides tools for loading, processing, and analyzing STM data,
with a focus on QPI pattern extraction and lattice analysis.

Main components:
- QPICalculator: Core QPI calculation and analysis
- NanonisFileLoader: Loader for Nanonis file format
- LatticeOperations: 2D lattice operations and utilities
- Visualization tools: Topography plotting and mask creation
"""

__author__ = "hunfen.gpt"

from . import lattice, nanonis_loader, preview_plot, qpi_core, utility

__all__ = [
    "lattice",
    "nanonis_loader",
    "preview_plot",
    "qpi_core",
    "utility",
]
