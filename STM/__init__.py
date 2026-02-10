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

__version__ = "0.9.0"
__author__ = "hunfen.gpt"


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qpi_core import QPICalculator
    from .lattice import LATTICE2D, LatticeOperations
    from .nanonis_loader import NanonisFileLoader
    from .preview_plot import plot_topo

# Regular imports (eager loading for most use cases)
from .lattice import LATTICE2D, LatticeOperations, create_polygon_mask
from .nanonis_loader import NanonisFileLoader
from .preview_plot import plot_topo
from .qpi_core import QPICalculator

__all__ = [
    "QPICalculator",
    "LATTICE2D", 
    "LatticeOperations",
    "create_polygon_mask",
    "NanonisFileLoader",
    "plot_topo",
]

