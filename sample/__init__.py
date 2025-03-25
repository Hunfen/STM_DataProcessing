# -*- coding: utf-8 -*-
"""
Nanonis Data File Loader Package

A Python toolkit for parsing Nanonis scanning probe microscopy data files
(.sxm/.dat/.3ds). Provides unified interfaces to access scan parameters,
channel data, and metadata.

Key Features:
    - Auto-detection of file types (SXM/DAT/3DS)
    - Structured extraction of scan parameters (bias, range, setpoint)
    - Multi-channel data support with directional scan correction

Example:
    >>> from sample import NanonisFileLoader
    >>> loader = NanonisFileLoader("scan.3ds")
    >>> print(loader.header["Grid dim"])  # Get grid dimensions
    >>> spectrum = loader.data  # Get 3D spectroscopy data

Dependencies:
    Requires numpy (for array operations) and pandas (for tabular data).
"""

__all__ = ['np', 'pd', 'NanonisFileLoader']

import numpy as np
import pandas as pd
from .nanonis_loader import NanonisFileLoader
