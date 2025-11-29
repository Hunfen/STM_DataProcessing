# -*- coding: utf-8 -*-
"""Utility functions for physics calculations.

This module provides utility functions for various
physics-related calculations, particularly focused
on superconducting and vortex physics applications.

Functions:
    vortex_num: Calculate the number of magnetic flux quanta (vortices)
    based on magnetic field and area.

Constants:
    phi_0: Magnetic flux quantum (2.067833848e-15 Wb)

Example:
    >>> vortex_num(field=0.1, area=1e-12)
    4.835978e+13
"""

__all__ = ['np', 'pd', 'vortex_num']

from . import np, pd


def vortex_num(field: float = 0, area: float = 0):
    """Calculate the vortex number using the given filed and area.

    Args:
        filed (float): The filed value.
        area (float): The area value.

    Returns:
        float: The calculated vortex number.
    """
    phi_0 = 2.067833848e-15  # 1 Wb <=> 1 T*m**2
    return field * area / phi_0
