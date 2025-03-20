# -*- coding: utf-8 -*-
"""
Python module for simple functions.
Name: func.py
"""
__all__ = ['vortex_num']


def vortex_num(filed: float, area: float):
    """Calculate the vortex number using the given filed and area.

    Args:
        filed (float): The filed value.
        area (float): The area value.

    Returns:
        float: The calculated vortex number.
    """
    phi_0 = 2e-15
    return filed * area / phi_0
