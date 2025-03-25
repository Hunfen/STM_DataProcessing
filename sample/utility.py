# -*- coding: utf-8 -*-

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
