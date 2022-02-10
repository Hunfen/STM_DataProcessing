# -*- coding: utf-8 -*-
__all__ = ['slice_3ds']

from typing import Any
import numpy as np


def slice_3ds(f: Any, bias: float = 0, full: bool = False):
    """[summary]

    Args:
        f (Any): [description]
        bias (float, optional): [description]. Defaults to 0.
        full (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    dim = f.header['Grid dim']
    points = f.header['Points']

    sweep = np.linspace(round(f.Parameters[0][0], 3),
                        round(f.Parameters[0][1], 3), f.header['Points'])

    # z
    z = np.zeros((dim[1], dim[0]))
    for row in range(dim[1]):
        for col in range(dim[0]):
            z[row][col] = f.Parameters[(dim[1] - row - 1) * dim[0] + col][4]

    # di/dv
    if full:
        sliced_data = np.zeros((points, dim[1], dim[0]))
        for i in range(points):  # points of sweep for each position
            for row in range(dim[1]):  # number of row for matplotlib
                for col in range(dim[0]):  # number of column for matplotlib
                    sliced_data[i][row][col] = (
                        f.data[(dim[1] - row - 1) * dim[0] + col][1][i] +
                        f.data[(dim[1] - row - 1) * dim[0] +
                               col][4][i]) / 2  # re-distribution of data
        return sweep, sliced_data, z
    else:
        sliced_data = np.zeros((dim[1], dim[0]))
        bias_idx = find_nearest(sweep, bias)
        for row in range(dim[1]):
            for col in range(dim[0]):
                sliced_data[row][col] = (
                    f.data[(dim[1] - row - 1) * dim[0] + col][1][bias_idx] +
                    f.data[(dim[1] - row - 1) * dim[0] + col][4][bias_idx]) / 2
        return sweep[bias_idx], sliced_data, z


def find_nearest(array: np.ndarray, value: float):
    """[summary]

    Args:
        array (np.ndarray): An array of numbers.
        value (float): A specified value.

    Returns:
        int: serial number of the element, which is the nearest value.
    """
    idx, val = min(enumerate(array), key=lambda x: abs(x[1] - value))
    return idx
