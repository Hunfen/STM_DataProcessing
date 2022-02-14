# -*- coding: utf-8 -*-
__all__ = ['slice_3ds', 'level_plain', 'align_row_diff_median', 'topo_extent']

from typing import Any
import numpy as np
from scipy.optimize import curve_fit


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


def level_plain(raw_topo: np.ndarray) -> np.ndarray:

    vector: list = []
    x: np.ndarray = np.zeros((raw_topo.ndim, raw_topo.size))
    topo_plain: np.ndarray = np.zeros(raw_topo.shape)
    for i in range(raw_topo.ndim):
        vector.append(np.linspace(0, raw_topo.shape[i] - 1, raw_topo.shape[i]))
    mesh_grid: list[np.ndarray] = np.meshgrid(*vector)
    for i in range(raw_topo.ndim):
        x[i] = mesh_grid[i].flatten()

    if raw_topo.ndim == 2:
        parameter = curve_fit(linear_2d, x, raw_topo.flatten())
        topo_plain_flatten = raw_topo.flatten() - linear_2d(x, *parameter[0])
        topo_plain = topo_plain_flatten.reshape(raw_topo.shape)
        topo_plain -= topo_plain.min()
    return topo_plain


def align_row_diff_median(raw_topo: np.ndarray) -> np.ndarray:
    topo_shift = np.zeros(raw_topo.shape)
    row_median = np.median(raw_topo, axis=1)
    for i in range(len(row_median)):
        topo_shift[i] = raw_topo[i] - (row_median[i] - row_median[0])
    topo_shift -= topo_shift.min()
    return topo_shift


# TODO: Create multi-dimensional linear function
def linear_2d(x: np.ndarray, a: float, b: float, d: float) -> np.ndarray:
    return a * x[0] + b * x[1] + d


def topo_extent(header: dict) -> 'tuple[float, float, float, float]':
    """
    Calculate position of topograph.

    Parameter
    ---------
    header : reformed header of .sxm

    Return
    ------
    position tuple (left[X], right[X], bottom[Y], top[Y])
    """

    center_X = header['Scan>Scanfield'][0]
    center_Y = header['Scan>Scanfield'][1]
    range_X = header['Scan>Scanfield'][2]
    range_Y = header['Scan>Scanfield'][3]
    return (center_X - range_X / 2, center_X + range_X / 2,
            center_Y - range_Y / 2, center_Y + range_Y / 2)
